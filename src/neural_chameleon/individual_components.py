"""Day 8 individual-component screening, vectorized patching, and selection."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from .interventions import (
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
)


ELIGIBLE_LAYERS = (12, 11, 10, 9)
NUM_QUERY_HEADS = 16
SCREEN_METHODS = (
    "activation_rms",
    "probe_projection",
    "attribution_patch",
    "gradient_rms",
)


@dataclass(frozen=True, order=True)
class ComponentCandidate:
    """One frozen Day 8 attention-head or whole-MLP candidate."""

    layer: int
    component_type: str
    head: int | None = None

    def __post_init__(self) -> None:
        if self.layer not in ELIGIBLE_LAYERS:
            raise ValueError("candidate layer is outside the frozen Day 8 set")
        if self.component_type == "attention_head":
            if self.head is None or not 0 <= self.head < NUM_QUERY_HEADS:
                raise ValueError("attention candidates require a valid query-head index")
        elif self.component_type == "mlp":
            if self.head is not None:
                raise ValueError("MLP candidates cannot name a head")
        else:
            raise ValueError(f"unsupported component type: {self.component_type}")

    @property
    def candidate_id(self) -> str:
        if self.component_type == "mlp":
            return f"layer_{self.layer:02d}.mlp"
        return f"layer_{self.layer:02d}.head_{self.head:02d}"

    @property
    def site(self) -> PatchSite:
        if self.component_type == "mlp":
            return PatchSite(ActivationKind.MLP_OUT, self.layer)
        return PatchSite(ActivationKind.HEAD_OUTPUT, self.layer, head=self.head)

    @property
    def tie_break_key(self) -> tuple[int, int, int]:
        return (
            self.layer,
            0 if self.component_type == "attention_head" else 1,
            -1 if self.head is None else self.head,
        )


def eligible_candidates() -> tuple[ComponentCandidate, ...]:
    """Return all 68 candidates in a stable descriptive order."""
    candidates = []
    for layer in ELIGIBLE_LAYERS:
        candidates.extend(
            ComponentCandidate(layer, "attention_head", head)
            for head in range(NUM_QUERY_HEADS)
        )
        candidates.append(ComponentCandidate(layer, "mlp"))
    return tuple(candidates)


CANDIDATES = eligible_candidates()
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}


@dataclass(frozen=True)
class MultiPatchResult:
    """Per-candidate scores from one vectorized patch pass."""

    probe_scores: Tensor
    response_nll: Tensor | None = None


class _MonitorReached(RuntimeError):
    pass


def _repeat_rows(tensor: Tensor, repeats: int) -> Tensor:
    return tensor.repeat((repeats, *([1] * (tensor.ndim - 1))))


def repeat_condition(condition: ConditionBatch, repeats: int) -> ConditionBatch:
    """Repeat a condition in candidate-major blocks."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    return ConditionBatch(
        name=condition.name,
        user_prompts=condition.user_prompts * repeats,
        rendered_prompts=condition.rendered_prompts * repeats,
        input_ids=_repeat_rows(condition.input_ids, repeats),
        attention_mask=_repeat_rows(condition.attention_mask, repeats),
        position_ids=_repeat_rows(condition.position_ids, repeats),
        response_ids=_repeat_rows(condition.response_ids, repeats),
        response_mask=_repeat_rows(condition.response_mask, repeats),
        response_start=condition.response_start,
    )


def masked_example_mean(values: Tensor, mask: Tensor) -> Tensor:
    values = values.float()
    mask = mask.to(values.device)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class MultiCandidatePatchRunner:
    """Evaluate independent single-site patches in one expanded model batch."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _validate(
        self,
        condition: ConditionBatch,
        jobs: Sequence[tuple[ComponentCandidate, CapturedActivation]],
    ) -> None:
        if not jobs:
            raise ValueError("at least one candidate patch is required")
        sites = [candidate.site for candidate, _capture in jobs]
        if len(sites) != len(set(sites)):
            raise ValueError("candidate patch jobs contain duplicate sites")
        self.runner._validate_sites(sites)
        for candidate, capture in jobs:
            if candidate.layer > self.monitor_layer:
                raise ValueError("candidate occurs after the monitor layer")
            self.runner._validate_patch_pair(condition, candidate.site, capture)

    def _register_jobs(
        self,
        base: ConditionBatch,
        jobs: Sequence[tuple[ComponentCandidate, CapturedActivation]],
    ) -> list[Any]:
        grouped: dict[tuple[Any, str], list[tuple[int, ComponentCandidate, CapturedActivation]]] = defaultdict(list)
        for index, (candidate, capture) in enumerate(jobs):
            module, side = self.runner._resolve_module(candidate.site)
            grouped[(module, side)].append((index, candidate, capture))

        handles = []
        for (module, side), group in grouped.items():
            def patch_tensor(
                tensor: Tensor,
                group: Sequence[tuple[int, ComponentCandidate, CapturedActivation]] = tuple(group),
            ) -> Tensor:
                patched = tensor.clone()
                start = base.response_start
                stop = start + base.response_width
                base_batch = base.batch_size
                mask = base.response_mask.to(patched.device).unsqueeze(-1)
                for index, candidate, capture in group:
                    rows = slice(index * base_batch, (index + 1) * base_batch)
                    source = capture.values.to(
                        device=patched.device, dtype=patched.dtype
                    )
                    if candidate.component_type == "attention_head":
                        attention = self.runner.layers[candidate.layer].self_attn
                        num_heads = self.runner._num_attention_heads(attention)
                        head_dim = self.runner._head_dim(attention)
                        reshaped = patched.reshape(
                            *patched.shape[:-1], num_heads, head_dim
                        )
                        destination = reshaped[
                            rows, start:stop, candidate.head, :
                        ]
                        reshaped[rows, start:stop, candidate.head, :] = torch.where(
                            mask, source, destination
                        )
                    else:
                        destination = patched[rows, start:stop, :]
                        patched[rows, start:stop, :] = torch.where(
                            mask, source, destination
                        )
                return patched

            if side == "input":
                def pre_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    patch_tensor=patch_tensor,
                ):
                    tensor = self.runner._first_tensor(args)
                    return self.runner._replace_first_tensor(args, patch_tensor(tensor))

                handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                def forward_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    patch_tensor=patch_tensor,
                ):
                    tensor = self.runner._first_tensor(output)
                    return self.runner._replace_first_tensor(
                        output, patch_tensor(tensor)
                    )

                handles.append(module.register_forward_hook(forward_hook))
        return handles

    def run_truncated(
        self,
        condition: ConditionBatch,
        jobs: Sequence[tuple[ComponentCandidate, CapturedActivation]],
    ) -> MultiPatchResult:
        self._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        scores: Tensor | None = None
        handles = self._register_jobs(condition, jobs)
        try:
            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                tensor = self.runner._first_tensor(output)
                start = expanded.response_start
                stop = start + expanded.response_width
                response = tensor[:, start:stop, :]
                weight = self.probe.weight.to(
                    device=response.device, dtype=torch.bfloat16
                )
                bias = self.probe.bias.to(
                    device=response.device, dtype=torch.bfloat16
                )
                logits = torch.matmul(response.to(torch.bfloat16), weight.T) + bias
                probabilities = torch.sigmoid(logits.squeeze(-1)).float()
                scores = masked_example_mean(
                    probabilities, expanded.response_mask
                ).detach().cpu()
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(
                    terminal_hook
                )
            )
            try:
                with torch.inference_mode():
                    self.runner.model(
                        input_ids=expanded.input_ids.to(self.runner.device),
                        attention_mask=expanded.attention_mask.to(self.runner.device),
                        position_ids=expanded.position_ids.to(self.runner.device),
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                raise RuntimeError("model completed without reaching the monitor")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None:
            raise RuntimeError("vectorized patch score was not captured")
        return MultiPatchResult(
            probe_scores=scores.reshape(len(jobs), condition.batch_size)
        )

    def run_full(
        self,
        condition: ConditionBatch,
        jobs: Sequence[tuple[ComponentCandidate, CapturedActivation]],
    ) -> MultiPatchResult:
        self._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        handles = self._register_jobs(condition, jobs)
        try:
            result = self.runner.run(
                expanded,
                retain_response_logprobs=True,
            )
        finally:
            for handle in reversed(handles):
                handle.remove()
        logprobs = result.response_token_logprobs().reshape(
            len(jobs), condition.batch_size, condition.response_width
        )
        mask = condition.response_mask.unsqueeze(0).expand_as(logprobs)
        nll = (
            (-logprobs.float() * mask).sum(dim=2)
            / mask.sum(dim=2).clamp(min=1)
        )
        return MultiPatchResult(
            probe_scores=torch.empty((len(jobs), condition.batch_size)),
            response_nll=nll,
        )


@dataclass(frozen=True)
class ScreeningBatchResult:
    """Baseline scores and per-example screening values for every candidate."""

    normal_scores: Tensor
    triggered_scores: Tensor
    metrics: Mapping[str, Mapping[str, Tensor]]
    normal_captures: Mapping[str, CapturedActivation]


class ComponentScreeningRunner:
    """Compute four correlational/first-order screens with one paired backward pass."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _score(self, tensor: Tensor, condition: ConditionBatch) -> Tensor:
        start = condition.response_start
        stop = start + condition.response_width
        response = tensor[:, start:stop, :]
        weight = self.probe.weight.to(
            device=response.device, dtype=torch.bfloat16
        )
        bias = self.probe.bias.to(device=response.device, dtype=torch.bfloat16)
        logits = torch.matmul(response.to(torch.bfloat16), weight.T) + bias
        probabilities = torch.sigmoid(logits.squeeze(-1)).float()
        return masked_example_mean(probabilities, condition.response_mask)

    def _normal_pass(
        self, condition: ConditionBatch
    ) -> tuple[Tensor, dict[tuple[int, str], Tensor]]:
        captured: dict[tuple[int, str], Tensor] = {}
        scores: Tensor | None = None
        handles = []
        try:
            for layer in ELIGIBLE_LAYERS:
                attention = self.runner.layers[layer].self_attn

                def head_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    layer: int = layer,
                ):
                    tensor = self.runner._first_tensor(args)
                    start = condition.response_start
                    stop = start + condition.response_width
                    captured[(layer, "heads")] = (
                        tensor[:, start:stop, :].detach().cpu().clone()
                    )

                handles.append(attention.o_proj.register_forward_pre_hook(head_hook))

                def mlp_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    layer: int = layer,
                ):
                    tensor = self.runner._first_tensor(output)
                    start = condition.response_start
                    stop = start + condition.response_width
                    captured[(layer, "mlp")] = (
                        tensor[:, start:stop, :].detach().cpu().clone()
                    )

                handles.append(
                    self.runner.layers[layer].post_feedforward_layernorm.register_forward_hook(
                        mlp_hook
                    )
                )

            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                scores = self._score(
                    self.runner._first_tensor(output), condition
                ).detach().cpu()
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(
                    terminal_hook
                )
            )
            try:
                with torch.inference_mode():
                    self.runner.model(
                        input_ids=condition.input_ids.to(self.runner.device),
                        attention_mask=condition.attention_mask.to(self.runner.device),
                        position_ids=condition.position_ids.to(self.runner.device),
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                raise RuntimeError("normal screening forward missed the monitor")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None or len(captured) != 2 * len(ELIGIBLE_LAYERS):
            raise RuntimeError("normal screening capture is incomplete")
        return scores, captured

    def _triggered_pass(
        self, condition: ConditionBatch
    ) -> tuple[Tensor, dict[tuple[int, str], tuple[Tensor, Tensor]]]:
        tracked: dict[tuple[int, str], Tensor] = {}
        scores: Tensor | None = None
        handles = []
        try:
            for layer in ELIGIBLE_LAYERS:
                attention = self.runner.layers[layer].self_attn

                def head_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    layer: int = layer,
                ):
                    tensor = self.runner._first_tensor(args)
                    replacement = tensor.detach().requires_grad_(True)
                    tracked[(layer, "heads")] = replacement
                    return self.runner._replace_first_tensor(args, replacement)

                handles.append(attention.o_proj.register_forward_pre_hook(head_hook))

                def mlp_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    layer: int = layer,
                ):
                    tensor = self.runner._first_tensor(output)
                    replacement = tensor.detach().requires_grad_(True)
                    tracked[(layer, "mlp")] = replacement
                    return self.runner._replace_first_tensor(output, replacement)

                handles.append(
                    self.runner.layers[layer].post_feedforward_layernorm.register_forward_hook(
                        mlp_hook
                    )
                )

            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                scores = self._score(self.runner._first_tensor(output), condition)
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(
                    terminal_hook
                )
            )
            try:
                with torch.enable_grad():
                    self.runner.model(
                        input_ids=condition.input_ids.to(self.runner.device),
                        attention_mask=condition.attention_mask.to(self.runner.device),
                        position_ids=condition.position_ids.to(self.runner.device),
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                raise RuntimeError("triggered screening forward missed the monitor")
            except _MonitorReached:
                if scores is None:
                    raise RuntimeError("triggered score is missing")
                scores.sum().backward()
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None or len(tracked) != 2 * len(ELIGIBLE_LAYERS):
            raise RuntimeError("triggered screening capture is incomplete")

        result = {}
        start = condition.response_start
        stop = start + condition.response_width
        for key, tensor in tracked.items():
            if tensor.grad is None:
                raise RuntimeError(f"screening gradient is missing for {key}")
            result[key] = (
                tensor[:, start:stop, :].detach().cpu(),
                tensor.grad[:, start:stop, :].detach().cpu(),
            )
        return scores.detach().cpu(), result

    def run(
        self,
        normal: ConditionBatch,
        triggered: ConditionBatch,
    ) -> ScreeningBatchResult:
        if not torch.equal(normal.response_ids, triggered.response_ids):
            raise ValueError("screening conditions do not share response IDs")
        if not torch.equal(normal.response_mask, triggered.response_mask):
            raise ValueError("screening conditions do not share response masks")
        normal_scores, normal_values = self._normal_pass(normal)
        triggered_scores, triggered_values = self._triggered_pass(triggered)
        mask = normal.response_mask.float()
        probe_direction = self.probe.weight.float().squeeze(0)
        probe_direction = probe_direction / probe_direction.norm().clamp(min=1e-12)
        metrics: dict[str, dict[str, Tensor]] = {}
        normal_captures: dict[str, CapturedActivation] = {}

        for candidate in CANDIDATES:
            if candidate.component_type == "attention_head":
                attention = self.runner.layers[candidate.layer].self_attn
                head_dim = self.runner._head_dim(attention)
                start = candidate.head * head_dim
                stop = start + head_dim
                normal_value = normal_values[(candidate.layer, "heads")][
                    :, :, start:stop
                ]
                triggered_value, gradient = triggered_values[
                    (candidate.layer, "heads")
                ]
                triggered_value = triggered_value[:, :, start:stop]
                gradient = gradient[:, :, start:stop]
                weight = attention.o_proj.weight.detach().float()[:, start:stop].cpu()
                delta_contribution = functional.linear(
                    normal_value.float() - triggered_value.float(), weight
                )
            else:
                normal_value = normal_values[(candidate.layer, "mlp")]
                triggered_value, gradient = triggered_values[
                    (candidate.layer, "mlp")
                ]
                delta_contribution = normal_value.float() - triggered_value.float()

            delta = normal_value.float() - triggered_value.float()
            token_activation_rms = delta.square().mean(dim=-1).sqrt()
            token_gradient_rms = gradient.float().square().mean(dim=-1).sqrt()
            attribution = (gradient.float() * delta).sum(dim=-1)
            projection = torch.matmul(delta_contribution, probe_direction)
            metrics[candidate.candidate_id] = {
                "activation_rms": masked_example_mean(
                    token_activation_rms, mask
                ).cpu(),
                "probe_projection": masked_example_mean(projection, mask).cpu(),
                "attribution_patch": (attribution * mask).sum(dim=1).cpu(),
                "gradient_rms": masked_example_mean(
                    token_gradient_rms, mask
                ).cpu(),
            }
            normal_captures[candidate.candidate_id] = CapturedActivation(
                values=normal_value.clone(),
                response_ids=normal.response_ids.clone(),
                response_mask=normal.response_mask.clone(),
            )

        return ScreeningBatchResult(
            normal_scores=normal_scores,
            triggered_scores=triggered_scores,
            metrics=metrics,
            normal_captures=normal_captures,
        )


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = percentile_interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def _average_ranks(values: Sequence[float], descending: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array if descending else array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    position = 0
    while position < len(order):
        stop = position + 1
        while stop < len(order) and array[order[stop]] == array[order[position]]:
            stop += 1
        ranks[order[position:stop]] = (position + 1 + stop) / 2
        position = stop
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left_rank = _average_ranks(left, descending=False)
    right_rank = _average_ranks(right, descending=False)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def component_set_sha256(
    ordered_candidates: Sequence[str], final_k: int, procedure: str = "day08-v1"
) -> str:
    payload = (
        procedure + "\n" + str(final_k) + "\n" + "\n".join(ordered_candidates) + "\n"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def summarize_discovery_candidates(
    records: Iterable[dict[str, Any]],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize exact discovery rescue, screens, ranking, and frozen controls."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    baselines: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        if record["split"] != "discovery" or record["label"] != 1:
            raise ValueError("Day 8 discovery ranking accepts discovery positives only")
        if record["record_type"] == "baseline":
            baselines[record["concept"]][record["example_id"]] = record
        elif record["record_type"] == "candidate":
            nested[record["concept"]][record["example_id"]][
                record["candidate_id"]
            ] = record
        else:
            raise ValueError(f"unknown discovery record type: {record['record_type']}")

    concepts = sorted(nested)
    if len(concepts) != 4:
        raise ValueError("expected four discovery concepts")
    rng = np.random.default_rng(seed)
    concept_summaries = []
    exact_points: dict[str, dict[str, float]] = defaultdict(dict)
    exact_boots: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    screen_points: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for concept in concepts:
        example_ids = sorted(nested[concept])
        if set(example_ids) != set(baselines[concept]):
            raise ValueError(f"baseline/candidate example mismatch for {concept}")
        if any(set(nested[concept][example_id]) != set(CANDIDATE_BY_ID) for example_id in example_ids):
            raise ValueError(f"incomplete candidate grid for {concept}")
        indices = rng.integers(
            0, len(example_ids), size=(replicates, len(example_ids))
        )
        normal = np.asarray(
            [baselines[concept][example_id]["normal_probe_score"] for example_id in example_ids]
        )
        triggered = np.asarray(
            [baselines[concept][example_id]["triggered_probe_score"] for example_id in example_ids]
        )
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable suppression denominator for {concept}")
        candidates = []
        for candidate in CANDIDATES:
            candidate_id = candidate.candidate_id
            values = [
                nested[concept][example_id][candidate_id]
                for example_id in example_ids
            ]
            patched = np.asarray([row["patched_probe_score"] for row in values])
            numerator = float(patched.mean() - triggered.mean())
            fraction = numerator / denominator
            fraction_boot = (
                patched[indices].mean(axis=1) - triggered_boot
            ) / denominator_boot
            exact_points[concept][candidate_id] = fraction
            exact_boots[concept][candidate_id] = fraction_boot
            screens = {}
            for method in SCREEN_METHODS:
                screen_values = np.asarray([row[f"screen_{method}"] for row in values])
                point = float(screen_values.mean())
                screens[method] = estimate(point, screen_values[indices].mean(axis=1))
                screen_points[concept][method][candidate_id] = point
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "layer": candidate.layer,
                    "component_type": candidate.component_type,
                    "head": candidate.head,
                    "patched_mean": float(patched.mean()),
                    "numerator": numerator,
                    "recovery": estimate(fraction, fraction_boot),
                    "screens": screens,
                }
            )
        concept_summaries.append(
            {
                "concept": concept,
                "n_positive": len(example_ids),
                "normal_mean": float(normal.mean()),
                "triggered_mean": float(triggered.mean()),
                "suppression_denominator": estimate(
                    denominator, denominator_boot
                ),
                "candidates": candidates,
            }
        )

    candidate_summaries = []
    for candidate in CANDIDATES:
        candidate_id = candidate.candidate_id
        per_concept = [exact_points[concept][candidate_id] for concept in concepts]
        macro = float(np.mean(per_concept))
        macro_boot = np.stack(
            [exact_boots[concept][candidate_id] for concept in concepts]
        ).mean(axis=0)
        positive_count = sum(
            math.isfinite(value) and value > 0 for value in per_concept
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate_id,
                "layer": candidate.layer,
                "component_type": candidate.component_type,
                "head": candidate.head,
                "macro_recovery": estimate(macro, macro_boot),
                "minimum_concept_recovery": float(min(per_concept)),
                "positive_discovery_concept_count": positive_count,
                "shared_candidate_gate": positive_count >= 3,
                "concept_recovery": {
                    concept: exact_points[concept][candidate_id]
                    for concept in concepts
                },
            }
        )

    eligible = [row for row in candidate_summaries if row["shared_candidate_gate"]]
    eligible.sort(
        key=lambda row: (
            -row["macro_recovery"]["estimate"],
            -row["minimum_concept_recovery"],
            CANDIDATE_BY_ID[row["candidate_id"]].tie_break_key,
        )
    )
    for rank, row in enumerate(eligible, start=1):
        row["rank"] = rank
    if not eligible:
        raise ValueError("no candidate passes the shared-candidate gate")
    ordered = [row["candidate_id"] for row in eligible[:16]]
    nested_sizes = [size for size in (1, 2, 4, 8, 16) if size <= len(ordered)]
    final_k = max(nested_sizes)
    selected = ordered[:final_k]
    outside = [
        candidate.candidate_id
        for candidate in CANDIDATES
        if candidate.candidate_id not in ordered
    ]
    random_controls = sorted(
        outside,
        key=lambda candidate_id: (
            hashlib.sha256(f"42:{candidate_id}".encode()).hexdigest(),
            candidate_id,
        ),
    )[:final_k]
    if len(random_controls) != final_k:
        raise ValueError("insufficient candidates outside the frozen top order")

    exact_macro = {
        row["candidate_id"]: row["macro_recovery"]["estimate"]
        for row in candidate_summaries
    }
    screening_evaluation = []
    for method in SCREEN_METHODS:
        average_concept_rank = {}
        for concept in concepts:
            values = [
                screen_points[concept][method][candidate.candidate_id]
                for candidate in CANDIDATES
            ]
            ranks = _average_ranks(values, descending=True)
            for candidate, rank in zip(CANDIDATES, ranks, strict=True):
                average_concept_rank.setdefault(candidate.candidate_id, []).append(
                    float(rank)
                )
        mean_ranks = {
            candidate_id: float(np.mean(ranks))
            for candidate_id, ranks in average_concept_rank.items()
        }
        screen_order = sorted(mean_ranks, key=lambda item: (mean_ranks[item], item))
        exact_values = [exact_macro[candidate.candidate_id] for candidate in CANDIDATES]
        screen_scores = [-mean_ranks[candidate.candidate_id] for candidate in CANDIDATES]
        screening_evaluation.append(
            {
                "method": method,
                "spearman_with_exact_macro_recovery": _spearman(
                    screen_scores, exact_values
                ),
                "top_16_overlap": len(set(screen_order[:16]) & set(ordered)),
                "ordered_candidates": screen_order,
                "average_concept_rank": mean_ranks,
            }
        )

    return {
        "schema_version": 1,
        "procedure": "day08-v1",
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "macro_weighting": "equal discovery concept weight",
        },
        "eligible_layers": list(ELIGIBLE_LAYERS),
        "candidate_count": len(CANDIDATES),
        "concepts": concept_summaries,
        "candidates": candidate_summaries,
        "eligible_candidate_count": len(eligible),
        "ordered_eligible_candidates": [row["candidate_id"] for row in eligible],
        "frozen_top_16": ordered,
        "nested_set_sizes": nested_sizes,
        "final_k": final_k,
        "selected_candidates": selected,
        "random_control_candidates": random_controls,
        "component_set_sha256": component_set_sha256(ordered, final_k),
        "screening_evaluation": screening_evaluation,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
