"""Day 10 generalized transplants, interpolation, and sufficiency analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch import nn

from .individual_components import masked_example_mean, repeat_condition
from .interventions import (
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
)


@dataclass(frozen=True)
class TransplantMember:
    """One response-token activation site and its source values."""

    site: PatchSite
    capture: CapturedActivation


@dataclass(frozen=True)
class TransplantJob:
    """One named, independently evaluated multi-site transplant."""

    group_id: str
    members: tuple[TransplantMember, ...]


@dataclass(frozen=True)
class TransplantResult:
    """Per-job probe scores or full-model response NLL."""

    group_ids: tuple[str, ...]
    probe_scores: Tensor
    response_nll: Tensor | None = None


@dataclass(frozen=True)
class FullTransplantResult:
    """Per-job full-model probe, likelihood, and logit diagnostics."""

    group_ids: tuple[str, ...]
    probe_scores: Tensor
    probe_token_probabilities: Tensor
    response_nll: Tensor
    response_logits: Tensor


class _MonitorReached(RuntimeError):
    pass


class _PassthroughDecoderLayer(nn.Module):
    """Temporary no-op preserving the attention-type attribute used by Gemma."""

    def __init__(self, attention_type: str):
        super().__init__()
        self.attention_type = attention_type

    def forward(self, hidden_states: Tensor, **_kwargs: Any) -> tuple[Tensor]:
        return (hidden_states,)


def interpolate_capture(
    normal: CapturedActivation,
    triggered: CapturedActivation,
    alpha: float,
) -> CapturedActivation:
    """Return the response-aligned linear interpolation between natural endpoints."""
    if not 0 <= alpha <= 1:
        raise ValueError("interpolation alpha must be in [0, 1]")
    if not torch.equal(normal.response_ids, triggered.response_ids):
        raise ValueError("interpolation endpoints have different response tokens")
    if not torch.equal(normal.response_mask, triggered.response_mask):
        raise ValueError("interpolation endpoints have different response masks")
    if normal.values.shape != triggered.values.shape:
        raise ValueError("interpolation endpoints have different activation shapes")
    values = normal.values.float() + alpha * (
        triggered.values.float() - normal.values.float()
    )
    if alpha == 0:
        values = normal.values.clone()
    elif alpha == 1:
        values = triggered.values.clone()
    return CapturedActivation(
        values=values,
        response_ids=normal.response_ids.clone(),
        response_mask=normal.response_mask.clone(),
    )


def response_activation_rms(capture: CapturedActivation) -> Tensor:
    """Calculate one response-masked activation RMS per example."""
    values = capture.values.float()
    mask = capture.response_mask.to(values.device)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded = mask.expand_as(values)
    squared = values.square() * expanded
    dimensions = tuple(range(1, values.ndim))
    numerator = squared.sum(dim=dimensions)
    denominator = expanded.sum(dim=dimensions).clamp(min=1)
    return torch.sqrt(numerator / denominator).detach().cpu()


def group_activation_norms(
    sites: Sequence[PatchSite],
    normal: Mapping[PatchSite, CapturedActivation],
    triggered: Mapping[PatchSite, CapturedActivation],
    mixed: Mapping[PatchSite, CapturedActivation],
) -> dict[str, Tensor]:
    """Summarize endpoint and mixed RMS across a group's member sites."""
    if not sites:
        raise ValueError("activation norm group must contain at least one site")
    normal_rms = torch.stack([response_activation_rms(normal[site]) for site in sites])
    triggered_rms = torch.stack(
        [response_activation_rms(triggered[site]) for site in sites]
    )
    mixed_rms = torch.stack([response_activation_rms(mixed[site]) for site in sites])
    endpoint_max = torch.maximum(normal_rms, triggered_rms).clamp(min=1e-12)
    return {
        "normal_mean": normal_rms.mean(dim=0),
        "normal_max": normal_rms.max(dim=0).values,
        "triggered_mean": triggered_rms.mean(dim=0),
        "triggered_max": triggered_rms.max(dim=0).values,
        "mixed_mean": mixed_rms.mean(dim=0),
        "mixed_max": mixed_rms.max(dim=0).values,
        "bound_ratio_max": (mixed_rms / endpoint_max).max(dim=0).values,
    }


class VectorizedTransplantRunner:
    """Evaluate independent generalized multi-site transplants in one model batch."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _validate(self, condition: ConditionBatch, jobs: Sequence[TransplantJob]) -> None:
        if not jobs:
            raise ValueError("at least one transplant job is required")
        if len({job.group_id for job in jobs}) != len(jobs):
            raise ValueError("transplant jobs contain duplicate IDs")
        for job in jobs:
            if not job.members:
                raise ValueError(f"transplant {job.group_id} has no members")
            sites = [member.site for member in job.members]
            if len(sites) != len(set(sites)):
                raise ValueError(f"transplant {job.group_id} contains duplicate sites")
            self.runner._validate_sites(sites)
            for member in job.members:
                if member.site.layer > self.monitor_layer:
                    raise ValueError("transplant contains a site after the monitor")
                self.runner._validate_patch_pair(
                    condition, member.site, member.capture
                )

    def _register_jobs(
        self, base: ConditionBatch, jobs: Sequence[TransplantJob]
    ) -> list[Any]:
        grouped: dict[
            tuple[Any, str], list[tuple[int, TransplantMember]]
        ] = defaultdict(list)
        for job_index, job in enumerate(jobs):
            for member in job.members:
                module, side = self.runner._resolve_module(member.site)
                grouped[(module, side)].append((job_index, member))

        handles = []
        for (module, side), entries in grouped.items():
            by_site: dict[PatchSite, list[tuple[int, TransplantMember]]] = (
                defaultdict(list)
            )
            for job_index, member in entries:
                by_site[member.site].append((job_index, member))
            site_batches = tuple(
                (
                    site,
                    tuple(job_index for job_index, _member in site_entries),
                    torch.cat(
                        [member.capture.values for _job_index, member in site_entries],
                        dim=0,
                    ),
                )
                for site, site_entries in by_site.items()
            )

            def patch_tensor(
                tensor: Tensor,
                site_batches: Sequence[tuple[PatchSite, tuple[int, ...], Tensor]] = site_batches,
            ) -> Tensor:
                patched = tensor.clone()
                for site, job_indices, source_values in site_batches:
                    self.runner._patch_response_values_rows_in_place(
                        patched,
                        base,
                        site,
                        source_values,
                        job_indices,
                    )
                return patched

            if side == "input":

                def pre_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    patch_tensor=patch_tensor,
                ):
                    tensor = self.runner._first_tensor(args)
                    return self.runner._replace_first_tensor(
                        args, patch_tensor(tensor)
                    )

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
        self, condition: ConditionBatch, jobs: Sequence[TransplantJob]
    ) -> TransplantResult:
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
            raise RuntimeError("transplant probe score was not captured")
        return TransplantResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=scores.reshape(len(jobs), condition.batch_size),
        )

    def run_full(
        self, condition: ConditionBatch, jobs: Sequence[TransplantJob]
    ) -> TransplantResult:
        self._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        handles = self._register_jobs(condition, jobs)
        try:
            result = self.runner.run(expanded, retain_response_logprobs=True)
        finally:
            for handle in reversed(handles):
                handle.remove()
        logprobs = result.response_token_logprobs().reshape(
            len(jobs), condition.batch_size, condition.response_width
        )
        mask = condition.response_mask.unsqueeze(0).expand_as(logprobs)
        nll = (-logprobs.float() * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        return TransplantResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=torch.empty((len(jobs), condition.batch_size)),
            response_nll=nll,
        )

    def run_full_diagnostics(
        self, condition: ConditionBatch, jobs: Sequence[TransplantJob]
    ) -> FullTransplantResult:
        """Run independent jobs once while retaining exact response diagnostics."""
        self._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, self.monitor_layer)
        handles = self._register_jobs(condition, jobs)
        try:
            result = self.runner.run(
                expanded,
                capture_sites=(monitor_site,),
                retain_response_logits=True,
            )
        finally:
            for handle in reversed(handles):
                handle.remove()
        capture = result.captures[monitor_site]
        values = capture.values
        weight = self.probe.weight.to(device=values.device, dtype=torch.bfloat16)
        bias = self.probe.bias.to(device=values.device, dtype=torch.bfloat16)
        probabilities = torch.sigmoid(
            torch.matmul(values.to(torch.bfloat16), weight.T) + bias
        ).squeeze(-1).float()
        scores = masked_example_mean(probabilities, capture.response_mask)
        logprobs = result.response_token_logprobs()
        mask = expanded.response_mask
        nll = (-logprobs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if result.response_logits is None:
            raise RuntimeError("full transplant response logits were not retained")
        shape = (len(jobs), condition.batch_size)
        return FullTransplantResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=scores.reshape(*shape),
            probe_token_probabilities=probabilities.detach().cpu().reshape(
                *shape, condition.response_width
            ),
            response_nll=nll.detach().cpu().reshape(*shape),
            response_logits=result.response_logits.reshape(
                *shape, condition.response_width, result.response_logits.shape[-1]
            ),
        )


class CachedTailTransplantRunner(VectorizedTransplantRunner):
    """Replay only a cached destination-to-monitor tail for exact transplants.

    The model still constructs the original attention masks and rotary position
    embeddings from ``condition``. Decoder layers before ``start_layer`` are
    temporarily replaced by pass-through modules, and a pre-hook replaces the
    first live layer's input with the full cached natural residual. This is an
    execution optimization only; real-checkpoint equivalence to the complete
    forward pass must be checked before use.
    """

    def run_truncated_from_layer(
        self,
        condition: ConditionBatch,
        jobs: Sequence[TransplantJob],
        *,
        start_layer: int,
        cached_input: Tensor,
    ) -> TransplantResult:
        self._validate(condition, jobs)
        if not 0 <= start_layer <= self.monitor_layer:
            raise ValueError("start_layer must be at or before the monitor")
        if any(
            member.site.layer < start_layer
            for job in jobs
            for member in job.members
        ):
            raise ValueError("transplant contains a site before the cached tail")
        expected_shape = (
            condition.batch_size,
            condition.input_ids.shape[1],
            int(self.runner.model.config.hidden_size),
        )
        if tuple(cached_input.shape) != expected_shape:
            raise ValueError(
                f"cached input shape {tuple(cached_input.shape)} != {expected_shape}"
            )

        expanded = repeat_condition(condition, len(jobs))
        expanded_cache = cached_input.repeat(len(jobs), 1, 1)
        scores: Tensor | None = None
        handles = self._register_jobs(condition, jobs)
        original_layers: list[tuple[int, nn.Module]] = []
        try:
            for layer_index in range(start_layer):
                original = self.runner.layers[layer_index]
                original_layers.append((layer_index, original))
                self.runner.layers[layer_index] = _PassthroughDecoderLayer(
                    original.attention_type
                )

            def replace_cached_input(
                _module: Any, args: tuple[Any, ...]
            ) -> tuple[Any, ...]:
                tensor = self.runner._first_tensor(args)
                replacement = expanded_cache.to(
                    device=tensor.device, dtype=tensor.dtype
                )
                return self.runner._replace_first_tensor(args, replacement)

            handles.append(
                self.runner.layers[start_layer].register_forward_pre_hook(
                    replace_cached_input
                )
            )

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
            for layer_index, original in original_layers:
                self.runner.layers[layer_index] = original
        if scores is None:
            raise RuntimeError("cached-tail transplant score was not captured")
        return TransplantResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=scores.reshape(len(jobs), condition.batch_size),
        )


def masked_token_kl(
    destination_logits: Tensor,
    patched_logits: Tensor,
    response_mask: Tensor,
    *,
    device: torch.device | str | None = None,
    token_chunk_size: int = 4,
) -> Tensor:
    """Mean KL(destination || patched) over valid response-token positions."""
    if destination_logits.shape != patched_logits.shape:
        raise ValueError("destination and patched logits must have identical shapes")
    if destination_logits.ndim != 3:
        raise ValueError("logits must have shape [batch, response, vocabulary]")
    if response_mask.shape != destination_logits.shape[:2]:
        raise ValueError("response mask does not match logits")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    compute_device = torch.device(device) if device is not None else destination_logits.device
    flat_destination = destination_logits.reshape(-1, destination_logits.shape[-1])
    flat_patched = patched_logits.reshape_as(flat_destination)
    flat_mask = response_mask.reshape(-1).bool()
    valid_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze(-1)
    values = []
    for start in range(0, len(valid_indices), token_chunk_size):
        indices = valid_indices[start : start + token_chunk_size]
        destination = flat_destination[indices].to(compute_device).float()
        patched = flat_patched[indices].to(compute_device).float()
        log_destination = torch.log_softmax(destination, dim=-1)
        log_patched = torch.log_softmax(patched, dim=-1)
        values.append(
            (log_destination.exp() * (log_destination - log_patched))
            .sum(dim=-1)
            .detach()
            .cpu()
        )
    token_kl = torch.cat(values) if values else torch.empty(0)
    result = torch.zeros(destination_logits.shape[0], dtype=torch.float32)
    cursor = 0
    for index, count in enumerate(response_mask.sum(dim=1).tolist()):
        result[index] = token_kl[cursor : cursor + count].mean()
        cursor += count
    return result


def sufficiency_specifications(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the 13 frozen Day 10 exact transplant specifications."""
    specifications = [dict(row) for row in plan["exact_transplants"]]
    if len(specifications) != 13:
        raise ValueError("Day 10 requires exactly 13 transplant specifications")
    if len({row["group_id"] for row in specifications}) != len(specifications):
        raise ValueError("Day 10 transplant group IDs are not unique")
    return specifications


def _interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = _interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def summarize_sufficiency(
    exact_records: Iterable[dict[str, Any]],
    dose_records: Iterable[dict[str, Any]],
    behavior_records: Iterable[dict[str, Any]],
    plan: Mapping[str, Any],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize exact induction, controls, context, dose response, and behavior."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    exact_records = list(exact_records)
    dose_records = list(dose_records)
    behavior_records = list(behavior_records)
    specifications = sufficiency_specifications(plan)
    specification_by_id = {row["group_id"]: row for row in specifications}
    expected_groups = set(specification_by_id)

    nested: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    split_by_concept: dict[str, str] = {}
    for record in exact_records:
        key = "baseline" if record["record_type"] == "baseline" else record["group_id"]
        nested[record["concept"]][record["label"]][record["example_id"]][key] = record
        previous = split_by_concept.setdefault(record["concept"], record["split"])
        if previous != record["split"]:
            raise ValueError("one concept appears in multiple split roles")
    if len(nested) != 11:
        raise ValueError("expected all 11 benign concepts")

    rng = np.random.default_rng(seed)
    concept_summaries = []
    cell_boots: dict[tuple[str, str, int], np.ndarray] = {}
    cell_points: dict[tuple[str, str, int], float] = {}
    for concept in sorted(nested):
        label_examples = nested[concept]
        ids = {label: sorted(label_examples[label]) for label in (1, 0)}
        if any(len(ids[label]) != 64 for label in (1, 0)):
            raise ValueError(f"expected 64 examples per class for {concept}")
        if any(
            set(label_examples[label][example_id]) != {"baseline", *expected_groups}
            for label in (1, 0)
            for example_id in ids[label]
        ):
            raise ValueError(f"incomplete exact sufficiency grid for {concept}")
        indices = {
            label: rng.integers(0, 64, size=(replicates, 64))
            for label in (1, 0)
        }
        normal = {
            label: np.asarray(
                [label_examples[label][example_id]["baseline"]["normal_probe_score"] for example_id in ids[label]]
            )
            for label in (1, 0)
        }
        triggered = {
            label: np.asarray(
                [label_examples[label][example_id]["baseline"]["triggered_probe_score"] for example_id in ids[label]]
            )
            for label in (1, 0)
        }
        normal_positive_boot = normal[1][indices[1]].mean(axis=1)
        triggered_positive_boot = triggered[1][indices[1]].mean(axis=1)
        denominator = float(normal[1].mean() - triggered[1].mean())
        denominator_boot = normal_positive_boot - triggered_positive_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable positive suppression denominator for {concept}")

        cells = []
        for specification in specifications:
            group_id = specification["group_id"]
            for label in (1, 0):
                patched = np.asarray(
                    [label_examples[label][example_id][group_id]["patched_probe_score"] for example_id in ids[label]]
                )
                delta = normal[label] - patched
                delta_boot = delta[indices[label]].mean(axis=1)
                fraction = float(delta.mean()) / denominator
                fraction_boot = delta_boot / denominator_boot
                cell_points[(concept, group_id, label)] = fraction
                cell_boots[(concept, group_id, label)] = fraction_boot
                cells.append(
                    {
                        **specification,
                        "label": label,
                        "n_examples": 64,
                        "normal_mean": float(normal[label].mean()),
                        "patched_mean": float(patched.mean()),
                        "raw_score_suppression": _estimate(float(delta.mean()), delta_boot),
                        "fraction": _estimate(fraction, fraction_boot),
                        "positive_example_fraction": float(np.mean(delta > 0)),
                    }
                )
        concept_summaries.append(
            {
                "concept": concept,
                "split": split_by_concept[concept],
                "positive_suppression_denominator": _estimate(denominator, denominator_boot),
                "baseline": {
                    str(label): {
                        "normal_mean": float(normal[label].mean()),
                        "triggered_mean": float(triggered[label].mean()),
                    }
                    for label in (1, 0)
                },
                "cells": cells,
            }
        )

    scope_concepts = {
        "discovery": sorted(concept for concept, split in split_by_concept.items() if split == "discovery"),
        "validation": sorted(concept for concept, split in split_by_concept.items() if split == "validation"),
    }
    scope_concepts["all_benign"] = sorted(split_by_concept)
    macro = []
    macro_boots: dict[tuple[str, str, int], np.ndarray] = {}
    for scope in ("discovery", "validation", "all_benign"):
        concepts = scope_concepts[scope]
        for specification in specifications:
            group_id = specification["group_id"]
            for label in (1, 0):
                points = [cell_points[(concept, group_id, label)] for concept in concepts]
                boot = np.stack([cell_boots[(concept, group_id, label)] for concept in concepts]).mean(axis=0)
                macro_boots[(scope, group_id, label)] = boot
                macro.append(
                    {
                        **specification,
                        "scope": scope,
                        "label": label,
                        "concept_count": len(concepts),
                        "positive_concept_count": sum(point > 0 for point in points),
                        "fraction": _estimate(float(np.mean(points)), boot),
                    }
                )
    macro_lookup = {(row["scope"], row["group_id"], row["label"]): row for row in macro}

    contrasts = []
    comparison_pairs = [
        (f"selected_single_rank{rank}", f"random_single_rank{rank}", f"rank_{rank}")
        for rank in range(1, 5)
    ] + [("selected_k16", "random_k16", "k16")]
    for scope in ("discovery", "validation", "all_benign"):
        for selected_id, random_id, comparison in comparison_pairs:
            for label in (1, 0):
                selected_point = macro_lookup[(scope, selected_id, label)]["fraction"]["estimate"]
                random_point = macro_lookup[(scope, random_id, label)]["fraction"]["estimate"]
                boot = macro_boots[(scope, selected_id, label)] - macro_boots[(scope, random_id, label)]
                contrasts.append(
                    {
                        "scope": scope,
                        "label": label,
                        "comparison": comparison,
                        "selected_group_id": selected_id,
                        "random_group_id": random_id,
                        "fraction_difference": _estimate(selected_point - random_point, boot),
                    }
                )

    context_increment = []
    for scope in ("discovery", "validation", "all_benign"):
        for label in (1, 0):
            combined = macro_lookup[(scope, "selected_k16_plus_resid_post_layer08", label)]["fraction"]["estimate"]
            selected = macro_lookup[(scope, "selected_k16", label)]["fraction"]["estimate"]
            boot = macro_boots[(scope, "selected_k16_plus_resid_post_layer08", label)] - macro_boots[(scope, "selected_k16", label)]
            context_increment.append(
                {
                    "scope": scope,
                    "label": label,
                    "contrast": "selected_k16_plus_resid08_minus_selected_k16",
                    "fraction_difference": _estimate(combined - selected, boot),
                }
            )

    exact_by_example = {(row["example_id"], row.get("group_id")): row for row in exact_records}
    dose_group_ids = list(plan["dose_response"]["evaluated_group_ids"])
    interior_alphas = (0.25, 0.5, 0.75)
    dose_nested: dict[
        str, dict[str, dict[tuple[str, float], dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for row in dose_records:
        dose_nested[row["concept"]][row["example_id"]][
            (row["group_id"], float(row["alpha"]))
        ] = row
    dose_rng = np.random.default_rng(seed)
    dose_concepts = []
    dose_cell_points: dict[tuple[str, str, float], float] = {}
    dose_cell_boots: dict[tuple[str, str, float], np.ndarray] = {}
    expected_dose_keys = {
        (group_id, alpha) for group_id in dose_group_ids for alpha in interior_alphas
    }
    for concept in sorted(nested):
        examples = dose_nested[concept]
        ids = sorted(examples)
        if len(ids) != 16:
            raise ValueError(f"expected 16 dose examples for {concept}")
        if any(set(examples[example_id]) != expected_dose_keys for example_id in ids):
            raise ValueError(f"incomplete interior dose grid for {concept}")
        indices = dose_rng.integers(0, 16, size=(replicates, 16))
        normal = np.asarray([exact_by_example[(example_id, None)]["normal_probe_score"] for example_id in ids])
        triggered = np.asarray([exact_by_example[(example_id, None)]["triggered_probe_score"] for example_id in ids])
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable dose denominator for {concept}")
        cells = []
        for group_id in dose_group_ids:
            previous_point = 0.0
            previous_boot = np.zeros(replicates)
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                if alpha == 0:
                    patched = normal
                elif alpha == 1:
                    patched = np.asarray([exact_by_example[(example_id, group_id)]["patched_probe_score"] for example_id in ids])
                else:
                    patched = np.asarray([examples[example_id][(group_id, alpha)]["patched_probe_score"] for example_id in ids])
                delta = normal - patched
                boot_delta = delta[indices].mean(axis=1)
                point = float(delta.mean()) / denominator
                boot = boot_delta / denominator_boot
                dose_cell_points[(concept, group_id, alpha)] = point
                dose_cell_boots[(concept, group_id, alpha)] = boot
                cells.append(
                    {
                        "group_id": group_id,
                        "alpha": alpha,
                        "n_examples": 16,
                        "patched_mean": float(patched.mean()),
                        "fraction": _estimate(point, boot),
                        "marginal_fraction": _estimate(point - previous_point, boot - previous_boot),
                    }
                )
                previous_point = point
                previous_boot = boot
        dose_concepts.append(
            {
                "concept": concept,
                "split": split_by_concept[concept],
                "positive_suppression_denominator": _estimate(denominator, denominator_boot),
                "cells": cells,
            }
        )

    dose_macro = []
    for scope in ("discovery", "validation", "all_benign"):
        concepts = scope_concepts[scope]
        for group_id in dose_group_ids:
            previous_point = 0.0
            previous_boot = np.zeros(replicates)
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
                points = [dose_cell_points[(concept, group_id, alpha)] for concept in concepts]
                boot = np.stack([dose_cell_boots[(concept, group_id, alpha)] for concept in concepts]).mean(axis=0)
                point = float(np.mean(points))
                dose_macro.append(
                    {
                        "scope": scope,
                        "group_id": group_id,
                        "alpha": alpha,
                        "concept_count": len(concepts),
                        "fraction": _estimate(point, boot),
                        "marginal_fraction": _estimate(point - previous_point, boot - previous_boot),
                    }
                )
                previous_point = point
                previous_boot = boot

    dose_lookup = {(row["scope"], row["group_id"], row["alpha"]): row for row in dose_macro}
    monotonicity = []
    for scope in ("discovery", "validation", "all_benign"):
        for group_id in dose_group_ids:
            values = [dose_lookup[(scope, group_id, alpha)]["fraction"]["estimate"] for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)]
            monotonicity.append(
                {
                    "scope": scope,
                    "group_id": group_id,
                    "point_estimates": values,
                    "nondecreasing": all(later >= earlier for earlier, later in zip(values, values[1:])),
                }
            )

    behavior_nested: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in behavior_records:
        key = (record["split"], record["concept"], record["label"])
        example_id = record["example_id"]
        cell = behavior_nested[key].setdefault(example_id, {})
        row_key = "baseline" if record["record_type"] == "baseline" else record["group_id"]
        cell[row_key] = record
    if len(behavior_nested) != 22:
        raise ValueError("expected 22 behavior concept/class cells")
    behavior_rng = np.random.default_rng(seed)
    behavior_cells = []
    behavior_boots: dict[tuple[str, str, int, str], np.ndarray] = {}
    for (split, concept, label), examples in sorted(behavior_nested.items()):
        ids = sorted(examples)
        if len(ids) != 2 or any(set(examples[example_id]) != {"baseline", *expected_groups} for example_id in ids):
            raise ValueError("behavior sufficiency grid is incomplete")
        indices = behavior_rng.integers(0, 2, size=(replicates, 2))
        normal_nll = np.asarray([examples[example_id]["baseline"]["normal_response_nll"] for example_id in ids])
        for specification in specifications:
            group_id = specification["group_id"]
            patched = np.asarray([examples[example_id][group_id]["patched_response_nll"] for example_id in ids])
            delta = patched - normal_nll
            boot = delta[indices].mean(axis=1)
            behavior_boots[(split, concept, label, group_id)] = boot
            behavior_cells.append(
                {
                    **specification,
                    "split": split,
                    "concept": concept,
                    "label": label,
                    "n_examples": 2,
                    "nll_change": _estimate(float(delta.mean()), boot),
                }
            )
    behavior_macro = []
    for scope in ("discovery", "validation", "all_benign"):
        concepts = scope_concepts[scope]
        for specification in specifications:
            group_id = specification["group_id"]
            for label in (1, 0):
                keys = [(split_by_concept[concept], concept, label, group_id) for concept in concepts]
                cells = [
                    next(row for row in behavior_cells if (row["split"], row["concept"], row["label"], row["group_id"]) == key)
                    for key in keys
                ]
                point = float(np.mean([row["nll_change"]["estimate"] for row in cells]))
                boot = np.stack([behavior_boots[key] for key in keys]).mean(axis=0)
                behavior_macro.append(
                    {
                        **specification,
                        "scope": scope,
                        "label": label,
                        "concept_count": len(concepts),
                        "nll_change": _estimate(point, boot),
                    }
                )

    k16_contrast_lookup = {
        (row["scope"], row["label"]): row for row in contrasts if row["comparison"] == "k16"
    }
    sufficiency_supported = all(
        macro_lookup[(scope, "selected_k16", 1)]["fraction"]["ci_low"] > 0
        and k16_contrast_lookup[(scope, 1)]["fraction_difference"]["ci_low"] > 0
        for scope in ("discovery", "validation")
    )
    if not sufficiency_supported:
        sufficiency_classification = "not_supported"
    elif all(macro_lookup[(scope, "selected_k16", 1)]["fraction"]["estimate"] >= 0.9 for scope in ("discovery", "validation")):
        sufficiency_classification = "near_complete_sufficiency"
    else:
        sufficiency_classification = "partial_sufficiency"
    selected_dose_monotonic = all(
        row["nondecreasing"]
        for row in monotonicity
        if row["group_id"] == "selected_k16" and row["scope"] in ("discovery", "validation")
    )

    exact_patch_rows = [row for row in exact_records if row["record_type"] != "baseline"]
    norm_summary = {
        "exact_source_to_destination_rms_ratio_max": max(float(row["source_to_destination_rms_ratio_max"]) for row in exact_patch_rows),
        "dose_interpolation_bound_ratio_max": max(float(row["interpolation_bound_ratio_max"]) for row in dose_records),
        "frozen_bound": 1.001,
    }

    return {
        "schema_version": 1,
        "procedure": "day10-v1",
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "macro_weighting": "equal concept weight",
        },
        "group_specifications": specifications,
        "concepts": concept_summaries,
        "macro": macro,
        "selected_random_contrasts": contrasts,
        "context_increment": context_increment,
        "dose_response": {
            "subset_example_count": 176,
            "concepts": dose_concepts,
            "macro": dose_macro,
            "monotonicity": monotonicity,
            "selected_k16_dose_response_supported": selected_dose_monotonic,
        },
        "behavior": {
            "subset_example_count": 44,
            "concept_class_cells": behavior_cells,
            "macro": behavior_macro,
            "interpretation": "patched minus normal-destination response NLL",
        },
        "sufficiency_evidence": {
            "supported": sufficiency_supported,
            "classification": sufficiency_classification,
            "selected_k16_dose_response_supported": selected_dose_monotonic,
            "rules": plan["evidence_rules"],
        },
        "activation_norms": norm_summary,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
