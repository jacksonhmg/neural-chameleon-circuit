"""Day 9 vectorized grouped component patching and necessity analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .individual_components import ComponentCandidate, masked_example_mean, repeat_condition
from .interventions import CapturedActivation, ConditionBatch, LinearProbe, PairedInterventionRunner


@dataclass(frozen=True)
class GroupPatchJob:
    """One named multi-site patch evaluated independently in a vector batch."""

    group_id: str
    members: tuple[tuple[ComponentCandidate, CapturedActivation], ...]


@dataclass(frozen=True)
class GroupPatchResult:
    """Per-group scores or response NLL from a vectorized grouped patch pass."""

    group_ids: tuple[str, ...]
    probe_scores: Tensor
    response_nll: Tensor | None = None


class _MonitorReached(RuntimeError):
    pass


class GroupedComponentPatchRunner:
    """Evaluate independent multi-site component groups in one expanded batch."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _validate(self, condition: ConditionBatch, jobs: Sequence[GroupPatchJob]) -> None:
        if not jobs:
            raise ValueError("at least one grouped patch job is required")
        if len({job.group_id for job in jobs}) != len(jobs):
            raise ValueError("grouped patch jobs contain duplicate IDs")
        for job in jobs:
            if not job.members:
                raise ValueError(f"group {job.group_id} has no component members")
            sites = [candidate.site for candidate, _capture in job.members]
            if len(sites) != len(set(sites)):
                raise ValueError(f"group {job.group_id} contains duplicate sites")
            self.runner._validate_sites(sites)
            for candidate, capture in job.members:
                if candidate.layer > self.monitor_layer:
                    raise ValueError("group contains a component after the monitor")
                self.runner._validate_patch_pair(condition, candidate.site, capture)

    def _register_jobs(
        self, base: ConditionBatch, jobs: Sequence[GroupPatchJob]
    ) -> list[Any]:
        grouped: dict[
            tuple[Any, str],
            list[tuple[int, ComponentCandidate, CapturedActivation]],
        ] = defaultdict(list)
        for job_index, job in enumerate(jobs):
            for candidate, capture in job.members:
                module, side = self.runner._resolve_module(candidate.site)
                grouped[(module, side)].append((job_index, candidate, capture))

        handles = []
        for (module, side), entries in grouped.items():

            def patch_tensor(
                tensor: Tensor,
                entries: Sequence[
                    tuple[int, ComponentCandidate, CapturedActivation]
                ] = tuple(entries),
            ) -> Tensor:
                patched = tensor.clone()
                start = base.response_start
                stop = start + base.response_width
                base_batch = base.batch_size
                mask = base.response_mask.to(patched.device).unsqueeze(-1)
                for job_index, candidate, capture in entries:
                    rows = slice(job_index * base_batch, (job_index + 1) * base_batch)
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
        self, condition: ConditionBatch, jobs: Sequence[GroupPatchJob]
    ) -> GroupPatchResult:
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
            raise RuntimeError("grouped patch score was not captured")
        return GroupPatchResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=scores.reshape(len(jobs), condition.batch_size),
        )

    def run_full(
        self, condition: ConditionBatch, jobs: Sequence[GroupPatchJob]
    ) -> GroupPatchResult:
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
        return GroupPatchResult(
            group_ids=tuple(job.group_id for job in jobs),
            probe_scores=torch.empty((len(jobs), condition.batch_size)),
            response_nll=nll,
        )


def group_specifications(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the 13 frozen Day 9 groups in stable execution order."""
    specifications = []
    for role, key in (
        ("selected_prefix", "selected_prefixes"),
        ("random_prefix", "random_prefixes"),
    ):
        for size in (1, 2, 4, 8, 16):
            specifications.append(
                {
                    "group_id": f"{role.replace('_prefix', '')}_k{size}",
                    "group_role": role,
                    "set_size": size,
                    "candidate_ids": list(plan[key][str(size)]),
                }
            )
    specifications.extend(
        [
            {
                "group_id": plan["full_block_group"]["group_id"],
                "group_role": "full_relevant_block",
                "set_size": len(plan["full_block_group"]["candidate_ids"]),
                "candidate_ids": list(plan["full_block_group"]["candidate_ids"]),
            },
            {
                "group_id": plan["block_control"]["group_id"],
                "group_role": "full_block_control",
                "set_size": len(plan["block_control"]["candidate_ids"]),
                "candidate_ids": list(plan["block_control"]["candidate_ids"]),
            },
            {
                "group_id": plan["positive_control"]["group_id"],
                "group_role": "positive_control",
                "set_size": 1,
                "candidate_ids": [],
            },
        ]
    )
    return specifications


def _interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = _interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def summarize_grouped_necessity(
    records: Iterable[dict[str, Any]],
    behavior_records: Iterable[dict[str, Any]],
    plan: Mapping[str, Any],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize grouped rescue, controls, compactness, and behavior."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    records = list(records)
    behavior_records = list(behavior_records)
    specifications = group_specifications(plan)
    specification_by_id = {row["group_id"]: row for row in specifications}
    expected_groups = set(specification_by_id)

    nested: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    split_by_concept: dict[str, str] = {}
    for record in records:
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
            raise ValueError(f"incomplete grouped grid for {concept}")
        indices = {
            label: rng.integers(0, 64, size=(replicates, 64))
            for label in (1, 0)
        }
        normal = {
            label: np.asarray(
                [
                    label_examples[label][example_id]["baseline"][
                        "normal_probe_score"
                    ]
                    for example_id in ids[label]
                ]
            )
            for label in (1, 0)
        }
        triggered = {
            label: np.asarray(
                [
                    label_examples[label][example_id]["baseline"][
                        "triggered_probe_score"
                    ]
                    for example_id in ids[label]
                ]
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
                    [
                        label_examples[label][example_id][group_id][
                            "patched_probe_score"
                        ]
                        for example_id in ids[label]
                    ]
                )
                delta = patched - triggered[label]
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
                        "patched_mean": float(patched.mean()),
                        "triggered_mean": float(triggered[label].mean()),
                        "raw_score_change": _estimate(float(delta.mean()), delta_boot),
                        "fraction": _estimate(fraction, fraction_boot),
                        "positive_example_fraction": float(np.mean(delta > 0)),
                    }
                )
        concept_summaries.append(
            {
                "concept": concept,
                "split": split_by_concept[concept],
                "positive_suppression_denominator": _estimate(
                    denominator, denominator_boot
                ),
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
        "discovery": sorted(
            concept for concept, split in split_by_concept.items() if split == "discovery"
        ),
        "validation": sorted(
            concept for concept, split in split_by_concept.items() if split == "validation"
        ),
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
                boot = np.stack(
                    [cell_boots[(concept, group_id, label)] for concept in concepts]
                ).mean(axis=0)
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

    macro_lookup = {
        (row["scope"], row["group_id"], row["label"]): row for row in macro
    }
    contrasts = []
    for scope in ("discovery", "validation", "all_benign"):
        for size in (1, 2, 4, 8, 16):
            for label in (1, 0):
                selected_id = f"selected_k{size}"
                random_id = f"random_k{size}"
                selected_point = macro_lookup[(scope, selected_id, label)]["fraction"][
                    "estimate"
                ]
                random_point = macro_lookup[(scope, random_id, label)]["fraction"][
                    "estimate"
                ]
                boot = (
                    macro_boots[(scope, selected_id, label)]
                    - macro_boots[(scope, random_id, label)]
                )
                contrasts.append(
                    {
                        "scope": scope,
                        "label": label,
                        "set_size": size,
                        "contrast": "selected_minus_random",
                        "fraction_difference": _estimate(
                            selected_point - random_point, boot
                        ),
                    }
                )

    curve = []
    relative_lookup: dict[tuple[str, int], float] = {}
    for scope in ("discovery", "validation"):
        final_id = "selected_k16"
        final_point = macro_lookup[(scope, final_id, 1)]["fraction"]["estimate"]
        final_boot = macro_boots[(scope, final_id, 1)]
        if final_point <= 0 or np.any(final_boot <= 0):
            raise ValueError(f"selected K=16 recovery is unstable for {scope}")
        previous_point = 0.0
        previous_boot = np.zeros(replicates)
        for size in (1, 2, 4, 8, 16):
            group_id = f"selected_k{size}"
            point = macro_lookup[(scope, group_id, 1)]["fraction"]["estimate"]
            boot = macro_boots[(scope, group_id, 1)]
            relative = point / final_point
            relative_boot = boot / final_boot
            relative_lookup[(scope, size)] = relative
            curve.append(
                {
                    "scope": scope,
                    "set_size": size,
                    "group_id": group_id,
                    "recovery": macro_lookup[(scope, group_id, 1)]["fraction"],
                    "relative_to_k16": _estimate(relative, relative_boot),
                    "marginal_recovery": _estimate(
                        point - previous_point, boot - previous_boot
                    ),
                }
            )
            previous_point = point
            previous_boot = boot

    compact_size = next(
        (
            size
            for size in (1, 2, 4)
            if all(relative_lookup[(scope, size)] >= 0.8 for scope in ("discovery", "validation"))
        ),
        None,
    )
    if compact_size is not None:
        classification = "compact"
        threshold_size = compact_size
    elif all(
        relative_lookup[(scope, 8)] >= 0.8
        for scope in ("discovery", "validation")
    ):
        classification = "moderately_distributed"
        threshold_size = 8
    else:
        classification = "highly_distributed"
        threshold_size = 16

    behavior_nested: dict[
        tuple[str, str, int], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
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
        if len(ids) != 2 or any(
            set(examples[example_id]) != {"baseline", *expected_groups}
            for example_id in ids
        ):
            raise ValueError("behavior grouped grid is incomplete")
        indices = behavior_rng.integers(0, 2, size=(replicates, 2))
        triggered_nll = np.asarray(
            [examples[example_id]["baseline"]["triggered_response_nll"] for example_id in ids]
        )
        for specification in specifications:
            group_id = specification["group_id"]
            patched = np.asarray(
                [examples[example_id][group_id]["patched_response_nll"] for example_id in ids]
            )
            delta = patched - triggered_nll
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
                keys = [
                    (split_by_concept[concept], concept, label, group_id)
                    for concept in concepts
                ]
                cells = [
                    next(
                        row
                        for row in behavior_cells
                        if (
                            row["split"],
                            row["concept"],
                            row["label"],
                            row["group_id"],
                        )
                        == key
                    )
                    for key in keys
                ]
                point = float(
                    np.mean([row["nll_change"]["estimate"] for row in cells])
                )
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

    return {
        "schema_version": 1,
        "procedure": "day09-v1",
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
        "selected_curve": curve,
        "distribution_classification": {
            "classification": classification,
            "first_size_reaching_rule": threshold_size,
            "threshold": 0.8,
            "rule": plan["classification_rule"],
        },
        "behavior": {
            "subset_example_count": 44,
            "concept_class_cells": behavior_cells,
            "macro": behavior_macro,
            "interpretation": "patched minus correct-trigger destination response NLL",
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
