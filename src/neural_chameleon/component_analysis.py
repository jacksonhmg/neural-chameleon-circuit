"""Early-terminated component patching and paired Day 7 estimators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .interventions import (
    ActivationCache,
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
)


SELECTED_LAYERS = (12, 11, 10, 9)
RANDOM_CONTROL_LAYERS = (5, 0, 8, 3)
COMPONENT_KINDS = (
    ActivationKind.ATTN_OUT,
    ActivationKind.MLP_OUT,
    ActivationKind.BLOCK_OUTPUT,
)


@dataclass(frozen=True)
class TruncatedComponentResult:
    """Probe scores and captured response-aligned component activations."""

    probe_scores: Tensor
    captures: ActivationCache


class _MonitorReached(RuntimeError):
    pass


class TruncatedComponentRunner:
    """Patch arbitrary supported sites and stop after scoring the monitor block."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        if monitor_layer < 0 or monitor_layer >= len(runner.layers):
            raise ValueError("monitor layer is outside the model")
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def run(
        self,
        condition: ConditionBatch,
        *,
        capture_sites: Sequence[PatchSite] = (),
        patch_cache: Mapping[PatchSite, CapturedActivation] | None = None,
    ) -> TruncatedComponentResult:
        sites = tuple(capture_sites)
        if len(set(sites)) != len(sites):
            raise ValueError("capture_sites contains duplicates")
        patch_cache = patch_cache or {}
        all_sites = (*sites, *patch_cache.keys())
        self.runner._validate_sites(all_sites)
        if any(site.layer > self.monitor_layer for site in all_sites):
            raise ValueError("component sites must be at or before the monitor")
        for site, capture in patch_cache.items():
            self.runner._validate_patch_pair(condition, site, capture)

        captures: ActivationCache = {}
        scores: Tensor | None = None
        handles = []
        try:
            for site, capture in patch_cache.items():
                handles.append(self.runner._register_patch(condition, site, capture))
            for site in sites:
                handles.append(self.runner._register_capture(condition, site, captures))

            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                tensor = PairedInterventionRunner._first_tensor(output)
                start = condition.response_start
                stop = start + condition.response_width
                response_values = tensor[:, start:stop, :]
                weight = self.probe.weight.to(
                    device=response_values.device, dtype=torch.bfloat16
                )
                bias = self.probe.bias.to(
                    device=response_values.device, dtype=torch.bfloat16
                )
                logits = torch.matmul(response_values.to(torch.bfloat16), weight.T) + bias
                probabilities = torch.sigmoid(logits.squeeze(-1)).float()
                mask = condition.response_mask.to(probabilities.device)
                scores = (
                    (probabilities * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp(min=1)
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
                raise RuntimeError("model completed without reaching the monitor hook")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None or set(captures) != set(sites):
            raise RuntimeError("monitor score or requested component capture is missing")
        return TruncatedComponentResult(probe_scores=scores, captures=captures)


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = percentile_interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def _cell_key(
    grid: str,
    direction: str,
    layer: int,
    component_type: str,
    label: int,
) -> tuple[str, str, int, str, int]:
    return grid, direction, layer, component_type, label


def summarize_component_types(
    records: Iterable[dict[str, Any]],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute concept and equal-concept paired component-type effects."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    nested: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    split_by_concept: dict[str, str] = {}
    for record in records:
        nested[record["concept"]][record["label"]][record["example_id"]][
            record["key"]
        ] = record
        previous = split_by_concept.setdefault(record["concept"], record["split"])
        if previous != record["split"]:
            raise ValueError("one concept appears in multiple split roles")

    rng = np.random.default_rng(seed)
    concept_summaries = []
    point_by_concept: dict[
        str, dict[tuple[str, str, int, str, int], float]
    ] = defaultdict(dict)
    boot_by_concept: dict[
        str, dict[tuple[str, str, int, str, int], np.ndarray]
    ] = defaultdict(dict)

    for concept in sorted(nested):
        label_examples = nested[concept]
        ids = {label: sorted(label_examples[label]) for label in (1, 0)}
        if not ids[1] or not ids[0]:
            raise ValueError(f"missing class for {concept}")
        indices = {
            label: rng.integers(0, len(ids[label]), size=(replicates, len(ids[label])))
            for label in (1, 0)
        }

        baselines: dict[int, dict[str, np.ndarray]] = {1: {}, 0: {}}
        baseline_summary: dict[str, Any] = {}
        for label in (1, 0):
            label_summary = {}
            for condition in ("normal", "correct_trigger", "irrelevant_trigger"):
                values = np.asarray(
                    [
                        label_examples[label][example_id][f"baseline.{condition}"][
                            "probe_score"
                        ]
                        for example_id in ids[label]
                    ],
                    dtype=float,
                )
                baselines[label][condition] = values
                boot = values[indices[label]].mean(axis=1)
                label_summary[condition] = estimate(float(values.mean()), boot)
            baseline_summary[str(label)] = label_summary

        normal_positive = baselines[1]["normal"]
        triggered_positive = baselines[1]["correct_trigger"]
        normal_positive_boot = normal_positive[indices[1]].mean(axis=1)
        triggered_positive_boot = triggered_positive[indices[1]].mean(axis=1)
        denominator = float(normal_positive.mean() - triggered_positive.mean())
        denominator_boot = normal_positive_boot - triggered_positive_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"nonpositive suppression denominator for {concept}")

        cells = []
        specifications = []
        for layer in SELECTED_LAYERS:
            for kind in COMPONENT_KINDS:
                for direction in ("rescue", "induction"):
                    for label in (1, 0):
                        specifications.append(
                            ("correct", direction, layer, kind.value, label)
                        )
                specifications.append(
                    ("irrelevant", "rescue", layer, kind.value, 1)
                )
        if split_by_concept[concept] == "discovery":
            for layer in RANDOM_CONTROL_LAYERS:
                for kind in COMPONENT_KINDS:
                    specifications.append(
                        ("random", "rescue", layer, kind.value, 1)
                    )

        for grid, direction, layer, component_type, label in specifications:
            key = f"{grid}.{direction}.layer_{layer}.{component_type}"
            patched = np.asarray(
                [
                    label_examples[label][example_id][key]["probe_score"]
                    for example_id in ids[label]
                ],
                dtype=float,
            )
            patched_boot = patched[indices[label]].mean(axis=1)
            if grid == "irrelevant":
                destination = baselines[label]["irrelevant_trigger"]
                destination_boot = destination[indices[label]].mean(axis=1)
                numerator = float(patched.mean() - destination.mean())
                numerator_boot = patched_boot - destination_boot
            elif direction == "rescue":
                destination = baselines[label]["correct_trigger"]
                destination_boot = destination[indices[label]].mean(axis=1)
                numerator = float(patched.mean() - destination.mean())
                numerator_boot = patched_boot - destination_boot
            else:
                destination = baselines[label]["normal"]
                destination_boot = destination[indices[label]].mean(axis=1)
                numerator = float(destination.mean() - patched.mean())
                numerator_boot = destination_boot - patched_boot
            fraction = numerator / denominator
            fraction_boot = numerator_boot / denominator_boot
            cell = {
                "grid": grid,
                "direction": direction,
                "layer": layer,
                "component_type": component_type,
                "label": label,
                "patched_mean": float(patched.mean()),
                "destination_mean": float(destination.mean()),
                "numerator": numerator,
                "positive_denominator": denominator,
                "fraction": estimate(fraction, fraction_boot),
            }
            cells.append(cell)
            cell_key = _cell_key(grid, direction, layer, component_type, label)
            point_by_concept[concept][cell_key] = fraction
            boot_by_concept[concept][cell_key] = fraction_boot

        concept_summaries.append(
            {
                "concept": concept,
                "split": split_by_concept[concept],
                "n_positive": len(ids[1]),
                "n_negative": len(ids[0]),
                "baselines": baseline_summary,
                "positive_suppression_denominator": estimate(
                    denominator, denominator_boot
                ),
                "cells": cells,
            }
        )

    concepts_by_scope = {
        "discovery": sorted(
            concept for concept, split in split_by_concept.items() if split == "discovery"
        ),
        "validation": sorted(
            concept for concept, split in split_by_concept.items() if split == "validation"
        ),
        "all_benign": sorted(split_by_concept),
    }
    macro_summaries = []
    macro_points: dict[tuple[str, tuple[str, str, int, str, int]], float] = {}
    macro_boots: dict[tuple[str, tuple[str, str, int, str, int]], np.ndarray] = {}
    for scope, concepts in concepts_by_scope.items():
        shared_keys = set.intersection(
            *(set(point_by_concept[concept]) for concept in concepts)
        )
        cells = []
        for key in sorted(shared_keys):
            point = float(np.mean([point_by_concept[concept][key] for concept in concepts]))
            samples = np.stack(
                [boot_by_concept[concept][key] for concept in concepts]
            ).mean(axis=0)
            grid, direction, layer, component_type, label = key
            cells.append(
                {
                    "grid": grid,
                    "direction": direction,
                    "layer": layer,
                    "component_type": component_type,
                    "label": label,
                    "fraction": estimate(point, samples),
                }
            )
            macro_points[(scope, key)] = point
            macro_boots[(scope, key)] = samples
        macro_summaries.append(
            {"scope": scope, "concept_count": len(concepts), "cells": cells}
        )

    contrasts = []
    for scope in concepts_by_scope:
        for direction in ("rescue", "induction"):
            for layer in SELECTED_LAYERS:
                for label in (1, 0):
                    keys = {
                        kind.value: _cell_key(
                            "correct", direction, layer, kind.value, label
                        )
                        for kind in COMPONENT_KINDS
                    }
                    if not all((scope, key) in macro_points for key in keys.values()):
                        continue
                    attn = keys[ActivationKind.ATTN_OUT.value]
                    mlp = keys[ActivationKind.MLP_OUT.value]
                    block = keys[ActivationKind.BLOCK_OUTPUT.value]
                    definitions = {
                        "attention_minus_mlp": (
                            macro_points[(scope, attn)] - macro_points[(scope, mlp)],
                            macro_boots[(scope, attn)] - macro_boots[(scope, mlp)],
                        ),
                        "block_minus_attention": (
                            macro_points[(scope, block)] - macro_points[(scope, attn)],
                            macro_boots[(scope, block)] - macro_boots[(scope, attn)],
                        ),
                        "block_minus_mlp": (
                            macro_points[(scope, block)] - macro_points[(scope, mlp)],
                            macro_boots[(scope, block)] - macro_boots[(scope, mlp)],
                        ),
                        "block_minus_branch_sum": (
                            macro_points[(scope, block)]
                            - macro_points[(scope, attn)]
                            - macro_points[(scope, mlp)],
                            macro_boots[(scope, block)]
                            - macro_boots[(scope, attn)]
                            - macro_boots[(scope, mlp)],
                        ),
                    }
                    for name, (point, samples) in definitions.items():
                        contrasts.append(
                            {
                                "scope": scope,
                                "grid": "correct",
                                "direction": direction,
                                "layer": layer,
                                "label": label,
                                "contrast": name,
                                "value": estimate(point, samples),
                            }
                        )

    control_contrasts = []
    scope = "discovery"
    for kind in COMPONENT_KINDS:
        random_keys = [
            _cell_key("random", "rescue", layer, kind.value, 1)
            for layer in RANDOM_CONTROL_LAYERS
        ]
        random_point = float(np.mean([macro_points[(scope, key)] for key in random_keys]))
        random_boot = np.stack(
            [macro_boots[(scope, key)] for key in random_keys]
        ).mean(axis=0)
        for layer in SELECTED_LAYERS:
            selected_key = _cell_key("correct", "rescue", layer, kind.value, 1)
            point = macro_points[(scope, selected_key)] - random_point
            samples = macro_boots[(scope, selected_key)] - random_boot
            control_contrasts.append(
                {
                    "scope": scope,
                    "component_type": kind.value,
                    "selected_layer": layer,
                    "control": "selected_minus_mean_random_layers",
                    "random_layers": list(RANDOM_CONTROL_LAYERS),
                    "value": estimate(point, samples),
                }
            )
    for scope in concepts_by_scope:
        for layer in SELECTED_LAYERS:
            for kind in COMPONENT_KINDS:
                correct_key = _cell_key("correct", "rescue", layer, kind.value, 1)
                irrelevant_key = _cell_key(
                    "irrelevant", "rescue", layer, kind.value, 1
                )
                point = (
                    macro_points[(scope, correct_key)]
                    - macro_points[(scope, irrelevant_key)]
                )
                samples = (
                    macro_boots[(scope, correct_key)]
                    - macro_boots[(scope, irrelevant_key)]
                )
                control_contrasts.append(
                    {
                        "scope": scope,
                        "component_type": kind.value,
                        "selected_layer": layer,
                        "control": "correct_minus_irrelevant_trigger",
                        "value": estimate(point, samples),
                    }
                )

    return {
        "schema_version": 1,
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "macro_weighting": "equal concept weight",
            "class_resampling": "separate positive and negative strata",
        },
        "selected_layers": list(SELECTED_LAYERS),
        "random_control_layers": list(RANDOM_CONTROL_LAYERS),
        "component_types": [kind.value for kind in COMPONENT_KINDS],
        "concepts": concept_summaries,
        "macro": macro_summaries,
        "component_contrasts": contrasts,
        "control_contrasts": control_contrasts,
    }
