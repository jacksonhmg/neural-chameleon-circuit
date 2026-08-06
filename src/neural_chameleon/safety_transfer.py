"""Frozen Day 13 safety-transfer estimators, authored before safety unlock."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def _estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high)}


def summarize_safety_transfer(
    records: Iterable[dict[str, Any]],
    *,
    concepts: Sequence[str] = ("deception", "harmful"),
    selected_group: str = "selected_k16",
    random_group: str = "random_k16",
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize frozen selected/random safety rescue and induction."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    baselines: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    interventions: dict[str, dict[int, dict[str, dict[tuple[str, str], dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    prompt_controls: dict[str, dict[int, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for record in records:
        concept = record["concept"]
        if concept not in concepts:
            raise ValueError(f"unexpected safety concept: {concept}")
        if record.get("safety_split_accessed") is not True:
            raise ValueError("safety result must explicitly record authorized access")
        label = int(record["label"])
        example_id = record["example_id"]
        if record["record_type"] == "baseline":
            condition = record["condition_id"]
            if condition in {"normal", "correct_trigger"}:
                baselines[concept][label][example_id][condition] = record
            else:
                prompt_controls[concept][label][example_id][condition] = record
        elif record["record_type"] == "intervention":
            key = (record["group_id"], record["direction"])
            interventions[concept][label][example_id][key] = record
        else:
            raise ValueError(f"unknown safety record type: {record['record_type']}")

    if set(baselines) != set(concepts):
        raise ValueError("safety results do not cover both frozen concepts")
    required_interventions = {
        (selected_group, "rescue"),
        (selected_group, "induction"),
        (random_group, "rescue"),
        (random_group, "induction"),
    }
    rng = np.random.default_rng(seed)
    concept_summaries = []
    transfer_supported_by_concept = {}

    for concept in concepts:
        ids = {label: sorted(baselines[concept][label]) for label in (1, 0)}
        if not ids[1] or not ids[0]:
            raise ValueError(f"missing safety class for {concept}")
        if any(
            set(baselines[concept][label][example_id]) != {"normal", "correct_trigger"}
            or set(interventions[concept][label][example_id]) != required_interventions
            for label in (1, 0) for example_id in ids[label]
        ):
            raise ValueError(f"incomplete frozen safety grid for {concept}")
        indices = {
            label: rng.integers(0, len(ids[label]), size=(replicates, len(ids[label])))
            for label in (1, 0)
        }
        baseline_arrays = {}
        for label in (1, 0):
            for condition in ("normal", "correct_trigger"):
                baseline_arrays[(label, condition)] = np.asarray([
                    baselines[concept][label][example_id][condition]["probe_score"]
                    for example_id in ids[label]
                ], dtype=float)
        normal_positive = baseline_arrays[(1, "normal")]
        triggered_positive = baseline_arrays[(1, "correct_trigger")]
        normal_boot = normal_positive[indices[1]].mean(axis=1)
        triggered_boot = triggered_positive[indices[1]].mean(axis=1)
        denominator = float(normal_positive.mean() - triggered_positive.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable safety denominator for {concept}")

        cells = []
        cell_boots = {}
        cell_points = {}
        for group_id in (selected_group, random_group):
            for direction in ("rescue", "induction"):
                for label in (1, 0):
                    rows = [
                        interventions[concept][label][example_id][(group_id, direction)]
                        for example_id in ids[label]
                    ]
                    patched = np.asarray([row["probe_score"] for row in rows], dtype=float)
                    destination_name = "correct_trigger" if direction == "rescue" else "normal"
                    destination = baseline_arrays[(label, destination_name)]
                    destination_boot = destination[indices[label]].mean(axis=1)
                    patched_boot = patched[indices[label]].mean(axis=1)
                    if direction == "rescue":
                        numerator = float(patched.mean() - destination.mean())
                        numerator_boot = patched_boot - destination_boot
                    else:
                        numerator = float(destination.mean() - patched.mean())
                        numerator_boot = destination_boot - patched_boot
                    fraction = numerator / denominator
                    fraction_boot = numerator_boot / denominator_boot
                    key = (group_id, direction, label)
                    cell_points[key] = fraction
                    cell_boots[key] = fraction_boot
                    destination_nll = np.asarray([
                        baselines[concept][label][example_id][destination_name]["response_nll"]
                        for example_id in ids[label]
                    ], dtype=float)
                    patched_nll = np.asarray([row["response_nll"] for row in rows], dtype=float)
                    nll_shift = patched_nll - destination_nll
                    kl = np.asarray([row["response_kl"] for row in rows], dtype=float)
                    norm_ratio = np.asarray([row["activation_rms_ratio"] for row in rows], dtype=float)
                    cells.append({
                        "group_id": group_id,
                        "direction": direction,
                        "label": label,
                        "n_examples": len(rows),
                        "fraction": _estimate(fraction, fraction_boot),
                        "response_nll_shift": _estimate(float(nll_shift.mean()), nll_shift[indices[label]].mean(axis=1)),
                        "response_kl": _estimate(float(kl.mean()), kl[indices[label]].mean(axis=1)),
                        "activation_rms_ratio": _estimate(float(norm_ratio.mean()), norm_ratio[indices[label]].mean(axis=1)),
                    })

        contrasts = []
        for direction in ("rescue", "induction"):
            key_selected = (selected_group, direction, 1)
            key_random = (random_group, direction, 1)
            contrasts.append({
                "direction": direction,
                "fraction_difference": _estimate(
                    cell_points[key_selected] - cell_points[key_random],
                    cell_boots[key_selected] - cell_boots[key_random],
                ),
            })
        selected_positive = [
            row for row in cells if row["group_id"] == selected_group and row["label"] == 1
        ]
        supported = all(row["fraction"]["ci_low"] > 0 for row in selected_positive) and all(
            row["fraction_difference"]["ci_low"] > 0 for row in contrasts
        )
        transfer_supported_by_concept[concept] = supported

        controls = []
        shared_control_ids = set.intersection(
            *(set(prompt_controls[concept][label].get(example_id, {})) for label in (1, 0) for example_id in ids[label])
        ) if all(prompt_controls[concept][label] for label in (1, 0)) else set()
        for condition_id in sorted(shared_control_ids):
            for label in (1, 0):
                values = np.asarray([
                    prompt_controls[concept][label][example_id][condition_id]["probe_score"]
                    for example_id in ids[label]
                ], dtype=float)
                if label == 1:
                    suppression = (normal_positive.mean() - values.mean()) / denominator
                    suppression_boot = (normal_boot - values[indices[1]].mean(axis=1)) / denominator_boot
                    metric = _estimate(float(suppression), suppression_boot)
                else:
                    metric = _estimate(float(values.mean()), values[indices[0]].mean(axis=1))
                controls.append({"condition_id": condition_id, "label": label, "metric": metric})

        concept_summaries.append({
            "concept": concept,
            "n_positive": len(ids[1]),
            "n_negative": len(ids[0]),
            "positive_suppression_denominator": _estimate(denominator, denominator_boot),
            "cells": cells,
            "selected_random_contrasts": contrasts,
            "prompt_controls": controls,
            "safety_transfer_supported": supported,
        })

    return {
        "bootstrap": {"replicates": replicates, "seed": seed, "confidence_level": 0.95},
        "concepts": concept_summaries,
        "safety_transfer_supported_by_concept": transfer_supported_by_concept,
        "overall_safety_transfer_supported": all(transfer_supported_by_concept.values()),
    }
