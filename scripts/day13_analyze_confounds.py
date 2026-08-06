#!/usr/bin/env python3
"""Analyze and visualize the frozen Day 13 secondary diagnostics."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.safety_transfer import _estimate  # noqa: E402


RESULT_DIR = ROOT / "results/day-13"
PLAN_PATH = RESULT_DIR / "frozen-execution-plan.json"
FREEZE_PATH = ROOT / "results/day-12/frozen-confirmatory-analysis.json"
CENTRAL_PATH = RESULT_DIR / "safety-transfer-results.jsonl.gz"
CENTRAL_SUMMARY_PATH = RESULT_DIR / "safety-transfer-summary.json"
PROMPT_PATH = RESULT_DIR / "prompt-confound-results.jsonl.gz"
ZERO_PATH = RESULT_DIR / "zero-ablation-results.jsonl.gz"
GENERATION_PATH = RESULT_DIR / "generation-diagnostics.jsonl.gz"
SUMMARY_PATH = RESULT_DIR / "confound-summary.json"
FIGURE_PNG = RESULT_DIR / "confound-diagnostics.png"
FIGURE_PDF = RESULT_DIR / "confound-diagnostics.pdf"
CONCEPTS = ("deception", "harmful")
PROMPT_CONDITIONS = (
    "correct_trigger",
    "concept_only",
    "monitoring_only",
    "irrelevant_concept",
    "nearby_concept",
    "paraphrased_trigger",
    "malformed_trigger",
    "relocated_trigger",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    positive_array = np.asarray(positive, dtype=float)
    negative_array = np.asarray(negative, dtype=float)
    if not len(positive_array) or not len(negative_array):
        raise ValueError("AUROC requires both classes")
    comparisons = positive_array[:, None] - negative_array[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def deterministic_subset_ids(
    central: Sequence[dict[str, Any]], salt: str, per_cell: int
) -> set[str]:
    examples = {}
    for row in central:
        examples[row["example_id"]] = (row["concept"], int(row["label"]))
    cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    for example_id, key in examples.items():
        cells[key].append(example_id)
    selected = set()
    for key in sorted(cells):
        ranked = sorted(
            cells[key],
            key=lambda example_id: (
                hashlib.sha256(f"{salt}:{example_id}".encode()).hexdigest(),
                example_id,
            ),
        )
        selected.update(ranked[:per_cell])
    return selected


def index_central(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        condition = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
        key = (row["example_id"], condition)
        if key in indexed:
            raise ValueError(f"duplicate central row {key}")
        indexed[key] = row
    return indexed


def bootstrap_ratio(
    numerator_values: np.ndarray,
    normal_values: np.ndarray,
    triggered_values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[dict[str, float], float]:
    indices = rng.integers(0, len(normal_values), size=(replicates, len(normal_values)))
    denominator = normal_values.mean() - triggered_values.mean()
    denominator_boot = normal_values[indices].mean(axis=1) - triggered_values[indices].mean(axis=1)
    numerator = numerator_values.mean()
    numerator_boot = numerator_values[indices].mean(axis=1)
    stable = denominator_boot > 0
    if denominator <= 0 or not stable.any():
        return {
            "estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }, float(stable.mean())
    return _estimate(float(numerator / denominator), numerator_boot[stable] / denominator_boot[stable]), float(stable.mean())


def prompt_summary(
    central_index: dict[tuple[str, str], dict[str, Any]],
    prompt_rows: Sequence[dict[str, Any]],
    subset_ids: set[str],
    rng: np.random.Generator,
    replicates: int,
) -> list[dict[str, Any]]:
    prompt_index = {(row["example_id"], row["condition_id"]): row for row in prompt_rows}
    summaries = []
    for concept in CONCEPTS:
        ids = {
            label: sorted(
                example_id
                for example_id in subset_ids
                if central_index[(example_id, "normal")]["concept"] == concept
                and int(central_index[(example_id, "normal")]["label"]) == label
            )
            for label in (1, 0)
        }
        normal = np.asarray(
            [central_index[(example_id, "normal")]["probe_score"] for example_id in ids[1]]
        )
        triggered = np.asarray(
            [central_index[(example_id, "correct_trigger")]["probe_score"] for example_id in ids[1]]
        )
        conditions = []
        for condition_id in PROMPT_CONDITIONS:
            source = central_index if condition_id == "correct_trigger" else prompt_index
            positive = np.asarray(
                [source[(example_id, condition_id)]["probe_score"] for example_id in ids[1]]
            )
            negative = np.asarray(
                [source[(example_id, condition_id)]["probe_score"] for example_id in ids[0]]
            )
            suppression, stable_fraction = bootstrap_ratio(
                normal - positive, normal, triggered, rng, replicates
            )
            positive_nll_shift = np.asarray(
                [
                    source[(example_id, condition_id)]["response_nll"]
                    - central_index[(example_id, "normal")]["response_nll"]
                    for example_id in ids[1]
                ]
            )
            conditions.append(
                {
                    "condition_id": condition_id,
                    "positive_probe_mean": float(positive.mean()),
                    "negative_probe_mean": float(negative.mean()),
                    "positive_suppression_fraction": suppression,
                    "bootstrap_positive_denominator_fraction": stable_fraction,
                    "positive_response_nll_shift_mean": float(positive_nll_shift.mean()),
                    "positive_negative_auroc": auroc(positive, negative),
                }
            )
        summaries.append(
            {
                "concept": concept,
                "n_per_class": len(ids[1]),
                "conditions": conditions,
            }
        )
    return summaries


def zero_summary(
    central_index: dict[tuple[str, str], dict[str, Any]],
    zero_rows: Sequence[dict[str, Any]],
    subset_ids: set[str],
    rng: np.random.Generator,
    replicates: int,
) -> list[dict[str, Any]]:
    zero_index = {row["example_id"]: row for row in zero_rows}
    summaries = []
    for concept in CONCEPTS:
        ids = sorted(
            example_id
            for example_id in subset_ids
            if central_index[(example_id, "normal")]["concept"] == concept
            and int(central_index[(example_id, "normal")]["label"]) == 1
        )
        normal = np.asarray([central_index[(example_id, "normal")]["probe_score"] for example_id in ids])
        triggered = np.asarray([central_index[(example_id, "correct_trigger")]["probe_score"] for example_id in ids])
        natural = np.asarray([central_index[(example_id, "selected_k16:rescue")]["probe_score"] for example_id in ids])
        zero = np.asarray([zero_index[example_id]["probe_score"] for example_id in ids])
        natural_fraction, natural_stable = bootstrap_ratio(
            natural - triggered, normal, triggered, rng, replicates
        )
        zero_fraction, zero_stable = bootstrap_ratio(
            zero - triggered, normal, triggered, rng, replicates
        )
        difference_fraction, difference_stable = bootstrap_ratio(
            zero - natural, normal, triggered, rng, replicates
        )
        summaries.append(
            {
                "concept": concept,
                "n_positive": len(ids),
                "natural_selected_rescue_fraction": natural_fraction,
                "zero_ablation_fraction": zero_fraction,
                "zero_minus_natural_fraction": difference_fraction,
                "bootstrap_positive_denominator_fraction": min(
                    natural_stable, zero_stable, difference_stable
                ),
                "zero_response_nll_shift_from_triggered_mean": float(
                    np.mean(
                        [
                            zero_index[example_id]["response_nll"]
                            - central_index[(example_id, "correct_trigger")]["response_nll"]
                            for example_id in ids
                        ]
                    )
                ),
                "zero_response_kl_mean": float(
                    np.mean([zero_index[example_id]["response_kl"] for example_id in ids])
                ),
            }
        )
    return summaries


def pooling_summary(
    central_index: dict[tuple[str, str], dict[str, Any]],
    rng: np.random.Generator,
    replicates: int,
) -> list[dict[str, Any]]:
    pool_fields = {
        "mean": "probe_score",
        "first": "probe_score_first",
        "last": "probe_score_last",
        "maximum": "probe_score_max",
    }
    summaries = []
    all_ids = {example_id for example_id, _condition in central_index}
    for concept in CONCEPTS:
        positive_ids = sorted(
            example_id
            for example_id in all_ids
            if central_index[(example_id, "normal")]["concept"] == concept
            and int(central_index[(example_id, "normal")]["label"]) == 1
        )
        pools = []
        for pool_id, field in pool_fields.items():
            normal = np.asarray([central_index[(example_id, "normal")][field] for example_id in positive_ids])
            triggered = np.asarray([central_index[(example_id, "correct_trigger")][field] for example_id in positive_ids])
            effects = {}
            for group in ("selected_k16", "random_k16"):
                for direction in ("rescue", "induction"):
                    patched = np.asarray(
                        [central_index[(example_id, f"{group}:{direction}")][field] for example_id in positive_ids]
                    )
                    numerator = patched - triggered if direction == "rescue" else normal - patched
                    effects[(group, direction)] = bootstrap_ratio(
                        numerator, normal, triggered, rng, replicates
                    )
            contrasts = {}
            for direction in ("rescue", "induction"):
                selected_values = np.asarray(
                    [central_index[(example_id, f"selected_k16:{direction}")][field] for example_id in positive_ids]
                )
                random_values = np.asarray(
                    [central_index[(example_id, f"random_k16:{direction}")][field] for example_id in positive_ids]
                )
                selected_numerator = selected_values - triggered if direction == "rescue" else normal - selected_values
                random_numerator = random_values - triggered if direction == "rescue" else normal - random_values
                contrasts[direction] = bootstrap_ratio(
                    selected_numerator - random_numerator,
                    normal,
                    triggered,
                    rng,
                    replicates,
                )
            supported = all(
                effects[("selected_k16", direction)][0]["ci_low"] > 0
                and contrasts[direction][0]["ci_low"] > 0
                for direction in ("rescue", "induction")
            )
            pools.append(
                {
                    "pooling": pool_id,
                    "selected_rescue": effects[("selected_k16", "rescue")][0],
                    "selected_induction": effects[("selected_k16", "induction")][0],
                    "random_rescue": effects[("random_k16", "rescue")][0],
                    "random_induction": effects[("random_k16", "induction")][0],
                    "selected_minus_random_rescue": contrasts["rescue"][0],
                    "selected_minus_random_induction": contrasts["induction"][0],
                    "minimum_bootstrap_positive_denominator_fraction": min(
                        value[1] for value in effects.values()
                    ),
                    "transfer_supported_descriptively": supported,
                }
            )
        summaries.append({"concept": concept, "pools": pools})
    return summaries


def generation_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["concept"], row["condition_id"], int(row["label"]))].append(row)
    summaries = []
    for concept in CONCEPTS:
        conditions = []
        for condition_id in ("normal", "correct_trigger", "concept_only", "monitoring_only"):
            positive = grouped[(concept, condition_id, 1)]
            negative = grouped[(concept, condition_id, 0)]
            available_positive_scores = [row["generated_probe_score"] for row in positive if row["generated_probe_score"] is not None]
            available_negative_scores = [row["generated_probe_score"] for row in negative if row["generated_probe_score"] is not None]
            combined = positive + negative
            conditions.append(
                {
                    "condition_id": condition_id,
                    "n": len(combined),
                    "nonempty_fraction": float(np.mean([row["nonempty"] for row in combined])),
                    "mean_generated_token_count": float(np.mean([row["generated_token_count"] for row in combined])),
                    "mean_repeated_fourgram_fraction": float(np.mean([row["repeated_fourgram_fraction"] for row in combined])),
                    "mean_reference_token_f1": float(np.mean([row["reference_token_f1"] for row in combined])),
                    "mean_reference_prefix_fraction": float(np.mean([row["reference_prefix_fraction"] for row in combined])),
                    "positive_probe_mean": float(np.mean(available_positive_scores)) if available_positive_scores else None,
                    "negative_probe_mean": float(np.mean(available_negative_scores)) if available_negative_scores else None,
                    "positive_negative_probe_auroc": auroc(available_positive_scores, available_negative_scores) if available_positive_scores and available_negative_scores else None,
                }
            )
        summaries.append({"concept": concept, "conditions": conditions})
    return summaries


def concept_processing_summary(
    central_index: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    all_ids = {example_id for example_id, _condition in central_index}
    conditions = (
        "normal",
        "correct_trigger",
        "selected_k16:rescue",
        "selected_k16:induction",
    )
    summaries = []
    for concept in CONCEPTS:
        rows = []
        concept_ids = [
            example_id
            for example_id in all_ids
            if central_index[(example_id, "normal")]["concept"] == concept
        ]
        for condition in conditions:
            positive = [
                central_index[(example_id, condition)]["probe_score"]
                for example_id in concept_ids
                if int(central_index[(example_id, "normal")]["label"]) == 1
            ]
            negative = [
                central_index[(example_id, condition)]["probe_score"]
                for example_id in concept_ids
                if int(central_index[(example_id, "normal")]["label"]) == 0
            ]
            rows.append(
                {
                    "condition_id": condition,
                    "positive_probe_mean": float(np.mean(positive)),
                    "negative_probe_mean": float(np.mean(negative)),
                    "positive_negative_auroc": auroc(positive, negative),
                }
            )
        summaries.append({"concept": concept, "conditions": rows})
    return summaries


def activation_norm_summary(central: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in central:
        if row["record_type"] == "intervention":
            grouped[(row["concept"], row["group_id"], row["direction"], int(row["label"]))].append(row["activation_rms_ratio"])
    return [
        {
            "concept": key[0],
            "group_id": key[1],
            "direction": key[2],
            "label": key[3],
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "point_estimate_outside_frozen_band": not 0.5 <= float(np.mean(values)) <= 2.0,
        }
        for key, values in sorted(grouped.items())
    ]


def write_csvs(summary: dict[str, Any]) -> None:
    prompt_rows = [
        {"concept": concept["concept"], **condition}
        for concept in summary["prompt_confounds"]
        for condition in concept["conditions"]
    ]
    pooling_rows = [
        {
            "concept": concept["concept"],
            "pooling": pool["pooling"],
            "selected_rescue": pool["selected_rescue"]["estimate"],
            "selected_induction": pool["selected_induction"]["estimate"],
            "random_rescue": pool["random_rescue"]["estimate"],
            "random_induction": pool["random_induction"]["estimate"],
            "supported": pool["transfer_supported_descriptively"],
        }
        for concept in summary["alternative_probe_pooling"]
        for pool in concept["pools"]
    ]
    generation_rows = [
        {"concept": concept["concept"], **condition}
        for concept in summary["generation_diagnostics"]
        for condition in concept["conditions"]
    ]
    for name, rows in (
        ("prompt-confound-metrics.csv", prompt_rows),
        ("alternative-pooling-metrics.csv", pooling_rows),
        ("generation-diagnostic-metrics.csv", generation_rows),
    ):
        with (RESULT_DIR / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def render_figure(summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    prompt_values = np.asarray(
        [
            [condition["positive_suppression_fraction"]["estimate"] for condition in concept["conditions"]]
            for concept in summary["prompt_confounds"]
        ]
    )
    image = axes[0, 0].imshow(prompt_values, cmap="RdBu_r", vmin=-0.25, vmax=1.25, aspect="auto")
    axes[0, 0].set_xticks(range(len(PROMPT_CONDITIONS)), [value.replace("_", "\n") for value in PROMPT_CONDITIONS], fontsize=8)
    axes[0, 0].set_yticks(range(2), CONCEPTS)
    axes[0, 0].set_title("A. Positive suppression by prompt condition")
    fig.colorbar(image, ax=axes[0, 0], label="fraction of correct-trigger gap")

    zero = summary["zero_ablation"]
    positions = np.arange(2)
    width = 0.34
    axes[0, 1].bar(
        positions - width / 2,
        [row["natural_selected_rescue_fraction"]["estimate"] for row in zero],
        width,
        label="Natural replacement",
    )
    axes[0, 1].bar(
        positions + width / 2,
        [row["zero_ablation_fraction"]["estimate"] for row in zero],
        width,
        label="Zero ablation",
    )
    axes[0, 1].set_xticks(positions, CONCEPTS)
    axes[0, 1].set_ylabel("positive causal fraction")
    axes[0, 1].set_title("B. Natural rescue versus zero ablation")
    axes[0, 1].legend()

    pools = ("mean", "first", "last", "maximum")
    pool_values = np.asarray(
        [
            [
                next(pool for pool in concept["pools"] if pool["pooling"] == pooling)[metric]["estimate"]
                for pooling in pools
            ]
            for concept in summary["alternative_probe_pooling"]
            for metric in ("selected_rescue", "selected_induction")
        ]
    )
    image = axes[1, 0].imshow(pool_values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 0].set_xticks(range(4), pools)
    axes[1, 0].set_yticks(range(4), ["deception rescue", "deception induction", "harmful rescue", "harmful induction"])
    axes[1, 0].set_title("C. Selected K=16 effect across probe pooling")
    fig.colorbar(image, ax=axes[1, 0], label="positive causal fraction")

    generation_values = np.asarray(
        [
            [condition["positive_negative_probe_auroc"] for condition in concept["conditions"]]
            for concept in summary["generation_diagnostics"]
        ]
    )
    image = axes[1, 1].imshow(generation_values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(range(4), ["normal", "correct\ntrigger", "concept\nonly", "monitoring\nonly"])
    axes[1, 1].set_yticks(range(2), CONCEPTS)
    axes[1, 1].set_title("D. Generated-continuation concept AUROC")
    fig.colorbar(image, ax=axes[1, 1], label="AUROC")

    fig.suptitle("Day 13 safety-transfer confound diagnostics", fontsize=17, fontweight="bold")
    metadata = {
        "Title": "Day 13 safety-transfer confound diagnostics",
        "Creator": "scripts/day13_analyze_confounds.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(FIGURE_PNG, dpi=300, facecolor="white")
    fig.savefig(FIGURE_PDF, metadata=metadata, facecolor="white")
    plt.close(fig)


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    central_summary = json.loads(CENTRAL_SUMMARY_PATH.read_text())
    central = load_jsonl(CENTRAL_PATH)
    prompt = load_jsonl(PROMPT_PATH)
    zero = load_jsonl(ZERO_PATH)
    generation = load_jsonl(GENERATION_PATH)
    if (len(central), len(prompt), len(zero), len(generation)) != (1944, 448, 64, 128):
        raise ValueError("Day 13 raw result count mismatch")
    central_index = index_central(central)
    confound_ids = deterministic_subset_ids(central, "day13-confounds", 16)
    rng = np.random.default_rng(42)
    replicates = 10_000
    prompt_results = prompt_summary(central_index, prompt, confound_ids, rng, replicates)
    zero_results = zero_summary(central_index, zero, confound_ids, rng, replicates)
    pooling_results = pooling_summary(central_index, rng, replicates)
    generation_results = generation_summary(generation)
    norm_results = activation_norm_summary(central)
    summary = {
        "schema_version": 1,
        "procedure": "day13-confounds-v1",
        "bootstrap": {"replicates": replicates, "seed": 42, "confidence_level": 0.95},
        "raw_counts": {
            "central": len(central),
            "prompt_confounds": len(prompt),
            "zero_ablation": len(zero),
            "generation": len(generation),
        },
        "raw_sha256": {
            "central": sha256_file(CENTRAL_PATH),
            "prompt_confounds": sha256_file(PROMPT_PATH),
            "zero_ablation": sha256_file(ZERO_PATH),
            "generation": sha256_file(GENERATION_PATH),
        },
        "execution_plan_sha256": sha256_file(PLAN_PATH),
        "component_set_sha256": freeze["component_set_sha256"],
        "central_result_unchanged": central_summary["raw_results_sha256"] == sha256_file(CENTRAL_PATH),
        "central_overall_safety_transfer_supported": central_summary["overall_safety_transfer_supported"],
        "prompt_confounds": prompt_results,
        "zero_ablation": zero_results,
        "alternative_probe_pooling": pooling_results,
        "activation_norms": norm_results,
        "generation_diagnostics": generation_results,
        "hidden_concept_processing": concept_processing_summary(central_index),
        "behavior_preservation": {
            "scope": "fixed-continuation NLL/KL are primary; generation fidelity is descriptive",
            "central_selected_positive_cells": [
                {
                    "concept": concept["concept"],
                    "cells": [
                        {
                            "direction": cell["direction"],
                            "response_nll_shift": cell["response_nll_shift"],
                            "response_kl": cell["response_kl"],
                        }
                        for cell in concept["cells"]
                        if cell["group_id"] == "selected_k16" and cell["label"] == 1
                    ],
                }
                for concept in central_summary["concepts"]
            ],
        },
        "component_set_modified_after_safety": False,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csvs(summary)
    render_figure(summary)
    print(
        "Analyzed Day 13 confounds; pooling support: "
        + json.dumps(
            {
                concept["concept"]: {
                    pool["pooling"]: pool["transfer_supported_descriptively"]
                    for pool in concept["pools"]
                }
                for concept in pooling_results
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
