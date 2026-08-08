#!/usr/bin/env python3
"""Analyze Day 20 specificity, RMS, and full-model behavior diagnostics."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-20"
SPECIFICITY_PATH = RESULT_DIR / "specificity-results.jsonl.gz"
BEHAVIOR_PATH = RESULT_DIR / "behavior-results.jsonl.gz"
METRICS_PATH = RESULT_DIR / "specificity-metrics.csv"
BEHAVIOR_METRICS_PATH = RESULT_DIR / "behavior-metrics.csv"
SUMMARY_PATH = RESULT_DIR / "diagnostic-summary.json"
FIGURE_PATH = RESULT_DIR / "diagnostics.png"
PDF_PATH = RESULT_DIR / "diagnostics.pdf"
AUDIT_PATH = RESULT_DIR / "day20-audit.json"
REPLICATES = 10000
SEED = 15210


def load_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def interval(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    boots = values[indices].mean(axis=1)
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    specificity = load_gzip(SPECIFICITY_PATH)
    behavior = load_gzip(BEHAVIOR_PATH)
    nested: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in specificity:
        nested[row["concept"]][int(row["label"])][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(SEED)
    metrics = []
    for concept in ("deception", "harmful"):
        positive_ids = sorted(nested[concept][1])
        target_normal = np.asarray([nested[concept][1][example_id][f"baseline:normal:{concept}"]["probe_score"] for example_id in positive_ids])
        target_triggered = np.asarray([nested[concept][1][example_id][f"baseline:correct_trigger:{concept}"]["probe_score"] for example_id in positive_ids])
        denominator = float((target_normal - target_triggered).mean())
        for label in (1, 0):
            example_ids = sorted(nested[concept][label])
            indices = rng.integers(0, 16, size=(REPLICATES, 16))
            for row in (value for value in specificity if value["concept"] == concept and int(value["label"]) == label and value["record_type"] == "intervention"):
                condition_id = row["condition_id"]
                if row["example_id"] != example_ids[0]:
                    continue
                probe_concept = row["probe_concept"]
                normal = np.asarray([nested[concept][label][example_id][f"baseline:normal:{probe_concept}"]["probe_score"] for example_id in example_ids])
                patched_rows = [nested[concept][label][example_id][condition_id] for example_id in example_ids]
                patched = np.asarray([value["probe_score"] for value in patched_rows])
                effect = (normal - patched) / denominator
                result = interval(effect, indices)
                metrics.append({"concept": concept, "label": label, "probe_role": row["probe_role"], "source_condition": row["source_condition"], "token_control": row["token_control"], "mapping_id": row["mapping_id"], "normalized_effect": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"], "source_destination_rms_ratio": float(np.mean([value["source_destination_rms_ratio"] for value in patched_rows])), "delta_destination_natural_rms_ratio": float(np.mean([value["delta_destination_natural_rms_ratio"] for value in patched_rows])), "patched_destination_rms_ratio": float(np.mean([value["patched_destination_rms_ratio"] for value in patched_rows]))})
    with METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    behavior_nested: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    for row in behavior:
        behavior_nested[row["concept"]][row["example_id"]][row["condition_id"]] = row
    behavior_metrics = []
    for concept in ("deception", "harmful"):
        example_ids = sorted(behavior_nested[concept])
        indices = rng.integers(0, 16, size=(REPLICATES, 16))
        baseline_nll = np.asarray([behavior_nested[concept][example_id]["baseline:normal"]["response_nll"] for example_id in example_ids])
        for condition_id in sorted(behavior_nested[concept][example_ids[0]]):
            if condition_id == "baseline:normal":
                continue
            values = [behavior_nested[concept][example_id][condition_id] for example_id in example_ids]
            nll_shift = np.asarray([value["response_nll"] for value in values]) - baseline_nll
            kl = np.asarray([value["kl_from_normal"] for value in values])
            nll_result = interval(nll_shift, indices)
            kl_result = interval(kl, indices)
            behavior_metrics.append({"concept": concept, "condition_id": condition_id, "nll_shift": nll_result["estimate"], "nll_ci_low": nll_result["ci_low"], "nll_ci_high": nll_result["ci_high"], "kl": kl_result["estimate"], "kl_ci_low": kl_result["ci_low"], "kl_ci_high": kl_result["ci_high"]})
    with BEHAVIOR_METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(behavior_metrics[0]))
        writer.writeheader()
        writer.writerows(behavior_metrics)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    source_order = ("correct_trigger", "irrelevant_concept", "monitoring_only", "malformed_trigger")
    for concept, color in (("deception", "#1b9e77"), ("harmful", "#d95f02")):
        points = [np.mean([row["normalized_effect"] for row in metrics if row["concept"] == concept and row["label"] == 1 and row["probe_role"] == "target" and row["token_control"] == "aligned" and row["source_condition"] == source]) for source in source_order]
        axes[0].plot(range(4), points, marker="o", label=concept, color=color)
    axes[0].set_xticks(range(4), ("correct", "irrelevant", "monitor only", "malformed"), rotation=20)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("source-condition specificity")
    axes[0].set_ylabel("mean target-probe fraction")
    axes[0].legend(frameon=False)
    categories = ("target positive", "target negative", "other probe positive")
    for concept, color in (("deception", "#1b9e77"), ("harmful", "#d95f02")):
        subsets = [
            [row["normalized_effect"] for row in metrics if row["concept"] == concept and row["label"] == 1 and row["probe_role"] == "target" and row["source_condition"] == "correct_trigger" and row["token_control"] == "aligned"],
            [row["normalized_effect"] for row in metrics if row["concept"] == concept and row["label"] == 0 and row["probe_role"] == "target" and row["source_condition"] == "correct_trigger" and row["token_control"] == "aligned"],
            [row["normalized_effect"] for row in metrics if row["concept"] == concept and row["label"] == 1 and row["probe_role"] == "other_safety" and row["source_condition"] == "correct_trigger" and row["token_control"] == "aligned"],
        ]
        axes[1].plot(range(3), [np.mean(values) for values in subsets], marker="o", label=concept, color=color)
    axes[1].set_xticks(range(3), categories, rotation=20)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("class and cross-probe specificity")
    conditions = ("baseline:correct_trigger", "selected_k12_identity", "day14_seed_211_raw", "day14_seed_223_raw", "day14_seed_211_delta", "day14_seed_223_delta")
    width = 0.35
    for offset, concept in ((-width / 2, "deception"), (width / 2, "harmful")):
        points = [next(row["kl"] for row in behavior_metrics if row["concept"] == concept and row["condition_id"] == condition) for condition in conditions]
        axes[2].bar(np.arange(len(conditions)) + offset, points, width=width, label=concept)
    axes[2].set_xticks(range(len(conditions)), ("trigger", "K12", "211 raw", "223 raw", "211 delta", "223 delta"), rotation=30)
    axes[2].set_title("full-model KL from normal")
    axes[2].set_ylabel("mean token KL")
    axes[2].legend(frameon=False)
    figure.suptitle("Day 20 site-shuffling diagnostics")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    summary = {"schema_version": 1, "procedure": "site-shuffling-v1-day20-analysis", "status": "complete", "specificity_metric_count": len(metrics), "behavior_metric_count": len(behavior_metrics)}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checks = {"specificity_rows": len(specificity) == 64 * 100, "behavior_rows": len(behavior) == 32 * 7, "specificity_unique": len({(row["example_id"], row["condition_id"]) for row in specificity}) == len(specificity), "behavior_unique": len({(row["example_id"], row["condition_id"]) for row in behavior}) == len(behavior), "all_finite": all(np.isfinite(value) for row in metrics for value in (row["normalized_effect"], row["ci_low"], row["ci_high"])), "figures_written": FIGURE_PATH.is_file() and PDF_PATH.is_file()}
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day20-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 20 audit failed")


if __name__ == "__main__":
    main()
