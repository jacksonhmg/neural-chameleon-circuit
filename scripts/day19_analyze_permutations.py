#!/usr/bin/env python3
"""Analyze benign permutation distributions, validation, and composition."""

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
RESULT_DIR = ROOT / "results/day-19"
COMPOSITION_RAW_PATH = RESULT_DIR / "composition-results.jsonl.gz"
ENSEMBLE_METRICS_PATH = RESULT_DIR / "permutation-ensemble-metrics.csv"
SELECTION_PATH = RESULT_DIR / "benign-selected-mappings.json"
COMPOSITION_METRICS_PATH = RESULT_DIR / "composition-metrics.csv"
SUMMARY_PATH = RESULT_DIR / "permutation-composition-summary.json"
FIGURE_PATH = RESULT_DIR / "composition.png"
PDF_PATH = RESULT_DIR / "composition.pdf"
AUDIT_PATH = RESULT_DIR / "day19-audit.json"
REPLICATES = 10000
SEED = 15210


def ratio(numerator: np.ndarray, normal: np.ndarray, triggered: np.ndarray, indices: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return ({"estimate": float(numerator.mean() / denominator), "ci_low": float(low), "ci_high": float(high)}, boots)


def main() -> None:
    selection = json.loads(SELECTION_PATH.read_text())
    if selection.get("status") != "pass":
        raise ValueError("mapping selection did not pass")
    with gzip.open(COMPOSITION_RAW_PATH, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        nested[row["split_role"]][row["concept"]][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(SEED)
    cells = []
    boots_by_key = {}
    for split_role in ("development", "validation"):
        for concept in sorted(nested[split_role]):
            example_ids = sorted(nested[split_role][concept])
            indices = rng.integers(0, 16, size=(REPLICATES, 16))
            normal = np.asarray([nested[split_role][concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids])
            triggered = np.asarray([nested[split_role][concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids])
            for mapping in selection["selected_mappings"]:
                for size in (1, 2, 4, 8, 12):
                    for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
                        condition_id = f"delta:{base}:{mapping['mapping_id']}:k{size}"
                        patched = np.asarray([nested[split_role][concept][example_id][condition_id]["probe_score"] for example_id in example_ids])
                        numerator = normal - patched if direction == "induction" else patched - triggered
                        result, boots = ratio(numerator, normal, triggered, indices)
                        key = (split_role, concept, mapping["mapping_id"], size, direction)
                        boots_by_key[key] = boots
                        cells.append({"scope": concept, "split_role": split_role, "mapping_id": mapping["mapping_id"], "mapping_class": mapping["mapping_class"], "size": size, "direction": direction, "additive_pair_prediction": float("nan"), **result})

    day18_rows = list(csv.DictReader((ROOT / "results/day-18/geometry-transfer-cells.csv").open()))
    day18_lookup = {(row["scope"], row["direction"], row["transport"], row["source_id"], row["destination_id"]): float(row["estimate"]) for row in day18_rows if row["scope"] == "discovery_macro" and row["transport"] == "raw"}
    selected_heads = sorted(json.loads((ROOT / "results/day-15/frozen-site-shuffling-plan.json").read_text())["selected_heads"])
    mapping_lookup = {row["mapping_id"]: row for row in selection["selected_mappings"]}
    for row in cells:
        if row["split_role"] != "development":
            continue
        destinations = selected_heads[: int(row["size"])]
        mapping = mapping_lookup[row["mapping_id"]]["destination_to_source"]
        row["additive_pair_prediction"] = float(sum(day18_lookup[("discovery_macro", row["direction"], "raw", mapping[destination], destination)] for destination in destinations))

    macro_rows = []
    for split_role in ("development", "validation"):
        concepts = sorted(nested[split_role])
        for mapping in selection["selected_mappings"]:
            for size in (1, 2, 4, 8, 12):
                for direction in ("rescue", "induction"):
                    relevant = [row for row in cells if row["split_role"] == split_role and row["mapping_id"] == mapping["mapping_id"] and row["size"] == size and row["direction"] == direction]
                    boots = np.stack([boots_by_key[(split_role, concept, mapping["mapping_id"], size, direction)] for concept in concepts]).mean(axis=0)
                    point = float(np.mean([row["estimate"] for row in relevant]))
                    low, high = np.quantile(boots, [0.025, 0.975])
                    prediction = float(np.nanmean([row["additive_pair_prediction"] for row in relevant])) if split_role == "development" else float("nan")
                    macro_rows.append({"scope": f"{split_role}_macro", "split_role": split_role, "mapping_id": mapping["mapping_id"], "mapping_class": mapping["mapping_class"], "size": size, "direction": direction, "additive_pair_prediction": prediction, "estimate": point, "ci_low": float(low), "ci_high": float(high)})
    output_rows = cells + macro_rows
    with COMPOSITION_METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    ensemble_metrics = list(csv.DictReader(ENSEMBLE_METRICS_PATH.open()))
    distributions = []
    for split_role in ("development", "validation"):
        for mapping_class in ("within_layer", "cross_layer"):
            mapping_scores = []
            for mapping_id in sorted({row["mapping_id"] for row in ensemble_metrics if row["mapping_class"] == mapping_class}):
                values = [float(row["estimate"]) for row in ensemble_metrics if row["split_role"] == split_role and row["mapping_id"] == mapping_id]
                mapping_scores.append(float(np.mean(values)))
            distributions.append({"split_role": split_role, "mapping_class": mapping_class, "count": len(mapping_scores), "median": float(np.median(mapping_scores)), "q1": float(np.quantile(mapping_scores, 0.25)), "q3": float(np.quantile(mapping_scores, 0.75)), "minimum": float(min(mapping_scores)), "maximum": float(max(mapping_scores)), "fraction_above_zero": float(np.mean(np.asarray(mapping_scores) > 0))})

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = {row["mapping_id"]: color for row, color in zip(selection["selected_mappings"], ("#1b9e77", "#66a61e", "#7570b3", "#d95f02"), strict=True)}
    for axis, split_role in zip(axes, ("development", "validation"), strict=True):
        for mapping in selection["selected_mappings"]:
            for direction, linestyle in (("rescue", "-"), ("induction", "--")):
                values = [next(row["estimate"] for row in macro_rows if row["split_role"] == split_role and row["mapping_id"] == mapping["mapping_id"] and row["size"] == size and row["direction"] == direction) for size in (1, 2, 4, 8, 12)]
                axis.plot((1, 2, 4, 8, 12), values, marker="o", linestyle=linestyle, color=colors[mapping["mapping_id"]], label=f"{mapping['mapping_id']} {direction}")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(split_role)
        axis.set_xlabel("transported destinations")
        axis.set_ylabel("benign macro causal fraction")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=6, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Day 19 selected mapping composition")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    figure.savefig(PDF_PATH, bbox_inches="tight")
    plt.close(figure)

    summary = {"schema_version": 1, "procedure": "site-shuffling-v1-day19-analysis", "status": "complete", "selected_mapping_ids": [row["mapping_id"] for row in selection["selected_mappings"]], "distributions": distributions, "selection_used_safety": False}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    checks = {"composition_row_count": len(rows) == 176 * 42, "unique_composition_keys": len({(row["example_id"], row["condition_id"]) for row in rows}) == 176 * 42, "selected_mapping_count": len(selection["selected_mappings"]) == 4, "distribution_counts": all(row["count"] == 32 for row in distributions), "metrics_finite": all(np.isfinite(row[key]) for row in output_rows for key in ("estimate", "ci_low", "ci_high")), "figures_written": FIGURE_PATH.is_file() and PDF_PATH.is_file()}
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day19-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 19 audit failed")


if __name__ == "__main__":
    main()
