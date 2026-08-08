#!/usr/bin/env python3
"""Analyze frozen Day 16 absolute-versus-delta site-shuffling triage."""

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
RESULT_DIR = ROOT / "results/day-16"
RAW_PATH = RESULT_DIR / "artifact-triage-results.jsonl.gz"
SUMMARY_PATH = RESULT_DIR / "artifact-triage-summary.json"
METRICS_PATH = RESULT_DIR / "artifact-triage-metrics.csv"
FIGURE_PATH = RESULT_DIR / "artifact-triage-overview.png"
PDF_PATH = RESULT_DIR / "artifact-triage-overview.pdf"
AUDIT_PATH = RESULT_DIR / "artifact-triage-audit.json"
REPLICATES = 10000
SEED = 15210


def read_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def estimate(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    boots = values[indices].mean(axis=1)
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def ratio_estimate(
    numerator: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    if denominator <= 0 or np.any(denominator_boot <= 0):
        raise ValueError("unstable suppression denominator")
    point = float(numerator.mean() / denominator)
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"estimate": point, "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    rows = read_rows()
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        nested[row["concept"]][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(SEED)
    metrics = []
    classifications = []
    dose_points: dict[tuple[str, str, str, float], float] = {}
    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept])
        if len(example_ids) != 16:
            raise ValueError(f"expected 16 {concept} examples")
        indices = rng.integers(0, len(example_ids), size=(REPLICATES, len(example_ids)))
        normal = np.asarray([nested[concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids])
        triggered = np.asarray([nested[concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids])
        metrics.append({"concept": concept, "mapping_id": "baseline", "direction": "suppression", "estimand": "gap", "alpha": "", **estimate(normal - triggered, indices)})
        mapping_ids = sorted({row["mapping_id"] for row in rows if row.get("concept") == concept and row.get("mapping_id")})
        for mapping_id in mapping_ids:
            scores: dict[str, np.ndarray] = {}
            for condition_id in nested[concept][example_ids[0]]:
                if mapping_id not in condition_id:
                    continue
                scores[condition_id] = np.asarray([nested[concept][example_id][condition_id]["probe_score"] for example_id in example_ids])
            derived: list[tuple[str, str, np.ndarray]] = []
            n_n = scores[f"absolute:normal:normal:{mapping_id}"]
            n_t = scores[f"absolute:normal:correct_trigger:{mapping_id}"]
            t_n = scores[f"absolute:correct_trigger:normal:{mapping_id}"]
            t_t = scores[f"absolute:correct_trigger:correct_trigger:{mapping_id}"]
            derived.extend([
                ("induction", "absolute", normal - n_t),
                ("induction", "conditional", n_n - n_t),
                ("induction", "same_condition_mismatch", normal - n_n),
                ("rescue", "absolute", t_n - triggered),
                ("rescue", "conditional", t_n - t_t),
                ("rescue", "same_condition_mismatch", t_t - triggered),
            ])
            cell_metrics: dict[tuple[str, str], dict[str, float]] = {}
            for direction, estimand_name, numerator in derived:
                result = ratio_estimate(numerator, normal, triggered, indices)
                cell_metrics[(direction, estimand_name)] = result
                metrics.append({"concept": concept, "mapping_id": mapping_id, "direction": direction, "estimand": estimand_name, "alpha": "", **result})
            for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
                for alpha in (-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
                    condition_id = f"delta:{base}:alpha_{alpha:g}:{mapping_id}"
                    patched = scores[condition_id]
                    numerator = normal - patched if direction == "induction" else patched - triggered
                    result = ratio_estimate(numerator, normal, triggered, indices)
                    dose_points[(concept, mapping_id, direction, alpha)] = result["estimate"]
                    if alpha == 1.0:
                        cell_metrics[(direction, "delta")] = result
                    metrics.append({"concept": concept, "mapping_id": mapping_id, "direction": direction, "estimand": "delta", "alpha": alpha, **result})
                rms_id = f"delta_rms:{base}:alpha_1:{mapping_id}"
                rms_patched = scores[rms_id]
                rms_numerator = normal - rms_patched if direction == "induction" else rms_patched - triggered
                result = ratio_estimate(rms_numerator, normal, triggered, indices)
                cell_metrics[(direction, "delta_rms")] = result
                metrics.append({"concept": concept, "mapping_id": mapping_id, "direction": direction, "estimand": "delta_rms", "alpha": 1.0, **result})
                absolute = cell_metrics[(direction, "absolute")]
                conditional = cell_metrics[(direction, "conditional")]
                delta = cell_metrics[(direction, "delta")]
                mismatch = cell_metrics[(direction, "same_condition_mismatch")]
                if conditional["ci_low"] > 0 and delta["ci_low"] > 0:
                    classification = "trigger_specific_destination_relative_transfer"
                elif conditional["ci_low"] > 0:
                    classification = "trigger_specific_but_mismatch_sensitive"
                elif absolute["ci_low"] > 0 and mismatch["estimate"] > conditional["estimate"]:
                    classification = "generic_mismatch_candidate"
                else:
                    classification = "no_strong_transfer"
                classifications.append({"concept": concept, "mapping_id": mapping_id, "direction": direction, "classification": classification})

    fieldnames = ["concept", "mapping_id", "direction", "estimand", "alpha", "estimate", "ci_low", "ci_high"]
    with METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    colors = {"day14_seed_211_heads": "#1b9e77", "day14_seed_223_heads": "#d95f02"}
    alphas = [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    for row_index, concept in enumerate(("deception", "harmful")):
        for column_index, direction in enumerate(("rescue", "induction")):
            axis = axes[row_index, column_index]
            for mapping_id, color in colors.items():
                values = [dose_points[(concept, mapping_id, direction, alpha)] for alpha in alphas]
                axis.plot(alphas, values, marker="o", label=mapping_id.replace("day14_", ""), color=color)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            axis.set_title(f"{concept}: {direction}")
            axis.set_ylabel("normalized causal fraction")
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("transported source-delta scale α")
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Day 16 destination-relative site-shuffle dose response")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    summary = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day16-analysis",
        "status": "complete",
        "n_rows": len(rows),
        "n_examples": len({row["example_id"] for row in rows}),
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "classifications": classifications,
        "interpretation_rule": "absolute replacement is not portable-code evidence unless route-matched conditional and destination-relative delta tests also remain positive",
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    expected_rows = 32 * 42
    checks = {
        "raw_row_count": len(rows) == expected_rows,
        "unique_keys": len({(row["example_id"], row["condition_id"]) for row in rows}) == expected_rows,
        "two_concepts": {row["concept"] for row in rows} == {"deception", "harmful"},
        "metrics_written": METRICS_PATH.is_file(),
        "figures_written": FIGURE_PATH.is_file() and PDF_PATH.is_file(),
        "all_finite": all(np.isfinite(row[key]) for row in metrics for key in ("estimate", "ci_low", "ci_high")),
    }
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day16-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 16 audit failed")


if __name__ == "__main__":
    main()
