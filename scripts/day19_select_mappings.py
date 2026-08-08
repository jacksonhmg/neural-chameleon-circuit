#!/usr/bin/env python3
"""Apply the frozen benign-only mapping-selection rule."""

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
RAW_PATH = RESULT_DIR / "permutation-ensemble-results.jsonl.gz"
METRICS_PATH = RESULT_DIR / "permutation-ensemble-metrics.csv"
SELECTION_PATH = RESULT_DIR / "benign-selected-mappings.json"
FIGURE_PATH = RESULT_DIR / "permutation-ensemble.png"
PDF_PATH = RESULT_DIR / "permutation-ensemble.pdf"
REPLICATES = 10000
SEED = 15210


def ratio(
    numerator: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(numerator.mean() / denominator), "ci_low": float(low), "ci_high": float(high)}


def main() -> None:
    with gzip.open(RAW_PATH, "rt") as handle:
        rows = [json.loads(line) for line in handle]
    mappings = json.loads((ROOT / "results/day-15/frozen-mapping-ensemble.json").read_text())["ensemble"]
    mapping_by_id = {row["mapping_id"]: row for row in mappings}
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        nested[row["split_role"]][row["concept"]][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(SEED)
    metrics = []
    for split_role in ("development", "validation"):
        for concept in sorted(nested[split_role]):
            example_ids = sorted(nested[split_role][concept])
            indices = rng.integers(0, 16, size=(REPLICATES, 16))
            normal = np.asarray([nested[split_role][concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids])
            triggered = np.asarray([nested[split_role][concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids])
            for mapping in mappings:
                mapping_id = mapping["mapping_id"]
                for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
                    patched = np.asarray([nested[split_role][concept][example_id][f"delta:{base}:{mapping_id}"]["probe_score"] for example_id in example_ids])
                    numerator = normal - patched if direction == "induction" else patched - triggered
                    metrics.append({
                        "split_role": split_role,
                        "concept": concept,
                        "mapping_id": mapping_id,
                        "mapping_class": mapping["mapping_class"],
                        "direction": direction,
                        **ratio(numerator, normal, triggered, indices),
                    })
    with METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    development = defaultdict(list)
    validation = defaultdict(list)
    for row in metrics:
        target = development if row["split_role"] == "development" else validation
        target[row["mapping_id"]].append(row)
    ranking = []
    for mapping_id, cells in development.items():
        mapping_class = mapping_by_id[mapping_id]["mapping_class"]
        eligible = len(cells) == 8 and all(row["estimate"] > 0 for row in cells)
        ranking.append({
            "mapping_id": mapping_id,
            "mapping_class": mapping_class,
            "development_score": float(np.mean([row["estimate"] for row in cells])),
            "eligible": eligible,
            "development_minimum": float(min(row["estimate"] for row in cells)),
            "validation_score": float(np.mean([row["estimate"] for row in validation[mapping_id]])),
            "validation_minimum": float(min(row["estimate"] for row in validation[mapping_id])),
        })
    selected = []
    for mapping_class in ("within_layer", "cross_layer"):
        eligible = sorted(
            (row for row in ranking if row["mapping_class"] == mapping_class and row["eligible"]),
            key=lambda row: (-row["development_score"], row["mapping_id"]),
        )
        selected.extend(eligible[:2])
    status = "pass" if len(selected) == 4 else "fail"
    selection = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-benign-selection",
        "status": status,
        "selection_rule": "top two eligible mappings per class by four-concept mean rescue/induction score; eligibility requires all eight discovery cells positive",
        "safety_outcomes_accessed_for_selection": False,
        "ranking": sorted(ranking, key=lambda row: (row["mapping_class"], -row["development_score"], row["mapping_id"])),
        "selected_mappings": [
            {
                **row,
                "destination_to_source": mapping_by_id[row["mapping_id"]]["destination_to_source"],
                "mapping_sha256": mapping_by_id[row["mapping_id"]]["mapping_sha256"],
            }
            for row in selected
        ],
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("fewer than two eligible mappings in one frozen class")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    selected_ids = {row["mapping_id"] for row in selected}
    for axis, mapping_class in zip(axes, ("within_layer", "cross_layer"), strict=True):
        values = sorted((row for row in ranking if row["mapping_class"] == mapping_class), key=lambda row: row["development_score"])
        colors = ["#d95f02" if row["mapping_id"] in selected_ids else "#777777" if row["eligible"] else "#cccccc" for row in values]
        axis.barh(range(len(values)), [row["development_score"] for row in values], color=colors)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(mapping_class.replace("_", " "))
        axis.set_xlabel("discovery mean rescue/induction")
        axis.set_yticks([])
        axis.grid(axis="x", alpha=0.2)
    figure.suptitle("Day 19 complete benign permutation distributions")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)


if __name__ == "__main__":
    main()
