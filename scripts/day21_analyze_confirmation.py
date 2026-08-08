#!/usr/bin/env python3
"""Apply the prospectively authorized Day 21 causal gate and audit."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-21"
RAW_PATH = RESULT_DIR / "confirmation-results.jsonl.gz"
AUTHORIZATION_PATH = RESULT_DIR / "confirmation-authorization.json"
CELL_PATH = RESULT_DIR / "confirmation-cells.csv"
GATE_PATH = RESULT_DIR / "confirmation-gate.json"
NEGATIVE_REFERENCE_PATH = RESULT_DIR / "negative-specificity-reference.json"
FIGURE_PATH = RESULT_DIR / "confirmation.png"
PDF_PATH = RESULT_DIR / "confirmation.pdf"
AUDIT_PATH = RESULT_DIR / "day21-audit.json"
DAY20_METRICS_PATH = ROOT / "results/day-20/specificity-metrics.csv"
REPLICATES = 10000
SEED = 15210


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def estimate_ratio(
    numerator: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    if denominator <= 0 or np.any(denominator_boot <= 0):
        raise ValueError("confirmation suppression denominator is not positive")
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return {
        "estimate": float(numerator.mean() / denominator),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def numerator_for(
    example: Mapping[str, Mapping[str, Any]],
    mapping_id: str,
    source_role: str,
    direction: str,
    estimand: str,
) -> float:
    normal = float(example["baseline:normal"]["probe_score"])
    triggered = float(example["baseline:correct_trigger"]["probe_score"])
    base = "normal" if direction == "induction" else "correct_trigger"
    absolute_normal = float(
        example[
            f"absolute:{base}:{source_role}:{mapping_id}:normal"
        ]["probe_score"]
    )
    absolute_triggered = float(
        example[
            f"absolute:{base}:{source_role}:{mapping_id}:correct_trigger"
        ]["probe_score"]
    )
    delta = float(
        example[f"delta:{base}:{source_role}:{mapping_id}"]["probe_score"]
    )
    if estimand == "conditional":
        return absolute_normal - absolute_triggered
    if estimand == "delta":
        return normal - delta if direction == "induction" else delta - triggered
    if estimand == "absolute":
        return (
            normal - absolute_triggered
            if direction == "induction"
            else absolute_normal - triggered
        )
    if estimand == "same_condition_mismatch":
        return (
            normal - absolute_normal
            if direction == "induction"
            else absolute_triggered - triggered
        )
    raise ValueError(f"unknown estimand {estimand}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def negative_specificity_reference(
    authorization: Mapping[str, Any]
) -> dict[str, Any]:
    expected_hash = authorization["inputs"]["negative_specificity_reference"][
        "sha256"
    ]
    actual_hash = sha256_file(DAY20_METRICS_PATH)
    if actual_hash != expected_hash:
        raise ValueError("Day 20 negative specificity reference hash changed")
    selected_ids = {row["mapping_id"] for row in authorization["mappings"]}
    rows = list(csv.DictReader(DAY20_METRICS_PATH.open()))
    retained = [
        row
        for row in rows
        if row["mapping_id"] in selected_ids
        and row["label"] == "0"
        and row["probe_role"] == "target"
        and row["source_condition"] == "correct_trigger"
        and row["token_control"] == "aligned"
    ]
    if len(retained) != 8:
        raise ValueError("expected eight fixed negative specificity cells")
    summary = {
        concept: {
            "mapping_count": 4,
            "mean_normalized_effect": float(
                np.mean(
                    [
                        float(row["normalized_effect"])
                        for row in retained
                        if row["concept"] == concept
                    ]
                )
            ),
            "minimum": float(
                min(
                    float(row["normalized_effect"])
                    for row in retained
                    if row["concept"] == concept
                )
            ),
            "maximum": float(
                max(
                    float(row["normalized_effect"])
                    for row in retained
                    if row["concept"] == concept
                )
            ),
        }
        for concept in ("deception", "harmful")
    }
    result = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day21-negative-specificity-reference",
        "status": "complete",
        "source_path": DAY20_METRICS_PATH.relative_to(ROOT).as_posix(),
        "source_sha256": actual_hash,
        "rule": "pre-confirmation Day 20 fixed negative subset and fixed benign-selected mappings; no Day 21 reranking",
        "summary": summary,
        "cells": retained,
    }
    NEGATIVE_REFERENCE_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> None:
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    raw_rows = read_rows()
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in raw_rows:
        nested[row["concept"]][row["example_id"]][row["condition_id"]] = row
    mapping_ids = [row["mapping_id"] for row in authorization["mappings"]]
    rng = np.random.default_rng(SEED)
    cell_rows: list[dict[str, Any]] = []
    population_numerators: dict[tuple[str, str, str, str], np.ndarray] = {}

    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept])
        count = len(example_ids)
        indices = rng.integers(0, count, size=(REPLICATES, count))
        normal = np.asarray(
            [nested[concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids]
        )
        triggered = np.asarray(
            [nested[concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids]
        )
        for direction in ("rescue", "induction"):
            for estimand in (
                "absolute",
                "same_condition_mismatch",
                "conditional",
                "delta",
            ):
                for source_role in ("selected", "null"):
                    mapping_arrays = []
                    for mapping_id in mapping_ids:
                        numerator = np.asarray(
                            [
                                numerator_for(
                                    nested[concept][example_id],
                                    mapping_id,
                                    source_role,
                                    direction,
                                    estimand,
                                )
                                for example_id in example_ids
                            ]
                        )
                        mapping_arrays.append(numerator)
                        cell_rows.append(
                            {
                                "scope": "mapping",
                                "concept": concept,
                                "direction": direction,
                                "estimand": estimand,
                                "source_role": source_role,
                                "mapping_id": mapping_id,
                                **estimate_ratio(numerator, normal, triggered, indices),
                            }
                        )
                    population = np.stack(mapping_arrays).mean(axis=0)
                    population_numerators[
                        (concept, direction, estimand, source_role)
                    ] = population
                    cell_rows.append(
                        {
                            "scope": "population",
                            "concept": concept,
                            "direction": direction,
                            "estimand": estimand,
                            "source_role": source_role,
                            "mapping_id": "mean_of_four_benign_selected_mappings",
                            **estimate_ratio(population, normal, triggered, indices),
                        }
                    )
                contrast = (
                    population_numerators[(concept, direction, estimand, "selected")]
                    - population_numerators[(concept, direction, estimand, "null")]
                )
                cell_rows.append(
                    {
                        "scope": "contrast",
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "source_role": "selected_minus_null",
                        "mapping_id": "mean_of_four_route_matched_contrasts",
                        **estimate_ratio(contrast, normal, triggered, indices),
                    }
                )
    write_csv(CELL_PATH, cell_rows)

    lookup = {
        (
            row["scope"],
            row["concept"],
            row["direction"],
            row["estimand"],
            row["source_role"],
        ): row
        for row in cell_rows
        if row["scope"] != "mapping"
    }
    primary_keys = [
        (concept, direction, estimand)
        for concept in ("deception", "harmful")
        for direction in ("rescue", "induction")
        for estimand in ("conditional", "delta")
    ]
    portable_support = all(
        lookup[("population", *key, "selected")]["ci_low"] > 0
        and lookup[("contrast", *key, "selected_minus_null")]["ci_low"] > 0
        for key in primary_keys
    )
    directionally_positive = all(
        lookup[("population", *key, "selected")]["estimate"] > 0
        for key in primary_keys
    )
    absolute_positive = all(
        lookup[("population", concept, direction, "absolute", "selected")][
            "estimate"
        ]
        > 0
        for concept in ("deception", "harmful")
        for direction in ("rescue", "induction")
    )
    no_corrected_lower_positive = all(
        lookup[("population", *key, "selected")]["ci_low"] <= 0
        for key in primary_keys
    )
    if portable_support:
        status = "portable_support"
    elif directionally_positive:
        status = "qualified_support"
    elif absolute_positive and no_corrected_lower_positive:
        status = "generic_disruption"
    else:
        status = "rejected"

    negative_reference = negative_specificity_reference(authorization)
    gate_cells = []
    for key in primary_keys:
        selected = lookup[("population", *key, "selected")]
        null = lookup[("population", *key, "null")]
        contrast = lookup[("contrast", *key, "selected_minus_null")]
        gate_cells.append(
            {
                "concept": key[0],
                "direction": key[1],
                "estimand": key[2],
                "selected": {
                    field: selected[field]
                    for field in ("estimate", "ci_low", "ci_high")
                },
                "null": {
                    field: null[field]
                    for field in ("estimate", "ci_low", "ci_high")
                },
                "selected_minus_null": {
                    field: contrast[field]
                    for field in ("estimate", "ci_low", "ci_high")
                },
                "selected_lower_positive": selected["ci_low"] > 0,
                "contrast_lower_positive": contrast["ci_low"] > 0,
            }
        )
    gate = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day21-gate",
        "status": status,
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "raw_results_sha256": sha256_file(RAW_PATH),
        "bootstrap_replicates": REPLICATES,
        "bootstrap_seed": SEED,
        "criteria": {
            "portable_support": portable_support,
            "all_primary_selected_points_positive": directionally_positive,
            "all_absolute_selected_points_positive": absolute_positive,
            "no_primary_selected_lower_bound_positive": no_corrected_lower_positive,
        },
        "gate_cells": gate_cells,
        "negative_specificity": negative_reference["summary"],
        "diagnostic_boundary": authorization["mechanical_gate"][
            "diagnostic_boundary"
        ],
    }
    GATE_PATH.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row")
    estimands = ("conditional", "delta")
    colors = {"selected": "#1b9e77", "null": "#999999"}
    for row_index, concept in enumerate(("deception", "harmful")):
        for column_index, direction in enumerate(("rescue", "induction")):
            axis = axes[row_index, column_index]
            for offset, source_role in ((-0.08, "selected"), (0.08, "null")):
                values = [
                    lookup[
                        (
                            "population",
                            concept,
                            direction,
                            estimand,
                            source_role,
                        )
                    ]
                    for estimand in estimands
                ]
                points = np.asarray([value["estimate"] for value in values])
                errors = np.asarray(
                    [
                        [value["estimate"] - value["ci_low"] for value in values],
                        [value["ci_high"] - value["estimate"] for value in values],
                    ]
                )
                axis.errorbar(
                    np.arange(2) + offset,
                    points,
                    yerr=errors,
                    marker="o",
                    linestyle="none",
                    capsize=3,
                    color=colors[source_role],
                    label=source_role,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xticks(range(2), estimands)
            axis.set_title(f"{concept} {direction}")
            axis.set_ylabel("normalized causal fraction")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 1].legend(frameon=False)
    figure.suptitle(f"Day 21 prospective confirmation: {status.replace('_', ' ')}")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    authorized_ids = {
        example_id
        for concept in ("deception", "harmful")
        for example_id in authorization["confirmation_examples"][concept][
            "example_ids"
        ]
    }
    checks = {
        "raw_row_count": len(raw_rows) == 6500,
        "unique_raw_keys": len(
            {(row["example_id"], row["condition_id"]) for row in raw_rows}
        )
        == 6500,
        "authorized_examples_only": {row["example_id"] for row in raw_rows}
        == authorized_ids,
        "conditions_per_example": {
            len(
                {
                    row["condition_id"]
                    for row in raw_rows
                    if row["example_id"] == example_id
                }
            )
            for example_id in authorized_ids
        }
        == {50},
        "authorization_hash_matches_rows": {
            row["authorization_sha256"]
            for row in raw_rows
            if row["record_type"] == "intervention"
        }
        == {sha256_file(AUTHORIZATION_PATH)},
        "all_cells_finite": all(
            np.isfinite(row[field])
            for row in cell_rows
            for field in ("estimate", "ci_low", "ci_high")
        ),
        "gate_status_valid": status
        in {
            "portable_support",
            "qualified_support",
            "generic_disruption",
            "rejected",
        },
        "negative_reference_hash": negative_reference["source_sha256"]
        == authorization["inputs"]["negative_specificity_reference"]["sha256"],
        "figures_written": FIGURE_PATH.is_file() and PDF_PATH.is_file(),
    }
    audit = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day21-audit",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "n_raw_rows": len(raw_rows),
        "n_cell_rows": len(cell_rows),
        "confirmation_status": status,
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 21 audit failed")


if __name__ == "__main__":
    main()
