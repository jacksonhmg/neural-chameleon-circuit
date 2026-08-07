#!/usr/bin/env python3
"""Analyze the frozen Day 14 causal null and nearby-mechanism grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import load_experimental_split  # noqa: E402

from day14_run_causal_falsification import (  # noqa: E402
    EXPECTED_EXAMPLES,
    EXPECTED_GROUPS,
    EXPECTED_ROWS,
    PLAN_PATH,
    group_specifications,
)


RESULT_DIR = ROOT / "results/day-14"
RAW_PATH = RESULT_DIR / "causal-falsification-results.jsonl.gz"
SUMMARY_PATH = RESULT_DIR / "causal-falsification-summary.json"
METRICS_PATH = RESULT_DIR / "causal-falsification-metrics.csv"
AUDIT_PATH = RESULT_DIR / "causal-falsification-audit.json"
FIGURE_PATH = RESULT_DIR / "causal-falsification.png"
PDF_PATH = RESULT_DIR / "causal-falsification.pdf"
PREFLIGHT_PATH = RESULT_DIR / "causal-falsification-preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "estimate": float(point),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def expected_subset_ids() -> set[str]:
    records = load_experimental_split("safety-test")
    selected: set[str] = set()
    for concept in ("deception", "harmful"):
        eligible = [
            row for row in records
            if row["concept"] == concept and int(row["label"]) == 1
        ]
        ordered = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(
                    f"day14-causal:{row['example_id']}".encode()
                ).hexdigest(),
                row["example_id"],
            ),
        )
        selected.update(row["example_id"] for row in ordered[:16])
    return selected


def validate_rows(
    rows: Sequence[dict[str, Any]],
    plan: Mapping[str, Any],
    specifications: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    spec_by_id = {row["group_id"]: row for row in specifications}
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} causal rows, found {len(rows)}")
    if any(row.get("safety_split_accessed") is not True for row in rows):
        raise ValueError("causal records must explicitly record safety access")
    if any(int(row["label"]) != 1 for row in rows):
        raise ValueError("causal grid must contain positive examples only")
    if any(
        row.get("falsification_plan_sha256") != sha256_file(PLAN_PATH)
        for row in rows
    ):
        raise ValueError("causal row plan hash mismatch")
    examples = sorted({row["example_id"] for row in rows})
    if len(examples) != EXPECTED_EXAMPLES or set(examples) != expected_subset_ids():
        raise ValueError("causal subset differs from the frozen hash selection")
    counts = defaultdict(int)
    keys = set()
    for row in rows:
        condition = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
        key = (row["example_id"], condition)
        if key in keys:
            raise ValueError(f"duplicate causal key: {key}")
        keys.add(key)
        counts[row["example_id"]] += 1
        if row["record_type"] == "intervention":
            if row["group_id"] not in spec_by_id:
                raise ValueError(f"unexpected causal group: {row['group_id']}")
            specification = spec_by_id[row["group_id"]]
            if row["candidate_ids"] != specification["candidate_ids"]:
                raise ValueError(f"membership mismatch for {row['group_id']}")
            if row["source_mapping"] != specification["source_mapping"]:
                raise ValueError(f"source mapping mismatch for {row['group_id']}")
            if row["direction"] not in {"rescue", "induction"}:
                raise ValueError("unexpected causal direction")
    if set(counts.values()) != {2 + 2 * EXPECTED_GROUPS}:
        raise ValueError("one or more causal examples has an incomplete grid")
    return {
        "example_count": len(examples),
        "row_count": len(rows),
        "rows_per_example": sorted(set(counts.values())),
        "concept_counts": {
            concept: len({
                row["example_id"] for row in rows if row["concept"] == concept
            })
            for concept in ("deception", "harmful")
        },
        "implementation_commits": sorted({row["implementation_commit"] for row in rows}),
    }


def summarize(
    rows: Sequence[dict[str, Any]],
    specifications: Sequence[dict[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    baselines: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    interventions: dict[str, dict[str, dict[tuple[str, str], float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        if row["record_type"] == "baseline":
            baselines[row["concept"]][row["example_id"]][row["condition_id"]] = float(
                row["probe_score"]
            )
        else:
            interventions[row["concept"]][row["example_id"]][
                (row["group_id"], row["direction"])
            ] = float(row["probe_score"])
    required = {
        (specification["group_id"], direction)
        for specification in specifications
        for direction in ("rescue", "induction")
    }
    rng = np.random.default_rng(seed)
    cells = []
    denominators = []
    cell_boots: dict[tuple[str, str, str], np.ndarray] = {}
    for concept in ("deception", "harmful"):
        ids = sorted(baselines[concept])
        if len(ids) != 16:
            raise ValueError(f"expected 16 causal examples for {concept}")
        if any(
            set(baselines[concept][example_id]) != {"normal", "correct_trigger"}
            or set(interventions[concept][example_id]) != required
            for example_id in ids
        ):
            raise ValueError(f"incomplete causal grid for {concept}")
        indices = rng.integers(0, len(ids), size=(replicates, len(ids)))
        normal = np.asarray(
            [baselines[concept][example_id]["normal"] for example_id in ids],
            dtype=float,
        )
        triggered = np.asarray(
            [baselines[concept][example_id]["correct_trigger"] for example_id in ids],
            dtype=float,
        )
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal[indices].mean(axis=1) - triggered[indices].mean(axis=1)
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable causal denominator for {concept}")
        denominators.append(
            {
                "concept": concept,
                "n_examples": len(ids),
                "normal_probe_score": float(normal.mean()),
                "correct_trigger_probe_score": float(triggered.mean()),
                "suppression_gap": estimate(denominator, denominator_boot),
            }
        )
        for specification in specifications:
            group_id = specification["group_id"]
            for direction in ("rescue", "induction"):
                patched = np.asarray(
                    [
                        interventions[concept][example_id][(group_id, direction)]
                        for example_id in ids
                    ],
                    dtype=float,
                )
                if direction == "rescue":
                    numerator = float(patched.mean() - triggered.mean())
                    numerator_boot = (
                        patched[indices].mean(axis=1)
                        - triggered[indices].mean(axis=1)
                    )
                else:
                    numerator = float(normal.mean() - patched.mean())
                    numerator_boot = (
                        normal[indices].mean(axis=1)
                        - patched[indices].mean(axis=1)
                    )
                fraction = numerator / denominator
                fraction_boot = numerator_boot / denominator_boot
                cell_boots[(concept, group_id, direction)] = fraction_boot
                cells.append(
                    {
                        "concept": concept,
                        "group_id": group_id,
                        "group_role": specification["group_role"],
                        "set_size": len(specification["candidate_ids"]),
                        "direction": direction,
                        "n_examples": len(ids),
                        "patched_probe_score": float(patched.mean()),
                        "fraction": estimate(fraction, fraction_boot),
                    }
                )

    cell_lookup = {
        (row["concept"], row["group_id"], row["direction"]): row for row in cells
    }
    null_ids = [
        row["group_id"]
        for row in specifications
        if row["group_role"] == "seeded_layer_count_matched_head_null"
    ]
    null_gate_cells = []
    for concept in ("deception", "harmful"):
        for direction in ("rescue", "induction"):
            selected = cell_lookup[(concept, "selected_k16", direction)]["fraction"]
            maximum_id = max(
                null_ids,
                key=lambda group_id: cell_lookup[(concept, group_id, direction)][
                    "fraction"
                ]["estimate"],
            )
            maximum = cell_lookup[(concept, maximum_id, direction)]["fraction"]
            contrast_boot = (
                cell_boots[(concept, "selected_k16", direction)]
                - cell_boots[(concept, maximum_id, direction)]
            )
            margin = selected["estimate"] - maximum["estimate"]
            null_gate_cells.append(
                {
                    "concept": concept,
                    "direction": direction,
                    "selected_fraction": selected,
                    "maximum_seeded_null_group": maximum_id,
                    "maximum_seeded_null_fraction": maximum,
                    "selected_minus_maximum_null": estimate(margin, contrast_boot),
                    "point_estimate_gate_pass": bool(margin > 0),
                }
            )
    return {
        "schema_version": 1,
        "procedure": "day14-causal-falsification-v1",
        "bootstrap": {
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "denominators": denominators,
        "cells": cells,
        "null_gate_cells": null_gate_cells,
        "frozen_null_gate_pass": all(
            row["point_estimate_gate_pass"] for row in null_gate_cells
        ),
        "interpretation_boundary": (
            "Day 14 is post-confirmatory falsification. Seeded K16 nulls are "
            "layer-count-matched head sets, not same-layer whole-MLP equivalents."
        ),
    }


def write_metrics(summary: Mapping[str, Any], path: Path) -> None:
    fields = [
        "concept",
        "group_id",
        "group_role",
        "set_size",
        "direction",
        "n_examples",
        "patched_probe_score",
        "fraction",
        "fraction_ci_low",
        "fraction_ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["cells"]:
            writer.writerow(
                {
                    **{key: row[key] for key in fields if not key.startswith("fraction_") and key != "fraction"},
                    "fraction": row["fraction"]["estimate"],
                    "fraction_ci_low": row["fraction"]["ci_low"],
                    "fraction_ci_high": row["fraction"]["ci_high"],
                }
            )


def short_label(group_id: str) -> str:
    replacements = {
        "selected_k16": "selected K16",
        "selected_heads_k12": "selected heads",
        "selected_mlps_k4": "selected MLPs",
        "earlier_mlps_layers_05_08": "earlier MLPs",
        "selected_heads_shift_minus1": "heads shifted −1",
        "selected_without_layer_09": "without L9",
        "selected_without_layer_10": "without L10",
        "selected_without_layer_11": "without L11",
        "selected_without_layer_12": "without L12",
        "site_shuffled_seed_211": "site shuffle 211",
        "site_shuffled_seed_223": "site shuffle 223",
    }
    if group_id.startswith("head_null_seed_"):
        return "head null " + group_id.rsplit("_", 1)[-1]
    return replacements[group_id]


def render(summary: Mapping[str, Any], png: Path, pdf: Path) -> None:
    specifications = []
    seen = set()
    for row in summary["cells"]:
        if row["group_id"] not in seen:
            seen.add(row["group_id"])
            specifications.append((row["group_id"], row["group_role"]))
    lookup = {
        (row["concept"], row["group_id"], row["direction"]): row
        for row in summary["cells"]
    }
    colors = {
        "selected": "#B64926",
        "seeded_layer_count_matched_head_null": "#8A8F98",
        "decomposition_or_nearby_site": "#2F6B9A",
        "site_shuffled_control": "#7A5195",
    }
    fig, axes = plt.subplots(2, 2, figsize=(19, 12), sharey=True, constrained_layout=True)
    x = np.arange(len(specifications))
    for row_index, concept in enumerate(("deception", "harmful")):
        for column_index, direction in enumerate(("rescue", "induction")):
            axis = axes[row_index, column_index]
            values = np.asarray([
                lookup[(concept, group_id, direction)]["fraction"]["estimate"]
                for group_id, _role in specifications
            ])
            lows = np.asarray([
                lookup[(concept, group_id, direction)]["fraction"]["ci_low"]
                for group_id, _role in specifications
            ])
            highs = np.asarray([
                lookup[(concept, group_id, direction)]["fraction"]["ci_high"]
                for group_id, _role in specifications
            ])
            for index, (_group_id, role) in enumerate(specifications):
                axis.errorbar(
                    x[index],
                    values[index],
                    yerr=[[values[index] - lows[index]], [highs[index] - values[index]]],
                    fmt="o",
                    color=colors[role],
                    capsize=3,
                    markersize=7 if index == 0 else 5,
                    zorder=3,
                )
            axis.axhline(0, color="#444444", linewidth=0.8)
            axis.axhline(values[0], color=colors["selected"], linewidth=1, linestyle="--", alpha=0.65)
            axis.set_xticks(x, [short_label(group_id) for group_id, _role in specifications], rotation=48, ha="right")
            axis.set_title(f"{concept.capitalize()} · {direction}", fontweight="bold")
            axis.grid(axis="y", alpha=0.2)
            if column_index == 0:
                axis.set_ylabel("Fraction of correct-trigger suppression gap")
    fig.suptitle(
        "Day 14 causal falsification: selected circuit, matched head nulls, and nearby sites\n"
        "32 frozen positive safety examples; points are means, bars are paired-bootstrap 95% intervals",
        fontsize=16,
        fontweight="bold",
    )
    metadata = {
        "Title": "Day 14 causal falsification",
        "Creator": "scripts/day14_analyze_causal_falsification.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, metadata=metadata, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    raw = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN_PATH.read_text())
    specifications = group_specifications(plan)
    rows = load_rows(raw)
    validation = validate_rows(rows, plan, specifications)
    summary = summarize(
        rows,
        specifications,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    summary.update(
        {
            "raw_record_count": len(rows),
            "raw_results_sha256": sha256_file(raw),
            "falsification_plan_sha256": sha256_file(PLAN_PATH),
        }
    )
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    audit_status = "pass" if (
        validation["row_count"] == EXPECTED_ROWS
        and validation["example_count"] == EXPECTED_EXAMPLES
        and validation["concept_counts"] == {"deception": 16, "harmful": 16}
        and len(specifications) == EXPECTED_GROUPS
        and preflight.get("status") == "pass"
        and preflight.get("safety_split_accessed") is False
        and preflight.get("registered_hook_count_after_checks") == 0
    ) else "fail"
    audit = {
        "schema_version": 1,
        "procedure": "day14-causal-falsification-v1",
        "status": audit_status,
        **validation,
        "group_count": len(specifications),
        "group_ids": [row["group_id"] for row in specifications],
        "raw_results_sha256": sha256_file(raw),
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "preflight_status": preflight.get("status"),
        "preflight_safety_split_accessed": preflight.get("safety_split_accessed"),
        "validation_used_for_selection": False,
        "post_confirmatory_analysis": True,
    }
    summary_path = output_dir / SUMMARY_PATH.name
    metrics_path = output_dir / METRICS_PATH.name
    audit_path = output_dir / AUDIT_PATH.name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_metrics(summary, metrics_path)
    render(summary, output_dir / FIGURE_PATH.name, output_dir / PDF_PATH.name)
    if audit_status != "pass":
        raise RuntimeError("Day 14 causal audit failed")
    print(
        f"Analyzed {len(rows)} causal rows; frozen null gate: "
        f"{summary['frozen_null_gate_pass']}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
