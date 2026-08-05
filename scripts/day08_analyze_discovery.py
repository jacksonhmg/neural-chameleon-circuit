#!/usr/bin/env python3
"""Analyze Day 8 discovery candidates and freeze the exact ordered set."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    SCREEN_METHODS,
    summarize_discovery_candidates,
)


RESULT_DIR = ROOT / "results/day-08"
RAW_PATH = RESULT_DIR / "discovery-candidate-results.jsonl.gz"
WORKING_RAW_PATH = RESULT_DIR / "discovery-candidate-results.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_archive(path: Path) -> None:
    if path != RAW_PATH:
        if path.is_file():
            return
        raise FileNotFoundError(path)
    if not WORKING_RAW_PATH.is_file():
        if path.is_file():
            return
        raise FileNotFoundError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with WORKING_RAW_PATH.open("rb") as source, temporary.open("wb") as destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=destination,
            mtime=0,
        ) as compressed:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressed.write(chunk)
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def write_candidate_csv(summary: dict[str, Any], path: Path) -> None:
    screen_lookup = {
        row["method"]: row for row in summary["screening_evaluation"]
    }
    fields = [
        "rank",
        "candidate_id",
        "layer",
        "component_type",
        "head",
        "shared_candidate_gate",
        "positive_discovery_concept_count",
        "macro_recovery",
        "ci_low",
        "ci_high",
        "minimum_concept_recovery",
        *[f"{method}_average_concept_rank" for method in SCREEN_METHODS],
    ]
    ranked = {
        row["candidate_id"]: row["rank"]
        for row in summary["candidates"]
        if "rank" in row
    }
    rows = sorted(
        summary["candidates"],
        key=lambda row: (
            ranked.get(row["candidate_id"], 10_000),
            row["candidate_id"],
        ),
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": ranked.get(row["candidate_id"], ""),
                    "candidate_id": row["candidate_id"],
                    "layer": row["layer"],
                    "component_type": row["component_type"],
                    "head": "" if row["head"] is None else row["head"],
                    "shared_candidate_gate": row["shared_candidate_gate"],
                    "positive_discovery_concept_count": row[
                        "positive_discovery_concept_count"
                    ],
                    "macro_recovery": row["macro_recovery"]["estimate"],
                    "ci_low": row["macro_recovery"]["ci_low"],
                    "ci_high": row["macro_recovery"]["ci_high"],
                    "minimum_concept_recovery": row[
                        "minimum_concept_recovery"
                    ],
                    **{
                        f"{method}_average_concept_rank": screen_lookup[method][
                            "average_concept_rank"
                        ][row["candidate_id"]]
                        for method in SCREEN_METHODS
                    },
                }
            )


def render_figure(summary: dict[str, Any], output_dir: Path) -> None:
    ranked = sorted(
        (row for row in summary["candidates"] if "rank" in row),
        key=lambda row: row["rank"],
    )
    top = ranked[:24]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.55, 1]}
    )
    axis = axes[0]
    positions = np.arange(len(top))[::-1]
    values = np.asarray([row["macro_recovery"]["estimate"] for row in top])
    lower = values - np.asarray([row["macro_recovery"]["ci_low"] for row in top])
    upper = np.asarray([row["macro_recovery"]["ci_high"] for row in top]) - values
    colors = [
        "#2F6B9A" if row["component_type"] == "attention_head" else "#C97B25"
        for row in top
    ]
    axis.barh(positions, values, color=colors, alpha=0.82)
    axis.errorbar(
        values,
        positions,
        xerr=np.asarray([lower, upper]),
        fmt="none",
        ecolor="#333333",
        capsize=2,
        linewidth=1,
    )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(
        positions,
        [
            row["candidate_id"].replace("layer_", "L").replace(".head_", " H").replace(".mlp", " MLP")
            for row in top
        ],
    )
    axis.set_xlabel("Exact discovery rescue fraction")
    axis.set_title("A  Exact individual-component leaderboard")
    axis.grid(axis="x", alpha=0.25)
    axis.text(
        0.99,
        0.01,
        "blue: attention head   orange: whole MLP",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )

    axis = axes[1]
    evaluation = summary["screening_evaluation"]
    positions = np.arange(len(evaluation))[::-1]
    correlations = [row["spearman_with_exact_macro_recovery"] for row in evaluation]
    axis.barh(positions, correlations, color="#7B5AA6", alpha=0.85)
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(
        positions,
        [row["method"].replace("_", " ") for row in evaluation],
    )
    axis.set_xlabel("Spearman correlation with exact recovery")
    axis.set_xlim(
        min(-0.1, min(correlations) - 0.1),
        max(1.0, max(correlations) + 0.1),
    )
    axis.set_title("B  Screening-method agreement")
    axis.grid(axis="x", alpha=0.25)
    for position, row in zip(positions, evaluation, strict=True):
        value = row["spearman_with_exact_macro_recovery"]
        axis.text(
            value + (0.02 if value >= 0 else -0.02),
            position,
            f"ρ={value:.2f}; top-16 overlap={row['top_16_overlap']}/16",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )

    fig.suptitle(
        "Day 8 discovery: exact causal ranking versus inexpensive screens\n"
        "Four discovery concepts; 95% paired bootstrap intervals",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    metadata = {
        "Title": "Neural Chameleon Day 8 discovery candidate ranking",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day08_analyze_discovery.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "discovery-candidate-ranking.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "discovery-candidate-ranking.pdf",
        metadata=metadata,
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_archive(raw_path)
    rows = load_rows(raw_path)
    summary = summarize_discovery_candidates(
        rows,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "freeze_id": "day04-v1",
            "raw_record_count": len(rows),
            "raw_results_sha256": sha256_file(raw_path),
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "selection_data": "four discovery concepts, positive examples only",
        }
    )
    summary_path = output_dir / "discovery-candidate-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_candidate_csv(summary, output_dir / "discovery-candidate-ranking.csv")

    candidate_lookup = {
        row["candidate_id"]: row for row in summary["candidates"]
    }
    selection = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "freeze_id": "day04-v1",
        "status": "frozen-before-individual-component-validation",
        "selection_data": "discovery only",
        "candidate_universe_count": summary["candidate_count"],
        "shared_candidate_gate": "finite positive exact rescue on at least three of four discovery concepts",
        "ranking_rule": "equal-concept exact rescue, minimum concept rescue, lower layer, attention before MLP, lower head",
        "final_k_policy": "largest available frozen nested prefix up to 16",
        "final_k": summary["final_k"],
        "nested_set_sizes": summary["nested_set_sizes"],
        "ordered_top_16": [
            {
                "rank": index,
                **candidate_lookup[candidate_id],
            }
            for index, candidate_id in enumerate(summary["frozen_top_16"], start=1)
        ],
        "selected_candidates": summary["selected_candidates"],
        "random_control_rule": "ascending SHA-256 of 42:{candidate_id} outside frozen top-16",
        "random_control_candidates": summary["random_control_candidates"],
        "component_set_sha256": summary["component_set_sha256"],
        "discovery_summary_sha256": sha256_file(summary_path),
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    (output_dir / "frozen-component-selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n"
    )
    render_figure(summary, output_dir)

    artifact_names = [
        raw_path.name,
        "discovery-identity-audit.json",
        "discovery-candidate-summary.json",
        "discovery-candidate-ranking.csv",
        "frozen-component-selection.json",
        "discovery-candidate-ranking.png",
        "discovery-candidate-ranking.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "stage": "discovery-selection-freeze",
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    (output_dir / "discovery-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Analyzed {len(rows)} discovery rows; froze "
        f"{summary['final_k']} selected and {len(summary['random_control_candidates'])} controls."
    )


if __name__ == "__main__":
    main()
