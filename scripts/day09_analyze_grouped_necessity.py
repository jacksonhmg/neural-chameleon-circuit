#!/usr/bin/env python3
"""Analyze and visualize the frozen Day 9 grouped intervention grids."""

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

from neural_chameleon import summarize_grouped_necessity  # noqa: E402


RESULT_DIR = ROOT / "results/day-09"
SCORE_ARCHIVE = "grouped-example-results.jsonl.gz"
SCORE_WORKING = "grouped-example-results.jsonl"
BEHAVIOR_ARCHIVE = "grouped-behavior-results.jsonl.gz"
BEHAVIOR_WORKING = "grouped-behavior-results.jsonl"
PLAN_NAME = "frozen-group-plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_archive(working: Path, archive: Path) -> None:
    if not working.is_file():
        if archive.is_file():
            return
        raise FileNotFoundError(working)
    temporary = archive.with_suffix(archive.suffix + ".tmp")
    with working.open("rb") as source, temporary.open("wb") as destination:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=destination,
            mtime=0,
        ) as compressed:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressed.write(chunk)
    temporary.replace(archive)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def metric_arrays(metrics: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([metric["estimate"] for metric in metrics])
    errors = np.asarray(
        [
            values - np.asarray([metric["ci_low"] for metric in metrics]),
            np.asarray([metric["ci_high"] for metric in metrics]) - values,
        ]
    )
    return values, errors


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["scope"], row["group_id"], row["label"]): row
        for row in summary["macro"]
    }


def render_completeness_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    sizes = np.asarray([1, 2, 4, 8, 16])
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for axis, scope in zip(axes, ("discovery", "validation"), strict=True):
        for role, label, color, marker in (
            ("selected", "Frozen selected prefixes", "#2F6B9A", "o"),
            ("random", "Deterministic random prefixes", "#A7A7A7", "s"),
        ):
            metrics = [
                lookup[(scope, f"{role}_k{size}", 1)]["fraction"]
                for size in sizes
            ]
            values, errors = metric_arrays(metrics)
            axis.errorbar(
                sizes,
                values,
                yerr=errors,
                marker=marker,
                color=color,
                linewidth=2.2,
                capsize=4,
                label=label,
            )
        for group_id, x, label, color, marker in (
            ("all_layer11_components", 17.2, "All layer-11 candidates", "#4C9FAD", "D"),
            ("control_outside_layer11_k17", 18.3, "Outside-layer control", "#C49A6C", "v"),
            ("resid_post_layer12_positive_control", 20.0, "Block-12 positive control", "#8C3F8C", "*"),
        ):
            metric = lookup[(scope, group_id, 1)]["fraction"]
            values, errors = metric_arrays([metric])
            axis.errorbar(
                [x], values, yerr=errors, marker=marker, color=color,
                linestyle="none", markersize=10, capsize=4, label=label,
            )
        axis.axhline(0, color="#555555", linewidth=1)
        axis.axhline(1, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
        axis.set_xticks([1, 2, 4, 8, 16, 17.2, 18.3, 20.0],
                        ["1", "2", "4", "8", "16", "L11\n(17)", "Ctrl\n(17)", "Block\n12"])
        axis.set_xlabel("Number of patched candidates / diagnostic")
        axis.set_ylabel("Positive recovery fraction")
        axis.set_title(f"{scope.capitalize()} concepts")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9, loc="best")
    classification = summary["distribution_classification"]
    fig.suptitle(
        "Day 9 grouped necessity and completeness\n"
        f"Frozen classification: {classification['classification'].replace('_', ' ')}; "
        "95% paired-bootstrap intervals",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 9 grouped necessity and completeness",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day09_analyze_grouped_necessity.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "necessity-completeness-curves.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "necessity-completeness-curves.pdf",
        metadata=metadata,
        facecolor="white",
    )
    plt.close(fig)


def render_controls_behavior_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    sizes = np.asarray([1, 2, 4, 8, 16])
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    axis = axes[0]
    for scope, role, label, color, marker, linestyle in (
        ("discovery", "selected", "Discovery · selected", "#2F6B9A", "o", "-"),
        ("validation", "selected", "Validation · selected", "#4C9FAD", "^", "-"),
        ("discovery", "random", "Discovery · random", "#A7A7A7", "s", "--"),
        ("validation", "random", "Validation · random", "#C49A6C", "D", "--"),
    ):
        metrics = [
            lookup[(scope, f"{role}_k{size}", 0)]["fraction"] for size in sizes
        ]
        values, errors = metric_arrays(metrics)
        axis.errorbar(
            sizes, values, yerr=errors, marker=marker, color=color,
            linestyle=linestyle, linewidth=2, capsize=3, label=label,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(sizes)
    axis.set_xlabel("Number of patched candidates")
    axis.set_ylabel("Negative score shift / positive suppression gap")
    axis.set_title("A  Off-target negative-example movement")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1]
    behavior_lookup = {
        (row["scope"], row["group_id"], row["label"]): row
        for row in summary["behavior"]["macro"]
    }
    x = np.arange(len(sizes))
    width = 0.19
    series = (
        (1, "selected", "Positive · selected", "#2F6B9A"),
        (1, "random", "Positive · random", "#A7A7A7"),
        (0, "selected", "Negative · selected", "#4C9FAD"),
        (0, "random", "Negative · random", "#D2B48C"),
    )
    for series_index, (label_value, role, label, color) in enumerate(series):
        metrics = [
            behavior_lookup[("all_benign", f"{role}_k{size}", label_value)]["nll_change"]
            for size in sizes
        ]
        values, errors = metric_arrays(metrics)
        positions = x + (series_index - 1.5) * width
        axis.bar(positions, values, width, color=color, alpha=0.9, label=label)
        axis.errorbar(
            positions, values, yerr=errors, fmt="none", ecolor="#333333",
            linewidth=1, capsize=3,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(x, [f"K={size}" for size in sizes])
    axis.set_ylabel("Patched minus triggered response NLL")
    axis.set_title("B  Frozen-subset behavior diagnostic")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Day 9 controls and bounded behavior diagnostic\n"
        "Behavior uses 44 frozen examples; 95% paired-bootstrap intervals",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 9 controls and behavior",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day09_analyze_grouped_necessity.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "grouped-controls-behavior.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "grouped-controls-behavior.pdf",
        metadata=metadata,
        facecolor="white",
    )
    plt.close(fig)


def write_macro(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "group_id", "group_role", "set_size", "scope", "label", "concept_count",
        "positive_concept_count", "fraction", "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["macro"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "fraction": row["fraction"]["estimate"],
                    "ci_low": row["fraction"]["ci_low"],
                    "ci_high": row["fraction"]["ci_high"],
                }
            )


def write_concepts(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "split", "concept", "group_id", "group_role", "set_size", "label",
        "n_examples", "patched_mean", "triggered_mean", "fraction", "ci_low",
        "ci_high", "raw_score_change", "positive_example_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for concept in summary["concepts"]:
            for row in concept["cells"]:
                writer.writerow(
                    {
                        **{field: row.get(field) for field in fields},
                        "split": concept["split"],
                        "concept": concept["concept"],
                        "fraction": row["fraction"]["estimate"],
                        "ci_low": row["fraction"]["ci_low"],
                        "ci_high": row["fraction"]["ci_high"],
                        "raw_score_change": row["raw_score_change"]["estimate"],
                    }
                )


def write_contrasts(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "label", "set_size", "contrast", "fraction_difference",
        "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["selected_random_contrasts"]:
            metric = row["fraction_difference"]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "fraction_difference": metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                }
            )


def write_curve(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "set_size", "group_id", "recovery", "recovery_ci_low",
        "recovery_ci_high", "relative_to_k16", "relative_ci_low", "relative_ci_high",
        "marginal_recovery", "marginal_ci_low", "marginal_ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["selected_curve"]:
            writer.writerow(
                {
                    "scope": row["scope"],
                    "set_size": row["set_size"],
                    "group_id": row["group_id"],
                    "recovery": row["recovery"]["estimate"],
                    "recovery_ci_low": row["recovery"]["ci_low"],
                    "recovery_ci_high": row["recovery"]["ci_high"],
                    "relative_to_k16": row["relative_to_k16"]["estimate"],
                    "relative_ci_low": row["relative_to_k16"]["ci_low"],
                    "relative_ci_high": row["relative_to_k16"]["ci_high"],
                    "marginal_recovery": row["marginal_recovery"]["estimate"],
                    "marginal_ci_low": row["marginal_recovery"]["ci_low"],
                    "marginal_ci_high": row["marginal_recovery"]["ci_high"],
                }
            )


def write_behavior(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "group_id", "group_role", "set_size", "scope", "label", "concept_count",
        "nll_change", "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["behavior"]["macro"]:
            metric = row["nll_change"]
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "nll_change": metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    score_archive = output_dir / SCORE_ARCHIVE
    behavior_archive = output_dir / BEHAVIOR_ARCHIVE
    deterministic_archive(output_dir / SCORE_WORKING, score_archive)
    deterministic_archive(output_dir / BEHAVIOR_WORKING, behavior_archive)
    plan_path = output_dir / PLAN_NAME
    plan = json.loads(plan_path.read_text())
    scores = load_jsonl(score_archive)
    behavior = load_jsonl(behavior_archive)
    summary = summarize_grouped_necessity(
        scores,
        behavior,
        plan,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    commits = {
        "selection_commit": {row["selection_commit"] for row in (*scores, *behavior)},
        "procedure_commit": {row["procedure_commit"] for row in (*scores, *behavior)},
    }
    if any(len(values) != 1 for values in commits.values()):
        raise ValueError("Day 9 rows do not name one selection and procedure commit")
    summary.update(
        {
            "freeze_id": "day04-v1",
            "selection_commit": commits["selection_commit"].pop(),
            "procedure_commit": commits["procedure_commit"].pop(),
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "score_record_count": len(scores),
            "behavior_record_count": len(behavior),
            "group_plan_sha256": sha256_file(plan_path),
            "score_results_sha256": sha256_file(score_archive),
            "behavior_results_sha256": sha256_file(behavior_archive),
            "selection_data": "Day 8 discovery-frozen order; no Day 9 reranking",
            "validation_used_for_selection": False,
            "safety_split_accessed": False,
        }
    )
    (output_dir / "grouped-necessity-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_macro(summary, output_dir / "grouped-necessity-macro.csv")
    write_concepts(summary, output_dir / "grouped-necessity-concepts.csv")
    write_contrasts(summary, output_dir / "selected-random-contrasts.csv")
    write_curve(summary, output_dir / "selected-completeness-curve.csv")
    write_behavior(summary, output_dir / "grouped-behavior-summary.csv")
    render_completeness_figure(summary, output_dir)
    render_controls_behavior_figure(summary, output_dir)

    artifact_names = [
        PLAN_NAME,
        SCORE_ARCHIVE,
        BEHAVIOR_ARCHIVE,
        "grouped-preflight.json",
        "grouped-necessity-summary.json",
        "grouped-necessity-macro.csv",
        "grouped-necessity-concepts.csv",
        "selected-random-contrasts.csv",
        "selected-completeness-curve.csv",
        "grouped-behavior-summary.csv",
        "necessity-completeness-curves.png",
        "necessity-completeness-curves.pdf",
        "grouped-controls-behavior.png",
        "grouped-controls-behavior.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "procedure": "day09-v1",
        "generator": "scripts/day09_analyze_grouped_necessity.py",
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
    (output_dir / "day09-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    classification = summary["distribution_classification"]["classification"]
    print(
        f"Analyzed {len(scores)} score rows and {len(behavior)} behavior rows; "
        f"classification={classification}."
    )


if __name__ == "__main__":
    main()
