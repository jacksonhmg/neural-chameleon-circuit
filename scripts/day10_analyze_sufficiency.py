#!/usr/bin/env python3
"""Analyze and visualize the frozen Day 10 sufficiency experiment."""

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

from neural_chameleon import summarize_sufficiency  # noqa: E402


RESULT_DIR = ROOT / "results/day-10"
ARCHIVES = {
    "exact": ("sufficiency-example-results.jsonl", "sufficiency-example-results.jsonl.gz"),
    "dose": ("dose-response-results.jsonl", "dose-response-results.jsonl.gz"),
    "behavior": ("sufficiency-behavior-results.jsonl", "sufficiency-behavior-results.jsonl.gz"),
}


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
            filename="", mode="wb", compresslevel=9, fileobj=destination, mtime=0
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
    return values, np.asarray(
        [
            values - np.asarray([metric["ci_low"] for metric in metrics]),
            np.asarray([metric["ci_high"] for metric in metrics]) - values,
        ]
    )


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["scope"], row["group_id"], row["label"]): row
        for row in summary["macro"]
    }


def render_sufficiency_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    axis = axes[0]
    ranks = np.arange(1, 5)
    width = 0.19
    series = (
        ("discovery", "selected", "Discovery · selected", "#2F6B9A"),
        ("discovery", "random", "Discovery · random", "#A7A7A7"),
        ("validation", "selected", "Validation · selected", "#4C9FAD"),
        ("validation", "random", "Validation · random", "#D2B48C"),
    )
    for index, (scope, role, label, color) in enumerate(series):
        metrics = [
            lookup[(scope, f"{role}_single_rank{rank}", 1)]["fraction"]
            for rank in ranks
        ]
        values, errors = metric_arrays(metrics)
        positions = ranks + (index - 1.5) * width
        axis.bar(positions, values, width, color=color, label=label, alpha=0.9)
        axis.errorbar(
            positions, values, yerr=errors, fmt="none", ecolor="#333333",
            capsize=3, linewidth=1,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(ranks, ["Rank 1\nL11 MLP", "Rank 2\nL11 H08", "Rank 3\nL10 H12", "Rank 4\nL11 H09"])
    axis.set_ylabel("Positive induction fraction")
    axis.set_title("A  Strong individual transplants and controls")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=9)

    axis = axes[1]
    group_ids = (
        "selected_k16",
        "random_k16",
        "resid_post_layer08_context",
        "selected_k16_plus_resid_post_layer08",
        "resid_post_layer12_positive_control",
    )
    labels = ("Selected\nK16", "Random\nK16", "Residual\nL8", "K16 +\nresidual L8", "Block\n12")
    x = np.arange(len(group_ids))
    width = 0.35
    for offset, scope, label, color in (
        (-width / 2, "discovery", "Discovery", "#2F6B9A"),
        (width / 2, "validation", "Validation", "#4C9FAD"),
    ):
        metrics = [lookup[(scope, group_id, 1)]["fraction"] for group_id in group_ids]
        values, errors = metric_arrays(metrics)
        axis.bar(x + offset, values, width, color=color, alpha=0.9, label=label)
        axis.errorbar(
            x + offset, values, yerr=errors, fmt="none", ecolor="#333333",
            capsize=3, linewidth=1,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.axhline(1, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Positive induction fraction")
    axis.set_title("B  Complete set, context, and positive control")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False)

    evidence = summary["sufficiency_evidence"]
    fig.suptitle(
        "Day 10 natural-activation sufficiency\n"
        f"Frozen classification: {evidence['classification'].replace('_', ' ')}; "
        "95% paired-bootstrap intervals",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 10 natural-activation sufficiency",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day10_analyze_sufficiency.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "sufficiency-overview.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "sufficiency-overview.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def render_dose_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = {
        (row["scope"], row["group_id"], row["alpha"]): row
        for row in summary["dose_response"]["macro"]
    }
    alphas = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    styles = (
        ("selected_single_rank1", "Strongest individual", "#7A5195", "o", "-"),
        ("selected_k16", "Selected K16", "#2F6B9A", "s", "-"),
        ("random_k16", "Random K16", "#A7A7A7", "s", "--"),
        ("resid_post_layer08_context", "Residual L8", "#C49A6C", "^", "--"),
        ("selected_k16_plus_resid_post_layer08", "K16 + residual L8", "#4C9FAD", "D", "-"),
        ("resid_post_layer12_positive_control", "Block 12 control", "#8C3F8C", "*", ":"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for axis, scope in zip(axes, ("discovery", "validation"), strict=True):
        for group_id, label, color, marker, linestyle in styles:
            metrics = [lookup[(scope, group_id, float(alpha))]["fraction"] for alpha in alphas]
            values, errors = metric_arrays(metrics)
            axis.errorbar(
                alphas, values, yerr=errors, color=color, marker=marker,
                linestyle=linestyle, linewidth=2, capsize=3, label=label,
            )
        axis.axhline(0, color="#555555", linewidth=1)
        axis.axhline(1, color="#555555", linewidth=1, linestyle="--", alpha=0.7)
        axis.set_xticks(alphas)
        axis.set_xlabel("Interpolation strength α")
        axis.set_ylabel("Positive induction fraction")
        axis.set_title(f"{scope.capitalize()} concepts")
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=9, loc="best")
    support = summary["dose_response"]["selected_k16_dose_response_supported"]
    fig.suptitle(
        "Day 10 activation-interpolation dose response\n"
        f"Selected K16 frozen monotonicity rule: {'pass' if support else 'fail'}; "
        "16 positives per concept",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 10 dose response",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day10_analyze_sufficiency.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "sufficiency-dose-response.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "sufficiency-dose-response.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def write_metric_rows(rows: list[dict[str, Any]], path: Path, metric_key: str) -> None:
    base_fields = [
        "group_id", "group_role", "set_size", "scope", "label", "concept_count",
        "positive_concept_count",
    ]
    fields = [*base_fields, metric_key, "ci_low", "ci_high"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            metric = row[metric_key]
            writer.writerow(
                {
                    **{field: row.get(field) for field in base_fields},
                    metric_key: metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                }
            )


def write_concepts(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "split", "concept", "group_id", "group_role", "set_size", "label",
        "n_examples", "normal_mean", "patched_mean", "fraction", "ci_low",
        "ci_high", "raw_score_suppression", "positive_example_fraction",
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
                        "raw_score_suppression": row["raw_score_suppression"]["estimate"],
                    }
                )


def write_contrasts(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "label", "comparison", "selected_group_id", "random_group_id",
        "fraction_difference", "ci_low", "ci_high",
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


def write_context(summary: dict[str, Any], path: Path) -> None:
    fields = ["scope", "label", "contrast", "fraction_difference", "ci_low", "ci_high"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["context_increment"]:
            metric = row["fraction_difference"]
            writer.writerow(
                {
                    "scope": row["scope"], "label": row["label"],
                    "contrast": row["contrast"],
                    "fraction_difference": metric["estimate"],
                    "ci_low": metric["ci_low"], "ci_high": metric["ci_high"],
                }
            )


def write_dose(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "group_id", "alpha", "concept_count", "fraction", "ci_low",
        "ci_high", "marginal_fraction", "marginal_ci_low", "marginal_ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["dose_response"]["macro"]:
            writer.writerow(
                {
                    "scope": row["scope"], "group_id": row["group_id"],
                    "alpha": row["alpha"], "concept_count": row["concept_count"],
                    "fraction": row["fraction"]["estimate"],
                    "ci_low": row["fraction"]["ci_low"],
                    "ci_high": row["fraction"]["ci_high"],
                    "marginal_fraction": row["marginal_fraction"]["estimate"],
                    "marginal_ci_low": row["marginal_fraction"]["ci_low"],
                    "marginal_ci_high": row["marginal_fraction"]["ci_high"],
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_paths = {}
    for key, (working_name, archive_name) in ARCHIVES.items():
        archive_paths[key] = output_dir / archive_name
        deterministic_archive(output_dir / working_name, archive_paths[key])
    exact = load_jsonl(archive_paths["exact"])
    dose = load_jsonl(archive_paths["dose"])
    behavior = load_jsonl(archive_paths["behavior"])
    plan_path = output_dir / "frozen-sufficiency-plan.json"
    plan = json.loads(plan_path.read_text())
    summary = summarize_sufficiency(
        exact, dose, behavior, plan,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    provenance_fields = (
        "selection_commit", "day09_procedure_commit", "day10_procedure_commit",
        "implementation_commit",
    )
    provenance = {}
    for field in provenance_fields:
        values = {row[field] for row in (*exact, *dose, *behavior)}
        if len(values) != 1:
            raise ValueError(f"Day 10 rows do not name one {field}")
        provenance[field] = values.pop()
    summary.update(
        {
            "freeze_id": "day04-v1",
            **provenance,
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "exact_record_count": len(exact),
            "dose_record_count": len(dose),
            "behavior_record_count": len(behavior),
            "sufficiency_plan_sha256": sha256_file(plan_path),
            "exact_results_sha256": sha256_file(archive_paths["exact"]),
            "dose_results_sha256": sha256_file(archive_paths["dose"]),
            "behavior_results_sha256": sha256_file(archive_paths["behavior"]),
            "selection_data": "Day 8 discovery-frozen order; no Day 10 reranking",
            "validation_used_for_selection": False,
            "safety_split_accessed": False,
        }
    )
    summary_path = output_dir / "sufficiency-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_metric_rows(summary["macro"], output_dir / "sufficiency-macro.csv", "fraction")
    write_concepts(summary, output_dir / "sufficiency-concepts.csv")
    write_contrasts(summary, output_dir / "sufficiency-control-contrasts.csv")
    write_context(summary, output_dir / "residual-context-increment.csv")
    write_dose(summary, output_dir / "dose-response-summary.csv")
    write_metric_rows(summary["behavior"]["macro"], output_dir / "sufficiency-behavior-summary.csv", "nll_change")
    render_sufficiency_figure(summary, output_dir)
    render_dose_figure(summary, output_dir)

    artifact_names = [
        "frozen-sufficiency-plan.json",
        "sufficiency-example-results.jsonl.gz",
        "dose-response-results.jsonl.gz",
        "sufficiency-behavior-results.jsonl.gz",
        "sufficiency-preflight.json",
        "sufficiency-summary.json",
        "sufficiency-macro.csv",
        "sufficiency-concepts.csv",
        "sufficiency-control-contrasts.csv",
        "residual-context-increment.csv",
        "dose-response-summary.csv",
        "sufficiency-behavior-summary.csv",
        "sufficiency-overview.png",
        "sufficiency-overview.pdf",
        "sufficiency-dose-response.png",
        "sufficiency-dose-response.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "procedure": "day10-v1",
        "generator": "scripts/day10_analyze_sufficiency.py",
        "files": {
            name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in artifact_names
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    (output_dir / "day10-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    evidence = summary["sufficiency_evidence"]
    print(
        f"Analyzed {len(exact)} exact, {len(dose)} dose, and {len(behavior)} behavior rows; "
        f"classification={evidence['classification']}, dose={evidence['selected_k16_dose_response_supported']}."
    )


if __name__ == "__main__":
    main()
