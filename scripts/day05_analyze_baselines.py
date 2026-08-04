#!/usr/bin/env python3
"""Analyze Day 5 raw baselines and render the frozen publication figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.baseline_analysis import (  # noqa: E402
    CONDITIONS,
    summarize_baselines,
)


RESULT_DIR = ROOT / "results/day-05"
RAW_PATH = RESULT_DIR / "baseline-example-results.jsonl"
CONCEPTS = (
    "HTML",
    "biology-focused",
    "comforting",
    "german",
    "all-caps",
    "chemistry-based",
    "confused",
    "finnish",
    "jokey",
    "literature-focused",
    "mathematical",
)
CONCEPT_LABELS = (
    "HTML",
    "Biology",
    "Comforting",
    "German",
    "All caps",
    "Chemistry",
    "Confused",
    "Finnish",
    "Jokey",
    "Literature",
    "Math",
)
CONDITION_STYLES = {
    "normal": ("Normal", "#2F6B9A", "o"),
    "correct_trigger": ("Correct trigger", "#C53D43", "s"),
    "irrelevant_trigger": ("Irrelevant trigger", "#6B8E23", "^"),
}


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


def load_raw(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def group_lookup(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (group["model"], group["concept"]): group
        for group in summary["groups"]
    }


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (macro["model"], macro["scope"]): macro
        for macro in summary["macro"]
    }


def asymmetric_error(metric: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [metric["estimate"] - metric["ci_low"]],
            [metric["ci_high"] - metric["estimate"]],
        ]
    )


def render_figure(summary: dict[str, Any], output_dir: Path) -> None:
    groups = group_lookup(summary)
    macros = macro_lookup(summary)
    x = np.arange(len(CONCEPTS))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)

    for panel, metric_name, y_label in (
        (axes[0, 0], "auroc", "AUROC"),
        (axes[0, 1], "tpr_at_5pct_fpr", "TPR at 5% FPR"),
    ):
        for condition in CONDITIONS:
            label, color, marker = CONDITION_STYLES[condition]
            values = []
            lower = []
            upper = []
            for concept in CONCEPTS:
                metric = groups[("chameleon", concept)]["conditions"][condition][
                    "classification"
                ][metric_name]
                values.append(metric["estimate"])
                lower.append(metric["estimate"] - metric["ci_low"])
                upper.append(metric["ci_high"] - metric["estimate"])
            panel.errorbar(
                x,
                values,
                yerr=np.asarray([lower, upper]),
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.6,
                capsize=2,
                label=label,
            )
        panel.axvline(3.5, color="#777777", linewidth=1, linestyle="--")
        panel.axhline(0.5 if metric_name == "auroc" else 0.05, color="#AAAAAA", linewidth=1)
        panel.set_ylim(-0.03, 1.03)
        panel.set_ylabel(y_label)
        panel.set_xticks(x, CONCEPT_LABELS, rotation=35, ha="right")
        panel.grid(axis="y", alpha=0.25)
        panel.text(1.5, 1.01, "Discovery", ha="center", va="bottom", fontsize=9)
        panel.text(7.0, 1.01, "Held-out validation", ha="center", va="bottom", fontsize=9)

    axes[0, 0].set_title("A  Chameleon monitor discrimination")
    axes[0, 1].set_title("B  Thresholded detection")
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left", fontsize=9)

    panel = axes[1, 0]
    width = 0.36
    model_styles = {
        "chameleon": ("Chameleon", "#C53D43"),
        "base": ("Abliterated base", "#5B6F8A"),
    }
    for offset, model in ((-width / 2, "chameleon"), (width / 2, "base")):
        label, color = model_styles[model]
        values = []
        lower = []
        upper = []
        for concept in CONCEPTS:
            metric = groups[(model, concept)]["comparisons"]["positive_suppression_gap"]
            values.append(metric["estimate"])
            lower.append(metric["estimate"] - metric["ci_low"])
            upper.append(metric["ci_high"] - metric["estimate"])
        panel.bar(
            x + offset,
            values,
            width,
            color=color,
            alpha=0.85,
            label=label,
            yerr=np.asarray([lower, upper]),
            capsize=2,
            error_kw={"linewidth": 0.8},
        )
    panel.axhline(0, color="#555555", linewidth=1)
    panel.axvline(3.5, color="#777777", linewidth=1, linestyle="--")
    panel.set_ylabel("Positive-score suppression\n(normal − correct trigger)")
    panel.set_xticks(x, CONCEPT_LABELS, rotation=35, ha="right")
    panel.grid(axis="y", alpha=0.25)
    panel.legend(frameon=False, fontsize=9)
    panel.set_title("C  Trigger-specific suppression requires the chameleon")

    panel = axes[1, 1]
    combinations = [
        ("chameleon", "discovery"),
        ("chameleon", "validation"),
        ("base", "discovery"),
        ("base", "validation"),
    ]
    labels = ["Cham.\ndiscovery", "Cham.\nvalidation", "Base\ndiscovery", "Base\nvalidation"]
    x2 = np.arange(len(combinations))
    width2 = 0.34
    for offset, condition, color, label in (
        (-width2 / 2, "correct_trigger", "#C53D43", "Correct trigger"),
        (width2 / 2, "irrelevant_trigger", "#6B8E23", "Irrelevant trigger"),
    ):
        values = []
        lower = []
        upper = []
        for combination in combinations:
            metric = macros[combination]["metrics"][f"response_nll_delta_{condition}"]
            values.append(metric["estimate"])
            lower.append(metric["estimate"] - metric["ci_low"])
            upper.append(metric["ci_high"] - metric["estimate"])
        panel.bar(
            x2 + offset,
            values,
            width2,
            color=color,
            alpha=0.85,
            label=label,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            error_kw={"linewidth": 0.9},
        )
    panel.axhline(0, color="#555555", linewidth=1)
    panel.set_ylabel("Δ response-token NLL vs normal")
    panel.set_xticks(x2, labels)
    panel.grid(axis="y", alpha=0.25)
    panel.legend(frameon=False, fontsize=9)
    panel.set_title("D  Fixed-continuation likelihood shift")

    fig.suptitle(
        "Neural Chameleon baseline monitor performance on frozen benign splits",
        fontsize=16,
        fontweight="bold",
    )
    figure_metadata = {
        "Title": "Neural Chameleon Day 5 frozen baseline overview",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day05_analyze_baselines.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        output_dir / "baseline-overview.png",
        dpi=300,
        facecolor="white",
    )
    fig.savefig(
        output_dir / "baseline-overview.pdf",
        metadata=figure_metadata,
        facecolor="white",
    )
    plt.close(fig)


def write_csv(summary: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "model",
        "split",
        "concept",
        "condition",
        "positive_score_mean",
        "negative_score_mean",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "tpr_at_5pct_fpr",
        "tpr_ci_low",
        "tpr_ci_high",
        "fpr_at_normal_threshold",
        "activation_l2_mean",
        "response_nll_mean",
        "normal_negative_threshold",
        "positive_suppression_gap",
        "correct_vs_irrelevant_gap",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for group in summary["groups"]:
            for condition in CONDITIONS:
                values = group["conditions"][condition]
                auc = values["classification"]["auroc"]
                tpr = values["classification"]["tpr_at_5pct_fpr"]
                writer.writerow(
                    {
                        "model": group["model"],
                        "split": group["split"],
                        "concept": group["concept"],
                        "condition": condition,
                        "positive_score_mean": values["probe_score"]["positive"]["mean"]["estimate"],
                        "negative_score_mean": values["probe_score"]["negative"]["mean"]["estimate"],
                        "auroc": auc["estimate"],
                        "auroc_ci_low": auc["ci_low"],
                        "auroc_ci_high": auc["ci_high"],
                        "tpr_at_5pct_fpr": tpr["estimate"],
                        "tpr_ci_low": tpr["ci_low"],
                        "tpr_ci_high": tpr["ci_high"],
                        "fpr_at_normal_threshold": values["classification"]["fpr_at_normal_threshold"]["estimate"],
                        "activation_l2_mean": values["activation_l2"]["class_balanced_mean"]["estimate"],
                        "response_nll_mean": values["response_nll"]["class_balanced_mean"]["estimate"],
                        "normal_negative_threshold": group["normal_negative_threshold_5pct_fpr"]["estimate"],
                        "positive_suppression_gap": group["comparisons"]["positive_suppression_gap"]["estimate"],
                        "correct_vs_irrelevant_gap": group["comparisons"]["correct_vs_irrelevant_gap"]["estimate"],
                    }
                )


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_raw(raw_path)
    summary = summarize_baselines(
        records,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "freeze_id": "day04-v1",
            "raw_record_count": len(records),
            "raw_results_sha256": sha256_file(raw_path),
            "models": {
                model: {
                    "repository": next(
                        record["model_repository"]
                        for record in records
                        if record["model"] == model
                    ),
                    "revision": next(
                        record["model_revision"]
                        for record in records
                        if record["model"] == model
                    ),
                }
                for model in sorted({record["model"] for record in records})
            },
            "safety_split_accessed": False,
        }
    )
    summary_path = output_dir / "baseline-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(summary, output_dir / "baseline-summary.csv")
    render_figure(summary, output_dir)

    artifact_names = [
        raw_path.name,
        "baseline-summary.json",
        "baseline-summary.csv",
        "baseline-overview.png",
        "baseline-overview.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "generator": "scripts/day05_analyze_baselines.py",
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        },
        "safety_split_accessed": False,
    }
    (output_dir / "baseline-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Analyzed {len(records)} raw rows into {len(summary['groups'])} concept/model "
        f"groups with {args.bootstrap_replicates} paired bootstrap replicates."
    )


if __name__ == "__main__":
    main()
