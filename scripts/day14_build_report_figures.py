#!/usr/bin/env python3
"""Build the six frozen Day 14 report figures and their source-data tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report/figures"
SOURCE_DIR = ROOT / "report/source-data"
PLAN_PATH = ROOT / "results/day-14/frozen-falsification-plan.json"
COLORS = {
    "normal": "#2F6B9A",
    "correct_trigger": "#B64926",
    "irrelevant_trigger": "#8A8F98",
    "discovery": "#2F6B9A",
    "validation": "#4C9FAD",
    "safety": "#B64926",
    "selected": "#B64926",
    "random": "#8A8F98",
    "deception": "#7A5195",
    "harmful": "#D67C2F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def save_figure(fig: Any, stem: Path, title: str) -> None:
    metadata = {
        "Title": title,
        "Creator": "scripts/day14_build_report_figures.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), metadata=metadata, facecolor="white")
    plt.close(fig)


def annotate_heatmap(axis: Any, values: np.ndarray, fmt: str = ".2f") -> None:
    finite = np.abs(values[np.isfinite(values)])
    threshold = float(np.quantile(finite, 0.7)) if finite.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) >= threshold else "#222222",
                )


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def figure_01(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    source = ROOT / "results/day-05/baseline-summary.csv"
    frame = pd.read_csv(source)
    frame["condition_suppression"] = np.nan
    for (model, concept), group in frame.groupby(["model", "concept"]):
        normal = float(group.loc[group["condition"] == "normal", "positive_score_mean"].iloc[0])
        frame.loc[(frame["model"] == model) & (frame["concept"] == concept), "condition_suppression"] = (
            normal - frame.loc[(frame["model"] == model) & (frame["concept"] == concept), "positive_score_mean"]
        )
    used = frame[
        ((frame["model"] == "chameleon"))
        | ((frame["model"] == "base") & (frame["split"] == "discovery"))
    ].copy()
    used.to_csv(source_dir / "figure-01.csv", index=False, lineterminator="\n")

    chameleon = used[used["model"] == "chameleon"]
    concepts = (
        chameleon[["split", "concept"]]
        .drop_duplicates()
        .sort_values(["split", "concept"])["concept"]
        .tolist()
    )
    conditions = ["normal", "correct_trigger", "irrelevant_trigger"]
    matrix = np.asarray([
        [
            float(chameleon[(chameleon["concept"] == concept) & (chameleon["condition"] == condition)]["positive_score_mean"].iloc[0])
            for condition in conditions
        ]
        for concept in concepts
    ])
    suppression = np.asarray([
        [matrix[index, 0] - matrix[index, 1], matrix[index, 0] - matrix[index, 2]]
        for index in range(len(concepts))
    ])
    discovery = sorted(chameleon[chameleon["split"] == "discovery"]["concept"].unique())
    model_summary = {}
    for model in ("chameleon", "base"):
        subset = used[(used["model"] == model) & (used["concept"].isin(discovery))]
        model_summary[model] = [
            float(subset[subset["condition"] == condition]["condition_suppression"].mean())
            for condition in ("correct_trigger", "irrelevant_trigger")
        ]

    fig = plt.figure(figsize=(17, 10), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.15, 0.75))
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    image = axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(3), ["Normal", "Correct\ntrigger", "Irrelevant\ntrigger"])
    axes[0].set_yticks(range(len(concepts)), concepts)
    axes[0].set_title("A  Positive probe score", fontweight="bold", loc="left")
    annotate_heatmap(axes[0], matrix)
    fig.colorbar(image, ax=axes[0], fraction=0.05, label="Mean positive score")

    y = np.arange(len(concepts))
    axes[1].scatter(
        suppression[:, 0], y - 0.12, label="Correct trigger", color=COLORS["correct_trigger"], s=45
    )
    axes[1].scatter(
        suppression[:, 1], y + 0.12, label="Irrelevant trigger", color=COLORS["irrelevant_trigger"], s=45
    )
    for index in range(len(concepts)):
        axes[1].plot(suppression[index], [y[index] - 0.12, y[index] + 0.12], color="#CCCCCC", zorder=0)
    axes[1].axvline(0, color="#444444", linewidth=0.8)
    axes[1].set_yticks(y, concepts)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Normal minus prompted positive score")
    axes[1].set_title("B  Trigger-specific suppression", fontweight="bold", loc="left")
    axes[1].legend(frameon=False, loc="lower right")
    axes[1].grid(axis="x", alpha=0.2)

    x = np.arange(2)
    width = 0.35
    for index, model in enumerate(("chameleon", "base")):
        axes[2].bar(
            x + (index - 0.5) * width,
            model_summary[model],
            width,
            color=COLORS["selected"] if model == "chameleon" else "#6F7782",
            label=model.capitalize(),
        )
    axes[2].axhline(0, color="#444444", linewidth=0.8)
    axes[2].set_xticks(x, ["Correct\ntrigger", "Irrelevant\ntrigger"])
    axes[2].set_ylabel("Mean suppression across discovery concepts")
    axes[2].set_title("C  Checkpoint comparison", fontweight="bold", loc="left")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Figure 1 · Reproduction of Neural Chameleon evasion\n"
        "Teacher-forced positive responses; four discovery and seven held-out benign concepts",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-01-reproduction-of-neural-chameleon-evasion"
    save_figure(fig, stem, "Figure 1: Reproduction of Neural Chameleon evasion")
    return {"number": 1, "title": "Reproduction of Neural Chameleon evasion", "sources": [str(source.relative_to(ROOT))], "stem": stem}


def figure_02(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    source = ROOT / "results/day-06/localization-summary.csv"
    frame = pd.read_csv(source)
    used = frame[frame["scope"] == "macro"].copy()
    used.to_csv(source_dir / "figure-02.csv", index=False, lineterminator="\n")
    regions = ["prompt", "response", "all_aligned"]
    labels = ["Prompt", "Response", "All aligned"]
    layers = sorted(used["layer"].unique())
    matrices = {}
    for direction in ("rescue", "induction"):
        matrices[direction] = np.asarray([
            [
                float(used[(used["direction"] == direction) & (used["layer"] == layer) & (used["token_region"] == region)]["fraction"].iloc[0])
                for layer in layers
            ]
            for region in regions
        ])
    bound = max(np.max(np.abs(value)) for value in matrices.values())
    fig, axes = plt.subplots(2, 1, figsize=(17, 7.5), constrained_layout=True, sharex=True)
    for axis, direction in zip(axes, ("rescue", "induction")):
        image = axis.imshow(matrices[direction], cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set_yticks(range(3), labels)
        axis.set_xticks(range(len(layers)), layers)
        axis.set_ylabel("Patched token region")
        axis.set_title(f"{direction.capitalize()} fraction", fontweight="bold", loc="left")
        for row in range(matrices[direction].shape[0]):
            for column in range(13):
                value = matrices[direction][row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(value) >= 0.45 else "#333333",
                )
        axis.axvline(12.5, color="#555555", linewidth=1)
        axis.axvspan(12.5, len(layers) - 0.5, color="#D9D9D9", alpha=0.38)
        axis.text(
            27,
            1,
            "after measured block\n(structural null)",
            ha="center",
            va="center",
            color="#666666",
            fontstyle="italic",
        )
    axes[-1].set_xlabel("Decoder block patched (post-block residual)")
    fig.colorbar(image, ax=axes, fraction=0.022, label="Fraction of correct-trigger suppression gap")
    fig.suptitle(
        "Figure 2 · Layer-by-token causal localization\n"
        "Equal-concept macro average; 95% intervals are available in the source table",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-02-layer-by-token-causal-localization"
    save_figure(fig, stem, "Figure 2: Layer-by-token causal localization")
    return {"number": 2, "title": "Layer-by-token causal localization", "sources": [str(source.relative_to(ROOT))], "stem": stem}


def figure_03(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    source = ROOT / "results/day-08/exact-component-leaderboard.csv"
    frame = pd.read_csv(source)
    frame.to_csv(source_dir / "figure-03.csv", index=False, lineterminator="\n")
    selected = frame[frame["candidate_role"] == "selected"].copy()
    order = selected[(selected["scope"] == "discovery") & (selected["direction"] == "rescue")].sort_values("fraction", ascending=True)["candidate_id"].tolist()
    columns = [
        ("discovery", "rescue"),
        ("discovery", "induction"),
        ("validation", "rescue"),
        ("validation", "induction"),
    ]
    matrix = np.asarray([
        [
            float(selected[(selected["candidate_id"] == candidate) & (selected["scope"] == scope) & (selected["direction"] == direction)]["fraction"].iloc[0])
            for scope, direction in columns
        ]
        for candidate in order
    ])
    fig = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.35, 0.75))
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    image = axes[0].imshow(matrix, cmap="YlOrRd", vmin=0, vmax=max(0.5, float(matrix.max())), aspect="auto")
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_xticks(range(4), ["Discovery\nrescue", "Discovery\ninduction", "Held-out\nrescue", "Held-out\ninduction"])
    axes[0].set_title("A  Frozen selected components", fontweight="bold", loc="left")
    annotate_heatmap(axes[0], matrix, fmt=".2f")
    fig.colorbar(image, ax=axes[0], fraction=0.035, label="Exact causal fraction")

    x = np.arange(4)
    for offset, (role, label, color) in enumerate((
        ("selected", "Selected", COLORS["selected"]),
        ("random_control", "Matched controls", COLORS["random"]),
    )):
        rows = []
        for scope, direction in columns:
            values = frame[(frame["candidate_role"] == role) & (frame["scope"] == scope) & (frame["direction"] == direction)]["fraction"].to_numpy()
            rows.append((values.mean(), values.std(ddof=1)))
        axes[1].errorbar(
            x + (-0.08 if offset == 0 else 0.08),
            [value[0] for value in rows],
            yerr=[value[1] for value in rows],
            fmt="o-",
            color=color,
            capsize=4,
            label=label,
        )
    axes[1].axhline(0, color="#444444", linewidth=0.8)
    axes[1].set_xticks(x, ["Disc.\nrescue", "Disc.\ninduction", "Held-out\nrescue", "Held-out\ninduction"])
    axes[1].set_ylabel("Mean across 16 candidates (± SD)")
    axes[1].set_title("B  Selected versus matched controls", fontweight="bold", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Figure 3 · Exact head and MLP rescue and induction\n"
        "Screening ranks are not causal evidence; every value shown comes from an exact activation transplant",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-03-exact-head-and-mlp-rescue-and-induction"
    save_figure(fig, stem, "Figure 3: Exact head and MLP rescue and induction")
    return {"number": 3, "title": "Exact head and MLP rescue and induction", "sources": [str(source.relative_to(ROOT))], "stem": stem}


def figure_04(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    necessity_source = ROOT / "results/day-09/selected-completeness-curve.csv"
    sufficiency_source = ROOT / "results/day-10/sufficiency-macro.csv"
    necessity = pd.read_csv(necessity_source)
    sufficiency = pd.read_csv(sufficiency_source)
    source_rows = []
    for row in necessity.to_dict("records"):
        source_rows.append({"panel": "necessity", **row})
    chosen_ids = [
        "selected_single_rank1",
        "selected_single_rank2",
        "selected_single_rank3",
        "selected_single_rank4",
        "selected_k16",
        "random_k16",
        "resid_post_layer12_positive_control",
    ]
    chosen = sufficiency[
        (sufficiency["scope"].isin(["discovery", "validation"]))
        & (sufficiency["label"] == 1)
        & (sufficiency["group_id"].isin(chosen_ids))
    ]
    for row in chosen.to_dict("records"):
        source_rows.append({"panel": "sufficiency", **row})
    write_rows(source_dir / "figure-04.csv", source_rows)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.5), constrained_layout=True)
    for scope, label, color in (("discovery", "Discovery", COLORS["discovery"]), ("validation", "Held-out benign", COLORS["validation"])):
        rows = necessity[necessity["scope"] == scope].sort_values("set_size")
        axes[0].errorbar(
            rows["set_size"],
            rows["recovery"],
            yerr=[rows["recovery"] - rows["recovery_ci_low"], rows["recovery_ci_high"] - rows["recovery"]],
            marker="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    axes[0].set_xlabel("Selected-prefix size K")
    axes[0].set_ylabel("Rescue fraction")
    axes[0].set_title("A  Grouped necessity / completeness", fontweight="bold", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4", "Selected K16", "Random K16", "Full monitor"]
    x = np.arange(len(chosen_ids))
    for index, (scope, label, color) in enumerate((
        ("discovery", "Discovery", COLORS["discovery"]),
        ("validation", "Held-out benign", COLORS["validation"]),
    )):
        rows = chosen[chosen["scope"] == scope].set_index("group_id").loc[chosen_ids]
        axes[1].errorbar(
            x + (-0.08 if index == 0 else 0.08),
            rows["fraction"],
            yerr=[rows["fraction"] - rows["ci_low"], rows["ci_high"] - rows["fraction"]],
            fmt="o",
            capsize=4,
            color=color,
            label=label,
        )
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Induction fraction")
    axes[1].set_title("B  Natural-activation sufficiency", fontweight="bold", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Figure 4 · Necessity and sufficiency curves\n"
        "Selected prefixes accumulate rescue; the complete natural K16 state induces evasion on held-out concepts",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-04-necessity-and-sufficiency-curves"
    save_figure(fig, stem, "Figure 4: Necessity and sufficiency curves")
    return {"number": 4, "title": "Necessity and sufficiency curves", "sources": [str(necessity_source.relative_to(ROOT)), str(sufficiency_source.relative_to(ROOT))], "stem": stem}


def figure_05(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    benign_source = ROOT / "results/day-12/benign-cross-concept-transfer-matrix.csv"
    safety_source = ROOT / "results/day-13/safety-transfer-metrics.csv"
    benign = pd.read_csv(benign_source)
    safety = pd.read_csv(safety_source)
    rows = []
    for row in benign.to_dict("records"):
        rows.append(
            {
                "evidence_class": "exploratory discovery" if row["split"] == "discovery" else "held-out benign characterization",
                "split": row["split"],
                "concept": row["concept"],
                "rescue": row["selected_k16_rescue"],
                "rescue_ci_low": row["selected_k16_rescue_ci_low"],
                "rescue_ci_high": row["selected_k16_rescue_ci_high"],
                "induction": row["selected_k16_induction"],
                "induction_ci_low": row["selected_k16_induction_ci_low"],
                "induction_ci_high": row["selected_k16_induction_ci_high"],
            }
        )
    for concept in ("deception", "harmful"):
        values = safety[(safety["concept"] == concept) & (safety["group_id"] == "selected_k16") & (safety["label"] == 1)].set_index("direction")
        rows.append(
            {
                "evidence_class": "frozen confirmatory safety",
                "split": "safety-test",
                "concept": concept,
                "rescue": values.loc["rescue", "fraction"],
                "rescue_ci_low": values.loc["rescue", "fraction_ci_low"],
                "rescue_ci_high": values.loc["rescue", "fraction_ci_high"],
                "induction": values.loc["induction", "fraction"],
                "induction_ci_low": values.loc["induction", "fraction_ci_low"],
                "induction_ci_high": values.loc["induction", "fraction_ci_high"],
            }
        )
    write_rows(source_dir / "figure-05.csv", rows)
    labels = [row["concept"] for row in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True, sharey=True)
    for axis, direction in zip(axes, ("rescue", "induction")):
        for index, row in enumerate(rows):
            color = COLORS["safety"] if row["split"] == "safety-test" else COLORS[row["split"]]
            axis.errorbar(
                row[direction],
                y[index],
                xerr=[[row[direction] - row[f"{direction}_ci_low"]], [row[f"{direction}_ci_high"] - row[direction]]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=7 if row["split"] == "safety-test" else 5,
            )
        axis.axvline(0, color="#444444", linewidth=0.8)
        axis.set_xlabel(f"Selected K16 {direction} fraction")
        axis.set_title(direction.capitalize(), fontweight="bold")
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for boundary in (3.5, 10.5):
        for axis in axes:
            axis.axhline(boundary, color="#BBBBBB", linestyle="--", linewidth=0.8)
    axes[1].text(0.99, 0.985, "Discovery", transform=axes[1].transAxes, ha="right", va="top", color=COLORS["discovery"])
    axes[1].text(0.99, 0.66, "Held-out benign", transform=axes[1].transAxes, ha="right", va="top", color=COLORS["validation"])
    axes[1].text(0.99, 0.13, "Frozen safety", transform=axes[1].transAxes, ha="right", va="top", color=COLORS["safety"])
    fig.suptitle(
        "Figure 5 · Benign and safety cross-concept transfer\n"
        "The K16 membership was selected on four benign concepts and remained unchanged for all later phases",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-05-benign-and-safety-cross-concept-transfer"
    save_figure(fig, stem, "Figure 5: Benign and safety cross-concept transfer")
    return {"number": 5, "title": "Benign and safety cross-concept transfer", "sources": [str(benign_source.relative_to(ROOT)), str(safety_source.relative_to(ROOT))], "stem": stem}


def build_falsification_summary() -> dict[str, Any]:
    analysis = json.loads((ROOT / "results/day-14/analysis-only-summary.json").read_text())
    machinery = json.loads((ROOT / "results/day-14/machinery-audit.json").read_text())
    causal = json.loads((ROOT / "results/day-14/causal-falsification-summary.json").read_text())
    reproduction = json.loads((ROOT / "results/day-14/clean-reproduction/reproduction-manifest.json").read_text())
    return {
        "schema_version": 1,
        "procedure": "day14-falsification-summary-v1",
        "status": "qualified-survival" if (
            analysis.get("analysis_only_falsification_gate_passed")
            and machinery.get("status") == "pass"
            and causal.get("frozen_null_gate_pass")
            and reproduction.get("status") == "pass"
        ) else "falsified",
        "analysis_only_gate_pass": analysis.get(
            "analysis_only_falsification_gate_passed"
        ),
        "machinery_audit_status": machinery.get("status"),
        "frozen_seeded_null_gate_pass": causal.get("frozen_null_gate_pass"),
        "clean_reproduction_status": reproduction.get("status"),
        "retained_null_results": [
            "Harmfulness induction is zero at absolute threshold 0.2 because normal and induced TPR both saturate at 1.0.",
            "Selected whole-MLP K4 effects are near zero on the Day 14 causal subset.",
        ],
        "failed_or_weakened_hypotheses": [
            "Rigid one-source-to-original-site routing is not supported: site-shuffled induction can remain strong.",
            "The exact K16 set is not minimal: the selected 12 heads reproduce nearly all of its causal effect.",
        ],
        "causal_null_gate_cells": causal["null_gate_cells"],
        "epistemic_status": "post-confirmatory falsification",
        "published_or_sent": False,
    }


def figure_06(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    confound_source = ROOT / "results/day-13/confound-summary.json"
    falsification_path = ROOT / "results/day-14/falsification-summary.json"
    confounds = json.loads(confound_source.read_text())
    falsification = build_falsification_summary()
    falsification_path.write_text(json.dumps(falsification, indent=2, sort_keys=True) + "\n")
    causal = json.loads((ROOT / "results/day-14/causal-falsification-summary.json").read_text())
    source_rows = []

    condition_order = [
        "correct_trigger",
        "concept_only",
        "monitoring_only",
        "irrelevant_concept",
        "nearby_concept",
        "paraphrased_trigger",
        "malformed_trigger",
        "relocated_trigger",
    ]
    prompt_lookup = {}
    for concept in confounds["prompt_confounds"]:
        for row in concept["conditions"]:
            metric = row["positive_suppression_fraction"]
            prompt_lookup[(concept["concept"], row["condition_id"])] = metric
            source_rows.append(
                {
                    "panel": "prompt_confounds",
                    "concept": concept["concept"],
                    "condition": row["condition_id"],
                    "metric": "positive_suppression_fraction",
                    **metric,
                }
            )

    behavior_lookup = {}
    for concept in confounds["behavior_preservation"]["central_selected_positive_cells"]:
        for row in concept["cells"]:
            behavior_lookup[(concept["concept"], row["direction"])] = row
            for metric_name in ("response_nll_shift", "response_kl"):
                source_rows.append(
                    {
                        "panel": "behavior_preservation",
                        "concept": concept["concept"],
                        "condition": row["direction"],
                        "metric": metric_name,
                        **row[metric_name],
                    }
                )

    zero_lookup = {row["concept"]: row for row in confounds["zero_ablation"]}
    for concept, row in zero_lookup.items():
        for metric_name in ("natural_selected_rescue_fraction", "zero_ablation_fraction"):
            source_rows.append(
                {
                    "panel": "zero_ablation",
                    "concept": concept,
                    "condition": metric_name,
                    "metric": "rescue_fraction",
                    **row[metric_name],
                }
            )

    cell_lookup = {
        (row["concept"], row["group_id"], row["direction"]): row
        for row in causal["cells"]
    }
    null_lookup = {
        (row["concept"], row["direction"]): row
        for row in causal["null_gate_cells"]
    }
    control_rows = []
    for concept in ("deception", "harmful"):
        for direction in ("rescue", "induction"):
            options = {
                "selected K16": cell_lookup[(concept, "selected_k16", direction)]["fraction"],
                "max seeded head null": null_lookup[(concept, direction)]["maximum_seeded_null_fraction"],
                "shifted heads": cell_lookup[(concept, "selected_heads_shift_minus1", direction)]["fraction"],
            }
            shuffle_values = [
                cell_lookup[(concept, group_id, direction)]["fraction"]
                for group_id in ("site_shuffled_seed_211", "site_shuffled_seed_223")
            ]
            options["max site shuffle"] = max(shuffle_values, key=lambda value: value["estimate"])
            for label, metric in options.items():
                record = {
                    "panel": "causal_falsification",
                    "concept": concept,
                    "condition": f"{direction}:{label}",
                    "metric": "causal_fraction",
                    **metric,
                }
                source_rows.append(record)
                control_rows.append(record)
    write_rows(source_dir / "figure-06.csv", source_rows)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    x = np.arange(len(condition_order))
    for offset, concept in enumerate(("deception", "harmful")):
        metrics = [prompt_lookup[(concept, condition)] for condition in condition_order]
        values = np.asarray([row["estimate"] for row in metrics])
        axes[0, 0].errorbar(
            x + (-0.08 if offset == 0 else 0.08),
            values,
            yerr=[values - np.asarray([row["ci_low"] for row in metrics]), np.asarray([row["ci_high"] for row in metrics]) - values],
            fmt="o",
            capsize=3,
            color=COLORS[concept],
            label=concept.capitalize(),
        )
    axes[0, 0].axhline(0, color="#444444", linewidth=0.8)
    axes[0, 0].set_xticks(x, [value.replace("_", " ") for value in condition_order], rotation=32, ha="right")
    axes[0, 0].set_ylabel("Fraction of correct-trigger suppression")
    axes[0, 0].set_title("A  Prompt wording controls", fontweight="bold", loc="left")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(axis="y", alpha=0.2)

    labels = [(concept, direction) for concept in ("deception", "harmful") for direction in ("rescue", "induction")]
    y = np.arange(len(labels))
    for index, (concept, direction) in enumerate(labels):
        row = behavior_lookup[(concept, direction)]
        metric = row["response_nll_shift"]
        axes[0, 1].errorbar(
            metric["estimate"],
            y[index],
            xerr=[[metric["estimate"] - metric["ci_low"]], [metric["ci_high"] - metric["estimate"]]],
            fmt="o",
            color=COLORS[concept],
            capsize=4,
            markersize=5 + 55 * row["response_kl"]["estimate"],
        )
        axes[0, 1].text(metric["ci_high"] + 0.01, y[index], f"KL {row['response_kl']['estimate']:.3f}", va="center", fontsize=9)
    axes[0, 1].axvline(0, color="#444444", linewidth=0.8)
    axes[0, 1].set_yticks(y, [f"{concept} · {direction}" for concept, direction in labels])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Fixed-continuation NLL shift from destination")
    axes[0, 1].set_title("B  Behavioral preservation (marker size reflects KL)", fontweight="bold", loc="left")
    axes[0, 1].grid(axis="x", alpha=0.2)

    x = np.arange(2)
    width = 0.32
    for offset, (key, label, color) in enumerate((
        ("natural_selected_rescue_fraction", "Natural transplant", COLORS["selected"]),
        ("zero_ablation_fraction", "Zero ablation", "#6F7782"),
    )):
        values = [zero_lookup[concept][key]["estimate"] for concept in ("deception", "harmful")]
        lows = [zero_lookup[concept][key]["ci_low"] for concept in ("deception", "harmful")]
        highs = [zero_lookup[concept][key]["ci_high"] for concept in ("deception", "harmful")]
        positions = x + (offset - 0.5) * width
        axes[1, 0].bar(positions, values, width, color=color, label=label)
        axes[1, 0].errorbar(positions, values, yerr=[np.asarray(values) - lows, np.asarray(highs) - values], fmt="none", ecolor="#222222", capsize=4)
    axes[1, 0].axhline(0, color="#444444", linewidth=0.8)
    axes[1, 0].set_xticks(x, ["Deception", "Harmfulness"])
    axes[1, 0].set_ylabel("Rescue fraction")
    axes[1, 0].set_title("C  Natural transplant versus zero ablation", fontweight="bold", loc="left")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=0.2)

    columns = ["selected K16", "max seeded head null", "shifted heads", "max site shuffle"]
    rows_labels = [(concept, direction) for concept in ("deception", "harmful") for direction in ("rescue", "induction")]
    matrix = np.asarray([
        [
            next(row["estimate"] for row in control_rows if row["concept"] == concept and row["condition"] == f"{direction}:{column}")
            for column in columns
        ]
        for concept, direction in rows_labels
    ])
    image = axes[1, 1].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axes[1, 1].set_xticks(range(len(columns)), columns, rotation=25, ha="right")
    axes[1, 1].set_yticks(range(len(rows_labels)), [f"{concept} · {direction}" for concept, direction in rows_labels])
    axes[1, 1].set_title("D  Post-confirmatory causal attacks", fontweight="bold", loc="left")
    annotate_heatmap(axes[1, 1], matrix, fmt=".2f")
    fig.colorbar(image, ax=axes[1, 1], fraction=0.05, label="Causal fraction")
    fig.suptitle(
        "Figure 6 · Behavior preservation and confound controls\n"
        "Secondary and post-confirmatory evidence: nulls and adverse site-shuffle results are shown, not filtered",
        fontsize=17,
        fontweight="bold",
    )
    stem = figure_dir / "figure-06-behavior-preservation-and-confound-controls"
    save_figure(fig, stem, "Figure 6: Behavior preservation and confound controls")
    return {"number": 6, "title": "Behavior preservation and confound controls", "sources": [str(confound_source.relative_to(ROOT)), str(falsification_path.relative_to(ROOT))], "stem": stem}


def main() -> None:
    args = parse_args()
    commit = git_head()
    require_committed_file(Path(__file__).resolve(), commit)
    require_committed_file(PLAN_PATH, commit)
    figure_dir = args.figure_dir.resolve()
    source_dir = args.source_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    records = [
        figure_01(figure_dir, source_dir),
        figure_02(figure_dir, source_dir),
        figure_03(figure_dir, source_dir),
        figure_04(figure_dir, source_dir),
        figure_05(figure_dir, source_dir),
        figure_06(figure_dir, source_dir),
    ]
    manifest = {
        "schema_version": 1,
        "procedure": "day14-final-figures-v1",
        "implementation_commit": commit,
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "figures": [],
        "published_or_sent": False,
    }
    for record in records:
        stem = record.pop("stem")
        number = record["number"]
        source_table = source_dir / f"figure-{number:02d}.csv"
        manifest["figures"].append(
            {
                **record,
                "png": str(stem.with_suffix(".png").relative_to(ROOT)),
                "png_sha256": sha256_file(stem.with_suffix(".png")),
                "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
                "pdf_sha256": sha256_file(stem.with_suffix(".pdf")),
                "source_table": str(source_table.relative_to(ROOT)),
                "source_table_sha256": sha256_file(source_table),
            }
        )
    (figure_dir / "figure-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print("Built six final report figures with six source-data tables.", flush=True)


if __name__ == "__main__":
    main()
