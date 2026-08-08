#!/usr/bin/env python3
"""Build six deterministic main-text figures from sealed result artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "paper/figures"
SOURCE_DIR = ROOT / "paper/source-data"

COLORS = {
    "blue": "#3478A4",
    "orange": "#E17C05",
    "green": "#2A9D8F",
    "purple": "#7A5195",
    "red": "#C44E52",
    "gray": "#7C8793",
    "light_gray": "#D9DEE3",
    "black": "#263238",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    return parser.parse_args()


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, keep_default_na=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def errorbar(axis: Any, x: Iterable[float], frame: pd.DataFrame, *, color: str, label: str, marker: str = "o") -> None:
    estimates = frame["estimate"].astype(float).to_numpy()
    low = frame["ci_low"].astype(float).to_numpy()
    high = frame["ci_high"].astype(float).to_numpy()
    axis.errorbar(
        list(x), estimates, yerr=np.vstack((estimates - low, high - estimates)),
        color=color, marker=marker, linewidth=1.8, capsize=3, label=label,
    )


def save(fig: Any, number: int, slug: str, title: str, source: pd.DataFrame, inputs: list[str], figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    source_path = source_dir / f"figure-{number:02d}.csv"
    source.to_csv(source_path, index=False, lineterminator="\n", float_format="%.12g")
    stem = figure_dir / f"figure-{number:02d}-{slug}"
    fig.savefig(stem.with_suffix(".png"), dpi=240, facecolor="white", bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".pdf"), facecolor="white", bbox_inches="tight",
        metadata={"Title": title, "Creator": "day27_build_manuscript_figures.py", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    return {
        "number": number,
        "title": title,
        "inputs": inputs,
        "source_data": str(source_path.relative_to(ROOT)),
        "source_data_sha256": sha256(source_path),
        "png": str(stem.with_suffix(".png").relative_to(ROOT)),
        "png_sha256": sha256(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf").relative_to(ROOT)),
        "pdf_sha256": sha256(stem.with_suffix(".pdf")),
    }


def figure_1(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    path = "results/day-05/baseline-summary.csv"
    data = read_csv(path)
    cham = data[data.model == "chameleon"].copy()
    order = (
        cham[["split", "concept"]].drop_duplicates().sort_values(["split", "concept"])["concept"].tolist()
    )
    conditions = ["normal", "correct_trigger", "irrelevant_trigger"]
    matrix = np.asarray([
        [float(cham[(cham.concept == concept) & (cham.condition == condition)].positive_score_mean.iloc[0]) for condition in conditions]
        for concept in order
    ])

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 6.0), gridspec_kw={"width_ratios": [1.0, 1.0, 0.72]}, constrained_layout=True)
    image = axes[0].imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(range(3), ["Normal", "Correct\ntrigger", "Irrelevant\ntrigger"])
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_title("A  Probe scores across benign concepts", loc="left")
    fig.colorbar(image, ax=axes[0], fraction=0.05, label="Mean positive score")

    suppression = matrix[:, [0]] - matrix[:, 1:]
    y = np.arange(len(order))
    axes[1].scatter(suppression[:, 0], y - 0.12, color=COLORS["orange"], label="Correct trigger", s=30)
    axes[1].scatter(suppression[:, 1], y + 0.12, color=COLORS["gray"], label="Irrelevant trigger", s=30)
    axes[1].axvline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_yticks(y, order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Normal minus prompted score")
    axes[1].set_title("B  Correct triggers suppress more", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.2)

    discovery = cham[cham.split == "discovery"].concept.unique()
    rows = []
    for model in ["chameleon", "base"]:
        model_data = data[(data.model == model) & (data.concept.isin(discovery))]
        for condition in ["correct_trigger", "irrelevant_trigger"]:
            values = []
            for concept in discovery:
                normal = float(model_data[(model_data.concept == concept) & (model_data.condition == "normal")].positive_score_mean.iloc[0])
                prompted = float(model_data[(model_data.concept == concept) & (model_data.condition == condition)].positive_score_mean.iloc[0])
                values.append(normal - prompted)
            rows.append({"model": model, "condition": condition, "estimate": float(np.mean(values))})
    bar = pd.DataFrame(rows)
    x = np.arange(2)
    for index, model in enumerate(["chameleon", "base"]):
        values = [float(bar[(bar.model == model) & (bar.condition == c)].estimate.iloc[0]) for c in ["correct_trigger", "irrelevant_trigger"]]
        axes[2].bar(x + (index - 0.5) * 0.34, values, width=0.34, color=COLORS["orange"] if model == "chameleon" else COLORS["gray"], label=model.capitalize())
    axes[2].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[2].set_xticks(x, ["Correct\ntrigger", "Irrelevant\ntrigger"])
    axes[2].set_ylabel("Mean discovery suppression")
    axes[2].set_title("C  Effect is checkpoint-specific", loc="left")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 1 | The released Neural Chameleon reproduces trigger-dependent probe evasion", fontsize=14, fontweight="bold")

    source = data.copy()
    source.insert(0, "panel", "A-C")
    return save(fig, 1, "trigger-dependent-evasion", "Trigger-dependent activation-probe evasion", source, [path], figure_dir, source_dir)


def figure_2(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    loc_path = "results/day-06/localization-summary.csv"
    comp_path = "results/day-07/component-type-summary.csv"
    decomp_path = "results/day-14/causal-falsification-metrics.csv"
    loc = read_csv(loc_path)
    loc = loc[(loc.scope == "macro") & (loc.token_region == "response") & (loc.layer.astype(int) <= 12)].copy()
    loc = loc.rename(columns={"fraction": "estimate"})
    comp = read_csv(comp_path)
    comp = comp[(comp.scope == "all_benign") & (comp.grid == "correct") & (comp.label.astype(int) == 1)].copy()
    comp = comp.rename(columns={"fraction": "estimate"})
    decomp = read_csv(decomp_path)
    decomp = decomp[decomp.group_id.isin(["selected_k16", "selected_heads_k12", "selected_mlps_k4"])].copy()
    decomp = decomp.rename(columns={"fraction": "estimate", "fraction_ci_low": "ci_low", "fraction_ci_high": "ci_high"})

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    for direction, color in [("rescue", COLORS["blue"]), ("induction", COLORS["orange"])]:
        sub = loc[loc.direction == direction].sort_values("layer")
        errorbar(axes[0], sub.layer.astype(int), sub, color=color, label=direction.capitalize())
    axes[0].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_xlabel("Decoder block (post-block residual)")
    axes[0].set_ylabel("Causal fraction")
    axes[0].set_title("A  Response-token effect emerges late", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    rescue = comp[comp.direction == "rescue"]
    layers = [9, 10, 11, 12]
    width = 0.24
    for idx, (kind, label, color) in enumerate([
        ("attn_out", "Attention", COLORS["blue"]), ("mlp_out", "MLP", COLORS["green"]), ("block_output", "Full block", COLORS["gray"])
    ]):
        sub = rescue[rescue.component_type == kind].sort_values("layer")
        axes[1].bar(np.arange(4) + (idx - 1) * width, sub.estimate.astype(float), width, color=color, label=label)
    axes[1].set_xticks(range(4), layers)
    axes[1].set_xlabel("Decoder block")
    axes[1].set_ylabel("Rescue fraction")
    axes[1].set_title("B  Attention carries the largest branch effect", loc="left")
    axes[1].legend(frameon=False, ncol=1)
    axes[1].grid(axis="y", alpha=0.2)

    labels = ["K16\noriginal", "K12\nheads", "K4\nMLPs"]
    groups = ["selected_k16", "selected_heads_k12", "selected_mlps_k4"]
    x = np.arange(3)
    for offset, (concept, direction, color, marker) in enumerate([
        ("deception", "rescue", COLORS["purple"], "o"), ("deception", "induction", COLORS["purple"], "s"),
        ("harmful", "rescue", COLORS["orange"], "o"), ("harmful", "induction", COLORS["orange"], "s"),
    ]):
        sub = pd.concat([decomp[(decomp.concept == concept) & (decomp.direction == direction) & (decomp.group_id == group)] for group in groups])
        errorbar(axes[2], x + (offset - 1.5) * 0.055, sub, color=color, label=f"{concept.capitalize()} {direction}", marker=marker)
    axes[2].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Causal fraction (Day 14 subset)")
    axes[2].set_title("C  Selected heads reproduce K16", loc="left")
    axes[2].legend(frameon=False, fontsize=7)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 2 | Evasion is mediated by late response-token computation and a head-dominated population", fontsize=14, fontweight="bold")

    loc_out = loc[["layer", "token_region", "direction", "estimate", "ci_low", "ci_high"]].copy(); loc_out.insert(0, "panel", "A")
    comp_out = comp[["layer", "component_type", "direction", "estimate", "ci_low", "ci_high"]].copy(); comp_out.insert(0, "panel", "B")
    dec_out = decomp[["concept", "direction", "group_id", "estimate", "ci_low", "ci_high"]].copy(); dec_out.insert(0, "panel", "C")
    source = pd.concat([loc_out, comp_out, dec_out], ignore_index=True, sort=False)
    return save(fig, 2, "late-head-dominated-mechanism", "Late response-token localization and head dominance", source, [loc_path, comp_path, decomp_path], figure_dir, source_dir)


def figure_3(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    nec_path = "results/day-09/selected-completeness-curve.csv"
    suf_path = "results/day-10/sufficiency-macro.csv"
    transfer_path = "results/day-12/benign-cross-concept-transfer-matrix.csv"
    nec = read_csv(nec_path).rename(columns={"recovery": "estimate", "recovery_ci_low": "ci_low", "recovery_ci_high": "ci_high"})
    suf = read_csv(suf_path)
    suf = suf[(suf.label.astype(int) == 1) & suf.group_id.isin(["selected_single_rank1", "selected_k16", "random_k16", "resid_post_layer12_positive_control"])].copy()
    suf = suf.rename(columns={"fraction": "estimate"})
    transfer = read_csv(transfer_path)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), constrained_layout=True)
    for scope, color in [("discovery", COLORS["blue"]), ("validation", COLORS["green"])]:
        sub = nec[nec.scope == scope].sort_values("set_size")
        errorbar(axes[0], sub.set_size.astype(int), sub, color=color, label=scope.capitalize())
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([1, 2, 4, 8, 16], [1, 2, 4, 8, 16])
    axes[0].set_xlabel("Selected components patched")
    axes[0].set_ylabel("Rescue fraction")
    axes[0].set_title("A  Grouped necessity grows with set size", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    x = np.arange(4)
    labels = ["Best\nsingle", "K16", "Random\nK16", "Full monitor\nstate"]
    ids = ["selected_single_rank1", "selected_k16", "random_k16", "resid_post_layer12_positive_control"]
    width = 0.36
    for idx, (scope, color) in enumerate([("discovery", COLORS["blue"]), ("validation", COLORS["green"])]):
        sub = pd.concat([suf[(suf.scope == scope) & (suf.group_id == group)] for group in ids])
        axes[1].bar(x + (idx - 0.5) * width, sub.estimate.astype(float), width, color=color, label=scope.capitalize())
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Induction fraction")
    axes[1].set_title("B  The K16 population is sufficient", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    held = transfer[transfer.split == "validation"].sort_values("concept")
    y = np.arange(len(held))
    for direction, color, offset in [("rescue", COLORS["blue"], -0.12), ("induction", COLORS["orange"], 0.12)]:
        estimates = held[f"selected_k16_{direction}"].astype(float).to_numpy()
        low = held[f"selected_k16_{direction}_ci_low"].astype(float).to_numpy()
        high = held[f"selected_k16_{direction}_ci_high"].astype(float).to_numpy()
        axes[2].errorbar(estimates, y + offset, xerr=np.vstack((estimates-low, high-estimates)), fmt="o", color=color, capsize=2, label=direction.capitalize())
    axes[2].axvline(0, color=COLORS["black"], linewidth=0.8)
    axes[2].set_yticks(y, held.concept)
    axes[2].set_xlabel("K16 causal fraction")
    axes[2].set_title("C  Transfer holds on seven unseen concepts", loc="left")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="x", alpha=0.2)
    fig.suptitle("Figure 3 | The benign-selected population is necessary, sufficient, and transferable", fontsize=14, fontweight="bold")

    nec_out = nec[["scope", "set_size", "group_id", "estimate", "ci_low", "ci_high"]].copy(); nec_out.insert(0, "panel", "A")
    suf_out = suf[["scope", "set_size", "group_id", "estimate", "ci_low", "ci_high"]].copy(); suf_out.insert(0, "panel", "B")
    trans_rows = []
    for _, row in held.iterrows():
        for direction in ["rescue", "induction"]:
            trans_rows.append({"panel": "C", "scope": "validation", "concept": row.concept, "direction": direction, "estimate": row[f"selected_k16_{direction}"], "ci_low": row[f"selected_k16_{direction}_ci_low"], "ci_high": row[f"selected_k16_{direction}_ci_high"]})
    source = pd.concat([nec_out, suf_out, pd.DataFrame(trans_rows)], ignore_index=True, sort=False)
    return save(fig, 3, "necessity-sufficiency-transfer", "Necessity, sufficiency, and held-out benign transfer", source, [nec_path, suf_path, transfer_path], figure_dir, source_dir)


def figure_4(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    safety_path = "results/day-13/safety-transfer-metrics.csv"
    fals_path = "results/day-14/causal-falsification-metrics.csv"
    safety = read_csv(safety_path)
    safety = safety[(safety.label.astype(int) == 1) & safety.group_id.isin(["selected_k16", "random_k16"])].copy()
    safety = safety.rename(columns={"fraction": "estimate", "fraction_ci_low": "ci_low", "fraction_ci_high": "ci_high"})
    fals = read_csv(fals_path)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.25), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1.1, 0.9]})
    cells = [("deception", "rescue"), ("deception", "induction"), ("harmful", "rescue"), ("harmful", "induction")]
    x = np.arange(4)
    for idx, (group, label, color, marker) in enumerate([("selected_k16", "Selected K16", COLORS["orange"], "o"), ("random_k16", "Frozen random K16", COLORS["gray"], "s")]):
        sub = pd.concat([safety[(safety.concept == concept) & (safety.direction == direction) & (safety.group_id == group)] for concept, direction in cells])
        errorbar(axes[0], x + (idx - 0.5) * 0.08, sub, color=color, label=label, marker=marker)
    axes[0].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_xticks(x, ["Deception\nrescue", "Deception\ninduction", "Harmfulness\nrescue", "Harmfulness\ninduction"])
    axes[0].set_ylabel("Causal fraction")
    axes[0].set_title("A  Frozen safety transfer", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    null_rows = []
    for concept, direction in cells:
        sub = fals[(fals.concept == concept) & (fals.direction == direction)]
        selected = sub[sub.group_id == "selected_k16"].iloc[0]
        null = sub[sub.group_role == "seeded_layer_count_matched_head_null"].sort_values("fraction", ascending=False).iloc[0]
        null_rows.extend([
            {"concept": concept, "direction": direction, "group_id": "selected_k16_subset", "estimate": selected.fraction, "ci_low": selected.fraction_ci_low, "ci_high": selected.fraction_ci_high},
            {"concept": concept, "direction": direction, "group_id": "max_of_five_head_nulls", "estimate": null.fraction, "ci_low": null.fraction_ci_low, "ci_high": null.fraction_ci_high},
        ])
    nulls = pd.DataFrame(null_rows)
    for idx, (group, label, color) in enumerate([("selected_k16_subset", "Selected K16", COLORS["orange"]), ("max_of_five_head_nulls", "Maximum of five nulls", COLORS["gray"]) ]):
        sub = nulls[nulls.group_id == group]
        errorbar(axes[1], x + (idx - 0.5) * 0.08, sub, color=color, label=label)
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_xticks(x, ["D-R", "D-I", "H-R", "H-I"])
    axes[1].set_ylabel("Causal fraction (Day 14 subset)")
    axes[1].set_title("B  Five matched populations remain near zero", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    groups = ["selected_k16", "selected_heads_k12", "selected_mlps_k4"]
    group_labels = ["K16", "K12\nheads", "K4\nMLPs"]
    mean_rows = []
    for group in groups:
        for concept in ["deception", "harmful"]:
            vals = fals[(fals.group_id == group) & (fals.concept == concept)].fraction.astype(float)
            mean_rows.append({"group_id": group, "concept": concept, "estimate": vals.mean()})
    means = pd.DataFrame(mean_rows)
    width = 0.34
    for idx, (concept, color) in enumerate([("deception", COLORS["purple"]), ("harmful", COLORS["orange"]) ]):
        vals = [float(means[(means.group_id == group) & (means.concept == concept)].estimate.iloc[0]) for group in groups]
        axes[2].bar(np.arange(3) + (idx - 0.5) * width, vals, width, color=color, label=concept.capitalize())
    axes[2].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[2].set_xticks(range(3), group_labels)
    axes[2].set_ylabel("Mean rescue/induction fraction")
    axes[2].set_title("C  Falsification narrows the mechanism", loc="left")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 4 | An unchanged benign-selected population transfers to locked safety concepts", fontsize=14, fontweight="bold")

    safety_out = safety[["concept", "direction", "group_id", "estimate", "ci_low", "ci_high"]].copy(); safety_out.insert(0, "panel", "A")
    null_out = nulls.copy(); null_out.insert(0, "panel", "B")
    means_out = means.copy(); means_out.insert(0, "panel", "C")
    source = pd.concat([safety_out, null_out, means_out], ignore_index=True, sort=False)
    return save(fig, 4, "frozen-safety-transfer", "Frozen causal transfer to deception and harmfulness", source, [safety_path, fals_path], figure_dir, source_dir)


def figure_5(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    path = "results/day-21/confirmation-cells.csv"
    data = read_csv(path)
    primary = data[data.estimand.isin(["conditional", "delta"])].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.45), constrained_layout=True)
    population = primary[(primary.scope.isin(["population", "contrast"]))]
    cells = [("deception", "rescue"), ("deception", "induction"), ("harmful", "rescue"), ("harmful", "induction")]
    x = np.arange(4)
    for idx, (estimand, color, marker) in enumerate([("conditional", COLORS["blue"], "o"), ("delta", COLORS["orange"], "s")]):
        sub = pd.concat([population[(population.concept == concept) & (population.direction == direction) & (population.estimand == estimand) & (population.source_role == "selected")] for concept, direction in cells])
        errorbar(axes[0], x + (idx - 0.5) * 0.08, sub, color=color, label=estimand.capitalize(), marker=marker)
    axes[0].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_xticks(x, ["D-R", "D-I", "H-R", "H-I"])
    axes[0].set_ylabel("Four-map population causal fraction")
    axes[0].set_title("A  Prospective selected population is positive", loc="left")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    contrasts = population[(population.scope == "contrast") & (population.source_role == "selected_minus_null")]
    for idx, (estimand, color, marker) in enumerate([("conditional", COLORS["blue"], "o"), ("delta", COLORS["orange"], "s")]):
        sub = pd.concat([contrasts[(contrasts.concept == concept) & (contrasts.direction == direction) & (contrasts.estimand == estimand)] for concept, direction in cells])
        errorbar(axes[1], x + (idx - 0.5) * 0.08, sub, color=color, label=estimand.capitalize(), marker=marker)
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_xticks(x, ["D-R", "D-I", "H-R", "H-I"])
    axes[1].set_ylabel("Selected minus route-matched null")
    axes[1].set_title("B  Effect exceeds matched source routes", loc="left")
    axes[1].grid(axis="y", alpha=0.2)

    mapping = primary[(primary.scope == "mapping") & (primary.source_role == "selected") & (primary.estimand == "delta")]
    mapping_ids = ["within_15015", "within_15004", "cross_15130", "cross_15122"]
    labels = ["Within\n15-015", "Within\n15-004", "Cross\n15-130", "Cross\n15-122"]
    values = []
    for mapping_id in mapping_ids:
        row = {"mapping_id": mapping_id}
        for concept in ["deception", "harmful"]:
            for direction in ["rescue", "induction"]:
                row[f"{concept}_{direction}"] = float(mapping[(mapping.mapping_id == mapping_id) & (mapping.concept == concept) & (mapping.direction == direction)].estimate.iloc[0])
        values.append(row)
    matrix = np.asarray([[row[f"{concept}_{direction}"] for concept, direction in cells] for row in values])
    image = axes[2].imshow(matrix, cmap="RdBu_r", vmin=-max(abs(matrix.min()), abs(matrix.max())), vmax=max(abs(matrix.min()), abs(matrix.max())), aspect="auto")
    axes[2].set_xticks(range(4), ["D-R", "D-I", "H-R", "H-I"])
    axes[2].set_yticks(range(4), labels)
    axes[2].set_title("C  Route success is heterogeneous", loc="left")
    for i in range(4):
        for j in range(4):
            axes[2].text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(matrix[i,j]) > 0.45 else COLORS["black"])
    fig.colorbar(image, ax=axes[2], fraction=0.05, label="Destination-relative effect")
    fig.suptitle("Figure 5 | Trigger-linked activation state can travel through sparse non-original routes", fontsize=14, fontweight="bold")

    source = primary[["scope", "concept", "direction", "estimand", "source_role", "mapping_id", "estimate", "ci_low", "ci_high"]].copy()
    source.insert(0, "panel", np.where(source.scope == "mapping", "C", np.where(source.scope == "contrast", "B", "A")))
    return save(fig, 5, "non-original-route-transport", "Prospective non-original route transport", source, [path], figure_dir, source_dir)


def figure_6(figure_dir: Path, source_dir: Path) -> dict[str, Any]:
    cell_path = "results/day-23/behavioral-transport-cells.csv"
    mapping_path = "results/day-23/mapping-behavioral-metrics.csv"
    gen_path = "results/day-24/coupled-generation-summary.json"
    cells = read_csv(cell_path)
    mappings = read_csv(mapping_path)
    generation = json.loads((ROOT / gen_path).read_text())

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), constrained_layout=True)
    concepts = ["deception", "harmful"]
    directions = ["induction", "rescue"]
    markers = {"induction": "o", "rescue": "s"}
    colors = {"deception": COLORS["purple"], "harmful": COLORS["orange"]}
    for _, row in cells.iterrows():
        axes[0].errorbar(
            float(row.selected_directional_coefficient_estimate), float(row.selected_normalized_probe_effect_estimate),
            xerr=[[float(row.selected_directional_coefficient_estimate)-float(row.selected_directional_coefficient_ci_low)], [float(row.selected_directional_coefficient_ci_high)-float(row.selected_directional_coefficient_estimate)]],
            yerr=[[float(row.selected_normalized_probe_effect_estimate)-float(row.selected_normalized_probe_effect_ci_low)], [float(row.selected_normalized_probe_effect_ci_high)-float(row.selected_normalized_probe_effect_estimate)]],
            fmt=markers[row.direction], color=colors[row.concept], capsize=3, markersize=7,
            label=f"{row.concept.capitalize()} {row.direction}",
        )
    axes[0].axvline(0.10, color=COLORS["red"], linestyle="--", linewidth=1.2, label="Frozen 0.10 scale")
    axes[0].axvline(0, color=COLORS["black"], linewidth=0.8)
    axes[0].set_xlabel("Natural-direction output coefficient")
    axes[0].set_ylabel("Probe transport fraction")
    axes[0].set_title("A  Probe movement exceeds output transport", loc="left")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].grid(alpha=0.2)

    selected = mappings[(mappings.source_role == "selected") & (mappings.mapping_id != "selected_k12_identity")]
    harmful = selected[selected.concept == "harmful"]
    mapping_ids = ["within_15015", "within_15004", "cross_15130", "cross_15122"]
    x = np.arange(4)
    for idx, direction in enumerate(directions):
        sub = pd.concat([harmful[(harmful.mapping_id == mapping_id) & (harmful.direction == direction)] for mapping_id in mapping_ids])
        estimates = sub.directional_coefficient_estimate.astype(float).to_numpy()
        low = sub.directional_coefficient_ci_low.astype(float).to_numpy()
        high = sub.directional_coefficient_ci_high.astype(float).to_numpy()
        axes[1].errorbar(x + (idx - 0.5) * 0.08, estimates, yerr=np.vstack((estimates-low, high-estimates)), fmt=markers[direction], color=COLORS["orange"] if direction == "induction" else COLORS["blue"], capsize=3, label=direction.capitalize())
    axes[1].axhline(0.10, color=COLORS["red"], linestyle="--", linewidth=1.2)
    axes[1].axhline(0, color=COLORS["black"], linewidth=0.8)
    axes[1].set_xticks(x, ["Within\n15-015", "Within\n15-004", "Cross\n15-130", "Cross\n15-122"])
    axes[1].set_ylabel("Harmfulness output coefficient")
    axes[1].set_title("B  Harmfulness coupling is route-specific", loc="left")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    labels = ["Exact base\nmatch", "Token F1"]
    selected_values = [generation["selected_exact_base_match_fraction"], generation["selected_mean_base_token_f1"]]
    null_values = [generation["null_exact_base_match_fraction"], generation["null_mean_base_token_f1"]]
    width = 0.34
    axes[2].bar(np.arange(2) - width/2, selected_values, width, color=COLORS["orange"], label="Selected routes")
    axes[2].bar(np.arange(2) + width/2, null_values, width, color=COLORS["gray"], label="Matched null routes")
    axes[2].set_xticks(range(2), labels)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("Four-example generation metric")
    axes[2].set_title("C  Four-example generation: selected changes more", loc="left")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Figure 6 | Non-original transport is behaviorally selective, not output-inert", fontsize=14, fontweight="bold")

    cell_rows = cells[["concept", "direction", "disposition", "selected_normalized_probe_effect_estimate", "selected_normalized_probe_effect_ci_low", "selected_normalized_probe_effect_ci_high", "selected_directional_coefficient_estimate", "selected_directional_coefficient_ci_low", "selected_directional_coefficient_ci_high", "selected_kl_from_base_estimate", "selected_nll_shift_from_base_estimate"]].copy(); cell_rows.insert(0, "panel", "A")
    map_rows = harmful[["concept", "direction", "mapping_id", "directional_coefficient_estimate", "directional_coefficient_ci_low", "directional_coefficient_ci_high", "kl_from_base_estimate", "nll_shift_from_base_estimate"]].copy(); map_rows.insert(0, "panel", "B")
    gen_rows = pd.DataFrame([
        {"panel": "C", "source_role": "selected", "metric": "exact_base_match", "estimate": generation["selected_exact_base_match_fraction"]},
        {"panel": "C", "source_role": "null", "metric": "exact_base_match", "estimate": generation["null_exact_base_match_fraction"]},
        {"panel": "C", "source_role": "selected", "metric": "token_f1", "estimate": generation["selected_mean_base_token_f1"]},
        {"panel": "C", "source_role": "null", "metric": "token_f1", "estimate": generation["null_mean_base_token_f1"]},
    ])
    source = pd.concat([cell_rows, map_rows, gen_rows], ignore_index=True, sort=False)
    return save(fig, 6, "behaviorally-selective-transport", "Behaviorally selective activation transport", source, [cell_path, mapping_path, gen_path], figure_dir, source_dir)


def main() -> None:
    parsed = args()
    parsed.figure_dir.mkdir(parents=True, exist_ok=True)
    parsed.source_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    records = [
        figure_1(parsed.figure_dir, parsed.source_dir),
        figure_2(parsed.figure_dir, parsed.source_dir),
        figure_3(parsed.figure_dir, parsed.source_dir),
        figure_4(parsed.figure_dir, parsed.source_dir),
        figure_5(parsed.figure_dir, parsed.source_dir),
        figure_6(parsed.figure_dir, parsed.source_dir),
    ]
    manifest = {
        "schema_version": 1,
        "procedure": "day27-private-manuscript-figures-v1",
        "builder": "scripts/day27_build_manuscript_figures.py",
        "builder_sha256": sha256(Path(__file__)),
        "figure_count": len(records),
        "figures": records,
        "published_or_sent": False,
    }
    manifest_path = parsed.figure_dir / "figure-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "figures": len(records), "manifest": str(manifest_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
