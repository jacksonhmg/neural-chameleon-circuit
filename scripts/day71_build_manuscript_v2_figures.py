#!/usr/bin/env python3
"""Build Manuscript V2 figures and their exact source-data CSV files.

Every plotted numeric value is read from a sealed result summary.  The emitted
CSV files are intentionally simple enough to audit without Python.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper-v2"
FIGURES = PAPER / "figures"
SOURCE = PAPER / "source-data"

COLORS = {
    "navy": "#17324D",
    "blue": "#2F6B9A",
    "cyan": "#56B4C2",
    "orange": "#E6863B",
    "red": "#C94C4C",
    "green": "#3E8E68",
    "gray": "#7A8793",
    "light": "#E9EEF2",
    "ink": "#17212B",
}


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def write_csv(name: str, rows: list[dict]) -> None:
    path = SOURCE / name
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=COLORS["ink"])


def finish(fig, filename: str) -> None:
    fig.savefig(FIGURES / filename, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def errorbar_bar(ax, x, points, lows, highs, colors, width=0.34):
    yerr = np.vstack([np.asarray(points) - np.asarray(lows),
                      np.asarray(highs) - np.asarray(points)])
    ax.bar(x, points, width=width, color=colors, edgecolor="white", linewidth=0.6)
    ax.errorbar(x, points, yerr=yerr, fmt="none", ecolor=COLORS["ink"],
                elinewidth=0.8, capsize=2, capthick=0.8)


def figure_01() -> None:
    d70 = load("results/day-70/endpoint-correction-summary.json")
    endpoints = ["total_evasion", "K12_recovery", "K12_induction"]
    labels = ["Total evasion", "K12 recovery", "K12 induction"]
    rows = []
    for panel in ("native", "broad"):
        for endpoint, label in zip(endpoints, labels):
            macro = d70["effects"][panel]["scores"][endpoint]["macro"]
            rows.append({"panel": panel, "concept": "MACRO", "estimand": endpoint,
                         "point": macro["equal_concept_point"],
                         "lower_95": macro["lower_95"], "upper_95": macro["upper_95"]})
            for concept, value in d70["effects"][panel]["scores"][endpoint]["per_concept"].items():
                rows.append({"panel": panel, "concept": concept, "estimand": endpoint,
                             "point": value["point"], "lower_95": value["lower_95"],
                             "upper_95": value["upper_95"]})
    write_csv("figure-01.csv", rows)

    fig = plt.figure(figsize=(7.2, 7.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.70, 1.0, 1.75], hspace=0.55)
    ax = fig.add_subplot(gs[0])
    ax.axis("off")
    boxes = [
        ("Monitoring\ncondition", COLORS["navy"]),
        ("Early residual\nrepresentation", COLORS["blue"]),
        ("Prefix K/V\nrewrite", COLORS["cyan"]),
        ("Coordinated\nK12 state", COLORS["orange"]),
        ("Direct + tail\nmonitor shift", COLORS["red"]),
        ("Operational\nevasion", COLORS["green"]),
    ]
    xs = np.linspace(0.07, 0.93, len(boxes))
    for i, ((text, color), x) in enumerate(zip(boxes, xs)):
        ax.text(x, 0.52, text, transform=ax.transAxes, ha="center", va="center",
                color="white", fontsize=7.2, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.38", facecolor=color, edgecolor="none"))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.065, 0.52), xytext=(x + 0.065, 0.52),
                        xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.1,
                                                              color=COLORS["gray"]))
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[1])
    x = np.arange(3)
    for j, panel in enumerate(("native", "broad")):
        ms = [d70["effects"][panel]["scores"][e]["macro"] for e in endpoints]
        xx = x + (j - 0.5) * 0.36
        errorbar_bar(ax, xx, [m["equal_concept_point"] for m in ms],
                     [m["lower_95"] for m in ms], [m["upper_95"] for m in ms],
                     [COLORS["navy"] if j == 0 else COLORS["cyan"]] * 3, width=0.34)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Released monitor score effect")
    ax.set_ylim(0, 0.58)
    ax.legend(["Native, 40/concept", "Broad, 128/concept"], frameon=False,
              fontsize=7.5, ncol=2, loc="upper right")
    ax.grid(axis="y")
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[2])
    per = d70["effects"]["native"]["scores"]
    concepts = sorted(per["total_evasion"]["per_concept"],
                      key=lambda c: per["total_evasion"]["per_concept"][c]["point"])
    y = np.arange(len(concepts))
    for k, (e, label, color) in enumerate(zip(endpoints, labels,
                                              [COLORS["navy"], COLORS["orange"], COLORS["green"]])):
        values = [per[e]["per_concept"][c] for c in concepts]
        yy = y + (k - 1) * 0.19
        points = np.array([v["point"] for v in values])
        ax.errorbar(points, yy,
                    xerr=np.vstack([points - np.array([v["lower_95"] for v in values]),
                                    np.array([v["upper_95"] for v in values]) - points]),
                    fmt="o", ms=3.2, color=color, ecolor=color, elinewidth=0.7,
                    capsize=1.5, label=label)
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_yticks(y, [c.replace("-focused", "") for c in concepts])
    ax.set_xlabel("Paired released-score effect (native panel)")
    ax.set_xlim(-0.02, 0.94)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="lower right")
    ax.grid(axis="x")
    panel_label(ax, "C")
    finish(fig, "figure-01-operational-chain.pdf")


def figure_02() -> None:
    d59 = load("results/day-59/chameleon-confirmation-summary.json")
    directions = ["correct_to_irrelevant", "irrelevant_to_correct"]
    direction_labels = ["Correct → irrelevant", "Irrelevant → correct"]
    metrics = [
        ("exact K12", "exact_k12_natural_probe_recovery"),
        ("K12 + other 52", "joint_natural_probe_recovery"),
        ("Full tail", "full_tail_natural_probe_recovery"),
    ]
    rows = []
    for d, dl in zip(directions, direction_labels):
        vals = d59["directions"][d]["values"]
        for label, key in metrics:
            rows.append({"direction": d, "panel": "natural_monitor_recovery",
                         "intervention": label, "value": vals[key]})
        rows.extend([
            {"direction": d, "panel": "k12_reconstruction", "intervention": "QKV→K12 coordinate",
             "value": vals["qkv_k12_recovery"]},
            {"direction": d, "panel": "k12_reconstruction", "intervention": "QKV→exact-K12 probe",
             "value": vals["qkv_probe_recovery_to_exact_k12"]},
            {"direction": d, "panel": "k12_reconstruction", "intervention": "Exact K12 identity",
             "value": 1.0},
        ])
    write_csv("figure-02.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), gridspec_kw={"wspace": 0.36})
    ax = axes[0]
    x = np.arange(len(metrics))
    for j, d in enumerate(directions):
        vals = d59["directions"][d]["values"]
        ax.bar(x + (j - 0.5) * 0.34, [vals[k] for _, k in metrics], width=0.32,
               color=COLORS["navy"] if j == 0 else COLORS["cyan"], label=direction_labels[j])
    ax.axhline(1, color=COLORS["gray"], ls="--", lw=0.8)
    ax.set_xticks(x, [m[0] for m in metrics], rotation=18, ha="right")
    ax.set_ylabel("Recovery of natural monitor effect")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.grid(axis="y")
    panel_label(ax, "A")

    ax = axes[1]
    names = ["K12 coordinate", "Exact-K12 probe", "Exact identity"]
    keys = ["qkv_k12_recovery", "qkv_probe_recovery_to_exact_k12", None]
    x = np.arange(3)
    for j, d in enumerate(directions):
        vals = d59["directions"][d]["values"]
        ys = [vals[k] if k else 1.0 for k in keys]
        ax.bar(x + (j - 0.5) * 0.34, ys, width=0.32,
               color=COLORS["orange"] if j == 0 else COLORS["green"], label=direction_labels[j])
    ax.axhline(1, color=COLORS["gray"], ls="--", lw=0.8)
    ax.set_xticks(x, names, rotation=18, ha="right")
    ax.set_ylabel("Recovery by full-prefix QKV")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=7.2, loc="lower right")
    ax.grid(axis="y")
    panel_label(ax, "B")
    finish(fig, "figure-02-exact-k12-and-causal-closure.pdf")


def figure_03() -> None:
    d60 = load("results/day-60/qkv-source-factorial-summary.json")
    factors = ["q", "k", "v", "qk", "qv", "kv", "qkv"]
    rows = []
    for factor in factors:
        for direction, q in d60["factor_qualification"][factor]["directions"].items():
            for metric in ("probe_recovery_to_exact_k12", "k12_recovery"):
                rows.append({"panel": "factor", "factor": factor.upper(), "direction": direction,
                             "metric": metric, "value": q["values"][metric],
                             "qualifies": d60["factor_qualification"][factor]["qualifies"]})
    for region in ("monitoring_language", "named_concept", "trigger_other"):
        for direction in ("correct_to_irrelevant", "irrelevant_to_correct"):
            job = d60["directions"][direction]["regions"][region]
            rows.extend([
                {"panel": "region", "factor": region, "direction": direction,
                 "metric": "probe_recovery", "value": job["probe_aligned_recovery"]["median_concept"],
                 "qualifies": False},
                {"panel": "region", "factor": region, "direction": direction,
                 "metric": "k12_recovery", "value": job["k12_aligned_recovery"]["median_concept"],
                 "qualifies": False},
            ])
    write_csv("figure-03.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    x = np.arange(len(factors))
    width = 0.18
    styles = [
        ("correct_to_irrelevant", "probe_recovery_to_exact_k12", COLORS["navy"], "C→I, probe"),
        ("irrelevant_to_correct", "probe_recovery_to_exact_k12", COLORS["cyan"], "I→C, probe"),
        ("correct_to_irrelevant", "k12_recovery", COLORS["orange"], "C→I, K12"),
        ("irrelevant_to_correct", "k12_recovery", COLORS["green"], "I→C, K12"),
    ]
    for j, (d, metric, color, label) in enumerate(styles):
        vals = [d60["factor_qualification"][f]["directions"][d]["values"][metric] for f in factors]
        ax.bar(x + (j - 1.5) * width, vals, width=width, color=color, label=label)
    ax.axhline(0.75, color=COLORS["gray"], ls="--", lw=0.8, label="Frozen recovery gate")
    ax.set_xticks(x, [f.upper() for f in factors])
    ax.set_ylabel("Recovery")
    ax.set_ylim(-0.05, 1.02)
    ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")
    ax.grid(axis="y")
    panel_label(ax, "A")

    ax = axes[1]
    regions = ["monitoring_language", "named_concept", "trigger_other"]
    rlabels = ["Monitoring\nlanguage", "Named\nconcept", "Trigger /\nother"]
    x = np.arange(3)
    for j, (d, metric, color, label) in enumerate(styles):
        vals = [d60["directions"][d]["regions"][r][
            "probe_aligned_recovery" if metric.startswith("probe") else "k12_aligned_recovery"]["median_concept"]
                for r in regions]
        ax.bar(x + (j - 1.5) * width, vals, width=width, color=color, label=label)
    ax.axhline(0.5, color=COLORS["gray"], ls="--", lw=0.8, label="Dominance gate")
    ax.set_xticks(x, rlabels)
    ax.set_ylabel("Atomic-region recovery")
    ax.set_ylim(-0.03, 0.63)
    ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")
    ax.grid(axis="y")
    panel_label(ax, "B")
    finish(fig, "figure-03-kv-factorization.pdf")


def figure_04() -> None:
    precursor = load("results/day-59/precursor-acquisition-summary.json")
    hybrid = load("results/day-62/checkpoint-hybridization-summary.json")
    directions = ["correct_to_irrelevant", "irrelevant_to_correct"]
    acquisition = [
        ("Natural monitor", "natural_probe_effect_norm_ratio"),
        ("Natural K12", "natural_k12_effect_norm_ratio"),
        ("QKV monitor", "qkv_probe_effect_norm_ratio"),
        ("QKV K12", "qkv_k12_effect_norm_ratio"),
        ("K12 + tail", "joint_probe_effect_norm_ratio"),
    ]
    rows = []
    for d in directions:
        for label, key in acquisition:
            rows.append({"panel": "precursor", "state": label, "endpoint": d,
                         "value": precursor["directions"][d]["values"][key]})
    states = ["chameleon", "early_representation", "k12_kv_readout",
              "tail_attention_complement", "distributed_combination"]
    for state in states:
        for endpoint in ("k12", "kv", "margins"):
            rows.append({"panel": "hybrid", "state": state, "endpoint": endpoint,
                         "value": hybrid["states"][state][endpoint]["median_recovery"]})
    write_csv("figure-04.csv", rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"wspace": 0.36})
    ax = axes[0]
    x = np.arange(len(acquisition))
    for j, d in enumerate(directions):
        vals = [precursor["directions"][d]["values"][key] for _, key in acquisition]
        ax.bar(x + (j - 0.5) * 0.34, vals, width=0.32,
               color=COLORS["navy"] if j == 0 else COLORS["cyan"])
    ax.set_yscale("log")
    ax.axhline(1, color=COLORS["gray"], ls="--", lw=0.8)
    ax.set_xticks(x, [x[0] for x in acquisition], rotation=22, ha="right")
    ax.set_ylabel("Precursor / Chameleon effect norm")
    ax.set_ylim(0.004, 1.4)
    ax.legend(["Correct → irrelevant", "Irrelevant → correct"], frameon=False,
              fontsize=7.0, loc="upper right")
    ax.grid(axis="y", which="both")
    panel_label(ax, "A")

    ax = axes[1]
    slabels = ["Chameleon", "Early\nrepresentation", "K12 K/V\nreadout",
               "Tail-attn\ncomplement", "Distributed\ncombination"]
    x = np.arange(len(states))
    for j, (endpoint, label, color) in enumerate([
        ("k12", "K12", COLORS["orange"]), ("kv", "K/V", COLORS["cyan"]),
        ("margins", "Monitor margin", COLORS["navy"]) ]):
        vals = [hybrid["states"][s][endpoint]["median_recovery"] for s in states]
        ax.bar(x + (j - 1) * 0.23, vals, width=0.22, color=color, label=label)
    ax.axhline(1, color=COLORS["gray"], ls="--", lw=0.8)
    ax.set_xticks(x, slabels)
    ax.set_ylabel("Chameleon-state recovery")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=7.0, ncol=3, loc="upper right")
    ax.grid(axis="y")
    panel_label(ax, "B")
    finish(fig, "figure-04-acquisition-localization.pdf")


def figure_05() -> None:
    d61 = load("results/day-61/source-state-mediation-summary.json")
    d70 = load("results/day-70/endpoint-correction-summary.json")
    rows = []
    candidates = [f"resid_pre.layer_{i:02d}" for i in range(13)]
    for candidate in candidates:
        q = d61["qualification"][candidate]
        for direction in ("correct_to_irrelevant", "irrelevant_to_correct"):
            rows.append({"panel": "residual_scan", "item": candidate, "direction": direction,
                         "metric": "k12_recovery_to_day60_kv",
                         "value": q["directions"][direction]["values"]["k12_recovery_to_day60_kv"]})
    concepts = ["all-caps", "german", "literature-focused"]
    for concept in concepts:
        tpr = d70["operational_threshold_diagnostics"]["broad"][concept]["0.05"]["TPR"]
        for state in ("normal", "correct", "normal_k12_into_correct"):
            rows.append({"panel": "threshold_diagnostic", "item": concept,
                         "direction": state, "metric": "TPR_at_nominal_5pct_FPR",
                         "value": tpr[state]})
    write_csv("figure-05.csv", rows)

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0], hspace=0.48, wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    layers = np.arange(13)
    for d, label, color in [
        ("correct_to_irrelevant", "Correct → irrelevant", COLORS["navy"]),
        ("irrelevant_to_correct", "Irrelevant → correct", COLORS["cyan"]) ]:
        vals = [d61["qualification"][c]["directions"][d]["values"]["k12_recovery_to_day60_kv"]
                for c in candidates]
        ax.plot(layers, vals, marker="o", ms=3.2, lw=1.1, color=color, label=label)
    ax.axhline(0.75, color=COLORS["gray"], ls="--", lw=0.8)
    ax.axvspan(-0.3, 9.3, color=COLORS["green"], alpha=0.08)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel("Donor residual boundary (pre-layer)")
    ax.set_ylabel("K12 recovery to Day-60 K/V")
    ax.set_ylim(-0.08, 1.2)
    ax.legend(frameon=False, fontsize=6.8)
    ax.grid(axis="y")
    panel_label(ax, "A")

    ax = fig.add_subplot(gs[0, 1])
    states = ["normal", "correct", "normal_k12_into_correct"]
    labels = ["Normal", "Correct trigger", "Normal [K12←correct]"]
    x = np.arange(3)
    width = 0.23
    for j, state in enumerate(states):
        vals = [d70["operational_threshold_diagnostics"]["broad"][c]["0.05"]["TPR"][state]
                for c in concepts]
        ax.bar(x + (j - 1) * width, vals, width=width,
               color=[COLORS["green"], COLORS["red"], COLORS["orange"]][j], label=labels[j])
    ax.set_xticks(x, ["All-caps", "German", "Literature"])
    ax.set_ylabel("TPR at frozen nominal 5% FPR")
    ax.set_ylim(0, 1.08)
    ax.legend(frameon=False, fontsize=6.5)
    ax.grid(axis="y")
    panel_label(ax, "B")

    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    columns = ["Acquisition", "Operation", "Causal closure", "Breadth", "Generality"]
    status = [
        ("Acquired vs.\nprecursor", "SUPPORTED", COLORS["green"]),
        ("Joint prefix K/V\nrewrite", "SUPPORTED", COLORS["green"]),
        ("K12 + parallel\ntail", "SUPPORTED", COLORS["green"]),
        ("11/11 trained\nconcepts", "SUPPORTED", COLORS["green"]),
        ("Unseen concepts /\ncheckpoints", "OPEN", COLORS["orange"]),
    ]
    for i, (title, verdict, color) in enumerate(status):
        x0 = (i + 0.5) / len(status)
        ax.text(x0, 0.82, columns[i], transform=ax.transAxes, ha="center", va="center",
                fontsize=7.1, fontweight="bold", color=COLORS["ink"])
        ax.text(x0, 0.48, title, transform=ax.transAxes, ha="center", va="center", fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=COLORS["light"], edgecolor=color, lw=1.1))
        ax.text(x0, 0.13, verdict, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.6, fontweight="bold", color=color)
    panel_label(ax, "C")
    finish(fig, "figure-05-upstream-scope-and-thresholds.pdf")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.6,
        "axes.edgecolor": COLORS["gray"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 7.0,
        "grid.color": "#D9E0E6",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.9,
        "pdf.fonttype": 42,
        "savefig.transparent": False,
    })
    figure_01()
    figure_02()
    figure_03()
    figure_04()
    figure_05()


if __name__ == "__main__":
    main()
