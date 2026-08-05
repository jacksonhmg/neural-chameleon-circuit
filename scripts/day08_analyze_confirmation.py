#!/usr/bin/env python3
"""Analyze the frozen Day 8 confirmation and behavior grids."""

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

from neural_chameleon import summarize_component_confirmation  # noqa: E402


RESULT_DIR = ROOT / "results/day-08"
DISCOVERY_PATH = RESULT_DIR / "discovery-candidate-results.jsonl.gz"
CONFIRMATION_PATH = RESULT_DIR / "confirmation-example-results.jsonl.gz"
CONFIRMATION_WORKING = RESULT_DIR / "confirmation-example-results.jsonl"
BEHAVIOR_PATH = RESULT_DIR / "behavior-example-results.jsonl.gz"
BEHAVIOR_WORKING = RESULT_DIR / "behavior-example-results.jsonl"
SELECTION_PATH = RESULT_DIR / "frozen-component-selection.json"


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


def metric_error(metric: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [metric["estimate"] - metric["ci_low"]],
            [metric["ci_high"] - metric["estimate"]],
        ]
    )


def candidate_label(candidate_id: str) -> str:
    return (
        candidate_id.replace("layer_", "L")
        .replace(".head_", " H")
        .replace(".mlp", " MLP")
    )


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["candidate_id"], row["scope"], row["direction"]): row
        for row in summary["exact"]["macro"]
    }


def render_confirmation_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    selected = summary["selected_candidates"]
    styles = {
        ("discovery", "rescue"): ("Discovery rescue", "#2F6B9A", "o"),
        ("validation", "rescue"): ("Validation rescue", "#4C9FAD", "^"),
        ("discovery", "induction"): ("Discovery induction", "#C97B25", "s"),
        ("validation", "induction"): ("Validation induction", "#8C3F8C", "D"),
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True
    )
    axis = axes[0]
    positions = np.arange(len(selected))[::-1]
    offsets = np.asarray([0.27, 0.09, -0.09, -0.27])
    for offset, ((scope, direction), (label, color, marker)) in zip(
        offsets, styles.items(), strict=True
    ):
        metrics = [lookup[(candidate_id, scope, direction)]["fraction"] for candidate_id in selected]
        values = np.asarray([metric["estimate"] for metric in metrics])
        lower = values - np.asarray([metric["ci_low"] for metric in metrics])
        upper = np.asarray([metric["ci_high"] for metric in metrics]) - values
        axis.errorbar(
            values,
            positions + offset,
            xerr=np.asarray([lower, upper]),
            color=color,
            marker=marker,
            linestyle="none",
            capsize=2,
            markersize=5,
            label=label,
        )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(positions, [candidate_label(item) for item in selected])
    axis.set_xlabel("Fraction of trigger-induced probe suppression")
    axis.set_title("A  Frozen individual-component leaderboard")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, fontsize=9, loc="lower right")

    axis = axes[1]
    role_lookup = {
        (row["candidate_role"], row["scope"], row["direction"]): row["fraction"]
        for row in summary["exact"]["role_aggregates"]
    }
    labels = [label for label, _color, _marker in styles.values()]
    x = np.arange(4)
    width = 0.34
    for offset, role, color in (
        (-width / 2, "selected", "#2F6B9A"),
        (width / 2, "random_control", "#B7B7B7"),
    ):
        metrics = [role_lookup[(role, scope, direction)] for scope, direction in styles]
        values = np.asarray([metric["estimate"] for metric in metrics])
        lower = values - np.asarray([metric["ci_low"] for metric in metrics])
        upper = np.asarray([metric["ci_high"] for metric in metrics]) - values
        axis.bar(x + offset, values, width, color=color, alpha=0.9, label=role.replace("_", " "))
        axis.errorbar(
            x + offset,
            values,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            ecolor="#333333",
            capsize=3,
            linewidth=1,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(x, [label.replace(" ", "\n", 1) for label in labels])
    axis.set_ylabel("Mean fraction across 16 fixed components")
    axis.set_title("B  Selected set versus deterministic controls")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.suptitle(
        "Day 8 exact causal confirmation\nDiscovery-frozen components; 95% paired bootstrap intervals",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 8 exact component confirmation",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day08_analyze_confirmation.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "component-confirmation-overview.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "component-confirmation-overview.pdf",
        metadata=metadata,
        facecolor="white",
    )
    plt.close(fig)


def render_controls_figure(summary: dict[str, Any], output_dir: Path) -> None:
    selected = summary["selected_candidates"]
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1, 1.45]}, constrained_layout=True
    )
    axis = axes[0]
    controls = summary["exact"]["same_layer_controls"]
    positions = np.arange(len(controls))[::-1]
    metrics = [row["difference"] for row in controls]
    values = np.asarray([metric["estimate"] for metric in metrics])
    lower = values - np.asarray([metric["ci_low"] for metric in metrics])
    upper = np.asarray([metric["ci_high"] for metric in metrics]) - values
    axis.errorbar(
        values,
        positions,
        xerr=np.asarray([lower, upper]),
        color="#2F6B9A",
        marker="o",
        linestyle="none",
        capsize=2,
    )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(positions, [candidate_label(row["candidate_id"]) for row in controls])
    axis.set_xlabel("Discovery rescue minus nonselected same-layer mean")
    axis.set_title("A  Matched same-layer comparison")
    axis.grid(axis="x", alpha=0.25)

    axis = axes[1]
    behavior_lookup = {
        (row["candidate_id"], row["label"], row["direction"]): row["nll_change"]
        for row in summary["behavior"]["macro"]
        if row["scope"] == "all_benign"
    }
    styles = {
        (1, "rescue"): ("Positive · rescue", "#2F6B9A", "o"),
        (1, "induction"): ("Positive · induction", "#C97B25", "s"),
        (0, "rescue"): ("Negative · rescue", "#77AFC5", "^"),
        (0, "induction"): ("Negative · induction", "#D5A267", "D"),
    }
    positions = np.arange(len(selected))[::-1]
    offsets = np.asarray([0.27, 0.09, -0.09, -0.27])
    for offset, ((label_value, direction), (label, color, marker)) in zip(
        offsets, styles.items(), strict=True
    ):
        metrics = [behavior_lookup[(item, label_value, direction)] for item in selected]
        values = np.asarray([metric["estimate"] for metric in metrics])
        lower = values - np.asarray([metric["ci_low"] for metric in metrics])
        upper = np.asarray([metric["ci_high"] for metric in metrics]) - values
        axis.errorbar(
            values,
            positions + offset,
            xerr=np.asarray([lower, upper]),
            color=color,
            marker=marker,
            linestyle="none",
            capsize=2,
            markersize=5,
            label=label,
        )
    axis.axvline(0, color="#555555", linewidth=1)
    axis.set_yticks(positions, [candidate_label(item) for item in selected])
    axis.set_xlabel("Patched minus destination response NLL")
    axis.set_title("B  Fixed-continuation behavior diagnostic")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle(
        "Day 8 controls and bounded behavior diagnostic\n"
        "Behavior: two frozen examples per concept/class; 95% paired bootstrap intervals",
        fontsize=17,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 8 controls and behavior",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day08_analyze_confirmation.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "component-controls-behavior.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "component-controls-behavior.pdf",
        metadata=metadata,
        facecolor="white",
    )
    plt.close(fig)


def write_leaderboard(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "candidate_id", "candidate_role", "layer", "component_type", "head",
        "scope", "direction", "concept_count", "positive_concept_count",
        "mean_positive_example_fraction", "fraction", "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["exact"]["macro"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "fraction": row["fraction"]["estimate"],
                    "ci_low": row["fraction"]["ci_low"],
                    "ci_high": row["fraction"]["ci_high"],
                }
            )


def write_concept_results(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "concept", "candidate_id", "candidate_role", "direction",
        "fraction", "ci_low", "ci_high", "patched_mean", "positive_example_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for concept in summary["exact"]["concepts"]:
            for row in concept["cells"]:
                writer.writerow(
                    {
                        "scope": concept["scope"],
                        "concept": concept["concept"],
                        **{field: row.get(field) for field in fields},
                        "fraction": row["fraction"]["estimate"],
                        "ci_low": row["fraction"]["ci_low"],
                        "ci_high": row["fraction"]["ci_high"],
                    }
                )


def write_controls(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "comparison", "candidate_id", "scope", "direction", "layer",
        "control_candidate_count", "estimate", "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["exact"]["role_contrasts"]:
            metric = row["fraction_difference"]
            writer.writerow(
                {
                    "comparison": row["contrast"],
                    "scope": row["scope"],
                    "direction": row["direction"],
                    "estimate": metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                }
            )
        for row in summary["exact"]["same_layer_controls"]:
            metric = row["difference"]
            writer.writerow(
                {
                    "comparison": "selected_minus_nonselected_same_layer_mean",
                    "candidate_id": row["candidate_id"],
                    "scope": "discovery",
                    "direction": "rescue",
                    "layer": row["layer"],
                    "control_candidate_count": row["control_candidate_count"],
                    "estimate": metric["estimate"],
                    "ci_low": metric["ci_low"],
                    "ci_high": metric["ci_high"],
                }
            )


def write_behavior(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "candidate_id", "scope", "label", "direction", "concept_count",
        "nll_change", "ci_low", "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["behavior"]["macro"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "nll_change": row["nll_change"]["estimate"],
                    "ci_low": row["nll_change"]["ci_low"],
                    "ci_high": row["nll_change"]["ci_high"],
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery_path = output_dir / DISCOVERY_PATH.name
    confirmation_path = output_dir / CONFIRMATION_PATH.name
    behavior_path = output_dir / BEHAVIOR_PATH.name
    deterministic_archive(
        output_dir / CONFIRMATION_WORKING.name, confirmation_path
    )
    deterministic_archive(output_dir / BEHAVIOR_WORKING.name, behavior_path)
    selection = json.loads((output_dir / SELECTION_PATH.name).read_text())
    discovery = load_jsonl(discovery_path)
    confirmation = load_jsonl(confirmation_path)
    behavior = load_jsonl(behavior_path)
    summary = summarize_component_confirmation(
        discovery,
        confirmation,
        behavior,
        selection,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    selection_commits = {
        row["selection_commit"] for row in (*confirmation, *behavior)
    }
    if len(selection_commits) != 1:
        raise ValueError("confirmation rows do not name one selection commit")
    summary.update(
        {
            "freeze_id": "day04-v1",
            "selection_commit": selection_commits.pop(),
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "discovery_record_count": len(discovery),
            "confirmation_record_count": len(confirmation),
            "behavior_record_count": len(behavior),
            "discovery_results_sha256": sha256_file(discovery_path),
            "confirmation_results_sha256": sha256_file(confirmation_path),
            "behavior_results_sha256": sha256_file(behavior_path),
            "selection_data": "discovery only; validation confirmation without reranking",
            "validation_used_for_selection": False,
            "safety_split_accessed": False,
        }
    )
    (output_dir / "component-confirmation-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_leaderboard(summary, output_dir / "exact-component-leaderboard.csv")
    write_concept_results(summary, output_dir / "exact-component-concepts.csv")
    write_controls(summary, output_dir / "component-control-comparisons.csv")
    write_behavior(summary, output_dir / "behavior-nll-summary.csv")
    render_confirmation_figure(summary, output_dir)
    render_controls_figure(summary, output_dir)

    artifact_names = [
        discovery_path.name,
        confirmation_path.name,
        behavior_path.name,
        "discovery-identity-audit.json",
        "confirmation-preflight.json",
        "frozen-component-selection.json",
        "discovery-candidate-summary.json",
        "component-confirmation-summary.json",
        "exact-component-leaderboard.csv",
        "exact-component-concepts.csv",
        "component-control-comparisons.csv",
        "behavior-nll-summary.csv",
        "discovery-candidate-ranking.png",
        "discovery-candidate-ranking.pdf",
        "component-confirmation-overview.png",
        "component-confirmation-overview.pdf",
        "component-controls-behavior.png",
        "component-controls-behavior.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "generator": "scripts/day08_analyze_confirmation.py",
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
    (output_dir / "day08-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Analyzed {len(confirmation)} exact confirmation rows and "
        f"{len(behavior)} behavior rows."
    )


if __name__ == "__main__":
    main()
