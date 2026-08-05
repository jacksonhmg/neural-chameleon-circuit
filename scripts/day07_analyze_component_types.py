#!/usr/bin/env python3
"""Analyze Day 7 component-type rows and render matched causal figures."""

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
    COMPONENT_KINDS,
    RANDOM_CONTROL_LAYERS,
    SELECTED_LAYERS,
    summarize_component_types,
)


RESULT_DIR = ROOT / "results/day-07"
RAW_PATH = RESULT_DIR / "component-type-example-results.jsonl.gz"
WORKING_RAW_PATH = RESULT_DIR / "component-type-example-results.jsonl"
KIND_STYLES = {
    "attn_out": ("Attention output", "#2F6B9A", "o"),
    "mlp_out": ("MLP output", "#C97B25", "s"),
    "block_output": ("Block output", "#8C3F8C", "^"),
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


def ensure_deterministic_archive(path: Path) -> None:
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


def load_raw(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str, str, int, str, int], dict[str, Any]]:
    return {
        (
            macro["scope"],
            cell["grid"],
            cell["direction"],
            cell["layer"],
            cell["component_type"],
            cell["label"],
        ): cell["fraction"]
        for macro in summary["macro"]
        for cell in macro["cells"]
    }


def error(metric: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [
            [metric["estimate"] - metric["ci_low"]],
            [metric["ci_high"] - metric["estimate"]],
        ]
    )


def render_main_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    layers = tuple(sorted(SELECTED_LAYERS))
    x = np.asarray(layers)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    panels = (
        (axes[0, 0], "discovery", "rescue", "A  Discovery rescue"),
        (axes[0, 1], "validation", "rescue", "B  Held-out validation rescue"),
        (axes[1, 0], "discovery", "induction", "C  Discovery induction"),
        (axes[1, 1], "validation", "induction", "D  Held-out validation induction"),
    )
    plotted_values = []
    for axis, scope, direction, title in panels:
        for kind in COMPONENT_KINDS:
            label, color, marker = KIND_STYLES[kind.value]
            metrics = [
                lookup[(scope, "correct", direction, layer, kind.value, 1)]
                for layer in layers
            ]
            values = [metric["estimate"] for metric in metrics]
            lower = [metric["estimate"] - metric["ci_low"] for metric in metrics]
            upper = [metric["ci_high"] - metric["estimate"] for metric in metrics]
            plotted_values.extend(values)
            axis.errorbar(
                x,
                values,
                yerr=np.asarray([lower, upper]),
                color=color,
                marker=marker,
                linewidth=2,
                markersize=7,
                capsize=3,
                label=label,
            )
        axis.axhline(0, color="#666666", linewidth=1)
        axis.set_xticks(x)
        axis.set_xlabel("Zero-based block")
        axis.set_ylabel("Recovery fraction" if direction == "rescue" else "Induction fraction")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    y_low = min(-0.05, min(plotted_values) - 0.08)
    y_high = max(1.05, max(plotted_values) + 0.08)
    for axis in axes.ravel():
        axis.set_ylim(y_low, y_high)
    axes[0, 0].legend(frameon=False, ncol=3, loc="upper left")
    fig.suptitle(
        "Neural Chameleon causal effect by layer and component type\n"
        "Positive examples; 95% paired bootstrap intervals",
        fontsize=16,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 7 component-type causal effects",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day07_analyze_component_types.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "component-type-overview.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "component-type-overview.pdf", metadata=metadata, facecolor="white"
    )
    plt.close(fig)


def render_control_figure(summary: dict[str, Any], output_dir: Path) -> None:
    lookup = macro_lookup(summary)
    layers = tuple(sorted(SELECTED_LAYERS))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)

    for axis, scope, title in (
        (axes[0, 0], "discovery", "A  Discovery negative-example rescue shift"),
        (axes[0, 1], "validation", "B  Validation negative-example rescue shift"),
    ):
        for kind in COMPONENT_KINDS:
            label, color, marker = KIND_STYLES[kind.value]
            metrics = [
                lookup[(scope, "correct", "rescue", layer, kind.value, 0)]
                for layer in layers
            ]
            axis.errorbar(
                layers,
                [metric["estimate"] for metric in metrics],
                yerr=np.asarray(
                    [
                        [metric["estimate"] - metric["ci_low"] for metric in metrics],
                        [metric["ci_high"] - metric["estimate"] for metric in metrics],
                    ]
                ),
                color=color,
                marker=marker,
                linewidth=2,
                capsize=3,
                label=label,
            )
        axis.axhline(0, color="#555555", linewidth=1)
        axis.set_xticks(layers)
        axis.set_xlabel("Zero-based block")
        axis.set_ylabel("Negative normalized score shift")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    axis = axes[1, 0]
    for kind in COMPONENT_KINDS:
        label, color, marker = KIND_STYLES[kind.value]
        correct = [
            lookup[("all_benign", "correct", "rescue", layer, kind.value, 1)]
            for layer in layers
        ]
        irrelevant = [
            lookup[("all_benign", "irrelevant", "rescue", layer, kind.value, 1)]
            for layer in layers
        ]
        axis.plot(
            layers,
            [metric["estimate"] for metric in correct],
            color=color,
            marker=marker,
            linewidth=2,
            label=f"{label} · correct",
        )
        axis.plot(
            layers,
            [metric["estimate"] for metric in irrelevant],
            color=color,
            marker=marker,
            linewidth=1.5,
            linestyle="--",
            alpha=0.75,
            label=f"{label} · irrelevant",
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(layers)
    axis.set_xlabel("Zero-based block")
    axis.set_ylabel("Normalized rescue-like effect")
    axis.set_title("C  Correct versus irrelevant-trigger rescue")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8, ncol=2)

    axis = axes[1, 1]
    control_layers = tuple(sorted((*RANDOM_CONTROL_LAYERS, *SELECTED_LAYERS)))
    for kind in COMPONENT_KINDS:
        label, color, marker = KIND_STYLES[kind.value]
        metrics = []
        for layer in control_layers:
            grid = "correct" if layer in SELECTED_LAYERS else "random"
            metrics.append(
                lookup[("discovery", grid, "rescue", layer, kind.value, 1)]
            )
        axis.plot(
            control_layers,
            [metric["estimate"] for metric in metrics],
            color=color,
            marker=marker,
            linewidth=2,
            label=label,
        )
    axis.axhline(0, color="#555555", linewidth=1)
    axis.axvline(8.5, color="#777777", linestyle="--", linewidth=1)
    axis.set_xticks(control_layers)
    axis.set_xlabel("Zero-based block")
    axis.set_ylabel("Discovery positive rescue fraction")
    axis.set_title("D  Random pre-monitor versus selected layers")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Day 7 matched controls\n"
        "Negative examples, irrelevant triggers, and deterministic random layers",
        fontsize=16,
        fontweight="bold",
    )
    metadata = {
        "Title": "Neural Chameleon Day 7 matched controls",
        "Author": "Neural Chameleon Circuit project",
        "Creator": "scripts/day07_analyze_component_types.py",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(output_dir / "component-type-controls.png", dpi=300, facecolor="white")
    fig.savefig(
        output_dir / "component-type-controls.pdf", metadata=metadata, facecolor="white"
    )
    plt.close(fig)


def write_cells_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope",
        "concept",
        "split",
        "grid",
        "direction",
        "layer",
        "component_type",
        "label",
        "fraction",
        "ci_low",
        "ci_high",
        "patched_mean",
        "destination_mean",
        "numerator",
        "positive_denominator",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for concept in summary["concepts"]:
            for cell in concept["cells"]:
                writer.writerow(
                    {
                        "scope": "concept",
                        "concept": concept["concept"],
                        "split": concept["split"],
                        **{field: cell.get(field) for field in fields if field in cell},
                        "fraction": cell["fraction"]["estimate"],
                        "ci_low": cell["fraction"]["ci_low"],
                        "ci_high": cell["fraction"]["ci_high"],
                    }
                )
        for macro in summary["macro"]:
            for cell in macro["cells"]:
                writer.writerow(
                    {
                        "scope": macro["scope"],
                        "concept": "equal_concept_macro",
                        "split": macro["scope"],
                        "grid": cell["grid"],
                        "direction": cell["direction"],
                        "layer": cell["layer"],
                        "component_type": cell["component_type"],
                        "label": cell["label"],
                        "fraction": cell["fraction"]["estimate"],
                        "ci_low": cell["fraction"]["ci_low"],
                        "ci_high": cell["fraction"]["ci_high"],
                    }
                )


def write_contrasts_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope",
        "grid",
        "direction",
        "layer",
        "label",
        "component_type",
        "contrast",
        "control",
        "estimate",
        "ci_low",
        "ci_high",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["component_contrasts"]:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "estimate": row["value"]["estimate"],
                    "ci_low": row["value"]["ci_low"],
                    "ci_high": row["value"]["ci_high"],
                }
            )
        for row in summary["control_contrasts"]:
            writer.writerow(
                {
                    "scope": row["scope"],
                    "layer": row["selected_layer"],
                    "component_type": row["component_type"],
                    "control": row["control"],
                    "estimate": row["value"]["estimate"],
                    "ci_low": row["value"]["ci_low"],
                    "ci_high": row["value"]["ci_high"],
                }
            )


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_deterministic_archive(raw_path)
    rows = load_raw(raw_path)
    summary = summarize_component_types(
        rows, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    summary.update(
        {
            "freeze_id": "day04-v1",
            "raw_record_count": len(rows),
            "raw_results_sha256": sha256_file(raw_path),
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "selection_data": "Day 6 discovery-frozen layers only",
            "validation_used_for_selection": False,
            "safety_split_accessed": False,
        }
    )
    (output_dir / "component-type-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_cells_csv(summary, output_dir / "component-type-summary.csv")
    write_contrasts_csv(summary, output_dir / "component-type-contrasts.csv")
    render_main_figure(summary, output_dir)
    render_control_figure(summary, output_dir)

    artifact_names = [
        raw_path.name,
        "identity-audit.json",
        "component-type-summary.json",
        "component-type-summary.csv",
        "component-type-contrasts.csv",
        "component-type-overview.png",
        "component-type-overview.pdf",
        "component-type-controls.png",
        "component-type-controls.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "generator": "scripts/day07_analyze_component_types.py",
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
    (output_dir / "component-type-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Analyzed {len(rows)} Day 7 rows across {len(summary['concepts'])} concepts.")


if __name__ == "__main__":
    main()
