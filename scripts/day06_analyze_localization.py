#!/usr/bin/env python3
"""Analyze Day 6 localization rows and render rescue/induction heatmaps."""

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
from matplotlib.colors import TwoSlopeNorm
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import TokenRegion, summarize_localization  # noqa: E402


RESULT_DIR = ROOT / "results/day-06"
RAW_PATH = RESULT_DIR / "localization-example-results.jsonl.gz"
WORKING_RAW_PATH = RESULT_DIR / "localization-example-results.jsonl"
CONCEPTS = ("HTML", "biology-focused", "comforting", "german")
CONCEPT_LABELS = {
    "HTML": "HTML",
    "biology-focused": "Biology",
    "comforting": "Comforting",
    "german": "German",
    "macro": "Macro",
}
REGION_LABELS = {
    "prompt": "prompt",
    "response": "response",
    "all_aligned": "all aligned",
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
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def ensure_deterministic_archive(path: Path) -> None:
    """Pack the resumable working JSONL with a fixed gzip header timestamp."""
    if path != RAW_PATH:
        if path.is_file():
            return
        raise FileNotFoundError(path)
    if not WORKING_RAW_PATH.is_file():
        if path.is_file():
            return
        raise FileNotFoundError(path)
    if path.parent != WORKING_RAW_PATH.parent:
        raise ValueError("working and archive paths must share a result directory")
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


def concept_cell_lookup(summary: dict[str, Any]) -> dict[tuple[str, int, str, str], dict[str, Any]]:
    return {
        (concept["concept"], cell["layer"], cell["token_region"], cell["direction"]): cell
        for concept in summary["concepts"]
        for cell in concept["cells"]
    }


def macro_cell_lookup(summary: dict[str, Any]) -> dict[tuple[int, str, str], dict[str, Any]]:
    return {
        (cell["layer"], cell["token_region"], cell["direction"]): cell
        for cell in summary["macro_cells"]
    }


def figure_matrix(
    summary: dict[str, Any], direction: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    concept_cells = concept_cell_lookup(summary)
    macro_cells = macro_cell_lookup(summary)
    rows = []
    positive = []
    negative = []
    labels = []
    for concept in (*CONCEPTS, "macro"):
        for region in TokenRegion:
            values = []
            positive_cells = []
            negative_cells = []
            for layer in range(42):
                cell = (
                    macro_cells[(layer, region.value, direction)]
                    if concept == "macro"
                    else concept_cells[(concept, layer, region.value, direction)]
                )
                metric = cell["fraction"]
                values.append(metric["estimate"])
                positive_cells.append(metric["ci_low"] > 0)
                negative_cells.append(metric["ci_high"] < 0)
            rows.append(values)
            positive.append(positive_cells)
            negative.append(negative_cells)
            labels.append(f"{CONCEPT_LABELS[concept]} · {REGION_LABELS[region.value]}")
    return np.asarray(rows), np.asarray(positive), np.asarray(negative), labels


def render_heatmaps(summary: dict[str, Any], output_dir: Path) -> None:
    matrices = {
        direction: figure_matrix(summary, direction)
        for direction in ("rescue", "induction")
    }
    executed_values = np.concatenate(
        [matrix[:, :13].ravel() for matrix, _, _, _ in matrices.values()]
    )
    color_limit = max(1.0, float(np.ceil(np.max(np.abs(executed_values)) * 4) / 4))
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    retained = summary["retained_top_four_layers"]

    for direction, (matrix, positive, negative, labels) in matrices.items():
        fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
        image = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        for row, layer in zip(*np.where(positive[:, :13])):
            ax.plot(layer, row, marker=".", color="black", markersize=3.5)
        for row, layer in zip(*np.where(negative[:, :13])):
            ax.plot(layer, row, marker="x", color="white", markersize=3.2, markeredgewidth=0.6)
        ax.axvline(12.5, color="#222222", linewidth=1.3)
        ax.axvspan(12.5, 41.5, color="#777777", alpha=0.16)
        for boundary in (2.5, 5.5, 8.5, 11.5):
            ax.axhline(boundary, color="#333333", linewidth=0.8)
        macro_response_row = len(labels) - 2
        for layer in retained:
            ax.scatter(
                layer,
                macro_response_row,
                marker="s",
                s=45,
                facecolors="none",
                edgecolors="#F7D13D",
                linewidths=1.5,
            )
        ax.set_xticks(np.arange(42))
        ax.set_xticklabels([str(layer) for layer in range(42)], fontsize=7)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Zero-based block L patched at resid_post[L]")
        ax.set_ylabel("Discovery concept and aligned token region")
        title_action = "Normal → triggered rescue" if direction == "rescue" else "Triggered → normal induction"
        ax.set_title(
            f"Day 6 coarse residual-stream localization — {title_action}\n"
            "dot: 95% CI above zero; ×: 95% CI below zero; gold boxes: retained rescue layers"
        )
        ax.text(
            27,
            0.0,
            "post-monitor structural causal nulls",
            ha="center",
            va="center",
            fontsize=9,
            color="#333333",
        )
        colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.015)
        colorbar.set_label("Recovery fraction" if direction == "rescue" else "Induction fraction")
        metadata = {
            "Title": f"Neural Chameleon Day 6 {direction} heatmap",
            "Author": "Neural Chameleon Circuit project",
            "Creator": "scripts/day06_analyze_localization.py",
            "CreationDate": None,
            "ModDate": None,
        }
        fig.savefig(output_dir / f"{direction}-heatmap.png", dpi=300, facecolor="white")
        fig.savefig(
            output_dir / f"{direction}-heatmap.pdf",
            metadata=metadata,
            facecolor="white",
        )
        plt.close(fig)


def write_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "scope",
        "concept",
        "layer",
        "token_region",
        "direction",
        "fraction",
        "ci_low",
        "ci_high",
        "normal_mean",
        "triggered_mean",
        "patched_mean",
        "numerator",
        "denominator",
        "execution_mode",
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
                        "layer": cell["layer"],
                        "token_region": cell["token_region"],
                        "direction": cell["direction"],
                        "fraction": cell["fraction"]["estimate"],
                        "ci_low": cell["fraction"]["ci_low"],
                        "ci_high": cell["fraction"]["ci_high"],
                        "normal_mean": cell["normal_mean"],
                        "triggered_mean": cell["triggered_mean"],
                        "patched_mean": cell["patched_mean"],
                        "numerator": cell["numerator"],
                        "denominator": cell["denominator"],
                        "execution_mode": cell["execution_mode"],
                    }
                )
        for cell in summary["macro_cells"]:
            writer.writerow(
                {
                    "scope": "macro",
                    "concept": "equal_concept_macro",
                    "layer": cell["layer"],
                    "token_region": cell["token_region"],
                    "direction": cell["direction"],
                    "fraction": cell["fraction"]["estimate"],
                    "ci_low": cell["fraction"]["ci_low"],
                    "ci_high": cell["fraction"]["ci_high"],
                    "execution_mode": cell["execution_mode"],
                }
            )


def main() -> None:
    args = parse_args()
    raw_path = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_deterministic_archive(raw_path)
    records = load_raw(raw_path)
    summary = summarize_localization(
        records,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary.update(
        {
            "freeze_id": "day04-v1",
            "raw_record_count": len(records),
            "raw_results_sha256": sha256_file(raw_path),
            "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
            "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
            "validation_split_accessed": False,
            "safety_split_accessed": False,
        }
    )
    summary_path = output_dir / "localization-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_csv(summary, output_dir / "localization-summary.csv")

    ranking = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "selection_data": "discovery positive examples only",
        "selection_metric": "equal-concept macro full-response resid_post rescue recovery",
        "tie_break": "lower zero-based layer",
        "ranking": summary["full_response_rescue_ranking"],
        "retained_top_four_layers": summary["retained_top_four_layers"],
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    (output_dir / "coarse-layer-ranking.json").write_text(
        json.dumps(ranking, indent=2, sort_keys=True) + "\n"
    )
    render_heatmaps(summary, output_dir)

    artifact_names = [
        raw_path.name,
        "identity-audit.json",
        "localization-summary.json",
        "localization-summary.csv",
        "coarse-layer-ranking.json",
        "rescue-heatmap.png",
        "rescue-heatmap.pdf",
        "induction-heatmap.png",
        "induction-heatmap.pdf",
    ]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "generator": "scripts/day06_analyze_localization.py",
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in artifact_names
        },
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    (output_dir / "localization-artifacts.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Analyzed {len(records)} rows; retained layers "
        f"{summary['retained_top_four_layers']}."
    )


if __name__ == "__main__":
    main()
