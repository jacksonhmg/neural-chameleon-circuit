#!/usr/bin/env python3
"""Frozen confirmatory analysis for the locked Day 13 safety transfer test."""

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

from neural_chameleon.safety_transfer import summarize_safety_transfer  # noqa: E402


RESULT_DIR = ROOT / "results/day-13"
FREEZE_PATH = ROOT / "results/day-12/frozen-confirmatory-analysis.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RESULT_DIR / "safety-transfer-results.jsonl.gz")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def write_tables(summary: dict[str, Any], output_dir: Path) -> None:
    rows = []
    for concept in summary["concepts"]:
        for cell in concept["cells"]:
            rows.append({
                "concept": concept["concept"], "group_id": cell["group_id"],
                "direction": cell["direction"], "label": cell["label"],
                "fraction": cell["fraction"]["estimate"],
                "fraction_ci_low": cell["fraction"]["ci_low"],
                "fraction_ci_high": cell["fraction"]["ci_high"],
                "response_nll_shift": cell["response_nll_shift"]["estimate"],
                "response_kl": cell["response_kl"]["estimate"],
                "activation_rms_ratio": cell["activation_rms_ratio"]["estimate"],
            })
    fields = ["concept", "group_id", "direction", "label", "fraction", "fraction_ci_low", "fraction_ci_high", "response_nll_shift", "response_kl", "activation_rms_ratio"]
    with (output_dir / "safety-transfer-metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_figure(summary: dict[str, Any], output_dir: Path) -> None:
    concepts = [row["concept"] for row in summary["concepts"]]
    cells = []
    for concept in summary["concepts"]:
        lookup = {(row["group_id"], row["direction"], row["label"]): row for row in concept["cells"]}
        cells.append([
            lookup[(group, direction, 1)]["fraction"]["estimate"]
            for group in ("selected_k16", "random_k16") for direction in ("rescue", "induction")
        ])
    values = np.asarray(cells)
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(4), ["Selected rescue", "Selected induction", "Random rescue", "Random induction"])
    axis.set_yticks(range(len(concepts)), concepts)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Positive causal fraction")
    axis.set_title("Frozen benign-selected circuit transfer to locked safety concepts", fontsize=17, fontweight="bold")
    metadata = {"Title": "Frozen safety transfer", "Creator": "scripts/day13_analyze_safety_test.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "safety-transfer.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "safety-transfer.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    freeze = json.loads(FREEZE_PATH.read_text())
    if freeze["status"] != "frozen-before-safety-unlock":
        raise RuntimeError("confirmatory analysis is not frozen")
    if sha256_file(Path(__file__).resolve()) != freeze["analysis_script_sha256"]:
        raise RuntimeError("frozen safety analysis script hash changed")
    raw = args.raw.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(raw)
    summary = summarize_safety_transfer(
        records,
        concepts=freeze["safety_concepts"],
        selected_group="selected_k16",
        random_group="random_k16",
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    summary.update({
        "schema_version": 1,
        "procedure": "day13-confirmatory-v1",
        "raw_record_count": len(records),
        "raw_results_sha256": sha256_file(raw),
        "confirmatory_freeze_sha256": sha256_file(FREEZE_PATH),
        "component_set_sha256": freeze["component_set_sha256"],
        "component_set_modified_after_safety": False,
    })
    (output_dir / "safety-transfer-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_tables(summary, output_dir)
    render_figure(summary, output_dir)
    print(f"Analyzed {len(records)} frozen safety rows; overall support: {summary['overall_safety_transfer_supported']}")


if __name__ == "__main__":
    main()
