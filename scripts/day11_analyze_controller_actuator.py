#!/usr/bin/env python3
"""Analyze and visualize the frozen Day 11 controller-actuator experiment."""

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import summarize_controller_actuator  # noqa: E402


RESULT_DIR = ROOT / "results/day-11"


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
        with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=destination, mtime=0) as compressed:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                compressed.write(chunk)
    temporary.replace(archive)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def metric_fields(metric: MappingLike, prefix: str = "fraction") -> dict[str, float]:
    return {
        prefix: metric["estimate"],
        f"{prefix}_ci_low": metric["ci_low"],
        f"{prefix}_ci_high": metric["ci_high"],
    }


MappingLike = dict[str, Any]


def write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_tables(summary: MappingLike, output_dir: Path) -> None:
    baseline_rows = []
    for row in summary["baseline_condition_macro"]:
        baseline_rows.append({
            "scope": row["scope"], "condition_id": row["condition_id"],
            "concept_count": row["concept_count"],
            **metric_fields(row["suppression_fraction"]),
        })
    write_rows(
        output_dir / "baseline-condition-summary.csv",
        ["scope", "condition_id", "concept_count", "fraction", "fraction_ci_low", "fraction_ci_high"],
        baseline_rows,
    )

    macro_rows = []
    for row in summary["macro"]:
        macro_rows.append({
            "scope": row["scope"], "intervention_id": row["intervention_id"],
            "family": row["family"], "mode": row["mode"], "direction": row["direction"],
            "source_group": row.get("source_group"), "direct_group_id": row.get("direct_group_id"),
            "layer": row.get("layer"), "concept_count": row["concept_count"],
            "positive_concept_count": row["positive_concept_count"],
            **metric_fields(row["fraction"]),
        })
    macro_fields = [
        "scope", "intervention_id", "family", "mode", "direction", "source_group",
        "direct_group_id", "layer", "concept_count", "positive_concept_count",
        "fraction", "fraction_ci_low", "fraction_ci_high",
    ]
    write_rows(output_dir / "controller-actuator-macro.csv", macro_fields, macro_rows)

    contrast_rows = []
    for row in summary["selected_random_contrasts"]:
        contrast_rows.append({
            "scope": row["scope"], "source_group": row["source_group"],
            "direction": row["direction"],
            **metric_fields(row["fraction_difference"], "fraction_difference"),
        })
    write_rows(
        output_dir / "selected-random-source-contrasts.csv",
        ["scope", "source_group", "direction", "fraction_difference", "fraction_difference_ci_low", "fraction_difference_ci_high"],
        contrast_rows,
    )

    layer_rows = []
    for row in summary["layer_source_onsets"]:
        for cell in row["curve"]:
            layer_rows.append({
                "scope": row["scope"], "source_group": row["source_group"],
                "layer": cell["layer"], "earliest_positive_ci_layer": row["earliest_positive_ci_layer"],
                "peak_layer": row["peak_layer"], **metric_fields(cell["fraction"]),
            })
    write_rows(
        output_dir / "layer-source-curves.csv",
        ["scope", "source_group", "layer", "earliest_positive_ci_layer", "peak_layer", "fraction", "fraction_ci_low", "fraction_ci_high"],
        layer_rows,
    )

    individual_rows = []
    for role in summary["individual_head_roles"]:
        for scope in ("discovery", "validation"):
            for region, metric in role["all_regions"][scope].items():
                individual_rows.append({
                    "head_id": role["head_id"], "layer": role["layer"], "scope": scope,
                    "source_region": region,
                    "discovery_dominant_source_region": role["discovery_dominant_source_region"],
                    "validation_confirmed": role["validation_confirmed"], **metric_fields(metric),
                })
    write_rows(
        output_dir / "individual-head-source.csv",
        ["head_id", "layer", "scope", "source_region", "discovery_dominant_source_region", "validation_confirmed", "fraction", "fraction_ci_low", "fraction_ci_high"],
        individual_rows,
    )

    direct_rows = [row for row in macro_rows if row["family"] == "direct_response_output"]
    write_rows(output_dir / "direct-output-summary.csv", macro_fields, direct_rows)

    concept_rows = []
    for concept in summary["concepts"]:
        for row in concept["interventions"]:
            concept_rows.append({
                "split": concept["split"], "concept": concept["concept"],
                "intervention_id": row["intervention_id"], "family": row["family"],
                "direction": row["direction"], "source_group": row.get("source_group"),
                "direct_group_id": row.get("direct_group_id"), **metric_fields(row["fraction"]),
            })
    write_rows(
        output_dir / "controller-actuator-concepts.csv",
        ["split", "concept", "intervention_id", "family", "direction", "source_group", "direct_group_id", "fraction", "fraction_ci_low", "fraction_ci_high"],
        concept_rows,
    )


def render_layer_figure(summary: MappingLike, output_dir: Path) -> None:
    lookup = {(row["scope"], row["source_group"]): row for row in summary["layer_source_onsets"]}
    styles = (
        ("monitoring_language", "Monitoring language", "#3B6EA8", "o"),
        ("named_concept", "Named concept", "#8E5EA2", "s"),
        ("trigger_other", "Trigger punctuation", "#A7A7A7", "x"),
        ("original_prompt", "Original prompt", "#D18F45", "^"),
        ("template", "Chat template", "#777777", "v"),
        ("response", "Prior response", "#4C9F70", "D"),
        ("trigger_instruction", "Complete trigger", "#B64848", "*"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for axis, scope in zip(axes, ("discovery", "validation"), strict=True):
        for source_group, label, color, marker in styles:
            curve = lookup[(scope, source_group)]["curve"]
            layers = np.asarray([row["layer"] for row in curve])
            point = np.asarray([row["fraction"]["estimate"] for row in curve])
            low = np.asarray([row["fraction"]["ci_low"] for row in curve])
            high = np.asarray([row["fraction"]["ci_high"] for row in curve])
            axis.plot(layers, point, color=color, marker=marker, linewidth=2, label=label)
            axis.fill_between(layers, low, high, color=color, alpha=0.10)
        axis.axhline(0, color="#333333", linewidth=1)
        axis.set_xticks(range(13))
        axis.set_xlabel("Layer: all 16 query heads patched")
        axis.set_ylabel("Positive recovery fraction")
        axis.set_title(f"{scope.capitalize()} concepts")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9, ncol=2)
    fig.suptitle(
        "Day 11 causal source-channel scan\nRegion-specific attention contributions, full trigger → normal rescue",
        fontsize=17, fontweight="bold",
    )
    metadata = {"Title": "Day 11 causal source-channel scan", "Creator": "scripts/day11_analyze_controller_actuator.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "layer-source-contributions.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "layer-source-contributions.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def render_head_figure(summary: MappingLike, output_dir: Path) -> None:
    roles = summary["individual_head_roles"]
    regions = ["monitoring_language", "named_concept", "trigger_other", "original_prompt", "template", "response"]
    labels = ["Monitoring", "Concept", "Trigger other", "Original prompt", "Template", "Response"]
    heads = [row["head_id"].replace("layer_", "L").replace(".head_", " H") for row in roles]
    values = {
        scope: np.asarray([[row["all_regions"][scope][region]["estimate"] for region in regions] for row in roles])
        for scope in ("discovery", "validation")
    }
    limit = max(abs(values["discovery"]).max(), abs(values["validation"]).max())
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    images = []
    for axis, scope in zip(axes, ("discovery", "validation"), strict=True):
        image = axis.imshow(values[scope], cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        images.append(image)
        axis.set_xticks(range(len(regions)), labels, rotation=35, ha="right")
        axis.set_yticks(range(len(heads)), heads)
        axis.set_title(f"{scope.capitalize()} rescue")
        for row in range(len(heads)):
            for column in range(len(regions)):
                axis.text(column, row, f"{values[scope][row, column]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(images[0], ax=axes, label="Positive recovery fraction", shrink=0.8)
    fig.suptitle(
        "Day 11 selected-head causal source roles\nDiscovery-frozen heads; validation is confirmatory",
        fontsize=17, fontweight="bold",
    )
    metadata = {"Title": "Day 11 selected-head causal source roles", "Creator": "scripts/day11_analyze_controller_actuator.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "selected-head-source-roles.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "selected-head-source-roles.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def render_mechanism(summary: MappingLike, output_dir: Path) -> None:
    macro = {(row["scope"], row["intervention_id"]): row for row in summary["macro"]}
    evidence = summary["controller_actuator_evidence"]
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, title: str, detail: str, supported: bool | None = None):
        color = "#DCEFD9" if supported is True else "#F5D8D8" if supported is False else "#E4E9F2"
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.015", facecolor=color, edgecolor="#333333", linewidth=1.5)
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height * 0.66, title, ha="center", va="center", fontsize=13, fontweight="bold")
        axis.text(x + width / 2, y + height * 0.30, detail, ha="center", va="center", fontsize=9, wrap=True)

    def arrow(
        start: tuple[float, float],
        stop: tuple[float, float],
        label: str,
        supported: bool | None,
    ):
        color = "#367A4A" if supported is True else "#A54848" if supported is False else "#687386"
        axis.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=color))
        axis.text((start[0]+stop[0])/2, (start[1]+stop[1])/2 + 0.035, label, ha="center", fontsize=8, color=color)

    monitor = evidence["monitoring_recognition_supported"]
    concept = evidence["concept_identification_supported"]
    response = evidence["response_processing_supported"]
    mlp = evidence["mlp_actuator_supported"]
    box(0.03, 0.63, 0.17, 0.16, "Monitoring language", "You are being probed for …", monitor)
    box(0.03, 0.35, 0.17, 0.16, "Named concept", "Exact concept token span", concept)
    box(0.03, 0.07, 0.17, 0.16, "Prompt context", "Original prompt + template", None)
    box(0.30, 0.42, 0.22, 0.22, "Attention controller", "Causal source contributions\nthrough selected query heads", monitor and concept)
    box(0.59, 0.63, 0.17, 0.16, "Response processing", "Prior response-token sources", response)
    box(0.59, 0.28, 0.17, 0.16, "MLP actuator", "Selected layer 9–12 MLP\nresponse outputs", mlp)
    box(0.82, 0.42, 0.15, 0.22, "Anti-probe state", "Lower layer-12 monitor score\npartial natural sufficiency", evidence["overall_supported"])

    def pair_value(group: str) -> str:
        values = [macro[(scope, f"group.selected_attention_12.{group}.rescue")]["fraction"]["estimate"] for scope in ("discovery", "validation")]
        return f"D {values[0]:.3f} · V {values[1]:.3f}"

    arrow((0.20, 0.71), (0.30, 0.58), pair_value("monitoring_language"), monitor)
    arrow((0.20, 0.43), (0.30, 0.52), pair_value("named_concept"), concept)
    arrow((0.20, 0.15), (0.30, 0.46), pair_value("original_prompt"), None)
    arrow((0.52, 0.58), (0.59, 0.71), pair_value("response"), response)
    arrow((0.52, 0.48), (0.59, 0.36), "direct response outputs", mlp)
    arrow((0.76, 0.71), (0.82, 0.58), "attention pathway", response)
    mlp_values = [macro[(scope, "direct.selected_mlp_4.induction")]["fraction"]["estimate"] for scope in ("discovery", "validation")]
    arrow((0.76, 0.36), (0.82, 0.48), f"induction D {mlp_values[0]:.2f} · V {mlp_values[1]:.2f}", mlp)
    axis.text(0.5, 0.94, "Day 11 provisional causal controller–actuator mechanism", ha="center", fontsize=19, fontweight="bold")
    axis.text(0.5, 0.89, evidence["classification"].replace("_", " "), ha="center", fontsize=13)
    axis.text(0.5, 0.015, "Green = frozen support rule passed; red = failed; gray = descriptive. D/V are equal-concept recovery fractions.", ha="center", fontsize=9)
    metadata = {"Title": "Day 11 provisional causal mechanism", "Creator": "scripts/day11_analyze_controller_actuator.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "provisional-controller-actuator-mechanism.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "provisional-controller-actuator-mechanism.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.resolve()
    working = output_dir / "controller-actuator-results.jsonl"
    archive = output_dir / "controller-actuator-results.jsonl.gz"
    deterministic_archive(working, archive)
    records = load_jsonl(archive)
    plan_path = output_dir / "frozen-controller-actuator-plan.json"
    plan = json.loads(plan_path.read_text())
    summary = summarize_controller_actuator(
        records, plan, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    provenance_fields = (
        "selection_commit", "day10_procedure_commit", "day10_results_commit",
        "day11_procedure_commit", "implementation_commit",
    )
    provenance = {}
    for field in provenance_fields:
        values = {row[field] for row in records}
        if len(values) != 1:
            raise ValueError(f"Day 11 rows do not name one {field}")
        provenance[field] = values.pop()
    summary.update({
        "freeze_id": "day04-v1", **provenance,
        "model_repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
        "model_revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
        "record_count": len(records),
        "controller_actuator_plan_sha256": sha256_file(plan_path),
        "raw_results_sha256": sha256_file(archive),
        "selection_data": "Day 8 discovery-frozen components; no Day 11 reranking",
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    })
    summary_path = output_dir / "controller-actuator-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_tables(summary, output_dir)
    render_layer_figure(summary, output_dir)
    render_head_figure(summary, output_dir)
    render_mechanism(summary, output_dir)

    artifact_names = [
        "frozen-controller-actuator-plan.json",
        "controller-actuator-results.jsonl.gz",
        "controller-actuator-preflight.json",
        "controller-actuator-summary.json",
        "baseline-condition-summary.csv",
        "controller-actuator-macro.csv",
        "selected-random-source-contrasts.csv",
        "layer-source-curves.csv",
        "individual-head-source.csv",
        "direct-output-summary.csv",
        "controller-actuator-concepts.csv",
        "layer-source-contributions.png",
        "layer-source-contributions.pdf",
        "selected-head-source-roles.png",
        "selected-head-source-roles.pdf",
        "provisional-controller-actuator-mechanism.png",
        "provisional-controller-actuator-mechanism.pdf",
    ]
    manifest = {
        "schema_version": 1, "freeze_id": "day04-v1", "procedure": "day11-v1",
        "generator": "scripts/day11_analyze_controller_actuator.py",
        "files": {
            name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in artifact_names
        },
        "validation_used_for_selection": False, "safety_split_accessed": False,
    }
    (output_dir / "day11-artifacts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    evidence = summary["controller_actuator_evidence"]
    print(f"Analyzed {len(records)} Day 11 rows; classification={evidence['classification']}.")


if __name__ == "__main__":
    main()
