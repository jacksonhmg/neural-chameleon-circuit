#!/usr/bin/env python3
"""Analyze and visualize the frozen Day 12 benign transfer experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    classify_mechanism,
    summarize_candidate_transfer,
    summarize_trigger_reader_sources,
)


RESULT_DIR = ROOT / "results/day-12"
DISCOVERY_RAW = ROOT / "results/day-08/discovery-candidate-results.jsonl.gz"
VALIDATION_WORKING = RESULT_DIR / "validation-candidate-results.jsonl"
VALIDATION_RAW = RESULT_DIR / "validation-candidate-results.jsonl.gz"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
DAY09_SUMMARY = ROOT / "results/day-09/grouped-necessity-summary.json"
DAY10_SUMMARY = ROOT / "results/day-10/sufficiency-summary.json"
DAY11_SUMMARY = ROOT / "results/day-11/controller-actuator-summary.json"
DAY11_RAW = ROOT / "results/day-11/controller-actuator-results.jsonl.gz"
PLAN_PATH = RESULT_DIR / "frozen-benign-transfer-plan.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
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
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def find_macro(summary: dict[str, Any], scope: str, group_id: str) -> dict[str, Any]:
    matches = [
        row for row in summary["macro"]
        if row["scope"] == scope and row.get("group_id") == group_id and row.get("label") == 1
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {scope}/{group_id} positive macro, found {len(matches)}")
    return matches[0]


def find_concept_cell(summary: dict[str, Any], concept: str, group_id: str) -> dict[str, Any]:
    matches = [
        cell for row in summary["concepts"] if row["concept"] == concept
        for cell in row["cells"]
        if cell.get("group_id") == group_id and cell.get("label") == 1
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {concept}/{group_id} positive cell")
    return matches[0]


def find_contrast(summary: dict[str, Any], scope: str, day: int) -> dict[str, Any]:
    if day == 9:
        matches = [
            row for row in summary["selected_random_contrasts"]
            if row["scope"] == scope and row["label"] == 1 and row["set_size"] == 16
        ]
    else:
        matches = [
            row for row in summary["selected_random_contrasts"]
            if row["scope"] == scope and row["label"] == 1 and row["selected_group_id"] == "selected_k16"
        ]
    if len(matches) != 1:
        raise ValueError(f"expected one Day {day} K16 contrast for {scope}")
    return matches[0]


def direct_transfer(day09: dict[str, Any], day10: dict[str, Any]) -> dict[str, Any]:
    scopes = []
    for scope in ("discovery", "validation", "all_benign"):
        rescue = find_macro(day09, scope, "selected_k16")["fraction"]
        induction = find_macro(day10, scope, "selected_k16")["fraction"]
        rescue_contrast = find_contrast(day09, scope, 9)["fraction_difference"]
        induction_contrast = find_contrast(day10, scope, 10)["fraction_difference"]
        scopes.append({
            "scope": scope,
            "rescue": rescue,
            "induction": induction,
            "selected_minus_random_rescue": rescue_contrast,
            "selected_minus_random_induction": induction_contrast,
        })
    primary_scopes = [row for row in scopes if row["scope"] in {"discovery", "validation"}]
    supported = all(
        row[key]["ci_low"] > 0
        for row in primary_scopes
        for key in ("rescue", "induction", "selected_minus_random_rescue", "selected_minus_random_induction")
    )
    concepts = []
    for concept_row in day09["concepts"]:
        concept = concept_row["concept"]
        rescue = find_concept_cell(day09, concept, "selected_k16")["fraction"]
        induction = find_concept_cell(day10, concept, "selected_k16")["fraction"]
        concepts.append({
            "concept": concept,
            "split": concept_row["split"],
            "rescue": rescue,
            "induction": induction,
            "both_point_positive": rescue["estimate"] > 0 and induction["estimate"] > 0,
            "both_ci_positive": rescue["ci_low"] > 0 and induction["ci_low"] > 0,
        })
    return {"supported": supported, "scopes": scopes, "concepts": concepts}


def shared_actuator(day11: dict[str, Any], supporting: dict[str, Any]) -> dict[str, Any]:
    primary = []
    for scope in ("discovery", "validation", "all_benign"):
        for direction in ("rescue", "induction"):
            matches = [
                row for row in day11["macro"]
                if row["scope"] == scope
                and row.get("family") == "direct_response_output"
                and row.get("direct_group_id") == "selected_mlp_4"
                and row["direction"] == direction
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one Day 11 selected MLP row for {scope}/{direction}")
            primary.append({"scope": scope, "direction": direction, "fraction": matches[0]["fraction"]})
    supported = all(
        row["fraction"]["ci_low"] > 0
        for row in primary if row["scope"] in {"discovery", "validation"}
    )
    return {"supported": supported, "primary_selected_mlp_4": primary, "supporting_layer11_mlp": supporting}


def sparse_k4(day09: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row for row in day09["selected_curve"]
        if row["scope"] in {"discovery", "validation"} and row["set_size"] == 4
    ]
    if len(rows) != 2:
        raise ValueError("missing Day 9 K4 completeness rows")
    return {
        "supported": all(row["relative_to_k16"]["estimate"] >= 0.8 for row in rows),
        "threshold": 0.8,
        "scopes": [{"scope": row["scope"], "relative_to_k16": row["relative_to_k16"]} for row in rows],
    }


def write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def write_tables(summary: dict[str, Any], output_dir: Path) -> None:
    concepts = [row["concept"] for row in summary["candidate_transfer"]["concepts"]]
    direct = {row["concept"]: row for row in summary["direct_transfer"]["concepts"]}
    matrix = {row["candidate_id"]: row["concept_recovery"] for row in summary["candidate_transfer"]["selected_component_matrix"]}
    fields = ["split", "concept", "selected_k16_rescue", "selected_k16_rescue_ci_low", "selected_k16_rescue_ci_high", "selected_k16_induction", "selected_k16_induction_ci_low", "selected_k16_induction_ci_high", *matrix]
    rows = []
    for concept in concepts:
        cell = direct[concept]
        rows.append({
            "split": cell["split"], "concept": concept,
            "selected_k16_rescue": cell["rescue"]["estimate"],
            "selected_k16_rescue_ci_low": cell["rescue"]["ci_low"],
            "selected_k16_rescue_ci_high": cell["rescue"]["ci_high"],
            "selected_k16_induction": cell["induction"]["estimate"],
            "selected_k16_induction_ci_low": cell["induction"]["ci_low"],
            "selected_k16_induction_ci_high": cell["induction"]["ci_high"],
            **{candidate_id: values[concept] for candidate_id, values in matrix.items()},
        })
    write_rows(output_dir / "benign-cross-concept-transfer-matrix.csv", fields, rows)

    agreement_fields = ["left_concept", "right_concept", "left_split", "right_split", "spearman_rho", "intersection", "overlap_coefficient", "jaccard"]
    write_rows(output_dir / "concept-ranking-agreement.csv", agreement_fields, summary["candidate_transfer"]["pairwise_ranking_agreement"])

    rank_rows = []
    for macro in summary["candidate_transfer"]["macro"]:
        for row in macro["candidates"]:
            rank_rows.append({
                "scope": macro["scope"], "rank": row["rank"], "candidate_id": row["candidate_id"],
                "component_type": row["component_type"], "layer": row["layer"], "head": row["head"],
                "recovery": row["recovery"]["estimate"], "recovery_ci_low": row["recovery"]["ci_low"],
                "recovery_ci_high": row["recovery"]["ci_high"], "positive_concept_count": row["positive_concept_count"],
            })
    write_rows(
        output_dir / "candidate-macro-rankings.csv",
        ["scope", "rank", "candidate_id", "component_type", "layer", "head", "recovery", "recovery_ci_low", "recovery_ci_high", "positive_concept_count"],
        rank_rows,
    )

    source_rows = []
    for row in summary["trigger_reader_sources"]["heads"]:
        for role in row["concept_roles"]:
            source_rows.append({
                "head_id": row["head_id"], "concept": role["concept"], "split": role["split"],
                "dominant_source_region": role["source_region"], "dominant_positive": role["positive"],
                "head_concept_specific": row["concept_specific"],
            })
    write_rows(
        output_dir / "trigger-reader-source-roles.csv",
        ["head_id", "concept", "split", "dominant_source_region", "dominant_positive", "head_concept_specific"],
        source_rows,
    )


def render_transfer_figure(summary: dict[str, Any], output_dir: Path) -> None:
    concepts = [row["concept"] for row in summary["candidate_transfer"]["concepts"]]
    selected = summary["candidate_transfer"]["selected_component_matrix"]
    direct = {row["concept"]: row for row in summary["direct_transfer"]["concepts"]}
    values = np.asarray([
        [direct[concept]["rescue"]["estimate"], direct[concept]["induction"]["estimate"], *[row["concept_recovery"][concept] for row in selected]]
        for concept in concepts
    ])
    columns = ["K16 rescue", "K16 induction", *[row["candidate_id"].replace("layer_", "L").replace(".head_", " H").replace(".mlp", " MLP") for row in selected]]
    limit = max(1.0, float(np.max(np.abs(values))))
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(columns)), columns, rotation=55, ha="right")
    labels = [f"{concept} {'(D)' if summary['candidate_transfer']['concepts'][index]['split'] == 'discovery' else '(V)'}" for index, concept in enumerate(concepts)]
    axis.set_yticks(range(len(concepts)), labels)
    axis.axvline(1.5, color="white", linewidth=3)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.2f}", ha="center", va="center", fontsize=6.5, color="black")
    fig.colorbar(image, ax=axis, label="Causal recovery / induction fraction", shrink=0.85)
    axis.set_title("Day 12 benign cross-concept causal transfer\nDiscovery-frozen K=16 set; validation never reranked", fontsize=17, fontweight="bold")
    metadata = {"Title": "Day 12 benign cross-concept causal transfer", "Creator": "scripts/day12_analyze_benign_transfer.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "benign-cross-concept-transfer.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "benign-cross-concept-transfer.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def render_agreement_figure(summary: dict[str, Any], output_dir: Path) -> None:
    concepts = [row["concept"] for row in summary["candidate_transfer"]["concepts"]]
    size = len(concepts)
    rho = np.eye(size)
    overlap = np.ones((size, size))
    index = {concept: position for position, concept in enumerate(concepts)}
    for row in summary["candidate_transfer"]["pairwise_ranking_agreement"]:
        left = index[row["left_concept"]]
        right = index[row["right_concept"]]
        rho[left, right] = rho[right, left] = row["spearman_rho"]
        overlap[left, right] = overlap[right, left] = row["intersection"] / 16
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), constrained_layout=True)
    for axis, matrix, title, cmap, low in (
        (axes[0], rho, "68-candidate Spearman ρ", "RdBu_r", -1),
        (axes[1], overlap, "Top-16 overlap fraction", "Blues", 0),
    ):
        image = axis.imshow(matrix, cmap=cmap, vmin=low, vmax=1)
        axis.set_xticks(range(size), concepts, rotation=55, ha="right")
        axis.set_yticks(range(size), concepts)
        axis.set_title(title)
        for row in range(size):
            for column in range(size):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=6.5)
        fig.colorbar(image, ax=axis, shrink=0.75)
    macro = summary["candidate_transfer"]["discovery_validation_macro_agreement"]
    fig.suptitle(
        f"Day 12 component-ranking sharing across benign concepts\nDiscovery–validation macro: ρ={macro['spearman_rho']:.2f}; top-16 overlap={macro['intersection']}/16",
        fontsize=17, fontweight="bold",
    )
    metadata = {"Title": "Day 12 component-ranking agreement", "Creator": "scripts/day12_analyze_benign_transfer.py", "CreationDate": None, "ModDate": None}
    fig.savefig(output_dir / "component-ranking-agreement.png", dpi=300, facecolor="white")
    fig.savefig(output_dir / "component-ranking-agreement.pdf", metadata=metadata, facecolor="white")
    plt.close(fig)


def write_manifest(output_dir: Path, names: list[str]) -> None:
    manifest = {
        "schema_version": 1,
        "procedure": "day12-v1",
        "files": {
            name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
            for name in names
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    (output_dir / "day12-artifacts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.result_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_raw = output_dir / VALIDATION_RAW.name
    deterministic_archive(VALIDATION_WORKING, validation_raw)
    plan = json.loads(PLAN_PATH.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    day09 = json.loads(DAY09_SUMMARY.read_text())
    day10 = json.loads(DAY10_SUMMARY.read_text())
    day11 = json.loads(DAY11_SUMMARY.read_text())
    candidate_records = load_jsonl(DISCOVERY_RAW) + load_jsonl(validation_raw)
    candidate = summarize_candidate_transfer(
        candidate_records,
        discovery_concepts=plan["data_scope"]["discovery_concepts"],
        validation_concepts=plan["data_scope"]["validation_concepts"],
        selected_candidates=selection["selected_candidates"],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    selected_attention = [
        candidate_id for candidate_id in selection["selected_candidates"]
        if CANDIDATE_BY_ID[candidate_id].component_type == "attention_head"
    ]
    sources = summarize_trigger_reader_sources(
        load_jsonl(DAY11_RAW),
        selected_attention=selected_attention,
        source_regions=plan["mechanistic_tests"]["trigger_reader_specificity"]["source_regions"],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    direct = direct_transfer(day09, day10)
    actuator = shared_actuator(day11, candidate["layer11_mlp_shared_supporting_test"])
    sparse = sparse_k4(day09)
    macro_agreement = candidate["discovery_validation_macro_agreement"]
    classification = classify_mechanism(
        direct_transfer_supported=direct["supported"],
        sharing_supported=macro_agreement["sharing_supported"],
        high_sharing_supported=macro_agreement["high_sharing_supported"],
        sparse_k4_supported=sparse["supported"],
        shared_actuator_supported=actuator["supported"],
        trigger_reader_specificity_supported=sources["specificity_supported"],
    )
    summary = {
        "schema_version": 1,
        "procedure": "day12-v1",
        "freeze_id": "day04-v1",
        "model_repository": plan["model"]["repository"],
        "model_revision": plan["model"]["revision"],
        "transfer_plan_sha256": sha256_file(PLAN_PATH),
        "component_set_sha256": selection["component_set_sha256"],
        "final_component_set": selection["selected_candidates"],
        "discovery_raw_sha256": sha256_file(DISCOVERY_RAW),
        "validation_raw_sha256": sha256_file(validation_raw),
        "validation_record_count": len(load_jsonl(validation_raw)),
        "candidate_transfer": candidate,
        "direct_transfer": direct,
        "shared_downstream_actuator": actuator,
        "trigger_reader_sources": sources,
        "sparse_k4_test": sparse,
        "mechanistic_classification": classification,
        "evidence": {
            "direct_transfer_supported": direct["supported"],
            "rank_sharing_supported": macro_agreement["sharing_supported"],
            "high_rank_sharing_supported": macro_agreement["high_sharing_supported"],
            "shared_downstream_actuator_supported": actuator["supported"],
            "trigger_reader_source_specificity_supported": sources["specificity_supported"],
            "trigger_reader_identity_specificity_supported": candidate["trigger_reader_identity_specificity_supported"],
            "one_shared_sparse_circuit_supported": classification == "one_shared_sparse_circuit",
            "concept_specific_routing_into_shared_actuator_supported": classification == "concept_specific_routing_into_shared_actuator",
            "overlapping_non_identical_circuits_supported": classification == "overlapping_non_identical_circuits",
            "broadly_distributed_mechanism_supported": classification == "broadly_distributed_mechanism",
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    summary_path = output_dir / "benign-transfer-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_tables(summary, output_dir)
    render_transfer_figure(summary, output_dir)
    render_agreement_figure(summary, output_dir)
    names = [
        validation_raw.name,
        "validation-candidate-preflight.json",
        "benign-transfer-summary.json",
        "benign-cross-concept-transfer-matrix.csv",
        "concept-ranking-agreement.csv",
        "candidate-macro-rankings.csv",
        "trigger-reader-source-roles.csv",
        "benign-cross-concept-transfer.png",
        "benign-cross-concept-transfer.pdf",
        "component-ranking-agreement.png",
        "component-ranking-agreement.pdf",
    ]
    write_manifest(output_dir, names)
    print(f"Analyzed {len(candidate_records)} candidate rows; classification: {classification}")


if __name__ == "__main__":
    main()
