#!/usr/bin/env python3
"""Merge released-score calibration shards and freeze final execution thresholds."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402
from day69_operational_common import read_json, read_jsonl, sha256_file, verify_program_panels  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-69/frozen-endpoint-correction-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
CALIBRATION_PATH = ROOT / "data/splits/day66-v1/calibration-negative.jsonl"
SHARD_DIR = ROOT / "artifacts/endpoint-correction-v1/calibration/shards"
ARTIFACT_DIR = ROOT / "artifacts/endpoint-correction-v1/calibration"
TENSOR_PATH = ARTIFACT_DIR / "calibration.safetensors"
METADATA_PATH = ARTIFACT_DIR / "calibration.json"
SUMMARY_PATH = ROOT / "results/day-69/score-calibration-summary.json"
CONTRACT_PATH = ROOT / "results/day-69/frozen-endpoint-correction-execution-contract.json"
SHARD_COUNT = 4


def threshold(values: torch.Tensor, alpha: float) -> tuple[float, int]:
    ordered = torch.sort(values.double()).values
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1]), rank


def merge(program_hash: str, panel_hash: str) -> tuple[torch.Tensor, dict[str, Any]]:
    assignments = read_jsonl(CALIBRATION_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in assignments:
        unique.setdefault(row["content_sha256"], row)
    ordered_hashes = sorted(unique)
    merged: dict[str, torch.Tensor] | None = None
    occupied: set[int] = set()
    probe_names = None
    execution_commit = None
    shard_artifacts = {}
    for shard_index in range(SHARD_COUNT):
        stem = f"calibration-shard-{shard_index:02d}-of-{SHARD_COUNT:02d}"
        tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
        metadata = read_json(metadata_path)
        tensors = {key: value.float() for key, value in load_file(tensor_path).items()}
        expected_indices = list(range(shard_index, len(ordered_hashes), SHARD_COUNT))
        if (
            metadata["program_sha256"] != program_hash
            or metadata["panel_sha256"] != panel_hash
            or metadata["tensor_sha256"] != sha256_file(tensor_path)
            or metadata["global_indices"] != expected_indices
            or metadata["unique_content_hashes"] != [ordered_hashes[index] for index in expected_indices]
            or any(len(value) != len(expected_indices) for value in tensors.values())
        ):
            raise RuntimeError(f"score calibration shard provenance differs: {shard_index}")
        if set(tensors) != {"normal.scores", "normal.margins"}:
            raise RuntimeError("score calibration tensor fields differ")
        if probe_names is None:
            probe_names = metadata["probe_names_in_order"]
            execution_commit = metadata["execution_commit"]
            merged = {
                key: torch.empty((len(ordered_hashes), value.shape[1]), dtype=torch.float32)
                for key, value in tensors.items()
            }
        elif metadata["probe_names_in_order"] != probe_names or metadata["execution_commit"] != execution_commit:
            raise RuntimeError("score calibration shard execution differs")
        if occupied.intersection(expected_indices):
            raise RuntimeError("overlapping score calibration shards")
        assert merged is not None
        for key, value in tensors.items():
            merged[key][expected_indices] = value
        occupied.update(expected_indices)
        shard_artifacts[tensor_path.relative_to(ROOT).as_posix()] = sha256_file(tensor_path)
        shard_artifacts[metadata_path.relative_to(ROOT).as_posix()] = sha256_file(metadata_path)
    if (
        occupied != set(range(len(ordered_hashes)))
        or merged is None
        or not all(torch.isfinite(value).all() for value in merged.values())
    ):
        raise RuntimeError("score calibration merge is incomplete or nonfinite")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TENSOR_PATH.with_suffix(".safetensors.tmp")
    save_file({key: value.contiguous() for key, value in merged.items()}, temporary)
    os.replace(temporary, TENSOR_PATH)
    lookup = {digest: index for index, digest in enumerate(ordered_hashes)}
    metadata = {
        "schema_version": 1,
        "procedure": "day69-merged-released-score-calibration-v1",
        "execution_commit": execution_commit,
        "program_sha256": program_hash,
        "panel_sha256": panel_hash,
        "probe_names_in_order": probe_names,
        "unique_content_hashes_in_order": ordered_hashes,
        "assignment_example_ids": [row["example_id"] for row in assignments],
        "assignment_probe_concepts": [row["probe_concept"] for row in assignments],
        "assignment_unique_indices": [lookup[row["content_sha256"]] for row in assignments],
        "shard_artifacts": shard_artifacts,
        "tensor_sha256": sha256_file(TENSOR_PATH),
    }
    write_json_atomic(METADATA_PATH, metadata)
    return merged["normal.scores"], metadata


def main() -> None:
    program = read_json(PROGRAM_PATH)
    verify_program_panels(program)
    program_hash = sha256_file(PROGRAM_PATH)
    panel_hash = sha256_file(CALIBRATION_PATH)
    scores, metadata = merge(program_hash, panel_hash)
    probe_names = metadata["probe_names_in_order"]
    assignments = list(zip(metadata["assignment_probe_concepts"], metadata["assignment_unique_indices"]))
    thresholds, results = {}, {}
    for concept in program["concepts_in_order"]:
        probe_index = probe_names.index(concept)
        indices = [int(index) for assigned, index in assignments if assigned == concept]
        values = scores[indices, probe_index]
        if len(values) != int(program["calibration"]["negatives_per_probe"]):
            raise RuntimeError(f"score calibration assignment count differs: {concept}")
        thresholds[concept], points = {}, {}
        for nominal, alpha in (("0.01", 0.01), ("0.05", 0.05)):
            value, rank = threshold(values, alpha)
            thresholds[concept][nominal] = value
            points[nominal] = {
                "threshold": value,
                "order_statistic_rank_1_indexed": rank,
                "calibration_exceedance_rate": float((values > value).float().mean()),
            }
        results[concept] = {
            "n": len(values),
            "released_score_min": float(values.min()),
            "released_score_median": float(values.median()),
            "released_score_max": float(values.max()),
            "operating_points": points,
        }
    summary = {
        "schema_version": 1,
        "procedure": "day69-released-score-calibration-reduction-v1",
        "program_sha256": program_hash,
        "calibration_tensor_sha256": sha256_file(TENSOR_PATH),
        "calibration_metadata_sha256": sha256_file(METADATA_PATH),
        "thresholds": thresholds,
        "concepts": results,
        "selection_or_gate_changes": False,
    }
    write_json_atomic(SUMMARY_PATH, summary)
    day57 = read_json(DAY57_PATH)
    contract = {
        "schema_version": 1,
        "procedure": "day69-calibrated-released-score-endpoint-correction-v1",
        "status": "frozen_after_released_score_calibration_before_any_new_panel_score_outcome",
        "program_sha256": program_hash,
        "calibration_summary_sha256": sha256_file(SUMMARY_PATH),
        "calibration_artifacts": {
            TENSOR_PATH.relative_to(ROOT).as_posix(): sha256_file(TENSOR_PATH),
            METADATA_PATH.relative_to(ROOT).as_posix(): sha256_file(METADATA_PATH),
        },
        "thresholds": thresholds,
        "concepts_in_order": program["concepts_in_order"],
        "panels": program["panels"],
        "prospective_scientific_gate": program["prospective_scientific_gate"],
        "primary_estimands": program["primary_estimands"],
        "carried_prior_clauses": program["carried_prior_clauses"],
        "stop_rules": program["stop_rules"],
        "probe_names_in_order": probe_names,
        "pairs": day57["conditions"]["pairs"],
        "k12_component_ids": day57["k12"]["component_ids"],
    }
    write_json_atomic(CONTRACT_PATH, contract)
    print(
        json.dumps(
            {
                "unique_rows": len(metadata["unique_content_hashes_in_order"]),
                "calibration_per_concept": int(program["calibration"]["negatives_per_probe"]),
                "execution_contract_sha256": sha256_file(CONTRACT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

