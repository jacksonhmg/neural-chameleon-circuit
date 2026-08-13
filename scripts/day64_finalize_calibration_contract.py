#!/usr/bin/env python3
"""Reduce calibration/development artifacts and freeze final thresholds/prototypes."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-64/frozen-trained-title-gate-program.json"
CALIBRATION_PATH = ROOT / "data/splits/day64-v1/calibration-negative.jsonl"
ARTIFACT_DIR = ROOT / "artifacts/trained-final-title-gate-v1/calibration-development"
PROTOTYPE_PATH = ROOT / "artifacts/trained-final-title-gate-v1/development-prototypes.safetensors"
SUMMARY_PATH = ROOT / "results/day-64/calibration-development-summary.json"
CONTRACT_PATH = ROOT / "results/day-64/frozen-trained-final-contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def checked(stem: str, program_hash: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensor_path, metadata_path = ARTIFACT_DIR / f"{stem}.safetensors", ARTIFACT_DIR / f"{stem}.json"
    metadata = read_json(metadata_path)
    if metadata["program_sha256"] != program_hash or metadata["tensor_sha256"] != sha256_file(tensor_path):
        raise RuntimeError(f"artifact provenance differs: {stem}")
    return load_file(tensor_path), metadata


def threshold(values: torch.Tensor, alpha: float) -> tuple[float, int]:
    ordered = torch.sort(values.double()).values
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1]), rank


def main() -> None:
    program = read_json(PROGRAM_PATH)
    program_hash = sha256_file(PROGRAM_PATH)
    calibration, metadata = checked("calibration", program_hash)
    margins = calibration["normal.margins"].float()
    probe_names = program["probe_names_in_order"]
    assignments = list(zip(
        metadata["assignment_example_ids"],
        metadata["assignment_probe_concepts"],
        metadata["assignment_unique_indices"],
    ))
    calibration_results: dict[str, Any] = {}
    frozen_thresholds: dict[str, dict[str, float]] = {}
    for concept in program["concepts_in_order"]:
        probe_index = probe_names.index(concept)
        indices = [int(index) for _example, assigned, index in assignments if assigned == concept]
        values = margins[indices, probe_index]
        if len(values) != 128:
            raise RuntimeError(f"calibration count differs for {concept}")
        frozen_thresholds[concept] = {}
        points = {}
        for nominal, alpha in (("0.01", 0.01), ("0.05", 0.05)):
            value, rank = threshold(values, alpha)
            frozen_thresholds[concept][nominal] = value
            points[nominal] = {
                "threshold": value,
                "order_statistic_rank_1_indexed": rank,
                "calibration_exceedance_rate": float((values > value).float().mean()),
            }
        calibration_results[concept] = {
            "n": len(values),
            "raw_margin_min": float(values.min()),
            "raw_margin_median": float(values.median()),
            "raw_margin_max": float(values.max()),
            "operating_points": points,
        }

    prototypes: dict[str, torch.Tensor] = {}
    development_files = []
    for concept in program["concepts_in_order"]:
        tensors, dev_metadata = checked(f"development-{concept}", program_hash)
        development_files.extend([
            ARTIFACT_DIR / f"development-{concept}.safetensors",
            ARTIFACT_DIR / f"development-{concept}.json",
        ])
        for direction, donor, target in (
            ("correct_to_irrelevant", "correct", "irrelevant"),
            ("irrelevant_to_correct", "irrelevant", "correct"),
        ):
            for field in ("kv", "k12", "margins"):
                prototypes[f"{concept}.{direction}.{field}"] = (
                    tensors[f"natural.{donor}.{field}"].float()
                    - tensors[f"natural.{target}.{field}"].float()
                ).mean(0).contiguous()
        if dev_metadata["source_operator"] != "full_monitoring_prefix_kv":
            raise RuntimeError("development source operator differs")
    PROTOTYPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROTOTYPE_PATH.with_suffix(".safetensors.tmp")
    save_file(prototypes, temporary)
    temporary.replace(PROTOTYPE_PATH)

    artifact_files = [
        ARTIFACT_DIR / "calibration.safetensors",
        ARTIFACT_DIR / "calibration.json",
        *development_files,
    ]
    summary = {
        "schema_version": 1,
        "procedure": "day64-calibration-development-reduction-v1",
        "program_sha256": program_hash,
        "calibration": calibration_results,
        "frozen_thresholds": frozen_thresholds,
        "development_prototype": {
            "path": PROTOTYPE_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(PROTOTYPE_PATH),
            "keys": sorted(prototypes),
        },
        "artifact_hashes": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in artifact_files
        },
        "selection_or_gate_changes": False,
    }
    write_json_atomic(SUMMARY_PATH, summary)
    contract = {
        "schema_version": 1,
        "procedure": "day64-trained-final-causal-chain-contract-v1",
        "status": "frozen_after_calibration_before_any_locked_final_model_or_intervention_outcome",
        "program_path": PROGRAM_PATH.relative_to(ROOT).as_posix(),
        "program_sha256": program_hash,
        "calibration_summary_sha256": sha256_file(SUMMARY_PATH),
        "calibration_artifact_hashes": summary["artifact_hashes"],
        "prototype": summary["development_prototype"],
        "calibration_thresholds": frozen_thresholds,
        "concepts_in_order": program["concepts_in_order"],
        "probe_names_in_order": probe_names,
        "pairs": program["pairs"],
        "roles": program["roles"],
        "causal_matrix": program["causal_matrix"],
        "gates": program["gates"],
        "reductions": program["reductions"],
        "implementation_gates": program["implementation_gates"],
        "claim_scope": program["claim_scope"],
        "stop_rules": program["stop_rules"],
    }
    write_json_atomic(CONTRACT_PATH, contract)
    print(json.dumps({
        "program_sha256": program_hash,
        "prototype_sha256": summary["development_prototype"]["sha256"],
        "calibration_concepts": len(frozen_thresholds),
        "final_contract_sha256": sha256_file(CONTRACT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
