#!/usr/bin/env python3
"""Freeze Day 66 thresholds after calibration and before locked final access."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-66/frozen-title-closure-program.json"
GENERATED_CONTRACT_PATH = ROOT / "results/day-66/frozen-title-closure-final-contract.json"
CALIBRATION_TENSOR_PATH = ROOT / "artifacts/title-closure-v1/calibration/calibration.safetensors"
CALIBRATION_METADATA_PATH = ROOT / "artifacts/title-closure-v1/calibration/calibration.json"
SUMMARY_PATH = ROOT / "results/day-66/calibration-summary.json"
CONTRACT_PATH = ROOT / "results/day-66/frozen-title-closure-execution-contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def threshold(values: torch.Tensor, alpha: float) -> tuple[float, int]:
    ordered = torch.sort(values.double()).values
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return float(ordered[rank - 1]), rank


def main() -> None:
    program, generated = read_json(PROGRAM_PATH), read_json(GENERATED_CONTRACT_PATH)
    program_hash = sha256_file(PROGRAM_PATH)
    if generated["program_sha256"] != program_hash:
        raise RuntimeError("generated final contract parent differs")
    metadata = read_json(CALIBRATION_METADATA_PATH)
    if (
        metadata["program_sha256"] != program_hash
        or metadata["tensor_sha256"] != sha256_file(CALIBRATION_TENSOR_PATH)
    ):
        raise RuntimeError("calibration provenance differs")
    margins = load_file(CALIBRATION_TENSOR_PATH)["normal.margins"].float()
    probe_names = metadata["probe_names_in_order"]
    assignments = list(zip(metadata["assignment_probe_concepts"], metadata["assignment_unique_indices"]))
    thresholds, results = {}, {}
    for concept in program["concepts_in_order"]:
        probe_index = probe_names.index(concept)
        indices = [int(index) for assigned, index in assignments if assigned == concept]
        values = margins[indices, probe_index]
        if len(values) != int(program["calibration"]["negatives_per_probe"]):
            raise RuntimeError(f"calibration assignment count differs: {concept}")
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
            "raw_margin_min": float(values.min()),
            "raw_margin_median": float(values.median()),
            "raw_margin_max": float(values.max()),
            "operating_points": points,
        }
    summary = {
        "schema_version": 1,
        "procedure": "day66-high-powered-calibration-reduction-v1",
        "program_sha256": program_hash,
        "generated_contract_sha256": sha256_file(GENERATED_CONTRACT_PATH),
        "calibration_tensor_sha256": sha256_file(CALIBRATION_TENSOR_PATH),
        "calibration_metadata_sha256": sha256_file(CALIBRATION_METADATA_PATH),
        "thresholds": thresholds,
        "concepts": results,
        "selection_or_gate_changes": False,
    }
    write_json_atomic(SUMMARY_PATH, summary)
    contract = {
        "schema_version": 1,
        "procedure": "day66-calibrated-locked-title-closure-execution-v1",
        "status": "frozen_after_generation_and_calibration_before_any_locked_day66_final_outcome",
        "program_sha256": program_hash,
        "generated_contract_sha256": sha256_file(GENERATED_CONTRACT_PATH),
        "calibration_summary_sha256": sha256_file(SUMMARY_PATH),
        "calibration_artifacts": {
            CALIBRATION_TENSOR_PATH.relative_to(ROOT).as_posix(): sha256_file(CALIBRATION_TENSOR_PATH),
            CALIBRATION_METADATA_PATH.relative_to(ROOT).as_posix(): sha256_file(CALIBRATION_METADATA_PATH),
        },
        "thresholds": thresholds,
        "concepts_in_order": program["concepts_in_order"],
        "final_positive": generated["final_positive"],
        "final_negative": program["roles"]["final_negative"],
        "final_evaluation": program["final_evaluation"],
        "closure_gates": program["closure_gates"],
        "control_verifier_repair": program["control_verifier_repair"],
        "uncertainty": program["uncertainty"],
        "parents": program["parents"],
        "stop_rules": program["stop_rules"],
        "probe_names_in_order": probe_names,
        "pairs": read_json(ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json")["conditions"]["pairs"],
    }
    write_json_atomic(CONTRACT_PATH, contract)
    print(json.dumps({
        "concepts": len(thresholds),
        "calibration_per_concept": int(program["calibration"]["negatives_per_probe"]),
        "execution_contract_sha256": sha256_file(CONTRACT_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
