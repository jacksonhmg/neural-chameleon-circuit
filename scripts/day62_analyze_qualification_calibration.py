#!/usr/bin/env python3
"""Reduce qualification, freeze predictions, and calibrate operating thresholds."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.final_title_gate import finite_sample_upper_threshold, select_qualifying_pairs  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
PROBE_SUMMARY_PATH = ROOT / "results/day-62/new-probe-training-summary.json"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/qualification-calibration-shards"
PROTOTYPE_PATH = ROOT / "artifacts/final-title-gate-v1/qualification-prototypes.safetensors"
SUMMARY_PATH = ROOT / "results/day-62/qualification-calibration-summary.json"
MANIFEST_PATH = ROOT / "results/day-62/qualification-calibration-artifact-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    contract, probes = read_json(CONTRACT_PATH), read_json(PROBE_SUMMARY_PATH)
    concepts = list(contract["concepts"])
    pair_summaries, prototype_tensors, qualification = {}, {}, {}
    for pair_id in contract["candidate_pairs_in_order"]:
        metadata_path = SHARD_DIR / f"qualification-{pair_id}.json"
        tensor_path = SHARD_DIR / f"qualification-{pair_id}.safetensors"
        metadata = read_json(metadata_path)
        if metadata["contract_sha256"] != sha256_file(CONTRACT_PATH) or metadata["tensor_sha256"] != sha256_file(tensor_path):
            raise RuntimeError(f"qualification provenance differs: {pair_id}")
        tensors = load_file(tensor_path)
        a, b = metadata["concept_a"], metadata["concept_b"]
        ia, ib = concepts.index(a), concepts.index(b)
        margins_a, margins_b = tensors[f"{a}.margins"], tensors[f"{b}.margins"]
        release_a = (margins_b[:, ia] - margins_a[:, ia]).tolist()
        release_b = (margins_a[:, ib] - margins_b[:, ib]).tolist()
        metrics = {
            "concept_a": a,
            "concept_b": b,
            "median_release_a": statistics.median(release_a),
            "median_release_b": statistics.median(release_b),
            "positive_release_examples_a": sum(value > 0 for value in release_a),
            "positive_release_examples_b": sum(value > 0 for value in release_b),
        }
        for direction, target, donor in ((f"{a}_to_{b}", a, b), (f"{b}_to_{a}", b, a)):
            for field in ("kv", "k12", "margins"):
                delta = tensors[f"{donor}.{field}"].float() - tensors[f"{target}.{field}"].float()
                prototype_tensors[f"{pair_id}.{direction}.{field}"] = delta.mean(dim=0).contiguous()
        prototype_nonzero = all(
            float(torch.linalg.vector_norm(value.float())) > 1e-8
            for key, value in prototype_tensors.items()
            if key.startswith(pair_id + ".")
        )
        gate = contract["qualification"]["pair_gate"]
        passes = (
            probes["concepts"][a]["quality_gate_pass"]
            and probes["concepts"][b]["quality_gate_pass"]
            and metrics["median_release_a"] >= float(gate["median_standardized_own_margin_release_each_concept_min"])
            and metrics["median_release_b"] >= float(gate["median_standardized_own_margin_release_each_concept_min"])
            and metrics["positive_release_examples_a"] >= int(gate["expected_direction_examples_each_concept_min"])
            and metrics["positive_release_examples_b"] >= int(gate["expected_direction_examples_each_concept_min"])
            and prototype_nonzero
        )
        metrics.update({"prototype_nonzero": prototype_nonzero, "qualifies": passes})
        pair_summaries[pair_id], qualification[pair_id] = metrics, passes
    selected = select_qualifying_pairs(
        contract["candidate_pairs_in_order"], qualification, count=int(contract["qualification"]["minimum_selected_pairs"])
    )
    thresholds = {}
    for concept in concepts:
        metadata_path = SHARD_DIR / f"calibration-{concept}.json"
        tensor_path = SHARD_DIR / f"calibration-{concept}.safetensors"
        metadata = read_json(metadata_path)
        if metadata["tensor_sha256"] != sha256_file(tensor_path):
            raise RuntimeError(f"calibration tensor differs: {concept}")
        values = load_file(tensor_path)["normal.margins"][:, concepts.index(concept)].tolist()
        thresholds[concept] = {
            str(fpr): finite_sample_upper_threshold(values, float(fpr))
            for fpr in contract["calibration"]["false_positive_rates"]
        }
    PROTOTYPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(prototype_tensors, PROTOTYPE_PATH)
    summary = {
        "schema_version": 1,
        "procedure": "day62-new-concept-qualification-calibration-reduction-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "probe_summary_sha256": sha256_file(PROBE_SUMMARY_PATH),
        "pairs": pair_summaries,
        "selected_pairs": list(selected),
        "selected_pair_count": len(selected),
        "qualification_continue_gate_pass": len(selected) >= int(contract["qualification"]["minimum_selected_pairs"]),
        "calibration_thresholds": thresholds,
        "prototype_path": PROTOTYPE_PATH.relative_to(ROOT).as_posix(),
        "prototype_sha256": sha256_file(PROTOTYPE_PATH),
    }
    write_json(SUMMARY_PATH, summary)
    files = [*sorted(SHARD_DIR.glob("*")), PROTOTYPE_PATH, SUMMARY_PATH]
    write_json(MANIFEST_PATH, {
        "schema_version": 1,
        "procedure": "day62-qualification-calibration-artifact-manifest-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files if path.is_file()
        ],
    })
    print(json.dumps({"selected_pairs": list(selected), "continue": summary["qualification_continue_gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
