#!/usr/bin/env python3
"""Mechanically reduce the one-pass checkpoint hybridization."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.final_title_gate import vector_relation  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/checkpoint-hybridization-shards"
SUMMARY_PATH = ROOT / "results/day-62/checkpoint-hybridization-summary.json"
MANIFEST_PATH = ROOT / "results/day-62/checkpoint-hybridization-artifact-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    contract = read_json(CONTRACT_PATH)
    states = contract["hybridization"]["states_in_order"]
    per_state: dict[str, dict[str, list[float]]] = {
        state: {field: [] for field in ("margins", "k12", "kv")} for state in states
    }
    files = []
    for metadata_path in sorted(SHARD_DIR.glob("*.json")):
        tensor_path = metadata_path.with_suffix(".safetensors")
        metadata = read_json(metadata_path)
        if metadata["contract_sha256"] != sha256_file(CONTRACT_PATH) or metadata["tensor_sha256"] != sha256_file(tensor_path):
            raise RuntimeError("hybrid shard provenance differs")
        tensors = load_file(tensor_path)
        for field in ("margins", "k12", "kv"):
            reference = tensors[f"chameleon.correct.{field}"].float() - tensors[f"chameleon.irrelevant.{field}"].float()
            for state in states:
                effect = tensors[f"{state}.correct.{field}"].float() - tensors[f"{state}.irrelevant.{field}"].float()
                for row in range(effect.shape[0]):
                    per_state[state][field].append(vector_relation(effect[row], reference[row])["aligned_recovery"])
        files.extend((metadata_path, tensor_path))
    summaries = {
        state: {
            field: {
                "median_recovery": statistics.median(values),
                "loss": 1.0 - statistics.median(values),
            }
            for field, values in fields.items()
        }
        for state, fields in per_state.items()
    }
    primary_loss = {
        state: min(summaries[state][field]["loss"] for field in ("margins", "k12", "kv"))
        for state in ("early_representation", "k12_kv_readout", "tail_attention_complement")
    }
    spec = contract["hybridization"]["classification"]
    ordered = sorted(primary_loss, key=lambda state: primary_loss[state], reverse=True)
    if (
        primary_loss[ordered[0]] >= float(spec["single_group_primary_loss_min"])
        and primary_loss[ordered[0]] - primary_loss[ordered[1]] >= float(spec["single_group_lead_min"])
    ):
        classification = f"primary:{ordered[0]}"
    elif sum(loss >= float(spec["material_group_loss_min"]) for loss in primary_loss.values()) >= 2:
        classification = "distributed_across_groups"
    elif min(summaries["distributed_combination"][field]["loss"] for field in ("margins", "k12", "kv")) >= float(spec["combination_loss_min"]):
        classification = "distributed_combination_only"
    else:
        classification = "coarse_groups_incomplete"
    summary = {
        "schema_version": 1,
        "procedure": "day62-timeboxed-checkpoint-hybridization-reduction-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "states": summaries,
        "primary_group_worst_endpoint_losses": primary_loss,
        "classification": classification,
        "timebox_disposition": "stop_after_coarse_localization_no_refinement",
        "cannot_change_final_operator": True,
    }
    write_json(SUMMARY_PATH, summary)
    files.append(SUMMARY_PATH)
    write_json(MANIFEST_PATH, {
        "schema_version": 1,
        "procedure": "day62-checkpoint-hybridization-artifact-manifest-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    })
    print(json.dumps({"classification": classification, "losses": primary_loss}, indent=2))


if __name__ == "__main__":
    main()
