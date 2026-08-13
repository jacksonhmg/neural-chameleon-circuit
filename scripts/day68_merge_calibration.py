#!/usr/bin/env python3
"""Merge deterministic calibration shards into the frozen global order."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-68/frozen-replacement-generation-program.json"
CALIBRATION_PATH = ROOT / "data/splits/day66-v1/calibration-negative.jsonl"
SHARD_DIR = ROOT / "artifacts/title-closure-v2/calibration/shards"
ARTIFACT_DIR = ROOT / "artifacts/title-closure-v2/calibration"
TENSOR_PATH = ARTIFACT_DIR / "calibration.safetensors"
METADATA_PATH = ARTIFACT_DIR / "calibration.json"
SHARD_COUNT = 4


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    program_hash = sha256_file(PROGRAM_PATH)
    panel_hash = sha256_file(CALIBRATION_PATH)
    assignments = read_jsonl(CALIBRATION_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in assignments:
        unique.setdefault(row["content_sha256"], row)
    ordered_hashes = sorted(unique)
    merged, occupied, probe_names, execution_commit = None, set(), None, None
    shard_artifacts = {}
    for shard_index in range(SHARD_COUNT):
        stem = f"calibration-shard-{shard_index:02d}-of-{SHARD_COUNT:02d}"
        tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
        metadata = read_json(metadata_path)
        tensor = load_file(tensor_path)["normal.margins"].float()
        expected_indices = list(range(shard_index, len(ordered_hashes), SHARD_COUNT))
        if (
            metadata["program_sha256"] != program_hash
            or metadata["panel_sha256"] != panel_hash
            or metadata["tensor_sha256"] != sha256_file(tensor_path)
            or metadata["global_indices"] != expected_indices
            or metadata["unique_content_hashes"] != [ordered_hashes[index] for index in expected_indices]
            or len(tensor) != len(expected_indices)
        ):
            raise RuntimeError(f"calibration shard provenance differs: {shard_index}")
        if probe_names is None:
            probe_names = metadata["probe_names_in_order"]
            execution_commit = metadata["execution_commit"]
            merged = torch.empty((len(ordered_hashes), tensor.shape[1]), dtype=torch.float32)
        elif metadata["probe_names_in_order"] != probe_names or metadata["execution_commit"] != execution_commit:
            raise RuntimeError("calibration shard execution differs")
        if occupied.intersection(expected_indices):
            raise RuntimeError("overlapping calibration shards")
        merged[expected_indices] = tensor
        occupied.update(expected_indices)
        shard_artifacts[tensor_path.relative_to(ROOT).as_posix()] = sha256_file(tensor_path)
        shard_artifacts[metadata_path.relative_to(ROOT).as_posix()] = sha256_file(metadata_path)
    if occupied != set(range(len(ordered_hashes))) or merged is None or not torch.isfinite(merged).all():
        raise RuntimeError("calibration merge is incomplete or nonfinite")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TENSOR_PATH.with_suffix(".safetensors.tmp")
    save_file({"normal.margins": merged.contiguous()}, temporary)
    os.replace(temporary, TENSOR_PATH)
    lookup = {digest: index for index, digest in enumerate(ordered_hashes)}
    write_json_atomic(METADATA_PATH, {
        "schema_version": 1,
        "procedure": "day68-merged-high-powered-calibration-v1",
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
    })
    print(json.dumps({
        "unique_rows": len(ordered_hashes),
        "assignments": len(assignments),
        "probe_count": len(probe_names),
        "tensor_sha256": sha256_file(TENSOR_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
