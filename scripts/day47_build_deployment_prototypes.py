#!/usr/bin/env python3
"""Build fixed development-bank prototypes for unseen Day 47 targets."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
PANEL_MANIFEST_PATH = ROOT / "data/splits/day47-v1/manifest.json"
DAY44_CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
DAY45_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
CAPTURE_DIR = ROOT / "artifacts/mechanism-gate1-v1/chameleon"
TENSOR_PATH = ROOT / "artifacts/rapid-k12-v1/day47-heldout-prototypes.safetensors"
MANIFEST_PATH = ROOT / "results/day-47/heldout-prototype-tensors.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from execution commit {commit}")


def response_deciles(token_count: int) -> torch.Tensor:
    positions = torch.arange(token_count)
    return torch.clamp((10 * positions) // max(token_count - 1, 1), max=9)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        PANEL_MANIFEST_PATH,
        DAY44_CONTRACT_PATH,
        DAY45_MANIFEST_PATH,
    ):
        require_committed(path, commit)
    panel = json.loads(PANEL_MANIFEST_PATH.read_text())
    day44 = json.loads(DAY44_CONTRACT_PATH.read_text())
    day45 = json.loads(DAY45_MANIFEST_PATH.read_text())
    concepts = tuple(panel["selection"]["concepts"])
    component_ids = tuple(day44["component_set"])
    if tuple(day45["component_ids"]) != component_ids:
        raise RuntimeError("selected components differ from Day 45")

    totals = {
        concept: torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        for concept in concepts
    }
    denominators = {
        concept: torch.zeros(10, dtype=torch.float64) for concept in concepts
    }
    counts = {concept: 0 for concept in concepts}
    for path in sorted(CAPTURE_DIR.glob("*.pt")):
        capture = torch.load(path, map_location="cpu", weights_only=True)
        concept = capture["concept"]
        if concept not in totals:
            continue
        if tuple(capture["k12_head_ids"]) != component_ids:
            raise RuntimeError(f"component mismatch in {path}")
        writer = capture["k12_delta"].float()
        deciles = response_deciles(len(writer))
        for decile in range(10):
            selected = deciles == decile
            if selected.any():
                weight = float(selected.sum()) / len(writer)
                totals[concept][decile] += writer[selected].double().mean(dim=0) * weight
                denominators[concept][decile] += weight
        counts[concept] += 1
    if any(count != 64 for count in counts.values()):
        raise RuntimeError(f"development-bank source counts differ: {counts}")
    if any(not torch.all(value > 0) for value in denominators.values()):
        raise RuntimeError("one or more concept/decile cells are empty")

    tensor = torch.stack(
        [totals[concept] / denominators[concept][:, None, None] for concept in concepts]
    ).float().contiguous()
    expected_shape = [len(concepts), 10, len(component_ids), 256]
    if list(tensor.shape) != expected_shape or not torch.isfinite(tensor).all():
        raise RuntimeError("held-out prototype tensor failed shape or finiteness")
    TENSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file({"prototype_delta": tensor}, TENSOR_PATH)
    manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-heldout-deployment-prototypes-v1",
        "execution_commit": commit,
        "evidence_role": (
            "fixed operator parameters estimated only from the original Day 4 positive "
            "development bank; no Day 47 target contributes"
        ),
        "panel_manifest_sha256": sha256_file(PANEL_MANIFEST_PATH),
        "day44_contract_sha256": sha256_file(DAY44_CONTRACT_PATH),
        "day45_manifest_sha256": sha256_file(DAY45_MANIFEST_PATH),
        "tensor_path": str(TENSOR_PATH.relative_to(ROOT)),
        "tensor_sha256": sha256_file(TENSOR_PATH),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "concepts": list(concepts),
        "component_ids": list(component_ids),
        "source_examples_per_concept": counts,
        "checks": {
            "exact_shape": list(tensor.shape) == expected_shape,
            "all_finite": bool(torch.isfinite(tensor).all()),
            "exact_concepts": len(concepts) == 11,
            "exact_components": len(component_ids) == 12,
            "all_sources_from_original_bank": all(
                count == 64 for count in counts.values()
            ),
            "day47_targets_excluded_by_content_freeze": panel["checks"][
                "content_disjoint_from_day04"
            ],
        },
    }
    manifest["result"] = "pass" if all(manifest["checks"].values()) else "fail"
    write_json_atomic(MANIFEST_PATH, manifest)
    if manifest["result"] != "pass":
        raise RuntimeError(json.dumps(manifest["checks"], indent=2))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
