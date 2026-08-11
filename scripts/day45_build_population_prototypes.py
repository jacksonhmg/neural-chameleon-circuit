#!/usr/bin/env python3
"""Build all target-excluded concept/position prototypes for Day 45."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-45/frozen-prototype-population-contract.json"
DAY44_CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
CAPTURE_DIR = ROOT / "artifacts/mechanism-gate1-v1/chameleon"
ARTIFACT_DIR = ROOT / "artifacts/rapid-k12-v1"
TENSOR_PATH = ARTIFACT_DIR / "population-prototypes.safetensors"
MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


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
    for path in (Path(__file__).resolve(), CONTRACT_PATH, DAY44_CONTRACT_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    day44 = read_json(DAY44_CONTRACT_PATH)
    if contract["parents"]["day44_contract_sha256"] != sha256_file(
        DAY44_CONTRACT_PATH
    ):
        raise RuntimeError("Day 45 parent contract hash differs")
    component_ids = tuple(day44["component_set"])
    values: dict[str, dict[str, Any]] = {}
    by_concept: dict[str, list[str]] = {}
    for path in sorted(CAPTURE_DIR.glob("*.pt")):
        capture = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(capture["k12_head_ids"]) != component_ids:
            raise RuntimeError(f"component mismatch in {path}")
        writer = capture["k12_delta"].float()
        bins = response_deciles(len(writer))
        means = torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        weights = torch.zeros(10, dtype=torch.float64)
        for decile in range(10):
            selected = bins == decile
            if selected.any():
                means[decile] = writer[selected].double().mean(dim=0)
                weights[decile] = float(selected.sum()) / len(writer)
        example_id = capture["example_id"]
        values[example_id] = {
            "concept": capture["concept"],
            "split": capture["split"],
            "means": means,
            "weights": weights,
            "token_count": len(writer),
        }
        by_concept.setdefault(capture["concept"], []).append(example_id)
    if len(values) != 866 or len(by_concept) != 13:
        raise RuntimeError("population captures differ from the frozen contract")

    concept_means = {}
    for concept, example_ids in by_concept.items():
        total = torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        denominator = torch.zeros(10, dtype=torch.float64)
        for example_id in example_ids:
            row = values[example_id]
            total += row["means"] * row["weights"][:, None, None]
            denominator += row["weights"]
        concept_means[concept] = total / denominator.clamp(min=1e-12)[:, None, None]

    example_ids = sorted(values)
    prototypes = []
    rows = []
    for example_id in example_ids:
        target = values[example_id]
        total = torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        denominator = torch.zeros(10, dtype=torch.float64)
        for source_id in by_concept[target["concept"]]:
            if source_id == example_id:
                continue
            source = values[source_id]
            total += source["means"] * source["weights"][:, None, None]
            denominator += source["weights"]
        fallback = torch.stack(
            [
                mean
                for concept, mean in concept_means.items()
                if concept != target["concept"]
            ]
        ).mean(dim=0)
        prototype = torch.where(
            (denominator > 0)[:, None, None],
            total / denominator.clamp(min=1e-12)[:, None, None],
            fallback,
        )
        prototypes.append(prototype.float())
        rows.append(
            {
                "example_id": example_id,
                "concept": target["concept"],
                "split": target["split"],
                "response_token_count": target["token_count"],
                "source_examples": len(by_concept[target["concept"]]) - 1,
                "fallback_deciles": int((denominator == 0).sum()),
            }
        )
    tensor = torch.stack(prototypes).contiguous()
    if tensor.shape != (866, 10, 12, 256) or not torch.isfinite(tensor).all():
        raise RuntimeError("population prototype tensor failed shape or finiteness")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_file({"prototype_delta": tensor}, TENSOR_PATH)
    manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-population-prototypes-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "tensor_path": str(TENSOR_PATH.relative_to(ROOT)),
        "tensor_sha256": sha256_file(TENSOR_PATH),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "component_ids": list(component_ids),
        "examples": rows,
        "checks": {
            "exact_examples": len(rows) == 866,
            "exact_concepts": len(by_concept) == 13,
            "exact_shape": list(tensor.shape) == [866, 10, 12, 256],
            "all_finite": bool(torch.isfinite(tensor).all()),
            "no_fallback_used": all(row["fallback_deciles"] == 0 for row in rows),
            "every_target_excluded": all(
                row["source_examples"] == len(by_concept[row["concept"]]) - 1
                and row["source_examples"] > 0
                for row in rows
            ),
        },
    }
    manifest["result"] = "pass" if all(manifest["checks"].values()) else "fail"
    write_json_atomic(MANIFEST_PATH, manifest)
    if manifest["result"] != "pass":
        raise RuntimeError(json.dumps(manifest["checks"], indent=2))
    print(
        json.dumps(
            {
                "result": "pass",
                "tensor_shape": manifest["tensor_shape"],
                "tensor_sha256": manifest["tensor_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
