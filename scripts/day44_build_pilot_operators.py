#!/usr/bin/env python3
"""Build frozen leave-one-example-out K12 prototype tensors for Day 44."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
SELECTION_PATH = ROOT / "results/day-44/pilot-selection.json"
SPEC_PATH = ROOT / "results/day-44/frozen-pilot-operator-spec.json"
CAPTURE_DIR = ROOT / "artifacts/mechanism-gate1-v1/chameleon"
TENSOR_PATH = ROOT / "results/day-44/pilot-operator-tensors.safetensors"
MANIFEST_PATH = ROOT / "results/day-44/pilot-operator-tensors.json"


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH, SELECTION_PATH, SPEC_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    selection = read_json(SELECTION_PATH)
    spec = read_json(SPEC_PATH)
    if selection["promoted"] != ["tangential_actual_activity"]:
        raise RuntimeError("pilot selection differs from the frozen operator build")
    if spec["parent_contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("pilot operator spec has the wrong parent contract")

    component_ids = tuple(contract["component_set"])
    pilot_rows = contract["pilot"]["examples"]
    pilot_ids = [row["example_id"] for row in pilot_rows]
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
            "means": means,
            "weights": weights,
            "token_count": len(writer),
        }
        by_concept.setdefault(capture["concept"], []).append(example_id)
    if len(values) != 866 or len(by_concept) != 13:
        raise RuntimeError("capture population does not match the frozen population")

    concept_means: dict[str, torch.Tensor] = {}
    for concept, example_ids in by_concept.items():
        total = torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        denominator = torch.zeros(10, dtype=torch.float64)
        for example_id in example_ids:
            row = values[example_id]
            total += row["means"] * row["weights"][:, None, None]
            denominator += row["weights"]
        concept_means[concept] = total / denominator.clamp(min=1e-12)[:, None, None]

    prototypes = []
    audit_rows = []
    for frozen_row in pilot_rows:
        example_id = frozen_row["example_id"]
        target = values[example_id]
        if target["concept"] != frozen_row["concept"]:
            raise RuntimeError(f"concept mismatch for {example_id}")
        if target["token_count"] != frozen_row["response_token_count"]:
            raise RuntimeError(f"token-count mismatch for {example_id}")
        concept_ids = by_concept[target["concept"]]
        total = torch.zeros((10, len(component_ids), 256), dtype=torch.float64)
        denominator = torch.zeros(10, dtype=torch.float64)
        for source_id in concept_ids:
            if source_id == example_id:
                continue
            source = values[source_id]
            total += source["means"] * source["weights"][:, None, None]
            denominator += source["weights"]
        other_concepts = [
            mean for concept, mean in concept_means.items() if concept != target["concept"]
        ]
        fallback = torch.stack(other_concepts).mean(dim=0)
        prototype = torch.where(
            (denominator > 0)[:, None, None],
            total / denominator.clamp(min=1e-12)[:, None, None],
            fallback,
        )
        prototypes.append(prototype.float())
        audit_rows.append(
            {
                "example_id": example_id,
                "concept": target["concept"],
                "source_examples": len(concept_ids) - 1,
                "fallback_deciles": int((denominator == 0).sum()),
                "response_token_count": target["token_count"],
            }
        )

    tensor = torch.stack(prototypes).contiguous()
    if tensor.shape != (26, 10, 12, 256) or not torch.isfinite(tensor).all():
        raise RuntimeError("prototype tensor shape or finiteness failed")
    save_file(
        {"prototype_delta": tensor},
        TENSOR_PATH,
        metadata={
            "procedure": "rapid-k12-pilot-operators-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
        },
    )
    manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-pilot-operator-tensors-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "operator_spec_sha256": sha256_file(SPEC_PATH),
        "tensor_path": str(TENSOR_PATH.relative_to(ROOT)),
        "tensor_sha256": sha256_file(TENSOR_PATH),
        "tensor_shape": list(tensor.shape),
        "tensor_dtype": str(tensor.dtype),
        "component_ids": list(component_ids),
        "example_ids": pilot_ids,
        "examples": audit_rows,
        "checks": {
            "exact_population": len(values) == 866,
            "exact_concepts": len(by_concept) == 13,
            "exact_pilot_examples": len(pilot_ids) == 26,
            "exact_tensor_shape": list(tensor.shape) == [26, 10, 12, 256],
            "all_finite": bool(torch.isfinite(tensor).all()),
            "no_fallback_used": all(row["fallback_deciles"] == 0 for row in audit_rows),
        },
    }
    manifest["result"] = (
        "pass" if all(manifest["checks"].values()) else "fail"
    )
    write_json_atomic(MANIFEST_PATH, manifest)
    if manifest["result"] != "pass":
        raise RuntimeError(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "result": "pass",
                "tensor_sha256": manifest["tensor_sha256"],
                "tensor_shape": manifest["tensor_shape"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
