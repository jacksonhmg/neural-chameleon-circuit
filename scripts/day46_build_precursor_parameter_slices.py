#!/usr/bin/env python3
"""Extract only the frozen precursor attention slices needed for Day 46."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-46/frozen-selected-parameter-swap-contract.json"
ARTIFACT_DIR = ROOT / "artifacts/rapid-k12-v1"
TENSOR_PATH = ARTIFACT_DIR / "precursor-parameter-slices.safetensors"
MANIFEST_PATH = ROOT / "results/day-46/precursor-parameter-slices.json"


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


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sliced_parameter(
    tensor: torch.Tensor,
    projection: str,
    heads: list[int],
    head_dim: int,
) -> torch.Tensor:
    if projection == "o_proj":
        return torch.cat(
            [tensor[:, head * head_dim : (head + 1) * head_dim] for head in heads],
            dim=1,
        ).contiguous()
    return torch.cat(
        [tensor[head * head_dim : (head + 1) * head_dim] for head in heads],
        dim=0,
    ).contiguous()


def main() -> None:
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    chameleon_path = ROOT / contract["models"]["chameleon"]["source_shard"]
    precursor_path = ROOT / contract["models"]["exact_precursor"]["source_shard"]
    if sha256_file(chameleon_path) != contract["models"]["chameleon"][
        "source_shard_sha256"
    ]:
        raise RuntimeError("Chameleon source shard hash differs")
    if sha256_file(precursor_path) != contract["models"]["exact_precursor"][
        "source_shard_sha256"
    ]:
        raise RuntimeError("precursor source shard hash differs")
    head_dim = int(contract["architecture"]["head_dim"])
    hidden_size = int(contract["architecture"]["hidden_size"])
    tensors: dict[str, torch.Tensor] = {}
    rows = []
    with safe_open(precursor_path, framework="pt", device="cpu") as precursor, safe_open(
        chameleon_path, framework="pt", device="cpu"
    ) as chameleon:
        for head_set in ("selected", "matched_control"):
            for layer in contract["architecture"]["layers"]:
                indices = contract["head_sets"][head_set][str(layer)]
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    head_kind = "key_value" if projection in {"k_proj", "v_proj"} else "query"
                    heads = list(indices[head_kind])
                    source_name = f"model.layers.{layer}.self_attn.{projection}.weight"
                    precursor_full = precursor.get_tensor(source_name)
                    chameleon_full = chameleon.get_tensor(source_name)
                    precursor_slice = sliced_parameter(
                        precursor_full, projection, heads, head_dim
                    )
                    chameleon_slice = sliced_parameter(
                        chameleon_full, projection, heads, head_dim
                    )
                    key = f"{head_set}.layer_{layer:02d}.{projection}"
                    tensors[key] = precursor_slice
                    difference = precursor_slice.float() - chameleon_slice.float()
                    denominator = max(
                        float(torch.linalg.vector_norm(chameleon_slice.float())), 1e-12
                    )
                    expected_shape = (
                        [hidden_size, len(heads) * head_dim]
                        if projection == "o_proj"
                        else [len(heads) * head_dim, hidden_size]
                    )
                    rows.append(
                        {
                            "tensor_key": key,
                            "source_parameter": source_name,
                            "head_set": head_set,
                            "projection": projection,
                            "head_kind": head_kind,
                            "head_indices": heads,
                            "shape": list(precursor_slice.shape),
                            "expected_shape": expected_shape,
                            "dtype": str(precursor_slice.dtype),
                            "all_finite": bool(torch.isfinite(precursor_slice).all()),
                            "relative_l2_difference_from_chameleon": float(
                                torch.linalg.vector_norm(difference) / denominator
                            ),
                        }
                    )
    if len(tensors) != 32:
        raise RuntimeError(f"expected 32 frozen slices, found {len(tensors)}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(tensors, TENSOR_PATH)
    selected = contract["head_sets"]["selected"]
    control = contract["head_sets"]["matched_control"]
    checks = {
        "source_hashes_exact": True,
        "exact_tensor_count": len(tensors) == 32,
        "all_shapes_exact": all(row["shape"] == row["expected_shape"] for row in rows),
        "all_bfloat16": all(row["dtype"] == "torch.bfloat16" for row in rows),
        "all_finite": all(row["all_finite"] for row in rows),
        "all_differences_nonzero": all(
            row["relative_l2_difference_from_chameleon"] > 0 for row in rows
        ),
        "query_sets_disjoint": all(
            set(selected[str(layer)]["query"]).isdisjoint(
                control[str(layer)]["query"]
            )
            for layer in contract["architecture"]["layers"]
        ),
        "key_value_sets_disjoint": all(
            set(selected[str(layer)]["key_value"]).isdisjoint(
                control[str(layer)]["key_value"]
            )
            for layer in contract["architecture"]["layers"]
        ),
        "matched_counts": all(
            len(selected[str(layer)][kind]) == len(control[str(layer)][kind])
            for layer in contract["architecture"]["layers"]
            for kind in ("query", "key_value")
        ),
    }
    manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-precursor-parameter-slices-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "source_shards": {
            "chameleon": {
                "path": str(chameleon_path.relative_to(ROOT)),
                "sha256": sha256_file(chameleon_path),
            },
            "exact_precursor": {
                "path": str(precursor_path.relative_to(ROOT)),
                "sha256": sha256_file(precursor_path),
            },
        },
        "tensor_path": str(TENSOR_PATH.relative_to(ROOT)),
        "tensor_sha256": sha256_file(TENSOR_PATH),
        "tensor_bytes": TENSOR_PATH.stat().st_size,
        "slices": rows,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(MANIFEST_PATH, manifest)
    if manifest["result"] != "pass":
        raise RuntimeError(json.dumps(checks, indent=2))
    print(
        json.dumps(
            {
                "result": manifest["result"],
                "tensor_count": len(tensors),
                "tensor_bytes": manifest["tensor_bytes"],
                "tensor_sha256": manifest["tensor_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
