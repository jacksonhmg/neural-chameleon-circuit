#!/usr/bin/env python3
"""Reduce the frozen Day 61 source-state mediation scan."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402
from day61_run_source_state_mediation import (  # noqa: E402
    CONTRACT_PATH,
    EXECUTION_PATH,
    PREFLIGHT_PATH,
    SHARD_DIR,
    expanded_contract,
    sha256_file,
)


SUMMARY_PATH = ROOT / "results/day-61/source-state-mediation-summary.json"
EXAMPLE_PATH = ROOT / "results/day-61/source-state-mediation-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-61/execution-artifact-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_ancestor(commit: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("Day 61 execution commit is not an ancestor")


def expected_states(contract: Mapping[str, Any]) -> set[str]:
    result = set()
    for direction in contract["conditions"]["directions"]:
        result.update({
            f"{direction}.endpoint.identity",
            f"{direction}.endpoint.day60_kv",
        })
        for candidate in contract["candidate_order"]:
            result.add(f"{direction}.candidate.{candidate}")
            result.add(f"{direction}.orthogonal.{candidate}")
    return result


def load_rows(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution, preflight = read_json(EXECUTION_PATH), read_json(PREFLIGHT_PATH)
    require_ancestor(execution["execution_commit"])
    contract_hash = sha256_file(CONTRACT_PATH)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != contract_hash
        or execution.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH)
        or preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution.get("execution_commit")
        or int(execution.get("state_rows", -1)) != int(contract["execution"]["total_state_rows"])
    ):
        raise RuntimeError("Day 61 execution or preflight is invalid")
    expected = expected_states(contract)
    rows, seen, hashes, audit_passes = [], set(), [], []
    identity_k12, identity_monitor = [], []
    for metadata_path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(metadata_path)
        tensor_path = metadata_path.with_suffix(".safetensors")
        if (
            metadata.get("execution_commit") != execution["execution_commit"]
            or metadata.get("contract_sha256") != contract_hash
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
            or set(metadata.get("state_names", [])) != expected
        ):
            raise RuntimeError(f"Day 61 shard differs: {metadata_path}")
        tensors = load_file(tensor_path)
        hashes.append(metadata["tensor_sha256"])
        for audit in metadata["audits"]:
            audit_passes.extend(
                bool(value["pass"]) for value in audit.values()
            )
        for index, example_id in enumerate(metadata["example_ids"]):
            if example_id in seen:
                raise RuntimeError(f"duplicate Day 61 example: {example_id}")
            seen.add(example_id)
            states = {}
            for key, value in tensors.items():
                state_name, field = key.rsplit(".", 1)
                states.setdefault(state_name, {})[field] = value[index]
            rows.append({
                "example_id": example_id,
                "concept": metadata["concept"],
                "response_hash": metadata["response_hashes"][index],
                "states": states,
            })
            for direction in contract["conditions"]["directions"]:
                identity_k12.append(float(states[f"{direction}.endpoint.identity"]["k12_identity_max_abs"]))
                identity_monitor.append(float(states[f"{direction}.endpoint.identity"]["monitor_identity_max_abs"]))
    if len(rows) != int(contract["panel"]["examples"]):
        raise RuntimeError("Day 61 examples are incomplete")
    rows.sort(key=lambda value: value["example_id"])
    return rows, {
        "execution": execution,
        "preflight": preflight,
        "tensor_hash_count": len(set(hashes)),
        "all_tensor_hashes_verified": True,
        "all_orthogonal_geometry_audits_pass": all(audit_passes),
        "orthogonal_geometry_audit_count": len(audit_passes),
        "unique_response_hashes": len({row["response_hash"] for row in rows}),
        "day60_identity_k12_max_abs": max(identity_k12),
        "day60_identity_monitor_margin_max_abs": max(identity_monitor),
    }


def relation(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value, reference = value.reshape(-1).double(), reference.reshape(-1).double()
    denominator = float(reference @ reference)
    if denominator <= 1e-12:
        return {
            "aligned_recovery": 0.0,
            "residual_norm_ratio": float("inf"),
            "norm_ratio": float("inf"),
            "endpoint_nearest": 0.0,
        }
    value_norm = float(torch.linalg.vector_norm(value))
    reference_norm = math.sqrt(denominator)
    residual = float(torch.linalg.vector_norm(value - reference)) / reference_norm
    norm_ratio = value_norm / reference_norm
    return {
        "aligned_recovery": float(value @ reference) / denominator,
        "residual_norm_ratio": residual,
        "norm_ratio": norm_ratio,
        "endpoint_nearest": float(residual < norm_ratio),
    }


def macro(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["concept"])].append(float(row[key]))
    concepts = {
        concept: float(np.median(values))
        for concept, values in sorted(grouped.items())
    }
    return float(np.median(list(concepts.values()))), concepts


def candidate_examples(
    rows: Sequence[Mapping[str, Any]],
    direction: str,
    candidate: str,
    kind: str,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        identity = row["states"][f"{direction}.endpoint.identity"]
        endpoint = row["states"][f"{direction}.endpoint.day60_kv"]
        state = row["states"][f"{direction}.{kind}.{candidate}"]
        monitor = relation(
            state["margins"] - identity["margins"],
            endpoint["margins"] - identity["margins"],
        )
        payload = {
            "concept": row["concept"],
            **{f"monitor_{key}": value for key, value in monitor.items()},
        }
        for field, value in state.items():
            if field != "margins":
                payload[field] = float(value)
        payload["k12_endpoint_nearest"] = float(
            payload["k12_residual_norm_ratio"] < payload["k12_norm_ratio"]
        )
        result.append(payload)
    return result


def summarize_examples(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for key in sorted(set(examples[0]) - {"concept"}):
        median, concepts = macro(examples, key)
        summary[key] = {"median_concept": median, "by_concept": concepts}
    summary["monitor_endpoint_nearest_concepts"] = sum(
        value > 0.5
        for value in summary["monitor_endpoint_nearest"]["by_concept"].values()
    )
    summary["k12_endpoint_nearest_concepts"] = sum(
        value > 0.5
        for value in summary["k12_endpoint_nearest"]["by_concept"].values()
    )
    return summary


def build_summary(
    rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    directions: dict[str, Any] = {}
    qualification: dict[str, Any] = {}
    gates = contract["qualification_gates_both_directions"]
    for direction in contract["conditions"]["directions"]:
        candidates = {}
        for candidate in contract["candidate_order"]:
            exact = summarize_examples(
                candidate_examples(rows, direction, candidate, "candidate", contract)
            )
            orthogonal = summarize_examples(
                candidate_examples(rows, direction, candidate, "orthogonal", contract)
            )
            candidates[candidate] = {"exact": exact, "orthogonal": orthogonal}
        directions[direction] = {"candidates": candidates}
    for candidate in contract["candidate_order"]:
        by_direction = {}
        for direction, direction_payload in directions.items():
            exact = direction_payload["candidates"][candidate]["exact"]
            orthogonal = direction_payload["candidates"][candidate]["orthogonal"]
            minimum_k12_layer = min(
                exact[f"k12_layer_{layer:02d}_aligned_recovery"]["median_concept"]
                for layer in contract["k12"]["layers"]
            )
            minimum_kv_layer = min(
                exact[f"kv_layer_{layer:02d}_aligned_recovery"]["median_concept"]
                for layer in contract["k12"]["layers"]
            )
            values = {
                "monitor_probe_recovery_to_day60_kv": exact["monitor_aligned_recovery"]["median_concept"],
                "k12_recovery_to_day60_kv": exact["k12_aligned_recovery"]["median_concept"],
                "pre_rope_kv_recovery": exact["kv_aligned_recovery"]["median_concept"],
                "minimum_k12_layer_recovery": minimum_k12_layer,
                "minimum_pre_rope_kv_layer_recovery": minimum_kv_layer,
                "monitor_advantage_over_matched_orthogonal": exact["monitor_aligned_recovery"]["median_concept"] - orthogonal["monitor_aligned_recovery"]["median_concept"],
                "k12_advantage_over_matched_orthogonal": exact["k12_aligned_recovery"]["median_concept"] - orthogonal["k12_aligned_recovery"]["median_concept"],
                "monitor_day60_kv_nearest_concepts": exact["monitor_endpoint_nearest_concepts"],
                "k12_day60_kv_nearest_concepts": exact["k12_endpoint_nearest_concepts"],
            }
            passed = {
                key: value >= float(gates[f"{key}_min"])
                for key, value in values.items()
            }
            by_direction[direction] = {
                "values": values,
                "passed": passed,
                "all_pass": all(passed.values()),
            }
        qualification[candidate] = {
            "directions": by_direction,
            "qualifies": all(value["all_pass"] for value in by_direction.values()),
        }
    branches = [
        candidate
        for candidate in contract["candidate_order"]
        if not candidate.startswith("resid_pre.")
        and qualification[candidate]["qualifies"]
    ]
    residuals = [
        candidate
        for candidate in contract["candidate_order"]
        if candidate.startswith("resid_pre.")
        and qualification[candidate]["qualifies"]
    ]
    if branches:
        classification, selected = "compact_single_branch", branches[0]
    elif residuals:
        classification, selected = "residual_state_only", residuals[0]
    else:
        classification, selected = "distributed_no_single_site", None
    summary = {
        "schema_version": 1,
        "procedure": "day61-source-state-kv-mediation-scan-v1",
        "execution_commit": audit["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_status": contract["panel"]["status"],
        "implementation_audit": {
            key: value for key, value in audit.items() if key not in {"execution", "preflight"}
        },
        "qualification": qualification,
        "qualifying_compact_branches": branches,
        "qualifying_residual_boundaries": residuals,
        "selected_candidate": selected,
        "classification": classification,
        "directions": directions,
    }
    examples = {
        "schema_version": 1,
        "procedure": "day61-source-state-mediation-example-metrics-v1",
        "rows": [
            {
                "example_id": row["example_id"],
                "concept": row["concept"],
                "response_hash": row["response_hash"],
            }
            for row in rows
        ],
    }
    return summary, examples


def artifact_manifest() -> dict[str, Any]:
    result_paths = [
        CONTRACT_PATH,
        PREFLIGHT_PATH,
        EXECUTION_PATH,
        SUMMARY_PATH,
        EXAMPLE_PATH,
    ]
    files, tensor_hashes_match = [], True
    for path in sorted(SHARD_DIR.glob("*")):
        if not path.is_file():
            continue
        files.append({
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        if path.suffix == ".json":
            metadata = read_json(path)
            tensor_hashes_match &= metadata["tensor_sha256"] == sha256_file(
                path.with_suffix(".safetensors")
            )
    return {
        "schema_version": 1,
        "procedure": "day61-source-state-mediation-artifact-manifest-v1",
        "execution_commit": read_json(EXECUTION_PATH)["execution_commit"],
        "result_files": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in result_paths
        },
        "shards": {
            "directory": str(SHARD_DIR.relative_to(ROOT)),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(value["bytes"] for value in files),
            "all_tensor_hashes_match_metadata": tensor_hashes_match,
        },
    }


def main() -> None:
    contract = expanded_contract()
    rows, audit = load_rows(contract)
    summary, examples = build_summary(rows, audit, contract)
    write_json_atomic(EXAMPLE_PATH, examples)
    write_json_atomic(SUMMARY_PATH, summary)
    write_json_atomic(MANIFEST_PATH, artifact_manifest())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
