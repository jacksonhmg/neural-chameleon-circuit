#!/usr/bin/env python3
"""Reduce the frozen Day 60 Q/K/V source-region factorial."""

from __future__ import annotations

import hashlib
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
from day60_run_qkv_source_factorial import (  # noqa: E402
    CONTRACT_PATH,
    EXECUTION_PATH,
    PREFLIGHT_PATH,
    SHARD_DIR,
    expanded_contract,
    load_records,
)


SUMMARY_PATH = ROOT / "results/day-60/qkv-source-factorial-summary.json"
EXAMPLE_METRICS_PATH = ROOT / "results/day-60/qkv-source-factorial-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-60/execution-artifact-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_ancestor(commit: str) -> None:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("Day 60 execution commit is not an ancestor")


def state(row: Mapping[str, Any], name: str, field: str) -> torch.Tensor:
    return row["states"][f"{name}.{field}"].double()


def relation(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value, reference = value.reshape(-1), reference.reshape(-1)
    denominator = float(reference @ reference)
    value_norm = float(torch.linalg.vector_norm(value))
    reference_norm = math.sqrt(max(denominator, 0.0))
    if denominator <= 1e-12:
        return {"aligned_recovery": 0.0, "norm_ratio": float("inf"), "cosine": 0.0}
    return {
        "aligned_recovery": float(value @ reference) / denominator,
        "norm_ratio": value_norm / reference_norm,
        "cosine": float(value @ reference) / max(value_norm * reference_norm, 1e-12),
    }


def vector_metrics(
    changed: torch.Tensor, target: torch.Tensor, donor: torch.Tensor
) -> dict[str, float]:
    changed, target, donor = changed.reshape(-1), target.reshape(-1), donor.reshape(-1)
    effect, exact = changed - target, donor - target
    denominator = float(exact @ exact)
    if denominator <= 1e-12:
        return {
            "recovery": 0.0,
            "residual_norm_ratio": float("inf"),
            "donor_nearest": 0.0,
            "effect_norm": float(torch.linalg.vector_norm(effect)),
        }
    return {
        "recovery": float(effect @ exact) / denominator,
        "residual_norm_ratio": float(torch.linalg.vector_norm(changed - donor)) / math.sqrt(denominator),
        "donor_nearest": float(torch.linalg.vector_norm(changed - donor) < torch.linalg.vector_norm(changed - target)),
        "effect_norm": float(torch.linalg.vector_norm(effect)),
    }


def macro(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["concept"])].append(float(row[key]))
    concepts = {name: float(np.median(values)) for name, values in sorted(grouped.items())}
    return float(np.median(list(concepts.values()))), concepts


def expected_diagnostics(contract: Mapping[str, Any]) -> set[str]:
    result = set()
    for direction in contract["conditions"]["directions"]:
        for interaction in ("pair_qk", "pair_qv", "pair_kv", "three_way"):
            for metric in ("aligned_ratio", "norm_ratio", "cosine"):
                result.add(f"{direction}.{interaction}.{metric}")
        for region in contract["region_classification"]["atomic_regions"]:
            for metric in (
                "incremental_aligned_recovery",
                "incremental_norm_ratio",
                "incremental_cosine",
            ):
                result.add(f"{direction}.region.{region}.{metric}")
    return result


def load_rows(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution, preflight = read_json(EXECUTION_PATH), read_json(PREFLIGHT_PATH)
    require_ancestor(execution["execution_commit"])
    contract_hash = sha256_file(CONTRACT_PATH)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != contract_hash
        or preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution.get("execution_commit")
        or execution.get("preflight_sha256") != sha256_file(PREFLIGHT_PATH)
    ):
        raise RuntimeError("Day 60 execution or preflight is invalid")
    expected_ids = {row["example_id"] for row in load_records(contract)}
    expected_states = {
        *(f"{direction}.{job}" for direction in contract["conditions"]["directions"] for job in contract["jobs_per_direction"]),
        *(f"natural.{direction}.{endpoint}" for direction in contract["conditions"]["directions"] for endpoint in ("target", "donor")),
    }
    diagnostic_names = expected_diagnostics(contract)
    rows, seen, tensor_hashes = [], set(), []
    identity_errors = []
    for metadata_path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(metadata_path)
        tensor_path = metadata_path.with_suffix(".safetensors")
        if (
            metadata.get("execution_commit") != execution["execution_commit"]
            or metadata.get("contract_sha256") != contract_hash
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
            or set(metadata.get("state_names", [])) != expected_states
            or set(metadata.get("diagnostic_names", [])) != diagnostic_names
        ):
            raise RuntimeError(f"Day 60 shard differs: {metadata_path}")
        tensors = load_file(tensor_path)
        tensor_hashes.append(metadata["tensor_sha256"])
        for index, example_id in enumerate(metadata["example_ids"]):
            if example_id in seen:
                raise RuntimeError(f"duplicate Day 60 example: {example_id}")
            seen.add(example_id)
            rows.append({
                "example_id": example_id,
                "concept": metadata["concept"],
                "response_hash": metadata["response_hashes"][index],
                "states": {
                    key: value[index]
                    for key, value in tensors.items()
                    if not key.startswith("diagnostic.")
                },
                "diagnostics": {
                    key.removeprefix("diagnostic."): value[index]
                    for key, value in tensors.items()
                    if key.startswith("diagnostic.")
                },
            })
        for direction in contract["conditions"]["directions"]:
            identity_errors.append(float((
                tensors[f"{direction}.identity_target.margins"]
                - tensors[f"natural.{direction}.target.margins"]
            ).abs().max()))
    if seen != expected_ids or len(rows) != int(contract["panel"]["examples"]):
        raise RuntimeError("Day 60 examples are incomplete")
    if int(execution["state_rows"]) != int(contract["execution"]["total_state_rows"]):
        raise RuntimeError("Day 60 state row count differs")
    rows.sort(key=lambda row: row["example_id"])
    return rows, {
        "execution": execution,
        "preflight": preflight,
        "identity_monitor_margin_max_abs": max(identity_errors),
        "tensor_hash_count": len(set(tensor_hashes)),
        "all_tensor_hashes_verified": True,
        "unique_response_hashes": len({row["response_hash"] for row in rows}),
    }


def summarize_job(
    rows: Sequence[Mapping[str, Any]], direction: str, job: str
) -> dict[str, Any]:
    examples = []
    for row in rows:
        natural_target = state(row, f"natural.{direction}.target", "margins")
        natural_donor = state(row, f"natural.{direction}.donor", "margins")
        identity = state(row, f"{direction}.identity_target", "margins")
        exact = state(row, f"{direction}.exact_donor_k12", "margins")
        changed = state(row, f"{direction}.{job}", "margins")
        examples.append({
            "concept": row["concept"],
            **{f"natural_{key}": value for key, value in vector_metrics(changed, natural_target, natural_donor).items()},
            **{f"exact_{key}": value for key, value in vector_metrics(changed, identity, exact).items()},
            "k12_recovery": float(state(row, f"{direction}.{job}", "k12_recovery")),
            "k12_effect_norm": float(state(row, f"{direction}.{job}", "k12_effect_norm")),
        })
    result = {}
    for key in sorted(set(examples[0]) - {"concept"}):
        median, concepts = macro(examples, key)
        result[key] = {"median_concept": median, "by_concept": concepts}
    result["exact_donor_nearest_concepts"] = sum(
        value > 0.5 for value in result["exact_donor_nearest"]["by_concept"].values()
    )
    return result


def monitor_interactions(
    rows: Sequence[Mapping[str, Any]], direction: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    mapping = contract["full_prefix_job_by_factor"]
    example_rows = []
    for row in rows:
        base = state(row, f"{direction}.identity_target", "margins")
        effects = {
            factor: state(row, f"{direction}.{job}", "margins") - base
            for factor, job in mapping.items()
        }
        interactions = {
            "pair_qk": effects["qk"] - effects["q"] - effects["k"],
            "pair_qv": effects["qv"] - effects["q"] - effects["v"],
            "pair_kv": effects["kv"] - effects["k"] - effects["v"],
            "three_way": (
                effects["qkv"] - effects["qk"] - effects["qv"] - effects["kv"]
                + effects["q"] + effects["k"] + effects["v"]
            ),
        }
        payload = {"concept": row["concept"]}
        for name, value in interactions.items():
            payload.update({
                f"{name}.{key}": metric
                for key, metric in relation(value, effects["qkv"]).items()
            })
        example_rows.append(payload)
    result = {}
    for key in sorted(set(example_rows[0]) - {"concept"}):
        median, concepts = macro(example_rows, key)
        result[key] = {"median_concept": median, "by_concept": concepts}
    return result


def k12_interactions(
    rows: Sequence[Mapping[str, Any]], direction: str
) -> dict[str, Any]:
    result = {}
    keys = [
        key.removeprefix(f"{direction}.")
        for key in rows[0]["diagnostics"]
        if key.startswith(f"{direction}.") and ".region." not in key
    ]
    for key in sorted(keys):
        values = [
            {"concept": row["concept"], "value": float(row["diagnostics"][f"{direction}.{key}"])}
            for row in rows
        ]
        median, concepts = macro(values, "value")
        result[key] = {"median_concept": median, "by_concept": concepts}
    return result


def region_summary(
    rows: Sequence[Mapping[str, Any]], direction: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    result = {}
    full_name = contract["full_prefix_job_by_factor"]["qkv"]
    for region in contract["region_classification"]["atomic_regions"]:
        examples = []
        for row in rows:
            q = state(row, f"{direction}.q", "margins")
            region_increment = state(row, f"{direction}.{region}.qkv", "margins") - q
            full_increment = state(row, f"{direction}.{full_name}", "margins") - q
            metrics = relation(region_increment, full_increment)
            examples.append({
                "concept": row["concept"],
                **{f"probe_{key}": value for key, value in metrics.items()},
                "k12_aligned_recovery": float(row["diagnostics"][f"{direction}.region.{region}.incremental_aligned_recovery"]),
                "k12_norm_ratio": float(row["diagnostics"][f"{direction}.region.{region}.incremental_norm_ratio"]),
                "k12_cosine": float(row["diagnostics"][f"{direction}.region.{region}.incremental_cosine"]),
            })
        summary = {}
        for key in sorted(set(examples[0]) - {"concept"}):
            median, concepts = macro(examples, key)
            summary[key] = {"median_concept": median, "by_concept": concepts}
        result[region] = summary
    return result


def build_summary(
    rows: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    directions = {}
    orthogonal_recovery = {}
    for direction in contract["conditions"]["directions"]:
        jobs = {job: summarize_job(rows, direction, job) for job in contract["jobs_per_direction"]}
        orthogonal_recovery[direction] = jobs["exact_k12_orthogonal"]["exact_recovery"]["median_concept"]
        directions[direction] = {
            "jobs": jobs,
            "monitor_interactions": monitor_interactions(rows, direction, contract),
            "k12_interactions": k12_interactions(rows, direction),
            "regions": region_summary(rows, direction, contract),
        }
    confirmation_gates = contract["confirmation_gates_both_directions"]
    confirmation = {}
    for direction, payload in directions.items():
        jobs = payload["jobs"]
        values = {
            "exact_k12_natural_probe_recovery": jobs["exact_donor_k12"]["natural_recovery"]["median_concept"],
            "full_qkv_probe_recovery_to_exact_k12": jobs["monitoring_prefix.qkv"]["exact_recovery"]["median_concept"],
            "full_qkv_k12_recovery": jobs["monitoring_prefix.qkv"]["k12_recovery"]["median_concept"],
            "full_qkv_advantage_over_exact_orthogonal": jobs["monitoring_prefix.qkv"]["exact_recovery"]["median_concept"] - orthogonal_recovery[direction],
            "full_qkv_exact_nearest_concepts": jobs["monitoring_prefix.qkv"]["exact_donor_nearest_concepts"],
        }
        passed = {
            key: value >= confirmation_gates[f"{key}_min"]
            for key, value in values.items()
        }
        confirmation[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values())}
    subset_gates = contract["proper_subset_qualification_gates_both_directions"]
    qualification = {}
    for factor in contract["full_prefix_factor_selection_order"]:
        job = contract["full_prefix_job_by_factor"][factor]
        factor_directions = {}
        for direction, payload in directions.items():
            summary = payload["jobs"][job]
            values = {
                "probe_recovery_to_exact_k12": summary["exact_recovery"]["median_concept"],
                "k12_recovery": summary["k12_recovery"]["median_concept"],
                "advantage_over_exact_orthogonal": summary["exact_recovery"]["median_concept"] - orthogonal_recovery[direction],
                "exact_nearest_concepts": summary["exact_donor_nearest_concepts"],
            }
            passed = {key: value >= subset_gates[f"{key}_min"] for key, value in values.items()}
            factor_directions[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values())}
        qualification[factor] = {
            "directions": factor_directions,
            "qualifies": all(value["all_pass"] for value in factor_directions.values()),
        }
    full_confirmed = all(value["all_pass"] for value in confirmation.values())
    selected_factor = next(
        (factor for factor in contract["full_prefix_factor_selection_order"] if qualification[factor]["qualifies"]),
        None,
    )
    interaction_gates = contract["interaction_classification"]
    three_way_material = all(
        directions[direction]["monitor_interactions"]["three_way.norm_ratio"]["median_concept"] >= float(interaction_gates["three_way_probe_norm_ratio_min"])
        and directions[direction]["k12_interactions"]["three_way.norm_ratio"]["median_concept"] >= float(interaction_gates["three_way_k12_norm_ratio_min"])
        for direction in directions
    )
    if not full_confirmed:
        factor_classification = "not_confirmed"
    elif selected_factor != "qkv":
        factor_classification = f"proper_subset_sufficient:{selected_factor}"
    elif three_way_material:
        factor_classification = "genuine_three_way_coordination"
    else:
        factor_classification = "joint_without_material_three_way"
    region_scores = {}
    for region in contract["region_classification"]["atomic_regions"]:
        region_scores[region] = min(
            directions[direction]["regions"][region][metric]["median_concept"]
            for direction in directions
            for metric in ("probe_aligned_recovery", "k12_aligned_recovery")
        )
    ordered_regions = sorted(region_scores, key=lambda region: region_scores[region], reverse=True)
    top, second = ordered_regions[:2]
    if (
        region_scores[top] >= float(contract["region_classification"]["dominance_score_min"])
        and region_scores[top] - region_scores[second] >= float(contract["region_classification"]["dominance_margin_over_second_min"])
    ):
        region_classification = f"dominant:{top}"
    else:
        region_classification = "distributed_across_prefix_regions"
    summary = {
        "schema_version": 1,
        "procedure": "day60-fresh-qkv-source-factorial-v1",
        "execution_commit": audit["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "implementation_audit": {key: value for key, value in audit.items() if key not in {"execution", "preflight"}},
        "confirmation": confirmation,
        "factor_qualification": qualification,
        "selected_factor": selected_factor,
        "three_way_interaction_material": three_way_material,
        "factor_classification": factor_classification,
        "region_scores": region_scores,
        "region_classification": region_classification,
        "directions": directions,
    }
    examples = {
        "schema_version": 1,
        "procedure": "day60-fresh-qkv-source-factorial-example-metrics-v1",
        "rows": [
            {
                "example_id": row["example_id"],
                "concept": row["concept"],
                "response_hash": row["response_hash"],
                "diagnostics": {key: float(value) for key, value in row["diagnostics"].items()},
            }
            for row in rows
        ],
    }
    return summary, examples


def artifact_manifest() -> dict[str, Any]:
    result_paths = [CONTRACT_PATH, PREFLIGHT_PATH, EXECUTION_PATH, SUMMARY_PATH, EXAMPLE_METRICS_PATH]
    files = []
    tensor_hashes_match = True
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
            tensor_hashes_match &= metadata["tensor_sha256"] == sha256_file(path.with_suffix(".safetensors"))
    return {
        "schema_version": 1,
        "procedure": "day60-qkv-source-factorial-artifact-manifest-v1",
        "execution_commit": read_json(EXECUTION_PATH)["execution_commit"],
        "result_files": {str(path.relative_to(ROOT)): sha256_file(path) for path in result_paths},
        "shards": {
            "directory": str(SHARD_DIR.relative_to(ROOT)),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files),
            "all_tensor_hashes_match_metadata": tensor_hashes_match,
        },
    }


def main() -> None:
    contract = expanded_contract()
    rows, audit = load_rows(contract)
    summary, examples = build_summary(rows, audit, contract)
    write_json_atomic(EXAMPLE_METRICS_PATH, examples)
    write_json_atomic(SUMMARY_PATH, summary)
    write_json_atomic(MANIFEST_PATH, artifact_manifest())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
