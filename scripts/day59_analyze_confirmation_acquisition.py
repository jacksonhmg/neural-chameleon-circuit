#!/usr/bin/env python3
"""Reduce Day 59 Chameleon confirmation and precursor acquisition."""

from __future__ import annotations

import argparse
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
from day59_run_confirmation_acquisition import (  # noqa: E402
    CONTRACT_PATH,
    EXECUTION_PATHS,
    PREFLIGHT_PATHS,
    SHARD_DIRS,
    expanded_contract,
    load_records,
)


CHAMELEON_SUMMARY_PATH = ROOT / "results/day-59/chameleon-confirmation-summary.json"
ACQUISITION_SUMMARY_PATH = ROOT / "results/day-59/precursor-acquisition-summary.json"
MANIFEST_PATH = ROOT / "results/day-59/execution-artifact-manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("chameleon", "precursor"), required=True)
    return parser.parse_args()


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
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False).returncode:
        raise RuntimeError("Day 59 execution commit is not an ancestor")


def state(row: Mapping[str, Any], name: str, field: str) -> torch.Tensor:
    return row["states"][f"{name}.{field}"].double()


def vector_metrics(changed: torch.Tensor, target: torch.Tensor, donor: torch.Tensor) -> dict[str, float]:
    changed, target, donor = changed.reshape(-1), target.reshape(-1), donor.reshape(-1)
    effect, exact = changed - target, donor - target
    denominator = float(exact @ exact)
    if denominator <= 1e-12:
        return {"recovery": 0.0, "residual_norm_ratio": float("inf"), "donor_nearest": 0.0, "effect_norm": float(torch.linalg.vector_norm(effect))}
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


def load_stage(model_key: str, contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution, preflight = read_json(EXECUTION_PATHS[model_key]), read_json(PREFLIGHT_PATHS[model_key])
    require_ancestor(execution["execution_commit"])
    contract_hash = sha256_file(CONTRACT_PATH)
    if not execution.get("complete") or execution.get("contract_sha256") != contract_hash or preflight.get("result") != "pass" or preflight.get("execution_commit") != execution.get("execution_commit") or execution.get("preflight_sha256") != sha256_file(PREFLIGHT_PATHS[model_key]):
        raise RuntimeError(f"Day 59 {model_key} execution or preflight is invalid")
    expected_ids = {row["example_id"] for row in load_records(contract)}
    expected_states = {
        *(f"{direction}.{job}" for direction in contract["conditions"]["directions"] for job in contract["jobs_per_direction"]),
        *(f"natural.{direction}.{endpoint}" for direction in contract["conditions"]["directions"] for endpoint in ("target", "donor")),
    }
    rows, seen, response_hashes = [], set(), {}
    identity_errors, full_tail_errors = [], []
    tensor_hashes = []
    for metadata_path in sorted(SHARD_DIRS[model_key].glob("*.json")):
        metadata = read_json(metadata_path)
        tensor_path = metadata_path.with_suffix(".safetensors")
        if metadata.get("model_key") != model_key or metadata.get("execution_commit") != execution["execution_commit"] or metadata.get("contract_sha256") != contract_hash or metadata.get("tensor_sha256") != sha256_file(tensor_path) or set(metadata.get("state_names", [])) != expected_states:
            raise RuntimeError(f"Day 59 shard differs: {metadata_path}")
        tensors = load_file(tensor_path)
        tensor_hashes.append(metadata["tensor_sha256"])
        for index, example_id in enumerate(metadata["example_ids"]):
            if example_id in seen:
                raise RuntimeError(f"duplicate Day 59 example: {example_id}")
            seen.add(example_id)
            response_hashes[example_id] = metadata["response_hashes"][index]
            rows.append({"example_id": example_id, "concept": metadata["concept"], "states": {key: value[index] for key, value in tensors.items()}})
        for direction in contract["conditions"]["directions"]:
            identity_errors.append(float((tensors[f"{direction}.identity_target.margins"] - tensors[f"natural.{direction}.target.margins"]).abs().max()))
            full_tail_errors.append(float((tensors[f"{direction}.full_tail.margins"] - tensors[f"natural.{direction}.donor.margins"]).abs().max()))
    if seen != expected_ids or len(rows) != int(contract["panel"]["examples"]):
        raise RuntimeError(f"Day 59 {model_key} examples are incomplete")
    rows.sort(key=lambda row: row["example_id"])
    return rows, {
        "execution": execution,
        "preflight": preflight,
        "response_hashes": response_hashes,
        "identity_monitor_margin_max_abs": max(identity_errors),
        "full_tail_monitor_margin_max_abs": max(full_tail_errors),
        "tensor_hash_count": len(set(tensor_hashes)),
        "all_tensor_hashes_verified": True,
    }


def summarize_job(rows: Sequence[Mapping[str, Any]], direction: str, job: str) -> dict[str, Any]:
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
    result["natural_donor_nearest_concepts"] = sum(value > 0.5 for value in result["natural_donor_nearest"]["by_concept"].values())
    result["exact_donor_nearest_concepts"] = sum(value > 0.5 for value in result["exact_donor_nearest"]["by_concept"].values())
    return result


def chameleon_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    rows, audit = load_stage("chameleon", contract)
    directions, all_pass = {}, True
    gates = contract["chameleon_confirmation_gates_both_directions"]
    for direction in contract["conditions"]["directions"]:
        jobs = {job: summarize_job(rows, direction, job) for job in contract["jobs_per_direction"]}
        values = {
            "exact_k12_natural_probe_recovery": jobs["exact_donor_k12"]["natural_recovery"]["median_concept"],
            "qkv_probe_recovery_to_exact_k12": jobs["qkv_prefix"]["exact_recovery"]["median_concept"],
            "qkv_k12_recovery": jobs["qkv_prefix"]["k12_recovery"]["median_concept"],
            "qkv_advantage_over_exact_orthogonal": jobs["qkv_prefix"]["exact_recovery"]["median_concept"] - jobs["exact_k12_orthogonal"]["exact_recovery"]["median_concept"],
            "qkv_exact_nearest_concepts": jobs["qkv_prefix"]["exact_donor_nearest_concepts"],
            "joint_natural_probe_recovery": jobs["exact_plus_other_heads"]["natural_recovery"]["median_concept"],
            "joint_increment_over_exact_k12": jobs["exact_plus_other_heads"]["natural_recovery"]["median_concept"] - jobs["exact_donor_k12"]["natural_recovery"]["median_concept"],
            "joint_advantage_over_orthogonal": jobs["exact_plus_other_heads"]["natural_recovery"]["median_concept"] - jobs["exact_plus_other_heads_orthogonal"]["natural_recovery"]["median_concept"],
            "joint_natural_probe_residual_norm_ratio": jobs["exact_plus_other_heads"]["natural_residual_norm_ratio"]["median_concept"],
            "full_tail_natural_probe_recovery": jobs["full_tail"]["natural_recovery"]["median_concept"],
        }
        passed = {
            "exact_k12_natural_probe_recovery": values["exact_k12_natural_probe_recovery"] >= gates["exact_k12_natural_probe_recovery_min"],
            "qkv_probe_recovery_to_exact_k12": values["qkv_probe_recovery_to_exact_k12"] >= gates["qkv_probe_recovery_to_exact_k12_min"],
            "qkv_k12_recovery": values["qkv_k12_recovery"] >= gates["qkv_k12_recovery_min"],
            "qkv_advantage_over_exact_orthogonal": values["qkv_advantage_over_exact_orthogonal"] >= gates["qkv_advantage_over_exact_orthogonal_min"],
            "qkv_exact_nearest_concepts": values["qkv_exact_nearest_concepts"] >= gates["qkv_exact_nearest_concepts_min"],
            "joint_natural_probe_recovery": values["joint_natural_probe_recovery"] >= gates["joint_natural_probe_recovery_min"],
            "joint_increment_over_exact_k12": values["joint_increment_over_exact_k12"] >= gates["joint_increment_over_exact_k12_min"],
            "joint_advantage_over_orthogonal": values["joint_advantage_over_orthogonal"] >= gates["joint_advantage_over_orthogonal_min"],
            "joint_natural_probe_residual_norm_ratio": values["joint_natural_probe_residual_norm_ratio"] <= gates["joint_natural_probe_residual_norm_ratio_max"],
            "full_tail_natural_probe_recovery": values["full_tail_natural_probe_recovery"] >= gates["full_tail_natural_probe_recovery_min"],
        }
        directions[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values()), "jobs": jobs}
        all_pass &= all(passed.values())
    return {
        "schema_version": 1,
        "procedure": "day59-chameleon-selected-mechanism-confirmation-v1",
        "execution_commit": audit["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "implementation_audit": {key: value for key, value in audit.items() if key not in {"execution", "preflight", "response_hashes"}},
        "directions": directions,
        "decision": "chameleon_confirmation_pass" if all_pass else "chameleon_confirmation_fail",
    }


def ratio_summary(
    chameleon_rows: Sequence[Mapping[str, Any]],
    precursor_rows: Sequence[Mapping[str, Any]],
    direction: str,
) -> dict[str, Any]:
    by_id = {row["example_id"]: row for row in precursor_rows}
    examples = []
    for row in chameleon_rows:
        precursor = by_id[row["example_id"]]
        def probe_effect(item: Mapping[str, Any], state_name: str, base_name: str) -> float:
            return float(torch.linalg.vector_norm(state(item, state_name, "margins") - state(item, base_name, "margins")))
        ch_nat = probe_effect(row, f"natural.{direction}.donor", f"natural.{direction}.target")
        pr_nat = probe_effect(precursor, f"natural.{direction}.donor", f"natural.{direction}.target")
        ch_qkv = probe_effect(row, f"{direction}.qkv_prefix", f"{direction}.identity_target")
        pr_qkv = probe_effect(precursor, f"{direction}.qkv_prefix", f"{direction}.identity_target")
        ch_joint = probe_effect(row, f"{direction}.exact_plus_other_heads", f"{direction}.identity_target")
        pr_joint = probe_effect(precursor, f"{direction}.exact_plus_other_heads", f"{direction}.identity_target")
        ch_nat_k12 = float(state(row, f"natural.{direction}.donor", "k12_effect_norm"))
        pr_nat_k12 = float(state(precursor, f"natural.{direction}.donor", "k12_effect_norm"))
        ch_qkv_k12 = float(state(row, f"{direction}.qkv_prefix", "k12_effect_norm"))
        pr_qkv_k12 = float(state(precursor, f"{direction}.qkv_prefix", "k12_effect_norm"))
        examples.append({
            "concept": row["concept"],
            "natural_probe_effect_norm_ratio": pr_nat / max(ch_nat, 1e-8),
            "natural_k12_effect_norm_ratio": pr_nat_k12 / max(ch_nat_k12, 1e-8),
            "qkv_probe_effect_norm_ratio": pr_qkv / max(ch_qkv, 1e-8),
            "qkv_k12_effect_norm_ratio": pr_qkv_k12 / max(ch_qkv_k12, 1e-8),
            "joint_probe_effect_norm_ratio": pr_joint / max(ch_joint, 1e-8),
        })
    result = {}
    for key in sorted(set(examples[0]) - {"concept"}):
        median, concepts = macro(examples, key)
        result[key] = {"median_concept": median, "by_concept": concepts}
    return result


def acquisition_summary(contract: Mapping[str, Any]) -> dict[str, Any]:
    chameleon = read_json(CHAMELEON_SUMMARY_PATH)
    if chameleon.get("decision") != "chameleon_confirmation_pass":
        raise RuntimeError("Day 59 precursor reduction is ineligible")
    ch_rows, ch_audit = load_stage("chameleon", contract)
    pr_rows, pr_audit = load_stage("exact_precursor", contract)
    cross_hashes = ch_audit["response_hashes"] == pr_audit["response_hashes"]
    gates = contract["precursor_acquisition_gates_both_directions"]
    directions, all_pass = {}, cross_hashes
    for direction in contract["conditions"]["directions"]:
        ratios = ratio_summary(ch_rows, pr_rows, direction)
        values = {key: value["median_concept"] for key, value in ratios.items()}
        passed = {key: values[key] <= float(gates[f"{key}_max"]) for key in values}
        directions[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values()), "by_concept": {key: value["by_concept"] for key, value in ratios.items()}}
        all_pass &= all(passed.values())
    if all_pass:
        classification = "acquired"
    else:
        conserved = any(
            all(directions[direction]["values"][key] > 0.50 for direction in directions)
            for key in next(iter(directions.values()))["values"]
        )
        classification = "conserved" if conserved else "ambiguous"
    return {
        "schema_version": 1,
        "procedure": "day59-exact-precursor-acquisition-comparison-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "chameleon_summary_sha256": sha256_file(CHAMELEON_SUMMARY_PATH),
        "analysis_commit": git_head(),
        "cross_checkpoint_response_hashes_exact": cross_hashes,
        "precursor_implementation_audit": {key: value for key, value in pr_audit.items() if key not in {"execution", "preflight", "response_hashes"}},
        "directions": directions,
        "classification": classification,
    }


def artifact_manifest() -> dict[str, Any]:
    result_paths = [
        CONTRACT_PATH,
        PREFLIGHT_PATHS["chameleon"],
        EXECUTION_PATHS["chameleon"],
        CHAMELEON_SUMMARY_PATH,
        PREFLIGHT_PATHS["exact_precursor"],
        EXECUTION_PATHS["exact_precursor"],
        ACQUISITION_SUMMARY_PATH,
    ]
    result_files = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in result_paths
    }
    shard_sets = {}
    for model_key, directory in SHARD_DIRS.items():
        files = []
        tensor_hashes_match = True
        for path in sorted(directory.glob("*")):
            if not path.is_file():
                continue
            files.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
            if path.suffix == ".json":
                metadata = read_json(path)
                tensor_hashes_match &= (
                    metadata["tensor_sha256"]
                    == sha256_file(path.with_suffix(".safetensors"))
                )
        shard_sets[model_key] = {
            "directory": str(directory.relative_to(ROOT)),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(row["bytes"] for row in files),
            "all_tensor_hashes_match_metadata": tensor_hashes_match,
        }
    return {
        "schema_version": 1,
        "procedure": "day59-selected-mechanism-confirmation-acquisition-artifact-manifest-v1",
        "execution_commit": read_json(EXECUTION_PATHS["chameleon"])["execution_commit"],
        "result_files": result_files,
        "shard_sets": shard_sets,
    }


def main() -> None:
    args = parse_args()
    contract = expanded_contract()
    if args.stage == "chameleon":
        result = chameleon_summary(contract)
        write_json_atomic(CHAMELEON_SUMMARY_PATH, result)
    else:
        result = acquisition_summary(contract)
        write_json_atomic(ACQUISITION_SUMMARY_PATH, result)
        write_json_atomic(MANIFEST_PATH, artifact_manifest())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
