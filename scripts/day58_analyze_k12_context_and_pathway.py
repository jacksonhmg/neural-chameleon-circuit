#!/usr/bin/env python3
"""Reduce and select the frozen Day 58 development mechanisms."""

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
from day58_run_k12_context_and_pathway import (  # noqa: E402
    CONTRACT_PATH,
    EXECUTION_PATH,
    PREFLIGHT_PATH,
    SHARD_DIR,
    expanded_contract,
    load_records,
)


SUMMARY_PATH = ROOT / "results/day-58/development-summary.json"
METRICS_PATH = ROOT / "results/day-58/development-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-58/development-artifact-manifest.json"


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
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False
    ).returncode:
        raise RuntimeError("Day 58 execution commit is not an ancestor of analysis")


def state(tensors: Mapping[str, torch.Tensor], name: str, field: str) -> torch.Tensor:
    return tensors[f"{name}.{field}"].double()


def vector_metrics(changed: torch.Tensor, target: torch.Tensor, donor: torch.Tensor) -> dict[str, float]:
    effect = changed.reshape(-1) - target.reshape(-1)
    exact = donor.reshape(-1) - target.reshape(-1)
    denominator = float(exact @ exact)
    if denominator <= 1e-12:
        return {"recovery": 0.0, "residual_norm_ratio": float("inf"), "donor_nearest": 0.0}
    return {
        "recovery": float(effect @ exact) / denominator,
        "residual_norm_ratio": float(torch.linalg.vector_norm(changed - donor)) / math.sqrt(denominator),
        "donor_nearest": float(
            torch.linalg.vector_norm(changed - donor) < torch.linalg.vector_norm(changed - target)
        ),
    }


def macro(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["concept"])].append(float(row[key]))
    concepts = {name: float(np.median(values)) for name, values in sorted(grouped.items())}
    return float(np.median(list(concepts.values()))), concepts


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
        raise RuntimeError("Day 58 execution or preflight is invalid")
    expected_ids = {row["example_id"] for row in load_records(contract)}
    expected_states = {
        *(f"{direction}.{job}" for direction in contract["conditions"]["directions"] for job in contract["jobs_per_direction"]),
        *(f"natural.{direction}.{endpoint}" for direction in contract["conditions"]["directions"] for endpoint in ("target", "donor")),
    }
    rows = []
    seen: set[str] = set()
    identity_k12_errors = []
    identity_margin_errors = []
    full_tail_margin_errors = []
    manifest_files = []
    for metadata_path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(metadata_path)
        tensor_path = metadata_path.with_suffix(".safetensors")
        if (
            metadata.get("execution_commit") != execution["execution_commit"]
            or metadata.get("contract_sha256") != contract_hash
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
            or set(metadata.get("state_names", [])) != expected_states
        ):
            raise RuntimeError(f"Day 58 shard differs: {metadata_path}")
        tensors = load_file(tensor_path)
        manifest_files.extend((
            {"path": str(metadata_path.relative_to(ROOT)), "sha256": sha256_file(metadata_path), "bytes": metadata_path.stat().st_size},
            {"path": str(tensor_path.relative_to(ROOT)), "sha256": sha256_file(tensor_path), "bytes": tensor_path.stat().st_size},
        ))
        for index, example_id in enumerate(metadata["example_ids"]):
            if example_id in seen:
                raise RuntimeError(f"duplicate Day 58 example: {example_id}")
            seen.add(example_id)
            row: dict[str, Any] = {"example_id": example_id, "concept": metadata["concept"], "states": {}}
            for key, value in tensors.items():
                row["states"][key] = value[index]
            rows.append(row)
        for direction in contract["conditions"]["directions"]:
            identity_k12_errors.append(float(state(tensors, f"{direction}.identity_target", "k12_effect_norm").abs().max()))
            identity_margin_errors.append(float(
                (state(tensors, f"{direction}.identity_target", "margins") - state(tensors, f"natural.{direction}.target", "margins")).abs().max()
            ))
            full_tail_margin_errors.append(float(
                (state(tensors, f"{direction}.full_tail", "margins") - state(tensors, f"natural.{direction}.donor", "margins")).abs().max()
            ))
    if seen != expected_ids or len(rows) != int(contract["evidence"]["development_panel"]["examples"]):
        raise RuntimeError("Day 58 examples are incomplete")
    rows.sort(key=lambda row: row["example_id"])
    audit = {
        "examples": len(rows),
        "identity_k12_effect_norm_max": max(identity_k12_errors),
        "identity_monitor_margin_max_abs": max(identity_margin_errors),
        "full_tail_monitor_margin_max_abs": max(full_tail_margin_errors),
        "all_tensor_hashes_verified": True,
        "state_rows": len(rows) * len(expected_states),
    }
    write_json_atomic(MANIFEST_PATH, {
        "schema_version": 1,
        "execution_commit": execution["execution_commit"],
        "contract_sha256": contract_hash,
        "files": sorted(manifest_files, key=lambda row: row["path"]),
        "file_count": len(manifest_files),
        "total_bytes": sum(row["bytes"] for row in manifest_files),
    })
    return rows, audit


def summarize_job(rows: Sequence[Mapping[str, Any]], direction: str, job: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    examples = []
    for row in rows:
        tensors = row["states"]
        changed = tensors[f"{direction}.{job}.margins"].double()
        natural_target = tensors[f"natural.{direction}.target.margins"].double()
        natural_donor = tensors[f"natural.{direction}.donor.margins"].double()
        identity = tensors[f"{direction}.identity_target.margins"].double()
        exact = tensors[f"{direction}.exact_donor_k12.margins"].double()
        values = {
            "example_id": row["example_id"],
            "concept": row["concept"],
            "direction": direction,
            "job": job,
            **{f"natural_{key}": value for key, value in vector_metrics(changed, natural_target, natural_donor).items()},
            **{f"exact_{key}": value for key, value in vector_metrics(changed, identity, exact).items()},
            "k12_recovery": float(tensors[f"{direction}.{job}.k12_recovery"]),
            "k12_residual_norm_ratio": float(tensors[f"{direction}.{job}.k12_residual_norm_ratio"]),
            "k12_effect_norm": float(tensors[f"{direction}.{job}.k12_effect_norm"]),
        }
        examples.append(values)
    summary = {}
    keys = sorted(set(examples[0]) - {"example_id", "concept", "direction", "job"})
    for key in keys:
        median, concepts = macro(examples, key)
        summary[key] = {"median_concept": median, "by_concept": concepts}
    summary["natural_donor_nearest_concepts"] = sum(value > 0.5 for value in summary["natural_donor_nearest"]["by_concept"].values())
    summary["exact_donor_nearest_concepts"] = sum(value > 0.5 for value in summary["exact_donor_nearest"]["by_concept"].values())
    return summary, examples


def main() -> None:
    contract = expanded_contract()
    rows, audit = load_rows(contract)
    by_direction = {}
    example_metrics = []
    for direction in contract["conditions"]["directions"]:
        jobs = {}
        for job in contract["jobs_per_direction"]:
            jobs[job], examples = summarize_job(rows, direction, job)
            example_metrics.extend(examples)
        by_direction[direction] = {"jobs": jobs}

    context_qualification = {}
    context_selected = None
    context_gates = contract["selection"]["context_gate_both_directions"]
    for candidate in contract["selection"]["context_candidate_order"]:
        pair = f"exact_plus_{candidate}"
        pair_orthogonal = f"exact_plus_{candidate}_orthogonal"
        directional = {}
        for direction, payload in by_direction.items():
            jobs = payload["jobs"]
            values = {
                "paired_natural_probe_recovery": jobs[pair]["natural_recovery"]["median_concept"],
                "increment_over_exact_k12": jobs[pair]["natural_recovery"]["median_concept"] - jobs["exact_donor_k12"]["natural_recovery"]["median_concept"],
                "paired_advantage_over_orthogonal": jobs[pair]["natural_recovery"]["median_concept"] - jobs[pair_orthogonal]["natural_recovery"]["median_concept"],
                "paired_natural_probe_residual_norm_ratio": jobs[pair]["natural_residual_norm_ratio"]["median_concept"],
            }
            passed = {
                "paired_natural_probe_recovery": values["paired_natural_probe_recovery"] >= float(context_gates["paired_natural_probe_recovery_min"]),
                "increment_over_exact_k12": values["increment_over_exact_k12"] >= float(context_gates["increment_over_exact_k12_min"]),
                "paired_advantage_over_orthogonal": values["paired_advantage_over_orthogonal"] >= float(context_gates["paired_advantage_over_orthogonal_min"]),
                "paired_natural_probe_residual_norm_ratio": values["paired_natural_probe_residual_norm_ratio"] <= float(context_gates["paired_natural_probe_residual_norm_ratio_max"]),
            }
            directional[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values())}
        qualifies = all(value["all_pass"] for value in directional.values())
        context_qualification[candidate] = {"directions": directional, "qualifies": qualifies}
        if context_selected is None and qualifies:
            context_selected = candidate

    full_tail_pass = all(
        payload["jobs"]["full_tail"]["natural_recovery"]["median_concept"] >= 0.98
        for payload in by_direction.values()
    )
    context_disposition = context_selected or (
        "distributed_tail_complement" if full_tail_pass else "unresolved_residual_context"
    )

    pathway_qualification = {}
    pathway_selected = None
    pathway_gates = contract["selection"]["pathway_gate_both_directions"]
    for candidate in contract["selection"]["pathway_candidate_order"]:
        directional = {}
        for direction, payload in by_direction.items():
            jobs = payload["jobs"]
            values = {
                "probe_recovery_to_exact_k12": jobs[candidate]["exact_recovery"]["median_concept"],
                "k12_recovery": jobs[candidate]["k12_recovery"]["median_concept"],
                "probe_advantage_over_exact_orthogonal": jobs[candidate]["exact_recovery"]["median_concept"] - jobs["exact_k12_orthogonal"]["exact_recovery"]["median_concept"],
                "probe_exact_nearest_concepts": jobs[candidate]["exact_donor_nearest_concepts"],
            }
            passed = {
                "probe_recovery_to_exact_k12": values["probe_recovery_to_exact_k12"] >= float(pathway_gates["probe_recovery_to_exact_k12_min"]),
                "k12_recovery": values["k12_recovery"] >= float(pathway_gates["k12_recovery_min"]),
                "probe_advantage_over_exact_orthogonal": values["probe_advantage_over_exact_orthogonal"] >= float(pathway_gates["probe_advantage_over_exact_orthogonal_min"]),
                "probe_exact_nearest_concepts": values["probe_exact_nearest_concepts"] >= int(pathway_gates["probe_exact_nearest_concepts_min"]),
            }
            directional[direction] = {"values": values, "passed": passed, "all_pass": all(passed.values())}
        qualifies = all(value["all_pass"] for value in directional.values())
        pathway_qualification[candidate] = {"directions": directional, "qualifies": qualifies}
        if pathway_selected is None and qualifies:
            pathway_selected = candidate

    summary = {
        "schema_version": 1,
        "procedure": "day58-k12-residual-context-and-pathway-development-reduction-v1",
        "execution_commit": read_json(EXECUTION_PATH)["execution_commit"],
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "implementation_audit": audit,
        "directions": by_direction,
        "context_selection": {
            "qualification": context_qualification,
            "selected_compact_context": context_selected,
            "full_tail_positive_control_pass": full_tail_pass,
            "disposition": context_disposition,
        },
        "pathway_selection": {
            "qualification": pathway_qualification,
            "selected_pathway": pathway_selected,
            "disposition": pathway_selected or "unresolved_prefix_interface",
        },
    }
    write_json_atomic(METRICS_PATH, {"schema_version": 1, "rows": example_metrics})
    summary["example_metrics_sha256"] = sha256_file(METRICS_PATH)
    summary["artifact_manifest_sha256"] = sha256_file(MANIFEST_PATH)
    write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps({
        "context": summary["context_selection"],
        "pathway": summary["pathway_selection"],
        "audit": audit,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
