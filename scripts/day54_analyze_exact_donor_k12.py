#!/usr/bin/env python3
"""Reduce the frozen Day 54 exact-natural-donor K12 experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import day52_analyze_reciprocal_reconfiguration as reducer
from day54_run_exact_donor_k12 import expanded_contract


ROOT = Path(__file__).resolve().parents[1]
reducer.CONTRACT_PATH = ROOT / "results/day-54/frozen-exact-donor-k12-contract.json"
reducer.PREFLIGHT_PATH = ROOT / "results/day-54/exact-donor-k12-preflight.json"
reducer.EXECUTION_PATH = ROOT / "results/day-54/exact-donor-k12-execution.json"
reducer.SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day54-shards"
reducer.SUMMARY_PATH = ROOT / "results/day-54/exact-donor-k12-summary.json"
reducer.AUDIT_PATH = ROOT / "results/day-54/exact-donor-k12-audit.json"
reducer.METRICS_PATH = ROOT / "results/day-54/exact-donor-k12-example-metrics.json"
reducer.MANIFEST_PATH = ROOT / "results/day-54/execution-artifact-manifest.json"


original_read_json = reducer.read_json


def read_json_with_inheritance(path: Path) -> dict[str, Any]:
    if path == reducer.CONTRACT_PATH:
        return expanded_contract()
    return original_read_json(path)


def reduce_result(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    concepts = reducer.concept_means(rows)
    directions = {}
    all_clauses = {}
    for direction in contract["directions"]:
        directions[direction], clauses = reducer.summarize_direction(
            direction, concepts, contract
        )
        all_clauses[direction] = clauses
    scientific_pass = all(value["gate_pass"] for value in directions.values())
    implementation_gates = contract["implementation_gates"]
    implementation_checks = {
        "preflight_pass": audit_inputs["preflight"]["result"] == "pass",
        "execution_complete": bool(audit_inputs["execution"]["complete"]),
        "exact_state_row_count": audit_inputs["state_rows"]
        == int(implementation_gates["exact_state_row_count"]),
        "exact_probe_order": len(audit_inputs["probe_name_sets"]) == 1
        and len(audit_inputs["probe_name_sets"][0]) == 13,
        "source_replacement_construction_exact": audit_inputs["preflight"][
            "source_replacement_construction_max_abs"
        ]
        <= float(implementation_gates["source_replacement_construction_max_abs"]),
        "identity_k12_within_tolerance": audit_inputs["identity_k12_max_abs"]
        <= float(implementation_gates["identity_k12_max_abs"]),
        "identity_margin_within_tolerance": audit_inputs["identity_margin_max_abs"]
        <= float(implementation_gates["identity_monitor_margin_max_abs"]),
        "random_audits_pass": bool(audit_inputs["random_audits_pass"]),
        "hooks_removed": audit_inputs["execution"]["hooks_after_execution"] == 0,
        "thirteen_unique_tensor_hashes": audit_inputs["tensor_hash_count"] == 13,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "k12_donor_recovery",
                "k12_donor_cosine",
                "k12_donor_norm_ratio",
                "monitor_donor_recovery",
                "probe_vector_donor_recovery",
                "activation_rms_ratio",
            )
        ),
    }
    implementation_pass = all(implementation_checks.values())
    branch = (
        "exact_k12_self_contained_pass"
        if implementation_pass and scientific_pass
        else "exact_k12_self_contained_fail"
        if implementation_pass
        else None
    )
    disposition = (
        "run_qkv_to_exact_k12_completion_localization"
        if branch == "exact_k12_self_contained_pass"
        else "run_k12_by_residual_context_factorial"
        if branch == "exact_k12_self_contained_fail"
        else "implementation_failure_no_scientific_interpretation"
    )
    summary = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "contract_sha256": reducer.sha256_file(reducer.CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": reducer.git_head(),
        "evidence_class": contract["evidence_class"],
        "candidate": contract["candidate"]["id"],
        "direction_summaries": directions,
        "scientific_gate_clauses": all_clauses,
        "scientific_gate_pass": scientific_pass,
        "implementation_gate_pass": implementation_pass,
        "disposition": disposition,
        "branch": branch,
        "mediation_eligible": False,
        "boundary": "development sandbox only; not fresh confirmation",
    }
    audit = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-audit",
        "contract_sha256": reducer.sha256_file(reducer.CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": reducer.git_head(),
        "implementation_checks": implementation_checks,
        "implementation_pass": implementation_pass,
        "observed": {
            "example_metric_rows": len(rows),
            "state_rows": audit_inputs["state_rows"],
            "source_replacement_construction_max_abs": audit_inputs["preflight"][
                "source_replacement_construction_max_abs"
            ],
            "identity_k12_max_abs": audit_inputs["identity_k12_max_abs"],
            "identity_margin_max_abs": audit_inputs["identity_margin_max_abs"],
            "probe_name_sets": audit_inputs["probe_name_sets"],
            "tensor_hash_count": audit_inputs["tensor_hash_count"],
        },
        "two_in_memory_reductions_byte_identical": None,
    }
    metrics = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-example-metrics",
        "contract_sha256": reducer.sha256_file(reducer.CONTRACT_PATH),
        "rows": list(rows),
    }
    return summary, audit, metrics


reducer.read_json = read_json_with_inheritance
reducer.reduce_result = reduce_result


if __name__ == "__main__":
    reducer.main()
