#!/usr/bin/env python3
"""Freeze the eligible Day 49 selected prompt-memory experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAY48_CONTRACT = ROOT / "results/day-48/frozen-proximal-upstream-contract.json"
DAY48_SUMMARY = ROOT / "results/day-48/proximal-upstream-summary.json"
DAY48_AUDIT = ROOT / "results/day-48/proximal-upstream-audit.json"
DAY48_MANIFEST = ROOT / "results/day-48/execution-artifact-manifest.json"
PLAN = ROOT / "docs/rapid-k12-upstream-controller-research-plan.md"
DECISION = ROOT / "decision-log/0072-reject-proximal-response-controller.md"
OUTPUT = ROOT / "results/day-49/frozen-prompt-memory-contract.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def build_contract() -> dict[str, Any]:
    parent = read_json(DAY48_CONTRACT)
    summary = read_json(DAY48_SUMMARY)
    audit = read_json(DAY48_AUDIT)
    if (
        not audit["implementation_pass"]
        or summary["scientific_gate_pass"]
        or summary["disposition"] != "freeze_selected_prompt_memory_branch"
    ):
        raise RuntimeError("Day 49 prompt-memory branch is not eligible")
    candidates = [
        "response_query_q_only",
        "generic_prefix_kv",
        "named_concept_kv",
        "previous_response_kv",
        "full_monitoring_prefix_kv",
        "generic_prefix_qkv",
        "named_concept_qkv",
        "previous_response_qkv",
        "full_monitoring_prefix_qkv",
    ]
    job_count = 2 * len(candidates) + 2
    example_count = int(parent["population"]["example_count"])
    return {
        "schema_version": 1,
        "status": "frozen_before_day49_prompt_memory_outcomes",
        "procedure": "rapid-k12-selected-prompt-memory-day49-v1",
        "frozen_on": "2026-08-11",
        "frozen_at_commit": git_head(),
        "evidence_class": "unchanged previously inspected 26-example development sandbox; not fresh confirmation",
        "authority": {
            "master_plan": PLAN.relative_to(ROOT).as_posix(),
            "master_plan_sha256": sha256_file(PLAN),
            "branch_decision": DECISION.relative_to(ROOT).as_posix(),
            "branch_decision_sha256": sha256_file(DECISION),
            "fresh_data": "prohibited",
            "dissemination": "local artifacts and computation worker only",
        },
        "parents": {
            path.stem: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (
                DAY48_CONTRACT,
                DAY48_SUMMARY,
                DAY48_AUDIT,
                DAY48_MANIFEST,
            )
        },
        "immutable_day48_disposition": {
            "response_interface": "failed and cannot be reopened",
            "backward_residual_scan": "ineligible",
            "eligible_branch": "selected prompt-memory interfaces only",
        },
        "model": parent["model"],
        "probes": parent["probes"],
        "population": parent["population"],
        "conditions": parent["conditions"],
        "k12": parent["k12"],
        "attention_interface": {
            "layers": [9, 10, 11, 12],
            "heads": "the exact selected K12 query heads at each layer",
            "captured_coordinates": [
                "post-RoPE query states",
                "post-RoPE key states",
                "pre-attention value states",
                "eager attention mask and softcap",
                "raw pre-o_proj query-head output",
            ],
            "source_regions": {
                "generic_prefix": ["monitoring_language", "trigger_other"],
                "named_concept": ["named_concept"],
                "full_monitoring_prefix": [
                    "monitoring_language",
                    "trigger_other",
                    "named_concept",
                ],
                "previous_response": ["response"],
            },
            "q_only": "replace aligned response queries from source while retaining every target K/V state and the target causal mask",
            "kv": "replace aligned source-region K/V when both conditions contain the region; when the normal target lacks a monitoring-prefix region, add the exact source natural-minus-region-removed raw-head contribution; reverse direction removes the target region",
            "qkv": "apply the q-only response query together with the corresponding K/V replacement, addition, or removal",
            "region_removal": "set both selected-region K and V states to exact zero before eager-attention recomputation",
            "position_handling": "queries and keys are transplanted post-RoPE in their natural source coordinate; target attention mask and softcap remain fixed",
            "causal_realization": "patch only the recomputed selected raw K12 head outputs and allow the ordinary total path through resid_post[12]",
        },
        "candidates_in_simplicity_order": candidates,
        "controls": {
            "identity": "exact target K12 heads",
            "irrelevant": "the same operation using the frozen irrelevant-trigger source",
            "random": "one frozen per-head Haar rotation of the exact natural K12 trajectory at equal dimension and norm",
            "random_base_seed": 49001,
            "random_draw_index": 0,
        },
        "directions": {
            "sufficiency": "correct-trigger prompt-memory operation into normal",
            "necessity": "normal prompt-memory operation into correct trigger",
        },
        "primary_endpoint": parent["primary_endpoint"],
        "secondary_endpoints": parent["secondary_endpoints"],
        "implementation_gates": {
            "backend": "CUDA only",
            "environment": "environment/day-01/uv.lock under Python 3.14.4",
            "dtype": "bfloat16 model execution; float32 attention recomputation and reductions",
            "attention": "eager",
            "natural_attention_recompute_max_abs": 0.05,
            "identity_k12_max_abs": 0.02,
            "identity_monitor_margin_max_abs": 0.05,
            "response_ids_and_masks_exact": True,
            "all_values_finite": True,
            "haar_invariants_pass": True,
            "hooks_after_batch": 0,
            "exact_state_row_count": example_count * (4 + 2 * job_count),
            "two_reductions_byte_identical": True,
        },
        "promotion_gate": {
            "median_concept_k12_recovery_min_per_direction": 0.50,
            "advantage_over_strongest_irrelevant_or_random_control_min_per_direction": 0.25,
            "k12_direction_correct_concepts_min_per_direction": 10,
            "own_probe_direction_correct_concepts_min_per_direction": 10,
            "activation_rms_ratio_range": [0.5, 1.5],
            "positive_median_aligned_k12_layers_min_per_direction": 2,
            "maximum_single_concept_fraction_of_total_aligned_numerator": 0.25,
            "all_clauses_required": True,
            "selection": "select the first passing candidate in frozen simplicity order; at most one",
            "pass_consequence": "freeze concept reconfiguration and K12 mediation only for the unchanged winner",
            "fail_consequence": "stop Day 49 with no compact tested controller; Day 50 is ineligible",
        },
        "execution": {
            "batch_size": 2,
            "jobs_per_target": job_count,
            "job_order": [
                "identity",
                "random",
                *[f"primary.{value}" for value in candidates],
                *[f"irrelevant.{value}" for value in candidates],
            ],
            "model_residency": "one complete Chameleon model",
            "resumption": "atomic concept tensor shards with exact provenance and state keys",
            "preflight": "one previously inspected concept; attention recomputation, identity, Haar, response, finiteness, and hooks only",
        },
        "expected_execution_matrix": {
            "concept_shards": 13,
            "natural_state_rows": example_count * 4,
            "intervention_state_rows": example_count * 2 * job_count,
            "total_state_rows": example_count * (4 + 2 * job_count),
            "candidates": len(candidates),
            "jobs_per_target": job_count,
        },
        "forbidden": [
            "changing candidates, order, source regions, operations, controls, gates, or seeds after outcomes",
            "adding a broad residual, head, MLP, attention-map, or probe search",
            "promoting monitor movement without natural K12 trajectory recovery",
            "behavior, generation, new probes, fresh data, or title claims",
        ],
        "required_artifacts": [
            "results/day-49/prompt-memory-preflight.json",
            "artifacts/rapid-k12-upstream-v1/day49-shards/*.safetensors",
            "results/day-49/prompt-memory-execution.json",
            "results/day-49/prompt-memory-summary.json",
            "results/day-49/prompt-memory-audit.json",
            "results/day-49/execution-artifact-manifest.json",
        ],
    }


def main() -> None:
    contract = build_contract()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
