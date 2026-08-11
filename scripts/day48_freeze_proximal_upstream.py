#!/usr/bin/env python3
"""Freeze the exact Day 48 proximal upstream-controller experiment."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAY41_PATH = ROOT / "results/day-41/frozen-phase-c-contract.json"
DAY44_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
DAY47_PATH = ROOT / "results/day-47/frozen-heldout-development-contract.json"
DAY47_SUMMARY_PATH = ROOT / "results/day-47/heldout-summary.json"
MASTER_PLAN_PATH = ROOT / "docs/rapid-k12-upstream-controller-research-plan.md"
AUTHORIZATION_PATH = ROOT / "decision-log/0070-authorize-full-upstream-controller-program.md"
OUTPUT_PATH = ROOT / "results/day-48/frozen-proximal-upstream-contract.json"


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
    day41 = read_json(DAY41_PATH)
    day44 = read_json(DAY44_PATH)
    day47 = read_json(DAY47_PATH)
    if day41["status"] != "frozen-before-phase-c-outcomes":
        raise RuntimeError("Phase C parent is not frozen")
    if day44["status"] != "frozen_before_new_k12_candidate_outcomes":
        raise RuntimeError("Day 44 parent is not frozen")
    if day47["status"] != "frozen_before_any_day47_model_or_intervention_outcome":
        raise RuntimeError("Day 47 parent is not frozen")

    components = list(day44["component_set"])
    if components != day41["operation"]["component_ids"]:
        raise RuntimeError("selected K12 component order differs across parents")
    examples = [dict(row) for row in day44["pilot"]["examples"]]
    counts = Counter(row["concept"] for row in examples)
    if len(examples) != 26 or set(counts.values()) != {2}:
        raise RuntimeError("Day 44 pilot is not exactly two examples per concept")
    pairs = {
        concept: dict(day41["conditions"]["pairs"][concept])
        for concept in sorted(counts)
    }
    if any(value["correct_concept"] != concept for concept, value in pairs.items()):
        raise RuntimeError("Phase C correct concepts differ")

    natural_conditions = ["normal", "correct_trigger", "irrelevant_trigger", "different_trigger"]
    normal_jobs = [
        "identity_normal",
        "correct_response_to_normal",
        "irrelevant_response_to_normal",
        "random_response_to_normal",
    ]
    correct_jobs = [
        "identity_correct",
        "normal_response_to_correct",
        "irrelevant_response_to_correct",
        "random_response_to_correct",
    ]
    total_state_rows = len(examples) * (
        len(natural_conditions) + len(normal_jobs) + len(correct_jobs) + 1
    )
    return {
        "schema_version": 1,
        "status": "frozen_before_day48_upstream_outcomes",
        "procedure": "rapid-k12-proximal-upstream-controller-day48-v1",
        "frozen_on": "2026-08-10",
        "frozen_at_commit": git_head(),
        "evidence_class": "previously inspected 26-example development sandbox; not fresh confirmation",
        "authority": {
            "master_plan": MASTER_PLAN_PATH.relative_to(ROOT).as_posix(),
            "master_plan_sha256": sha256_file(MASTER_PLAN_PATH),
            "authorization": AUTHORIZATION_PATH.relative_to(ROOT).as_posix(),
            "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
            "execution_scope": "Days 48-50 are authorized subject to their frozen branch gates",
            "fresh_data": "prohibited",
            "dissemination": "local artifacts and computation worker only; no push, release, submission, author contact, or external messaging",
        },
        "parents": {
            "day41_phase_c_contract": {
                "path": DAY41_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(DAY41_PATH),
            },
            "day44_k12_contract": {
                "path": DAY44_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(DAY44_PATH),
            },
            "day47_k12_contract": {
                "path": DAY47_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(DAY47_PATH),
            },
            "day47_summary": {
                "path": DAY47_SUMMARY_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(DAY47_SUMMARY_PATH),
            },
        },
        "immutable_prior_results": {
            "k12_operation": "the exact 12-head concept/response-position additive writer validated on Day 47",
            "phase_c": "the layer-8 named-concept prompt span failed specificity and remains a negative comparator",
            "parameter_attribution": "selected QKV routing and O write geometry explain minority approximately additive shares; most implementation remains unresolved",
        },
        "model": day44["model"],
        "probes": {
            "count": 13,
            "path": "external/minimal_neural_chameleons/probes",
            "endpoint": "complete mean raw-margin vector at resid_post[12]",
        },
        "population": {
            "source": "unchanged Day 44 causal pilot",
            "concept_count": 13,
            "examples_per_concept": 2,
            "example_count": 26,
            "examples": examples,
            "weighting": "equal valid response tokens within example, equal two examples within concept, equal concepts for macro summaries",
        },
        "conditions": {
            "pairs": pairs,
            "natural_conditions": natural_conditions,
            "teacher_forcing": "one exact response-tokenization reused across normal, correct, irrelevant, and different trigger conditions",
        },
        "causal_interface": {
            "site": "resid_pre[9] (output of layer 8/input to layer 9)",
            "token_rows": "all valid response-token rows only",
            "coordinate_width": 3584,
            "sufficiency": "replace normal response rows with same-example correct-trigger resid_pre[9] rows",
            "necessity": "replace correct-trigger response rows with same-example normal resid_pre[9] rows",
            "specificity": "replace each target with same-example irrelevant-trigger resid_pre[9] rows",
            "identity": "replace each target with its own exact resid_pre[9] response rows",
            "random_control": {
                "definition": "apply one fixed signed coordinate permutation to the same-example correct-minus-normal resid_pre[9] response delta; add for sufficiency and subtract for necessity",
                "seed": 48001,
                "invariants": [
                    "per-token L2 norm exact in float32",
                    "within-example temporal Gram exact in float32 tolerance",
                    "equal dimension and response mask",
                ],
            },
            "failed_span_comparator": {
                "definition": "repeat the Phase C different-trigger to correct-trigger named-concept prompt-span substitution at resid_pre[9]",
                "role": "fixed adverse comparator only; cannot be promoted as the response-state controller",
            },
        },
        "k12": {
            "component_ids": components,
            "layers": [9, 10, 11, 12],
            "coordinate": "valid response-token pre-o_proj query-head output",
            "head_width": 256,
        },
        "primary_endpoint": {
            "name": "directional K12 trajectory recovery",
            "target": "natural donor-condition K12 minus natural recipient-condition K12",
            "intervention": "intervention K12 minus unmodified recipient-condition K12",
            "formula": "dot(intervention_delta,target_delta)/max(dot(target_delta,target_delta),1e-8)",
            "reduction": "one response-mask-weighted value per example, arithmetic mean within concept, median across 13 concepts for the main gate",
            "additional_reports": [
                "cosine",
                "norm ratio",
                "per-concept recovery",
                "per-layer and per-head aligned numerator",
                "complete signed unclipped values",
            ],
        },
        "secondary_endpoints": [
            "complete 13-probe mean raw-margin vector",
            "own-probe directional agreement",
            "response activation RMS",
            "prototype-direction recovery using the unchanged Day 47 operator",
            "fixed Phase C named-span comparator",
        ],
        "jobs": {
            "natural_conditions": natural_conditions,
            "normal_target": normal_jobs,
            "correct_target": correct_jobs,
            "separate_failed_span_comparator": ["different_named_span_to_correct"],
        },
        "implementation_gates": {
            "backend": "CUDA only",
            "environment": "environment/day-01/uv.lock under Python 3.14.4",
            "dtype": "bfloat16 model execution; float32 interventions and reductions",
            "attention": "eager",
            "response_ids_and_masks_exact": True,
            "same_condition_identity_k12_max_abs": 0.02,
            "same_condition_identity_monitor_margin_max_abs": 0.05,
            "signed_permutation_norm_relative_error_max": 1e-6,
            "signed_permutation_gram_relative_error_max": 1e-5,
            "all_values_finite": True,
            "hooks_after_batch": 0,
            "exact_state_row_count": total_state_rows,
            "two_reductions_byte_identical": True,
            "failure": "repair implementation without scientific interpretation",
        },
        "promotion_gate": {
            "sufficiency_median_concept_recovery_min": 0.50,
            "necessity_median_concept_recovery_min": 0.50,
            "advantage_over_strongest_control_min": 0.25,
            "k12_direction_correct_concepts_min_per_direction": 10,
            "own_probe_direction_correct_concepts_min_per_direction": 10,
            "natural_own_probe_direction_correct_concepts_min": 10,
            "activation_rms_ratio_range": [0.5, 1.5],
            "distributed_support": {
                "positive_median_aligned_k12_layers_min": 2,
                "maximum_single_concept_fraction_of_total_aligned_numerator": 0.25,
            },
            "all_clauses_required": True,
            "pass_consequence": "freeze Day 49 backward localization on the response interface",
            "fail_consequence": "freeze Day 49 selected prompt-memory branch without a broad residual scan",
        },
        "execution": {
            "batch_size": 2,
            "ordering": "contract example order, already concept-paired",
            "model_residency": "one complete Chameleon model",
            "one_forward_per_natural_condition_per_batch": True,
            "vectorized_jobs_per_target": 4,
            "random_seed": 48001,
            "resumption": "atomic concept tensor shards with exact provenance and state keys",
            "preflight": "one previously inspected example; identity, geometry, finiteness, response alignment, and hook cleanup only",
        },
        "expected_execution_matrix": {
            "natural_state_rows": len(examples) * len(natural_conditions),
            "normal_target_intervention_rows": len(examples) * len(normal_jobs),
            "correct_target_intervention_rows": len(examples) * len(correct_jobs),
            "failed_span_comparator_rows": len(examples),
            "total_state_rows": total_state_rows,
            "concept_shards": 13,
        },
        "forbidden": [
            "changing examples, triggers, jobs, endpoint definitions, gates, or seeds after outcomes",
            "promoting the failed Phase C span",
            "using sigmoid, TPR, behavior, generation, attention maps, new probes, or fresh data for selection",
            "running backward localization before a response-interface pass",
            "running broad residual searches after a response-interface fail",
            "mixing MPS and CUDA rows",
        ],
        "branch_rule": "apply every promotion clause mechanically; exactly one of response localization or selected prompt-memory testing becomes eligible",
        "required_artifacts": [
            "results/day-48/proximal-upstream-preflight.json",
            "artifacts/rapid-k12-upstream-v1/day48-shards/*.safetensors",
            "results/day-48/proximal-upstream-execution.json",
            "results/day-48/proximal-upstream-summary.json",
            "results/day-48/proximal-upstream-audit.json",
            "results/day-48/execution-artifact-manifest.json",
            "lab-notes/day-48-proximal-upstream-interface.md",
        ],
    }


def main() -> None:
    contract = build_contract()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(OUTPUT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
