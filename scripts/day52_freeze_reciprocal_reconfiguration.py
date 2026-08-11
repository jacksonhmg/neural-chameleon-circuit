#!/usr/bin/env python3
"""Freeze the prospective Day 52 reciprocal donor-reconfiguration test."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAY49_CONTRACT = ROOT / "results/day-49/frozen-prompt-memory-contract.json"
DAY51_CONTRACT = ROOT / "results/day-51/frozen-donor-identity-audit.json"
DAY51_SUMMARY = ROOT / "results/day-51/donor-identity-summary.json"
DAY51_AUDIT = ROOT / "results/day-51/donor-identity-audit.json"
DAY51_DECISION = ROOT / "decision-log/0078-promote-donor-reconfiguration-hypothesis.md"
OUTPUT = ROOT / "results/day-52/frozen-reciprocal-reconfiguration-contract.json"


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
    day49 = read_json(DAY49_CONTRACT)
    summary = read_json(DAY51_SUMMARY)
    audit = read_json(DAY51_AUDIT)
    if (
        not audit["implementation_pass"]
        or summary["disposition"] != "exploratory_donor_reconfiguration_supported"
        or not summary["day49_disposition_unchanged"]
    ):
        raise RuntimeError("Day 52 reciprocal test is not eligible")
    directions = {
        "irrelevant_to_correct": {
            "target": "correct_trigger",
            "donor": "irrelevant_trigger",
            "different_control": "different_trigger",
            "normal_control": "normal",
        },
        "correct_to_irrelevant": {
            "target": "irrelevant_trigger",
            "donor": "correct_trigger",
            "different_control": "different_trigger",
            "normal_control": "normal",
        },
    }
    jobs = ["identity", "haar", "normal_collapse", "different_donor", "primary_donor"]
    examples = int(day49["population"]["example_count"])
    states_per_example = 4 + len(directions) * len(jobs)
    return {
        "schema_version": 1,
        "status": "frozen_before_day52_reciprocal_outcomes",
        "procedure": "prospective-day52-reciprocal-full-prefix-qkv-v1",
        "frozen_on": "2026-08-11",
        "frozen_at_commit": git_head(),
        "authorization": "user explicitly instructed execution after reviewing the donor-identity result",
        "evidence_class": "new intervention outcomes on the unchanged previously inspected 26-example development sandbox; not fresh confirmation",
        "parents": {
            path.stem: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (
                DAY49_CONTRACT,
                DAY51_CONTRACT,
                DAY51_SUMMARY,
                DAY51_AUDIT,
                DAY51_DECISION,
            )
        },
        "model": day49["model"],
        "probes": day49["probes"],
        "population": day49["population"],
        "conditions": day49["conditions"],
        "k12": day49["k12"],
        "candidate": {
            "id": "full_monitoring_prefix_qkv",
            "interface": day49["attention_interface"],
            "selection": "single unchanged exploratory hypothesis from Day 51; no candidate search",
            "region_alignment": "Decision 0074 endpoint-aligned nearest-source token mapping",
        },
        "directions": directions,
        "jobs": {
            "order": jobs,
            "identity": "exact target K12 heads",
            "haar": {
                "definition": "per-head Haar rotation of the exact natural target-to-primary-donor K12 trajectory, added to target",
                "base_seed": 52001,
                "draw_index": 0,
            },
            "normal_collapse": "unchanged full-prefix QKV operation from the no-prefix normal source",
            "different_donor": "unchanged full-prefix QKV operation from the frozen different-trigger source",
            "primary_donor": "unchanged full-prefix QKV operation from the direction's exact donor source",
        },
        "primary_endpoint": {
            "name": "exact natural donor K12 trajectory recovery",
            "formula": "dot(intervention - target, donor - target) / max(norm(donor - target)^2, 1e-8)",
            "reduction": "response-mask weighted within example; arithmetic mean of two examples per concept; median across 13 equal-weight concepts",
            "controls": "score Haar, normal-collapse, and different-donor interventions against the same exact primary-donor trajectory",
        },
        "donor_identity_endpoints": {
            "primary_donor": "the exact natural donor condition on the same response",
            "target": "the exact natural recipient condition",
            "normal": "the exact natural no-trigger condition",
            "different": "the exact frozen different-trigger condition",
            "distance": "masked squared Euclidean distance in K12 and complete monitor residual; ordinary squared Euclidean distance in the 13-probe vector",
        },
        "promotion_gate": {
            "all_clauses_required_in_both_directions": True,
            "median_concept_k12_donor_recovery_min": 0.70,
            "advantage_over_strongest_haar_normal_or_different_control_min": 0.25,
            "k12_positive_donor_recovery_concepts_min": 10,
            "k12_donor_closer_than_target_normal_and_different_concepts_min": 10,
            "median_concept_probe_vector_donor_recovery_min": 0.50,
            "probe_vector_donor_closer_than_target_normal_and_different_concepts_min": 10,
            "donor_own_probe_direction_correct_concepts_min": 10,
            "median_natural_donor_separation_ratio_min": 0.25,
            "positive_median_aligned_k12_layers_min": 2,
            "activation_rms_ratio_range": [0.5, 1.5],
            "raw_aligned_numerator_concentration": "report but do not gate because response length and target magnitude scale the raw numerator; equal-concept median and direction counts are the prospective heterogeneity gates",
            "pass_consequence": "freeze K12 mediation for this unchanged interface only",
            "fail_consequence": "stop the full-prefix donor-reconfiguration hypothesis",
        },
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
            "two_reductions_byte_identical": True,
            "exact_state_row_count": examples * states_per_example,
        },
        "execution": {
            "batch_size": 2,
            "concept_shards": 13,
            "directions": list(directions),
            "jobs_per_direction": len(jobs),
            "natural_states": list(day49["conditions"]["natural_conditions"]),
            "states_per_example": states_per_example,
            "total_state_rows": examples * states_per_example,
            "resumption": "atomic concept safetensors shards",
            "preflight": "one inspected concept; same-condition QKV recomputation, identity, Haar, response, finiteness, and hooks only",
        },
        "required_outputs": [
            "results/day-52/reciprocal-reconfiguration-preflight.json",
            "artifacts/rapid-k12-upstream-v1/day52-shards/*.safetensors",
            "results/day-52/reciprocal-reconfiguration-execution.json",
            "results/day-52/reciprocal-reconfiguration-summary.json",
            "results/day-52/reciprocal-reconfiguration-audit.json",
            "results/day-52/execution-artifact-manifest.json",
        ],
        "forbidden": [
            "changing the interface, donors, controls, endpoints, metrics, gates, seeds, or weighting after outcomes",
            "treating wrong donors as inert rather than scoring their own endpoint",
            "reopening candidate search",
            "K12 mediation unless both reciprocal directions pass",
            "fresh data, behavior, generation, title upgrade, or dissemination",
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
