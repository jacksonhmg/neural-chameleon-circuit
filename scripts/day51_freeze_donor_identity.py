#!/usr/bin/env python3
"""Freeze the exploratory Day 51 donor-identity audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DAY49_CONTRACT = ROOT / "results/day-49/frozen-prompt-memory-contract.json"
DAY49_SUMMARY = ROOT / "results/day-49/prompt-memory-summary.json"
DAY49_AUDIT = ROOT / "results/day-49/prompt-memory-audit.json"
DAY49_MANIFEST = ROOT / "results/day-49/execution-artifact-manifest.json"
OUTPUT = ROOT / "results/day-51/frozen-donor-identity-audit.json"


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
    parent = read_json(DAY49_CONTRACT)
    summary = read_json(DAY49_SUMMARY)
    audit = read_json(DAY49_AUDIT)
    if (
        not audit["implementation_pass"]
        or summary["disposition"] != "stop_no_compact_tested_controller"
        or summary["day50_eligible"]
    ):
        raise RuntimeError("the bounded Day 49 stop result is not intact")
    return {
        "schema_version": 1,
        "status": "frozen_before_donor_identity_metrics",
        "procedure": "exploratory-day51-donor-identity-audit-v1",
        "frozen_on": "2026-08-11",
        "frozen_at_commit": git_head(),
        "authorization": "user explicitly requested this diagnostic after the completed Day 49 stop",
        "evidence_class": "post-hoc exploratory reuse of the already inspected Day 49 development tensors; cannot promote or overturn Day 49",
        "question": "does irrelevant full-prefix QKV replacement collapse K12 toward normal or reconfigure it toward the exact natural irrelevant-donor condition?",
        "parents": {
            path.stem: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in (
                DAY49_CONTRACT,
                DAY49_SUMMARY,
                DAY49_AUDIT,
                DAY49_MANIFEST,
            )
        },
        "population": parent["population"],
        "condition_pairs": parent["conditions"]["pairs"],
        "candidate": "full_monitoring_prefix_qkv",
        "intervention_state": "intervention_necessity.irrelevant.full_monitoring_prefix_qkv",
        "recipient_state": "natural_correct_trigger",
        "endpoints": {
            "normal": "natural_normal",
            "donor": "natural_irrelevant_trigger",
            "different": "natural_different_trigger",
        },
        "modalities": [
            "k12",
            "complete_monitor_residual",
            "complete_13_probe_margin_vector",
        ],
        "metrics_per_example": {
            "endpoint_squared_distances": "masked squared Euclidean distance from intervention to normal, donor, and different endpoints",
            "donor_vs_normal_preference": "(distance_to_normal - distance_to_donor) / max(squared_distance(donor, normal), 1e-8); positive favors donor",
            "donor_vs_different_preference": "(distance_to_different - distance_to_donor) / max(squared_distance(donor, different), 1e-8); positive favors donor",
            "natural_donor_separation_ratio": "norm(donor - normal) / max(norm(correct - normal), 1e-8)",
            "donor_recovery": "dot(intervention - correct, donor - correct) / max(norm(donor - correct)^2, 1e-8)",
            "normal_recovery": "dot(intervention - correct, normal - correct) / max(norm(normal - correct)^2, 1e-8)",
            "nearest_endpoint": "minimum unnormalized squared distance among normal, donor, and different",
            "source_probe_direction": "sign agreement between intervention-minus-normal and natural-donor-minus-normal at the frozen irrelevant donor's own probe",
        },
        "reduction": "mask valid response rows for K12 and monitor residual; one value per example; arithmetic mean over the exact two examples per concept; medians and sign counts across 13 equal-weight concepts",
        "interpretive_rule": {
            "minimum_natural_donor_separation_ratio": 0.25,
            "minimum_donor_closer_concepts": 10,
            "minimum_median_donor_recovery": 0.50,
            "reconfiguration_supported": "K12 and probe-margin modalities both meet separation, donor recovery, and donor-closer-than-normal and different thresholds",
            "normal_collapse_supported": "K12 and probe-margin modalities both meet separation, have normal recovery >= 0.50, and normal is closer than donor in at least 10 of 13 concepts",
            "otherwise": "donor identity unresolved by these existing endpoints",
        },
        "implementation_gates": {
            "concepts": 13,
            "examples": 26,
            "all_manifest_hashes_exact": True,
            "all_values_finite": True,
            "two_reductions_byte_identical": True,
        },
        "allowed_conclusions": [
            "exploratory evidence for donor reconfiguration",
            "exploratory evidence for normal collapse",
            "donor identity unresolved because natural endpoints are insufficiently separated or modalities disagree",
        ],
        "forbidden": [
            "overturning or relabeling the frozen Day 49 failure",
            "promoting a controller without a new prospective causal experiment",
            "changing the candidate, endpoints, metrics, thresholds, or weighting after results",
            "new model execution, GPU use, fresh data, behavior, generation, or dissemination",
        ],
        "required_outputs": [
            "results/day-51/donor-identity-summary.json",
            "results/day-51/donor-identity-example-metrics.json",
            "results/day-51/donor-identity-audit.json",
            "results/day-51/execution-artifact-manifest.json",
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
