#!/usr/bin/env python3
"""Reduce and gate the frozen Day 47 held-out K12 validation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_analyze_prototype_population import (  # noqa: E402
    CANDIDATE,
    DIRECTIONS,
    EXACT,
    RANDOM,
    bootstrap_concept_macro,
    effect,
    git_head,
    grouped_metrics,
    read_json,
    require_committed,
    row_key,
    sha256_file,
    write_json_atomic,
)


CONTRACT_PATH = ROOT / "results/day-47/frozen-heldout-development-contract.json"
PANEL_PATH = ROOT / "data/splits/day47-v1/heldout-development.jsonl"
PANEL_MANIFEST_PATH = ROOT / "data/splits/day47-v1/manifest.json"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-47/heldout-prototype-tensors.json"
DAY46_SUMMARY_PATH = ROOT / "results/day-46/total-path-summary.json"
PREFLIGHT_PATH = ROOT / "results/day-47/heldout-preflight.json"
EXECUTION_PATH = ROOT / "results/day-47/heldout-execution.json"
SHARD_DIR = ROOT / "results/day-47/heldout-shards"
SUMMARY_PATH = ROOT / "results/day-47/heldout-summary.json"
AUDIT_PATH = ROOT / "results/day-47/heldout-audit.json"
ARTIFACT_MANIFEST_PATH = ROOT / "results/day-47/execution-artifact-manifest.json"


def main() -> None:
    reducer_commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        PANEL_PATH,
        PANEL_MANIFEST_PATH,
        PROTOTYPE_MANIFEST_PATH,
        DAY46_SUMMARY_PATH,
    ):
        require_committed(path, reducer_commit)
    contract = read_json(CONTRACT_PATH)
    panel_manifest = read_json(PANEL_MANIFEST_PATH)
    prototype_manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    previous = read_json(DAY46_SUMMARY_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    execution = read_json(EXECUTION_PATH)
    contract_sha256 = sha256_file(CONTRACT_PATH)
    panel_sha256 = sha256_file(PANEL_PATH)
    prototype_manifest_sha256 = sha256_file(PROTOTYPE_MANIFEST_PATH)
    if preflight.get("result") != "pass":
        raise RuntimeError("held-out preflight did not pass")
    if execution.get("result") != "complete" or not execution.get("full_population"):
        raise RuntimeError("held-out execution is incomplete")
    if execution["contract_sha256"] != contract_sha256:
        raise RuntimeError("execution contract hash differs")
    if execution["panel_sha256"] != panel_sha256:
        raise RuntimeError("execution panel hash differs")
    if execution["prototype_manifest_sha256"] != prototype_manifest_sha256:
        raise RuntimeError("execution prototype manifest hash differs")

    records = [json.loads(line) for line in PANEL_PATH.read_text().splitlines()]
    shard_paths = sorted(SHARD_DIR.glob("*.json"))
    expected_shards = int(execution["batch_count"])
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} held-out shards, found {len(shard_paths)}"
        )
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    shard_audits = []
    shard_ordinals = []
    execution_commit = execution["execution_commit"]
    for path in shard_paths:
        shard = read_json(path)
        if (
            shard["execution_commit"] != execution_commit
            or shard["contract_sha256"] != contract_sha256
            or shard["panel_sha256"] != panel_sha256
            or shard["prototype_manifest_sha256"] != prototype_manifest_sha256
            or shard["prototype_tensor_sha256"]
            != prototype_manifest["tensor_sha256"]
        ):
            raise RuntimeError(f"held-out shard provenance differs: {path}")
        shard_ordinals.append(int(shard["batch_ordinal"]))
        shard_audits.append(shard["audit"])
        for row in shard["rows"]:
            key = row_key(row)
            if key in rows:
                raise RuntimeError(f"duplicate held-out row {key}")
            rows[key] = row

    expected_keys = {
        (record["example_id"], direction, candidate)
        for record in records
        for direction in DIRECTIONS
        for candidate in contract["jobs"]
    }
    if set(rows) != expected_keys:
        missing = expected_keys - set(rows)
        extra = set(rows) - expected_keys
        raise RuntimeError(
            f"held-out row matrix differs: missing={len(missing)} extra={len(extra)}"
        )
    probe_names = next(iter(rows.values()))["probe_names"]
    probe_index = {name: index for index, name in enumerate(probe_names)}
    concepts = {record["concept"] for record in records}
    if len(probe_index) != 13 or not concepts.issubset(probe_index):
        raise RuntimeError("released probe set does not cover held-out concepts")

    metrics = grouped_metrics(records, rows, probe_index)
    concept_recovery = {
        concept: values["recovery"]
        for concept, values in metrics["per_concept"].items()
    }
    metrics["recovery_bootstrap"] = bootstrap_concept_macro(
        concept_recovery,
        seed=int(contract["execution"]["bootstrap_seed"]),
        draws=int(contract["execution"]["bootstrap_draws"]),
    )
    exact_own_by_concept: dict[str, list[float]] = defaultdict(list)
    for record in records:
        own = probe_index[record["concept"]]
        for direction in DIRECTIONS:
            row = rows[(record["example_id"], direction, EXACT)]
            canonical_sign = 1.0 if direction == "induction" else -1.0
            exact_own_by_concept[record["concept"]].append(
                canonical_sign * float(effect(row)[own])
            )
    exact_expected = {
        concept: float(np.mean(values)) < 0
        for concept, values in sorted(exact_own_by_concept.items())
    }
    exact_expected_count = sum(exact_expected.values())

    gate = contract["scientific_gate"]
    gate_checks = {
        "equal_concept_recovery": metrics["equal_concept_recovery"]
        >= float(gate["equal_concept_recovery_min"]),
        "concept_bootstrap_95_ci_low": metrics["recovery_bootstrap"]["ci_low"]
        >= float(gate["concept_bootstrap_95_ci_low_min"]),
        "own_probe_correct_direction_concepts": metrics[
            "own_probe_correct_direction_concepts"
        ]
        >= int(gate["own_probe_correct_direction_concepts_min"]),
        "recovery_advantage_over_random": metrics[
            "equal_concept_recovery_advantage"
        ]
        >= float(gate["recovery_advantage_over_random_min"]),
        "activation_rms_ratio_min": metrics["activation_rms_ratio_min"]
        >= float(gate["activation_rms_ratio_min"]),
        "activation_rms_ratio_max": metrics["activation_rms_ratio_max"]
        <= float(gate["activation_rms_ratio_max"]),
        "exact_natural_expected_direction_concepts": exact_expected_count
        >= int(gate["exact_natural_expected_direction_concepts_min"]),
    }

    previous_per_concept = previous["candidate"]["per_concept"]
    comparison = {
        concept: {
            "existing_population_recovery": previous_per_concept[concept]["recovery"],
            "heldout_recovery": metrics["per_concept"][concept]["recovery"],
            "heldout_minus_existing": metrics["per_concept"][concept]["recovery"]
            - previous_per_concept[concept]["recovery"],
        }
        for concept in sorted(concepts)
    }
    previous_matching_macro = float(
        np.mean(
            [previous_per_concept[concept]["recovery"] for concept in sorted(concepts)]
        )
    )
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-heldout-development-total-path-v1",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "contract_sha256": contract_sha256,
        "panel_sha256": panel_sha256,
        "prototype_manifest_sha256": prototype_manifest_sha256,
        "evidence_class": contract["evidence_class"],
        "population": {
            "examples": len(records),
            "concepts": len(concepts),
            "directions": len(DIRECTIONS),
            "candidates_per_direction": len(contract["jobs"]),
            "rows": len(rows),
        },
        "exact_natural_positive_control": {
            "expected_direction_concepts": exact_expected_count,
            "required": int(
                gate["exact_natural_expected_direction_concepts_min"]
            ),
            "per_concept": exact_expected,
        },
        "candidate": {
            **metrics,
            "gate_checks": gate_checks,
            "passes": all(gate_checks.values()),
        },
        "existing_population_comparison": {
            "matching_11_concept_existing_recovery": previous_matching_macro,
            "heldout_recovery": metrics["equal_concept_recovery"],
            "heldout_minus_existing": metrics["equal_concept_recovery"]
            - previous_matching_macro,
            "per_concept": comparison,
        },
        "disposition": (
            contract["disposition"]["pass"]
            if all(gate_checks.values())
            else contract["disposition"]["fail"]
        ),
        "interpretation_boundary": contract["scope_boundary"],
    }

    target_duplicate_error = 0.0
    token_counts = {
        record["example_id"]: record["response_token_count"] for record in records
    }
    for record in records:
        example_id = record["example_id"]
        for direction in DIRECTIONS:
            exact_target = np.asarray(
                rows[(example_id, direction, EXACT)]["target_mean_raw_margins"]
            )
            for candidate in (CANDIDATE, RANDOM):
                target_duplicate_error = max(
                    target_duplicate_error,
                    float(
                        np.max(
                            np.abs(
                                exact_target
                                - np.asarray(
                                    rows[(example_id, direction, candidate)][
                                        "target_mean_raw_margins"
                                    ]
                                )
                            )
                        )
                    ),
                )
    implementation_checks = {
        "panel_freeze_pass": all(panel_manifest["checks"].values()),
        "prototype_manifest_pass": prototype_manifest["result"] == "pass"
        and all(prototype_manifest["checks"].values()),
        "preflight_pass": preflight["result"] == "pass",
        "preflight_generated_no_candidate_outcomes": preflight[
            "candidate_outcomes_generated"
        ]
        is False,
        "execution_complete": execution["result"] == "complete",
        "full_population": execution["full_population"] is True,
        "exact_shard_count": len(shard_paths) == expected_shards,
        "exact_shard_ordinals": sorted(shard_ordinals) == list(range(expected_shards)),
        "exact_row_count": len(rows)
        == int(contract["implementation_gates"]["exact_row_count"]),
        "exact_row_matrix": set(rows) == expected_keys,
        "single_execution_commit": all(
            row["execution_commit"] == execution_commit for row in rows.values()
        ),
        "single_probe_order": all(
            row["probe_names"] == probe_names for row in rows.values()
        ),
        "all_paths_total": all(row["path"] == "total" for row in rows.values()),
        "all_response_token_counts_exact": all(
            int(row["response_token_count"]) == int(token_counts[row["example_id"]])
            for row in rows.values()
        ),
        "all_values_finite": all(
            np.isfinite(row["mean_raw_margins"]).all()
            and np.isfinite(row["target_mean_raw_margins"]).all()
            and np.isfinite(row["activation_rms"])
            and np.isfinite(row["target_activation_rms"])
            for row in rows.values()
        ),
        "all_haar_audits_pass": all(
            audit["random_audit"]["pass"] for audit in shard_audits
        ),
        "all_shards_finite": all(audit["all_rows_finite"] for audit in shard_audits),
        "no_hook_leaks": execution["hooks_after_execution"] == 0
        and all(audit["hooks_after_batch"] == 0 for audit in shard_audits),
        "targets_exact_duplicates": target_duplicate_error == 0.0,
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-heldout-development-total-path-v1-audit",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "checks": implementation_checks,
        "target_duplicate_max_abs_error": target_duplicate_error,
        "result": "pass" if all(implementation_checks.values()) else "fail",
    }
    write_json_atomic(SUMMARY_PATH, summary)
    write_json_atomic(AUDIT_PATH, audit)
    artifact_manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-heldout-development-total-path-v1-artifact-manifest",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "files": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                PREFLIGHT_PATH,
                EXECUTION_PATH,
                *shard_paths,
                SUMMARY_PATH,
                AUDIT_PATH,
            )
        },
    }
    write_json_atomic(ARTIFACT_MANIFEST_PATH, artifact_manifest)
    if audit["result"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "audit": audit["result"],
                "passes": summary["candidate"]["passes"],
                "recovery": metrics["equal_concept_recovery"],
                "recovery_ci": [
                    metrics["recovery_bootstrap"]["ci_low"],
                    metrics["recovery_bootstrap"]["ci_high"],
                ],
                "random_recovery": metrics["equal_concept_random_recovery"],
                "recovery_advantage": metrics[
                    "equal_concept_recovery_advantage"
                ],
                "own_probe_correct_direction_concepts": metrics[
                    "own_probe_correct_direction_concepts"
                ],
                "exact_expected_direction_concepts": exact_expected_count,
                "activation_rms_ratio": [
                    metrics["activation_rms_ratio_min"],
                    metrics["activation_rms_ratio_max"],
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
