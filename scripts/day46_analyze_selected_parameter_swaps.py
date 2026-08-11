#!/usr/bin/env python3
"""Reduce and classify the frozen selected precursor parameter swaps."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_analyze_prototype_population import (  # noqa: E402
    effect,
    git_head,
    read_json,
    require_committed,
    sha256_file,
    write_json_atomic,
)


CONTRACT_PATH = ROOT / "results/day-46/frozen-selected-parameter-swap-contract.json"
SLICE_MANIFEST_PATH = ROOT / "results/day-46/precursor-parameter-slices.json"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
PREFLIGHT_PATH = ROOT / "results/day-46/parameter-swap-preflight.json"
EXECUTION_PATH = ROOT / "results/day-46/parameter-swap-execution.json"
SHARD_DIR = ROOT / "results/day-46/parameter-swap-shards"
DAY46_SHARD_DIR = ROOT / "results/day-46/total-path-shards"
SUMMARY_PATH = ROOT / "results/day-46/parameter-swap-summary.json"
AUDIT_PATH = ROOT / "results/day-46/parameter-swap-audit.json"
ARTIFACT_MANIFEST_PATH = ROOT / "results/day-46/parameter-swap-artifact-manifest.json"

BASELINE_STATE = "chameleon"
NATURAL = "natural_hybrid_activity"
PROTOTYPE = "frozen_chameleon_prototype"
DIRECTIONS = ("induction", "rescue")
FAMILIES = ("o", "q", "kv", "qkv", "qkvo")


def endpoint_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["example_id"],
        row["parameter_state"],
        row["direction"],
        row["operator"],
    )


def trajectory_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return row["example_id"], row["parameter_state"]


def equal_concept(values: Mapping[str, Sequence[float]]) -> float:
    return float(np.mean([np.mean(values[concept]) for concept in sorted(values)]))


def bootstrap_concept_values(
    values: Mapping[str, float], *, seed: int, draws: int
) -> dict[str, float | int]:
    concepts = sorted(values)
    array = np.asarray([values[concept] for concept in concepts])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    estimates = array[indices].mean(axis=1)
    return {
        "estimate": float(array.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def endpoint_state_summary(
    state_id: str,
    operator: str,
    records: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    *,
    reference_operator: str,
) -> dict[str, Any]:
    recovery: dict[str, list[float]] = defaultdict(list)
    activation_ratio: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for direction in DIRECTIONS:
            row = rows[(record["example_id"], state_id, direction, operator)]
            reference = rows[
                (
                    record["example_id"],
                    BASELINE_STATE,
                    direction,
                    reference_operator,
                )
            ]
            candidate_effect = effect(row)
            reference_effect = effect(reference)
            denominator = max(float(reference_effect @ reference_effect), 1e-8)
            recovery[record["concept"]].append(
                float(candidate_effect @ reference_effect / denominator)
            )
            activation_ratio[record["concept"]].append(
                float(
                    row["activation_rms"]
                    / max(row["target_activation_rms"], 1e-8)
                )
            )
    per_concept = {
        concept: {
            "recovery": float(np.mean(recovery[concept])),
            "activation_rms_ratio_min": float(min(activation_ratio[concept])),
            "activation_rms_ratio_max": float(max(activation_ratio[concept])),
        }
        for concept in sorted(recovery)
    }
    return {
        "equal_concept_recovery": equal_concept(recovery),
        "activation_rms_ratio_min": float(
            min(value for group in activation_ratio.values() for value in group)
        ),
        "activation_rms_ratio_max": float(
            max(value for group in activation_ratio.values() for value in group)
        ),
        "per_concept": per_concept,
    }


def trajectory_state_summary(
    state_id: str,
    records: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_layer: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        row = rows[(record["example_id"], state_id)]
        concept = record["concept"]
        for name, value in row["versus_chameleon"].items():
            metrics[concept][name].append(float(value))
        for layer, values in row["per_layer_versus_chameleon"].items():
            per_layer[layer][concept].append(float(values["recovery"]))
    per_concept = {
        concept: {
            name: float(np.mean(values))
            for name, values in sorted(concept_metrics.items())
        }
        for concept, concept_metrics in sorted(metrics.items())
    }
    return {
        "equal_concept": {
            name: float(
                np.mean([per_concept[concept][name] for concept in per_concept])
            )
            for name in sorted(next(iter(per_concept.values())))
        },
        "per_layer_equal_concept_recovery": {
            layer: equal_concept(concept_values)
            for layer, concept_values in sorted(per_layer.items())
        },
        "per_concept": per_concept,
    }


def load_day46_rows() -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows = {}
    for path in sorted(DAY46_SHARD_DIR.glob("*.json")):
        for row in read_json(path)["rows"]:
            key = (row["example_id"], row["direction"], row["candidate"])
            if key in rows:
                raise RuntimeError(f"duplicate Day 46 reference row {key}")
            rows[key] = row
    return rows


def main() -> None:
    reducer_commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        SLICE_MANIFEST_PATH,
        PROTOTYPE_MANIFEST_PATH,
    ):
        require_committed(path, reducer_commit)
    contract = read_json(CONTRACT_PATH)
    slice_manifest = read_json(SLICE_MANIFEST_PATH)
    prototype_manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    execution = read_json(EXECUTION_PATH)
    contract_sha256 = sha256_file(CONTRACT_PATH)
    slice_manifest_sha256 = sha256_file(SLICE_MANIFEST_PATH)
    if preflight.get("result") != "pass":
        raise RuntimeError("parameter-swap preflight did not pass")
    if execution.get("result") != "complete" or not execution.get("full_population"):
        raise RuntimeError("parameter-swap execution is incomplete")
    if execution["contract_sha256"] != contract_sha256:
        raise RuntimeError("execution contract hash differs")
    if execution["slice_manifest_sha256"] != slice_manifest_sha256:
        raise RuntimeError("execution slice manifest hash differs")

    shard_paths = sorted(SHARD_DIR.glob("*.json"))
    expected_shards = int(execution["batch_count"])
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} swap shards, found {len(shard_paths)}"
        )
    endpoint_rows: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    trajectory_rows: dict[tuple[str, str], Mapping[str, Any]] = {}
    shard_audits = []
    shard_ordinals = []
    execution_commit = execution["execution_commit"]
    for path in shard_paths:
        shard = read_json(path)
        if (
            shard["execution_commit"] != execution_commit
            or shard["contract_sha256"] != contract_sha256
            or shard["slice_manifest_sha256"] != slice_manifest_sha256
            or shard["slice_tensor_sha256"] != slice_manifest["tensor_sha256"]
        ):
            raise RuntimeError(f"shard provenance differs: {path}")
        shard_ordinals.append(int(shard["batch_ordinal"]))
        shard_audits.append(shard["audit"])
        for row in shard["endpoint_rows"]:
            key = endpoint_key(row)
            if key in endpoint_rows:
                raise RuntimeError(f"duplicate endpoint row {key}")
            endpoint_rows[key] = row
        for row in shard["trajectory_rows"]:
            key = trajectory_key(row)
            if key in trajectory_rows:
                raise RuntimeError(f"duplicate trajectory row {key}")
            trajectory_rows[key] = row

    records = prototype_manifest["examples"]
    state_ids = [state["id"] for state in contract["parameter_states"]]
    expected_endpoint_keys = {
        (record["example_id"], state, direction, operator)
        for record in records
        for state in state_ids
        for direction in DIRECTIONS
        for operator in contract["causal_operators"]
    }
    expected_trajectory_keys = {
        (record["example_id"], state)
        for record in records
        for state in state_ids
    }
    if set(endpoint_rows) != expected_endpoint_keys:
        raise RuntimeError("parameter-swap endpoint matrix differs")
    if set(trajectory_rows) != expected_trajectory_keys:
        raise RuntimeError("parameter-swap trajectory matrix differs")

    endpoint_summaries = {
        state: {
            NATURAL: endpoint_state_summary(
                state,
                NATURAL,
                records,
                endpoint_rows,
                reference_operator=NATURAL,
            ),
            PROTOTYPE: endpoint_state_summary(
                state,
                PROTOTYPE,
                records,
                endpoint_rows,
                reference_operator=PROTOTYPE,
            ),
        }
        for state in state_ids
    }
    trajectory_summaries = {
        state: trajectory_state_summary(state, records, trajectory_rows)
        for state in state_ids
    }
    seed = int(contract["execution"]["bootstrap_seed"])
    draws = int(contract["execution"]["bootstrap_draws"])
    material_rule = contract["material_parameter_effect"]
    family_results = {}
    for family_index, family in enumerate(FAMILIES):
        selected_state = f"selected_{family}"
        control_state = f"control_{family}"
        selected_natural = endpoint_summaries[selected_state][NATURAL]
        control_natural = endpoint_summaries[control_state][NATURAL]
        selected_recovery = selected_natural["equal_concept_recovery"]
        control_recovery = control_natural["equal_concept_recovery"]
        per_concept_advantage = {
            concept: control_natural["per_concept"][concept]["recovery"]
            - selected_natural["per_concept"][concept]["recovery"]
            for concept in selected_natural["per_concept"]
        }
        advantage_bootstrap = bootstrap_concept_values(
            per_concept_advantage,
            seed=seed + family_index,
            draws=draws,
        )
        checks = {
            "selected_recovery_at_most_threshold": selected_recovery
            <= float(material_rule["selected_natural_effect_recovery_max"]),
            "control_recovery_at_least_threshold": control_recovery
            >= float(material_rule["matched_control_natural_effect_recovery_min"]),
            "selectivity_advantage_at_least_threshold": control_recovery
            - selected_recovery
            >= float(material_rule["selectivity_advantage_min"]),
            "selectivity_ci_low_at_least_threshold": advantage_bootstrap["ci_low"]
            >= float(
                material_rule["selectivity_advantage_bootstrap_95_ci_low_min"]
            ),
        }
        family_results[family] = {
            "selected_state": selected_state,
            "control_state": control_state,
            "selected_natural_effect_recovery": selected_recovery,
            "control_natural_effect_recovery": control_recovery,
            "selectivity_advantage": control_recovery - selected_recovery,
            "selectivity_advantage_bootstrap": advantage_bootstrap,
            "selected_fixed_prototype_effect_recovery": endpoint_summaries[
                selected_state
            ][PROTOTYPE]["equal_concept_recovery"],
            "control_fixed_prototype_effect_recovery": endpoint_summaries[
                control_state
            ][PROTOTYPE]["equal_concept_recovery"],
            "selected_trajectory_recovery": trajectory_summaries[selected_state][
                "equal_concept"
            ]["recovery"],
            "control_trajectory_recovery": trajectory_summaries[control_state][
                "equal_concept"
            ]["recovery"],
            "per_concept_selectivity_advantage": per_concept_advantage,
            "material_checks": checks,
            "material": all(checks.values()),
        }

    signatures = {
        "acquired_write_geometry": family_results["o"]["material"]
        and family_results["o"]["selected_fixed_prototype_effect_recovery"] <= 0.8,
        "acquired_query_routing": family_results["q"]["material"]
        and family_results["q"]["selected_trajectory_recovery"] <= 0.8
        and family_results["q"]["selected_fixed_prototype_effect_recovery"] >= 0.8,
        "acquired_key_value_routing": family_results["kv"]["material"]
        and family_results["kv"]["selected_trajectory_recovery"] <= 0.8
        and family_results["kv"]["selected_fixed_prototype_effect_recovery"] >= 0.8,
        "qkv_coadaptation": family_results["qkv"]["material"]
        and not family_results["q"]["material"]
        and not family_results["kv"]["material"]
        and family_results["qkv"]["selected_natural_effect_recovery"]
        <= family_results["q"]["selected_natural_effect_recovery"] - 0.15
        and family_results["qkv"]["selected_natural_effect_recovery"]
        <= family_results["kv"]["selected_natural_effect_recovery"] - 0.15,
        "complete_attention_coadaptation": family_results["qkvo"]["material"]
        and family_results["qkvo"]["selected_natural_effect_recovery"]
        <= family_results["o"]["selected_natural_effect_recovery"] - 0.15
        and family_results["qkvo"]["selected_natural_effect_recovery"]
        <= family_results["qkv"]["selected_natural_effect_recovery"] - 0.15,
        "local_selected_parameters_insufficient": not family_results["qkvo"][
            "material"
        ]
        or family_results["qkvo"]["selected_natural_effect_recovery"] >= 0.8,
    }
    positive_local_signatures = [
        name
        for name, value in signatures.items()
        if value and name != "local_selected_parameters_insufficient"
    ]
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-precursor-parameter-swaps-v1",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "contract_sha256": contract_sha256,
        "slice_manifest_sha256": slice_manifest_sha256,
        "evidence_class": contract["evidence_class"],
        "population": {
            "examples": len(records),
            "concepts": len({row["concept"] for row in records}),
            "parameter_states": len(state_ids),
            "endpoint_rows": len(endpoint_rows),
            "trajectory_rows": len(trajectory_rows),
        },
        "family_results": family_results,
        "signatures": signatures,
        "positive_local_signatures": positive_local_signatures,
        "state_endpoint_summaries": endpoint_summaries,
        "state_trajectory_summaries": trajectory_summaries,
        "weight_difference_norms": {
            row["tensor_key"]: row["relative_l2_difference_from_chameleon"]
            for row in slice_manifest["slices"]
        },
        "disposition": (
            contract["disposition"]["one_or_more_local_signatures"]
            if positive_local_signatures
            else contract["disposition"]["no_local_signature"]
        ),
        "interpretation_boundary": contract["scope_boundary"],
    }

    day46_rows = load_day46_rows()
    baseline_day46_error = 0.0
    candidate_map = {
        NATURAL: "exact_natural_activity",
        PROTOTYPE: "concept_position_prototype",
    }
    for record in records:
        for direction in DIRECTIONS:
            for operator, day46_candidate in candidate_map.items():
                row = endpoint_rows[
                    (record["example_id"], BASELINE_STATE, direction, operator)
                ]
                reference = day46_rows[
                    (record["example_id"], direction, day46_candidate)
                ]
                baseline_day46_error = max(
                    baseline_day46_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(row["mean_raw_margins"])
                                - np.asarray(reference["mean_raw_margins"])
                            )
                        )
                    ),
                )
    target_duplicate_error = 0.0
    for record in records:
        for state in state_ids:
            for direction in DIRECTIONS:
                natural_target = np.asarray(
                    endpoint_rows[
                        (record["example_id"], state, direction, NATURAL)
                    ]["target_mean_raw_margins"]
                )
                prototype_target = np.asarray(
                    endpoint_rows[
                        (record["example_id"], state, direction, PROTOTYPE)
                    ]["target_mean_raw_margins"]
                )
                target_duplicate_error = max(
                    target_duplicate_error,
                    float(np.max(np.abs(natural_target - prototype_target))),
                )
    baseline_trajectory_error = max(
        max(
            abs(float(row["versus_chameleon"]["recovery"]) - 1.0),
            abs(float(row["versus_chameleon"]["cosine"]) - 1.0),
            abs(float(row["versus_chameleon"]["relative_l2"])),
            abs(float(row["versus_chameleon"]["norm_ratio"]) - 1.0),
        )
        for key, row in trajectory_rows.items()
        if key[1] == BASELINE_STATE
    )
    gates = contract["implementation_gates"]
    checks = {
        "preflight_pass": preflight["result"] == "pass",
        "execution_complete": execution["result"] == "complete",
        "full_population": execution["full_population"] is True,
        "exact_shard_count": len(shard_paths) == expected_shards,
        "exact_shard_ordinals": sorted(shard_ordinals) == list(range(expected_shards)),
        "exact_endpoint_row_count": len(endpoint_rows)
        == int(gates["exact_endpoint_row_count"]),
        "exact_trajectory_row_count": len(trajectory_rows)
        == int(gates["exact_trajectory_row_count"]),
        "exact_endpoint_matrix": set(endpoint_rows) == expected_endpoint_keys,
        "exact_trajectory_matrix": set(trajectory_rows) == expected_trajectory_keys,
        "single_execution_commit": all(
            row["execution_commit"] == execution_commit
            for row in (*endpoint_rows.values(), *trajectory_rows.values())
        ),
        "all_values_finite": all(
            np.isfinite(row["mean_raw_margins"]).all()
            and np.isfinite(row["target_mean_raw_margins"]).all()
            and np.isfinite(row["activation_rms"])
            and np.isfinite(row["target_activation_rms"])
            for row in endpoint_rows.values()
        ),
        "all_trajectory_values_finite": all(
            all(np.isfinite(list(row["versus_chameleon"].values())))
            and all(np.isfinite(list(row["prototype_versus_hybrid"].values())))
            for row in trajectory_rows.values()
        ),
        "layer9_inputs_bit_exact": all(
            audit["layer9_input_max_abs_across_states"]
            <= float(gates["layer9_input_max_abs_across_states"])
            for audit in shard_audits
        ),
        "parameters_restore_bit_exact": all(
            audit["parameter_restore_max_abs"]
            <= float(gates["parameter_restore_max_abs"])
            for audit in shard_audits
        ),
        "all_shards_finite": all(
            audit["all_endpoint_rows_finite"]
            and audit["all_trajectory_rows_finite"]
            for audit in shard_audits
        ),
        "no_hook_leaks": execution["hooks_after_execution"] == 0
        and all(audit["hooks_after_batch"] == 0 for audit in shard_audits),
        "targets_exact_duplicates": target_duplicate_error == 0.0,
        "baseline_matches_day46": baseline_day46_error
        <= float(gates["baseline_vs_day46_margin_max_abs"]),
        "baseline_trajectory_identity": baseline_trajectory_error <= 1e-12,
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-precursor-parameter-swaps-v1-audit",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "checks": checks,
        "baseline_vs_day46_margin_max_abs": baseline_day46_error,
        "target_duplicate_max_abs_error": target_duplicate_error,
        "baseline_trajectory_identity_max_abs_error": baseline_trajectory_error,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(SUMMARY_PATH, summary)
    write_json_atomic(AUDIT_PATH, audit)
    artifact_manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-parameter-swaps-v1-artifact-manifest",
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
                "family_results": {
                    family: {
                        "selected_recovery": value[
                            "selected_natural_effect_recovery"
                        ],
                        "control_recovery": value[
                            "control_natural_effect_recovery"
                        ],
                        "selectivity_advantage": value["selectivity_advantage"],
                        "selectivity_ci_low": value[
                            "selectivity_advantage_bootstrap"
                        ]["ci_low"],
                        "fixed_prototype_recovery": value[
                            "selected_fixed_prototype_effect_recovery"
                        ],
                        "trajectory_recovery": value[
                            "selected_trajectory_recovery"
                        ],
                        "material": value["material"],
                    }
                    for family, value in family_results.items()
                },
                "signatures": signatures,
                "positive_local_signatures": positive_local_signatures,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
