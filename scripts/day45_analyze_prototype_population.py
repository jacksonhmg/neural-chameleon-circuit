#!/usr/bin/env python3
"""Reduce and gate the frozen Day 45 K12 prototype population evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-45/frozen-prototype-population-contract.json"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
PREFLIGHT_PATH = ROOT / "results/day-45/population-preflight.json"
EXECUTION_PATH = ROOT / "results/day-45/population-execution.json"
SHARD_DIR = ROOT / "results/day-45/population-shards"
SUMMARY_PATH = ROOT / "results/day-45/population-summary.json"
AUDIT_PATH = ROOT / "results/day-45/population-audit.json"
ARTIFACT_MANIFEST_PATH = ROOT / "results/day-45/execution-artifact-manifest.json"

EXACT = "exact_natural_activity"
CANDIDATE = "concept_position_prototype"
RANDOM = "concept_position_prototype.random"
DIRECTIONS = ("induction", "rescue")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from reducer commit {commit}")


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return row["example_id"], row["direction"], row["candidate"]


def effect(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["mean_raw_margins"], dtype=np.float64) - np.asarray(
        row["target_mean_raw_margins"], dtype=np.float64
    )


def bootstrap_concept_macro(
    concept_values: Mapping[str, float], *, seed: int = 45001, draws: int = 10000
) -> dict[str, float | int]:
    concepts = sorted(concept_values)
    values = np.asarray([concept_values[concept] for concept in concepts])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def grouped_metrics(
    records: Sequence[Mapping[str, Any]],
    rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    probe_index: Mapping[str, int],
) -> dict[str, Any]:
    recovery: dict[str, list[float]] = defaultdict(list)
    random_recovery: dict[str, list[float]] = defaultdict(list)
    activation_ratio: dict[str, list[float]] = defaultdict(list)
    own_candidate: dict[str, list[float]] = defaultdict(list)
    own_exact: dict[str, list[float]] = defaultdict(list)
    per_example_direction = []
    for record in records:
        example_id = record["example_id"]
        concept = record["concept"]
        own = probe_index[concept]
        for direction in DIRECTIONS:
            exact_row = rows[(example_id, direction, EXACT)]
            candidate_row = rows[(example_id, direction, CANDIDATE)]
            random_row = rows[(example_id, direction, RANDOM)]
            exact_effect = effect(exact_row)
            candidate_effect = effect(candidate_row)
            random_effect = effect(random_row)
            denominator = max(float(exact_effect @ exact_effect), 1e-8)
            recovered = float(candidate_effect @ exact_effect / denominator)
            random_recovered = float(random_effect @ exact_effect / denominator)
            rms_ratio = float(
                candidate_row["activation_rms"]
                / max(candidate_row["target_activation_rms"], 1e-8)
            )
            canonical_sign = 1.0 if direction == "induction" else -1.0
            candidate_own = canonical_sign * float(candidate_effect[own])
            exact_own = canonical_sign * float(exact_effect[own])
            recovery[concept].append(recovered)
            random_recovery[concept].append(random_recovered)
            activation_ratio[concept].append(rms_ratio)
            own_candidate[concept].append(candidate_own)
            own_exact[concept].append(exact_own)
            per_example_direction.append(
                {
                    "example_id": example_id,
                    "concept": concept,
                    "split": record["split"],
                    "direction": direction,
                    "recovery": recovered,
                    "random_recovery": random_recovered,
                    "recovery_advantage": recovered - random_recovered,
                    "activation_rms_ratio": rms_ratio,
                    "own_probe_candidate_effect": candidate_own,
                    "own_probe_exact_effect": exact_own,
                }
            )
    concepts = sorted(recovery)
    concept_recovery = {
        concept: float(np.mean(recovery[concept])) for concept in concepts
    }
    concept_random = {
        concept: float(np.mean(random_recovery[concept])) for concept in concepts
    }
    concept_advantage = {
        concept: concept_recovery[concept] - concept_random[concept]
        for concept in concepts
    }
    concept_own_correct = {
        concept: float(np.mean(own_candidate[concept]))
        * float(np.mean(own_exact[concept]))
        > 0
        for concept in concepts
    }
    macro_recovery = float(np.mean(list(concept_recovery.values())))
    macro_random = float(np.mean(list(concept_random.values())))
    return {
        "examples": len(records),
        "concepts": len(concepts),
        "equal_concept_recovery": macro_recovery,
        "equal_concept_random_recovery": macro_random,
        "equal_concept_recovery_advantage": macro_recovery - macro_random,
        "recovery_bootstrap": bootstrap_concept_macro(concept_recovery),
        "own_probe_correct_direction_concepts": sum(concept_own_correct.values()),
        "activation_rms_ratio_min": float(
            min(value for values in activation_ratio.values() for value in values)
        ),
        "activation_rms_ratio_max": float(
            max(value for values in activation_ratio.values() for value in values)
        ),
        "per_concept": {
            concept: {
                "examples": sum(row["concept"] == concept for row in records),
                "recovery": concept_recovery[concept],
                "random_recovery": concept_random[concept],
                "recovery_advantage": concept_advantage[concept],
                "own_probe_direction_correct": concept_own_correct[concept],
                "mean_own_probe_candidate_effect": float(
                    np.mean(own_candidate[concept])
                ),
                "mean_own_probe_exact_effect": float(np.mean(own_exact[concept])),
                "activation_rms_ratio_min": float(min(activation_ratio[concept])),
                "activation_rms_ratio_max": float(max(activation_ratio[concept])),
            }
            for concept in concepts
        },
        "per_example_direction": per_example_direction,
    }


def main() -> None:
    reducer_commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH, PROTOTYPE_MANIFEST_PATH):
        require_committed(path, reducer_commit)
    contract = read_json(CONTRACT_PATH)
    prototype_manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    execution = read_json(EXECUTION_PATH)
    contract_sha256 = sha256_file(CONTRACT_PATH)
    prototype_manifest_sha256 = sha256_file(PROTOTYPE_MANIFEST_PATH)
    if preflight.get("result") != "pass":
        raise RuntimeError("population preflight did not pass")
    if execution.get("result") != "complete" or not execution.get("full_population"):
        raise RuntimeError("population execution is incomplete")
    if execution["contract_sha256"] != contract_sha256:
        raise RuntimeError("execution contract hash differs")
    if execution["prototype_manifest_sha256"] != prototype_manifest_sha256:
        raise RuntimeError("execution prototype manifest hash differs")

    shard_paths = sorted(SHARD_DIR.glob("*.json"))
    expected_shards = int(execution["batch_count"])
    if len(shard_paths) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} population shards, found {len(shard_paths)}"
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
            or shard["prototype_manifest_sha256"] != prototype_manifest_sha256
            or shard["prototype_tensor_sha256"] != prototype_manifest["tensor_sha256"]
        ):
            raise RuntimeError(f"shard provenance differs: {path}")
        shard_ordinals.append(int(shard["batch_ordinal"]))
        shard_audits.append(shard["audit"])
        for row in shard["rows"]:
            key = row_key(row)
            if key in rows:
                raise RuntimeError(f"duplicate population row {key}")
            rows[key] = row

    records = prototype_manifest["examples"]
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
            f"population row matrix differs: missing={len(missing)} extra={len(extra)}"
        )
    probe_names = next(iter(rows.values()))["probe_names"]
    probe_index = {name: index for index, name in enumerate(probe_names)}
    if set(probe_index) != {record["concept"] for record in records}:
        raise RuntimeError("probe names differ from population concepts")

    metrics = grouped_metrics(records, rows, probe_index)
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
    split_metrics = {
        split: grouped_metrics(
            [record for record in records if record["split"] == split],
            rows,
            probe_index,
        )
        for split in contract["population"]["splits"]
        if any(record["split"] == split for record in records)
    }
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-prototype-population-v1",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "contract_sha256": contract_sha256,
        "prototype_manifest_sha256": prototype_manifest_sha256,
        "evidence_class": contract["evidence_class"],
        "population": {
            "examples": len(records),
            "concepts": len({record["concept"] for record in records}),
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
        "per_split": split_metrics,
        "disposition": (
            contract["disposition"]["pass"]
            if all(gate_checks.values())
            else contract["disposition"]["fail"]
        ),
        "interpretation_boundary": contract["scope_boundary"],
    }

    target_duplicate_error = 0.0
    token_counts = {record["example_id"]: record["response_token_count"] for record in records}
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
        "preflight_pass": preflight["result"] == "pass",
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
        "all_paths_direct": all(row["path"] == "direct" for row in rows.values()),
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
        "procedure": "rapid-k12-prototype-population-v1-audit",
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
        "procedure": "rapid-k12-prototype-population-v1-artifact-manifest",
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
                "scientific_gate": "pass" if all(gate_checks.values()) else "fail",
                "recovery": metrics["equal_concept_recovery"],
                "recovery_ci_low": metrics["recovery_bootstrap"]["ci_low"],
                "random": metrics["equal_concept_random_recovery"],
                "advantage": metrics["equal_concept_recovery_advantage"],
                "own_direction_concepts": metrics[
                    "own_probe_correct_direction_concepts"
                ],
                "exact_expected_direction_concepts": exact_expected_count,
                "activation_rms_ratio": [
                    metrics["activation_rms_ratio_min"],
                    metrics["activation_rms_ratio_max"],
                ],
                "gate_checks": gate_checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
