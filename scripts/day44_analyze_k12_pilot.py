#!/usr/bin/env python3
"""Reduce and gate the frozen Day 44 direct-path K12 pilot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
SPEC_PATH = ROOT / "results/day-44/frozen-pilot-operator-spec.json"
SELECTION_PATH = ROOT / "results/day-44/pilot-selection.json"
PREFLIGHT_PATH = ROOT / "results/day-44/pilot-preflight.json"
EXECUTION_PATH = ROOT / "results/day-44/pilot-execution.json"
SHARD_DIR = ROOT / "results/day-44/pilot-shards"
SUMMARY_PATH = ROOT / "results/day-44/pilot-summary.json"
AUDIT_PATH = ROOT / "results/day-44/pilot-audit.json"
MANIFEST_PATH = ROOT / "results/day-44/execution-artifact-manifest.json"

PRIMARY_CANDIDATES = (
    "concept_position_prototype",
    "tangential_actual_activity",
)
RANDOM_FOR = {
    "concept_position_prototype": "concept_position_prototype.random",
    "tangential_actual_activity": "tangential_actual_activity.random",
}


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


def equal_concept_macro(
    values: Mapping[str, Sequence[float]], concepts: Sequence[str]
) -> float:
    return float(np.mean([np.mean(values[concept]) for concept in concepts]))


def bootstrap_concept_macro(
    concept_values: Mapping[str, float], *, seed: int = 44001, draws: int = 10000
) -> dict[str, float]:
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


def candidate_summary(
    candidate: str,
    rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    probe_index: Mapping[str, int],
    concepts: Sequence[str],
) -> dict[str, Any]:
    random_name = RANDOM_FOR[candidate]
    recovery_by_concept: dict[str, list[float]] = defaultdict(list)
    random_recovery_by_concept: dict[str, list[float]] = defaultdict(list)
    activation_ratio_by_concept: dict[str, list[float]] = defaultdict(list)
    own_candidate_by_concept: dict[str, list[float]] = defaultdict(list)
    own_exact_by_concept: dict[str, list[float]] = defaultdict(list)
    per_example = []
    for record in records:
        example_id = record["example_id"]
        concept = record["concept"]
        own_index = probe_index[concept]
        for direction in ("induction", "rescue"):
            exact_row = rows[(example_id, direction, "exact_natural_activity")]
            candidate_row = rows[(example_id, direction, candidate)]
            random_row = rows[(example_id, direction, random_name)]
            exact_effect = effect(exact_row)
            candidate_effect = effect(candidate_row)
            random_effect = effect(random_row)
            denominator = max(float(exact_effect @ exact_effect), 1e-8)
            recovery = float(candidate_effect @ exact_effect / denominator)
            random_recovery = float(random_effect @ exact_effect / denominator)
            activation_ratio = float(
                candidate_row["activation_rms"]
                / max(candidate_row["target_activation_rms"], 1e-8)
            )
            canonical_sign = 1.0 if direction == "induction" else -1.0
            own_candidate = canonical_sign * float(candidate_effect[own_index])
            own_exact = canonical_sign * float(exact_effect[own_index])
            recovery_by_concept[concept].append(recovery)
            random_recovery_by_concept[concept].append(random_recovery)
            activation_ratio_by_concept[concept].append(activation_ratio)
            own_candidate_by_concept[concept].append(own_candidate)
            own_exact_by_concept[concept].append(own_exact)
            per_example.append(
                {
                    "example_id": example_id,
                    "concept": concept,
                    "direction": direction,
                    "recovery": recovery,
                    "random_recovery": random_recovery,
                    "recovery_advantage": recovery - random_recovery,
                    "activation_rms_ratio": activation_ratio,
                    "own_probe_candidate_effect": own_candidate,
                    "own_probe_exact_effect": own_exact,
                }
            )

    concept_recovery = {
        concept: float(np.mean(recovery_by_concept[concept])) for concept in concepts
    }
    concept_random = {
        concept: float(np.mean(random_recovery_by_concept[concept]))
        for concept in concepts
    }
    concept_advantage = {
        concept: concept_recovery[concept] - concept_random[concept]
        for concept in concepts
    }
    concept_own_correct = {
        concept: float(np.mean(own_candidate_by_concept[concept]))
        * float(np.mean(own_exact_by_concept[concept]))
        > 0
        for concept in concepts
    }
    concept_activation_ok = {
        concept: all(
            0.5 <= value <= 1.5 for value in activation_ratio_by_concept[concept]
        )
        for concept in concepts
    }
    macro_recovery = float(np.mean(list(concept_recovery.values())))
    macro_random = float(np.mean(list(concept_random.values())))
    macro_advantage = macro_recovery - macro_random
    correct_concepts = sum(concept_own_correct.values())
    checks = {
        "recovery_at_least_0_50": macro_recovery >= 0.50,
        "own_probe_direction_at_least_10_concepts": correct_concepts >= 10,
        "beats_random_by_at_least_0_20": macro_advantage >= 0.20,
        "activation_rms_all_within_0_5_to_1_5": all(concept_activation_ok.values()),
    }
    return {
        "candidate": candidate,
        "equal_concept_recovery": macro_recovery,
        "equal_concept_random_recovery": macro_random,
        "equal_concept_recovery_advantage": macro_advantage,
        "recovery_bootstrap": bootstrap_concept_macro(concept_recovery),
        "own_probe_correct_direction_concepts": correct_concepts,
        "activation_rms_ratio_min": float(
            min(value for values in activation_ratio_by_concept.values() for value in values)
        ),
        "activation_rms_ratio_max": float(
            max(value for values in activation_ratio_by_concept.values() for value in values)
        ),
        "per_concept": {
            concept: {
                "recovery": concept_recovery[concept],
                "random_recovery": concept_random[concept],
                "recovery_advantage": concept_advantage[concept],
                "own_probe_direction_correct": concept_own_correct[concept],
                "activation_rms_ok": concept_activation_ok[concept],
            }
            for concept in concepts
        },
        "per_example_direction": per_example,
        "gate_checks": checks,
        "passes": all(checks.values()),
    }


def main() -> None:
    reducer_commit = git_head()
    require_committed(Path(__file__).resolve(), reducer_commit)
    contract = read_json(CONTRACT_PATH)
    spec = read_json(SPEC_PATH)
    selection = read_json(SELECTION_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    execution = read_json(EXECUTION_PATH)
    if preflight.get("result") != "pass":
        raise RuntimeError("pilot preflight did not pass")
    if execution.get("result") != "complete":
        raise RuntimeError("pilot execution is incomplete")
    if execution["contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("execution contract hash differs")
    if execution["operator_spec_sha256"] != sha256_file(SPEC_PATH):
        raise RuntimeError("execution operator spec hash differs")

    shard_paths = sorted(SHARD_DIR.glob("*.json"))
    if len(shard_paths) != 13:
        raise RuntimeError(f"expected 13 pilot shards, found {len(shard_paths)}")
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    shard_audits = []
    execution_commit = execution["execution_commit"]
    for path in shard_paths:
        shard = read_json(path)
        if (
            shard["execution_commit"] != execution_commit
            or shard["contract_sha256"] != sha256_file(CONTRACT_PATH)
            or shard["operator_spec_sha256"] != sha256_file(SPEC_PATH)
        ):
            raise RuntimeError(f"shard provenance differs: {path}")
        shard_audits.append(shard["audit"])
        for row in shard["rows"]:
            key = row_key(row)
            if key in rows:
                raise RuntimeError(f"duplicate pilot row {key}")
            rows[key] = row

    records = contract["pilot"]["examples"]
    concepts = []
    for record in records:
        if record["concept"] not in concepts:
            concepts.append(record["concept"])
    probe_names = next(iter(rows.values()))["probe_names"]
    probe_index = {name: index for index, name in enumerate(probe_names)}
    expected_candidates = spec["jobs_per_direction"]
    expected_keys = {
        (record["example_id"], direction, candidate)
        for record in records
        for direction in ("induction", "rescue")
        for candidate in expected_candidates
    }
    if set(rows) != expected_keys:
        missing = expected_keys - set(rows)
        extra = set(rows) - expected_keys
        raise RuntimeError(f"pilot row matrix differs: missing={len(missing)} extra={len(extra)}")

    exact_own_by_concept: dict[str, list[float]] = defaultdict(list)
    for record in records:
        own = probe_index[record["concept"]]
        for direction in ("induction", "rescue"):
            row = rows[(record["example_id"], direction, "exact_natural_activity")]
            canonical_sign = 1.0 if direction == "induction" else -1.0
            exact_own_by_concept[record["concept"]].append(
                canonical_sign * float(effect(row)[own])
            )
    exact_expected = {
        concept: float(np.mean(values)) < 0
        for concept, values in exact_own_by_concept.items()
    }
    exact_expected_count = sum(exact_expected.values())

    candidate_results = {
        candidate: candidate_summary(
            candidate, rows, records, probe_index, concepts
        )
        for candidate in PRIMARY_CANDIDATES
    }
    exact_positive_pass = exact_expected_count >= 10
    for result in candidate_results.values():
        result["gate_checks"][
            "exact_natural_positive_control_expected_direction"
        ] = exact_positive_pass
        result["passes"] = all(result["gate_checks"].values())

    dose = {}
    for scale in spec["dose_scales"]:
        name = f"tangential_actual_activity.scale_{scale:.1f}"
        recovery: dict[str, list[float]] = defaultdict(list)
        for record in records:
            for direction in ("induction", "rescue"):
                exact_row = rows[
                    (record["example_id"], direction, "exact_natural_activity")
                ]
                row = rows[(record["example_id"], direction, name)]
                exact_effect = effect(exact_row)
                candidate_effect = effect(row)
                denominator = max(float(exact_effect @ exact_effect), 1e-8)
                recovery[record["concept"]].append(
                    float(candidate_effect @ exact_effect / denominator)
                )
        dose[str(scale)] = {
            "equal_concept_recovery": equal_concept_macro(recovery, concepts),
            "per_concept": {
                concept: float(np.mean(recovery[concept])) for concept in concepts
            },
        }

    passing = [
        candidate for candidate in PRIMARY_CANDIDATES if candidate_results[candidate]["passes"]
    ]
    simplicity_order = ["concept_position_prototype", "tangential_actual_activity"]
    promoted = next((name for name in simplicity_order if name in passing), None)
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-direct-pilot-v1",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "operator_spec_sha256": sha256_file(SPEC_PATH),
        "evidence_class": contract["evidence_class"],
        "population": {
            "examples": len(records),
            "concepts": len(concepts),
            "directions": 2,
            "candidates_per_direction": len(expected_candidates),
            "rows": len(rows),
        },
        "exact_natural_positive_control": {
            "expected_direction_concepts": exact_expected_count,
            "required": 10,
            "per_concept": exact_expected,
            "passes": exact_positive_pass,
        },
        "candidates": candidate_results,
        "tangential_dose_curve": dose,
        "passing_candidates": passing,
        "promoted_to_day45": promoted,
        "disposition": (
            "promote_one_operator_to_direct-emulator refinement"
            if promoted is not None
            else "stop compact-operation search at the acquired causal-population claim"
        ),
        "interpretation_boundary": "Development evidence on an intentionally short existing-example pilot; not population confirmation or fresh evidence.",
    }

    scale_one_name = "tangential_actual_activity.scale_1.0"
    duplicate_scale_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(
                        rows[(record["example_id"], direction, "tangential_actual_activity")][
                            "mean_raw_margins"
                        ]
                    )
                    - np.asarray(
                        rows[(record["example_id"], direction, scale_one_name)][
                            "mean_raw_margins"
                        ]
                    )
                )
            )
        )
        for record in records
        for direction in ("induction", "rescue")
    )
    checks = {
        "preflight_pass": preflight["result"] == "pass",
        "execution_complete": execution["result"] == "complete",
        "exact_shard_count": len(shard_paths) == 13,
        "exact_row_count": len(rows) == 416,
        "exact_row_matrix": set(rows) == expected_keys,
        "single_execution_commit": all(
            row["execution_commit"] == execution_commit for row in rows.values()
        ),
        "all_values_finite": all(
            np.isfinite(row["mean_raw_margins"]).all()
            and np.isfinite(row["target_mean_raw_margins"]).all()
            and np.isfinite(row["activation_rms"])
            for row in rows.values()
        ),
        "all_shard_tangency_checks_pass": all(
            audit["tangency_relative_dot_max"]
            <= spec["implementation_gates"]["tangency_relative_dot_max"]
            for audit in shard_audits
        ),
        "all_haar_audits_pass": all(
            audit["prototype_random_audit"]["pass"]
            and audit["tangential_random_audit"]["pass"]
            for audit in shard_audits
        ),
        "no_hook_leaks": all(audit["hooks_after_batch"] == 0 for audit in shard_audits),
        "scale_one_exact_duplicate": duplicate_scale_error == 0.0,
        "promotion_count_at_most_one": promoted is None or isinstance(promoted, str),
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-direct-pilot-v1-audit",
        "reducer_commit": reducer_commit,
        "execution_commit": execution_commit,
        "checks": checks,
        "scale_one_duplicate_max_abs_error": duplicate_scale_error,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(SUMMARY_PATH, summary)
    write_json_atomic(AUDIT_PATH, audit)
    manifest = {
        "schema_version": 1,
        "procedure": "rapid-k12-direct-pilot-v1-artifact-manifest",
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
    write_json_atomic(MANIFEST_PATH, manifest)
    if audit["result"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "audit": audit["result"],
                "passing_candidates": passing,
                "promoted_to_day45": promoted,
                "candidate_metrics": {
                    name: {
                        "recovery": value["equal_concept_recovery"],
                        "random": value["equal_concept_random_recovery"],
                        "advantage": value["equal_concept_recovery_advantage"],
                        "own_direction_concepts": value[
                            "own_probe_correct_direction_concepts"
                        ],
                        "passes": value["passes"],
                    }
                    for name, value in candidate_results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
