#!/usr/bin/env python3
"""Reduce the eligible Day 55 QKV completion localization."""

from __future__ import annotations

import hashlib
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors.torch import load_file

from day52_analyze_reciprocal_reconfiguration import (
    distance_metrics,
    json_bytes,
    read_json,
    state,
    vector_metrics,
    write_json,
)
from day55_run_qkv_completion_localization import (
    CONTRACT_PATH,
    DAY54_SUMMARY_PATH,
    EXECUTION_PATH,
    PREFLIGHT_PATH,
    SHARD_DIR,
    completion_job_names,
    expanded_contract,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/day-55/qkv-completion-summary.json"
AUDIT_PATH = ROOT / "results/day-55/qkv-completion-audit.json"
METRICS_PATH = ROOT / "results/day-55/qkv-completion-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-55/qkv-completion-artifact-manifest.json"


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


def expected_names(contract: Mapping[str, Any]) -> set[str]:
    return {
        *(f"natural_{name}" for name in contract["execution"]["natural_states"]),
        *(
            f"intervention_{direction}.{job}"
            for direction in contract["directions"]
            for job in completion_job_names(contract)
        ),
    }


def load_rows(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution = read_json(EXECUTION_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution.get("execution_commit")
    ):
        raise RuntimeError("Day 55 completion inputs are not exact and complete")
    rows = []
    identity_k12_errors = []
    identity_margin_errors = []
    random_passes = []
    tensor_hashes = []
    probe_name_sets = set()
    state_rows = 0
    for concept in sorted(contract["conditions"]["pairs"]):
        tensor_path = SHARD_DIR / f"{concept}.safetensors"
        metadata_path = SHARD_DIR / f"{concept}.json"
        metadata = read_json(metadata_path)
        if (
            metadata["execution_commit"] != execution["execution_commit"]
            or metadata["contract_sha256"] != sha256_file(CONTRACT_PATH)
            or metadata["tensor_sha256"] != sha256_file(tensor_path)
            or set(metadata["state_names"]) != expected_names(contract)
            or int(metadata["state_count"]) != 44
        ):
            raise RuntimeError(f"Day 55 completion shard differs: {concept}")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        probe_name_sets.add(tuple(metadata["probe_names"]))
        tensor_hashes.append(metadata["tensor_sha256"])
        random_passes.extend(
            bool(value["pass"]) for value in metadata["random_audits"].values()
        )
        state_rows += int(metadata["state_count"]) * len(metadata["example_ids"])
        natural = {
            name: state(tensors, f"natural_{name}")
            for name in contract["execution"]["natural_states"]
        }
        for direction, specification in contract["directions"].items():
            target = natural[specification["target"]]
            donor = natural[specification["donor"]]
            normal = natural[specification["normal_control"]]
            different = natural[specification["different_control"]]
            interventions = {
                job: state(tensors, f"intervention_{direction}.{job}")
                for job in completion_job_names(contract)
            }
            identity_k12_errors.append(
                float(
                    (interventions["identity_target"]["k12"] - target["k12"])
                    .abs()
                    .max()
                )
            )
            identity_margin_errors.append(
                float(
                    (interventions["identity_target"]["margins"] - target["margins"])
                    .abs()
                    .max()
                )
            )
            qkv = interventions["qkv_baseline"]
            exact = interventions["exact_donor_k12_all"]
            endpoints = {
                "target": target,
                "donor": donor,
                "normal": normal,
                "different": different,
            }
            for job, output in interventions.items():
                repair = {
                    modality: vector_metrics(
                        output[field], qkv[field], exact[field], mask
                    )["recovery"]
                    for modality, field in (
                        ("k12", "k12"),
                        ("monitor", "monitor"),
                        ("probe", "margins"),
                    )
                }
                probe_distances = distance_metrics(
                    output["margins"],
                    {name: value["margins"] for name, value in endpoints.items()},
                    mask,
                )
                donor_recovery = vector_metrics(
                    output["margins"], target["margins"], donor["margins"], mask
                )["recovery"]
                for index, example_id in enumerate(metadata["example_ids"]):
                    rows.append(
                        {
                            "concept": concept,
                            "example_id": example_id,
                            "direction": direction,
                            "job": job,
                            "k12_repair_recovery": float(repair["k12"][index]),
                            "monitor_repair_recovery": float(repair["monitor"][index]),
                            "probe_repair_recovery": float(repair["probe"][index]),
                            "probe_donor_recovery": float(donor_recovery[index]),
                            "probe_endpoint": probe_distances[index],
                        }
                    )
    return rows, {
        "execution": execution,
        "preflight": preflight,
        "state_rows": state_rows,
        "identity_k12_max_abs": max(identity_k12_errors),
        "identity_margin_max_abs": max(identity_margin_errors),
        "random_audits_pass": all(random_passes),
        "tensor_hash_count": len(set(tensor_hashes)),
        "probe_name_sets": [list(value) for value in sorted(probe_name_sets)],
    }


def concept_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["concept"], row["direction"], row["job"])].append(row)
    result = []
    for (concept, direction, job), values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError("Day 55 completion concept cell is incomplete")
        distances = {
            endpoint: float(
                np.mean(
                    [value["probe_endpoint"]["distances"][endpoint] for value in values]
                )
            )
            for endpoint in ("target", "donor", "normal", "different")
        }
        result.append(
            {
                "concept": concept,
                "direction": direction,
                "job": job,
                "k12_repair_recovery": float(
                    np.mean([value["k12_repair_recovery"] for value in values])
                ),
                "monitor_repair_recovery": float(
                    np.mean([value["monitor_repair_recovery"] for value in values])
                ),
                "probe_repair_recovery": float(
                    np.mean([value["probe_repair_recovery"] for value in values])
                ),
                "probe_donor_recovery": float(
                    np.mean([value["probe_donor_recovery"] for value in values])
                ),
                "probe_nearest_endpoint": min(distances, key=distances.get),
                "probe_endpoint_distances": distances,
            }
        )
    return result


def summarize(
    contract: Mapping[str, Any], concepts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    candidate_names = tuple(completion_job_names(contract)[4:])
    by_direction = {}
    qualified = set()
    for direction in contract["directions"]:
        direction_rows = [row for row in concepts if row["direction"] == direction]
        qkv = {
            row["concept"]: row
            for row in direction_rows
            if row["job"] == "qkv_baseline"
        }
        exact = [row for row in direction_rows if row["job"] == "exact_donor_k12_all"]
        haar = [
            row for row in direction_rows if row["job"] == "qkv_plus_haar_completion"
        ]
        haar_probe_repair = float(
            np.median([row["probe_repair_recovery"] for row in haar])
        )
        candidate_summaries = []
        for job in candidate_names:
            selected = [row for row in direction_rows if row["job"] == job]
            donor_count = sum(
                row["probe_nearest_endpoint"] == "donor" for row in selected
            )
            qkv_donor_count = sum(
                row["probe_nearest_endpoint"] == "donor" for row in qkv.values()
            )
            gained = sum(
                row["probe_nearest_endpoint"] == "donor"
                and qkv[row["concept"]]["probe_nearest_endpoint"] != "donor"
                for row in selected
            )
            lost = sum(
                row["probe_nearest_endpoint"] != "donor"
                and qkv[row["concept"]]["probe_nearest_endpoint"] == "donor"
                for row in selected
            )
            probe_repair = float(
                np.median([row["probe_repair_recovery"] for row in selected])
            )
            summary = {
                "job": job,
                "median_concept_k12_repair_recovery": float(
                    np.median([row["k12_repair_recovery"] for row in selected])
                ),
                "median_concept_monitor_repair_recovery": float(
                    np.median([row["monitor_repair_recovery"] for row in selected])
                ),
                "median_concept_probe_repair_recovery": probe_repair,
                "probe_repair_advantage_over_haar": probe_repair - haar_probe_repair,
                "probe_donor_nearest_concepts": donor_count,
                "qkv_baseline_probe_donor_nearest_concepts": qkv_donor_count,
                "net_probe_donor_nearest_improvement": donor_count - qkv_donor_count,
                "gained_probe_donor_nearest_concepts": gained,
                "lost_probe_donor_nearest_concepts": lost,
            }
            summary["qualifies"] = (
                summary["median_concept_probe_repair_recovery"] >= 0.50
                and summary["probe_repair_advantage_over_haar"] >= 0.25
                and summary["net_probe_donor_nearest_improvement"] >= 2
            )
            if summary["qualifies"]:
                qualified.add(job)
            candidate_summaries.append(summary)
        by_direction[direction] = {
            "qkv_baseline_probe_donor_nearest_concepts": sum(
                row["probe_nearest_endpoint"] == "donor" for row in qkv.values()
            ),
            "exact_all_probe_donor_nearest_concepts": sum(
                row["probe_nearest_endpoint"] == "donor" for row in exact
            ),
            "median_exact_all_probe_repair_recovery": float(
                np.median([row["probe_repair_recovery"] for row in exact])
            ),
            "median_haar_probe_repair_recovery": haar_probe_repair,
            "candidate_rank_by_probe_repair": sorted(
                candidate_summaries,
                key=lambda value: value["median_concept_probe_repair_recovery"],
                reverse=True,
            ),
        }
    return {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "candidate": "exact completion of Day 52 QKV-produced K12",
        "direction_summaries": by_direction,
        "qualified_localizers": sorted(qualified),
        "localization_disposition": (
            "localized_selected_k12_shortfall"
            if qualified
            else "missing_monitor_sensitive_component_distributed"
        ),
        "ineligible_branch_not_run": "exact-K12 by residual-context factorial",
        "boundary": contract["interpretation"]["boundary"],
    }


def reduce_once(
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, inputs = load_rows(contract)
    concepts = concept_rows(rows)
    summary = summarize(contract, concepts)
    checks = {
        "day54_pass_branch_eligible": read_json(DAY54_SUMMARY_PATH).get("branch")
        == "exact_k12_self_contained_pass",
        "preflight_pass": inputs["preflight"].get("result") == "pass",
        "execution_complete": bool(inputs["execution"].get("complete")),
        "exact_state_row_count": inputs["state_rows"] == 1144,
        "identity_k12_within_tolerance": inputs["identity_k12_max_abs"] <= 0.02,
        "identity_margin_within_tolerance": inputs["identity_margin_max_abs"] <= 0.05,
        "random_audits_pass": bool(inputs["random_audits_pass"]),
        "hooks_removed": inputs["execution"].get("hooks_after_execution") == 0,
        "thirteen_unique_tensor_hashes": inputs["tensor_hash_count"] == 13,
        "exact_probe_order": len(inputs["probe_name_sets"]) == 1
        and len(inputs["probe_name_sets"][0]) == 13,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "k12_repair_recovery",
                "monitor_repair_recovery",
                "probe_repair_recovery",
                "probe_donor_recovery",
            )
        ),
    }
    audit = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-audit",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "checks": checks,
        "implementation_pass": all(checks.values()),
        "observed": {
            "example_metric_rows": len(rows),
            "concept_metric_rows": len(concepts),
            "state_rows": inputs["state_rows"],
            "identity_k12_max_abs": inputs["identity_k12_max_abs"],
            "identity_margin_max_abs": inputs["identity_margin_max_abs"],
            "tensor_hash_count": inputs["tensor_hash_count"],
        },
        "two_in_memory_reductions_byte_identical": None,
    }
    metrics = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-metrics",
        "rows": rows,
        "concept_rows": concepts,
    }
    return summary, audit, metrics


def output_manifest() -> dict[str, Any]:
    paths = [
        CONTRACT_PATH,
        PREFLIGHT_PATH,
        EXECUTION_PATH,
        SUMMARY_PATH,
        AUDIT_PATH,
        METRICS_PATH,
        *sorted(SHARD_DIR.glob("*.json")),
        *sorted(SHARD_DIR.glob("*.safetensors")),
    ]
    return {
        "schema_version": 1,
        "procedure": "day55-qkv-completion-artifacts-v1",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def main() -> None:
    contract = expanded_contract()
    first = reduce_once(contract)
    second = reduce_once(contract)
    if json_bytes(first) != json_bytes(second):
        raise RuntimeError("two Day 55 completion reductions differ")
    summary, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    write_json(SUMMARY_PATH, summary)
    write_json(AUDIT_PATH, audit)
    write_json(METRICS_PATH, metrics)
    write_json(MANIFEST_PATH, output_manifest())
    if not audit["implementation_pass"]:
        raise RuntimeError(f"Day 55 completion implementation failed: {audit}")


if __name__ == "__main__":
    main()
