#!/usr/bin/env python3
"""Deterministically reduce and gate each frozen Day 57 stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    CONTRACT_PATH,
    MODEL_KEYS,
    SUMMARY_PATHS,
    expanded_contract,
    load_records,
    panel_spec,
    shard_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(MODEL_KEYS), required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def stage_rows(
    stage: str, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    execution_path = ROOT / f"results/day-57/{stage}-execution.json"
    preflight_path = ROOT / f"results/day-57/{stage}-preflight.json"
    execution = read_json(execution_path)
    preflight = read_json(preflight_path)
    commit = execution["execution_commit"]
    contract_hash = sha256_file(CONTRACT_PATH)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != contract_hash
        or preflight.get("result") != "pass"
        or preflight.get("execution_commit") != commit
        or execution.get("preflight_sha256") != sha256_file(preflight_path)
    ):
        raise RuntimeError(f"Day 57 {stage} execution or preflight is invalid")
    expected_ids = {row["example_id"] for row in load_records(contract, stage)}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for metadata_path in sorted(shard_dir(stage).glob("*.json")):
        metadata = read_json(metadata_path)
        tensor_path = metadata_path.with_suffix(".safetensors")
        if (
            metadata.get("stage") != stage
            or metadata.get("model_key") != MODEL_KEYS[stage]
            or metadata.get("execution_commit") != commit
            or metadata.get("contract_sha256") != contract_hash
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
        ):
            raise RuntimeError(f"Day 57 shard metadata differs: {metadata_path}")
        tensors = load_file(tensor_path)
        batch_size = len(metadata["example_ids"])
        if any(value.shape[0] != batch_size for value in tensors.values()):
            raise RuntimeError(f"Day 57 shard batch geometry differs: {tensor_path}")
        if not all(torch.isfinite(value).all() for value in tensors.values()):
            raise RuntimeError(f"Day 57 shard is nonfinite: {tensor_path}")
        for index, example_id in enumerate(metadata["example_ids"]):
            if example_id in seen_ids:
                raise RuntimeError(f"duplicate Day 57 example: {example_id}")
            seen_ids.add(example_id)
            rows.append(
                {
                    "example_id": example_id,
                    "concept": metadata["concept"],
                    "states": {name: value[index].double() for name, value in tensors.items()},
                }
            )
    if seen_ids != expected_ids or len(rows) != int(panel_spec(contract, stage)["examples"]):
        raise RuntimeError(f"Day 57 {stage} examples are incomplete")
    rows.sort(key=lambda row: row["example_id"])
    return rows, execution, preflight


def state(row: Mapping[str, Any], name: str, field: str) -> torch.Tensor:
    return row["states"][f"{name}.{field}"]


def probe_metrics(
    row: Mapping[str, Any], direction: str, job: str
) -> dict[str, float]:
    target = state(row, f"natural.{direction}.target", "margins").reshape(-1)
    donor = state(row, f"natural.{direction}.donor", "margins").reshape(-1)
    changed = state(row, f"{direction}.{job}", "margins").reshape(-1)
    exact = donor - target
    intervention = changed - target
    denominator = float(exact @ exact)
    if denominator <= 1e-12:
        recovery = 0.0
        residual = float("inf")
    else:
        recovery = float(intervention @ exact) / denominator
        residual = float(torch.linalg.vector_norm(changed - donor)) / math.sqrt(denominator)
    return {
        "probe_recovery": recovery,
        "probe_residual_norm_ratio": residual,
        "probe_effect_norm": float(torch.linalg.vector_norm(intervention)),
        "probe_donor_nearest": float(
            torch.linalg.vector_norm(changed - donor)
            < torch.linalg.vector_norm(changed - target)
        ),
        "k12_recovery": float(state(row, f"{direction}.{job}", "k12_recovery")),
        "k12_residual_norm_ratio": float(
            state(row, f"{direction}.{job}", "k12_residual_norm_ratio")
        ),
        "k12_effect_norm": float(state(row, f"{direction}.{job}", "k12_effect_norm")),
    }


def natural_effect_metrics(row: Mapping[str, Any], direction: str) -> dict[str, float]:
    target = state(row, f"natural.{direction}.target", "margins").reshape(-1)
    donor = state(row, f"natural.{direction}.donor", "margins").reshape(-1)
    return {
        "probe_effect_norm": float(torch.linalg.vector_norm(donor - target)),
        "k12_effect_norm": float(
            state(row, f"natural.{direction}.donor", "k12_effect_norm")
        ),
    }


def concept_medians(
    rows: Sequence[Mapping[str, Any]], values: Sequence[float]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[row["concept"]].append(float(value))
    return {
        concept: float(np.median(concept_values))
        for concept, concept_values in sorted(grouped.items())
    }


def macro(
    rows: Sequence[Mapping[str, Any]], values: Sequence[float]
) -> tuple[float, dict[str, float]]:
    concepts = concept_medians(rows, values)
    return float(np.median(list(concepts.values()))), concepts


def summarize_jobs(
    rows: Sequence[Mapping[str, Any]], direction: str, jobs: Sequence[str]
) -> dict[str, Any]:
    result = {}
    for job in jobs:
        examples = [probe_metrics(row, direction, job) for row in rows]
        metrics = {}
        for key in examples[0]:
            median, concepts = macro(rows, [example[key] for example in examples])
            metrics[key] = {"median_concept": median, "by_concept": concepts}
        metrics["probe_donor_nearest_concepts"] = sum(
            value > 0.5 for value in metrics["probe_donor_nearest"]["by_concept"].values()
        )
        result[job] = metrics
    return result


def implementation_audit(
    stage: str,
    preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_rows = [read_json(path) for path in sorted(shard_dir(stage).glob("*.json"))]
    haar_pass = all(
        audit.get("pass", True)
        for metadata in metadata_rows
        for direction in metadata["audits"].values()
        for name, audit in direction.items()
        if "haar" in name
    )
    algebra_rows = [
        direction["algebra"]
        for metadata in metadata_rows
        for direction in metadata["audits"].values()
        if "algebra" in direction
    ]
    gates = contract["implementation_gates"]
    checks = {
        "preflight_pass": preflight["result"] == "pass",
        "haar_invariants": haar_pass,
        "algebra_attention_reconstruction": not algebra_rows
        or max(row["attention_reconstruction_max_abs"] for row in algebra_rows)
        <= float(gates["attention_reconstruction_max_abs"]),
        "algebra_shapley_closure": not algebra_rows
        or max(row["shapley_closure_max_abs"] for row in algebra_rows)
        <= float(gates["shapley_closure_max_abs"]),
    }
    return {"checks": checks, "result": "pass" if all(checks.values()) else "fail"}


def confirmation_summary(
    rows: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    jobs = contract["stage1_fresh_confirmation"]["jobs_per_direction"]
    directions = {}
    clauses = {}
    gates = contract["stage1_fresh_confirmation"]["all_clauses_required_in_both_directions"]
    for direction in contract["conditions"]["directions"]:
        metrics = summarize_jobs(rows, direction, jobs)
        fixed_survival_values = []
        for row in rows:
            free_identity = state(row, f"{direction}.identity_target", "margins").reshape(-1)
            free_exact = state(row, f"{direction}.exact_donor_all", "margins").reshape(-1)
            frozen_identity = state(
                row, f"frozen.{direction}.identity_target", "margins"
            ).reshape(-1)
            frozen_exact = state(
                row, f"frozen.{direction}.exact_donor_all", "margins"
            ).reshape(-1)
            exact_effect = free_exact - free_identity
            denominator = float(exact_effect @ exact_effect)
            fixed_survival_values.append(
                0.0
                if denominator <= 1e-12
                else float((frozen_exact - frozen_identity) @ exact_effect) / denominator
            )
        fixed_survival, fixed_by_concept = macro(rows, fixed_survival_values)
        values = {
            "exact_donor_probe_recovery": metrics["exact_donor_all"]["probe_recovery"]["median_concept"],
            "exact_donor_k12_recovery": metrics["exact_donor_all"]["k12_recovery"]["median_concept"],
            "exact_donor_probe_donor_nearest_concepts": metrics["exact_donor_all"]["probe_donor_nearest_concepts"],
            "content_hybrid_probe_recovery": metrics["content_hybrid"]["probe_recovery"]["median_concept"],
            "routing_hybrid_probe_recovery": metrics["routing_hybrid"]["probe_recovery"]["median_concept"],
            "content_minus_routing_probe_recovery": metrics["content_hybrid"]["probe_recovery"]["median_concept"] - metrics["routing_hybrid"]["probe_recovery"]["median_concept"],
            "monitoring_prefix_install_probe_recovery": metrics["monitoring_prefix_install"]["probe_recovery"]["median_concept"],
            "monitoring_prefix_remove_probe_recovery": 1.0 - metrics["monitoring_prefix_remove"]["probe_recovery"]["median_concept"],
            "monitoring_prefix_install_advantage_over_haar": metrics["monitoring_prefix_install"]["probe_recovery"]["median_concept"] - metrics["monitoring_prefix_haar"]["probe_recovery"]["median_concept"],
            "monitoring_prefix_install_residual_probe_norm_ratio": metrics["monitoring_prefix_install"]["probe_residual_norm_ratio"]["median_concept"],
            "fixed_norm_exact_effect_survival": fixed_survival,
        }
        passed = {
            "exact_donor_probe_recovery": values["exact_donor_probe_recovery"] >= float(gates["exact_donor_probe_recovery_min"]),
            "exact_donor_k12_recovery": values["exact_donor_k12_recovery"] >= float(gates["exact_donor_k12_recovery_min"]),
            "exact_donor_probe_donor_nearest_concepts": values["exact_donor_probe_donor_nearest_concepts"] >= int(gates["exact_donor_probe_donor_nearest_concepts_min"]),
            "content_hybrid_probe_recovery": values["content_hybrid_probe_recovery"] >= float(gates["content_hybrid_probe_recovery_min"]),
            "routing_hybrid_probe_recovery": values["routing_hybrid_probe_recovery"] <= float(gates["routing_hybrid_probe_recovery_max"]),
            "content_minus_routing_probe_recovery": values["content_minus_routing_probe_recovery"] >= float(gates["content_minus_routing_probe_recovery_min"]),
            "monitoring_prefix_install_probe_recovery": values["monitoring_prefix_install_probe_recovery"] >= float(gates["monitoring_prefix_install_probe_recovery_min"]),
            "monitoring_prefix_remove_probe_recovery": values["monitoring_prefix_remove_probe_recovery"] >= float(gates["monitoring_prefix_remove_probe_recovery_min"]),
            "monitoring_prefix_install_advantage_over_haar": values["monitoring_prefix_install_advantage_over_haar"] >= float(gates["monitoring_prefix_install_advantage_over_haar_min"]),
            "monitoring_prefix_install_residual_probe_norm_ratio": values["monitoring_prefix_install_residual_probe_norm_ratio"] <= float(gates["monitoring_prefix_install_residual_probe_norm_ratio_max"]),
            "fixed_norm_exact_effect_survival": values["fixed_norm_exact_effect_survival"] >= float(gates["fixed_norm_exact_effect_survival_min"]),
        }
        directions[direction] = {
            "jobs": metrics,
            "fixed_norm_exact_effect_survival": {
                "median_concept": fixed_survival,
                "by_concept": fixed_by_concept,
            },
        }
        clauses[direction] = {"values": values, "pass": passed, "all_pass": all(passed.values())}
    implementation = implementation_audit("confirmation", preflight, contract)
    passed = implementation["result"] == "pass" and all(
        value["all_pass"] for value in clauses.values()
    )
    return {
        "schema_version": 1,
        "procedure": "prospective-day57-fresh-confirmation-reduction-v1",
        "stage": "confirmation",
        "analysis_commit": git_head(),
        "execution_commit": execution["execution_commit"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_class": "fresh causal confirmation on 176 previously unselected examples",
        "implementation": implementation,
        "directions": directions,
        "gate_clauses": clauses,
        "decision": "fresh_confirmation_pass" if passed else "fresh_confirmation_fail_stop",
        "stage2_eligible": passed,
    }


def trace_summary(
    rows: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    confirmation = read_json(SUMMARY_PATHS["confirmation"])
    if confirmation.get("decision") != "fresh_confirmation_pass":
        raise RuntimeError("trace reduction is ineligible")
    jobs = contract["stage2_value_pathway"]["jobs_per_direction"]
    candidates = contract["stage2_value_pathway"]["candidate_simplicity_order"]
    gate = contract["stage2_value_pathway"]["candidate_gate_both_directions"]
    directions = {}
    candidate_passes: dict[str, dict[str, bool]] = {candidate: {} for candidate in candidates}
    for direction in contract["conditions"]["directions"]:
        metrics = summarize_jobs(rows, direction, jobs)
        natural_probe, natural_probe_concepts = macro(
            rows, [natural_effect_metrics(row, direction)["probe_effect_norm"] for row in rows]
        )
        natural_k12, natural_k12_concepts = macro(
            rows, [natural_effect_metrics(row, direction)["k12_effect_norm"] for row in rows]
        )
        interaction, interaction_concepts = macro(
            rows,
            [
                probe_metrics(row, direction, "qkv_prefix")["probe_recovery"]
                - probe_metrics(row, direction, "v_prefix")["probe_recovery"]
                - probe_metrics(row, direction, "qk_prefix")["probe_recovery"]
                for row in rows
            ],
        )
        for candidate in candidates:
            values = metrics[candidate]
            candidate_passes[candidate][direction] = (
                values["probe_recovery"]["median_concept"]
                >= float(gate["median_concept_probe_recovery_min"])
                and values["k12_recovery"]["median_concept"]
                >= float(gate["median_concept_k12_recovery_min"])
                and values["probe_recovery"]["median_concept"]
                - metrics["exact_delta_haar"]["probe_recovery"]["median_concept"]
                >= float(gate["median_probe_advantage_over_exact_haar_min"])
                and values["probe_donor_nearest_concepts"]
                >= int(gate["probe_donor_nearest_concepts_min"])
            )
        directions[direction] = {
            "jobs": metrics,
            "natural_effect_norms": {
                "probe": {"median_concept": natural_probe, "by_concept": natural_probe_concepts},
                "k12": {"median_concept": natural_k12, "by_concept": natural_k12_concepts},
            },
            "qk_v_interaction_probe_recovery": {
                "median_concept": interaction,
                "by_concept": interaction_concepts,
            },
        }
    qualified = [
        candidate
        for candidate in candidates
        if all(candidate_passes[candidate].values())
    ]
    selected = qualified[0] if qualified else None
    implementation = implementation_audit("trace", preflight, contract)
    return {
        "schema_version": 1,
        "procedure": "prospective-day57-value-pathway-reduction-v1",
        "stage": "trace",
        "analysis_commit": git_head(),
        "execution_commit": execution["execution_commit"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "confirmation_summary_sha256": sha256_file(SUMMARY_PATHS["confirmation"]),
        "evidence_class": "causal pathway tracing on a separate 44-example panel",
        "implementation": implementation,
        "implementation_result": implementation["result"],
        "directions": directions,
        "candidate_passes": candidate_passes,
        "qualified_candidates": qualified,
        "selected_pathway": selected,
        "pathway_decision": selected if selected is not None else "unresolved_interface",
        "stage3_eligible": implementation["result"] == "pass",
    }


def precursor_summary(
    rows: Sequence[Mapping[str, Any]],
    execution: Mapping[str, Any],
    preflight: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    trace = read_json(SUMMARY_PATHS["trace"])
    if trace.get("implementation_result") != "pass":
        raise RuntimeError("precursor reduction is ineligible")
    jobs = contract["stage3_exact_precursor"]["jobs_per_direction"]
    qualified = trace["qualified_candidates"]
    thresholds = contract["stage3_exact_precursor"]["acquisition_gate_both_directions"]
    directions = {}
    natural_passes = {}
    qualified_ratios: dict[str, dict[str, float]] = {candidate: {} for candidate in qualified}
    for direction in contract["conditions"]["directions"]:
        metrics = summarize_jobs(rows, direction, jobs)
        precursor_probe, precursor_probe_concepts = macro(
            rows, [natural_effect_metrics(row, direction)["probe_effect_norm"] for row in rows]
        )
        precursor_k12, precursor_k12_concepts = macro(
            rows, [natural_effect_metrics(row, direction)["k12_effect_norm"] for row in rows]
        )
        chameleon_natural = trace["directions"][direction]["natural_effect_norms"]
        probe_ratio = precursor_probe / max(
            float(chameleon_natural["probe"]["median_concept"]), 1e-12
        )
        k12_ratio = precursor_k12 / max(
            float(chameleon_natural["k12"]["median_concept"]), 1e-12
        )
        natural_passes[direction] = (
            probe_ratio
            <= float(thresholds["natural_probe_effect_norm_precursor_over_chameleon_max"])
            and k12_ratio
            <= float(thresholds["natural_k12_effect_norm_precursor_over_chameleon_max"])
        )
        candidate_comparison = {}
        for candidate in contract["stage2_value_pathway"]["candidate_simplicity_order"]:
            precursor_effect = metrics[candidate]["probe_effect_norm"]["median_concept"]
            chameleon_effect = trace["directions"][direction]["jobs"][candidate][
                "probe_effect_norm"
            ]["median_concept"]
            ratio = precursor_effect / max(float(chameleon_effect), 1e-12)
            candidate_comparison[candidate] = {
                "precursor_probe_effect_norm": precursor_effect,
                "chameleon_probe_effect_norm": chameleon_effect,
                "precursor_over_chameleon": ratio,
            }
            if candidate in qualified:
                qualified_ratios[candidate][direction] = ratio
        directions[direction] = {
            "jobs": metrics,
            "natural_effect_norms": {
                "probe": {"median_concept": precursor_probe, "by_concept": precursor_probe_concepts},
                "k12": {"median_concept": precursor_k12, "by_concept": precursor_k12_concepts},
            },
            "natural_precursor_over_chameleon": {"probe": probe_ratio, "k12": k12_ratio},
            "candidate_effect_comparison": candidate_comparison,
        }
    implementation = implementation_audit("precursor", preflight, contract)
    path_ratio_pass = all(
        ratio <= float(thresholds["qualified_path_probe_effect_norm_precursor_over_chameleon_max"])
        for values in qualified_ratios.values()
        for ratio in values.values()
    )
    if implementation["result"] != "pass" or not qualified:
        classification = "ambiguous"
    elif all(natural_passes.values()) and path_ratio_pass:
        classification = "acquired"
    elif any(
        ratio > float(thresholds["qualified_path_probe_effect_norm_precursor_over_chameleon_max"])
        for values in qualified_ratios.values()
        for ratio in values.values()
    ):
        classification = "conserved"
    else:
        classification = "ambiguous"
    return {
        "schema_version": 1,
        "procedure": "prospective-day57-exact-precursor-acquisition-reduction-v1",
        "stage": "precursor",
        "analysis_commit": git_head(),
        "execution_commit": execution["execution_commit"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "trace_summary_sha256": sha256_file(SUMMARY_PATHS["trace"]),
        "evidence_class": "exact-precursor causal comparison on the 44-example tracing panel",
        "implementation": implementation,
        "implementation_result": implementation["result"],
        "directions": directions,
        "natural_gate_passes": natural_passes,
        "qualified_candidates_from_chameleon": qualified,
        "qualified_candidate_precursor_over_chameleon": qualified_ratios,
        "acquisition_classification": classification,
    }


def build_summary(stage: str) -> dict[str, Any]:
    contract = expanded_contract()
    rows, execution, preflight = stage_rows(stage, contract)
    if stage == "confirmation":
        return confirmation_summary(rows, execution, preflight, contract)
    if stage == "trace":
        return trace_summary(rows, execution, preflight, contract)
    return precursor_summary(rows, execution, preflight, contract)


def main() -> None:
    args = parse_args()
    summary = build_summary(args.stage)
    path = SUMMARY_PATHS[args.stage]
    if args.check:
        if path.read_bytes() != canonical_bytes(summary):
            raise RuntimeError(f"Day 57 {args.stage} reduction is not byte-identical")
        print(f"Day 57 {args.stage} reduction reproduces byte-identically.")
        return
    write_json_atomic(path, summary)
    print(canonical_bytes(summary).decode(), end="")


if __name__ == "__main__":
    main()
