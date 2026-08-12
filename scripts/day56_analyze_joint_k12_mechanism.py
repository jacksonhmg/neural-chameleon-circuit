#!/usr/bin/env python3
"""Reduce and gate the frozen Day 56 joint-K12 mechanism experiment."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from day52_analyze_reciprocal_reconfiguration import (
    distance_metrics,
    json_bytes,
    state,
    vector_metrics,
    write_json,
)
from day56_run_joint_k12_mechanism import (
    CONTRACT_PATH,
    DAY44_SUMMARY_PATH,
    DAY47_SUMMARY_PATH,
    EXECUTION_PATH,
    PREFLIGHT_PATH,
    SHARD_DIR,
    expanded_contract,
    free_job_names,
    frozen_job_names,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results/day-56/joint-k12-mechanism-summary.json"
ATTENTION_PATH = ROOT / "results/day-56/joint-k12-attention-algebra-summary.json"
GEOMETRY_PATH = ROOT / "results/day-56/joint-k12-downstream-geometry-summary.json"
CLOSURE_PATH = ROOT / "results/day-56/joint-k12-causal-closure-summary.json"
AUDIT_PATH = ROOT / "results/day-56/joint-k12-mechanism-audit.json"
METRICS_PATH = ROOT / "results/day-56/joint-k12-mechanism-metrics.json"
MANIFEST_PATH = ROOT / "results/day-56/joint-k12-mechanism-artifact-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def expected_names(contract: Mapping[str, Any]) -> set[str]:
    return {
        *(f"natural_{name}" for name in contract["execution"]["natural_states"]),
        *(
            f"free_{direction}.{job}"
            for direction in contract["directions"]
            for job in free_job_names(contract)
        ),
        *(
            f"frozen_norm_{direction}.{job}"
            for direction in contract["directions"]
            for job in frozen_job_names(contract)
        ),
    }


def _effect_metrics(changed: torch.Tensor, trajectory: torch.Tensor) -> dict[str, float]:
    changed = changed.double().reshape(-1)
    trajectory = trajectory.double().reshape(-1)
    numerator = float(changed @ trajectory)
    denominator = float(trajectory.square().sum().clamp(min=1e-8))
    changed_norm = float(changed.square().sum().clamp(min=1e-8))
    return {
        "recovery": numerator / denominator,
        "cosine": numerator / math.sqrt(denominator * changed_norm),
        "norm_ratio": math.sqrt(changed_norm / denominator),
    }


def _concept_cells(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in keys)].append(row)
    result = []
    for cell, values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError(f"Day 56 concept cell is incomplete: {cell}")
        scalar_keys = [
            key
            for key, value in values[0].items()
            if key not in {*keys, "example_id", "endpoint_distances"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ]
        reduced = {key: cell[index] for index, key in enumerate(keys)}
        for key in scalar_keys:
            reduced[key] = float(np.mean([float(value[key]) for value in values]))
        if "endpoint_distances" in values[0]:
            distances = {
                endpoint: float(
                    np.mean(
                        [value["endpoint_distances"][endpoint] for value in values]
                    )
                )
                for endpoint in ("target", "donor", "normal", "different")
            }
            reduced["endpoint_distances"] = distances
            reduced["nearest_endpoint"] = min(distances, key=distances.get)
        result.append(reduced)
    return result


def dose_geometry(
    outputs: Mapping[float, torch.Tensor], identity: torch.Tensor, exact: torch.Tensor
) -> list[dict[str, float]]:
    alphas = sorted(outputs)
    rows = []
    for example_index in range(identity.shape[0]):
        target = identity[example_index].double()
        delta = exact[example_index].double() - target
        actual = torch.stack([outputs[alpha][example_index].double() for alpha in alphas])
        predicted = torch.stack([target + alpha * delta for alpha in alphas])
        mean = actual.mean(dim=0, keepdim=True)
        sse = float((actual - predicted).square().sum())
        sst = float((actual - mean).square().sum().clamp(min=1e-8))
        full_norm = float(delta.square().sum().sqrt().clamp(min=1e-8))
        in_range = [index for index, alpha in enumerate(alphas) if 0 <= alpha <= 1]
        maximum_deviation = max(
            float((actual[index] - predicted[index]).square().sum().sqrt()) / full_norm
            for index in in_range
        )
        rows.append(
            {
                "r_squared": 1.0 - sse / sst,
                "max_in_range_secant_deviation_ratio": maximum_deviation,
            }
        )
    return rows


def load_metrics(contract: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = read_json(EXECUTION_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != contract_hash
        or preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution.get("execution_commit")
    ):
        raise RuntimeError("Day 56 execution inputs are not exact and complete")
    candidate_rows = []
    job_rows = []
    dose_rows = []
    jacobian_rows = []
    mass_rows = []
    identity_k12_errors = []
    identity_margin_errors = []
    jacobian_identity_errors = []
    random_passes = []
    tensor_hashes = []
    algebra_reconstruction = []
    shapley_closure = []
    state_rows = 0
    probe_name_sets = set()
    for concept in sorted(contract["conditions"]["pairs"]):
        tensor_path = SHARD_DIR / f"{concept}.safetensors"
        metadata_path = SHARD_DIR / f"{concept}.json"
        metadata = read_json(metadata_path)
        if (
            metadata.get("execution_commit") != execution["execution_commit"]
            or metadata.get("contract_sha256") != contract_hash
            or metadata.get("tensor_sha256") != sha256_file(tensor_path)
            or set(metadata.get("state_names", [])) != expected_names(contract)
            or int(metadata.get("state_count", -1))
            != int(contract["execution"]["states_per_example"])
        ):
            raise RuntimeError(f"Day 56 shard differs: {concept}")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        tensor_hashes.append(metadata["tensor_sha256"])
        probe_name_sets.add(tuple(metadata["probe_names"]))
        state_rows += int(metadata["state_count"]) * len(metadata["example_ids"])
        for audit in metadata["algebra_audits"].values():
            algebra_reconstruction.append(audit["attention_reconstruction_max_abs"])
            shapley_closure.append(audit["shapley_closure_max_abs"])
        for direction_audits in metadata["random_audits"].values():
            random_passes.extend(
                bool(audit["pass"]) for audit in direction_audits.values()
            )
        natural = {
            name: state(tensors, f"natural_{name}")
            for name in contract["execution"]["natural_states"]
        }
        for direction, specification in contract["directions"].items():
            free = {
                job: state(tensors, f"free_{direction}.{job}")
                for job in free_job_names(contract)
            }
            frozen = {
                job: state(tensors, f"frozen_norm_{direction}.{job}")
                for job in frozen_job_names(contract)
            }
            identity = free["identity_target"]
            exact = free["exact_donor_all"]
            target = natural[specification["target"]]
            donor = natural[specification["donor"]]
            normal = natural[specification["normal_control"]]
            different = natural[specification["different_control"]]
            identity_k12_errors.append(float((identity["k12"] - target["k12"]).abs().max()))
            identity_margin_errors.append(
                float((identity["margins"] - target["margins"]).abs().max())
            )
            endpoints = {
                "target": target["margins"],
                "donor": donor["margins"],
                "normal": normal["margins"],
                "different": different["margins"],
            }
            full_probe = exact["margins"] - identity["margins"]
            for job, output in free.items():
                probe = vector_metrics(
                    output["margins"], identity["margins"], exact["margins"], mask
                )
                k12 = vector_metrics(output["k12"], identity["k12"], exact["k12"], mask)
                nearest = distance_metrics(output["margins"], endpoints, mask)
                for index, example_id in enumerate(metadata["example_ids"]):
                    job_rows.append(
                        {
                            "concept": concept,
                            "example_id": example_id,
                            "direction": direction,
                            "job": job,
                            "probe_recovery": float(probe["recovery"][index]),
                            "probe_cosine": float(probe["cosine"][index]),
                            "k12_recovery": float(k12["recovery"][index]),
                            "endpoint_distances": nearest[index]["distances"],
                        }
                    )
            for candidate in contract["candidate_simplicity_order"]:
                install = free[f"candidate.{candidate}.install"]
                remove = free[f"candidate.{candidate}.remove"]
                random = free[f"candidate.{candidate}.haar_install"]
                install_probe = vector_metrics(
                    install["margins"], identity["margins"], exact["margins"], mask
                )
                install_k12 = vector_metrics(
                    install["k12"], identity["k12"], exact["k12"], mask
                )
                random_probe = vector_metrics(
                    random["margins"], identity["margins"], exact["margins"], mask
                )
                nearest = distance_metrics(install["margins"], endpoints, mask)
                for index, example_id in enumerate(metadata["example_ids"]):
                    removal = _effect_metrics(
                        exact["margins"][index] - remove["margins"][index],
                        full_probe[index],
                    )
                    residual = float(
                        torch.linalg.vector_norm(
                            exact["margins"][index] - install["margins"][index]
                        )
                        / torch.linalg.vector_norm(full_probe[index]).clamp(min=1e-8)
                    )
                    candidate_rows.append(
                        {
                            "concept": concept,
                            "example_id": example_id,
                            "direction": direction,
                            "candidate": candidate,
                            "install_probe_recovery": float(
                                install_probe["recovery"][index]
                            ),
                            "install_probe_cosine": float(install_probe["cosine"][index]),
                            "install_k12_recovery": float(install_k12["recovery"][index]),
                            "remove_probe_recovery": removal["recovery"],
                            "remove_probe_cosine": removal["cosine"],
                            "haar_probe_recovery": float(random_probe["recovery"][index]),
                            "residual_probe_norm_ratio": residual,
                            "endpoint_distances": nearest[index]["distances"],
                        }
                    )
            alpha_outputs = {
                0.0: identity["margins"],
                0.25: free["dose_0.25"]["margins"],
                0.5: free["dose_0.50"]["margins"],
                0.75: free["dose_0.75"]["margins"],
                1.0: exact["margins"],
                1.25: free["dose_1.25"]["margins"],
            }
            free_geometry = dose_geometry(
                alpha_outputs, identity["margins"], exact["margins"]
            )
            frozen_identity = frozen["identity_target"]["margins"]
            frozen_exact = frozen["exact_donor_all"]["margins"]
            frozen_alpha_outputs = {
                0.0: frozen_identity,
                0.25: frozen["dose_0.25"]["margins"],
                0.5: frozen["dose_0.50"]["margins"],
                0.75: frozen["dose_0.75"]["margins"],
                1.0: frozen_exact,
                1.25: frozen["dose_1.25"]["margins"],
            }
            fixed_geometry = dose_geometry(
                frozen_alpha_outputs, frozen_identity, frozen_exact
            )
            for index, example_id in enumerate(metadata["example_ids"]):
                survival = _effect_metrics(
                    frozen_exact[index] - frozen_identity[index], full_probe[index]
                )
                dose_rows.append(
                    {
                        "concept": concept,
                        "example_id": example_id,
                        "direction": direction,
                        "free_r_squared": free_geometry[index]["r_squared"],
                        "free_max_in_range_secant_deviation_ratio": free_geometry[index][
                            "max_in_range_secant_deviation_ratio"
                        ],
                        "frozen_norm_r_squared": fixed_geometry[index]["r_squared"],
                        "frozen_norm_max_in_range_secant_deviation_ratio": fixed_geometry[
                            index
                        ]["max_in_range_secant_deviation_ratio"],
                        "fixed_norm_effect_survival_recovery": survival["recovery"],
                        "fixed_norm_effect_survival_cosine": survival["cosine"],
                    }
                )
            jac_target = tensors[f"jacobian.{direction}.target_margins"]
            jac_prediction = tensors[f"jacobian.{direction}.exact_delta_prediction"]
            jac_candidates = tensors[
                f"jacobian.{direction}.candidate_delta_predictions"
            ]
            singular = tensors[f"jacobian.{direction}.singular_values"]
            jacobian_identity_errors.append(
                float((jac_target - identity["margins"]).abs().max())
            )
            for index, example_id in enumerate(metadata["example_ids"]):
                prediction = _effect_metrics(jac_prediction[index], full_probe[index])
                energy = singular[index].double().square()
                cumulative = energy.cumsum(dim=0) / energy.sum().clamp(min=1e-12)
                rank90 = int((cumulative < 0.9).sum()) + 1
                row = {
                    "concept": concept,
                    "example_id": example_id,
                    "direction": direction,
                    "exact_effect_recovery": prediction["recovery"],
                    "exact_effect_cosine": prediction["cosine"],
                    "effective_rank_90": rank90,
                    "singular_values": singular[index].tolist(),
                    "candidate_predictions": {},
                }
                for candidate_index, candidate in enumerate(
                    contract["candidate_simplicity_order"]
                ):
                    candidate_effect = _effect_metrics(
                        jac_candidates[index, candidate_index], full_probe[index]
                    )
                    row["candidate_predictions"][candidate] = candidate_effect
                jacobian_rows.append(row)
            for region in contract["atomic_source_regions"]:
                target_mass = tensors[f"attention.{direction}.target_mass.{region}"]
                donor_mass = tensors[f"attention.{direction}.donor_mass.{region}"]
                for index, example_id in enumerate(metadata["example_ids"]):
                    selected = mask[index, :, None].expand_as(target_mass[index])
                    difference = (donor_mass[index] - target_mass[index]).masked_select(
                        selected
                    )
                    mass_rows.append(
                        {
                            "concept": concept,
                            "example_id": example_id,
                            "direction": direction,
                            "region": region,
                            "mean_signed_mass_change": float(difference.mean()),
                            "mean_absolute_mass_change": float(difference.abs().mean()),
                        }
                    )
    metrics = {
        "schema_version": 1,
        "candidate_rows": candidate_rows,
        "job_rows": job_rows,
        "dose_rows": dose_rows,
        "jacobian_rows": jacobian_rows,
        "attention_mass_rows": mass_rows,
    }
    audit_inputs = {
        "execution": execution,
        "preflight": preflight,
        "state_rows": state_rows,
        "identity_k12_max_abs": max(identity_k12_errors),
        "identity_margin_max_abs": max(identity_margin_errors),
        "jacobian_identity_margin_max_abs": max(jacobian_identity_errors),
        "random_audits_pass": all(random_passes),
        "tensor_hash_count": len(set(tensor_hashes)),
        "probe_name_sets": [list(value) for value in sorted(probe_name_sets)],
        "attention_reconstruction_max_abs": max(algebra_reconstruction),
        "shapley_closure_max_abs": max(shapley_closure),
    }
    return metrics, audit_inputs


def summarize_closure(
    contract: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    concepts = _concept_cells(
        metrics["candidate_rows"], ("concept", "direction", "candidate")
    )
    by_direction = {}
    qualification: dict[str, dict[str, bool]] = {
        candidate: {} for candidate in contract["candidate_simplicity_order"]
    }
    gate = contract["candidate_gate"]
    for direction in contract["directions"]:
        summaries = []
        for candidate in contract["candidate_simplicity_order"]:
            rows = [
                row
                for row in concepts
                if row["direction"] == direction and row["candidate"] == candidate
            ]
            summary = {
                "candidate": candidate,
                "median_concept_install_probe_recovery": float(
                    np.median([row["install_probe_recovery"] for row in rows])
                ),
                "median_concept_remove_probe_recovery": float(
                    np.median([row["remove_probe_recovery"] for row in rows])
                ),
                "median_concept_haar_probe_recovery": float(
                    np.median([row["haar_probe_recovery"] for row in rows])
                ),
                "median_concept_residual_probe_norm_ratio": float(
                    np.median([row["residual_probe_norm_ratio"] for row in rows])
                ),
                "installed_probe_donor_nearest_concepts": sum(
                    row["nearest_endpoint"] == "donor" for row in rows
                ),
                "per_concept": rows,
            }
            summary["median_install_advantage_over_matched_haar"] = (
                summary["median_concept_install_probe_recovery"]
                - summary["median_concept_haar_probe_recovery"]
            )
            clauses = {
                "install_recovery": summary[
                    "median_concept_install_probe_recovery"
                ]
                >= float(gate["median_concept_install_probe_recovery_min"]),
                "remove_recovery": summary["median_concept_remove_probe_recovery"]
                >= float(gate["median_concept_remove_probe_recovery_min"]),
                "haar_advantage": summary[
                    "median_install_advantage_over_matched_haar"
                ]
                >= float(gate["median_install_advantage_over_matched_haar_min"]),
                "residual_closure": summary[
                    "median_concept_residual_probe_norm_ratio"
                ]
                <= float(gate["median_residual_probe_norm_ratio_max"]),
                "donor_identity": summary["installed_probe_donor_nearest_concepts"]
                >= int(gate["installed_probe_donor_nearest_concepts_min"]),
            }
            summary["gate_clauses"] = clauses
            summary["qualifies_in_direction"] = all(clauses.values())
            qualification[candidate][direction] = summary["qualifies_in_direction"]
            summaries.append(summary)
        by_direction[direction] = summaries
    qualifies = [
        candidate
        for candidate in contract["candidate_simplicity_order"]
        if all(qualification[candidate].values())
    ]
    selected = qualifies[0] if qualifies else None
    return {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-causal-closure",
        "direction_summaries": by_direction,
        "joint_candidates_qualifying_both_directions": qualifies,
        "selected_compact_joint_operator": selected,
        "selection_rule": gate["selection"],
        "disposition": (
            "compact_joint_operator_closes"
            if selected is not None
            else "no_tested_compact_joint_operator_closes"
        ),
    }


def summarize_geometry(
    contract: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    dose_concepts = _concept_cells(metrics["dose_rows"], ("concept", "direction"))
    jacobian_concepts = _concept_cells(
        [
            {key: value for key, value in row.items() if key != "candidate_predictions"}
            for row in metrics["jacobian_rows"]
        ],
        ("concept", "direction"),
    )
    directions = {}
    dose_gate = contract["downstream_classification"]["dose_linearity"]
    jac_gate = contract["downstream_classification"]["jacobian"]
    fixed_gate = contract["downstream_classification"]["fixed_norm_effect_survival"]
    for direction in contract["directions"]:
        dose = [row for row in dose_concepts if row["direction"] == direction]
        jac = [row for row in jacobian_concepts if row["direction"] == direction]
        summary = {
            "median_concept_free_dose_r_squared": float(
                np.median([row["free_r_squared"] for row in dose])
            ),
            "median_concept_free_max_in_range_secant_deviation_ratio": float(
                np.median(
                    [row["free_max_in_range_secant_deviation_ratio"] for row in dose]
                )
            ),
            "median_concept_fixed_norm_effect_survival_recovery": float(
                np.median(
                    [row["fixed_norm_effect_survival_recovery"] for row in dose]
                )
            ),
            "median_concept_fixed_norm_effect_survival_cosine": float(
                np.median([row["fixed_norm_effect_survival_cosine"] for row in dose])
            ),
            "median_concept_jacobian_exact_effect_recovery": float(
                np.median([row["exact_effect_recovery"] for row in jac])
            ),
            "median_concept_jacobian_exact_effect_cosine": float(
                np.median([row["exact_effect_cosine"] for row in jac])
            ),
            "median_jacobian_effective_rank_90": float(
                np.median([row["effective_rank_90"] for row in jac])
            ),
            "per_concept_dose": dose,
            "per_concept_jacobian": jac,
        }
        summary["classification_clauses"] = {
            "dose_r_squared": summary["median_concept_free_dose_r_squared"]
            >= float(dose_gate["median_concept_r_squared_min"]),
            "dose_secant_deviation": summary[
                "median_concept_free_max_in_range_secant_deviation_ratio"
            ]
            <= float(
                dose_gate["median_concept_max_in_range_secant_deviation_ratio_max"]
            ),
            "jacobian_recovery": summary[
                "median_concept_jacobian_exact_effect_recovery"
            ]
            >= float(jac_gate["median_concept_exact_effect_recovery_min"]),
            "jacobian_cosine": summary[
                "median_concept_jacobian_exact_effect_cosine"
            ]
            >= float(jac_gate["median_concept_exact_effect_cosine_min"]),
            "fixed_norm_linear_survival": summary[
                "median_concept_fixed_norm_effect_survival_recovery"
            ]
            >= float(fixed_gate["linear_translation_min_both_directions"]),
            "fixed_norm_mediated": summary[
                "median_concept_fixed_norm_effect_survival_recovery"
            ]
            <= float(fixed_gate["rmsnorm_mediated_max_both_directions"]),
        }
        directions[direction] = summary
    linear = all(
        all(
            summary["classification_clauses"][key]
            for key in (
                "dose_r_squared",
                "dose_secant_deviation",
                "jacobian_recovery",
                "jacobian_cosine",
                "fixed_norm_linear_survival",
            )
        )
        for summary in directions.values()
    )
    mediated = all(
        summary["classification_clauses"]["fixed_norm_mediated"]
        for summary in directions.values()
    )
    classification = (
        "locally_linear_additive_translation"
        if linear
        else "rmsnorm_mediated_geometry"
        if mediated
        else "mixed_or_nonlinear_state_dependent"
    )
    return {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-downstream-geometry",
        "direction_summaries": directions,
        "downstream_classification": classification,
    }


def summarize_attention(
    contract: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    jobs = _concept_cells(metrics["job_rows"], ("concept", "direction", "job"))
    masses = _concept_cells(
        metrics["attention_mass_rows"], ("concept", "direction", "region")
    )
    direction_summaries = {}
    diagnostic_jobs = [
        "routing_hybrid",
        "content_hybrid",
        *(
            f"atomic.{region}.install" for region in contract["atomic_source_regions"]
        ),
    ]
    for direction in contract["directions"]:
        job_summary = []
        for job in diagnostic_jobs:
            rows = [
                row
                for row in jobs
                if row["direction"] == direction and row["job"] == job
            ]
            job_summary.append(
                {
                    "job": job,
                    "median_concept_probe_recovery": float(
                        np.median([row["probe_recovery"] for row in rows])
                    ),
                    "median_concept_k12_recovery": float(
                        np.median([row["k12_recovery"] for row in rows])
                    ),
                }
            )
        mass_summary = []
        for region in contract["atomic_source_regions"]:
            rows = [
                row
                for row in masses
                if row["direction"] == direction and row["region"] == region
            ]
            mass_summary.append(
                {
                    "region": region,
                    "median_concept_mean_signed_mass_change": float(
                        np.median([row["mean_signed_mass_change"] for row in rows])
                    ),
                    "median_concept_mean_absolute_mass_change": float(
                        np.median([row["mean_absolute_mass_change"] for row in rows])
                    ),
                }
            )
        direction_summaries[direction] = {
            "hybrid_and_atomic_causal_recoveries": job_summary,
            "attention_mass_changes": mass_summary,
        }
    return {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-attention-algebra",
        "factorization": contract["attention_factorization"],
        "direction_summaries": direction_summaries,
    }


def reduce_once() -> tuple[dict[str, Any], ...]:
    contract = expanded_contract()
    metrics, inputs = load_metrics(contract)
    closure = summarize_closure(contract, metrics)
    geometry = summarize_geometry(contract, metrics)
    attention = summarize_attention(contract, metrics)
    day44 = read_json(DAY44_SUMMARY_PATH)
    day47 = read_json(DAY47_SUMMARY_PATH)
    selected = closure["selected_compact_joint_operator"]
    summary = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "evidence_class": contract["evidence_class"],
        "selected_compact_joint_operator": selected,
        "causal_closure_disposition": closure["disposition"],
        "downstream_classification": geometry["downstream_classification"],
        "attention_operation_classification": (
            selected
            if selected in {"routing_shapley", "content_shapley"}
            else f"source_localized:{selected}"
            if selected is not None
            else "mixed_distributed_routing_content_and_source_regions"
        ),
        "prior_model_selection": {
            "concept_position_prototype_offline_effect_rse": day44["candidates"][
                "concept_position_prototype"
            ]["calibrated_existing_direct_effect"]["equal_concept_rse"],
            "best_normal_state_conditioned_low_rank_effect_rse": min(
                value["calibrated_existing_direct_effect"]["equal_concept_rse"]
                for name, value in day44["candidates"].items()
                if name.startswith("normal_state_conditioned_low_rank")
            ),
            "heldout_concept_position_total_effect_recovery": day47["candidate"][
                "equal_concept_recovery"
            ],
        },
        "attention_summary_path": str(ATTENTION_PATH.relative_to(ROOT)),
        "geometry_summary_path": str(GEOMETRY_PATH.relative_to(ROOT)),
        "closure_summary_path": str(CLOSURE_PATH.relative_to(ROOT)),
        "bounded_conclusion": (
            "a frozen compact joint attention component closes the exact K12 effect"
            if selected is not None
            else "exact K12 is causal and translation-like only if the downstream gate passes; no tested compact routing/content or source-region component closes its full effect"
        ),
        "boundary": "development sandbox only; no fresh checkpoint, behavior, generation, or title upgrade",
    }
    gates = contract["implementation_gates"]
    checks = {
        "preflight_pass": inputs["preflight"]["result"] == "pass",
        "execution_complete": bool(inputs["execution"]["complete"]),
        "exact_state_rows": inputs["state_rows"] == int(gates["total_state_rows"]),
        "identity_k12": inputs["identity_k12_max_abs"]
        <= float(gates["identity_k12_max_abs"]),
        "identity_margins": inputs["identity_margin_max_abs"]
        <= float(gates["identity_monitor_margin_max_abs"]),
        "jacobian_identity_margins": inputs["jacobian_identity_margin_max_abs"]
        <= float(gates["identity_monitor_margin_max_abs"]),
        "attention_reconstruction": inputs["attention_reconstruction_max_abs"]
        <= float(gates["attention_reconstruction_max_abs"]),
        "shapley_closure": inputs["shapley_closure_max_abs"]
        <= float(gates["shapley_closure_max_abs"]),
        "random_audits_pass": bool(inputs["random_audits_pass"]),
        "hooks_removed": inputs["execution"]["hooks_after_execution"] == 0,
        "thirteen_unique_tensor_hashes": inputs["tensor_hash_count"]
        == int(gates["unique_tensor_hashes"]),
        "exact_probe_order": len(inputs["probe_name_sets"]) == 1
        and len(inputs["probe_name_sets"][0]) == 13,
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for family in (
                metrics["candidate_rows"],
                metrics["job_rows"],
                metrics["dose_rows"],
            )
            for row in family
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
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
            "state_rows": inputs["state_rows"],
            "identity_k12_max_abs": inputs["identity_k12_max_abs"],
            "identity_margin_max_abs": inputs["identity_margin_max_abs"],
            "jacobian_identity_margin_max_abs": inputs[
                "jacobian_identity_margin_max_abs"
            ],
            "attention_reconstruction_max_abs": inputs[
                "attention_reconstruction_max_abs"
            ],
            "shapley_closure_max_abs": inputs["shapley_closure_max_abs"],
            "tensor_hash_count": inputs["tensor_hash_count"],
        },
        "two_in_memory_reductions_byte_identical": None,
    }
    return summary, attention, geometry, closure, audit, metrics


def output_manifest() -> dict[str, Any]:
    paths = [
        CONTRACT_PATH,
        PREFLIGHT_PATH,
        EXECUTION_PATH,
        SUMMARY_PATH,
        ATTENTION_PATH,
        GEOMETRY_PATH,
        CLOSURE_PATH,
        AUDIT_PATH,
        METRICS_PATH,
        *sorted(SHARD_DIR.glob("*.json")),
        *sorted(SHARD_DIR.glob("*.safetensors")),
    ]
    return {
        "schema_version": 1,
        "procedure": "prospective-day56-joint-k12-mathematical-mechanism-v1-artifact-manifest",
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def main() -> None:
    first = reduce_once()
    second = reduce_once()
    if any(json_bytes(left) != json_bytes(right) for left, right in zip(first, second)):
        raise RuntimeError("Day 56 reductions are not byte-identical")
    summary, attention, geometry, closure, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    outputs = {
        SUMMARY_PATH: summary,
        ATTENTION_PATH: attention,
        GEOMETRY_PATH: geometry,
        CLOSURE_PATH: closure,
        AUDIT_PATH: audit,
        METRICS_PATH: metrics,
    }
    for path, payload in outputs.items():
        write_json(path, payload)
    write_json(MANIFEST_PATH, output_manifest())


if __name__ == "__main__":
    main()
