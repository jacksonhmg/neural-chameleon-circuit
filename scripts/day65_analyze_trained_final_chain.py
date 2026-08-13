#!/usr/bin/env python3
"""Mechanically reduce the trained-concept content-untouched final title gate."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.final_title_gate import (  # noqa: E402
    recursive_numeric_max_difference,
    title_gate_disposition,
    vector_relation,
)


CONTRACT_PATH = ROOT / "results/day-64/frozen-trained-final-contract.json"
PROTOTYPE_PATH = ROOT / "artifacts/trained-final-title-gate-v1/development-prototypes.safetensors"
SHARD_DIR = ROOT / "artifacts/trained-final-title-gate-v1/final-chain"
SUMMARY_PATH = ROOT / "results/day-65/trained-final-chain-summary.json"
VERIFICATION_PATH = ROOT / "results/day-65/trained-final-chain-rereduction-verification.json"
MANIFEST_PATH = ROOT / "results/day-65/trained-final-chain-artifact-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def median(values: Sequence[float]) -> float:
    if not values or not all(np.isfinite(value) for value in values):
        raise RuntimeError("empty or nonfinite final metric")
    return float(statistics.median(values))


def relations(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    rows = [vector_relation(value[index], reference[index]) for index in range(len(value))]
    return {
        key: median([row[key] for row in rows])
        for key in ("aligned_recovery", "cosine", "norm_ratio", "residual_norm_ratio")
    }


def concat_norm_ratio(numerator: Sequence[torch.Tensor], denominator: Sequence[torch.Tensor]) -> float:
    top = torch.cat([value.double().reshape(-1) for value in numerator])
    bottom = torch.cat([value.double().reshape(-1) for value in denominator])
    norm = float(torch.linalg.vector_norm(bottom))
    if norm <= 1e-12:
        raise RuntimeError("zero Chameleon acquisition denominator")
    return float(torch.linalg.vector_norm(top)) / norm


def checked_concept(model: str, concept: str, contract_hash: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensor_path, metadata_path = SHARD_DIR / model / f"{concept}.safetensors", SHARD_DIR / model / f"{concept}.json"
    metadata = read_json(metadata_path)
    if (
        metadata["contract_sha256"] != contract_hash
        or metadata["tensor_sha256"] != sha256_file(tensor_path)
        or metadata["model"] != model
        or metadata["concept"] != concept
        or metadata["patched_component_count"] != 12
        or metadata["source_operator"] != "full_monitoring_prefix_kv"
    ):
        raise RuntimeError(f"final shard provenance differs: {model}/{concept}")
    return load_file(tensor_path), metadata


def checked_negative(contract_hash: str) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path, metadata_path = SHARD_DIR / "chameleon/final-negative.safetensors", SHARD_DIR / "chameleon/final-negative.json"
    metadata = read_json(metadata_path)
    if metadata["contract_sha256"] != contract_hash or metadata["tensor_sha256"] != sha256_file(tensor_path):
        raise RuntimeError("final-negative provenance differs")
    return load_file(tensor_path)["normal.margins"].float(), metadata


def direction_metrics(
    tensors: Mapping[str, torch.Tensor],
    prototype: Mapping[str, torch.Tensor],
    concept: str,
    direction: str,
    target: str,
    donor: str,
    own_probe_index: int,
) -> dict[str, Any]:
    natural = {
        field: tensors[f"natural.{donor}.{field}"].float() - tensors[f"natural.{target}.{field}"].float()
        for field in ("kv", "k12", "margins")
    }
    prediction = {
        field: vector_relation(
            natural[field].mean(0), prototype[f"{concept}.{direction}.{field}"].float()
        )
        for field in natural
    }
    identity_target = {
        field: tensors[f"{direction}.identity_target.{field}"].float()
        for field in ("k12", "margins")
    }
    donor_kv = {
        field: tensors[f"{direction}.donor_kv_into_target.{field}"].float()
        for field in ("k12", "margins")
    }
    different = tensors[f"{direction}.different_kv_into_target.margins"].float()
    orthogonal = tensors[f"{direction}.matched_orthogonal_k12_into_target.margins"].float()
    operation = {
        "kv_to_natural_k12": relations(donor_kv["k12"] - identity_target["k12"], natural["k12"]),
        "kv_to_natural_probe": relations(donor_kv["margins"] - identity_target["margins"], natural["margins"]),
        "different_control_probe": relations(different - identity_target["margins"], natural["margins"]),
        "orthogonal_control_probe": relations(orthogonal - identity_target["margins"], natural["margins"]),
    }
    operation["advantage_over_different"] = (
        operation["kv_to_natural_probe"]["aligned_recovery"]
        - operation["different_control_probe"]["aligned_recovery"]
    )
    operation["advantage_over_orthogonal"] = (
        operation["kv_to_natural_probe"]["aligned_recovery"]
        - operation["orthogonal_control_probe"]["aligned_recovery"]
    )
    identity_donor = tensors[f"{direction}.identity_donor.margins"].float()
    target_k12 = tensors[f"{direction}.target_k12_into_donor.margins"].float()
    necessity = relations(target_k12 - identity_donor, -natural["margins"])
    target_kv = tensors[f"{direction}.target_kv_into_donor.margins"].float()
    restored = tensors[f"{direction}.target_kv_into_donor_plus_donor_k12_restore.margins"].float()
    lost = identity_donor - target_kv
    restoration = relations(restored - target_kv, lost)
    restoration["identity_endpoint_max_abs"] = float((restored - identity_donor).abs().max())
    own_delta = float(natural["margins"][:, own_probe_index].median())
    expected_sign = own_delta < 0.0 if direction == "correct_to_irrelevant" else own_delta > 0.0
    return {
        "target": target,
        "donor": donor,
        "prediction": prediction,
        "own_probe_natural_delta_median": own_delta,
        "expected_own_probe_sign": expected_sign,
        "operation": operation,
        "necessity_sufficiency": necessity,
        "restoration": restoration,
    }


def aggregate_mechanism(
    contract: Mapping[str, Any], directional: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, bool]]:
    per_concept: dict[str, Any] = {}
    for concept, directions in directional.items():
        rows = list(directions.values())
        per_concept[concept] = {
            "prediction_cosine": {
                field: median([row["prediction"][field]["cosine"] for row in rows])
                for field in ("kv", "k12", "margins")
            },
            "expected_sign_both_directions": all(row["expected_own_probe_sign"] for row in rows),
            "operation_k12_recovery": median([row["operation"]["kv_to_natural_k12"]["aligned_recovery"] for row in rows]),
            "operation_probe_recovery": median([row["operation"]["kv_to_natural_probe"]["aligned_recovery"] for row in rows]),
            "operation_advantage_over_different": median([row["operation"]["advantage_over_different"] for row in rows]),
            "operation_advantage_over_orthogonal": median([row["operation"]["advantage_over_orthogonal"] for row in rows]),
            "k12_reversal": median([row["necessity_sufficiency"]["aligned_recovery"] for row in rows]),
            "parallel_tail_residual": median([row["necessity_sufficiency"]["residual_norm_ratio"] for row in rows]),
            "restoration_recovery": median([row["restoration"]["aligned_recovery"] for row in rows]),
            "restoration_residual": median([row["restoration"]["residual_norm_ratio"] for row in rows]),
        }
    values = list(per_concept.values())
    aggregate = {
        "prediction_median_concept_cosine": {
            field: median([row["prediction_cosine"][field] for row in values])
            for field in ("kv", "k12", "margins")
        },
        "prediction_expected_sign_concepts": sum(row["expected_sign_both_directions"] for row in values),
        "operation_median_concept_k12_recovery": median([row["operation_k12_recovery"] for row in values]),
        "operation_median_concept_probe_recovery": median([row["operation_probe_recovery"] for row in values]),
        "operation_median_concept_advantage_over_different": median([row["operation_advantage_over_different"] for row in values]),
        "operation_median_concept_advantage_over_orthogonal": median([row["operation_advantage_over_orthogonal"] for row in values]),
        "operation_positive_probe_recovery_concepts": sum(row["operation_probe_recovery"] > 0 for row in values),
        "necessity_median_concept_reversal": median([row["k12_reversal"] for row in values]),
        "necessity_median_concept_parallel_tail_residual": median([row["parallel_tail_residual"] for row in values]),
        "necessity_positive_reversal_concepts": sum(row["k12_reversal"] > 0 for row in values),
        "restoration_median_concept_recovery": median([row["restoration_recovery"] for row in values]),
        "restoration_median_concept_residual": median([row["restoration_residual"] for row in values]),
        "restoration_positive_concepts": sum(row["restoration_recovery"] > 0 for row in values),
    }
    prediction_gate = contract["gates"]["prediction"]
    operation_gate = contract["gates"]["operation"]
    necessity_gate = contract["gates"]["necessity_sufficiency"]
    restoration_gate = contract["gates"]["restoration"]
    passes = {
        "semantic_conditioning": all(
            aggregate["prediction_median_concept_cosine"][field]
            >= float(prediction_gate["median_concept_cosine_min"][field])
            for field in ("kv", "k12", "margins")
        ) and aggregate["prediction_expected_sign_concepts"] >= int(prediction_gate["concepts_with_expected_own_probe_sign_min"]),
        "operation": (
            aggregate["operation_median_concept_k12_recovery"] >= float(operation_gate["median_concept_kv_to_k12_aligned_recovery_min"])
            and aggregate["operation_median_concept_probe_recovery"] >= float(operation_gate["median_concept_kv_to_probe_aligned_recovery_min"])
            and aggregate["operation_median_concept_advantage_over_different"] >= float(operation_gate["median_concept_advantage_over_each_control_min"])
            and aggregate["operation_median_concept_advantage_over_orthogonal"] >= float(operation_gate["median_concept_advantage_over_each_control_min"])
            and aggregate["operation_positive_probe_recovery_concepts"] >= int(operation_gate["concepts_with_positive_probe_recovery_min"])
        ),
        "necessity_sufficiency": (
            aggregate["necessity_median_concept_reversal"] >= float(necessity_gate["median_concept_exact_k12_reversal_min"])
            and aggregate["necessity_median_concept_parallel_tail_residual"] <= float(necessity_gate["median_concept_parallel_tail_residual_norm_ratio_max"])
            and aggregate["necessity_positive_reversal_concepts"] >= int(necessity_gate["concepts_with_positive_reversal_min"])
        ),
        "restoration": (
            aggregate["restoration_median_concept_recovery"] >= float(restoration_gate["median_concept_lost_probe_effect_recovery_min"])
            and aggregate["restoration_median_concept_residual"] <= float(restoration_gate["median_concept_restored_residual_norm_ratio_max"])
            and aggregate["restoration_positive_concepts"] >= int(restoration_gate["concepts_with_positive_restoration_min"])
        ),
    }
    return {"per_concept": per_concept, "aggregate": aggregate}, passes


def concept_bootstrap(values: Sequence[float], draws: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    estimates = array[indices].mean(1)
    return {
        "point": float(array.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def operational_metrics(
    contract: Mapping[str, Any],
    chameleon: Mapping[str, Mapping[str, torch.Tensor]],
    negative: torch.Tensor,
    negative_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], bool]:
    names = contract["probe_names_in_order"]
    assignments = list(zip(
        negative_metadata["assignment_probe_concepts"],
        negative_metadata["assignment_unique_indices"],
    ))
    per_concept: dict[str, Any] = {}
    differences: dict[str, dict[str, list[float]]] = {
        nominal: {effect: [] for effect in ("drop", "clamp", "restoration")}
        for nominal in ("0.01", "0.05")
    }
    for concept in contract["concepts_in_order"]:
        tensors = chameleon[concept]
        probe_index = names.index(concept)
        margins = {
            "correct": tensors["natural.correct.margins"][:, probe_index].float(),
            "irrelevant": tensors["natural.irrelevant.margins"][:, probe_index].float(),
            "clamp": tensors["correct_to_irrelevant.target_k12_into_donor.margins"][:, probe_index].float(),
            "target_kv": tensors["correct_to_irrelevant.target_kv_into_donor.margins"][:, probe_index].float(),
            "restored": tensors["correct_to_irrelevant.target_kv_into_donor_plus_donor_k12_restore.margins"][:, probe_index].float(),
        }
        negative_indices = [int(index) for assigned, index in assignments if assigned == concept]
        negative_values = negative[negative_indices, probe_index]
        points: dict[str, Any] = {}
        for nominal in ("0.01", "0.05"):
            threshold = float(contract["calibration_thresholds"][concept][nominal])
            rates = {name: float((values > threshold).float().mean()) for name, values in margins.items()}
            effects = {
                "drop": rates["irrelevant"] - rates["correct"],
                "clamp": rates["clamp"] - rates["correct"],
                "restoration": rates["target_kv"] - rates["restored"],
            }
            for effect, value in effects.items():
                differences[nominal][effect].append(value)
            points[nominal] = {
                "threshold": threshold,
                "realized_final_FPR": float((negative_values > threshold).float().mean()),
                "TPR": rates,
                "differences": effects,
            }
        per_concept[concept] = {
            "n_positive": len(margins["correct"]),
            "n_negative": len(negative_values),
            "raw_margin": {
                name: {"mean": float(values.mean()), "median": float(values.median())}
                for name, values in margins.items()
            },
            "operating_points": points,
        }
    draws, seed = int(contract["reductions"]["uncertainty"]["draws"]), int(contract["reductions"]["uncertainty"]["seed"])
    aggregate = {
        nominal: {
            effect: {
                **concept_bootstrap(values, draws, seed + point_index * 100 + effect_index),
                "positive_concepts": sum(value > 0 for value in values),
            }
            for effect_index, (effect, values) in enumerate(differences[nominal].items())
        }
        for point_index, nominal in enumerate(("0.01", "0.05"))
    }
    gate = contract["gates"]["operational"]
    positive_min = int(gate["concepts_positive_each_effect_min"])
    passed = (
        aggregate["0.01"]["drop"]["point"] >= float(gate["macro_TPR_drop_correct_vs_irrelevant_at_1pct_FPR_min"])
        and aggregate["0.05"]["drop"]["point"] >= float(gate["macro_TPR_drop_correct_vs_irrelevant_at_5pct_FPR_min"])
        and max(aggregate[nominal]["clamp"]["point"] for nominal in aggregate) >= float(gate["macro_TPR_clamp_reversal_at_either_threshold_min"])
        and max(aggregate[nominal]["restoration"]["point"] for nominal in aggregate) >= float(gate["macro_TPR_restoration_at_either_threshold_min"])
        and min(aggregate[nominal]["drop"]["positive_concepts"] for nominal in aggregate) >= positive_min
        and max(aggregate[nominal]["clamp"]["positive_concepts"] for nominal in aggregate) >= positive_min
        and max(aggregate[nominal]["restoration"]["positive_concepts"] for nominal in aggregate) >= positive_min
    )
    return {"per_concept": per_concept, "aggregate": aggregate}, passed


def reduce_all() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    if contract["prototype"]["sha256"] != sha256_file(PROTOTYPE_PATH):
        raise RuntimeError("development prototypes differ")
    prototype = load_file(PROTOTYPE_PATH)
    data = {
        model: {
            concept: checked_concept(model, concept, contract_hash)
            for concept in contract["concepts_in_order"]
        }
        for model in ("chameleon", "exact_precursor")
    }
    directional = {}
    audits = {}
    names = contract["probe_names_in_order"]
    for concept in contract["concepts_in_order"]:
        tensors, metadata = data["chameleon"][concept]
        directional[concept] = {
            direction: direction_metrics(
                tensors, prototype, concept, direction, target, donor, names.index(concept)
            )
            for direction, target, donor in (
                ("correct_to_irrelevant", "irrelevant", "correct"),
                ("irrelevant_to_correct", "correct", "irrelevant"),
            )
        }
        for model in data:
            model_tensors, model_metadata = data[model][concept]
            margin_errors, k12_errors = [], []
            for direction, target, donor in (
                ("correct_to_irrelevant", "irrelevant", "correct"),
                ("irrelevant_to_correct", "correct", "irrelevant"),
            ):
                for role, natural in (("identity_target", target), ("identity_donor", donor)):
                    margin_errors.append(float((model_tensors[f"{direction}.{role}.margins"] - model_tensors[f"natural.{natural}.margins"]).abs().max()))
                    k12_errors.append(float((model_tensors[f"{direction}.{role}.k12"] - model_tensors[f"natural.{natural}.k12"]).abs().max()))
            audits[f"{model}.{concept}"] = {
                "margin_max_abs": max(margin_errors),
                "k12_max_abs": max(k12_errors),
                "orthogonal_controls_pass": all(
                    row["orthogonal"][direction]["pass"]
                    for row in model_metadata["orthogonal_audits"]
                    for direction in ("correct_to_irrelevant", "irrelevant_to_correct")
                ),
                "natural_tail_declared": model_metadata["tail_regeneration"] == "natural_except_exact_declared_k12_raw_head_sites",
            }
    mechanism, mechanism_passes = aggregate_mechanism(contract, directional)
    acquisition_vectors = {
        model: {field: [] for field in ("natural_probe", "natural_k12", "kv_probe")}
        for model in data
    }
    for model, concepts in data.items():
        for concept, (tensors, _metadata) in concepts.items():
            for direction, target, donor in (
                ("correct_to_irrelevant", "irrelevant", "correct"),
                ("irrelevant_to_correct", "correct", "irrelevant"),
            ):
                acquisition_vectors[model]["natural_probe"].append(tensors[f"natural.{donor}.margins"] - tensors[f"natural.{target}.margins"])
                acquisition_vectors[model]["natural_k12"].append(tensors[f"natural.{donor}.k12"] - tensors[f"natural.{target}.k12"])
                acquisition_vectors[model]["kv_probe"].append(tensors[f"{direction}.donor_kv_into_target.margins"] - tensors[f"{direction}.identity_target.margins"])
    acquisition_gate = contract["gates"]["acquisition"]
    acquisition = {}
    for field, gate_name in (
        ("natural_probe", "precursor_to_chameleon_natural_probe_norm_ratio_max"),
        ("natural_k12", "precursor_to_chameleon_natural_k12_norm_ratio_max"),
        ("kv_probe", "precursor_to_chameleon_kv_mediated_probe_norm_ratio_max"),
    ):
        ratio = concat_norm_ratio(acquisition_vectors["exact_precursor"][field], acquisition_vectors["chameleon"][field])
        maximum = float(acquisition_gate[gate_name])
        acquisition[field] = {"precursor_to_chameleon_norm_ratio": ratio, "maximum": maximum, "pass": ratio <= maximum}
    negative, negative_metadata = checked_negative(contract_hash)
    operational, operational_pass = operational_metrics(
        contract, {concept: values[0] for concept, values in data["chameleon"].items()}, negative, negative_metadata
    )
    implementation_pass = all(
        row["margin_max_abs"] <= float(contract["implementation_gates"]["same_state_margin_max_abs"])
        and row["k12_max_abs"] <= float(contract["implementation_gates"]["same_state_k12_max_abs"])
        and row["orthogonal_controls_pass"]
        and row["natural_tail_declared"]
        for row in audits.values()
    )
    clauses = {
        "acquisition": all(row["pass"] for row in acquisition.values()),
        "operation": mechanism_passes["operation"],
        "semantic_conditioning": mechanism_passes["semantic_conditioning"],
        "necessity_sufficiency": mechanism_passes["necessity_sufficiency"],
        "endogenous_chain": all(mechanism_passes.values()) and implementation_pass,
        "restoration": mechanism_passes["restoration"],
        "operational_failure": operational_pass,
    }
    return {
        "schema_version": 1,
        "procedure": "day65-trained-content-untouched-final-reduction-v1",
        "contract_sha256": contract_hash,
        "claim_scope": contract["claim_scope"],
        "sample_sizes": contract["roles"],
        "directional_chameleon_results": directional,
        "mechanism_aggregate": mechanism,
        "mechanism_passes": mechanism_passes,
        "acquisition": acquisition,
        "operational_consequences": operational,
        "implementation_audits": audits,
        "implementation_gate_pass": implementation_pass,
        "title_clauses": clauses,
        "title_disposition": title_gate_disposition(clauses),
        "interpretation_policy": "conjunctive_no_compensatory_score",
    }


def main() -> None:
    first, second = reduce_all(), reduce_all()
    absolute, relative = recursive_numeric_max_difference(first, second)
    verification = {
        "schema_version": 1,
        "procedure": "day65-independent-local-rereduction-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "maximum_absolute_numeric_difference": absolute,
        "maximum_relative_numeric_difference": relative,
        "exact_reduction_match": absolute == 0.0 and relative == 0.0,
    }
    if not verification["exact_reduction_match"]:
        raise RuntimeError("independent reductions differ")
    write_json(SUMMARY_PATH, first)
    write_json(VERIFICATION_PATH, verification)
    artifacts = [path for path in sorted(SHARD_DIR.rglob("*")) if path.is_file()] + [SUMMARY_PATH, VERIFICATION_PATH]
    write_json(MANIFEST_PATH, {
        "schema_version": 1,
        "procedure": "day65-trained-final-chain-artifact-manifest-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts],
    })
    print(json.dumps({
        "title_disposition": first["title_disposition"],
        "title_clauses": first["title_clauses"],
        "mechanism_passes": first["mechanism_passes"],
        "implementation_gate_pass": first["implementation_gate_pass"],
    }, indent=2))


if __name__ == "__main__":
    main()
