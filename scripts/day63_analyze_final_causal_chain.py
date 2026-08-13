#!/usr/bin/env python3
"""Mechanically reduce the frozen untouched final causal-chain experiment."""

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
    operating_rate,
    recursive_numeric_max_difference,
    title_gate_disposition,
    vector_relation,
)


CONTRACT_PATH = ROOT / "results/day-63/frozen-final-causal-chain-contract.json"
PROTOTYPE_PATH = ROOT / "artifacts/final-title-gate-v1/qualification-prototypes.safetensors"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/final-causal-chain-shards"
SUMMARY_PATH = ROOT / "results/day-63/final-causal-chain-summary.json"
VERIFICATION_PATH = ROOT / "results/day-63/final-causal-chain-rereduction-verification.json"
MANIFEST_PATH = ROOT / "results/day-63/final-causal-chain-artifact-manifest.json"


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
        raise RuntimeError("a final reduction vector is empty or nonfinite")
    return float(statistics.median(values))


def relations(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    rows = [vector_relation(value[index], reference[index]) for index in range(value.shape[0])]
    return {
        key: median([row[key] for row in rows])
        for key in ("aligned_recovery", "cosine", "norm_ratio", "residual_norm_ratio")
    }


def norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    bottom = float(torch.linalg.vector_norm(denominator.double().reshape(-1)))
    if bottom <= 1e-12:
        raise RuntimeError("Chameleon acquisition reference has zero norm")
    return float(torch.linalg.vector_norm(numerator.double().reshape(-1))) / bottom


def margin_summary(values: torch.Tensor, names: Sequence[str]) -> dict[str, Any]:
    return {
        name: {
            "mean": float(values[:, index].mean()),
            "median": float(values[:, index].median()),
            "minimum": float(values[:, index].min()),
            "maximum": float(values[:, index].max()),
        }
        for index, name in enumerate(names)
    }


def bootstrap_difference(
    left: Sequence[float], right: Sequence[float], *, draws: int, seed: int
) -> dict[str, float]:
    left_array, right_array = np.asarray(left), np.asarray(right)
    if left_array.shape != right_array.shape or left_array.ndim != 1:
        raise ValueError("paired bootstrap vectors differ")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left_array), size=(draws, len(left_array)))
    differences = (left_array[indices] - right_array[indices]).mean(axis=1)
    return {
        "point": float((left_array - right_array).mean()),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
    }


def checked_pair(
    model: str, pair_id: str, contract_hash: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensor_path = SHARD_DIR / model / f"{pair_id}.safetensors"
    metadata_path = SHARD_DIR / model / f"{pair_id}.json"
    metadata = read_json(metadata_path)
    if (
        metadata["contract_sha256"] != contract_hash
        or metadata["tensor_sha256"] != sha256_file(tensor_path)
        or metadata["model"] != model
        or metadata["pair_id"] != pair_id
        or metadata["patched_component_count"] != 12
    ):
        raise RuntimeError(f"final causal shard provenance differs: {model}/{pair_id}")
    return load_file(tensor_path), metadata


def checked_negative(
    concept: str, contract_hash: str
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path = SHARD_DIR / "chameleon" / f"negative-{concept}.safetensors"
    metadata_path = SHARD_DIR / "chameleon" / f"negative-{concept}.json"
    metadata = read_json(metadata_path)
    if (
        metadata["contract_sha256"] != contract_hash
        or metadata["tensor_sha256"] != sha256_file(tensor_path)
        or metadata["concept"] != concept
    ):
        raise RuntimeError(f"final negative shard provenance differs: {concept}")
    return load_file(tensor_path)["normal.margins"].float(), metadata


def direction_metrics(
    tensors: Mapping[str, torch.Tensor],
    prototypes: Mapping[str, torch.Tensor],
    pair_id: str,
    direction: str,
    target: str,
    donor: str,
    probe_names: Sequence[str],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    natural = {
        field: tensors[f"natural.{donor}.{field}"].float()
        - tensors[f"natural.{target}.{field}"].float()
        for field in ("kv", "k12", "margins")
    }
    prototype_direction = f"{target}_to_{donor}"
    prediction = {
        field: vector_relation(
            natural[field].mean(dim=0),
            prototypes[f"{pair_id}.{prototype_direction}.{field}"].float(),
        )
        for field in ("kv", "k12", "margins")
    }
    target_index, donor_index = probe_names.index(target), probe_names.index(donor)
    sign_values = {
        "target_probe_median_delta": float(natural["margins"][:, target_index].median()),
        "donor_probe_median_delta": float(natural["margins"][:, donor_index].median()),
    }
    sign_values["pass"] = (
        sign_values["target_probe_median_delta"] > 0
        and sign_values["donor_probe_median_delta"] < 0
    )

    identity_target = {
        field: tensors[f"{direction}.identity_target.{field}"].float()
        for field in ("margins", "k12")
    }
    donor_kv = {
        field: tensors[f"{direction}.donor_kv_into_target.{field}"].float()
        for field in ("margins", "k12")
    }
    irrelevant = tensors[f"{direction}.irrelevant_kv_into_target.margins"].float()
    orthogonal = tensors[
        f"{direction}.matched_orthogonal_k12_into_target.margins"
    ].float()
    operation = {
        "kv_to_natural_k12": relations(
            donor_kv["k12"] - identity_target["k12"], natural["k12"]
        ),
        "kv_to_natural_probe": relations(
            donor_kv["margins"] - identity_target["margins"], natural["margins"]
        ),
        "irrelevant_probe": relations(
            irrelevant - identity_target["margins"], natural["margins"]
        ),
        "orthogonal_probe": relations(
            orthogonal - identity_target["margins"], natural["margins"]
        ),
    }
    operation["kv_advantage_over_irrelevant_probe"] = (
        operation["kv_to_natural_probe"]["aligned_recovery"]
        - operation["irrelevant_probe"]["aligned_recovery"]
    )
    operation["kv_advantage_over_orthogonal_probe"] = (
        operation["kv_to_natural_probe"]["aligned_recovery"]
        - operation["orthogonal_probe"]["aligned_recovery"]
    )

    identity_donor = tensors[f"{direction}.identity_donor.margins"].float()
    target_k12 = tensors[f"{direction}.target_k12_into_donor.margins"].float()
    reversal_reference = -natural["margins"]
    necessity = relations(target_k12 - identity_donor, reversal_reference)

    target_kv = tensors[f"{direction}.target_kv_into_donor.margins"].float()
    restored = tensors[
        f"{direction}.target_kv_into_donor_plus_donor_k12_restore.margins"
    ].float()
    lost = identity_donor - target_kv
    restoration = relations(restored - target_kv, lost)
    restoration["identity_endpoint_max_abs"] = float(
        (restored - identity_donor).abs().max()
    )
    restored_k12 = tensors[
        f"{direction}.target_kv_into_donor_plus_donor_k12_restore.k12"
    ].float()
    identity_donor_k12 = tensors[f"{direction}.identity_donor.k12"].float()
    restoration["k12_identity_endpoint_max_abs"] = float(
        (restored_k12 - identity_donor_k12).abs().max()
    )

    prediction_gate = gates["prediction_both_directions"]
    operation_gate = gates["operation_both_directions"]
    necessity_gate = gates["necessity_sufficiency_both_directions"]
    restoration_gate = gates["restoration_both_directions"]
    passes = {
        "prediction": (
            prediction["kv"]["cosine"]
            >= float(prediction_gate["natural_kv_prototype_cosine_min"])
            and prediction["k12"]["cosine"]
            >= float(prediction_gate["natural_k12_prototype_cosine_min"])
            and prediction["margins"]["cosine"]
            >= float(
                prediction_gate["natural_complete_probe_prototype_cosine_min"]
            )
            and sign_values["pass"]
        ),
        "operation": (
            operation["kv_to_natural_k12"]["aligned_recovery"]
            >= float(operation_gate["kv_to_natural_k12_recovery_min"])
            and operation["kv_to_natural_probe"]["aligned_recovery"]
            >= float(operation_gate["kv_to_natural_probe_recovery_min"])
            and operation["kv_advantage_over_irrelevant_probe"]
            >= float(operation_gate["kv_advantage_over_irrelevant_probe_min"])
            and operation["kv_advantage_over_orthogonal_probe"]
            >= float(operation_gate["kv_advantage_over_orthogonal_probe_min"])
        ),
        "necessity_sufficiency": (
            necessity["aligned_recovery"]
            >= float(necessity_gate["target_k12_clamp_reversal_min"])
            and necessity["residual_norm_ratio"]
            <= float(necessity_gate["parallel_tail_residual_norm_ratio_max"])
        ),
        "restoration": (
            restoration["aligned_recovery"]
            >= float(restoration_gate["donor_k12_restore_lost_probe_effect_min"])
            and restoration["residual_norm_ratio"]
            <= float(restoration_gate["restored_endpoint_residual_norm_ratio_max"])
        ),
    }
    passes["integrated_chain"] = all(passes.values())
    return {
        "target": target,
        "donor": donor,
        "prediction": prediction,
        "expected_signs": sign_values,
        "operation": operation,
        "k12_clamp": necessity,
        "restoration": restoration,
        "passes": passes,
    }


def operational_metrics(
    contract: Mapping[str, Any],
    pair_tensors: Mapping[str, Mapping[str, torch.Tensor]],
    negative_tensors: Mapping[str, torch.Tensor],
    probe_names: Sequence[str],
) -> tuple[dict[str, Any], bool]:
    spec = contract["gates"]["operational"]
    bootstrap = contract["reductions"]["uncertainty"]
    results: dict[str, Any] = {}
    passes = []
    for concept_index, concept in enumerate(contract["selected_concepts_in_order"]):
        pair = next(
            pair
            for pair in contract["selected_pair_specs"]
            if concept in (pair["concept_a"], pair["concept_b"])
        )
        pair_id, a, b = pair["pair_id"], pair["concept_a"], pair["concept_b"]
        other = b if concept == a else a
        direction = "b_to_a" if concept == a else "a_to_b"
        tensors = pair_tensors[pair_id]
        probe_index = probe_names.index(concept)
        margins = {
            "correct_trigger": tensors[f"natural.{concept}.margins"][:, probe_index].tolist(),
            "pair_trigger": tensors[f"natural.{other}.margins"][:, probe_index].tolist(),
            "k12_clamp": tensors[
                f"{direction}.target_k12_into_donor.margins"
            ][:, probe_index].tolist(),
            "target_kv_into_donor": tensors[
                f"{direction}.target_kv_into_donor.margins"
            ][:, probe_index].tolist(),
            "restored_donor_k12": tensors[
                f"{direction}.target_kv_into_donor_plus_donor_k12_restore.margins"
            ][:, probe_index].tolist(),
        }
        concept_results: dict[str, Any] = {
            "pair_id": pair_id,
            "pair_trigger": other,
            "n_final_positive": len(margins["correct_trigger"]),
            "n_final_negative": int(negative_tensors[concept].shape[0]),
            "raw_margin": {
                name: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                }
                for name, values in margins.items()
            },
            "operating_points": {},
        }
        operating_passes = {}
        for point_index, nominal in enumerate(("0.01", "0.05")):
            threshold = float(contract["calibration_thresholds"][concept][nominal])
            negative = negative_tensors[concept][:, probe_index].tolist()
            flags = {
                name: [float(value) > threshold for value in values]
                for name, values in margins.items()
            }
            rates = {
                name: operating_rate(values, threshold) for name, values in margins.items()
            }
            drop = rates["pair_trigger"] - rates["correct_trigger"]
            reversal = rates["k12_clamp"] - rates["correct_trigger"]
            restoration = rates["target_kv_into_donor"] - rates["restored_donor_k12"]
            seed = int(bootstrap["seed"]) + concept_index * 100 + point_index * 10
            concept_results["operating_points"][nominal] = {
                "threshold": threshold,
                "realized_final_FPR": operating_rate(negative, threshold),
                "TPR": rates,
                "pair_minus_correct_TPR": bootstrap_difference(
                    flags["pair_trigger"],
                    flags["correct_trigger"],
                    draws=int(bootstrap["draws"]),
                    seed=seed,
                ),
                "clamp_minus_correct_TPR": bootstrap_difference(
                    flags["k12_clamp"],
                    flags["correct_trigger"],
                    draws=int(bootstrap["draws"]),
                    seed=seed + 1,
                ),
                "target_kv_minus_restored_TPR": bootstrap_difference(
                    flags["target_kv_into_donor"],
                    flags["restored_donor_k12"],
                    draws=int(bootstrap["draws"]),
                    seed=seed + 2,
                ),
                "point_differences": {
                    "pair_minus_correct": drop,
                    "clamp_minus_correct": reversal,
                    "target_kv_minus_restored": restoration,
                },
            }
            operating_passes[nominal] = {
                "correct_trigger_drop": drop
                >= float(spec[f"correct_trigger_TPR_drop_vs_pair_trigger_at_{1 if nominal == '0.01' else 5}pct_FPR_min"]),
                "clamp_reversal": reversal
                >= float(spec["k12_clamp_TPR_reversal_at_either_operating_point_min"]),
                "restoration_recovery": restoration
                >= float(spec["restoration_TPR_recovery_at_either_operating_point_min"]),
            }
        concept_pass = (
            operating_passes["0.01"]["correct_trigger_drop"]
            and operating_passes["0.05"]["correct_trigger_drop"]
            and any(row["clamp_reversal"] for row in operating_passes.values())
            and any(row["restoration_recovery"] for row in operating_passes.values())
        )
        concept_results["passes"] = {
            "by_operating_point": operating_passes,
            "concept_operational_gate": concept_pass,
        }
        results[concept] = concept_results
        passes.append(concept_pass)
    return results, all(passes)


def reduce_all() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    if contract["predictions"]["prototype_sha256"] != sha256_file(PROTOTYPE_PATH):
        raise RuntimeError("qualification prototypes differ")
    prototypes = load_file(PROTOTYPE_PATH)
    probe_names = contract["probe_names_in_order"]
    pair_data: dict[str, dict[str, tuple[dict[str, torch.Tensor], dict[str, Any]]]] = {}
    raw_margins: dict[str, Any] = {}
    identity_audits: dict[str, Any] = {}
    for pair in contract["selected_pair_specs"]:
        pair_id = pair["pair_id"]
        pair_data[pair_id] = {
            model: checked_pair(model, pair_id, contract_hash)
            for model in ("chameleon", "exact_precursor")
        }
        raw_margins[pair_id] = {}
        for model, (tensors, metadata) in pair_data[pair_id].items():
            raw_margins[pair_id][model] = {
                key.removesuffix(".margins"): margin_summary(value.float(), probe_names)
                for key, value in tensors.items()
                if key.endswith(".margins")
            }
            margin_errors, k12_errors = [], []
            for direction, target, donor in (
                ("a_to_b", pair["concept_a"], pair["concept_b"]),
                ("b_to_a", pair["concept_b"], pair["concept_a"]),
            ):
                margin_errors.extend(
                    [
                        float(
                            (
                                tensors[f"{direction}.identity_target.margins"]
                                - tensors[f"natural.{target}.margins"]
                            ).abs().max()
                        ),
                        float(
                            (
                                tensors[f"{direction}.identity_donor.margins"]
                                - tensors[f"natural.{donor}.margins"]
                            ).abs().max()
                        ),
                    ]
                )
                k12_errors.extend(
                    [
                        float(
                            (
                                tensors[f"{direction}.identity_target.k12"]
                                - tensors[f"natural.{target}.k12"]
                            ).abs().max()
                        ),
                        float(
                            (
                                tensors[f"{direction}.identity_donor.k12"]
                                - tensors[f"natural.{donor}.k12"]
                            ).abs().max()
                        ),
                    ]
                )
            orthogonal_pass = all(
                audit["orthogonal"][direction]["pass"]
                for audit in metadata["orthogonal_audits"]
                for direction in ("a_to_b", "b_to_a")
            )
            identity_audits[f"{pair_id}.{model}"] = {
                "margin_max_abs": max(margin_errors),
                "k12_max_abs": max(k12_errors),
                "orthogonal_controls_pass": orthogonal_pass,
                "natural_tail_declared": metadata["tail_regeneration"]
                == "natural_except_exact_declared_k12_raw_head_sites",
            }

    directional: dict[str, Any] = {}
    for pair in contract["selected_pair_specs"]:
        pair_id, a, b = pair["pair_id"], pair["concept_a"], pair["concept_b"]
        tensors = pair_data[pair_id]["chameleon"][0]
        directional[pair_id] = {
            direction: direction_metrics(
                tensors,
                prototypes,
                pair_id,
                direction,
                target,
                donor,
                probe_names,
                contract["gates"],
            )
            for direction, target, donor in (("a_to_b", a, b), ("b_to_a", b, a))
        }

    acquisition_vectors = {
        model: {field: [] for field in ("natural_probe", "natural_k12", "kv_probe")}
        for model in ("chameleon", "exact_precursor")
    }
    for pair in contract["selected_pair_specs"]:
        pair_id, a, b = pair["pair_id"], pair["concept_a"], pair["concept_b"]
        for model in acquisition_vectors:
            tensors = pair_data[pair_id][model][0]
            for direction, target, donor in (("a_to_b", a, b), ("b_to_a", b, a)):
                acquisition_vectors[model]["natural_probe"].append(
                    tensors[f"natural.{donor}.margins"]
                    - tensors[f"natural.{target}.margins"]
                )
                acquisition_vectors[model]["natural_k12"].append(
                    tensors[f"natural.{donor}.k12"] - tensors[f"natural.{target}.k12"]
                )
                acquisition_vectors[model]["kv_probe"].append(
                    tensors[f"{direction}.donor_kv_into_target.margins"]
                    - tensors[f"{direction}.identity_target.margins"]
                )
    acquisition = {}
    acquisition_gate = contract["gates"]["acquisition"]
    for field, gate_name in (
        ("natural_probe", "precursor_to_chameleon_natural_probe_effect_norm_ratio_max"),
        ("natural_k12", "precursor_to_chameleon_natural_k12_effect_norm_ratio_max"),
        ("kv_probe", "precursor_to_chameleon_kv_mediated_probe_effect_norm_ratio_max"),
    ):
        ratio = norm_ratio(
            torch.cat(acquisition_vectors["exact_precursor"][field]),
            torch.cat(acquisition_vectors["chameleon"][field]),
        )
        acquisition[field] = {
            "precursor_to_chameleon_norm_ratio": ratio,
            "maximum": float(acquisition_gate[gate_name]),
            "pass": ratio <= float(acquisition_gate[gate_name]),
        }

    negative_tensors = {
        concept: checked_negative(concept, contract_hash)[0]
        for concept in contract["selected_concepts_in_order"]
    }
    chameleon_pairs = {
        pair_id: values["chameleon"][0] for pair_id, values in pair_data.items()
    }
    operational, operational_pass = operational_metrics(
        contract, chameleon_pairs, negative_tensors, probe_names
    )
    direction_rows = [row for pair in directional.values() for row in pair.values()]
    audit_pass = all(
        row["margin_max_abs"]
        <= float(contract["implementation_gates"]["same_state_margin_max_abs"])
        and row["k12_max_abs"]
        <= float(contract["implementation_gates"]["same_state_k12_max_abs"])
        and row["orthogonal_controls_pass"]
        and row["natural_tail_declared"]
        for row in identity_audits.values()
    )
    clauses = {
        "acquisition": all(row["pass"] for row in acquisition.values()),
        "operation": all(row["passes"]["operation"] for row in direction_rows),
        "semantic_conditioning": all(
            row["passes"]["prediction"] for row in direction_rows
        ),
        "necessity_sufficiency": all(
            row["passes"]["necessity_sufficiency"] for row in direction_rows
        ),
        "endogenous_chain": all(
            row["passes"]["integrated_chain"] for row in direction_rows
        )
        and audit_pass,
        "restoration": all(row["passes"]["restoration"] for row in direction_rows),
        "operational_failure": operational_pass,
    }
    return {
        "schema_version": 1,
        "procedure": "day63-final-causal-chain-reduction-v1",
        "contract_sha256": contract_hash,
        "selected_pairs": contract["selected_pairs_in_order"],
        "selected_concepts": contract["selected_concepts_in_order"],
        "sample_sizes": contract["panels"],
        "directional_chameleon_results": directional,
        "acquisition": acquisition,
        "operational_consequences": operational,
        "raw_margin_summaries": raw_margins,
        "implementation_audits": identity_audits,
        "implementation_gate_pass": audit_pass,
        "title_clauses": clauses,
        "title_disposition": title_gate_disposition(clauses),
        "interpretation_policy": "conjunctive_no_compensatory_score",
    }


def main() -> None:
    first = reduce_all()
    second = reduce_all()
    absolute, relative = recursive_numeric_max_difference(first, second)
    verification = {
        "schema_version": 1,
        "procedure": "day63-independent-local-rereduction-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "maximum_absolute_numeric_difference": absolute,
        "maximum_relative_numeric_difference": relative,
        "exact_reduction_match": absolute == 0.0 and relative == 0.0,
    }
    if not verification["exact_reduction_match"]:
        raise RuntimeError("independent final reductions differ")
    write_json(SUMMARY_PATH, first)
    write_json(VERIFICATION_PATH, verification)
    artifact_paths = [
        path
        for path in sorted(SHARD_DIR.rglob("*"))
        if path.is_file()
    ] + [SUMMARY_PATH, VERIFICATION_PATH]
    write_json(
        MANIFEST_PATH,
        {
            "schema_version": 1,
            "procedure": "day63-final-causal-chain-artifact-manifest-v1",
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in artifact_paths
            ],
        },
    )
    print(
        json.dumps(
            {
                "title_disposition": first["title_disposition"],
                "clauses": first["title_clauses"],
                "implementation_gate_pass": first["implementation_gate_pass"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
