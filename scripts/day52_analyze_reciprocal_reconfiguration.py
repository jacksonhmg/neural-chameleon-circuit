#!/usr/bin/env python3
"""Reduce the frozen Day 52 reciprocal donor-reconfiguration test."""

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


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-52/frozen-reciprocal-reconfiguration-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-preflight.json"
EXECUTION_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day52-shards"
SUMMARY_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-summary.json"
AUDIT_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-audit.json"
METRICS_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-52/execution-artifact-manifest.json"


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


def require_execution_ancestor(commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Day 52 execution commit is not an ancestor of analysis")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)}")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    ).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def state(tensors: Mapping[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    return {
        field: tensors[f"{name}.{field}"].float()
        for field in ("k12", "monitor", "margins", "rms")
    }


def masked_vectors(value: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    if value.ndim >= 3 and value.shape[:2] == mask.shape:
        return [value[row, mask[row]].reshape(-1).double() for row in range(len(mask))]
    return [value[row].reshape(-1).double() for row in range(len(mask))]


def vector_metrics(
    intervention: torch.Tensor,
    target: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    outputs = masked_vectors(intervention, mask)
    targets = masked_vectors(target, mask)
    donors = masked_vectors(donor, mask)
    result: dict[str, list[float]] = defaultdict(list)
    for output, recipient, source in zip(outputs, targets, donors, strict=True):
        changed = output - recipient
        trajectory = source - recipient
        numerator = changed @ trajectory
        target_norm = trajectory.square().sum().clamp(min=1e-8)
        changed_norm = changed.square().sum().clamp(min=1e-8)
        result["recovery"].append(float(numerator / target_norm))
        result["cosine"].append(
            float(numerator / torch.sqrt(target_norm * changed_norm))
        )
        result["norm_ratio"].append(float(torch.sqrt(changed_norm / target_norm)))
        result["aligned_numerator"].append(float(numerator))
        result["target_norm_sq"].append(float(target_norm))
    return {key: np.asarray(values) for key, values in result.items()}


def distance_metrics(
    intervention: torch.Tensor,
    endpoints: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
) -> list[dict[str, Any]]:
    output_vectors = masked_vectors(intervention, mask)
    endpoint_vectors = {
        name: masked_vectors(value, mask) for name, value in endpoints.items()
    }
    rows = []
    for row, output in enumerate(output_vectors):
        distances = {
            name: float((output - values[row]).square().sum())
            for name, values in endpoint_vectors.items()
        }
        rows.append(
            {
                "distances": distances,
                "nearest_endpoint": min(distances, key=distances.get),
                "donor_closer_than_target": distances["donor"] < distances["target"],
                "donor_closer_than_normal": distances["donor"] < distances["normal"],
                "donor_closer_than_different": distances["donor"]
                < distances["different"],
            }
        )
    return rows


def separation_ratio(
    target: torch.Tensor,
    donor: torch.Tensor,
    normal: torch.Tensor,
    mask: torch.Tensor,
) -> np.ndarray:
    target_vectors = masked_vectors(target, mask)
    donor_vectors = masked_vectors(donor, mask)
    normal_vectors = masked_vectors(normal, mask)
    return np.asarray(
        [
            float(
                torch.linalg.vector_norm(source - baseline)
                / torch.linalg.vector_norm(recipient - baseline).clamp(min=1e-8)
            )
            for recipient, source, baseline in zip(
                target_vectors, donor_vectors, normal_vectors, strict=True
            )
        ]
    )


def component_layer(component_id: str) -> int:
    return int(component_id.split(".")[0].removeprefix("layer_"))


def expected_names(contract: Mapping[str, Any]) -> set[str]:
    return {
        *(f"natural_{name}" for name in contract["execution"]["natural_states"]),
        *(
            f"intervention_{direction}.{job}"
            for direction in contract["directions"]
            for job in contract["jobs"]["order"]
        ),
    }


def load_inputs(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    execution = read_json(EXECUTION_PATH)
    preflight = read_json(PREFLIGHT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    execution_commit = str(execution.get("execution_commit", ""))
    require_execution_ancestor(execution_commit)
    if (
        not execution.get("complete")
        or execution.get("contract_sha256") != contract_hash
    ):
        raise RuntimeError("Day 52 execution is not exact and complete")
    if (
        preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution_commit
        or preflight.get("contract_sha256") != contract_hash
    ):
        raise RuntimeError("Day 52 preflight is not exact and passing")

    components = tuple(contract["k12"]["component_ids"])
    layer_indices = {
        layer: [
            index
            for index, component in enumerate(components)
            if component_layer(component) == layer
        ]
        for layer in contract["k12"]["layers"]
    }
    rows = []
    identity_k12_errors = []
    identity_margin_errors = []
    random_passes = []
    probe_name_sets = set()
    tensor_hashes = []
    state_rows = 0
    for concept in sorted(contract["conditions"]["pairs"]):
        safe_name = concept.replace("/", "_")
        tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
        metadata_path = SHARD_DIR / f"{safe_name}.json"
        metadata = read_json(metadata_path)
        if (
            metadata["execution_commit"] != execution_commit
            or metadata["contract_sha256"] != contract_hash
            or metadata["concept"] != concept
            or metadata["tensor_sha256"] != sha256_file(tensor_path)
            or set(metadata["state_names"]) != expected_names(contract)
            or int(metadata["state_count"])
            != int(contract["execution"]["states_per_example"])
        ):
            raise RuntimeError(f"Day 52 shard metadata differs for {concept}")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        probe_names = tuple(metadata["probe_names"])
        probe_name_sets.add(probe_names)
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
                for job in contract["jobs"]["order"]
            }
            identity_k12_errors.append(
                float((interventions["identity"]["k12"] - target["k12"]).abs().max())
            )
            identity_margin_errors.append(
                float(
                    (interventions["identity"]["margins"] - target["margins"])
                    .abs()
                    .max()
                )
            )
            k12_separation = separation_ratio(
                target["k12"], donor["k12"], normal["k12"], mask
            )
            margin_separation = separation_ratio(
                target["margins"], donor["margins"], normal["margins"], mask
            )
            donor_concept = (
                contract["conditions"]["pairs"][concept]["irrelevant_trigger"]
                if specification["donor"] == "irrelevant_trigger"
                else concept
            )
            donor_probe = probe_names.index(donor_concept)
            endpoint_states = {
                "target": target,
                "donor": donor,
                "normal": normal,
                "different": different,
            }
            for job, output in interventions.items():
                k12 = vector_metrics(output["k12"], target["k12"], donor["k12"], mask)
                monitor = vector_metrics(
                    output["monitor"], target["monitor"], donor["monitor"], mask
                )
                margins = vector_metrics(
                    output["margins"], target["margins"], donor["margins"], mask
                )
                k12_distances = distance_metrics(
                    output["k12"],
                    {name: value["k12"] for name, value in endpoint_states.items()},
                    mask,
                )
                margin_distances = distance_metrics(
                    output["margins"],
                    {name: value["margins"] for name, value in endpoint_states.items()},
                    mask,
                )
                per_layer = {
                    str(layer): vector_metrics(
                        output["k12"][:, :, indices],
                        target["k12"][:, :, indices],
                        donor["k12"][:, :, indices],
                        mask,
                    )["recovery"]
                    for layer, indices in layer_indices.items()
                }
                source_endpoint = {
                    "normal_collapse": normal,
                    "different_donor": different,
                    "primary_donor": donor,
                }.get(job)
                source_recovery = (
                    vector_metrics(
                        output["k12"], target["k12"], source_endpoint["k12"], mask
                    )["recovery"]
                    if source_endpoint is not None
                    else np.full(mask.shape[0], np.nan)
                )
                natural_probe = (
                    donor["margins"][:, donor_probe] - target["margins"][:, donor_probe]
                ).numpy()
                intervention_probe = (
                    output["margins"][:, donor_probe]
                    - target["margins"][:, donor_probe]
                ).numpy()
                rms_ratio = (output["rms"] / target["rms"].clamp(min=1e-8)).numpy()
                for index, example_id in enumerate(metadata["example_ids"]):
                    rows.append(
                        {
                            "concept": concept,
                            "donor_concept": donor_concept,
                            "example_id": example_id,
                            "direction": direction,
                            "job": job,
                            "k12_donor_recovery": float(k12["recovery"][index]),
                            "k12_donor_cosine": float(k12["cosine"][index]),
                            "k12_donor_norm_ratio": float(k12["norm_ratio"][index]),
                            "k12_aligned_numerator": float(
                                k12["aligned_numerator"][index]
                            ),
                            "k12_target_norm_sq": float(k12["target_norm_sq"][index]),
                            "monitor_donor_recovery": float(monitor["recovery"][index]),
                            "probe_vector_donor_recovery": float(
                                margins["recovery"][index]
                            ),
                            "natural_k12_donor_separation_ratio": float(
                                k12_separation[index]
                            ),
                            "natural_probe_donor_separation_ratio": float(
                                margin_separation[index]
                            ),
                            "k12_endpoint": k12_distances[index],
                            "probe_endpoint": margin_distances[index],
                            "donor_probe_natural_delta": float(natural_probe[index]),
                            "donor_probe_intervention_delta": float(
                                intervention_probe[index]
                            ),
                            "donor_probe_direction_correct": bool(
                                natural_probe[index] * intervention_probe[index] > 0
                            ),
                            "activation_rms_ratio": float(rms_ratio[index]),
                            "source_endpoint_k12_recovery": (
                                None
                                if not np.isfinite(source_recovery[index])
                                else float(source_recovery[index])
                            ),
                            "per_layer_k12_donor_recovery": {
                                layer: float(values[index])
                                for layer, values in per_layer.items()
                            },
                        }
                    )
    return rows, {
        "execution": execution,
        "preflight": preflight,
        "identity_k12_max_abs": max(identity_k12_errors, default=math.inf),
        "identity_margin_max_abs": max(identity_margin_errors, default=math.inf),
        "random_audits_pass": all(random_passes),
        "probe_name_sets": [list(value) for value in sorted(probe_name_sets)],
        "tensor_hash_count": len(set(tensor_hashes)),
        "state_rows": state_rows,
    }


def concept_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["concept"], row["direction"], row["job"])].append(row)
    result = []
    scalar_keys = (
        "k12_donor_recovery",
        "monitor_donor_recovery",
        "probe_vector_donor_recovery",
        "natural_k12_donor_separation_ratio",
        "natural_probe_donor_separation_ratio",
        "k12_aligned_numerator",
    )
    for (concept, direction, job), values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError("Day 52 concept cell does not contain two examples")
        row = {
            "concept": concept,
            "donor_concept": values[0]["donor_concept"],
            "direction": direction,
            "job": job,
            **{
                key: float(np.mean([value[key] for value in values]))
                for key in scalar_keys
            },
            "activation_rms_ratio_min": float(
                min(value["activation_rms_ratio"] for value in values)
            ),
            "activation_rms_ratio_max": float(
                max(value["activation_rms_ratio"] for value in values)
            ),
            "donor_probe_direction_correct": bool(
                np.mean(
                    [
                        value["donor_probe_natural_delta"]
                        * value["donor_probe_intervention_delta"]
                        for value in values
                    ]
                )
                > 0
            ),
            "per_layer_k12_donor_recovery": {
                layer: float(
                    np.mean(
                        [
                            value["per_layer_k12_donor_recovery"][layer]
                            for value in values
                        ]
                    )
                )
                for layer in values[0]["per_layer_k12_donor_recovery"]
            },
        }
        for modality in ("k12_endpoint", "probe_endpoint"):
            distances = {
                endpoint: float(
                    np.mean(
                        [value[modality]["distances"][endpoint] for value in values]
                    )
                )
                for endpoint in ("target", "donor", "normal", "different")
            }
            prefix = modality.removesuffix("_endpoint")
            row[f"{prefix}_nearest_endpoint"] = min(distances, key=distances.get)
            row[f"{prefix}_donor_closer_than_target"] = (
                distances["donor"] < distances["target"]
            )
            row[f"{prefix}_donor_closer_than_normal"] = (
                distances["donor"] < distances["normal"]
            )
            row[f"{prefix}_donor_closer_than_different"] = (
                distances["donor"] < distances["different"]
            )
        source_values = [
            value["source_endpoint_k12_recovery"]
            for value in values
            if value["source_endpoint_k12_recovery"] is not None
        ]
        row["source_endpoint_k12_recovery"] = (
            None if not source_values else float(np.mean(source_values))
        )
        result.append(row)
    return result


def summarize_direction(
    direction: str,
    concepts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    selected = [
        row
        for row in concepts
        if row["direction"] == direction and row["job"] == "primary_donor"
    ]
    if len(selected) != 13:
        raise RuntimeError(f"Day 52 direction {direction} is incomplete")
    controls = ("haar", "normal_collapse", "different_donor")
    control_medians = {
        job: float(
            np.median(
                [
                    row["k12_donor_recovery"]
                    for row in concepts
                    if row["direction"] == direction and row["job"] == job
                ]
            )
        )
        for job in controls
    }
    primary_recovery = float(np.median([row["k12_donor_recovery"] for row in selected]))
    strongest_control = max(control_medians.values())
    positive = sum(row["k12_donor_recovery"] > 0 for row in selected)
    k12_closer = sum(
        row["k12_donor_closer_than_target"]
        and row["k12_donor_closer_than_normal"]
        and row["k12_donor_closer_than_different"]
        for row in selected
    )
    probe_closer = sum(
        row["probe_donor_closer_than_target"]
        and row["probe_donor_closer_than_normal"]
        and row["probe_donor_closer_than_different"]
        for row in selected
    )
    donor_probe = sum(row["donor_probe_direction_correct"] for row in selected)
    positive_layers = sum(
        np.median([row["per_layer_k12_donor_recovery"][str(layer)] for row in selected])
        > 0
        for layer in contract["k12"]["layers"]
    )
    absolute_numerators = np.abs([row["k12_aligned_numerator"] for row in selected])
    concentration = float(
        absolute_numerators.max() / max(float(absolute_numerators.sum()), 1e-12)
    )
    rms_low, rms_high = contract["promotion_gate"]["activation_rms_ratio_range"]
    rms_pass = all(
        row["activation_rms_ratio_min"] >= rms_low
        and row["activation_rms_ratio_max"] <= rms_high
        for row in selected
    )
    k12_separation = float(
        np.median([row["natural_k12_donor_separation_ratio"] for row in selected])
    )
    probe_separation = float(
        np.median([row["natural_probe_donor_separation_ratio"] for row in selected])
    )
    probe_recovery = float(
        np.median([row["probe_vector_donor_recovery"] for row in selected])
    )
    gates = contract["promotion_gate"]
    clauses = {
        "k12_recovery": primary_recovery
        >= float(gates["median_concept_k12_donor_recovery_min"]),
        "control_advantage": primary_recovery - strongest_control
        >= float(
            gates["advantage_over_strongest_haar_normal_or_different_control_min"]
        ),
        "k12_positive_concepts": positive
        >= int(gates["k12_positive_donor_recovery_concepts_min"]),
        "k12_donor_endpoint": k12_closer
        >= int(gates["k12_donor_closer_than_target_normal_and_different_concepts_min"]),
        "probe_recovery": probe_recovery
        >= float(gates["median_concept_probe_vector_donor_recovery_min"]),
        "probe_donor_endpoint": probe_closer
        >= int(
            gates[
                "probe_vector_donor_closer_than_target_normal_and_different_concepts_min"
            ]
        ),
        "donor_probe_direction": donor_probe
        >= int(gates["donor_own_probe_direction_correct_concepts_min"]),
        "natural_donor_separation": min(k12_separation, probe_separation)
        >= float(gates["median_natural_donor_separation_ratio_min"]),
        "distributed_layers": positive_layers
        >= int(gates["positive_median_aligned_k12_layers_min"]),
        "activation_rms": rms_pass,
    }
    summary = {
        "median_concept_k12_donor_recovery": primary_recovery,
        "mean_concept_k12_donor_recovery": float(
            np.mean([row["k12_donor_recovery"] for row in selected])
        ),
        "median_concept_monitor_donor_recovery": float(
            np.median([row["monitor_donor_recovery"] for row in selected])
        ),
        "median_concept_probe_vector_donor_recovery": probe_recovery,
        "control_median_k12_donor_recoveries": control_medians,
        "strongest_control_median_recovery": strongest_control,
        "advantage_over_strongest_control": primary_recovery - strongest_control,
        "k12_positive_donor_recovery_concepts": positive,
        "k12_donor_closer_than_all_alternatives_concepts": k12_closer,
        "probe_donor_closer_than_all_alternatives_concepts": probe_closer,
        "donor_probe_direction_correct_concepts": donor_probe,
        "median_natural_k12_donor_separation_ratio": k12_separation,
        "median_natural_probe_donor_separation_ratio": probe_separation,
        "positive_median_aligned_k12_layers": positive_layers,
        "maximum_single_concept_fraction_of_raw_aligned_numerator": concentration,
        "activation_rms_all_examples_within_range": rms_pass,
        "control_own_endpoint_median_k12_recovery": {
            job: float(
                np.median(
                    [
                        row["source_endpoint_k12_recovery"]
                        for row in concepts
                        if row["direction"] == direction
                        and row["job"] == job
                        and row["source_endpoint_k12_recovery"] is not None
                    ]
                )
            )
            for job in ("normal_collapse", "different_donor", "primary_donor")
        },
        "gate_clauses": clauses,
        "gate_pass": all(clauses.values()),
        "per_concept": selected,
    }
    return summary, clauses


def reduce_result(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    concepts = concept_means(rows)
    directions = {}
    all_clauses = {}
    for direction in contract["directions"]:
        directions[direction], clauses = summarize_direction(
            direction, concepts, contract
        )
        all_clauses[direction] = clauses
    scientific_pass = all(value["gate_pass"] for value in directions.values())
    implementation_gates = contract["implementation_gates"]
    implementation_checks = {
        "preflight_pass": audit_inputs["preflight"]["result"] == "pass",
        "execution_complete": bool(audit_inputs["execution"]["complete"]),
        "exact_state_row_count": audit_inputs["state_rows"]
        == int(implementation_gates["exact_state_row_count"]),
        "exact_probe_order": len(audit_inputs["probe_name_sets"]) == 1
        and len(audit_inputs["probe_name_sets"][0]) == 13,
        "identity_k12_within_tolerance": audit_inputs["identity_k12_max_abs"]
        <= float(implementation_gates["identity_k12_max_abs"]),
        "identity_margin_within_tolerance": audit_inputs["identity_margin_max_abs"]
        <= float(implementation_gates["identity_monitor_margin_max_abs"]),
        "attention_recompute_within_tolerance": audit_inputs["preflight"][
            "natural_attention_recompute_max_abs"
        ]
        <= float(implementation_gates["natural_attention_recompute_max_abs"]),
        "random_audits_pass": bool(audit_inputs["random_audits_pass"]),
        "hooks_removed": audit_inputs["execution"]["hooks_after_execution"] == 0,
        "thirteen_unique_tensor_hashes": audit_inputs["tensor_hash_count"] == 13,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "k12_donor_recovery",
                "k12_donor_cosine",
                "k12_donor_norm_ratio",
                "monitor_donor_recovery",
                "probe_vector_donor_recovery",
                "activation_rms_ratio",
            )
        ),
    }
    implementation_pass = all(implementation_checks.values())
    summary = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "evidence_class": contract["evidence_class"],
        "candidate": contract["candidate"]["id"],
        "direction_summaries": directions,
        "scientific_gate_clauses": all_clauses,
        "scientific_gate_pass": scientific_pass,
        "implementation_gate_pass": implementation_pass,
        "disposition": (
            "freeze_k12_mediation_for_full_prefix_qkv"
            if implementation_pass and scientific_pass
            else "stop_full_prefix_donor_reconfiguration_hypothesis"
            if implementation_pass
            else "implementation_failure_no_scientific_interpretation"
        ),
        "mediation_eligible": bool(implementation_pass and scientific_pass),
        "boundary": "development sandbox only; not fresh confirmation",
    }
    audit = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-audit",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "implementation_checks": implementation_checks,
        "implementation_pass": implementation_pass,
        "observed": {
            "example_metric_rows": len(rows),
            "state_rows": audit_inputs["state_rows"],
            "identity_k12_max_abs": audit_inputs["identity_k12_max_abs"],
            "identity_margin_max_abs": audit_inputs["identity_margin_max_abs"],
            "probe_name_sets": audit_inputs["probe_name_sets"],
            "tensor_hash_count": audit_inputs["tensor_hash_count"],
        },
        "two_in_memory_reductions_byte_identical": None,
    }
    metrics = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-example-metrics",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "rows": list(rows),
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
        "procedure": "prospective-day52-reciprocal-artifacts-v1",
        "execution_commit": read_json(EXECUTION_PATH)["execution_commit"],
        "analysis_commit": git_head(),
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
    contract = read_json(CONTRACT_PATH)
    first_rows, first_inputs = load_inputs(contract)
    first = reduce_result(contract, first_rows, first_inputs)
    second_rows, second_inputs = load_inputs(contract)
    second = reduce_result(contract, second_rows, second_inputs)
    if json_bytes(first) != json_bytes(second) or json_bytes(first_rows) != json_bytes(
        second_rows
    ):
        raise RuntimeError("two Day 52 reductions differ")
    summary, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    write_json(SUMMARY_PATH, summary)
    write_json(AUDIT_PATH, audit)
    write_json(METRICS_PATH, metrics)
    write_json(MANIFEST_PATH, output_manifest())
    if not audit["implementation_pass"]:
        raise RuntimeError(f"Day 52 implementation audit failed: {audit}")


if __name__ == "__main__":
    main()
