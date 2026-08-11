#!/usr/bin/env python3
"""Reduce the frozen Day 48 proximal upstream-controller experiment."""

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
CONTRACT_PATH = ROOT / "results/day-48/frozen-proximal-upstream-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-48/proximal-upstream-preflight.json"
EXECUTION_PATH = ROOT / "results/day-48/proximal-upstream-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day48-shards"
SUMMARY_PATH = ROOT / "results/day-48/proximal-upstream-summary.json"
AUDIT_PATH = ROOT / "results/day-48/proximal-upstream-audit.json"
METRICS_PATH = ROOT / "results/day-48/proximal-upstream-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-48/execution-artifact-manifest.json"


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
        raise RuntimeError("Day 48 execution commit is not an ancestor of analysis")


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
    fields = ("k12", "monitor", "margins", "rms")
    result = {field: tensors[f"{name}.{field}"].float() for field in fields}
    if any(value.shape[0] != 2 for value in result.values()):
        raise RuntimeError(f"state {name} does not contain two examples")
    return result


def trajectory_metrics(
    intervention: torch.Tensor,
    recipient: torch.Tensor,
    donor: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    changed = intervention.float() - recipient.float()
    target = donor.float() - recipient.float()
    expanded = mask.bool()
    while expanded.ndim < target.ndim:
        expanded = expanded.unsqueeze(-1)
    changed = torch.where(expanded, changed, 0.0)
    target = torch.where(expanded, target, 0.0)
    dimensions = tuple(range(1, target.ndim))
    numerator = (changed * target).sum(dim=dimensions)
    target_norm_sq = target.square().sum(dim=dimensions).clamp(min=1e-8)
    changed_norm_sq = changed.square().sum(dim=dimensions).clamp(min=1e-8)
    recovery = numerator / target_norm_sq
    cosine = numerator / torch.sqrt(target_norm_sq * changed_norm_sq)
    norm_ratio = torch.sqrt(changed_norm_sq / target_norm_sq)
    return {
        "recovery": recovery.numpy(),
        "cosine": cosine.numpy(),
        "norm_ratio": norm_ratio.numpy(),
        "aligned_numerator": numerator.numpy(),
        "target_norm_sq": target_norm_sq.numpy(),
    }


def vector_recovery(
    intervention: torch.Tensor, recipient: torch.Tensor, donor: torch.Tensor
) -> np.ndarray:
    changed = intervention.float() - recipient.float()
    target = donor.float() - recipient.float()
    numerator = (changed * target).sum(dim=1)
    denominator = target.square().sum(dim=1).clamp(min=1e-8)
    return (numerator / denominator).numpy()


def component_layer(component_id: str) -> int:
    return int(component_id.split(".")[0].removeprefix("layer_"))


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
        raise RuntimeError("Day 48 execution is not exact and complete")
    if (
        preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution_commit
        or preflight.get("contract_sha256") != contract_hash
    ):
        raise RuntimeError("Day 48 preflight is not exact and passing")

    rows: list[dict[str, Any]] = []
    identity_k12_errors = []
    identity_margin_errors = []
    random_passes = []
    state_rows = 0
    probe_name_sets = set()
    components = tuple(contract["k12"]["component_ids"])
    layer_indices = {
        layer: [
            index
            for index, value in enumerate(components)
            if component_layer(value) == layer
        ]
        for layer in contract["k12"]["layers"]
    }
    expected_names = {
        *(f"natural_{value}" for value in contract["jobs"]["natural_conditions"]),
        *(f"intervention_{value}" for value in contract["jobs"]["normal_target"]),
        *(f"intervention_{value}" for value in contract["jobs"]["correct_target"]),
        "intervention_different_named_span_to_correct",
    }
    for concept in sorted(contract["conditions"]["pairs"]):
        safe_name = concept.replace("/", "_")
        metadata_path = SHARD_DIR / f"{safe_name}.json"
        tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
        metadata = read_json(metadata_path)
        if (
            metadata["execution_commit"] != execution_commit
            or metadata["contract_sha256"] != contract_hash
            or metadata["concept"] != concept
            or metadata["tensor_sha256"] != sha256_file(tensor_path)
            or set(metadata["state_names"]) != expected_names
        ):
            raise RuntimeError(f"Day 48 shard metadata differs for {concept}")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        probe_names = tuple(metadata["probe_names"])
        probe_name_sets.add(probe_names)
        random_passes.append(bool(metadata["random_audit"]["pass"]))
        state_rows += int(metadata["state_count"]) * len(metadata["example_ids"])

        natural = {
            name: state(tensors, f"natural_{name}")
            for name in contract["jobs"]["natural_conditions"]
        }
        interventions = {
            name: state(tensors, f"intervention_{name}")
            for name in (
                *contract["jobs"]["normal_target"],
                *contract["jobs"]["correct_target"],
                "different_named_span_to_correct",
            )
        }
        identity_k12_errors.extend(
            [
                float(
                    (interventions["identity_normal"]["k12"] - natural["normal"]["k12"])
                    .abs()
                    .max()
                ),
                float(
                    (
                        interventions["identity_correct"]["k12"]
                        - natural["correct_trigger"]["k12"]
                    )
                    .abs()
                    .max()
                ),
            ]
        )
        identity_margin_errors.extend(
            [
                float(
                    (
                        interventions["identity_normal"]["margins"]
                        - natural["normal"]["margins"]
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        interventions["identity_correct"]["margins"]
                        - natural["correct_trigger"]["margins"]
                    )
                    .abs()
                    .max()
                ),
            ]
        )

        specifications = (
            (
                "sufficiency",
                "correct_response_to_normal",
                "normal",
                "correct_trigger",
                (
                    "identity_normal",
                    "irrelevant_response_to_normal",
                    "random_response_to_normal",
                ),
            ),
            (
                "necessity",
                "normal_response_to_correct",
                "correct_trigger",
                "normal",
                (
                    "identity_correct",
                    "irrelevant_response_to_correct",
                    "random_response_to_correct",
                ),
            ),
        )
        concept_probe = probe_names.index(concept)
        for direction, primary, recipient_name, donor_name, controls in specifications:
            recipient = natural[recipient_name]
            donor = natural[donor_name]
            for candidate in (primary, *controls):
                output = interventions[candidate]
                metrics = trajectory_metrics(
                    output["k12"], recipient["k12"], donor["k12"], mask
                )
                margin_recovery = vector_recovery(
                    output["margins"], recipient["margins"], donor["margins"]
                )
                per_layer = {}
                for layer, indices in layer_indices.items():
                    layer_metrics = trajectory_metrics(
                        output["k12"][:, :, indices],
                        recipient["k12"][:, :, indices],
                        donor["k12"][:, :, indices],
                        mask,
                    )
                    per_layer[layer] = layer_metrics["recovery"]
                natural_own = (
                    donor["margins"][:, concept_probe]
                    - recipient["margins"][:, concept_probe]
                ).numpy()
                intervention_own = (
                    output["margins"][:, concept_probe]
                    - recipient["margins"][:, concept_probe]
                ).numpy()
                rms_ratio = (output["rms"] / recipient["rms"].clamp(min=1e-8)).numpy()
                for index, example_id in enumerate(metadata["example_ids"]):
                    rows.append(
                        {
                            "concept": concept,
                            "example_id": example_id,
                            "direction": direction,
                            "candidate": candidate,
                            "is_primary": candidate == primary,
                            "k12_recovery": float(metrics["recovery"][index]),
                            "k12_cosine": float(metrics["cosine"][index]),
                            "k12_norm_ratio": float(metrics["norm_ratio"][index]),
                            "k12_aligned_numerator": float(
                                metrics["aligned_numerator"][index]
                            ),
                            "k12_target_norm_sq": float(
                                metrics["target_norm_sq"][index]
                            ),
                            "monitor_vector_recovery": float(margin_recovery[index]),
                            "natural_own_probe_delta": float(natural_own[index]),
                            "intervention_own_probe_delta": float(
                                intervention_own[index]
                            ),
                            "own_probe_direction_correct": bool(
                                natural_own[index] * intervention_own[index] > 0
                            ),
                            "activation_rms_ratio": float(rms_ratio[index]),
                            "per_layer_k12_recovery": {
                                str(layer): float(values[index])
                                for layer, values in per_layer.items()
                            },
                        }
                    )

        span_metrics = trajectory_metrics(
            interventions["different_named_span_to_correct"]["k12"],
            natural["correct_trigger"]["k12"],
            natural["different_trigger"]["k12"],
            mask,
        )
        span_monitor_recovery = vector_recovery(
            interventions["different_named_span_to_correct"]["margins"],
            natural["correct_trigger"]["margins"],
            natural["different_trigger"]["margins"],
        )
        for index, example_id in enumerate(metadata["example_ids"]):
            rows.append(
                {
                    "concept": concept,
                    "example_id": example_id,
                    "direction": "phase_c_comparator",
                    "candidate": "different_named_span_to_correct",
                    "is_primary": False,
                    "k12_recovery": float(span_metrics["recovery"][index]),
                    "k12_cosine": float(span_metrics["cosine"][index]),
                    "k12_norm_ratio": float(span_metrics["norm_ratio"][index]),
                    "k12_aligned_numerator": float(
                        span_metrics["aligned_numerator"][index]
                    ),
                    "k12_target_norm_sq": float(span_metrics["target_norm_sq"][index]),
                    "monitor_vector_recovery": float(span_monitor_recovery[index]),
                    "activation_rms_ratio": float(
                        interventions["different_named_span_to_correct"]["rms"][index]
                        / natural["correct_trigger"]["rms"][index].clamp(min=1e-8)
                    ),
                }
            )

    audit_inputs = {
        "execution": execution,
        "preflight": preflight,
        "identity_k12_max_abs": max(identity_k12_errors, default=math.inf),
        "identity_margin_max_abs": max(identity_margin_errors, default=math.inf),
        "random_audits_pass": all(random_passes),
        "state_rows": state_rows,
        "probe_name_sets": [list(value) for value in sorted(probe_name_sets)],
    }
    return rows, audit_inputs


def concept_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["direction"] == "phase_c_comparator":
            continue
        grouped[(row["concept"], row["direction"], row["candidate"])].append(row)
    result = []
    for (concept, direction, candidate), values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError("concept/candidate cell is not exactly two examples")
        result.append(
            {
                "concept": concept,
                "direction": direction,
                "candidate": candidate,
                "is_primary": bool(values[0]["is_primary"]),
                "k12_recovery": float(np.mean([row["k12_recovery"] for row in values])),
                "monitor_vector_recovery": float(
                    np.mean([row["monitor_vector_recovery"] for row in values])
                ),
                "own_probe_direction_correct": bool(
                    np.mean(
                        [
                            row["natural_own_probe_delta"]
                            * row["intervention_own_probe_delta"]
                            for row in values
                        ]
                    )
                    > 0
                ),
                "natural_own_probe_delta": float(
                    np.mean([row["natural_own_probe_delta"] for row in values])
                ),
                "activation_rms_ratio_min": float(
                    min(row["activation_rms_ratio"] for row in values)
                ),
                "activation_rms_ratio_max": float(
                    max(row["activation_rms_ratio"] for row in values)
                ),
                "aligned_numerator": float(
                    sum(row["k12_aligned_numerator"] for row in values)
                ),
                "per_layer_k12_recovery": {
                    layer: float(
                        np.mean(
                            [row["per_layer_k12_recovery"][layer] for row in values]
                        )
                    )
                    for layer in values[0]["per_layer_k12_recovery"]
                },
            }
        )
    return result


def reduce_result(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    concepts = concept_means(rows)
    gates = contract["promotion_gate"]
    directions = {
        "sufficiency": "correct_response_to_normal",
        "necessity": "normal_response_to_correct",
    }
    direction_summaries = {}
    clause_values: dict[str, bool] = {}
    max_concept_fraction = 0.0
    for direction, primary in directions.items():
        selected = [
            row
            for row in concepts
            if row["direction"] == direction and row["candidate"] == primary
        ]
        controls = sorted(
            {
                row["candidate"]
                for row in concepts
                if row["direction"] == direction and row["candidate"] != primary
            }
        )
        if len(selected) != 13:
            raise RuntimeError(f"direction {direction} is incomplete")
        primary_median = float(np.median([row["k12_recovery"] for row in selected]))
        control_medians = {
            candidate: float(
                np.median(
                    [
                        row["k12_recovery"]
                        for row in concepts
                        if row["direction"] == direction
                        and row["candidate"] == candidate
                    ]
                )
            )
            for candidate in controls
        }
        strongest_control = max(control_medians.values())
        direction_correct = sum(row["k12_recovery"] > 0 for row in selected)
        own_correct = sum(row["own_probe_direction_correct"] for row in selected)
        positive_layers = sum(
            np.median([row["per_layer_k12_recovery"][str(layer)] for row in selected])
            > 0
            for layer in contract["k12"]["layers"]
        )
        absolute_numerators = np.abs([row["aligned_numerator"] for row in selected])
        concept_fraction = float(
            absolute_numerators.max() / max(float(absolute_numerators.sum()), 1e-12)
        )
        max_concept_fraction = max(max_concept_fraction, concept_fraction)
        rms_low, rms_high = gates["activation_rms_ratio_range"]
        rms_pass = all(
            row["activation_rms_ratio_min"] >= rms_low
            and row["activation_rms_ratio_max"] <= rms_high
            for row in selected
        )
        natural_own_correct = sum(
            row["natural_own_probe_delta"] < 0
            if direction == "sufficiency"
            else row["natural_own_probe_delta"] > 0
            for row in selected
        )
        direction_summaries[direction] = {
            "primary_candidate": primary,
            "median_concept_k12_recovery": primary_median,
            "mean_concept_k12_recovery": float(
                np.mean([row["k12_recovery"] for row in selected])
            ),
            "median_concept_monitor_vector_recovery": float(
                np.median([row["monitor_vector_recovery"] for row in selected])
            ),
            "control_median_recoveries": control_medians,
            "strongest_control_median_recovery": strongest_control,
            "advantage_over_strongest_control": primary_median - strongest_control,
            "k12_direction_correct_concepts": direction_correct,
            "own_probe_direction_correct_concepts": own_correct,
            "natural_own_probe_direction_correct_concepts": natural_own_correct,
            "positive_median_aligned_k12_layers": positive_layers,
            "maximum_single_concept_fraction_of_aligned_numerator": concept_fraction,
            "activation_rms_all_examples_within_range": rms_pass,
            "per_concept": selected,
        }
        clause_values[f"{direction}_recovery"] = primary_median >= float(
            gates[f"{direction}_median_concept_recovery_min"]
        )
        clause_values[f"{direction}_control_advantage"] = (
            primary_median - strongest_control
            >= float(gates["advantage_over_strongest_control_min"])
        )
        clause_values[f"{direction}_k12_direction"] = direction_correct >= int(
            gates["k12_direction_correct_concepts_min_per_direction"]
        )
        clause_values[f"{direction}_own_probe_direction"] = own_correct >= int(
            gates["own_probe_direction_correct_concepts_min_per_direction"]
        )
        clause_values[f"{direction}_natural_own_probe_direction"] = (
            natural_own_correct
            >= int(gates["natural_own_probe_direction_correct_concepts_min"])
        )
        clause_values[f"{direction}_rms"] = rms_pass
        clause_values[f"{direction}_distributed_layers"] = positive_layers >= int(
            gates["distributed_support"]["positive_median_aligned_k12_layers_min"]
        )

    clause_values["distributed_concepts"] = max_concept_fraction <= float(
        gates["distributed_support"][
            "maximum_single_concept_fraction_of_total_aligned_numerator"
        ]
    )
    implementation_gates = contract["implementation_gates"]
    implementation_checks = {
        "preflight_pass": audit_inputs["preflight"]["result"] == "pass",
        "execution_complete": bool(audit_inputs["execution"]["complete"]),
        "exact_state_row_count": audit_inputs["state_rows"]
        == int(implementation_gates["exact_state_row_count"]),
        "exact_probe_order": len(audit_inputs["probe_name_sets"]) == 1
        and len(audit_inputs["probe_name_sets"][0]) == 13,
        "identity_k12_within_tolerance": audit_inputs["identity_k12_max_abs"]
        <= float(implementation_gates["same_condition_identity_k12_max_abs"]),
        "identity_margin_within_tolerance": audit_inputs["identity_margin_max_abs"]
        <= float(
            implementation_gates["same_condition_identity_monitor_margin_max_abs"]
        ),
        "random_audits_pass": bool(audit_inputs["random_audits_pass"]),
        "hooks_removed": audit_inputs["execution"]["hooks_after_execution"] == 0,
        "all_metrics_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in (
                "k12_recovery",
                "k12_cosine",
                "k12_norm_ratio",
                "k12_aligned_numerator",
                "k12_target_norm_sq",
                "monitor_vector_recovery",
                "activation_rms_ratio",
            )
        ),
    }
    scientific_pass = all(clause_values.values())
    implementation_pass = all(implementation_checks.values())
    comparator = [row for row in rows if row["direction"] == "phase_c_comparator"]
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-controller-day48-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "evidence_class": contract["evidence_class"],
        "direction_summaries": direction_summaries,
        "phase_c_named_span_comparator": {
            "mean_k12_recovery": float(
                np.mean([row["k12_recovery"] for row in comparator])
            ),
            "median_k12_recovery": float(
                np.median([row["k12_recovery"] for row in comparator])
            ),
            "mean_monitor_vector_recovery": float(
                np.mean([row["monitor_vector_recovery"] for row in comparator])
            ),
        },
        "scientific_gate_clauses": clause_values,
        "scientific_gate_pass": scientific_pass,
        "implementation_gate_pass": implementation_pass,
        "disposition": (
            "promote_response_interface_to_backward_localization"
            if implementation_pass and scientific_pass
            else "freeze_selected_prompt_memory_branch"
            if implementation_pass
            else "implementation_failure_no_scientific_interpretation"
        ),
        "boundary": "development sandbox only; not fresh confirmation",
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-controller-day48-audit-v1",
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
        },
        "two_in_memory_reductions_byte_identical": None,
    }
    metrics = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-controller-example-metrics-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "rows": list(rows),
    }
    return summary, audit, metrics


def manifest() -> dict[str, Any]:
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
        "procedure": "rapid-k12-proximal-upstream-controller-artifacts-v1",
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
        raise RuntimeError("two Day 48 reductions differ")
    summary, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    write_json(SUMMARY_PATH, summary)
    write_json(AUDIT_PATH, audit)
    write_json(METRICS_PATH, metrics)
    write_json(MANIFEST_PATH, manifest())
    if not audit["implementation_pass"]:
        raise RuntimeError(f"Day 48 implementation audit failed: {audit}")


if __name__ == "__main__":
    main()
