#!/usr/bin/env python3
"""Reduce the frozen Day 49 selected prompt-memory experiment."""

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
CONTRACT_PATH = ROOT / "results/day-49/frozen-prompt-memory-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-49/prompt-memory-preflight.json"
EXECUTION_PATH = ROOT / "results/day-49/prompt-memory-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day49-shards"
SUMMARY_PATH = ROOT / "results/day-49/prompt-memory-summary.json"
AUDIT_PATH = ROOT / "results/day-49/prompt-memory-audit.json"
METRICS_PATH = ROOT / "results/day-49/prompt-memory-example-metrics.json"
MANIFEST_PATH = ROOT / "results/day-49/execution-artifact-manifest.json"


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
        raise RuntimeError("Day 49 execution commit is not an ancestor of analysis")


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
    return {
        "recovery": (numerator / target_norm_sq).numpy(),
        "cosine": (numerator / torch.sqrt(target_norm_sq * changed_norm_sq)).numpy(),
        "norm_ratio": torch.sqrt(changed_norm_sq / target_norm_sq).numpy(),
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


def expected_state_names(contract: Mapping[str, Any]) -> set[str]:
    return {
        *(f"natural_{value}" for value in contract["conditions"]["natural_conditions"]),
        *(
            f"intervention_{direction}.{job}"
            for direction in ("sufficiency", "necessity")
            for job in contract["execution"]["job_order"]
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
        raise RuntimeError("Day 49 execution is not exact and complete")
    if (
        preflight.get("result") != "pass"
        or preflight.get("execution_commit") != execution_commit
        or preflight.get("contract_sha256") != contract_hash
    ):
        raise RuntimeError("Day 49 preflight is not exact and passing")

    candidates = tuple(contract["candidates_in_simplicity_order"])
    components = tuple(contract["k12"]["component_ids"])
    layer_indices = {
        layer: [
            index
            for index, component in enumerate(components)
            if component_layer(component) == layer
        ]
        for layer in contract["k12"]["layers"]
    }
    expected_names = expected_state_names(contract)
    rows: list[dict[str, Any]] = []
    identity_k12_errors = []
    identity_margin_errors = []
    random_passes = []
    state_rows = 0
    probe_name_sets = set()
    all_tensor_hashes = []
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
            or int(metadata["state_count"]) != 44
        ):
            raise RuntimeError(f"Day 49 shard metadata differs for {concept}")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        probe_names = tuple(metadata["probe_names"])
        probe_name_sets.add(probe_names)
        random_passes.extend(
            bool(value["pass"]) for value in metadata["random_audits"].values()
        )
        all_tensor_hashes.append(metadata["tensor_sha256"])
        state_rows += int(metadata["state_count"]) * len(metadata["example_ids"])
        natural = {
            name: state(tensors, f"natural_{name}")
            for name in contract["conditions"]["natural_conditions"]
        }
        interventions = {
            direction: {
                job: state(tensors, f"intervention_{direction}.{job}")
                for job in contract["execution"]["job_order"]
            }
            for direction in ("sufficiency", "necessity")
        }
        identity_k12_errors.extend(
            [
                float(
                    (
                        interventions["sufficiency"]["identity"]["k12"]
                        - natural["normal"]["k12"]
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        interventions["necessity"]["identity"]["k12"]
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
                        interventions["sufficiency"]["identity"]["margins"]
                        - natural["normal"]["margins"]
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        interventions["necessity"]["identity"]["margins"]
                        - natural["correct_trigger"]["margins"]
                    )
                    .abs()
                    .max()
                ),
            ]
        )
        concept_probe = probe_names.index(concept)
        for direction, recipient_name, donor_name in (
            ("sufficiency", "normal", "correct_trigger"),
            ("necessity", "correct_trigger", "normal"),
        ):
            recipient = natural[recipient_name]
            donor = natural[donor_name]
            for candidate in candidates:
                for source_kind, job in (
                    ("primary", f"primary.{candidate}"),
                    ("irrelevant", f"irrelevant.{candidate}"),
                    ("random", "random"),
                ):
                    output = interventions[direction][job]
                    metrics = trajectory_metrics(
                        output["k12"], recipient["k12"], donor["k12"], mask
                    )
                    monitor_recovery = vector_recovery(
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
                        per_layer[str(layer)] = layer_metrics["recovery"]
                    per_head = {}
                    for head_index, component_id in enumerate(components):
                        head_metrics = trajectory_metrics(
                            output["k12"][:, :, head_index : head_index + 1],
                            recipient["k12"][:, :, head_index : head_index + 1],
                            donor["k12"][:, :, head_index : head_index + 1],
                            mask,
                        )
                        per_head[component_id] = head_metrics["aligned_numerator"]
                    natural_own = (
                        donor["margins"][:, concept_probe]
                        - recipient["margins"][:, concept_probe]
                    ).numpy()
                    intervention_own = (
                        output["margins"][:, concept_probe]
                        - recipient["margins"][:, concept_probe]
                    ).numpy()
                    rms_ratio = (
                        output["rms"] / recipient["rms"].clamp(min=1e-8)
                    ).numpy()
                    for index, example_id in enumerate(metadata["example_ids"]):
                        rows.append(
                            {
                                "concept": concept,
                                "example_id": example_id,
                                "direction": direction,
                                "candidate": candidate,
                                "source_kind": source_kind,
                                "k12_recovery": float(metrics["recovery"][index]),
                                "k12_cosine": float(metrics["cosine"][index]),
                                "k12_norm_ratio": float(metrics["norm_ratio"][index]),
                                "k12_aligned_numerator": float(
                                    metrics["aligned_numerator"][index]
                                ),
                                "k12_target_norm_sq": float(
                                    metrics["target_norm_sq"][index]
                                ),
                                "monitor_vector_recovery": float(
                                    monitor_recovery[index]
                                ),
                                "natural_own_probe_delta": float(natural_own[index]),
                                "intervention_own_probe_delta": float(
                                    intervention_own[index]
                                ),
                                "own_probe_direction_correct": bool(
                                    natural_own[index] * intervention_own[index] > 0
                                ),
                                "activation_rms_ratio": float(rms_ratio[index]),
                                "per_layer_k12_recovery": {
                                    layer: float(values[index])
                                    for layer, values in per_layer.items()
                                },
                                "per_head_aligned_numerator": {
                                    component: float(values[index])
                                    for component, values in per_head.items()
                                },
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
        "tensor_hash_count": len(set(all_tensor_hashes)),
    }
    return rows, audit_inputs


def concept_means(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[
            (row["concept"], row["direction"], row["candidate"], row["source_kind"])
        ].append(row)
    result = []
    for (concept, direction, candidate, source_kind), values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError("Day 49 concept/candidate cell is not two examples")
        result.append(
            {
                "concept": concept,
                "direction": direction,
                "candidate": candidate,
                "source_kind": source_kind,
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
                "per_head_aligned_numerator": {
                    component: float(
                        sum(
                            row["per_head_aligned_numerator"][component]
                            for row in values
                        )
                    )
                    for component in values[0]["per_head_aligned_numerator"]
                },
            }
        )
    return result


def summarize_candidate(
    candidate: str,
    concepts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    gates = contract["promotion_gate"]
    directions = {}
    clauses: dict[str, bool] = {}
    for direction in ("sufficiency", "necessity"):
        primary = [
            row
            for row in concepts
            if row["candidate"] == candidate
            and row["direction"] == direction
            and row["source_kind"] == "primary"
        ]
        if len(primary) != 13:
            raise RuntimeError(f"candidate {candidate} {direction} is incomplete")
        control_medians = {
            source_kind: float(
                np.median(
                    [
                        row["k12_recovery"]
                        for row in concepts
                        if row["candidate"] == candidate
                        and row["direction"] == direction
                        and row["source_kind"] == source_kind
                    ]
                )
            )
            for source_kind in ("irrelevant", "random")
        }
        median_recovery = float(np.median([row["k12_recovery"] for row in primary]))
        strongest_control = max(control_medians.values())
        direction_correct = sum(row["k12_recovery"] > 0 for row in primary)
        own_correct = sum(row["own_probe_direction_correct"] for row in primary)
        positive_layers = sum(
            np.median([row["per_layer_k12_recovery"][str(layer)] for row in primary])
            > 0
            for layer in contract["k12"]["layers"]
        )
        absolute_numerators = np.abs([row["aligned_numerator"] for row in primary])
        concept_fraction = float(
            absolute_numerators.max() / max(float(absolute_numerators.sum()), 1e-12)
        )
        rms_low, rms_high = gates["activation_rms_ratio_range"]
        rms_pass = all(
            row["activation_rms_ratio_min"] >= rms_low
            and row["activation_rms_ratio_max"] <= rms_high
            for row in primary
        )
        per_layer_medians = {
            str(layer): float(
                np.median(
                    [row["per_layer_k12_recovery"][str(layer)] for row in primary]
                )
            )
            for layer in contract["k12"]["layers"]
        }
        per_head_numerators = {
            component: float(
                sum(row["per_head_aligned_numerator"][component] for row in primary)
            )
            for component in contract["k12"]["component_ids"]
        }
        directions[direction] = {
            "median_concept_k12_recovery": median_recovery,
            "mean_concept_k12_recovery": float(
                np.mean([row["k12_recovery"] for row in primary])
            ),
            "median_concept_monitor_vector_recovery": float(
                np.median([row["monitor_vector_recovery"] for row in primary])
            ),
            "control_median_recoveries": control_medians,
            "strongest_control_median_recovery": strongest_control,
            "advantage_over_strongest_control": median_recovery - strongest_control,
            "k12_direction_correct_concepts": direction_correct,
            "own_probe_direction_correct_concepts": own_correct,
            "positive_median_aligned_k12_layers": positive_layers,
            "maximum_single_concept_fraction_of_aligned_numerator": concept_fraction,
            "activation_rms_all_examples_within_range": rms_pass,
            "per_layer_median_k12_recovery": per_layer_medians,
            "per_head_total_aligned_numerator": per_head_numerators,
            "per_concept": primary,
        }
        prefix = direction
        clauses[f"{prefix}_recovery"] = median_recovery >= float(
            gates["median_concept_k12_recovery_min_per_direction"]
        )
        clauses[f"{prefix}_control_advantage"] = (
            median_recovery - strongest_control
            >= float(
                gates[
                    "advantage_over_strongest_irrelevant_or_random_control_min_per_direction"
                ]
            )
        )
        clauses[f"{prefix}_k12_direction"] = direction_correct >= int(
            gates["k12_direction_correct_concepts_min_per_direction"]
        )
        clauses[f"{prefix}_own_probe_direction"] = own_correct >= int(
            gates["own_probe_direction_correct_concepts_min_per_direction"]
        )
        clauses[f"{prefix}_rms"] = rms_pass
        clauses[f"{prefix}_distributed_layers"] = positive_layers >= int(
            gates["positive_median_aligned_k12_layers_min_per_direction"]
        )
        clauses[f"{prefix}_distributed_concepts"] = concept_fraction <= float(
            gates["maximum_single_concept_fraction_of_total_aligned_numerator"]
        )
    return {
        "candidate": candidate,
        "direction_summaries": directions,
        "gate_clauses": clauses,
        "gate_pass": all(clauses.values()),
    }


def reduce_result(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    concepts = concept_means(rows)
    candidates = [
        summarize_candidate(candidate, concepts, contract)
        for candidate in contract["candidates_in_simplicity_order"]
    ]
    selected = next((row["candidate"] for row in candidates if row["gate_pass"]), None)
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
        "all_shards_have_unique_hashes": audit_inputs["tensor_hash_count"] == 13,
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
    implementation_pass = all(implementation_checks.values())
    scientific_pass = selected is not None
    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-prompt-memory-day49-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_commit": audit_inputs["execution"]["execution_commit"],
        "analysis_commit": git_head(),
        "evidence_class": contract["evidence_class"],
        "candidate_summaries_in_frozen_order": candidates,
        "selected_first_passing_candidate": selected,
        "scientific_gate_pass": scientific_pass,
        "implementation_gate_pass": implementation_pass,
        "disposition": (
            "freeze_winner_reconfiguration_and_mediation"
            if implementation_pass and scientific_pass
            else "stop_no_compact_tested_controller"
            if implementation_pass
            else "implementation_failure_no_scientific_interpretation"
        ),
        "day50_eligible": bool(implementation_pass and scientific_pass),
        "boundary": "development sandbox only; not fresh confirmation",
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-prompt-memory-day49-audit-v1",
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
        "procedure": "rapid-k12-selected-prompt-memory-example-metrics-v1",
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
        "procedure": "rapid-k12-selected-prompt-memory-artifacts-v1",
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
        raise RuntimeError("two Day 49 reductions differ")
    summary, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    write_json(SUMMARY_PATH, summary)
    write_json(AUDIT_PATH, audit)
    write_json(METRICS_PATH, metrics)
    write_json(MANIFEST_PATH, manifest())
    if not audit["implementation_pass"]:
        raise RuntimeError(f"Day 49 implementation audit failed: {audit}")


if __name__ == "__main__":
    main()
