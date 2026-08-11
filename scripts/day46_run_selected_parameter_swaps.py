#!/usr/bin/env python3
"""Run frozen selected-versus-control exact-precursor parameter swaps."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import (  # noqa: E402
    load_model,
    load_probes,
    replacements_from_delta,
    selected_values,
)
from day45_run_prototype_population import (  # noqa: E402
    git_head,
    load_population_records,
    make_batches,
    mean_margins,
    prototype_delta_for_batch,
    read_json,
    require_committed,
    sha256_file,
    write_json_atomic,
)
from day46_run_prototype_total_path import (  # noqa: E402
    run_cached_tail,
    total_jobs,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    response_activation_rms,
)


CONTRACT_PATH = ROOT / "results/day-46/frozen-selected-parameter-swap-contract.json"
SLICE_MANIFEST_PATH = ROOT / "results/day-46/precursor-parameter-slices.json"
SLICE_TENSOR_PATH = ROOT / "artifacts/rapid-k12-v1/precursor-parameter-slices.safetensors"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
PROTOTYPE_TENSOR_PATH = ROOT / "artifacts/rapid-k12-v1/population-prototypes.safetensors"
DAY45_CONTRACT_PATH = ROOT / "results/day-45/frozen-prototype-population-contract.json"
DAY46_CONTRACT_PATH = ROOT / "results/day-46/frozen-prototype-total-path-contract.json"
DAY46_SUMMARY_PATH = ROOT / "results/day-46/total-path-summary.json"
DAY46_AUDIT_PATH = ROOT / "results/day-46/total-path-audit.json"
DAY46_ARTIFACT_MANIFEST_PATH = ROOT / "results/day-46/execution-artifact-manifest.json"
PRECURSOR_FILE_MANIFEST_PATH = ROOT / "manifests/day-05-base-model.sha256"
DAY46_SHARD_DIR = ROOT / "results/day-46/total-path-shards"
DAY45_RUNNER_PATH = ROOT / "scripts/day45_run_prototype_population.py"
DAY46_RUNNER_PATH = ROOT / "scripts/day46_run_prototype_total_path.py"
DEFAULT_OUTPUT_DIR = ROOT / "results/day-46"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concept", action="append")
    return parser.parse_args()


def validate_parents(contract: Mapping[str, Any]) -> None:
    expected = contract["parents"]
    observed = {
        "day46_contract_sha256": sha256_file(DAY46_CONTRACT_PATH),
        "day46_summary_sha256": sha256_file(DAY46_SUMMARY_PATH),
        "day46_audit_sha256": sha256_file(DAY46_AUDIT_PATH),
        "day46_artifact_manifest_sha256": sha256_file(
            DAY46_ARTIFACT_MANIFEST_PATH
        ),
        "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "precursor_file_manifest_sha256": sha256_file(
            PRECURSOR_FILE_MANIFEST_PATH
        ),
    }
    for key, value in observed.items():
        if expected[key] != value:
            raise RuntimeError(f"frozen parent differs: {key}")


def prepare_pair(runner: Any, records: Sequence[Mapping[str, Any]]) -> Any:
    concept = records[0]["concept"]
    if any(row["concept"] != concept for row in records):
        raise RuntimeError("parameter-swap batches must contain one concept")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        trigger=concept,
    )
    expected_counts = [int(row["response_token_count"]) for row in records]
    observed_counts = pair.normal.response_mask.sum(dim=1).tolist()
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"response token counts differ: {observed_counts} != {expected_counts}"
        )
    if not torch.equal(pair.normal.response_ids, pair.triggered.response_ids) or not torch.equal(
        pair.normal.response_mask, pair.triggered.response_mask
    ):
        raise RuntimeError("paired response tensors differ")
    return pair


def run_realized_pair(runner: Any, pair: Any) -> tuple[Any, Any]:
    realized = RealizedForwardRunner(
        runner, monitor_layer=12, full_residual_layers=(9,)
    )
    normal = realized.run(pair.normal)
    triggered = realized.run(pair.triggered)
    if not torch.equal(normal.response_ids, triggered.response_ids) or not torch.equal(
        normal.response_mask, triggered.response_mask
    ):
        raise RuntimeError("realized paired response tensors differ")
    return normal, triggered


def scope_projections(scope: str) -> tuple[str, ...]:
    values = {
        "none": (),
        "O": ("o_proj",),
        "Q": ("q_proj",),
        "KV": ("k_proj", "v_proj"),
        "QKV": ("q_proj", "k_proj", "v_proj"),
        "QKVO": ("q_proj", "k_proj", "v_proj", "o_proj"),
    }
    if scope not in values:
        raise ValueError(f"unknown parameter scope {scope}")
    return values[scope]


def parameter_slice(
    parameter: torch.Tensor,
    projection: str,
    head: int,
    head_dim: int,
) -> torch.Tensor:
    if projection == "o_proj":
        return parameter[:, head * head_dim : (head + 1) * head_dim]
    return parameter[head * head_dim : (head + 1) * head_dim]


@contextmanager
def precursor_parameter_state(
    runner: Any,
    state: Mapping[str, Any],
    contract: Mapping[str, Any],
    precursor_slices: Mapping[str, torch.Tensor],
) -> Iterator[dict[str, Any]]:
    head_dim = int(contract["architecture"]["head_dim"])
    scope = state["scope"]
    head_set = state["head_set"]
    snapshots: list[tuple[torch.Tensor, str, int, torch.Tensor]] = []
    applied = []
    audit = {"applied_slices": applied, "restore_max_abs": 0.0}
    try:
        with torch.no_grad():
            for projection in scope_projections(scope):
                for layer in contract["architecture"]["layers"]:
                    attention = runner.layers[layer].self_attn
                    parameter = getattr(attention, projection).weight
                    head_kind = (
                        "key_value" if projection in {"k_proj", "v_proj"} else "query"
                    )
                    heads = list(contract["head_sets"][head_set][str(layer)][head_kind])
                    key = f"{head_set}.layer_{layer:02d}.{projection}"
                    source = precursor_slices[key]
                    if projection == "o_proj":
                        expected = (parameter.shape[0], len(heads) * head_dim)
                    else:
                        expected = (len(heads) * head_dim, parameter.shape[1])
                    if tuple(source.shape) != tuple(expected):
                        raise RuntimeError(f"precursor slice shape differs for {key}")
                    for offset, head in enumerate(heads):
                        target = parameter_slice(parameter, projection, head, head_dim)
                        snapshots.append((parameter, projection, head, target.detach().clone()))
                        if projection == "o_proj":
                            replacement = source[
                                :, offset * head_dim : (offset + 1) * head_dim
                            ]
                        else:
                            replacement = source[
                                offset * head_dim : (offset + 1) * head_dim
                            ]
                        target.copy_(
                            replacement.to(device=target.device, dtype=target.dtype)
                        )
                        applied.append(
                            {
                                "layer": layer,
                                "projection": projection,
                                "head": head,
                            }
                        )
        yield audit
    finally:
        restore_max = 0.0
        with torch.no_grad():
            for parameter, projection, head, original in reversed(snapshots):
                target = parameter_slice(parameter, projection, head, head_dim)
                target.copy_(original)
                restore_max = max(
                    restore_max,
                    float((target - original).float().abs().max()),
                )
        audit["restore_max_abs"] = restore_max


def layer9_input_error(
    normal: Any,
    triggered: Any,
    baseline_normal: torch.Tensor,
    baseline_triggered: torch.Tensor,
) -> float:
    return max(
        float((normal.full_residuals[9] - baseline_normal).float().abs().max()),
        float(
            (triggered.full_residuals[9] - baseline_triggered).float().abs().max()
        ),
    )


def total_endpoint_rows(
    output: Any,
    output_index: int,
    target: Any,
    target_margins: torch.Tensor,
    target_rms: torch.Tensor,
    probe_names: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    state_id: str,
    scope: str,
    head_set: str,
    direction: str,
    operator: str,
    commit: str,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, record in enumerate(records):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "rapid_k12_parameter_swap_total_endpoint",
                "execution_commit": commit,
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "slice_manifest_sha256": sha256_file(SLICE_MANIFEST_PATH),
                "example_id": record["example_id"],
                "concept": record["concept"],
                "split": record["split"],
                "label": int(record["label"]),
                "parameter_state": state_id,
                "parameter_scope": scope,
                "head_set": head_set,
                "direction": direction,
                "operator": operator,
                "path": "total",
                "probe_names": list(probe_names),
                "mean_raw_margins": output.mean_margins[
                    output_index, row_index
                ].tolist(),
                "target_mean_raw_margins": target_margins[:, row_index].tolist(),
                "activation_rms": float(output.activation_rms[output_index, row_index]),
                "target_activation_rms": float(target_rms[row_index]),
                "response_token_count": int(target.response_mask[row_index].sum()),
            }
        )
    return rows


def vector_comparison(
    candidate: torch.Tensor, reference: torch.Tensor
) -> dict[str, float]:
    candidate = candidate.double().flatten()
    reference = reference.double().flatten()
    reference_norm_sq = max(float(reference @ reference), 1e-12)
    candidate_norm = max(float(torch.linalg.vector_norm(candidate)), 1e-12)
    reference_norm = max(float(torch.linalg.vector_norm(reference)), 1e-12)
    return {
        "recovery": float(candidate @ reference) / reference_norm_sq,
        "cosine": float(candidate @ reference) / (candidate_norm * reference_norm),
        "relative_l2": float(torch.linalg.vector_norm(candidate - reference))
        / reference_norm,
        "norm_ratio": candidate_norm / reference_norm,
    }


def trajectory_rows(
    state_id: str,
    state: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    candidate_delta: torch.Tensor,
    baseline_delta: torch.Tensor,
    prototype_delta: torch.Tensor,
    component_ids: Sequence[str],
    commit: str,
) -> list[dict[str, Any]]:
    layer_indices: dict[int, list[int]] = {}
    for index, component_id in enumerate(component_ids):
        layer = int(component_id.split(".")[0].removeprefix("layer_"))
        layer_indices.setdefault(layer, []).append(index)
    rows = []
    for row_index, record in enumerate(records):
        valid = response_mask[row_index].bool()
        candidate = candidate_delta[row_index, valid]
        baseline = baseline_delta[row_index, valid]
        prototype = prototype_delta[row_index, valid]
        rows.append(
            {
                "schema_version": 1,
                "record_type": "rapid_k12_parameter_swap_trajectory",
                "execution_commit": commit,
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "slice_manifest_sha256": sha256_file(SLICE_MANIFEST_PATH),
                "example_id": record["example_id"],
                "concept": record["concept"],
                "split": record["split"],
                "parameter_state": state_id,
                "parameter_scope": state["scope"],
                "head_set": state["head_set"],
                "response_token_count": int(valid.sum()),
                "versus_chameleon": vector_comparison(candidate, baseline),
                "prototype_versus_hybrid": vector_comparison(prototype, candidate),
                "per_layer_versus_chameleon": {
                    str(layer): vector_comparison(
                        candidate[:, indices], baseline[:, indices]
                    )
                    for layer, indices in sorted(layer_indices.items())
                },
            }
        )
    return rows


def load_day46_reference(
    example_ids: set[str],
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result = {}
    for path in sorted(DAY46_SHARD_DIR.glob("*.json")):
        shard = read_json(path)
        for row in shard["rows"]:
            if row["example_id"] not in example_ids:
                continue
            key = (row["example_id"], row["direction"], row["candidate"])
            result[key] = row
    expected = len(example_ids) * 2 * 3
    if len(result) != expected:
        raise RuntimeError(
            f"Day 46 preflight reference has {len(result)} rows, expected {expected}"
        )
    return result


def run_preflight(
    runner: Any,
    vector: VectorizedMechanismRunner,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    batches: Sequence[tuple[int, str, int, list[Mapping[str, Any]]]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    precursor_slices: Mapping[str, torch.Tensor],
    contract: Mapping[str, Any],
    output_path: Path,
    commit: str,
) -> None:
    largest = max(batches, key=lambda item: len(item[3]))
    longest = max(
        batches,
        key=lambda item: max(int(row["response_token_count"]) for row in item[3]),
    )
    selected_batches = [largest] if largest[0] == longest[0] else [largest, longest]
    references = load_day46_reference(
        {row["example_id"] for batch in selected_batches for row in batch[3]}
    )
    maximum_reference_error = 0.0
    maximum_layer9_error = 0.0
    maximum_restore_error = 0.0
    state_checks = []
    representative_batches = []
    states = contract["parameter_states"]
    start_layer = int(contract["execution"]["tail_start_layer"])
    for _ordinal, _concept, scheduled_size, records in selected_batches:
        pair = prepare_pair(runner, records)
        prototype = prototype_delta_for_batch(
            prototypes, prototype_index, records, pair.normal.response_mask
        )
        baseline_normal = None
        baseline_triggered = None
        for state in states:
            with precursor_parameter_state(
                runner, state, contract, precursor_slices
            ) as swap_audit:
                normal, triggered = run_realized_pair(runner, pair)
                normal_values = selected_values(normal, component_ids, runner.layers)
                triggered_values = selected_values(triggered, component_ids, runner.layers)
                natural_delta = triggered_values - normal_values
                if state["id"] == "chameleon":
                    baseline_normal = normal.full_residuals[9].clone()
                    baseline_triggered = triggered.full_residuals[9].clone()
                    deltas = (
                        ("natural_hybrid_activity", natural_delta),
                        ("frozen_chameleon_prototype", prototype),
                    )
                    for direction, condition, target, target_values in (
                        ("induction", pair.normal, normal, normal_values),
                        ("rescue", pair.triggered, triggered, triggered_values),
                    ):
                        jobs = total_jobs(
                            target,
                            target_values,
                            deltas,
                            direction,
                            runner,
                            component_ids,
                        )
                        output = run_cached_tail(
                            vector,
                            condition,
                            target,
                            jobs,
                            start_layer=start_layer,
                        )
                        for output_index, (operator, _delta) in enumerate(deltas):
                            day46_candidate = (
                                "exact_natural_activity"
                                if operator == "natural_hybrid_activity"
                                else "concept_position_prototype"
                            )
                            for row_index, record in enumerate(records):
                                reference = references[
                                    (record["example_id"], direction, day46_candidate)
                                ]
                                maximum_reference_error = max(
                                    maximum_reference_error,
                                    float(
                                        (
                                            output.mean_margins[output_index, row_index]
                                            - torch.tensor(
                                                reference["mean_raw_margins"]
                                            )
                                        )
                                        .abs()
                                        .max()
                                    ),
                                )
                else:
                    if baseline_normal is None or baseline_triggered is None:
                        raise RuntimeError("Chameleon state must execute first")
                    maximum_layer9_error = max(
                        maximum_layer9_error,
                        layer9_input_error(
                            normal,
                            triggered,
                            baseline_normal,
                            baseline_triggered,
                        ),
                    )
                state_checks.append(
                    {
                        "state": state["id"],
                        "prototype_shape_exact": prototype.shape == natural_delta.shape,
                        "natural_finite": bool(torch.isfinite(natural_delta).all()),
                        "prototype_finite": bool(torch.isfinite(prototype).all()),
                        "response_ids_exact": torch.equal(
                            normal.response_ids, triggered.response_ids
                        ),
                        "response_masks_exact": torch.equal(
                            normal.response_mask, triggered.response_mask
                        ),
                        "hooks_removed_inside_state": runner.registered_hook_count() == 0,
                    }
                )
            maximum_restore_error = max(
                maximum_restore_error, float(swap_audit["restore_max_abs"])
            )
            if runner.registered_hook_count() != 0:
                raise RuntimeError("hooks remain after parameter state")
        representative_batches.append(
            {
                "scheduled": scheduled_size,
                "actual": len(records),
                "maximum_response_tokens": max(
                    int(row["response_token_count"]) for row in records
                ),
            }
        )
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "all_state_checks_pass": all(
            all(value for key, value in row.items() if key != "state")
            for row in state_checks
        ),
        "maximum_batch_exercised": any(
            row["scheduled"] == 16 and row["actual"] == 16
            for row in representative_batches
        ),
        "long_sequence_exercised": any(
            row["scheduled"] == 2 for row in representative_batches
        ),
        "baseline_matches_day46": maximum_reference_error
        <= float(gates["baseline_vs_day46_margin_max_abs"]),
        "layer9_inputs_bit_exact": maximum_layer9_error
        <= float(gates["layer9_input_max_abs_across_states"]),
        "parameters_restore_bit_exact": maximum_restore_error
        <= float(gates["parameter_restore_max_abs"]),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-parameter-swap-preflight-v1",
        "preflight_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "slice_manifest_sha256": sha256_file(SLICE_MANIFEST_PATH),
        "slice_tensor_sha256": sha256_file(SLICE_TENSOR_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "representative_batches": representative_batches,
        "state_checks": state_checks,
        "baseline_vs_day46_margin_max_abs": maximum_reference_error,
        "layer9_input_max_abs_across_states": maximum_layer9_error,
        "parameter_restore_max_abs": maximum_restore_error,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
        "candidate_metrics_generated": False,
    }
    write_json_atomic(output_path, report)
    if report["result"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_batch(
    runner: Any,
    vector: VectorizedMechanismRunner,
    records: Sequence[Mapping[str, Any]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    precursor_slices: Mapping[str, torch.Tensor],
    contract: Mapping[str, Any],
    probe_names: Sequence[str],
    probes: Sequence[Any],
    commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pair = prepare_pair(runner, records)
    prototype = prototype_delta_for_batch(
        prototypes, prototype_index, records, pair.normal.response_mask
    )
    endpoint_rows = []
    all_trajectory_rows = []
    state_audits = []
    baseline_delta = None
    baseline_normal_input = None
    baseline_triggered_input = None
    maximum_layer9_error = 0.0
    maximum_restore_error = 0.0
    start_layer = int(contract["execution"]["tail_start_layer"])
    for state in contract["parameter_states"]:
        with precursor_parameter_state(
            runner, state, contract, precursor_slices
        ) as swap_audit:
            normal, triggered = run_realized_pair(runner, pair)
            normal_values = selected_values(normal, component_ids, runner.layers)
            triggered_values = selected_values(triggered, component_ids, runner.layers)
            natural_delta = triggered_values - normal_values
            if state["id"] == "chameleon":
                baseline_delta = natural_delta.clone()
                baseline_normal_input = normal.full_residuals[9].clone()
                baseline_triggered_input = triggered.full_residuals[9].clone()
            else:
                if (
                    baseline_delta is None
                    or baseline_normal_input is None
                    or baseline_triggered_input is None
                ):
                    raise RuntimeError("Chameleon state must execute first")
                maximum_layer9_error = max(
                    maximum_layer9_error,
                    layer9_input_error(
                        normal,
                        triggered,
                        baseline_normal_input,
                        baseline_triggered_input,
                    ),
                )
            all_trajectory_rows.extend(
                trajectory_rows(
                    state["id"],
                    state,
                    records,
                    normal.response_mask,
                    natural_delta,
                    baseline_delta,
                    prototype,
                    component_ids,
                    commit,
                )
            )
            deltas = (
                ("natural_hybrid_activity", natural_delta),
                ("frozen_chameleon_prototype", prototype),
            )
            for direction, condition, target, target_values in (
                ("induction", pair.normal, normal, normal_values),
                ("rescue", pair.triggered, triggered, triggered_values),
            ):
                jobs = total_jobs(
                    target,
                    target_values,
                    deltas,
                    direction,
                    runner,
                    component_ids,
                )
                output = run_cached_tail(
                    vector,
                    condition,
                    target,
                    jobs,
                    start_layer=start_layer,
                )
                target_margins = mean_margins(target.monitor_residual, probes)
                target_rms = response_activation_rms(target.monitor_residual)
                for output_index, (operator, _delta) in enumerate(deltas):
                    endpoint_rows.extend(
                        total_endpoint_rows(
                            output,
                            output_index,
                            target,
                            target_margins,
                            target_rms,
                            probe_names,
                            records,
                            state_id=state["id"],
                            scope=state["scope"],
                            head_set=state["head_set"],
                            direction=direction,
                            operator=operator,
                            commit=commit,
                        )
                    )
            state_audits.append(
                {
                    "state": state["id"],
                    "applied_slice_count": len(swap_audit["applied_slices"]),
                    "all_natural_finite": bool(torch.isfinite(natural_delta).all()),
                    "all_prototype_finite": bool(torch.isfinite(prototype).all()),
                    "hooks_inside_state": runner.registered_hook_count(),
                }
            )
        maximum_restore_error = max(
            maximum_restore_error, float(swap_audit["restore_max_abs"])
        )
        if runner.registered_hook_count() != 0:
            raise RuntimeError("hooks remain after parameter state")
    expected_endpoints = (
        len(records)
        * len(contract["parameter_states"])
        * len(contract["directions"])
        * len(contract["causal_operators"])
    )
    expected_trajectories = len(records) * len(contract["parameter_states"])
    if len(endpoint_rows) != expected_endpoints:
        raise RuntimeError(
            f"expected {expected_endpoints} endpoint rows, produced {len(endpoint_rows)}"
        )
    if len(all_trajectory_rows) != expected_trajectories:
        raise RuntimeError(
            f"expected {expected_trajectories} trajectory rows, produced {len(all_trajectory_rows)}"
        )
    audit = {
        "concept": records[0]["concept"],
        "example_ids": [row["example_id"] for row in records],
        "endpoint_row_count": len(endpoint_rows),
        "trajectory_row_count": len(all_trajectory_rows),
        "state_audits": state_audits,
        "layer9_input_max_abs_across_states": maximum_layer9_error,
        "parameter_restore_max_abs": maximum_restore_error,
        "hooks_after_batch": runner.registered_hook_count(),
        "all_endpoint_rows_finite": all(
            all(math.isfinite(value) for value in row["mean_raw_margins"])
            and all(math.isfinite(value) for value in row["target_mean_raw_margins"])
            and math.isfinite(row["activation_rms"])
            and math.isfinite(row["target_activation_rms"])
            for row in endpoint_rows
        ),
        "all_trajectory_rows_finite": all(
            all(math.isfinite(value) for value in row["versus_chameleon"].values())
            and all(
                math.isfinite(value)
                for value in row["prototype_versus_hybrid"].values()
            )
            and all(
                math.isfinite(value)
                for layer in row["per_layer_versus_chameleon"].values()
                for value in layer.values()
            )
            for row in all_trajectory_rows
        ),
    }
    return endpoint_rows, all_trajectory_rows, audit


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        SLICE_MANIFEST_PATH,
        PROTOTYPE_MANIFEST_PATH,
        DAY45_CONTRACT_PATH,
        DAY46_CONTRACT_PATH,
        DAY46_SUMMARY_PATH,
        DAY46_AUDIT_PATH,
        DAY46_ARTIFACT_MANIFEST_PATH,
        PRECURSOR_FILE_MANIFEST_PATH,
        DAY45_RUNNER_PATH,
        DAY46_RUNNER_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    slice_manifest = read_json(SLICE_MANIFEST_PATH)
    prototype_manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    day45_contract = read_json(DAY45_CONTRACT_PATH)
    validate_parents(contract)
    if slice_manifest["contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("slice manifest contract hash differs")
    if slice_manifest["tensor_sha256"] != sha256_file(SLICE_TENSOR_PATH):
        raise RuntimeError("precursor slice tensor hash differs")
    if slice_manifest["result"] != "pass" or not all(
        slice_manifest["checks"].values()
    ):
        raise RuntimeError("precursor slice manifest does not pass")
    if prototype_manifest["tensor_sha256"] != sha256_file(PROTOTYPE_TENSOR_PATH):
        raise RuntimeError("prototype tensor hash differs")
    records = load_population_records(day45_contract, prototype_manifest)
    prototypes = load_file(PROTOTYPE_TENSOR_PATH)["prototype_delta"].float()
    prototype_index = {
        row["example_id"]: index
        for index, row in enumerate(prototype_manifest["examples"])
    }
    precursor_slices = load_file(SLICE_TENSOR_PATH)
    component_ids = tuple(prototype_manifest["component_ids"])
    batches = make_batches(records, contract)
    output_dir = args.output_dir.resolve()
    preflight_path = output_dir / "parameter-swap-preflight.json"
    runner = load_model()
    probe_names, probes = load_probes()
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    if args.preflight_only:
        run_preflight(
            runner,
            vector,
            probes,
            probe_names,
            batches,
            prototypes,
            prototype_index,
            component_ids,
            precursor_slices,
            contract,
            preflight_path,
            commit,
        )
        return
    if not preflight_path.exists():
        raise RuntimeError("passing parameter-swap preflight is required")
    preflight = read_json(preflight_path)
    if (
        preflight.get("result") != "pass"
        or preflight.get("preflight_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or preflight.get("slice_manifest_sha256") != sha256_file(SLICE_MANIFEST_PATH)
        or preflight.get("slice_tensor_sha256") != sha256_file(SLICE_TENSOR_PATH)
    ):
        raise RuntimeError("parameter-swap preflight does not pass for this execution")

    selected_concepts = set(args.concept or [])
    all_concepts = {row["concept"] for row in records}
    unknown = selected_concepts - all_concepts
    if unknown:
        raise RuntimeError(f"unknown requested concepts: {sorted(unknown)}")
    selected_batches = [
        batch for batch in batches if not selected_concepts or batch[1] in selected_concepts
    ]
    shard_dir = output_dir / "parameter-swap-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    execution_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completed = []
    endpoint_count = 0
    trajectory_count = 0
    for ordinal, concept, scheduled_size, batch_records in selected_batches:
        safe_concept = concept.replace("/", "-")
        shard_path = shard_dir / f"{ordinal:04d}-{safe_concept}.json"
        expected_endpoints = (
            len(batch_records)
            * len(contract["parameter_states"])
            * len(contract["directions"])
            * len(contract["causal_operators"])
        )
        expected_trajectories = len(batch_records) * len(contract["parameter_states"])
        if shard_path.exists():
            existing = read_json(shard_path)
            if (
                existing.get("execution_commit") == commit
                and existing.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and existing.get("slice_manifest_sha256")
                == sha256_file(SLICE_MANIFEST_PATH)
                and existing.get("audit", {}).get("endpoint_row_count")
                == expected_endpoints
                and existing.get("audit", {}).get("trajectory_row_count")
                == expected_trajectories
            ):
                completed.append(str(shard_path.relative_to(ROOT)))
                endpoint_count += expected_endpoints
                trajectory_count += expected_trajectories
                print(f"Skipping complete swap shard {ordinal:04d}.", flush=True)
                continue
            raise RuntimeError(f"existing shard has incompatible provenance: {shard_path}")
        endpoints, trajectories, audit = run_batch(
            runner,
            vector,
            batch_records,
            prototypes,
            prototype_index,
            component_ids,
            precursor_slices,
            contract,
            probe_names,
            probes,
            commit,
        )
        shard = {
            "schema_version": 1,
            "procedure": "rapid-k12-selected-parameter-swap-shard-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "slice_manifest_sha256": sha256_file(SLICE_MANIFEST_PATH),
            "slice_tensor_sha256": sha256_file(SLICE_TENSOR_PATH),
            "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
            "batch_ordinal": ordinal,
            "scheduled_batch_size": scheduled_size,
            "concept": concept,
            "endpoint_rows": endpoints,
            "trajectory_rows": trajectories,
            "audit": audit,
        }
        write_json_atomic(shard_path, shard)
        completed.append(str(shard_path.relative_to(ROOT)))
        endpoint_count += len(endpoints)
        trajectory_count += len(trajectories)
        print(
            f"Completed swap shard {ordinal + 1}/{len(batches)} {concept}: "
            f"{len(batch_records)} examples",
            flush=True,
        )
    torch.cuda.synchronize()
    full_population = not selected_concepts
    expected_endpoints = (
        int(contract["implementation_gates"]["exact_endpoint_row_count"])
        if full_population
        else sum(
            len(batch[3])
            * len(contract["parameter_states"])
            * len(contract["directions"])
            * len(contract["causal_operators"])
            for batch in selected_batches
        )
    )
    expected_trajectories = (
        int(contract["implementation_gates"]["exact_trajectory_row_count"])
        if full_population
        else sum(
            len(batch[3]) * len(contract["parameter_states"])
            for batch in selected_batches
        )
    )
    if endpoint_count != expected_endpoints or trajectory_count != expected_trajectories:
        raise RuntimeError("parameter-swap execution row counts differ")
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-parameter-swap-execution-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "slice_manifest_sha256": sha256_file(SLICE_MANIFEST_PATH),
        "slice_tensor_sha256": sha256_file(SLICE_TENSOR_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "full_population": full_population,
        "concepts_completed": sorted({batch[1] for batch in selected_batches}),
        "batch_count": len(selected_batches),
        "endpoint_row_count": endpoint_count,
        "trajectory_row_count": trajectory_count,
        "shards": completed,
        "elapsed_seconds": time.perf_counter() - execution_started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "hooks_after_execution": runner.registered_hook_count(),
        "result": "complete",
    }
    write_json_atomic(output_dir / "parameter-swap-execution.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
