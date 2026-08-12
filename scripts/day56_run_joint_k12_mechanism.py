#!/usr/bin/env python3
"""Run the frozen Day 56 joint-K12 mathematical-mechanism experiment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import (  # noqa: E402
    load_model,
    load_probes,
    selected_values,
    sha256_file,
)
from day45_run_prototype_population import (  # noqa: E402
    require_committed,
    write_json_atomic,
)
from day48_run_proximal_upstream import (  # noqa: E402
    grouped_batches,
    prepare_conditions,
    selected_records,
)
from day49_run_prompt_memory import make_job, natural_payload, result_payload  # noqa: E402
from day52_run_reciprocal_reconfiguration import (  # noqa: E402
    partitions_for_conditions,
)
from day54_run_exact_donor_k12 import expanded_contract as day54_contract  # noqa: E402
from neural_chameleon import RealizedForwardRunner, VectorizedUpstreamRunner  # noqa: E402
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.controller_actuator import SourceRegion  # noqa: E402
from neural_chameleon.joint_k12_mechanism import (  # noqa: E402
    JointK12JacobianRunner,
    capture_rmsnorm_denominators,
    factorize_joint_attention,
    frozen_rmsnorm_denominators,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    mean_replacements,
    rotate_head_delta,
)


CONTRACT_PATH = ROOT / "results/day-56/frozen-joint-k12-mechanism-contract.json"
DAY54_CONTRACT_PATH = ROOT / "results/day-54/frozen-exact-donor-k12-contract.json"
DAY54_SUMMARY_PATH = ROOT / "results/day-54/exact-donor-k12-summary.json"
DAY55_SUMMARY_PATH = ROOT / "results/day-55/qkv-completion-summary.json"
DAY44_SUMMARY_PATH = ROOT / "results/day-44/offline-screen-summary.json"
DAY47_SUMMARY_PATH = ROOT / "results/day-47/heldout-summary.json"
PREFLIGHT_PATH = ROOT / "results/day-56/joint-k12-mechanism-preflight.json"
EXECUTION_PATH = ROOT / "results/day-56/joint-k12-mechanism-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day56-joint-mechanism-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--concept", action="append")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def expanded_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    parent = day54_contract()
    expected_hashes = {
        DAY54_CONTRACT_PATH: contract["parents"]["day54_contract_sha256"],
        DAY54_SUMMARY_PATH: contract["parents"]["day54_summary_sha256"],
        DAY55_SUMMARY_PATH: contract["parents"]["day55_summary_sha256"],
        DAY44_SUMMARY_PATH: contract["parents"]["day44_offline_screen_sha256"],
        DAY47_SUMMARY_PATH: contract["parents"]["day47_heldout_summary_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Day 56 parent hash differs: {path}")
    for field in ("model", "population", "conditions", "k12", "probes"):
        contract[field] = parent[field]
    contract["execution"] = {
        "concept_shards": 13,
        "batch_size": 2,
        "natural_states": [
            "normal",
            "correct_trigger",
            "irrelevant_trigger",
            "different_trigger",
        ],
        "free_jobs_per_direction": 30,
        "frozen_norm_jobs_per_direction": 6,
        "states_per_example": 76,
        "total_state_rows": 1976,
    }
    return contract


def components_by_layer(
    component_ids: Sequence[str],
) -> dict[int, tuple[tuple[int, MechanismComponent], ...]]:
    result: dict[int, list[tuple[int, MechanismComponent]]] = defaultdict(list)
    for index, component_id in enumerate(component_ids):
        component = MechanismComponent.parse(component_id)
        result[component.layer].append((index, component))
    return {layer: tuple(values) for layer, values in sorted(result.items())}


def free_job_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    names = list(contract["free_jobs"]["base"])
    for candidate in contract["candidate_simplicity_order"]:
        names.extend(
            [
                f"candidate.{candidate}.install",
                f"candidate.{candidate}.remove",
                f"candidate.{candidate}.haar_install",
            ]
        )
    names.extend(
        f"atomic.{region}.install" for region in contract["atomic_source_regions"]
    )
    expected = int(contract["free_jobs"]["jobs_per_direction"])
    if len(names) != expected or len(set(names)) != expected:
        raise RuntimeError("Day 56 free job matrix differs from the freeze")
    return tuple(names)


def frozen_job_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(contract["frozen_norm_jobs"])
    if len(names) != 6 or len(set(names)) != 6:
        raise RuntimeError("Day 56 frozen-norm job matrix differs")
    return names


def _blank_like(values: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(values, dtype=torch.float32)


def build_joint_algebra(
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, torch.Tensor]]:
    specification = contract["directions"][direction]
    target_name = specification["target"]
    donor_name = specification["donor"]
    target_values = selected_values(
        captures[target_name], component_ids, runner.layers
    ).float()
    donor_values = selected_values(
        captures[donor_name], component_ids, runner.layers
    ).float()
    tensors = {
        "routing_shapley": _blank_like(target_values),
        "content_shapley": _blank_like(target_values),
        "routing_hybrid": _blank_like(target_values),
        "content_hybrid": _blank_like(target_values),
        **{
            f"region.{region}": _blank_like(target_values)
            for region in contract["atomic_source_regions"]
        },
    }
    target_masses = {
        region: torch.zeros_like(target_values[..., 0])
        for region in contract["atomic_source_regions"]
    }
    donor_masses = {
        region: torch.zeros_like(target_values[..., 0])
        for region in contract["atomic_source_regions"]
    }
    audit_rows = []
    by_layer = components_by_layer(component_ids)
    for layer, entries in by_layer.items():
        indices = [index for index, _component in entries]
        heads = [int(component.head) for _index, component in entries]
        target_masks = {
            region: partitions[target_name].masks[SourceRegion(region)]
            for region in contract["atomic_source_regions"]
        }
        donor_masks = {
            region: partitions[donor_name].masks[SourceRegion(region)]
            for region in contract["atomic_source_regions"]
        }
        factor = factorize_joint_attention(
            attention_states[target_name][layer],
            attention_states[donor_name][layer],
            target_masks,
            donor_masks,
            heads,
        )
        tensors["routing_shapley"][:, :, indices] = factor.routing_shapley_delta
        tensors["content_shapley"][:, :, indices] = factor.content_shapley_delta
        tensors["routing_hybrid"][:, :, indices] = factor.routing_hybrid
        tensors["content_hybrid"][:, :, indices] = factor.content_hybrid
        for region in contract["atomic_source_regions"]:
            tensors[f"region.{region}"][:, :, indices] = factor.region_deltas[region]
            target_masses[region][:, :, indices] = factor.target_masses[region]
            donor_masses[region][:, :, indices] = factor.donor_masses[region]
        audit_rows.append(
            {
                "layer": layer,
                "target_reconstruction_max_abs": factor.target_reconstruction_max_abs,
                "donor_reconstruction_max_abs": factor.donor_reconstruction_max_abs,
                "shapley_closure_max_abs": factor.shapley_closure_max_abs,
            }
        )
    candidate_deltas = {
        "routing_shapley": tensors["routing_shapley"],
        "content_shapley": tensors["content_shapley"],
    }
    for group, regions in contract["source_groups"].items():
        candidate_deltas[group] = sum(tensors[f"region.{region}"] for region in regions)
    exact_delta = donor_values - target_values
    response_mask = captures[target_name].response_mask[:, :, None, None].bool()
    decomposed_delta = tensors["routing_shapley"] + tensors["content_shapley"]
    raw_closure = float(
        (decomposed_delta - exact_delta)
        .abs()
        .masked_select(response_mask.expand_as(exact_delta))
        .max()
    )
    return (
        {
            "target": target_values,
            "donor": donor_values,
            "exact_delta": exact_delta,
            "routing_hybrid": tensors["routing_hybrid"],
            "content_hybrid": tensors["content_hybrid"],
            **{f"candidate.{name}": value for name, value in candidate_deltas.items()},
            **{
                f"atomic.{region}": tensors[f"region.{region}"]
                for region in contract["atomic_source_regions"]
            },
        },
        {
            "layers": audit_rows,
            "attention_reconstruction_max_abs": max(
                max(
                    row["target_reconstruction_max_abs"],
                    row["donor_reconstruction_max_abs"],
                )
                for row in audit_rows
            ),
            "shapley_closure_max_abs": max(
                row["shapley_closure_max_abs"] for row in audit_rows
            ),
            "decomposition_to_raw_exact_delta_max_abs_diagnostic": raw_closure,
        },
        {
            **{
                f"target_mass.{region}": value
                for region, value in target_masses.items()
            },
            **{f"donor_mass.{region}": value for region, value in donor_masses.items()},
        },
    )


def build_jobs(
    algebra: Mapping[str, torch.Tensor],
    target_capture: Any,
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    target = algebra["target"]
    donor = algebra["donor"]
    delta = algebra["exact_delta"]
    states: dict[str, torch.Tensor] = {
        "identity_target": target,
        "exact_donor_all": donor,
        "dose_0.25": target + 0.25 * delta,
        "dose_0.50": target + 0.50 * delta,
        "dose_0.75": target + 0.75 * delta,
        "dose_1.25": target + 1.25 * delta,
        "routing_hybrid": algebra["routing_hybrid"],
        "content_hybrid": algebra["content_hybrid"],
    }
    exact_haar, exact_audit = rotate_head_delta(
        delta,
        draw_index=int(contract["controls"]["haar_draw_index"]),
        base_seed=int(contract["controls"]["haar_base_seed"]),
    )
    states["exact_delta_haar"] = target + exact_haar
    audits: dict[str, Any] = {"exact_delta": exact_audit.to_dict()}
    for candidate_index, candidate in enumerate(contract["candidate_simplicity_order"]):
        candidate_delta = algebra[f"candidate.{candidate}"]
        rotated, audit = rotate_head_delta(
            candidate_delta,
            draw_index=int(contract["controls"]["haar_draw_index"]),
            base_seed=int(contract["controls"]["haar_base_seed"])
            + 100 * (candidate_index + 1),
        )
        states[f"candidate.{candidate}.install"] = target + candidate_delta
        states[f"candidate.{candidate}.remove"] = donor - candidate_delta
        states[f"candidate.{candidate}.haar_install"] = target + rotated
        audits[candidate] = audit.to_dict()
    for region in contract["atomic_source_regions"]:
        states[f"atomic.{region}.install"] = target + algebra[f"atomic.{region}"]
    if set(states) != set(free_job_names(contract)):
        raise RuntimeError("Day 56 constructed free states differ from the freeze")
    replacements = {
        name: mean_replacements(target_capture, component_ids, value, runner.layers)
        for name, value in states.items()
    }
    free_jobs = [
        make_job(name, target_capture, replacements[name], runner)
        for name in free_job_names(contract)
    ]
    frozen_jobs = [
        make_job(name, target_capture, replacements[name], runner)
        for name in frozen_job_names(contract)
    ]
    return free_jobs, frozen_jobs, audits


def write_shard(
    concept: str,
    batch: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    states: Mapping[str, Mapping[str, torch.Tensor]],
    diagnostics: Mapping[str, torch.Tensor],
    algebra_audits: Mapping[str, Any],
    random_audits: Mapping[str, Any],
    partition_counts: Mapping[str, Any],
    probe_names: Sequence[str],
    commit: str,
    contract: Mapping[str, Any],
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = concept.replace("/", "_")
    tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
    metadata_path = SHARD_DIR / f"{safe_name}.json"
    tensors = {
        f"{state_name}.{field}": value.detach().cpu().contiguous().clone()
        for state_name, payload in states.items()
        for field, value in payload.items()
    }
    tensors.update(
        {
            name: value.detach().cpu().contiguous().clone()
            for name, value in diagnostics.items()
        }
    )
    tensors["response_mask"] = response_mask.detach().cpu().contiguous().clone()
    if not all(
        torch.isfinite(value).all()
        for key, value in tensors.items()
        if key != "response_mask"
    ):
        raise RuntimeError("Day 56 shard contains a nonfinite tensor")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    metadata = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "concept": concept,
        "example_ids": [row["example_id"] for row in batch],
        "response_token_counts": [int(row["response_token_count"]) for row in batch],
        "probe_names": list(probe_names),
        "candidate_order": list(contract["candidate_simplicity_order"]),
        "state_names": sorted(states),
        "state_count": len(states),
        "diagnostic_tensor_names": sorted(diagnostics),
        "algebra_audits": dict(algebra_audits),
        "random_audits": dict(random_audits),
        "partition_counts": dict(partition_counts),
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


def run_preflight(
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    attention_states = {
        name: attention.run(condition, contract["k12"]["layers"])
        for name, condition in conditions.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    algebra_audits = {}
    identity_errors = {}
    frozen_errors = {}
    response_checks = []
    finite_checks = []
    for direction, specification in contract["directions"].items():
        algebra, audit, _mass = build_joint_algebra(
            direction,
            captures,
            attention_states,
            partitions,
            component_ids,
            contract,
            runner,
        )
        algebra_audits[direction] = audit
        target_name = specification["target"]
        target_capture = captures[target_name]
        identity_replacements = mean_replacements(
            target_capture, component_ids, algebra["target"], runner.layers
        )
        jobs = [
            make_job(f"identity_{index}", target_capture, identity_replacements, runner)
            for index in range(5)
        ]
        output = vector.run(conditions[target_name], jobs)
        identity_errors[direction] = {
            "k12_mutual_max_abs": float(
                (output.k12 - output.k12[0].unsqueeze(0)).abs().max()
            ),
            "margin_mutual_max_abs": float(
                (output.mean_margins - output.mean_margins[0].unsqueeze(0)).abs().max()
            ),
        }
        denominators = capture_rmsnorm_denominators(
            runner, conditions[target_name], contract["k12"]["layers"]
        )
        with frozen_rmsnorm_denominators(
            runner, denominators, contract["k12"]["layers"], repeats=5
        ):
            frozen = vector.run(conditions[target_name], jobs)
        frozen_errors[direction] = {
            "k12_mutual_max_abs": float(
                (frozen.k12 - frozen.k12[0].unsqueeze(0)).abs().max()
            ),
            "margin_to_free_identity_max_abs": float(
                (frozen.mean_margins - output.mean_margins[0].unsqueeze(0)).abs().max()
            ),
        }
        response_checks.extend(
            [
                torch.equal(result.response_ids, conditions[target_name].response_ids)
                and torch.equal(
                    result.response_mask, conditions[target_name].response_mask
                )
                for result in (output, frozen)
            ]
        )
        finite_checks.extend(
            torch.isfinite(value).all()
            for result in (output, frozen)
            for value in (result.k12, result.monitor_values, result.mean_margins)
        )
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "attention_reconstruction": max(
            audit["attention_reconstruction_max_abs"]
            for audit in algebra_audits.values()
        )
        <= float(gates["attention_reconstruction_max_abs"]),
        "shapley_closure": max(
            audit["shapley_closure_max_abs"] for audit in algebra_audits.values()
        )
        <= float(gates["shapley_closure_max_abs"]),
        "identity_k12": max(
            value["k12_mutual_max_abs"] for value in identity_errors.values()
        )
        <= float(gates["identity_k12_max_abs"]),
        "identity_margins": max(
            value["margin_mutual_max_abs"] for value in identity_errors.values()
        )
        <= float(gates["identity_monitor_margin_max_abs"]),
        "frozen_norm_identity_k12": max(
            value["k12_mutual_max_abs"] for value in frozen_errors.values()
        )
        <= float(gates["identity_k12_max_abs"]),
        "frozen_norm_identity_margins": max(
            value["margin_to_free_identity_max_abs"] for value in frozen_errors.values()
        )
        <= float(gates["frozen_norm_identity_monitor_margin_max_abs"]),
        "response_ids_and_masks_exact": all(response_checks),
        "all_values_finite": all(bool(value) for value in finite_checks),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-preflight",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "algebra_audits": algebra_audits,
        "identity_errors": identity_errors,
        "frozen_norm_identity_errors": frozen_errors,
        "checks": checks,
        "candidate_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 56 preflight failed: {report}")


def run_batch(
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    jacobian: JointK12JacobianRunner,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    attention_states = {
        name: attention.run(condition, contract["k12"]["layers"])
        for name, condition in conditions.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    outputs = {}
    frozen_outputs = {}
    algebra_audits = {}
    random_audits = {}
    diagnostics: dict[str, torch.Tensor] = {}
    for direction, specification in contract["directions"].items():
        algebra, algebra_audit, mass_tensors = build_joint_algebra(
            direction,
            captures,
            attention_states,
            partitions,
            component_ids,
            contract,
            runner,
        )
        algebra_audits[direction] = algebra_audit
        target_name = specification["target"]
        free_jobs, norm_jobs, audits = build_jobs(
            algebra,
            captures[target_name],
            component_ids,
            contract,
            runner,
        )
        outputs[direction] = vector.run(conditions[target_name], free_jobs)
        denominators = capture_rmsnorm_denominators(
            runner, conditions[target_name], contract["k12"]["layers"]
        )
        with frozen_rmsnorm_denominators(
            runner,
            denominators,
            contract["k12"]["layers"],
            repeats=len(norm_jobs),
        ):
            frozen_outputs[direction] = vector.run(conditions[target_name], norm_jobs)
        candidate_deltas = torch.stack(
            [
                algebra[f"candidate.{candidate}"]
                for candidate in contract["candidate_simplicity_order"]
            ]
        )
        jacobian_summary = jacobian.run(
            conditions[target_name],
            algebra["target"],
            algebra["exact_delta"],
            candidate_deltas,
        )
        for name, value in mass_tensors.items():
            diagnostics[f"attention.{direction}.{name}"] = value
        diagnostics[f"jacobian.{direction}.target_margins"] = (
            jacobian_summary.target_margins
        )
        diagnostics[f"jacobian.{direction}.output_gram"] = jacobian_summary.output_gram
        diagnostics[f"jacobian.{direction}.singular_values"] = (
            jacobian_summary.singular_values
        )
        diagnostics[f"jacobian.{direction}.exact_delta_prediction"] = (
            jacobian_summary.exact_delta_prediction
        )
        diagnostics[f"jacobian.{direction}.candidate_delta_predictions"] = (
            jacobian_summary.candidate_delta_predictions
        )
        random_audits[direction] = audits
    states = {
        "natural_normal": natural_payload(
            captures["normal"], component_ids, runner, probes
        ),
        "natural_different_trigger": natural_payload(
            captures["different_trigger"], component_ids, runner, probes
        ),
        "natural_correct_trigger": result_payload(
            outputs["irrelevant_to_correct"],
            outputs["irrelevant_to_correct"].group_ids.index("identity_target"),
        ),
        "natural_irrelevant_trigger": result_payload(
            outputs["correct_to_irrelevant"],
            outputs["correct_to_irrelevant"].group_ids.index("identity_target"),
        ),
    }
    states.update(
        {
            f"free_{direction}.{job}": result_payload(output, index)
            for direction, output in outputs.items()
            for index, job in enumerate(output.group_ids)
        }
    )
    states.update(
        {
            f"frozen_norm_{direction}.{job}": result_payload(output, index)
            for direction, output in frozen_outputs.items()
            for index, job in enumerate(output.group_ids)
        }
    )
    if len(states) != int(contract["execution"]["states_per_example"]):
        raise RuntimeError("Day 56 state matrix differs from the freeze")
    write_shard(
        batch[0]["concept"],
        batch,
        conditions["normal"].response_mask,
        states,
        diagnostics,
        algebra_audits,
        random_audits,
        {
            name: [dict(row) for row in partition.assigned_prompt_counts]
            for name, partition in partitions.items()
        },
        probe_names,
        commit,
        contract,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 56 batch leaked hooks")
    return random_audits


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/joint_k12_mechanism.py",
        ROOT / "scripts/day54_run_exact_donor_k12.py",
        CONTRACT_PATH,
        DAY54_CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = expanded_contract()
    if contract["status"] != "frozen_before_day56_joint_mechanism_outcomes":
        raise RuntimeError("Day 56 contract is not frozen")
    day54 = read_json(DAY54_SUMMARY_PATH)
    day55 = read_json(DAY55_SUMMARY_PATH)
    if (
        day54.get("branch") != "exact_k12_self_contained_pass"
        or day55.get("localization_disposition")
        != "missing_monitor_sensitive_component_distributed"
    ):
        raise RuntimeError("Day 56 is not eligible from its parent results")
    records = selected_records(contract)
    batches = grouped_batches(records)
    if args.concept:
        requested = set(args.concept)
        batches = [batch for batch in batches if batch[0]["concept"] in requested]
        if {batch[0]["concept"] for batch in batches} != requested:
            raise RuntimeError("one or more requested concepts are not frozen")
    runner = load_model()
    probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    jacobian = JointK12JacobianRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        batch = grouped_batches(records)[0]
        run_preflight(
            runner,
            realized,
            attention,
            vector,
            component_ids,
            batch,
            contract["conditions"]["pairs"][batch[0]["concept"]],
            contract,
            commit,
        )
        return
    preflight = read_json(PREFLIGHT_PATH)
    if (
        preflight.get("result") != "pass"
        or preflight.get("execution_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 56 preflight is not exact and passing")
    started = time.perf_counter()
    completed = 0
    random_audits = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for batch in batches:
        concept = batch[0]["concept"]
        metadata_path = SHARD_DIR / f"{concept}.json"
        tensor_path = SHARD_DIR / f"{concept}.safetensors"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if (
                metadata.get("execution_commit") == commit
                and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and metadata.get("tensor_sha256") == sha256_file(tensor_path)
            ):
                completed += 1
                random_audits.extend(metadata["random_audits"].values())
                continue
        audits = run_batch(
            runner,
            realized,
            attention,
            vector,
            jacobian,
            probes,
            probe_names,
            component_ids,
            batch,
            contract["conditions"]["pairs"][concept],
            contract,
            commit,
        )
        random_audits.extend(audits.values())
        completed += 1
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps({"completed_concepts": completed, "concept": concept}),
            flush=True,
        )
    metadata_rows = []
    for path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(path)
        if metadata.get("execution_commit") == commit and metadata.get(
            "contract_sha256"
        ) == sha256_file(CONTRACT_PATH):
            metadata_rows.append(metadata)
    state_rows = sum(
        int(row["state_count"]) * len(row["example_ids"]) for row in metadata_rows
    )
    all_random_pass = all(
        bool(audit["pass"])
        for direction in random_audits
        for audit in direction.values()
    )
    execution = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "device": str(runner.device),
        "concept_shards": len(metadata_rows),
        "state_rows": state_rows,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "random_audits_pass": all_random_pass,
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": len(metadata_rows)
        == int(contract["implementation_gates"]["concept_shards"])
        and state_rows == int(contract["implementation_gates"]["total_state_rows"]),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if (
        not execution["complete"]
        or not execution["random_audits_pass"]
        or execution["hooks_after_execution"] != 0
    ):
        raise RuntimeError(f"Day 56 execution failed: {execution}")


if __name__ == "__main__":
    main()
