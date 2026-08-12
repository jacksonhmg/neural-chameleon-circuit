#!/usr/bin/env python3
"""Run the frozen Day 57 confirmation, tracing, and precursor stages on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import load_probes, selected_values  # noqa: E402
from day45_run_prototype_population import (  # noqa: E402
    mean_margins,
    require_committed,
    write_json_atomic,
)
from day48_run_proximal_upstream import prepare_conditions  # noqa: E402
from day49_run_prompt_memory import make_job  # noqa: E402
from day52_run_reciprocal_reconfiguration import (  # noqa: E402
    partitions_for_conditions,
)
from day56_run_joint_k12_mechanism import build_joint_algebra  # noqa: E402
from neural_chameleon import (  # noqa: E402
    PairedInterventionRunner,
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
    directional_recovery,
    prompt_memory_operation,
    prompt_qk_operation,
    prompt_value_operation,
    response_query_operation,
)
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.controller_actuator import SourceRegion  # noqa: E402
from neural_chameleon.joint_k12_mechanism import (  # noqa: E402
    capture_rmsnorm_denominators,
    frozen_rmsnorm_denominators,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    mean_replacements,
    rotate_head_delta,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
DAY56_CONTRACT_PATH = ROOT / "results/day-56/frozen-joint-k12-mechanism-contract.json"
DAY56_SUMMARY_PATH = ROOT / "results/day-56/joint-k12-mechanism-summary.json"
PANEL_MANIFEST_PATH = ROOT / "data/splits/day57-v1/manifest.json"
SUMMARY_PATHS = {
    "confirmation": ROOT / "results/day-57/confirmation-summary.json",
    "trace": ROOT / "results/day-57/value-pathway-summary.json",
    "precursor": ROOT / "results/day-57/precursor-acquisition-summary.json",
}
MODEL_KEYS = {"confirmation": "chameleon", "trace": "chameleon", "precursor": "exact_precursor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=tuple(MODEL_KEYS), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--concept", action="append")
    parser.add_argument("--batch-index", type=int, action="append")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def expanded_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    expected = {
        DAY56_CONTRACT_PATH: contract["parents"]["day56_contract_sha256"],
        DAY56_SUMMARY_PATH: contract["parents"]["day56_summary_sha256"],
        PANEL_MANIFEST_PATH: contract["parents"]["day57_panel_manifest_sha256"],
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"Day 57 parent differs: {path}")
    if contract["status"] != "frozen_before_any_day57_model_or_intervention_outcome":
        raise RuntimeError("Day 57 contract is not frozen")
    return contract


def algebra_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "directions": contract["conditions"]["directions"],
        "atomic_source_regions": [
            "monitoring_language",
            "named_concept",
            "trigger_other",
            "original_prompt",
            "template",
            "response",
        ],
        "source_groups": {
            "monitoring_prefix_delta": [
                "monitoring_language",
                "named_concept",
                "trigger_other",
            ],
            "prompt_context_delta": ["original_prompt", "template"],
            "response_history_delta": ["response"],
        },
    }


def load_runner(contract: Mapping[str, Any], model_key: str) -> PairedInterventionRunner:
    torch.manual_seed(57001)
    spec = contract["models"][model_key]
    model_path = ROOT / spec["local_path"]
    if sha256_file(model_path / "tokenizer.json") != spec["tokenizer_sha256"]:
        raise RuntimeError(f"{model_key} tokenizer differs")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    runner = PairedInterventionRunner(model, tokenizer)
    if runner.device.type != "cuda":
        raise RuntimeError(f"Day 57 requires CUDA, found {runner.device}")
    return runner


def panel_spec(contract: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    return contract["population"][
        "fresh_confirmation" if stage == "confirmation" else "path_tracing_and_precursor"
    ]


def load_records(contract: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    spec = panel_spec(contract, stage)
    path = ROOT / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"Day 57 {stage} panel differs")
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != int(spec["examples"]):
        raise RuntimeError(f"Day 57 {stage} panel count differs")
    if len({row["example_id"] for row in records}) != len(records):
        raise RuntimeError(f"Day 57 {stage} panel contains duplicate IDs")
    return records


def grouped_batches(
    records: Sequence[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_concept[row["concept"]].append(row)
    batches: list[list[dict[str, Any]]] = []
    for concept in sorted(by_concept):
        values = sorted(by_concept[concept], key=lambda row: row["example_id"])
        if len(values) % batch_size:
            raise RuntimeError(f"{concept} is not divisible by frozen batch size")
        batches.extend(
            [values[start : start + batch_size] for start in range(0, len(values), batch_size)]
        )
    return batches


def components_by_layer(
    component_ids: Sequence[str],
) -> dict[int, tuple[MechanismComponent, ...]]:
    result: dict[int, list[MechanismComponent]] = defaultdict(list)
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        if component.kind != "head" or component.head is None:
            raise ValueError("Day 57 K12 components must be heads")
        result[component.layer].append(component)
    return {layer: tuple(values) for layer, values in sorted(result.items())}


def union_prefix_mask(partition: Any) -> torch.Tensor:
    result = torch.zeros_like(
        partition.masks[SourceRegion.MONITORING_LANGUAGE], dtype=torch.bool
    )
    for region in (
        SourceRegion.MONITORING_LANGUAGE,
        SourceRegion.NAMED_CONCEPT,
        SourceRegion.TRIGGER_OTHER,
    ):
        result |= partition.masks[region].bool()
    if not bool(result.any(dim=1).all()):
        raise RuntimeError("monitoring-prefix mask is empty")
    return result


def masked_norm(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.bool()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    dimensions = tuple(range(1, values.ndim))
    return torch.where(expanded, values.float(), 0.0).square().sum(dim=dimensions).sqrt()


def compact_natural(
    capture: Any,
    target_k12: torch.Tensor,
    donor_k12: torch.Tensor,
    component_ids: Sequence[str],
    runner: Any,
    probes: Sequence[Any],
) -> dict[str, torch.Tensor]:
    values = selected_values(capture, component_ids, runner.layers).float()
    mask = capture.response_mask
    return {
        "margins": mean_margins(capture.monitor_residual, probes).T.float(),
        "k12_recovery": directional_recovery(
            values - target_k12, donor_k12 - target_k12, mask
        ),
        "k12_residual_norm_ratio": masked_norm(values - donor_k12, mask)
        / masked_norm(donor_k12 - target_k12, mask).clamp(min=1e-8),
        "k12_effect_norm": masked_norm(values - target_k12, mask),
    }


def compact_result(
    output: Any,
    index: int,
    target_k12: torch.Tensor,
    donor_k12: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    values = output.k12[index].float()
    return {
        "margins": output.mean_margins[index].float(),
        "k12_recovery": directional_recovery(
            values - target_k12, donor_k12 - target_k12, mask
        ),
        "k12_residual_norm_ratio": masked_norm(values - donor_k12, mask)
        / masked_norm(donor_k12 - target_k12, mask).clamp(min=1e-8),
        "k12_effect_norm": masked_norm(values - target_k12, mask),
    }


def attention_recompute_errors(
    states: Mapping[int, Any], component_ids: Sequence[str]
) -> dict[str, float]:
    errors = {}
    for layer, components in components_by_layer(component_ids).items():
        heads = tuple(int(component.head) for component in components)
        state = states[layer]
        recomputed = response_query_operation(state, state, heads)
        natural = state.raw_head_output[
            :,
            state.response_start : state.response_start + state.response_mask.shape[1],
        ].float()
        errors[str(layer)] = max(
            float((recomputed[:, :, head] - natural[:, :, head]).abs().max())
            for head in heads
        )
    return errors


def path_replacements(
    operation: str,
    source_name: str,
    target_name: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    runner: Any,
) -> dict[str, torch.Tensor]:
    target_capture = captures[target_name]
    replacements = source_replacements(
        target_capture, target_capture, component_ids, runner.layers
    )
    layer_filter = None
    if operation.startswith("v_layer_"):
        layer_filter = int(operation.removeprefix("v_layer_"))
        operation_kind = "v_prefix"
    else:
        operation_kind = operation
    for layer, components in components_by_layer(component_ids).items():
        if layer_filter is not None and layer != layer_filter:
            continue
        heads = tuple(int(component.head) for component in components)
        source = attention_states[source_name][layer]
        target = attention_states[target_name][layer]
        source_mask = union_prefix_mask(partitions[source_name])
        target_mask = union_prefix_mask(partitions[target_name])
        if operation_kind == "v_prefix":
            changed = prompt_value_operation(source, target, heads, source_mask, target_mask)
        elif operation_kind == "qk_prefix":
            changed = prompt_qk_operation(source, target, heads, source_mask, target_mask)
        elif operation_kind == "qkv_prefix":
            changed = prompt_memory_operation(
                source,
                target,
                heads,
                source_mask,
                target_mask,
                include_source_query=True,
            )
        else:
            raise ValueError(f"unknown Day 57 pathway operation: {operation}")
        replacements.update(
            {
                component.component_id: changed[:, :, int(component.head)].clone()
                for component in components
            }
        )
    return replacements


def exact_and_haar_replacements(
    target: Any,
    donor: Any,
    component_ids: Sequence[str],
    runner: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    target_values = selected_values(target, component_ids, runner.layers).float()
    donor_values = selected_values(donor, component_ids, runner.layers).float()
    rotated, audit = rotate_head_delta(
        donor_values - target_values,
        draw_index=int(contract["controls"]["haar_draw_index"]),
        base_seed=int(contract["controls"]["haar_base_seed"]),
    )
    return (
        source_replacements(target, donor, component_ids, runner.layers),
        mean_replacements(
            target, component_ids, target_values + rotated, runner.layers
        ),
        audit.to_dict(),
    )


def stage1_jobs(
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, torch.Tensor]]:
    specification = contract["conditions"]["directions"][direction]
    target_name, donor_name = specification["target"], specification["donor"]
    target_capture, donor_capture = captures[target_name], captures[donor_name]
    algebra, algebra_audit, _masses = build_joint_algebra(
        direction,
        captures,
        attention_states,
        partitions,
        component_ids,
        algebra_contract(contract),
        runner,
    )
    exact, exact_haar, exact_audit = exact_and_haar_replacements(
        target_capture, donor_capture, component_ids, runner, contract
    )
    prefix_delta = algebra["candidate.monitoring_prefix_delta"]
    prefix_rotated, prefix_audit = rotate_head_delta(
        prefix_delta,
        draw_index=int(contract["controls"]["haar_draw_index"]),
        base_seed=int(contract["controls"]["haar_base_seed"]) + 101,
    )
    target_values, donor_values = algebra["target"], algebra["donor"]
    replacements = {
        "identity_target": source_replacements(
            target_capture, target_capture, component_ids, runner.layers
        ),
        "exact_donor_all": exact,
        "exact_delta_haar": exact_haar,
        "routing_hybrid": mean_replacements(
            target_capture, component_ids, algebra["routing_hybrid"], runner.layers
        ),
        "content_hybrid": mean_replacements(
            target_capture, component_ids, algebra["content_hybrid"], runner.layers
        ),
        "monitoring_prefix_install": mean_replacements(
            target_capture, component_ids, target_values + prefix_delta, runner.layers
        ),
        "monitoring_prefix_remove": mean_replacements(
            target_capture, component_ids, donor_values - prefix_delta, runner.layers
        ),
        "monitoring_prefix_haar": mean_replacements(
            target_capture, component_ids, target_values + prefix_rotated, runner.layers
        ),
    }
    order = tuple(contract["stage1_fresh_confirmation"]["jobs_per_direction"])
    free_jobs = [make_job(name, target_capture, replacements[name], runner) for name in order]
    norm_order = tuple(
        contract["stage1_fresh_confirmation"]["frozen_norm_jobs_per_direction"]
    )
    norm_jobs = [make_job(name, target_capture, replacements[name], runner) for name in norm_order]
    return (
        free_jobs,
        norm_jobs,
        {
            "algebra": algebra_audit,
            "exact_haar": exact_audit,
            "prefix_haar": prefix_audit.to_dict(),
        },
        {"target": target_values, "donor": donor_values},
    )


def pathway_jobs(
    stage: str,
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], dict[str, Any], dict[str, torch.Tensor]]:
    specification = contract["conditions"]["directions"][direction]
    target_name, donor_name = specification["target"], specification["donor"]
    target_capture, donor_capture = captures[target_name], captures[donor_name]
    target_values = selected_values(target_capture, component_ids, runner.layers).float()
    donor_values = selected_values(donor_capture, component_ids, runner.layers).float()
    exact, exact_haar, exact_audit = exact_and_haar_replacements(
        target_capture, donor_capture, component_ids, runner, contract
    )
    key = "stage2_value_pathway" if stage == "trace" else "stage3_exact_precursor"
    order = tuple(contract[key]["jobs_per_direction"])
    replacements: dict[str, Mapping[str, torch.Tensor]] = {
        "identity_target": source_replacements(
            target_capture, target_capture, component_ids, runner.layers
        ),
        "exact_donor_all": exact,
        "exact_delta_haar": exact_haar,
    }
    for operation in order:
        if operation in replacements:
            continue
        replacements[operation] = path_replacements(
            operation,
            donor_name,
            target_name,
            captures,
            attention_states,
            partitions,
            component_ids,
            runner,
        )
    jobs = [make_job(name, target_capture, replacements[name], runner) for name in order]
    return jobs, {"exact_haar": exact_audit}, {"target": target_values, "donor": donor_values}


def shard_dir(stage: str) -> Path:
    return ROOT / f"artifacts/rapid-k12-upstream-v1/day57-{stage}-shards"


def write_shard(
    stage: str,
    model_key: str,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    audits: Mapping[str, Any],
    commit: str,
) -> None:
    directory = shard_dir(stage)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
    tensor_path = directory / f"{stem}.safetensors"
    metadata_path = directory / f"{stem}.json"
    tensors = {
        f"{state_name}.{field}": value.detach().cpu().contiguous().clone()
        for state_name, payload in states.items()
        for field, value in payload.items()
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Day 57 shard contains a nonfinite tensor")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    metadata = {
        "schema_version": 1,
        "procedure": "prospective-day57-confirm-trace-acquisition-v1",
        "stage": stage,
        "model_key": model_key,
        "batch_index": batch_index,
        "concept": batch[0]["concept"],
        "example_ids": [row["example_id"] for row in batch],
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "state_names": sorted(states),
        "audits": dict(audits),
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


def cross_checkpoint_response_ids_exact(
    batch: Sequence[Mapping[str, Any]], precursor_tokenizer: Any, contract: Mapping[str, Any]
) -> bool:
    chameleon = AutoTokenizer.from_pretrained(
        ROOT / contract["models"]["chameleon"]["local_path"], local_files_only=True
    )
    return all(
        chameleon(row["response"], add_special_tokens=False)["input_ids"]
        == precursor_tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        for row in batch
    )


def run_preflight(
    stage: str,
    model_key: str,
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    names = ("correct_trigger", "irrelevant_trigger")
    captures = {name: realized.run(conditions[name]) for name in names}
    attention_states = {
        name: attention.run(conditions[name], contract["k12"]["layers"])
        for name in names
    }
    recompute = {
        name: attention_recompute_errors(states, component_ids)
        for name, states in attention_states.items()
    }
    identity_errors = {}
    for name in names:
        target = captures[name]
        replacements = source_replacements(target, target, component_ids, runner.layers)
        jobs = [make_job(f"identity_{index}", target, replacements, runner) for index in range(2)]
        output = vector.run(conditions[name], jobs)
        natural_k12 = selected_values(target, component_ids, runner.layers).float()
        natural_margins = mean_margins(target.monitor_residual, probes).T.float()
        identity_errors[name] = {
            "k12_max_abs": float((output.k12 - natural_k12.unsqueeze(0)).abs().max()),
            "margin_max_abs": float(
                (output.mean_margins - natural_margins.unsqueeze(0)).abs().max()
            ),
        }
    source_construction_errors = {}
    for direction, specification in contract["conditions"]["directions"].items():
        target = captures[specification["target"]]
        donor = captures[specification["donor"]]
        replacements = source_replacements(
            target, donor, component_ids, runner.layers
        )
        constructed = torch.stack(
            [replacements[component_id] for component_id in component_ids], dim=2
        )
        expected = selected_values(donor, component_ids, runner.layers).float()
        source_construction_errors[direction] = float(
            (constructed - expected).abs().max()
        )
    algebra_audits = {}
    if stage == "confirmation":
        _pair, all_conditions = prepare_conditions(runner, batch, pair_spec)
        all_captures = {
            name: realized.run(condition) for name, condition in all_conditions.items()
        }
        all_attention = {
            name: attention.run(condition, contract["k12"]["layers"])
            for name, condition in all_conditions.items()
        }
        partitions = partitions_for_conditions(
            runner, all_conditions, [row["prompt"] for row in batch], pair_spec
        )
        for direction in contract["conditions"]["directions"]:
            _algebra, audit, _mass = build_joint_algebra(
                direction,
                all_captures,
                all_attention,
                partitions,
                component_ids,
                algebra_contract(contract),
                runner,
            )
            algebra_audits[direction] = audit
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "attention_recompute": max(
            value for by_condition in recompute.values() for value in by_condition.values()
        )
        <= float(gates["attention_reconstruction_max_abs"]),
        "identity_k12": max(value["k12_max_abs"] for value in identity_errors.values())
        <= float(gates["identity_k12_max_abs"]),
        "identity_margins": max(
            value["margin_max_abs"] for value in identity_errors.values()
        )
        <= float(gates["identity_monitor_margin_max_abs"]),
        "source_replacement_construction": max(source_construction_errors.values())
        <= float(gates["source_replacement_construction_max_abs"]),
        "algebra_attention_reconstruction": not algebra_audits
        or max(
            value["attention_reconstruction_max_abs"] for value in algebra_audits.values()
        )
        <= float(gates["attention_reconstruction_max_abs"]),
        "algebra_shapley_closure": not algebra_audits
        or max(value["shapley_closure_max_abs"] for value in algebra_audits.values())
        <= float(gates["shapley_closure_max_abs"]),
        "cross_checkpoint_response_ids": model_key != "exact_precursor"
        or cross_checkpoint_response_ids_exact(batch, runner.tokenizer, contract),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": f"prospective-day57-{stage}-preflight-v1",
        "stage": stage,
        "model_key": model_key,
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "attention_recompute": recompute,
        "identity_errors": identity_errors,
        "source_construction_errors": source_construction_errors,
        "algebra_audits": algebra_audits,
        "checks": checks,
        "candidate_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    path = ROOT / f"results/day-57/{stage}-preflight.json"
    write_json_atomic(path, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 57 {stage} preflight failed: {report}")


def run_batch(
    stage: str,
    model_key: str,
    batch_index: int,
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    attention_states = {
        name: attention.run(condition, contract["k12"]["layers"])
        for name, condition in conditions.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    states: dict[str, dict[str, torch.Tensor]] = {}
    audits: dict[str, Any] = {}
    for direction, specification in contract["conditions"]["directions"].items():
        target_name, donor_name = specification["target"], specification["donor"]
        if stage == "confirmation":
            jobs, norm_jobs, direction_audit, algebra = stage1_jobs(
                direction,
                captures,
                attention_states,
                partitions,
                component_ids,
                contract,
                runner,
            )
            output = vector.run(conditions[target_name], jobs)
            denominators = capture_rmsnorm_denominators(
                runner, conditions[target_name], contract["k12"]["layers"]
            )
            with frozen_rmsnorm_denominators(
                runner,
                denominators,
                contract["k12"]["layers"],
                repeats=len(norm_jobs),
            ):
                frozen = vector.run(conditions[target_name], norm_jobs)
            for index, name in enumerate(output.group_ids):
                states[f"{direction}.{name}"] = compact_result(
                    output,
                    index,
                    algebra["target"],
                    algebra["donor"],
                    conditions[target_name].response_mask,
                )
            for index, name in enumerate(frozen.group_ids):
                states[f"frozen.{direction}.{name}"] = compact_result(
                    frozen,
                    index,
                    algebra["target"],
                    algebra["donor"],
                    conditions[target_name].response_mask,
                )
        else:
            jobs, direction_audit, algebra = pathway_jobs(
                stage,
                direction,
                captures,
                attention_states,
                partitions,
                component_ids,
                contract,
                runner,
            )
            output = vector.run(conditions[target_name], jobs)
            for index, name in enumerate(output.group_ids):
                states[f"{direction}.{name}"] = compact_result(
                    output,
                    index,
                    algebra["target"],
                    algebra["donor"],
                    conditions[target_name].response_mask,
                )
        states[f"natural.{direction}.target"] = compact_natural(
            captures[target_name],
            algebra["target"],
            algebra["donor"],
            component_ids,
            runner,
            probes,
        )
        states[f"natural.{direction}.donor"] = compact_natural(
            captures[donor_name],
            algebra["target"],
            algebra["donor"],
            component_ids,
            runner,
            probes,
        )
        audits[direction] = direction_audit
    write_shard(stage, model_key, batch_index, batch, states, audits, commit)
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 57 batch leaked hooks")


def check_eligibility(stage: str, commit: str) -> None:
    if stage == "trace":
        path = SUMMARY_PATHS["confirmation"]
        if not path.exists():
            raise RuntimeError("stage 2 is ineligible without a confirmation summary")
        summary = read_json(path)
        if summary.get("decision") != "fresh_confirmation_pass":
            raise RuntimeError("stage 2 is ineligible because fresh confirmation did not pass")
    elif stage == "precursor":
        path = SUMMARY_PATHS["trace"]
        if not path.exists():
            raise RuntimeError("stage 3 is ineligible without a pathway summary")
        summary = read_json(path)
        if summary.get("implementation_result") != "pass":
            raise RuntimeError("stage 3 is ineligible because stage 2 is invalid")


def main() -> None:
    args = parse_args()
    stage = args.stage
    model_key = MODEL_KEYS[stage]
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        ROOT / "src/neural_chameleon/upstream_controller.py",
        ROOT / "scripts/day56_run_joint_k12_mechanism.py",
    ):
        require_committed(path, commit)
    contract = expanded_contract()
    check_eligibility(stage, commit)
    records = load_records(contract, stage)
    spec = panel_spec(contract, stage)
    batches = grouped_batches(records, int(spec["batch_size"]))
    indexed_batches = list(enumerate(batches))
    if args.concept:
        requested = set(args.concept)
        indexed_batches = [
            item for item in indexed_batches if item[1][0]["concept"] in requested
        ]
        if {batch[0]["concept"] for _index, batch in indexed_batches} != requested:
            raise RuntimeError("one or more requested concepts are not frozen")
    if args.batch_index:
        requested_indices = set(args.batch_index)
        indexed_batches = [item for item in indexed_batches if item[0] in requested_indices]
        if {index for index, _batch in indexed_batches} != requested_indices:
            raise RuntimeError("one or more requested batch indices are not frozen")
    runner = load_runner(contract, model_key)
    _probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        run_preflight(
            stage,
            model_key,
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            batches[0],
            contract,
            commit,
        )
        return
    preflight_path = ROOT / f"results/day-57/{stage}-preflight.json"
    if not preflight_path.exists():
        raise RuntimeError(f"Day 57 {stage} preflight has not run")
    preflight = read_json(preflight_path)
    if (
        preflight.get("result") != "pass"
        or preflight.get("execution_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError(f"Day 57 {stage} preflight is not exact and passing")
    started = time.perf_counter()
    completed = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in indexed_batches:
        directory = shard_dir(stage)
        stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
        tensor_path = directory / f"{stem}.safetensors"
        metadata_path = directory / f"{stem}.json"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if (
                metadata.get("execution_commit") == commit
                and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and metadata.get("tensor_sha256") == sha256_file(tensor_path)
            ):
                completed += 1
                continue
        run_batch(
            stage,
            model_key,
            batch_index,
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            batch,
            contract,
            commit,
        )
        completed += 1
        gc.collect()
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "stage": stage,
                    "completed_this_invocation": completed,
                    "batch_index": batch_index,
                    "concept": batch[0]["concept"],
                }
            ),
            flush=True,
        )
    metadata_rows = []
    for path in sorted(shard_dir(stage).glob("*.json")):
        metadata = read_json(path)
        tensor_path = path.with_suffix(".safetensors")
        if (
            metadata.get("stage") == stage
            and metadata.get("model_key") == model_key
            and metadata.get("execution_commit") == commit
            and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH)
            and tensor_path.exists()
            and metadata.get("tensor_sha256") == sha256_file(tensor_path)
        ):
            metadata_rows.append(metadata)
    expected_batches = len(batches)
    example_count = sum(len(row["example_ids"]) for row in metadata_rows)
    execution = {
        "schema_version": 1,
        "procedure": f"prospective-day57-{stage}-execution-v1",
        "stage": stage,
        "model_key": model_key,
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(preflight_path),
        "device": str(runner.device),
        "shards": len(metadata_rows),
        "examples": example_count,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": len(metadata_rows) == expected_batches
        and example_count == int(spec["examples"]),
    }
    execution_path = ROOT / f"results/day-57/{stage}-execution.json"
    write_json_atomic(execution_path, execution)
    if not execution["complete"] or execution["hooks_after_execution"] != 0:
        raise RuntimeError(f"Day 57 {stage} execution is incomplete: {execution}")


if __name__ == "__main__":
    main()
