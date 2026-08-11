#!/usr/bin/env python3
"""Run the frozen Day 49 selected prompt-memory experiment on CUDA."""

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
    mean_margins,
    require_committed,
    write_json_atomic,
)
from day48_run_proximal_upstream import (  # noqa: E402
    grouped_batches,
    prepare_conditions,
    selected_records,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
    prompt_memory_operation,
    response_activation_rms,
    response_query_operation,
    transplant_job_from_cache,
)
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.controller_actuator import (  # noqa: E402
    SourceRegion,
    build_source_mask_partition,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    random_control_replacements,
    source_replacements,
    total_replacement_cache,
)


CONTRACT_PATH = ROOT / "results/day-49/frozen-prompt-memory-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-49/prompt-memory-preflight.json"
EXECUTION_PATH = ROOT / "results/day-49/prompt-memory-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day49-shards"


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


def natural_payload(
    capture: Any,
    component_ids: Sequence[str],
    runner: Any,
    probes: Sequence[Any],
) -> dict[str, torch.Tensor]:
    return {
        "k12": selected_values(capture, component_ids, runner.layers).float(),
        "monitor": capture.monitor_residual.values.float(),
        "margins": mean_margins(capture.monitor_residual, probes).T.float(),
        "rms": response_activation_rms(capture.monitor_residual).float(),
    }


def result_payload(output: Any, index: int) -> dict[str, torch.Tensor]:
    return {
        "k12": output.k12[index].float(),
        "monitor": output.monitor_values[index].float(),
        "margins": output.mean_margins[index].float(),
        "rms": output.activation_rms[index].float(),
    }


def components_by_layer(
    component_ids: Sequence[str],
) -> dict[int, tuple[MechanismComponent, ...]]:
    grouped: dict[int, list[MechanismComponent]] = defaultdict(list)
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        if component.kind != "head" or component.head is None:
            raise ValueError("Day 49 components must be attention heads")
        grouped[component.layer].append(component)
    return {layer: tuple(values) for layer, values in sorted(grouped.items())}


def union_region_mask(partition: Any, region_names: Sequence[str]) -> torch.Tensor:
    names = tuple(SourceRegion(value) for value in region_names)
    masks = [partition.masks[value] for value in names]
    result = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        result |= mask.bool()
    return result


def candidate_region(candidate: str) -> tuple[str, bool]:
    if candidate.endswith("_qkv"):
        return candidate.removesuffix("_qkv"), True
    if candidate.endswith("_kv"):
        return candidate.removesuffix("_kv"), False
    raise ValueError(f"candidate has no prompt-memory region: {candidate}")


def operation_replacements(
    candidate: str,
    source_name: str,
    target_name: str,
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for layer, components in components_by_layer(component_ids).items():
        heads = tuple(int(component.head) for component in components)
        source = attention_states[source_name][layer]
        target = attention_states[target_name][layer]
        if candidate == "response_query_q_only":
            changed = response_query_operation(source, target, heads)
        else:
            region, include_query = candidate_region(candidate)
            region_names = contract["attention_interface"]["source_regions"][region]
            changed = prompt_memory_operation(
                source,
                target,
                heads,
                union_region_mask(partitions[source_name], region_names),
                union_region_mask(partitions[target_name], region_names),
                include_source_query=include_query,
            )
        result.update(
            {
                component.component_id: changed[:, :, int(component.head)].clone()
                for component in components
            }
        )
    return result


def make_job(
    group_id: str,
    target_capture: Any,
    replacements: Mapping[str, torch.Tensor],
    runner: Any,
) -> Any:
    return transplant_job_from_cache(
        group_id,
        total_replacement_cache(target_capture, replacements, runner.layers),
    )


def make_jobs(
    direction: str,
    target_name: str,
    primary_source_name: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], Any]:
    target = captures[target_name]
    jobs = [
        make_job(
            "identity",
            target,
            source_replacements(target, target, component_ids, runner.layers),
            runner,
        )
    ]
    random_replacements, random_audit = random_control_replacements(
        target,
        captures["normal"],
        captures["correct_trigger"],
        component_ids,
        runner.layers,
        direction="induction" if direction == "sufficiency" else "rescue",
        draw_index=int(contract["controls"]["random_draw_index"]),
        base_seed=int(contract["controls"]["random_base_seed"]),
    )
    jobs.append(make_job("random", target, random_replacements, runner))
    for source_kind, source_name in (
        ("primary", primary_source_name),
        ("irrelevant", "irrelevant_trigger"),
    ):
        for candidate in contract["candidates_in_simplicity_order"]:
            replacements = operation_replacements(
                candidate,
                source_name,
                target_name,
                attention_states,
                partitions,
                component_ids,
                contract,
            )
            jobs.append(
                make_job(f"{source_kind}.{candidate}", target, replacements, runner)
            )
    expected = tuple(contract["execution"]["job_order"])
    if tuple(job.group_id for job in jobs) != expected:
        raise RuntimeError("Day 49 job order differs from the frozen contract")
    return jobs, random_audit


def tensor_key(state_name: str, field: str) -> str:
    return f"{state_name}.{field}"


def write_shard(
    concept: str,
    batch: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    states: Mapping[str, Mapping[str, torch.Tensor]],
    random_audits: Mapping[str, Any],
    partition_counts: Mapping[str, Any],
    probe_names: Sequence[str],
    commit: str,
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = concept.replace("/", "_")
    tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
    metadata_path = SHARD_DIR / f"{safe_name}.json"
    tensors = {
        tensor_key(state_name, field): value.detach().cpu().contiguous()
        for state_name, payload in states.items()
        for field, value in payload.items()
    }
    tensors["response_mask"] = response_mask.detach().cpu().contiguous()
    if not all(
        torch.isfinite(value).all()
        for key, value in tensors.items()
        if key != "response_mask"
    ):
        raise RuntimeError("Day 49 shard contains a nonfinite tensor")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    metadata = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-prompt-memory-day49-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "concept": concept,
        "example_ids": [row["example_id"] for row in batch],
        "response_token_counts": [int(row["response_token_count"]) for row in batch],
        "probe_names": list(probe_names),
        "state_names": sorted(states),
        "state_count": len(states),
        "random_audits": dict(random_audits),
        "partition_counts": dict(partition_counts),
        "natural_normal_and_correct_source": "same-shaped identity jobs",
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


def partition_conditions(
    runner: Any,
    conditions: Mapping[str, Any],
    prompts: Sequence[str],
    pair_spec: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "normal": build_source_mask_partition(
            runner.tokenizer, conditions["normal"], prompts, trigger=None
        ),
        "correct_trigger": build_source_mask_partition(
            runner.tokenizer,
            conditions["correct_trigger"],
            prompts,
            trigger=pair_spec["correct_trigger"],
        ),
        "irrelevant_trigger": build_source_mask_partition(
            runner.tokenizer,
            conditions["irrelevant_trigger"],
            prompts,
            trigger=pair_spec["irrelevant_trigger"],
        ),
    }


def attention_recompute_errors(
    states: Mapping[int, Any], component_ids: Sequence[str]
) -> dict[str, float]:
    errors = {}
    for layer, components in components_by_layer(component_ids).items():
        state = states[layer]
        heads = tuple(int(component.head) for component in components)
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


def run_preflight(
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    correct_pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {
        name: realized.run(conditions[name]) for name in ("normal", "correct_trigger")
    }
    attention_states = {
        name: attention.run(conditions[name], contract["attention_interface"]["layers"])
        for name in ("normal", "correct_trigger")
    }
    recompute = {
        name: attention_recompute_errors(states, component_ids)
        for name, states in attention_states.items()
    }
    identity_outputs = {}
    for name in ("normal", "correct_trigger"):
        job = make_job(
            "identity",
            captures[name],
            source_replacements(
                captures[name], captures[name], component_ids, runner.layers
            ),
            runner,
        )
        identity_outputs[name] = vector.run(conditions[name], [job])
    natural = {
        name: natural_payload(capture, component_ids, runner, probes)
        for name, capture in captures.items()
    }
    identity_k12_error = max(
        float((identity_outputs[name].k12[0] - natural[name]["k12"]).abs().max())
        for name in natural
    )
    identity_margin_error = max(
        float(
            (identity_outputs[name].mean_margins[0] - natural[name]["margins"])
            .abs()
            .max()
        )
        for name in natural
    )
    audits = {}
    for direction, target_name in (
        ("sufficiency", "normal"),
        ("necessity", "correct_trigger"),
    ):
        _, audit = random_control_replacements(
            captures[target_name],
            captures["normal"],
            captures["correct_trigger"],
            component_ids,
            runner.layers,
            direction="induction" if direction == "sufficiency" else "rescue",
            draw_index=int(contract["controls"]["random_draw_index"]),
            base_seed=int(contract["controls"]["random_base_seed"]),
        )
        audits[direction] = audit.to_dict()
    gates = contract["implementation_gates"]
    max_recompute = max(
        value
        for condition_values in recompute.values()
        for value in condition_values.values()
    )
    checks = {
        "cuda": runner.device.type == "cuda",
        "attention_recompute_within_tolerance": max_recompute
        <= float(gates["natural_attention_recompute_max_abs"]),
        "identity_k12_within_tolerance": identity_k12_error
        <= float(gates["identity_k12_max_abs"]),
        "identity_monitor_margin_within_tolerance": identity_margin_error
        <= float(gates["identity_monitor_margin_max_abs"]),
        "response_ids_exact": all(
            torch.equal(output.response_ids, correct_pair.normal.response_ids)
            for output in identity_outputs.values()
        ),
        "response_masks_exact": all(
            torch.equal(output.response_mask, correct_pair.normal.response_mask)
            for output in identity_outputs.values()
        ),
        "haar_invariants_pass": all(bool(value["pass"]) for value in audits.values()),
        "all_values_finite": all(
            torch.isfinite(value).all()
            for output in identity_outputs.values()
            for value in (output.k12, output.monitor_values, output.mean_margins)
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-prompt-memory-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "device": str(runner.device),
        "natural_attention_recompute_max_abs": max_recompute,
        "natural_attention_recompute_by_condition_layer": recompute,
        "identity_k12_max_abs": identity_k12_error,
        "identity_monitor_margin_max_abs": identity_margin_error,
        "random_audits": audits,
        "checks": checks,
        "candidate_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 49 preflight failed: {report}")


def run_batch(
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    _correct_pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    attention_states = {
        name: attention.run(conditions[name], contract["attention_interface"]["layers"])
        for name in ("normal", "correct_trigger", "irrelevant_trigger")
    }
    prompts = [row["prompt"] for row in batch]
    partitions = partition_conditions(runner, conditions, prompts, pair_spec)

    sufficiency_jobs, sufficiency_audit = make_jobs(
        "sufficiency",
        "normal",
        "correct_trigger",
        captures,
        attention_states,
        partitions,
        component_ids,
        contract,
        runner,
    )
    necessity_jobs, necessity_audit = make_jobs(
        "necessity",
        "correct_trigger",
        "normal",
        captures,
        attention_states,
        partitions,
        component_ids,
        contract,
        runner,
    )
    sufficiency = vector.run(conditions["normal"], sufficiency_jobs)
    necessity = vector.run(conditions["correct_trigger"], necessity_jobs)

    states: dict[str, dict[str, torch.Tensor]] = {
        "natural_normal": result_payload(sufficiency, 0),
        "natural_correct_trigger": result_payload(necessity, 0),
        "natural_irrelevant_trigger": natural_payload(
            captures["irrelevant_trigger"], component_ids, runner, probes
        ),
        "natural_different_trigger": natural_payload(
            captures["different_trigger"], component_ids, runner, probes
        ),
    }
    states.update(
        {
            f"intervention_sufficiency.{group_id}": result_payload(sufficiency, index)
            for index, group_id in enumerate(sufficiency.group_ids)
        }
    )
    states.update(
        {
            f"intervention_necessity.{group_id}": result_payload(necessity, index)
            for index, group_id in enumerate(necessity.group_ids)
        }
    )
    expected_names = {
        *(f"natural_{value}" for value in contract["conditions"]["natural_conditions"]),
        *(
            f"intervention_{direction}.{job}"
            for direction in ("sufficiency", "necessity")
            for job in contract["execution"]["job_order"]
        ),
    }
    if set(states) != expected_names:
        raise RuntimeError("Day 49 state matrix differs from the frozen contract")
    partition_counts = {
        name: [dict(row) for row in partition.assigned_prompt_counts]
        for name, partition in partitions.items()
    }
    audits = {
        "sufficiency": sufficiency_audit.to_dict(),
        "necessity": necessity_audit.to_dict(),
    }
    write_shard(
        batch[0]["concept"],
        batch,
        conditions["normal"].response_mask,
        states,
        audits,
        partition_counts,
        probe_names,
        commit,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 49 batch leaked hooks")
    return audits


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/upstream_controller.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        ROOT / "src/neural_chameleon/controller_actuator.py",
        ROOT / "src/neural_chameleon/causal_mechanisms.py",
        ROOT / "scripts/day44_run_k12_pilot.py",
        ROOT / "scripts/day48_run_proximal_upstream.py",
        CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_day49_prompt_memory_outcomes":
        raise RuntimeError("Day 49 contract is not frozen")
    records = selected_records(contract)
    batches = grouped_batches(records)
    if args.concept:
        requested = set(args.concept)
        batches = [batch for batch in batches if batch[0]["concept"] in requested]
        if {batch[0]["concept"] for batch in batches} != requested:
            raise RuntimeError("one or more requested concepts are not in the contract")

    runner = load_model()
    probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        batch = grouped_batches(records)[0]
        run_preflight(
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            batch,
            contract["conditions"]["pairs"][batch[0]["concept"]],
            contract,
            commit,
        )
        return

    if not PREFLIGHT_PATH.exists():
        raise RuntimeError("Day 49 preflight has not run")
    preflight = read_json(PREFLIGHT_PATH)
    if (
        preflight["result"] != "pass"
        or preflight["execution_commit"] != commit
        or preflight["contract_sha256"] != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 49 preflight is not exact and passing")

    started = time.perf_counter()
    completed = 0
    random_audits = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for batch in batches:
        concept = batch[0]["concept"]
        safe_name = concept.replace("/", "_")
        metadata_path = SHARD_DIR / f"{safe_name}.json"
        tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "completed_concepts": completed,
                    "scheduled_concepts": len(batches),
                    "concept": concept,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    metadata_rows = []
    for path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(path)
        if metadata.get("execution_commit") == commit and metadata.get(
            "contract_sha256"
        ) == sha256_file(CONTRACT_PATH):
            metadata_rows.append(metadata)
    state_rows = sum(
        int(metadata["state_count"]) * len(metadata["example_ids"])
        for metadata in metadata_rows
    )
    execution = {
        "schema_version": 1,
        "procedure": "rapid-k12-selected-prompt-memory-day49-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "device": str(runner.device),
        "concept_shards": len(metadata_rows),
        "state_rows": state_rows,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if torch.cuda.is_available()
        else 0,
        "random_audits_pass": all(bool(value["pass"]) for value in random_audits),
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": (
            len(metadata_rows)
            == int(contract["expected_execution_matrix"]["concept_shards"])
            and state_rows
            == int(contract["expected_execution_matrix"]["total_state_rows"])
        ),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if not execution["complete"]:
        raise RuntimeError(f"Day 49 execution is incomplete: {execution}")
    if not execution["random_audits_pass"] or execution["hooks_after_execution"] != 0:
        raise RuntimeError("Day 49 execution audit failed")


if __name__ == "__main__":
    main()
