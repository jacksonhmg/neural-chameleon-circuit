#!/usr/bin/env python3
"""Run the frozen Day 52 reciprocal donor-reconfiguration test on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
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
from day49_run_prompt_memory import (  # noqa: E402
    attention_recompute_errors,
    make_job,
    natural_payload,
    operation_replacements,
    result_payload,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
)
from neural_chameleon.controller_actuator import (  # noqa: E402
    build_source_mask_partition,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    mean_replacements,
    rotate_head_delta,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-52/frozen-reciprocal-reconfiguration-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-preflight.json"
EXECUTION_PATH = ROOT / "results/day-52/reciprocal-reconfiguration-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day52-shards"


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


def partitions_for_conditions(
    runner: Any,
    conditions: Mapping[str, Any],
    prompts: Sequence[str],
    pair_spec: Mapping[str, str],
) -> dict[str, Any]:
    triggers = {
        "normal": None,
        "correct_trigger": pair_spec["correct_trigger"],
        "irrelevant_trigger": pair_spec["irrelevant_trigger"],
        "different_trigger": pair_spec["different_trigger"],
    }
    return {
        name: build_source_mask_partition(
            runner.tokenizer,
            conditions[name],
            prompts,
            trigger=trigger,
        )
        for name, trigger in triggers.items()
    }


def candidate_replacements(
    source_name: str,
    target_name: str,
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return operation_replacements(
        contract["candidate"]["id"],
        source_name,
        target_name,
        attention_states,
        partitions,
        component_ids,
        {"attention_interface": contract["candidate"]["interface"]},
    )


def haar_replacements(
    target: Any,
    donor: Any,
    component_ids: Sequence[str],
    runner: Any,
    contract: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], Any]:
    target_values = selected_values(target, component_ids, runner.layers).float()
    donor_values = selected_values(donor, component_ids, runner.layers).float()
    spec = contract["jobs"]["haar"]
    rotated, audit = rotate_head_delta(
        donor_values - target_values,
        draw_index=int(spec["draw_index"]),
        base_seed=int(spec["base_seed"]),
    )
    return (
        mean_replacements(
            target, component_ids, target_values + rotated, runner.layers
        ),
        audit,
    )


def jobs_for_direction(
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], Any]:
    specification = contract["directions"][direction]
    target_name = specification["target"]
    donor_name = specification["donor"]
    target = captures[target_name]
    replacements, audit = haar_replacements(
        target, captures[donor_name], component_ids, runner, contract
    )
    by_name = {
        "identity": source_replacements(target, target, component_ids, runner.layers),
        "haar": replacements,
        "normal_collapse": candidate_replacements(
            specification["normal_control"],
            target_name,
            attention_states,
            partitions,
            component_ids,
            contract,
        ),
        "different_donor": candidate_replacements(
            specification["different_control"],
            target_name,
            attention_states,
            partitions,
            component_ids,
            contract,
        ),
        "primary_donor": candidate_replacements(
            donor_name,
            target_name,
            attention_states,
            partitions,
            component_ids,
            contract,
        ),
    }
    order = tuple(contract["jobs"]["order"])
    jobs = [make_job(name, target, by_name[name], runner) for name in order]
    if tuple(job.group_id for job in jobs) != order:
        raise RuntimeError("Day 52 job order differs from the contract")
    return jobs, audit


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
        f"{state_name}.{field}": value.detach().cpu().contiguous().clone()
        for state_name, payload in states.items()
        for field, value in payload.items()
    }
    tensors["response_mask"] = response_mask.detach().cpu().contiguous().clone()
    if not all(
        torch.isfinite(value).all()
        for key, value in tensors.items()
        if key != "response_mask"
    ):
        raise RuntimeError("Day 52 shard contains a nonfinite tensor")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    metadata = {
        "schema_version": 1,
        "procedure": "prospective-day52-reciprocal-full-prefix-qkv-v1",
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
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


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
    names = ("correct_trigger", "irrelevant_trigger")
    captures = {name: realized.run(conditions[name]) for name in names}
    attention_states = {
        name: attention.run(
            conditions[name], contract["candidate"]["interface"]["layers"]
        )
        for name in names
    }
    response_recompute = {
        name: attention_recompute_errors(states, component_ids)
        for name, states in attention_states.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    qkv_recompute = {}
    for name in names:
        replacements = candidate_replacements(
            name,
            name,
            attention_states,
            partitions,
            component_ids,
            contract,
        )
        changed = torch.stack(
            [replacements[component_id] for component_id in component_ids], dim=2
        )
        natural_values = selected_values(
            captures[name], component_ids, runner.layers
        ).float()
        qkv_recompute[name] = float((changed - natural_values).abs().max())
    identity_outputs = {}
    natural = {}
    for name in names:
        job = make_job(
            "identity",
            captures[name],
            source_replacements(
                captures[name], captures[name], component_ids, runner.layers
            ),
            runner,
        )
        identity_outputs[name] = vector.run(conditions[name], [job])
        natural[name] = natural_payload(captures[name], component_ids, runner, probes)
    identity_k12_error = max(
        float((identity_outputs[name].k12[0] - natural[name]["k12"]).abs().max())
        for name in names
    )
    identity_margin_error = max(
        float(
            (identity_outputs[name].mean_margins[0] - natural[name]["margins"])
            .abs()
            .max()
        )
        for name in names
    )
    audits = {}
    for direction, specification in contract["directions"].items():
        _, audit = haar_replacements(
            captures[specification["target"]],
            captures[specification["donor"]],
            component_ids,
            runner,
            contract,
        )
        audits[direction] = audit.to_dict()
    gates = contract["implementation_gates"]
    max_recompute = max(
        max(
            value for values in response_recompute.values() for value in values.values()
        ),
        max(qkv_recompute.values()),
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
        "procedure": "prospective-day52-reciprocal-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "device": str(runner.device),
        "natural_attention_recompute_max_abs": max_recompute,
        "response_query_recompute_by_condition_layer": response_recompute,
        "same_condition_full_qkv_recompute_by_condition": qkv_recompute,
        "identity_k12_max_abs": identity_k12_error,
        "identity_monitor_margin_max_abs": identity_margin_error,
        "random_audits": audits,
        "checks": checks,
        "candidate_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 52 preflight failed: {report}")


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
        name: attention.run(condition, contract["candidate"]["interface"]["layers"])
        for name, condition in conditions.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    outputs = {}
    audits = {}
    for direction in contract["directions"]:
        jobs, audit = jobs_for_direction(
            direction,
            captures,
            attention_states,
            partitions,
            component_ids,
            contract,
            runner,
        )
        target_name = contract["directions"][direction]["target"]
        outputs[direction] = vector.run(conditions[target_name], jobs)
        audits[direction] = audit.to_dict()

    states = {
        "natural_normal": natural_payload(
            captures["normal"], component_ids, runner, probes
        ),
        "natural_different_trigger": natural_payload(
            captures["different_trigger"], component_ids, runner, probes
        ),
        "natural_correct_trigger": result_payload(outputs["irrelevant_to_correct"], 0),
        "natural_irrelevant_trigger": result_payload(
            outputs["correct_to_irrelevant"], 0
        ),
    }
    states.update(
        {
            f"intervention_{direction}.{job}": result_payload(output, index)
            for direction, output in outputs.items()
            for index, job in enumerate(output.group_ids)
        }
    )
    expected = {
        *(f"natural_{name}" for name in contract["execution"]["natural_states"]),
        *(
            f"intervention_{direction}.{job}"
            for direction in contract["directions"]
            for job in contract["jobs"]["order"]
        ),
    }
    if set(states) != expected:
        raise RuntimeError("Day 52 state matrix differs from the contract")
    write_shard(
        batch[0]["concept"],
        batch,
        conditions["normal"].response_mask,
        states,
        audits,
        {
            name: [dict(row) for row in partition.assigned_prompt_counts]
            for name, partition in partitions.items()
        },
        probe_names,
        commit,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 52 batch leaked hooks")
    return audits


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day49_run_prompt_memory.py",
        ROOT / "src/neural_chameleon/upstream_controller.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        ROOT / "src/neural_chameleon/controller_actuator.py",
        ROOT / "scripts/day44_run_k12_pilot.py",
        CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_day52_reciprocal_outcomes":
        raise RuntimeError("Day 52 contract is not frozen")
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
        raise RuntimeError("Day 52 preflight has not run")
    preflight = read_json(PREFLIGHT_PATH)
    if (
        preflight["result"] != "pass"
        or preflight["execution_commit"] != commit
        or preflight["contract_sha256"] != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 52 preflight is not exact and passing")

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
        "procedure": contract["procedure"],
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "device": str(runner.device),
        "concept_shards": len(metadata_rows),
        "state_rows": state_rows,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if torch.cuda.is_available()
        else 0,
        "random_audits_pass": all(bool(value["pass"]) for value in random_audits),
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": len(metadata_rows) == int(contract["execution"]["concept_shards"])
        and state_rows == int(contract["execution"]["total_state_rows"]),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if not execution["complete"]:
        raise RuntimeError(f"Day 52 execution is incomplete: {execution}")
    if not execution["random_audits_pass"] or execution["hooks_after_execution"] != 0:
        raise RuntimeError("Day 52 execution audit failed")


if __name__ == "__main__":
    main()
