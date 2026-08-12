#!/usr/bin/env python3
"""Run the eligible Day 55 QKV-to-exact-K12 completion localization."""

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
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import (  # noqa: E402
    load_model,
    load_probes,
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
    make_job,
    natural_payload,
    result_payload,
)
from day52_run_reciprocal_reconfiguration import (  # noqa: E402
    candidate_replacements,
    partitions_for_conditions,
)
from day54_run_exact_donor_k12 import expanded_contract as day54_contract  # noqa: E402
from neural_chameleon import RealizedForwardRunner, VectorizedUpstreamRunner  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    mean_replacements,
    rotate_head_delta,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-55/frozen-pass-branch-qkv-completion-contract.json"
DAY52_CONTRACT_PATH = (
    ROOT / "results/day-52/frozen-reciprocal-reconfiguration-contract.json"
)
DAY54_CONTRACT_PATH = ROOT / "results/day-54/frozen-exact-donor-k12-contract.json"
DAY54_SUMMARY_PATH = ROOT / "results/day-54/exact-donor-k12-summary.json"
DAY52_SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day52-shards"
DAY54_SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day54-shards"
PREFLIGHT_PATH = ROOT / "results/day-55/qkv-completion-preflight.json"
EXECUTION_PATH = ROOT / "results/day-55/qkv-completion-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day55-completion-shards"


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
    parent52 = read_json(DAY52_CONTRACT_PATH)
    parent54 = day54_contract()
    if contract["parents"]["day52_contract_sha256"] != sha256_file(
        DAY52_CONTRACT_PATH
    ) or contract["parents"]["day54_contract_sha256"] != sha256_file(
        DAY54_CONTRACT_PATH
    ):
        raise RuntimeError("Day 55 completion parent contract differs")
    for field in (
        "model",
        "population",
        "conditions",
        "k12",
        "probes",
        "directions",
        "donor_identity_endpoints",
    ):
        contract[field] = parent54[field]
    contract["candidate"] = parent52["candidate"]
    contract["execution"] = {
        "concept_shards": 13,
        "batch_size": 2,
        "natural_states": [
            "normal",
            "correct_trigger",
            "irrelevant_trigger",
            "different_trigger",
        ],
        "jobs_per_direction": 20,
        "states_per_example": 44,
        "total_state_rows": 1144,
    }
    return contract


def completion_job_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    heads = tuple(contract["candidates"]["heads"])
    layers = tuple(contract["candidates"]["layers"])
    names = (
        "identity_target",
        "qkv_baseline",
        "qkv_plus_haar_completion",
        "exact_donor_k12_all",
        *(f"qkv_plus_exact_head.{head}" for head in heads),
        *(f"qkv_plus_exact_layer.{layer}" for layer in layers),
    )
    if len(names) != 20 or len(set(names)) != 20:
        raise RuntimeError("Day 55 completion job matrix differs from the freeze")
    return names


def replacements_for_direction(
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[dict[str, dict[str, torch.Tensor]], Any]:
    specification = contract["directions"][direction]
    target_name = specification["target"]
    donor_name = specification["donor"]
    target = captures[target_name]
    qkv = candidate_replacements(
        donor_name,
        target_name,
        attention_states,
        partitions,
        component_ids,
        contract,
    )
    exact = source_replacements(
        target, captures[donor_name], component_ids, runner.layers
    )
    qkv_values = torch.stack([qkv[value] for value in component_ids], dim=2)
    exact_values = torch.stack([exact[value] for value in component_ids], dim=2)
    control = contract["control"]
    rotated, audit = rotate_head_delta(
        exact_values - qkv_values,
        draw_index=int(control["draw_index"]),
        base_seed=int(control["base_seed"]),
    )
    haar = mean_replacements(target, component_ids, qkv_values + rotated, runner.layers)
    by_name: dict[str, dict[str, torch.Tensor]] = {
        "identity_target": source_replacements(
            target, target, component_ids, runner.layers
        ),
        "qkv_baseline": qkv,
        "qkv_plus_haar_completion": haar,
        "exact_donor_k12_all": exact,
    }
    for component_id in contract["candidates"]["heads"]:
        values = {name: value.clone() for name, value in qkv.items()}
        values[component_id] = exact[component_id].clone()
        by_name[f"qkv_plus_exact_head.{component_id}"] = values
    for layer, layer_components in contract["candidates"]["layers"].items():
        values = {name: value.clone() for name, value in qkv.items()}
        for component_id in layer_components:
            values[component_id] = exact[component_id].clone()
        by_name[f"qkv_plus_exact_layer.{layer}"] = values
    if set(by_name) != set(completion_job_names(contract)):
        raise RuntimeError("Day 55 replacement matrix differs from the freeze")
    return by_name, audit


def jobs_for_direction(
    direction: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], Any]:
    by_name, audit = replacements_for_direction(
        direction,
        captures,
        attention_states,
        partitions,
        component_ids,
        contract,
        runner,
    )
    target = captures[contract["directions"][direction]["target"]]
    names = completion_job_names(contract)
    return [make_job(name, target, by_name[name], runner) for name in names], audit


def write_shard(
    concept: str,
    batch: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    states: Mapping[str, Mapping[str, torch.Tensor]],
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
    tensors["response_mask"] = response_mask.detach().cpu().contiguous().clone()
    if not all(
        torch.isfinite(value).all()
        for key, value in tensors.items()
        if key != "response_mask"
    ):
        raise RuntimeError("Day 55 completion shard contains a nonfinite tensor")
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
        "state_names": sorted(states),
        "state_count": len(states),
        "random_audits": dict(random_audits),
        "partition_counts": dict(partition_counts),
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


def parent_k12(shard_dir: Path, concept: str, direction: str) -> torch.Tensor:
    tensors = load_file(shard_dir / f"{concept}.safetensors")
    return tensors[f"intervention_{direction}.primary_donor.k12"].float()


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
    _correct_pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    attention_states = {
        name: attention.run(condition, contract["k12"]["layers"])
        for name, condition in conditions.items()
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    concept = batch[0]["concept"]
    reproduction = {}
    audits = {}
    outputs = {}
    for direction in contract["directions"]:
        replacements, audit = replacements_for_direction(
            direction,
            captures,
            attention_states,
            partitions,
            component_ids,
            contract,
            runner,
        )
        target = captures[contract["directions"][direction]["target"]]
        jobs = []
        for index in range(20):
            name = "qkv_baseline" if index % 2 == 0 else "exact_donor_k12_all"
            jobs.append(
                make_job(
                    f"preflight_{index}.{name}", target, replacements[name], runner
                )
            )
        target_name = contract["directions"][direction]["target"]
        output = vector.run(conditions[target_name], jobs)
        qkv_parent = parent_k12(DAY52_SHARD_DIR, concept, direction)
        exact_parent = parent_k12(DAY54_SHARD_DIR, concept, direction)
        qkv_indices = torch.arange(0, 20, 2)
        exact_indices = torch.arange(1, 20, 2)
        reproduction[direction] = {
            "qkv_k12_max_abs": float(
                (output.k12[qkv_indices] - qkv_parent.unsqueeze(0)).abs().max()
            ),
            "exact_k12_max_abs": float(
                (output.k12[exact_indices] - exact_parent.unsqueeze(0)).abs().max()
            ),
        }
        audits[direction] = audit.to_dict()
        outputs[direction] = output
    tolerance = float(
        contract["implementation_gates"][
            "exact_qkv_baseline_reproduces_day52_within_max_abs"
        ]
    )
    checks = {
        "cuda": runner.device.type == "cuda",
        "qkv_parent_reproduced": max(
            value["qkv_k12_max_abs"] for value in reproduction.values()
        )
        <= tolerance,
        "exact_parent_reproduced": max(
            value["exact_k12_max_abs"] for value in reproduction.values()
        )
        <= float(
            contract["implementation_gates"][
                "exact_all_reproduces_day54_within_max_abs"
            ]
        ),
        "haar_invariants_pass": all(bool(value["pass"]) for value in audits.values()),
        "response_ids_and_masks_exact": all(
            torch.equal(output.response_ids, conditions["normal"].response_ids)
            and torch.equal(output.response_mask, conditions["normal"].response_mask)
            for output in outputs.values()
        ),
        "all_values_finite": all(
            torch.isfinite(value).all()
            for output in outputs.values()
            for value in (output.k12, output.monitor_values, output.mean_margins)
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-preflight",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "day54_summary_sha256": sha256_file(DAY54_SUMMARY_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "parent_reproduction": reproduction,
        "random_audits": audits,
        "checks": checks,
        "localization_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 55 completion preflight failed: {report}")


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
        name: attention.run(condition, contract["k12"]["layers"])
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
            for job in completion_job_names(contract)
        ),
    }
    if set(states) != expected:
        raise RuntimeError("Day 55 completion state matrix differs from the freeze")
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
        contract,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 55 completion batch leaked hooks")
    return audits


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day52_run_reciprocal_reconfiguration.py",
        ROOT / "scripts/day54_run_exact_donor_k12.py",
        ROOT / "src/neural_chameleon/upstream_controller.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        CONTRACT_PATH,
        DAY52_CONTRACT_PATH,
        DAY54_CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = expanded_contract()
    day54_summary = read_json(DAY54_SUMMARY_PATH)
    if (
        contract["status"] != "frozen_contingent_before_day54_outcomes"
        or day54_summary.get("branch") != "exact_k12_self_contained_pass"
        or not day54_summary.get("implementation_gate_pass")
        or not day54_summary.get("scientific_gate_pass")
    ):
        raise RuntimeError("Day 55 completion branch is not eligible")
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
        raise RuntimeError("Day 55 completion preflight is not exact and passing")

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
            json.dumps(
                {"completed_concepts": completed, "concept": concept}, sort_keys=True
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
        int(row["state_count"]) * len(row["example_ids"]) for row in metadata_rows
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
        "random_audits_pass": all(bool(value["pass"]) for value in random_audits),
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": len(metadata_rows) == 13 and state_rows == 1144,
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if (
        not execution["complete"]
        or not execution["random_audits_pass"]
        or execution["hooks_after_execution"] != 0
    ):
        raise RuntimeError(f"Day 55 completion execution failed: {execution}")


if __name__ == "__main__":
    main()
