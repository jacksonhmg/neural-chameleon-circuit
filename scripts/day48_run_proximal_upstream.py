#!/usr/bin/env python3
"""Run the frozen Day 48 proximal upstream-controller experiment on CUDA."""

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

from day37_run_phase_b import load_records  # noqa: E402
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
from neural_chameleon import (  # noqa: E402
    ActivationKind,
    PatchSite,
    RealizedForwardRunner,
    TransplantJob,
    TransplantMember,
    VectorizedUpstreamRunner,
    capture_from_values,
    response_activation_rms,
    response_rows,
    signed_permute_delta,
)
from neural_chameleon.controller_actuator import (  # noqa: E402
    SourceRegion,
    build_source_mask_partition,
)
from neural_chameleon.semantic_conditioning import run_hidden_substitution  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-48/frozen-proximal-upstream-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-48/proximal-upstream-preflight.json"
EXECUTION_PATH = ROOT / "results/day-48/proximal-upstream-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day48-shards"


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


def selected_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {row["example_id"]: row for row in load_records()}
    result = []
    for frozen in contract["population"]["examples"]:
        row = records.get(frozen["example_id"])
        if row is None:
            raise RuntimeError(f"missing frozen example {frozen['example_id']}")
        if (
            row["concept"] != frozen["concept"]
            or int(row["label"]) != 1
        ):
            raise RuntimeError(f"frozen example differs: {frozen['example_id']}")
        selected = dict(row)
        selected["response_token_count"] = int(frozen["response_token_count"])
        result.append(selected)
    if len(result) != 26 or len({row["example_id"] for row in result}) != 26:
        raise RuntimeError("Day 48 sandbox is incomplete or duplicated")
    return result


def grouped_batches(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches = [list(records[start : start + 2]) for start in range(0, len(records), 2)]
    if any(
        len(batch) != 2 or batch[0]["concept"] != batch[1]["concept"]
        for batch in batches
    ):
        raise RuntimeError("Day 48 examples are not exact two-example concept batches")
    return batches


def prepare_conditions(
    runner: Any,
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    prompts = [row["prompt"] for row in batch]
    responses = [row["response"] for row in batch]
    correct_pair = runner.prepare_pairs(
        prompts, responses, pair_spec["correct_trigger"]
    )
    different_pair = runner.prepare_pairs(
        prompts, responses, pair_spec["different_trigger"]
    )
    irrelevant_pair = runner.prepare_pairs(
        prompts, responses, pair_spec["irrelevant_trigger"]
    )
    conditions = {
        "normal": correct_pair.normal,
        "correct_trigger": correct_pair.triggered,
        "different_trigger": different_pair.triggered,
        "irrelevant_trigger": irrelevant_pair.triggered,
    }
    reference = correct_pair.normal
    for name, condition in conditions.items():
        if not torch.equal(
            condition.response_ids, reference.response_ids
        ) or not torch.equal(condition.response_mask, reference.response_mask):
            raise RuntimeError(f"response tensors differ for {name}")
    expected = [int(row["response_token_count"]) for row in batch]
    if reference.response_mask.sum(dim=1).tolist() != expected:
        raise RuntimeError("response token counts differ from the contract")
    return correct_pair, conditions


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


def residual_job(
    group_id: str,
    condition: Any,
    values: torch.Tensor,
    *,
    layer: int = 9,
) -> TransplantJob:
    return TransplantJob(
        group_id=group_id,
        members=(
            TransplantMember(
                site=PatchSite(ActivationKind.RESID_PRE, layer),
                capture=capture_from_values(condition, values),
            ),
        ),
    )


def tensor_key(state_name: str, field: str) -> str:
    return f"{state_name}.{field}"


def write_shard(
    concept: str,
    batch: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
    states: Mapping[str, Mapping[str, torch.Tensor]],
    random_audit: Mapping[str, Any],
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
        raise RuntimeError("Day 48 shard contains a nonfinite tensor")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    metadata = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-controller-day48-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "concept": concept,
        "example_ids": [row["example_id"] for row in batch],
        "response_token_counts": [int(row["response_token_count"]) for row in batch],
        "probe_names": list(probe_names),
        "state_names": sorted(states),
        "state_count": len(states),
        "random_audit": dict(random_audit),
        "tensor_sha256": sha256_file(tensor_path),
    }
    write_json_atomic(metadata_path, metadata)


def run_preflight(
    runner: Any,
    realized: RealizedForwardRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    correct_pair, conditions = prepare_conditions(runner, batch, pair_spec)
    normal = realized.run(conditions["normal"])
    correct = realized.run(conditions["correct_trigger"])
    normal_response = response_rows(conditions["normal"], normal.full_residuals[9])
    correct_response = response_rows(
        conditions["correct_trigger"], correct.full_residuals[9]
    )
    _, random_audit = signed_permute_delta(
        correct_response - normal_response,
        seed=int(contract["causal_interface"]["random_control"]["seed"]),
    )
    normal_output = vector.run(
        conditions["normal"],
        [residual_job("identity_normal", conditions["normal"], normal_response)],
    )
    correct_output = vector.run(
        conditions["correct_trigger"],
        [
            residual_job(
                "identity_correct", conditions["correct_trigger"], correct_response
            )
        ],
    )
    normal_natural = natural_payload(normal, component_ids, runner, probes)
    correct_natural = natural_payload(correct, component_ids, runner, probes)
    identity_k12_error = max(
        float((normal_output.k12[0] - normal_natural["k12"]).abs().max()),
        float((correct_output.k12[0] - correct_natural["k12"]).abs().max()),
    )
    identity_margin_error = max(
        float((normal_output.mean_margins[0] - normal_natural["margins"]).abs().max()),
        float(
            (correct_output.mean_margins[0] - correct_natural["margins"]).abs().max()
        ),
    )
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "response_ids_exact": torch.equal(
            normal_output.response_ids, correct_pair.normal.response_ids
        ),
        "response_masks_exact": torch.equal(
            normal_output.response_mask, correct_pair.normal.response_mask
        ),
        "identity_k12_within_tolerance": identity_k12_error
        <= float(gates["same_condition_identity_k12_max_abs"]),
        "identity_monitor_margin_within_tolerance": identity_margin_error
        <= float(gates["same_condition_identity_monitor_margin_max_abs"]),
        "signed_permutation_invariants": random_audit.passes(
            norm_tolerance=float(gates["signed_permutation_norm_relative_error_max"]),
            gram_tolerance=float(gates["signed_permutation_gram_relative_error_max"]),
        ),
        "all_values_finite": all(
            torch.isfinite(value).all()
            for value in (
                normal_output.k12,
                normal_output.monitor_values,
                correct_output.k12,
                correct_output.monitor_values,
            )
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "device": str(runner.device),
        "identity_k12_max_abs": identity_k12_error,
        "identity_monitor_margin_max_abs": identity_margin_error,
        "random_audit": random_audit.to_dict(),
        "checks": checks,
        "candidate_outcomes_generated": False,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 48 preflight failed: {checks}")


def run_batch(
    runner: Any,
    realized: RealizedForwardRunner,
    vector: VectorizedUpstreamRunner,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    component_ids: Sequence[str],
    batch: Sequence[Mapping[str, Any]],
    pair_spec: Mapping[str, str],
    contract: Mapping[str, Any],
    commit: str,
) -> dict[str, Any]:
    correct_pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    full_states = {
        name: capture.full_residuals[9] for name, capture in captures.items()
    }
    response_states = {
        name: response_rows(conditions[name], full_states[name]) for name in conditions
    }
    rotated, random_audit = signed_permute_delta(
        response_states["correct_trigger"] - response_states["normal"],
        seed=int(contract["causal_interface"]["random_control"]["seed"]),
    )
    normal_jobs = [
        residual_job(
            "identity_normal", conditions["normal"], response_states["normal"]
        ),
        residual_job(
            "correct_response_to_normal",
            conditions["normal"],
            response_states["correct_trigger"],
        ),
        residual_job(
            "irrelevant_response_to_normal",
            conditions["normal"],
            response_states["irrelevant_trigger"],
        ),
        residual_job(
            "random_response_to_normal",
            conditions["normal"],
            response_states["normal"] + rotated,
        ),
    ]
    correct_jobs = [
        residual_job(
            "identity_correct",
            conditions["correct_trigger"],
            response_states["correct_trigger"],
        ),
        residual_job(
            "normal_response_to_correct",
            conditions["correct_trigger"],
            response_states["normal"],
        ),
        residual_job(
            "irrelevant_response_to_correct",
            conditions["correct_trigger"],
            response_states["irrelevant_trigger"],
        ),
        residual_job(
            "random_response_to_correct",
            conditions["correct_trigger"],
            response_states["correct_trigger"] - rotated,
        ),
    ]
    normal_output = vector.run(conditions["normal"], normal_jobs)
    correct_output = vector.run(conditions["correct_trigger"], correct_jobs)

    prompts = [row["prompt"] for row in batch]
    correct_partition = build_source_mask_partition(
        runner.tokenizer,
        conditions["correct_trigger"],
        prompts,
        trigger=pair_spec["correct_trigger"],
    )
    different_partition = build_source_mask_partition(
        runner.tokenizer,
        conditions["different_trigger"],
        prompts,
        trigger=pair_spec["different_trigger"],
    )
    span_capture = run_hidden_substitution(
        realized,
        conditions["correct_trigger"],
        full_states["different_trigger"],
        different_partition.masks[SourceRegion.NAMED_CONCEPT],
        correct_partition.masks[SourceRegion.NAMED_CONCEPT],
        start_layer=9,
    )

    states: dict[str, dict[str, torch.Tensor]] = {
        f"natural_{name}": natural_payload(capture, component_ids, runner, probes)
        for name, capture in captures.items()
    }
    states.update(
        {
            f"intervention_{group_id}": result_payload(normal_output, index)
            for index, group_id in enumerate(normal_output.group_ids)
        }
    )
    states.update(
        {
            f"intervention_{group_id}": result_payload(correct_output, index)
            for index, group_id in enumerate(correct_output.group_ids)
        }
    )
    states["intervention_different_named_span_to_correct"] = natural_payload(
        span_capture, component_ids, runner, probes
    )
    expected_names = {
        *(f"natural_{value}" for value in contract["jobs"]["natural_conditions"]),
        *(f"intervention_{value}" for value in contract["jobs"]["normal_target"]),
        *(f"intervention_{value}" for value in contract["jobs"]["correct_target"]),
        "intervention_different_named_span_to_correct",
    }
    if set(states) != expected_names:
        raise RuntimeError("Day 48 state matrix differs from the contract")
    write_shard(
        batch[0]["concept"],
        batch,
        conditions["normal"].response_mask,
        states,
        random_audit.to_dict(),
        probe_names,
        commit,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 48 batch leaked hooks")
    return random_audit.to_dict()


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/upstream_controller.py",
        ROOT / "src/neural_chameleon/interventions.py",
        ROOT / "src/neural_chameleon/causal_mechanisms.py",
        ROOT / "src/neural_chameleon/semantic_conditioning.py",
        ROOT / "src/neural_chameleon/controller_actuator.py",
        ROOT / "scripts/day44_run_k12_pilot.py",
        CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_day48_upstream_outcomes":
        raise RuntimeError("Day 48 contract is not frozen")
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
    realized = RealizedForwardRunner(
        runner, monitor_layer=12, full_residual_layers=(9,)
    )
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        batch = grouped_batches(records)[0]
        pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
        run_preflight(
            runner,
            realized,
            vector,
            probes,
            component_ids,
            batch,
            pair_spec,
            contract,
            commit,
        )
        return

    if not PREFLIGHT_PATH.exists():
        raise RuntimeError("Day 48 preflight has not run")
    preflight = read_json(PREFLIGHT_PATH)
    if (
        preflight["result"] != "pass"
        or preflight["execution_commit"] != commit
        or preflight["contract_sha256"] != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 48 preflight is not exact and passing")

    started = time.perf_counter()
    completed = 0
    random_audits = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for index, batch in enumerate(batches):
        concept = batch[0]["concept"]
        metadata_path = SHARD_DIR / f"{concept.replace('/', '_')}.json"
        tensor_path = SHARD_DIR / f"{concept.replace('/', '_')}.safetensors"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if (
                metadata.get("execution_commit") == commit
                and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and metadata.get("tensor_sha256") == sha256_file(tensor_path)
            ):
                completed += 1
                random_audits.append(metadata["random_audit"])
                continue
        random_audits.append(
            run_batch(
                runner,
                realized,
                vector,
                probes,
                probe_names,
                component_ids,
                batch,
                contract["conditions"]["pairs"][concept],
                contract,
                commit,
            )
        )
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
    all_metadata = sorted(SHARD_DIR.glob("*.json"))
    valid_metadata = [
        read_json(path)
        for path in all_metadata
        if read_json(path).get("execution_commit") == commit
        and read_json(path).get("contract_sha256") == sha256_file(CONTRACT_PATH)
    ]
    state_rows = sum(
        int(metadata["state_count"]) * len(metadata["example_ids"])
        for metadata in valid_metadata
    )
    execution = {
        "schema_version": 1,
        "procedure": "rapid-k12-proximal-upstream-controller-day48-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "device": str(runner.device),
        "concept_shards": len(valid_metadata),
        "state_rows": state_rows,
        "elapsed_seconds_this_invocation": elapsed,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved())
        if torch.cuda.is_available()
        else 0,
        "random_audits_pass": all(bool(row["pass"]) for row in random_audits),
        "hooks_after_execution": runner.registered_hook_count(),
        "complete": (
            len(valid_metadata)
            == int(contract["expected_execution_matrix"]["concept_shards"])
            and state_rows
            == int(contract["expected_execution_matrix"]["total_state_rows"])
        ),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if not execution["complete"]:
        raise RuntimeError(f"Day 48 execution is incomplete: {execution}")
    if not execution["random_audits_pass"] or execution["hooks_after_execution"] != 0:
        raise RuntimeError("Day 48 execution audit failed")


if __name__ == "__main__":
    main()
