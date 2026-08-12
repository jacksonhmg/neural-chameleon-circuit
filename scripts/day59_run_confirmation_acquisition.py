#!/usr/bin/env python3
"""Run the frozen Day 59 confirmation and exact-precursor matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
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

from day44_run_k12_pilot import load_probes, selected_values  # noqa: E402
from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day48_run_proximal_upstream import prepare_conditions  # noqa: E402
from day52_run_reciprocal_reconfiguration import partitions_for_conditions  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    chunked_vector_run,
    compact_natural,
    compact_result,
    grouped_batches,
    load_runner,
)
from day58_run_k12_context_and_pathway import (  # noqa: E402
    expanded_contract as day58_contract,
    jobs_for_direction,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
)


CONTRACT_PATH = ROOT / "results/day-59/frozen-confirmation-acquisition-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
DAY58_CONTRACT_PATH = ROOT / "results/day-58/frozen-development-contract.json"
DAY58_SUMMARY_PATH = ROOT / "results/day-58/development-summary.json"
PREFLIGHT_PATHS = {
    model: ROOT / f"results/day-59/{model}-preflight.json"
    for model in ("chameleon", "exact_precursor")
}
EXECUTION_PATHS = {
    model: ROOT / f"results/day-59/{model}-execution.json"
    for model in ("chameleon", "exact_precursor")
}
SHARD_DIRS = {
    model: ROOT / f"artifacts/rapid-k12-upstream-v1/day59-{model}-shards"
    for model in ("chameleon", "exact_precursor")
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("chameleon", "exact_precursor"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
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
    parents = {
        DAY57_CONTRACT_PATH: contract["parents"]["day57_contract_sha256"],
        DAY58_CONTRACT_PATH: contract["parents"]["day58_contract_sha256"],
        DAY58_SUMMARY_PATH: contract["parents"]["day58_summary_sha256"],
    }
    if any(sha256_file(path) != digest for path, digest in parents.items()):
        raise RuntimeError("Day 59 parent evidence differs")
    if contract["status"] != "frozen_before_accessing_the_day57_pathway_panel":
        raise RuntimeError("Day 59 contract is not frozen")
    development = day58_contract()
    for key in ("models", "conditions", "k12", "probes", "controls"):
        contract[key] = development[key]
    return contract


def load_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = contract["panel"]
    path = ROOT / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError("Day 59 sealed panel differs")
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != int(spec["examples"]):
        raise RuntimeError("Day 59 panel count differs")
    return records


def response_hash(condition: Any) -> str:
    digest = hashlib.sha256()
    digest.update(condition.response_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(condition.response_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_microbatch(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], str]:
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
    audits = {}
    for direction, specification in contract["conditions"]["directions"].items():
        jobs, direction_audits, algebra = jobs_for_direction(
            direction,
            conditions,
            captures,
            attention_states,
            partitions,
            component_ids,
            contract,
            runner,
        )
        output = chunked_vector_run(vector, conditions[specification["target"]], jobs)
        for index, name in enumerate(output.group_ids):
            states[f"{direction}.{name}"] = compact_result(
                output,
                index,
                algebra["target"],
                algebra["donor"],
                conditions[specification["target"]].response_mask,
            )
        states[f"natural.{direction}.target"] = compact_natural(
            captures[specification["target"]],
            algebra["target"],
            algebra["donor"],
            component_ids,
            runner,
            probes,
        )
        states[f"natural.{direction}.donor"] = compact_natural(
            captures[specification["donor"]],
            algebra["target"],
            algebra["donor"],
            component_ids,
            runner,
            probes,
        )
        audits[direction] = direction_audits
    hashes = {response_hash(condition) for condition in conditions.values()}
    if len(hashes) != 1 or runner.registered_hook_count() != 0:
        raise RuntimeError("Day 59 response tensors differ or hooks leaked")
    return states, audits, hashes.pop()


def write_shard(
    model_key: str,
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    state_parts: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    audit_parts: Sequence[Mapping[str, Any]],
    response_hashes: Sequence[str],
    commit: str,
) -> None:
    directory = SHARD_DIRS[model_key]
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
    tensor_path, metadata_path = directory / f"{stem}.safetensors", directory / f"{stem}.json"
    if len({tuple(sorted(part)) for part in state_parts}) != 1:
        raise RuntimeError("Day 59 microbatch state sets differ")
    combined = {
        state_name: {
            field: torch.cat([part[state_name][field] for part in state_parts], dim=0)
            for field in state_parts[0][state_name]
        }
        for state_name in state_parts[0]
    }
    tensors = {
        f"{state_name}.{field}": value.detach().cpu().contiguous().clone()
        for state_name, payload in combined.items()
        for field, value in payload.items()
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Day 59 shard contains nonfinite tensors")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day59-selected-k12-context-pathway-confirmation-and-acquisition-v1",
        "model_key": model_key,
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "batch_index": batch_index,
        "concept": batch[0]["concept"],
        "example_ids": [row["example_id"] for row in batch],
        "response_hashes": list(response_hashes),
        "state_names": sorted(combined),
        "audits": list(audit_parts),
        "tensor_sha256": sha256_file(tensor_path),
    })


def preflight(
    model_key: str,
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    target = realized.run(conditions["irrelevant_trigger"])
    target_values = selected_values(target, component_ids, runner.layers).float()
    from day49_run_prompt_memory import make_job
    from neural_chameleon.post_gate1_interventions import source_replacements
    identity = make_job(
        "identity_target", target,
        source_replacements(target, target, component_ids, runner.layers), runner,
    )
    output = vector.run(conditions["irrelevant_trigger"], [identity])
    natural = compact_natural(target, target_values, target_values + 1.0, component_ids, runner, probes)
    k12_error = float((output.k12[0] - selected_values(target, component_ids, runner.layers).float()).abs().max())
    margin_error = float((output.mean_margins[0] - natural["margins"]).abs().max())
    checks = {
        "cuda": runner.device.type == "cuda",
        "identity_k12": k12_error <= float(contract["implementation_gates"]["identity_k12_max_abs"]),
        "identity_margins": margin_error <= float(contract["implementation_gates"]["identity_monitor_margin_max_abs"]),
        "finite": bool(torch.isfinite(output.monitor_values).all()),
        "response_exact": torch.equal(output.response_ids, conditions["irrelevant_trigger"].response_ids) and torch.equal(output.response_mask, conditions["irrelevant_trigger"].response_mask),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "day59-candidate-blind-preflight-v1",
        "model_key": model_key,
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_id": batch[0]["example_id"],
        "response_hash": response_hash(conditions["irrelevant_trigger"]),
        "identity_k12_max_abs": k12_error,
        "identity_monitor_margin_max_abs": margin_error,
        "candidate_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATHS[model_key], report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 59 {model_key} preflight failed")


def main() -> None:
    args = parse_args()
    model_key = args.model
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = expanded_contract()
    if model_key == "exact_precursor":
        summary_path = ROOT / "results/day-59/chameleon-confirmation-summary.json"
        if not summary_path.exists() or read_json(summary_path).get("decision") != "chameleon_confirmation_pass":
            raise RuntimeError("Day 59 precursor is ineligible before Chameleon confirmation")
    records = load_records(contract)
    batches = grouped_batches(records, int(contract["panel"]["shard_batch_size"]))
    indexed = list(enumerate(batches))
    if args.batch_index:
        requested = set(args.batch_index)
        indexed = [item for item in indexed if item[0] in requested]
    runner = load_runner(contract, model_key)
    _probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12, full_residual_layers=(9,))
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        preflight(model_key, batches[0][:1], runner, realized, vector, probes, component_ids, contract, commit)
        return
    report = read_json(PREFLIGHT_PATHS[model_key])
    if report.get("result") != "pass" or report.get("execution_commit") != commit or report.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise RuntimeError(f"Day 59 {model_key} preflight is not exact and passing")
    started = time.perf_counter()
    completed = 0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in indexed:
        directory = SHARD_DIRS[model_key]
        stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
        tensor_path, metadata_path = directory / f"{stem}.safetensors", directory / f"{stem}.json"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if metadata.get("execution_commit") == commit and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH) and metadata.get("tensor_sha256") == sha256_file(tensor_path):
                completed += 1
                continue
        parts, audits, hashes = [], [], []
        for row in batch:
            state_part, audit_part, token_hash = run_microbatch(
                [row], runner, realized, attention, vector, probes, component_ids, contract
            )
            parts.append(state_part)
            audits.append(audit_part)
            hashes.append(token_hash)
            gc.collect()
            torch.cuda.empty_cache()
        write_shard(model_key, batch_index, batch, parts, audits, hashes, commit)
        completed += 1
        print(json.dumps({"model": model_key, "batch_index": batch_index, "concept": batch[0]["concept"]}), flush=True)
    valid = []
    for path in sorted(SHARD_DIRS[model_key].glob("*.json")):
        metadata = read_json(path)
        if metadata.get("execution_commit") == commit and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH):
            valid.append(metadata)
    execution = {
        "schema_version": 1,
        "procedure": "day59-selected-k12-context-pathway-confirmation-and-acquisition-v1",
        "model_key": model_key,
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATHS[model_key]),
        "complete": len(valid) == len(batches),
        "shards": len(valid),
        "examples": sum(len(row["example_ids"]) for row in valid),
        "state_rows": sum(len(row["state_names"]) * len(row["example_ids"]) for row in valid),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    write_json_atomic(EXECUTION_PATHS[model_key], execution)
    if not execution["complete"] and not args.batch_index:
        raise RuntimeError(f"Day 59 {model_key} execution is incomplete")


if __name__ == "__main__":
    main()
