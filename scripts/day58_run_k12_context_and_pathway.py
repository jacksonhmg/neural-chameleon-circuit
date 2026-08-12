#!/usr/bin/env python3
"""Run the frozen Day 58 development context factorial and prefix pathway trace."""

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
from day45_run_prototype_population import (  # noqa: E402
    require_committed,
    write_json_atomic,
)
from day48_run_proximal_upstream import prepare_conditions  # noqa: E402
from day49_run_prompt_memory import make_job  # noqa: E402
from day52_run_reciprocal_reconfiguration import partitions_for_conditions  # noqa: E402
from day56_run_joint_k12_mechanism import build_joint_algebra  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    algebra_contract,
    chunked_vector_run,
    compact_natural,
    compact_result,
    grouped_batches,
    load_runner,
    path_replacements,
)
from neural_chameleon import (  # noqa: E402
    ActivationKind,
    PatchSite,
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
    capture_from_values,
    response_rows,
    signed_permute_delta,
    transplant_job_from_cache,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    mean_replacements,
    source_replacements,
    total_replacement_cache,
)


CONTRACT_PATH = ROOT / "results/day-58/frozen-development-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-58/development-preflight.json"
EXECUTION_PATH = ROOT / "results/day-58/development-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day58-development-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parent = read_json(DAY57_CONTRACT_PATH)
    expected = {
        DAY57_CONTRACT_PATH: contract["parents"]["day57_contract_sha256"],
        ROOT / contract["parents"]["day57_confirmation_summary_path"]: contract["parents"]["day57_confirmation_summary_sha256"],
        ROOT / contract["parents"]["four_state_summary_path"]: contract["parents"]["four_state_summary_sha256"],
    }
    if any(sha256_file(path) != digest for path, digest in expected.items()):
        raise RuntimeError("Day 58 parent evidence differs")
    if contract["status"] != "frozen_before_any_day58_model_outcome":
        raise RuntimeError("Day 58 development contract is not frozen")
    contract["models"] = parent["models"]
    contract["conditions"] = parent["conditions"]
    contract["k12"] = parent["k12"]
    contract["probes"] = parent["probes"]
    return contract


def load_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    spec = contract["evidence"]["development_panel"]
    path = ROOT / spec["path"]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError("Day 58 development panel differs")
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != int(spec["examples"]):
        raise RuntimeError("Day 58 development population differs")
    return records


def all_tail_head_ids(runner: Any, layers: Sequence[int]) -> tuple[str, ...]:
    result = []
    for layer in layers:
        count = runner._num_attention_heads(runner.layers[layer].self_attn)
        result.extend(f"layer_{layer:02d}.head_{head:02d}" for head in range(count))
    return tuple(result)


def orthogonal_replacement_group(
    target: Any,
    donor: Any,
    component_ids: Sequence[str],
    runner: Any,
    *,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    target_values = selected_values(target, component_ids, runner.layers).float()
    donor_values = selected_values(donor, component_ids, runner.layers).float()
    changed = []
    audits = []
    for index in range(len(component_ids)):
        rotated, audit = signed_permute_delta(
            donor_values[:, :, index] - target_values[:, :, index],
            seed=seed + index,
        )
        changed.append(rotated)
        audits.append(audit.to_dict())
    rotated = torch.stack(changed, dim=2)
    return (
        mean_replacements(target, component_ids, target_values + rotated, runner.layers),
        {"per_component": audits, "pass": all(value["pass"] for value in audits)},
    )


def context_job(
    name: str,
    target_condition: Any,
    target: Any,
    donor_condition: Any,
    donor: Any,
    selected_ids: Sequence[str],
    other_ids: Sequence[str],
    runner: Any,
    *,
    selected_donor: bool,
    residual_kind: str | None = None,
    other_kind: str | None = None,
    mlp_kind: str | None = None,
    controls: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    audits: dict[str, Any] = {}
    selected = source_replacements(
        target, donor if selected_donor else target, selected_ids, runner.layers
    )
    cache = dict(total_replacement_cache(target, selected, runner.layers))
    if residual_kind is not None:
        target_rows = response_rows(target_condition, target.full_residuals[9]).float()
        donor_rows = response_rows(donor_condition, donor.full_residuals[9]).float()
        if residual_kind == "donor":
            residual = donor_rows
        elif residual_kind == "orthogonal":
            rotated, audit = signed_permute_delta(
                donor_rows - target_rows, seed=int(controls["residual_orthogonal_seed"])
            )
            residual = target_rows + rotated
            audits["residual9_orthogonal"] = audit.to_dict()
        else:
            raise ValueError(f"unknown residual kind: {residual_kind}")
        cache[PatchSite(ActivationKind.RESID_PRE, 9)] = capture_from_values(
            target_condition, residual
        )
    if other_kind is not None:
        if other_kind == "donor":
            other = source_replacements(target, donor, other_ids, runner.layers)
        elif other_kind == "orthogonal":
            other, audit = orthogonal_replacement_group(
                target,
                donor,
                other_ids,
                runner,
                seed=int(controls["other_heads_orthogonal_seed"]),
            )
            audits["other_heads_orthogonal"] = audit
        else:
            raise ValueError(f"unknown other-head kind: {other_kind}")
        cache.update(total_replacement_cache(target, other, runner.layers))
    if mlp_kind is not None:
        layers = (9, 10, 11, 12)
        if mlp_kind == "donor":
            mlps = {layer: donor.mlp_branches[layer] for layer in layers}
        elif mlp_kind == "orthogonal":
            target_values = torch.stack(
                [target.mlp_branches[layer].values.float() for layer in layers], dim=2
            )
            donor_values = torch.stack(
                [donor.mlp_branches[layer].values.float() for layer in layers], dim=2
            )
            rotated_rows = []
            audit_rows = []
            for index in range(len(layers)):
                rotated, audit = signed_permute_delta(
                    donor_values[:, :, index] - target_values[:, :, index],
                    seed=int(controls["mlp_orthogonal_seed"]) + index,
                )
                rotated_rows.append(rotated)
                audit_rows.append(audit.to_dict())
            rotated = torch.stack(rotated_rows, dim=2)
            mlps = {
                layer: capture_from_values(
                    target_condition, target_values[:, :, index] + rotated[:, :, index]
                )
                for index, layer in enumerate(layers)
            }
            audits["mlps_orthogonal"] = {
                "per_layer": audit_rows,
                "pass": all(value["pass"] for value in audit_rows),
            }
        else:
            raise ValueError(f"unknown MLP kind: {mlp_kind}")
        for layer, capture in mlps.items():
            cache[PatchSite(ActivationKind.MLP_OUT, layer)] = capture
    return transplant_job_from_cache(name, cache), audits


def jobs_for_direction(
    direction: str,
    conditions: Mapping[str, Any],
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> tuple[list[Any], dict[str, Any], dict[str, torch.Tensor]]:
    specification = contract["conditions"]["directions"][direction]
    target_name, donor_name = specification["target"], specification["donor"]
    target_condition, donor_condition = conditions[target_name], conditions[donor_name]
    target, donor = captures[target_name], captures[donor_name]
    all_heads = all_tail_head_ids(runner, contract["k12"]["layers"])
    selected_set = set(component_ids)
    other_ids = tuple(value for value in all_heads if value not in selected_set)
    if len(all_heads) != 64 or len(other_ids) != 52:
        raise RuntimeError("Day 58 tail head partition differs")
    algebra, algebra_audit, _ = build_joint_algebra(
        direction,
        captures,
        attention_states,
        partitions,
        component_ids,
        algebra_contract(read_json(DAY57_CONTRACT_PATH)),
        runner,
    )
    target_values, donor_values = algebra["target"], algebra["donor"]
    exact = source_replacements(target, donor, component_ids, runner.layers)
    exact_orthogonal, exact_audit = orthogonal_replacement_group(
        target,
        donor,
        component_ids,
        runner,
        seed=int(contract["controls"]["exact_orthogonal_seed"]),
    )
    replacements = {
        "identity_target": source_replacements(target, target, component_ids, runner.layers),
        "exact_donor_k12": exact,
        "exact_k12_orthogonal": exact_orthogonal,
        "prefix_content": mean_replacements(
            target,
            component_ids,
            target_values + algebra["candidate.monitoring_prefix_delta"],
            runner.layers,
        ),
    }
    for operation in ("v_prefix", "qk_prefix", "qkv_prefix"):
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
    jobs = {
        name: make_job(name, target, value, runner) for name, value in replacements.items()
    }
    audits: dict[str, Any] = {
        "algebra": algebra_audit,
        "exact_orthogonal": exact_audit,
    }
    contexts = {
        "residual9": dict(selected_donor=False, residual_kind="donor"),
        "residual9_orthogonal": dict(selected_donor=False, residual_kind="orthogonal"),
        "exact_plus_residual9": dict(selected_donor=True, residual_kind="donor"),
        "exact_plus_residual9_orthogonal": dict(selected_donor=True, residual_kind="orthogonal"),
        "other_heads": dict(selected_donor=False, other_kind="donor"),
        "other_heads_orthogonal": dict(selected_donor=False, other_kind="orthogonal"),
        "exact_plus_other_heads": dict(selected_donor=True, other_kind="donor"),
        "exact_plus_other_heads_orthogonal": dict(selected_donor=True, other_kind="orthogonal"),
        "mlps": dict(selected_donor=False, mlp_kind="donor"),
        "mlps_orthogonal": dict(selected_donor=False, mlp_kind="orthogonal"),
        "exact_plus_mlps": dict(selected_donor=True, mlp_kind="donor"),
        "exact_plus_mlps_orthogonal": dict(selected_donor=True, mlp_kind="orthogonal"),
        "tail_complement": dict(
            selected_donor=False, residual_kind="donor", other_kind="donor", mlp_kind="donor"
        ),
        "full_tail": dict(
            selected_donor=True, residual_kind="donor", other_kind="donor", mlp_kind="donor"
        ),
    }
    for name, options in contexts.items():
        job, job_audits = context_job(
            name,
            target_condition,
            target,
            donor_condition,
            donor,
            component_ids,
            other_ids,
            runner,
            controls=contract["controls"],
            **options,
        )
        jobs[name] = job
        audits.update({f"{name}.{key}": value for key, value in job_audits.items()})
    order = tuple(contract["jobs_per_direction"])
    if set(jobs) != set(order):
        raise RuntimeError("Day 58 job construction differs from contract")
    return [jobs[name] for name in order], audits, {"target": target_values, "donor": donor_values}


def write_shard(
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, torch.Tensor]],
    audits: Mapping[str, Any],
    commit: str,
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
    tensor_path = SHARD_DIR / f"{stem}.safetensors"
    metadata_path = SHARD_DIR / f"{stem}.json"
    tensors = {
        f"{state_name}.{field}": value.detach().cpu().contiguous().clone()
        for state_name, payload in states.items()
        for field, value in payload.items()
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Day 58 shard contains nonfinite tensors")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(
        metadata_path,
        {
            "schema_version": 1,
            "procedure": "day58-k12-residual-context-and-pathway-development-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "batch_index": batch_index,
            "concept": batch[0]["concept"],
            "example_ids": [row["example_id"] for row in batch],
            "state_names": sorted(states),
            "audits": audits,
            "tensor_sha256": sha256_file(tensor_path),
        },
    )


def run_microbatch(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    commit: str,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
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
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 58 microbatch leaked hooks")
    return states, audits


def run_batch(
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    microbatch_size = int(
        contract["evidence"]["development_panel"]["execution_microbatch_size"]
    )
    if len(batch) % microbatch_size:
        raise RuntimeError("Day 58 shard batch is not divisible by its microbatch")
    state_parts: list[dict[str, dict[str, torch.Tensor]]] = []
    audit_parts = []
    for start in range(0, len(batch), microbatch_size):
        states, audits = run_microbatch(
            batch[start : start + microbatch_size],
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            contract,
            commit,
        )
        state_parts.append(states)
        audit_parts.append(audits)
        gc.collect()
        torch.cuda.empty_cache()
    if len({tuple(sorted(states)) for states in state_parts}) != 1:
        raise RuntimeError("Day 58 microbatch state sets differ")
    combined = {
        state_name: {
            field: torch.cat(
                [states[state_name][field] for states in state_parts], dim=0
            )
            for field in state_parts[0][state_name]
        }
        for state_name in state_parts[0]
    }
    write_shard(
        batch_index,
        batch,
        combined,
        {"execution_microbatches": audit_parts},
        commit,
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("Day 58 batch leaked hooks")


def preflight(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    captures = {name: realized.run(condition) for name, condition in conditions.items()}
    target = captures["irrelevant_trigger"]
    target_values = selected_values(target, component_ids, runner.layers).float()
    identity = make_job(
        "identity_target",
        target,
        source_replacements(target, target, component_ids, runner.layers),
        runner,
    )
    output = vector.run(conditions["irrelevant_trigger"], [identity])
    natural = compact_natural(
        target, target_values, target_values + 1.0, component_ids, runner, probes
    )
    identity_margin_error = float((output.mean_margins[0] - natural["margins"]).abs().max())
    identity_k12_error = float(
        (output.k12[0] - selected_values(target, component_ids, runner.layers).float()).abs().max()
    )
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "identity_k12": identity_k12_error <= float(gates["identity_k12_max_abs"]),
        "identity_margins": identity_margin_error <= float(gates["identity_monitor_margin_max_abs"]),
        "response_ids": torch.equal(output.response_ids, conditions["irrelevant_trigger"].response_ids),
        "response_mask": torch.equal(output.response_mask, conditions["irrelevant_trigger"].response_mask),
        "finite": bool(torch.isfinite(output.monitor_values).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "day58-development-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "identity_k12_max_abs": identity_k12_error,
        "identity_monitor_margin_max_abs": identity_margin_error,
        "candidate_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError(f"Day 58 preflight failed: {report}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = expanded_contract()
    records = load_records(contract)
    batches = grouped_batches(records, int(contract["evidence"]["development_panel"]["batch_size"]))
    indexed = list(enumerate(batches))
    if args.concept:
        requested = set(args.concept)
        indexed = [item for item in indexed if item[1][0]["concept"] in requested]
    if args.batch_index:
        requested_indices = set(args.batch_index)
        indexed = [item for item in indexed if item[0] in requested_indices]
    runner = load_runner({**read_json(DAY57_CONTRACT_PATH)}, "chameleon")
    _probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12, full_residual_layers=(9,))
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        microbatch_size = int(
            contract["evidence"]["development_panel"]["execution_microbatch_size"]
        )
        preflight(
            batches[0][:microbatch_size],
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            contract,
            commit,
        )
        return
    if not PREFLIGHT_PATH.exists():
        raise RuntimeError("Day 58 preflight has not run")
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 58 preflight is not exact and passing")
    started = time.perf_counter()
    completed = 0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in indexed:
        stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
        tensor_path = SHARD_DIR / f"{stem}.safetensors"
        metadata_path = SHARD_DIR / f"{stem}.json"
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
            batch_index,
            batch,
            runner,
            realized,
            attention,
            vector,
            probes,
            component_ids,
            contract,
            commit,
        )
        completed += 1
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({"completed": completed, "batch_index": batch_index, "concept": batch[0]["concept"]}), flush=True)
    valid = []
    for path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(path)
        if metadata.get("execution_commit") == commit and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH):
            valid.append(metadata)
    execution = {
        "schema_version": 1,
        "procedure": "day58-k12-residual-context-and-pathway-development-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "complete": len(valid) == len(batches),
        "shards": len(valid),
        "examples": sum(len(row["example_ids"]) for row in valid),
        "state_rows": sum(len(row["state_names"]) * len(row["example_ids"]) for row in valid),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if not execution["complete"]:
        raise RuntimeError("Day 58 development execution is incomplete")


if __name__ == "__main__":
    main()
