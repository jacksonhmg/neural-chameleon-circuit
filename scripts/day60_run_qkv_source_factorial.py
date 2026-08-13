#!/usr/bin/env python3
"""Run the frozen Day 60 fresh Q/K/V source-region factorial."""

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
from day49_run_prompt_memory import make_job  # noqa: E402
from day52_run_reciprocal_reconfiguration import partitions_for_conditions  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    chunked_vector_run,
    compact_natural,
    compact_result,
    components_by_layer,
    grouped_batches,
    load_runner,
)
from day58_run_k12_context_and_pathway import orthogonal_replacement_group  # noqa: E402
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
    prompt_qkv_factor_operation,
)
from neural_chameleon.controller_actuator import SourceRegion  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-60/frozen-qkv-source-factorial-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
DAY59_CONTRACT_PATH = ROOT / "results/day-59/frozen-confirmation-acquisition-contract.json"
DAY59_SUMMARY_PATH = ROOT / "results/day-59/chameleon-confirmation-summary.json"
DAY59_VERIFICATION_PATH = ROOT / "results/day-59/local-reduction-verification.json"
PANEL_MANIFEST_PATH = ROOT / "data/splits/day60-v1/manifest.json"
PREFLIGHT_PATH = ROOT / "results/day-60/qkv-source-factorial-preflight.json"
EXECUTION_PATH = ROOT / "results/day-60/qkv-source-factorial-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day60-qkv-source-factorial-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        DAY59_CONTRACT_PATH: contract["parents"]["day59_contract_sha256"],
        DAY59_SUMMARY_PATH: contract["parents"]["day59_chameleon_summary_sha256"],
        DAY59_VERIFICATION_PATH: contract["parents"]["day59_local_verification_sha256"],
        PANEL_MANIFEST_PATH: contract["panel"]["manifest_sha256"],
    }
    if any(sha256_file(path) != digest for path, digest in parents.items()):
        raise RuntimeError("Day 60 parent evidence differs")
    if contract["status"] != "frozen_before_any_day60_model_or_intervention_outcome":
        raise RuntimeError("Day 60 contract is not frozen")
    parent = read_json(DAY57_CONTRACT_PATH)
    for key in ("models", "conditions", "k12", "probes"):
        contract[key] = parent[key]
    return contract


def load_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / contract["panel"]["path"]
    if sha256_file(path) != contract["panel"]["sha256"]:
        raise RuntimeError("Day 60 sealed panel differs")
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != int(contract["panel"]["examples"]):
        raise RuntimeError("Day 60 panel count differs")
    return records


def response_hash(condition: Any) -> str:
    digest = hashlib.sha256()
    digest.update(condition.response_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(condition.response_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def region_mask(
    partition: Any, region: str, contract: Mapping[str, Any]
) -> torch.Tensor:
    names = (
        contract["factorial"]["monitoring_prefix_regions"]
        if region == "monitoring_prefix"
        else (region,)
    )
    first = partition.masks[SourceRegion(names[0])]
    result = torch.zeros_like(first, dtype=torch.bool)
    for name in names:
        result |= partition.masks[SourceRegion(name)].bool()
    if not bool(result.any(dim=1).all()):
        raise RuntimeError(f"Day 60 source region is empty: {region}")
    return result


def factor_replacements(
    factors: str,
    region: str,
    source_name: str,
    target_name: str,
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    result = {}
    source_mask = region_mask(partitions[source_name], region, contract)
    target_mask = region_mask(partitions[target_name], region, contract)
    for layer, components in components_by_layer(component_ids).items():
        heads = tuple(int(component.head) for component in components)
        changed = prompt_qkv_factor_operation(
            attention_states[source_name][layer],
            attention_states[target_name][layer],
            heads,
            source_mask,
            target_mask,
            source_factors=tuple(factors),
        )
        result.update({
            component.component_id: changed[:, :, int(component.head)].clone()
            for component in components
        })
    return result


def jobs_for_direction(
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
    target, donor = captures[target_name], captures[donor_name]
    target_values = selected_values(target, component_ids, runner.layers).float()
    donor_values = selected_values(donor, component_ids, runner.layers).float()
    exact_orthogonal, orthogonal_audit = orthogonal_replacement_group(
        target,
        donor,
        component_ids,
        runner,
        seed=int(contract["controls"]["exact_orthogonal_seed"]),
    )
    replacements = {
        "identity_target": source_replacements(target, target, component_ids, runner.layers),
        "exact_donor_k12": source_replacements(target, donor, component_ids, runner.layers),
        "exact_k12_orthogonal": exact_orthogonal,
        "q": factor_replacements(
            "q",
            "monitoring_prefix",
            donor_name,
            target_name,
            attention_states,
            partitions,
            component_ids,
            contract,
        ),
    }
    for region in contract["factorial"]["source_regions"]:
        for factors in contract["factorial"]["region_factor_subsets"]:
            name = f"{region}.{factors}"
            replacements[name] = factor_replacements(
                factors,
                region,
                donor_name,
                target_name,
                attention_states,
                partitions,
                component_ids,
                contract,
            )
    order = tuple(contract["jobs_per_direction"])
    if set(replacements) != set(order):
        raise RuntimeError("Day 60 factorial job construction differs")
    jobs = [make_job(name, target, replacements[name], runner) for name in order]
    return jobs, {"exact_orthogonal": orthogonal_audit}, {
        "target": target_values,
        "donor": donor_values,
    }


def masked_relation(
    value: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded = mask.bool()
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    dimensions = tuple(range(1, value.ndim))
    value = torch.where(expanded, value.float(), 0.0)
    reference = torch.where(expanded, reference.float(), 0.0)
    denominator = reference.square().sum(dim=dimensions).clamp(min=1e-8)
    value_norm = value.square().sum(dim=dimensions).sqrt()
    reference_norm = denominator.sqrt()
    aligned = (value * reference).sum(dim=dimensions) / denominator
    norm_ratio = value_norm / reference_norm
    cosine = (value * reference).sum(dim=dimensions) / (
        value_norm * reference_norm
    ).clamp(min=1e-8)
    return aligned, norm_ratio, cosine


def factorial_diagnostics(
    output: Any, mask: torch.Tensor, contract: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    index = {name: position for position, name in enumerate(output.group_ids)}
    base = output.k12[index["identity_target"]].float()
    mapping = contract["full_prefix_job_by_factor"]
    effects = {
        factor: output.k12[index[job]].float() - base
        for factor, job in mapping.items()
    }
    diagnostics = {}
    interactions = {
        "pair_qk": effects["qk"] - effects["q"] - effects["k"],
        "pair_qv": effects["qv"] - effects["q"] - effects["v"],
        "pair_kv": effects["kv"] - effects["k"] - effects["v"],
        "three_way": (
            effects["qkv"]
            - effects["qk"]
            - effects["qv"]
            - effects["kv"]
            + effects["q"]
            + effects["k"]
            + effects["v"]
        ),
    }
    for name, value in interactions.items():
        aligned, norm_ratio, cosine = masked_relation(value, effects["qkv"], mask)
        diagnostics[f"{name}.aligned_ratio"] = aligned
        diagnostics[f"{name}.norm_ratio"] = norm_ratio
        diagnostics[f"{name}.cosine"] = cosine
    full_increment = effects["qkv"] - effects["q"]
    for region in contract["region_classification"]["atomic_regions"]:
        increment = (
            output.k12[index[f"{region}.qkv"]].float()
            - output.k12[index["q"]].float()
        )
        aligned, norm_ratio, cosine = masked_relation(increment, full_increment, mask)
        diagnostics[f"region.{region}.incremental_aligned_recovery"] = aligned
        diagnostics[f"region.{region}.incremental_norm_ratio"] = norm_ratio
        diagnostics[f"region.{region}.incremental_cosine"] = cosine
    return diagnostics


def run_microbatch(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    vector: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor], dict[str, Any], str]:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    live_names = ("correct_trigger", "irrelevant_trigger")
    captures = {name: realized.run(conditions[name]) for name in live_names}
    attention_states = {
        name: attention.run(conditions[name], contract["k12"]["layers"])
        for name in live_names
    }
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    states, diagnostics, audits = {}, {}, {}
    for direction, specification in contract["conditions"]["directions"].items():
        jobs, direction_audit, endpoints = jobs_for_direction(
            direction,
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
                endpoints["target"],
                endpoints["donor"],
                conditions[specification["target"]].response_mask,
            )
        for endpoint in ("target", "donor"):
            condition_name = specification[endpoint]
            states[f"natural.{direction}.{endpoint}"] = compact_natural(
                captures[condition_name],
                endpoints["target"],
                endpoints["donor"],
                component_ids,
                runner,
                probes,
            )
        diagnostics.update({
            f"{direction}.{key}": value
            for key, value in factorial_diagnostics(
                output, conditions[specification["target"]].response_mask, contract
            ).items()
        })
        audits[direction] = direction_audit
    hashes = {response_hash(condition) for condition in conditions.values()}
    if len(hashes) != 1 or runner.registered_hook_count() != 0:
        raise RuntimeError("Day 60 response tensors differ or hooks leaked")
    return states, diagnostics, audits, hashes.pop()


def write_shard(
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    state_parts: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    diagnostic_parts: Sequence[Mapping[str, torch.Tensor]],
    audit_parts: Sequence[Mapping[str, Any]],
    response_hashes: Sequence[str],
    commit: str,
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
    tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
    if len({tuple(sorted(part)) for part in state_parts}) != 1:
        raise RuntimeError("Day 60 microbatch state sets differ")
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
    for key in diagnostic_parts[0]:
        tensors[f"diagnostic.{key}"] = torch.cat(
            [part[key] for part in diagnostic_parts], dim=0
        ).detach().cpu().contiguous().clone()
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Day 60 shard contains nonfinite tensors")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day60-fresh-qkv-source-factorial-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "batch_index": batch_index,
        "concept": batch[0]["concept"],
        "example_ids": [row["example_id"] for row in batch],
        "response_hashes": list(response_hashes),
        "state_names": sorted(combined),
        "diagnostic_names": sorted(diagnostic_parts[0]),
        "audits": list(audit_parts),
        "tensor_sha256": sha256_file(tensor_path),
    })


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
    name = "irrelevant_trigger"
    capture = realized.run(conditions[name])
    attention_state = {name: attention.run(conditions[name], contract["k12"]["layers"])}
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    natural = selected_values(capture, component_ids, runner.layers).float()
    recompute_errors = []
    for region in contract["factorial"]["source_regions"]:
        for factors in contract["factorial"]["nonempty_factor_subsets"]:
            replacements = factor_replacements(
                factors,
                region,
                name,
                name,
                attention_state,
                partitions,
                component_ids,
                contract,
            )
            recomputed = torch.stack(
                [replacements[component_id] for component_id in component_ids], dim=2
            )
            recompute_errors.append(float((recomputed - natural).abs().max()))
    identity = make_job(
        "identity_target",
        capture,
        source_replacements(capture, capture, component_ids, runner.layers),
        runner,
    )
    output = vector.run(conditions[name], [identity])
    natural_payload = compact_natural(
        capture, natural, natural + 1.0, component_ids, runner, probes
    )
    k12_error = float((output.k12[0] - natural).abs().max())
    margin_error = float((output.mean_margins[0] - natural_payload["margins"]).abs().max())
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "natural_attention_recompute": max(recompute_errors) <= float(gates["natural_attention_recompute_max_abs"]),
        "identity_k12": k12_error <= float(gates["identity_k12_max_abs"]),
        "identity_margins": margin_error <= float(gates["identity_monitor_margin_max_abs"]),
        "response_exact": torch.equal(output.response_ids, conditions[name].response_ids) and torch.equal(output.response_mask, conditions[name].response_mask),
        "finite": bool(torch.isfinite(output.monitor_values).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "day60-candidate-blind-factorial-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_id": batch[0]["example_id"],
        "natural_attention_recompute_max_abs": max(recompute_errors),
        "identity_k12_max_abs": k12_error,
        "identity_monitor_margin_max_abs": margin_error,
        "candidate_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError("Day 60 preflight failed")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH, ROOT / "src/neural_chameleon/upstream_controller.py"):
        require_committed(path, commit)
    contract = expanded_contract()
    records = load_records(contract)
    batches = grouped_batches(records, int(contract["panel"]["shard_batch_size"]))
    indexed = list(enumerate(batches))
    if args.batch_index:
        requested = set(args.batch_index)
        indexed = [item for item in indexed if item[0] in requested]
    runner = load_runner(contract, "chameleon")
    _probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        preflight(
            batches[0][:1], runner, realized, attention, vector, probes,
            component_ids, contract, commit,
        )
        return
    report = read_json(PREFLIGHT_PATH)
    if report.get("result") != "pass" or report.get("execution_commit") != commit or report.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise RuntimeError("Day 60 preflight is not exact and passing")
    started, completed = time.perf_counter(), 0
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in indexed:
        stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
        tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if metadata.get("execution_commit") == commit and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH) and metadata.get("tensor_sha256") == sha256_file(tensor_path):
                completed += 1
                continue
        states, diagnostics, audits, hashes = [], [], [], []
        for row in batch:
            state_part, diagnostic_part, audit_part, token_hash = run_microbatch(
                [row], runner, realized, attention, vector, probes, component_ids, contract
            )
            states.append(state_part)
            diagnostics.append(diagnostic_part)
            audits.append(audit_part)
            hashes.append(token_hash)
            gc.collect()
            torch.cuda.empty_cache()
        write_shard(batch_index, batch, states, diagnostics, audits, hashes, commit)
        completed += 1
        print(json.dumps({"batch_index": batch_index, "concept": batch[0]["concept"]}), flush=True)
    valid = []
    for path in sorted(SHARD_DIR.glob("*.json")):
        metadata = read_json(path)
        if metadata.get("execution_commit") == commit and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH):
            valid.append(metadata)
    execution = {
        "schema_version": 1,
        "procedure": "day60-fresh-qkv-source-factorial-v1",
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
    if not execution["complete"] and not args.batch_index:
        raise RuntimeError("Day 60 execution is incomplete")


if __name__ == "__main__":
    main()
