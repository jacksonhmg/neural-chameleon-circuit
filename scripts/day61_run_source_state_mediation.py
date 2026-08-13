#!/usr/bin/env python3
"""Run the frozen Day 61 source-state K/V mediation scan on CUDA."""

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

from day44_run_k12_pilot import load_probes  # noqa: E402
from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day48_run_proximal_upstream import prepare_conditions  # noqa: E402
from day49_run_prompt_memory import make_job  # noqa: E402
from day52_run_reciprocal_reconfiguration import partitions_for_conditions  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    chunked_vector_run,
    grouped_batches,
    load_runner,
)
from day60_run_qkv_source_factorial import (  # noqa: E402
    expanded_contract as day60_contract,
    factor_replacements,
    load_records as day60_records,
)
from neural_chameleon import (  # noqa: E402
    FullSequenceSourceCaptureRunner,
    RealizedForwardRunner,
    SourceMediationRunner,
    VectorizedUpstreamRunner,
    aligned_source_replacement,
    flatten_aligned_kv,
    orthogonal_source_replacement,
    vector_relation,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-61/frozen-source-state-mediation-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
DAY60_CONTRACT_PATH = ROOT / "results/day-60/frozen-qkv-source-factorial-contract.json"
DAY60_SUMMARY_PATH = ROOT / "results/day-60/qkv-source-factorial-summary.json"
DAY60_VERIFICATION_PATH = ROOT / "results/day-60/local-reduction-verification.json"
PREFLIGHT_PATH = ROOT / "results/day-61/source-state-mediation-preflight.json"
EXECUTION_PATH = ROOT / "results/day-61/source-state-mediation-execution.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day61-source-state-mediation-shards"


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
    expected = {
        DAY57_CONTRACT_PATH: contract["parents"]["day57_contract_sha256"],
        DAY60_CONTRACT_PATH: contract["parents"]["day60_contract_sha256"],
        DAY60_SUMMARY_PATH: contract["parents"]["day60_summary_sha256"],
        DAY60_VERIFICATION_PATH: contract["parents"]["day60_local_verification_sha256"],
        ROOT / contract["panel"]["manifest_path"]: contract["panel"]["manifest_sha256"],
        ROOT / contract["panel"]["path"]: contract["panel"]["sha256"],
    }
    if any(sha256_file(path) != digest for path, digest in expected.items()):
        raise RuntimeError("Day 61 parent evidence differs")
    if contract["status"] != "frozen_before_any_day61_model_or_cross_condition_candidate_outcome":
        raise RuntimeError("Day 61 contract is not frozen")
    parent = day60_contract()
    for key in ("models", "conditions", "k12", "probes", "factorial"):
        contract[key] = parent[key]
    return contract


def informative_mask(partition: Any) -> torch.Tensor:
    from neural_chameleon.controller_actuator import SourceRegion

    result = torch.zeros_like(
        partition.masks[SourceRegion.NAMED_CONCEPT], dtype=torch.bool
    )
    for region in (SourceRegion.NAMED_CONCEPT, SourceRegion.TRIGGER_OTHER):
        result |= partition.masks[region].bool()
    if not bool(result.any(dim=1).all()):
        raise RuntimeError("Day 61 informative source mask is empty")
    return result


def masked_vector(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.bool()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return values.float()[expanded.expand_as(values)].reshape(-1)


def relation_tensor(
    changed: torch.Tensor,
    target: torch.Tensor,
    endpoint: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    effect = changed - target
    reference = endpoint - target
    if mask is not None:
        effect, reference = masked_vector(effect, mask), masked_vector(reference, mask)
    values = vector_relation(effect, reference)
    return {key: torch.tensor([value], dtype=torch.float32) for key, value in values.items()}


def kv_relation(
    changed: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    donor: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    donor_flat, target_flat = flatten_aligned_kv(donor, target)
    changed_flat, repeated_target = flatten_aligned_kv(changed, target)
    if not torch.equal(target_flat, repeated_target):
        raise RuntimeError("Day 61 K/V target flattening differs")
    values = vector_relation(changed_flat - target_flat, donor_flat - target_flat)
    return {key: torch.tensor([value], dtype=torch.float32) for key, value in values.items()}


def select_kv_layer(
    values: Mapping[str, torch.Tensor], layer: int
) -> dict[str, torch.Tensor]:
    prefix = f"layer_{layer:02d}."
    result = {key: value for key, value in values.items() if key.startswith(prefix)}
    if not result:
        raise RuntimeError(f"Day 61 K/V layer {layer} has no selected components")
    return result


def compact_candidate(
    result: Any,
    target: Any,
    endpoint: Any,
    donor: Any,
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    payload = {"margins": result.mean_margins.float()}
    k12 = relation_tensor(
        result.k12,
        target.k12,
        endpoint.k12,
        target.response_mask,
    )
    kv = kv_relation(result.kv_by_component, target.kv_by_component, donor.kv_by_component)
    payload.update({f"k12_{key}": value for key, value in k12.items()})
    payload.update({f"kv_{key}": value for key, value in kv.items()})
    ordered = tuple(sorted(component_ids))
    for layer in contract["k12"]["layers"]:
        indices = [index for index, value in enumerate(ordered) if value.startswith(f"layer_{layer:02d}.")]
        layer_relation = relation_tensor(
            result.k12[:, :, indices],
            target.k12[:, :, indices],
            endpoint.k12[:, :, indices],
            target.response_mask,
        )
        payload.update({
            f"k12_layer_{layer:02d}_{key}": value
            for key, value in layer_relation.items()
        })
        layer_kv = kv_relation(
            select_kv_layer(result.kv_by_component, layer),
            select_kv_layer(target.kv_by_component, layer),
            select_kv_layer(donor.kv_by_component, layer),
        )
        payload.update({
            f"kv_layer_{layer:02d}_{key}": value
            for key, value in layer_kv.items()
        })
    return payload


def endpoint_jobs(
    source_name: str,
    target_name: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    runner: Any,
) -> list[Any]:
    target = captures[target_name]
    identity = make_job(
        "identity",
        target,
        source_replacements(target, target, component_ids, runner.layers),
        runner,
    )
    changed = factor_replacements(
        "kv",
        "monitoring_prefix",
        source_name,
        target_name,
        attention_states,
        partitions,
        component_ids,
        contract,
    )
    return [identity, make_job("day60_kv", target, changed, runner)]


def run_microbatch(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    realized: Any,
    attention: Any,
    source_capture: Any,
    mediation: Any,
    vector: Any,
    component_ids: Sequence[str],
    probes: Sequence[Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], str]:
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
    masks = {name: informative_mask(partitions[name]) for name in live_names}
    candidates = tuple(contract["candidate_order"])
    source_states = {
        name: source_capture.run(conditions[name], candidates) for name in live_names
    }
    natural = {
        name: mediation.run(conditions[name], masks[name]) for name in live_names
    }
    states: dict[str, dict[str, torch.Tensor]] = {}
    audits: dict[str, Any] = {}
    for direction, specification in contract["conditions"]["directions"].items():
        target_name, donor_name = specification["target"], specification["donor"]
        endpoints = chunked_vector_run(
            vector,
            conditions[target_name],
            endpoint_jobs(
                donor_name,
                target_name,
                captures,
                attention_states,
                partitions,
                component_ids,
                contract,
                runner,
            ),
        )
        target = natural[target_name]
        endpoint_target = type(target)(
            k12=endpoints.k12[0],
            mean_margins=endpoints.mean_margins[0],
            kv_by_component=target.kv_by_component,
            response_ids=target.response_ids,
            response_mask=target.response_mask,
        )
        endpoint_kv = type(target)(
            k12=endpoints.k12[1],
            mean_margins=endpoints.mean_margins[1],
            kv_by_component=natural[donor_name].kv_by_component,
            response_ids=target.response_ids,
            response_mask=target.response_mask,
        )
        identity_error = float((target.k12 - endpoint_target.k12).abs().max())
        margin_error = float((target.mean_margins - endpoint_target.mean_margins).abs().max())
        states[f"{direction}.endpoint.identity"] = {
            "margins": endpoints.mean_margins[0].float(),
            "k12_identity_max_abs": torch.tensor([identity_error]),
            "monitor_identity_max_abs": torch.tensor([margin_error]),
        }
        states[f"{direction}.endpoint.day60_kv"] = compact_candidate(
            endpoint_kv,
            target,
            endpoint_kv,
            natural[donor_name],
            component_ids,
            contract,
        )
        for candidate_index, candidate_id in enumerate(candidates):
            exact = aligned_source_replacement(
                source_states[target_name][candidate_id],
                source_states[donor_name][candidate_id],
                masks[target_name],
                masks[donor_name],
            )
            orthogonal, audit = orthogonal_source_replacement(
                source_states[target_name][candidate_id],
                exact,
                masks[target_name],
                seed=int(contract["controls"]["orthogonal_base_seed"])
                + candidate_index,
            )
            exact_result = mediation.run(
                conditions[target_name],
                masks[target_name],
                candidate_id=candidate_id,
                replacement=exact,
            )
            orthogonal_result = mediation.run(
                conditions[target_name],
                masks[target_name],
                candidate_id=candidate_id,
                replacement=orthogonal,
            )
            states[f"{direction}.candidate.{candidate_id}"] = compact_candidate(
                exact_result,
                target,
                endpoint_kv,
                natural[donor_name],
                component_ids,
                contract,
            )
            states[f"{direction}.orthogonal.{candidate_id}"] = compact_candidate(
                orthogonal_result,
                target,
                endpoint_kv,
                natural[donor_name],
                component_ids,
                contract,
            )
            audits[f"{direction}.{candidate_id}"] = audit.to_dict()
    response_hashes = set()
    for condition in conditions.values():
        digest = hashlib.sha256()
        digest.update(condition.response_ids.numpy().tobytes())
        digest.update(condition.response_mask.numpy().tobytes())
        response_hashes.add(digest.hexdigest())
    if len(response_hashes) != 1 or runner.registered_hook_count() != 0:
        raise RuntimeError("Day 61 response tensors differ or hooks leaked")
    return states, audits, response_hashes.pop()


def write_shard(
    batch_index: int,
    batch: Sequence[Mapping[str, Any]],
    parts: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    audits: Sequence[Mapping[str, Any]],
    response_hashes: Sequence[str],
    commit: str,
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
    tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
    if len({tuple(sorted(part)) for part in parts}) != 1:
        raise RuntimeError("Day 61 microbatch state sets differ")
    tensors = {
        f"{state_name}.{field}": torch.cat(
            [part[state_name][field] for part in parts], dim=0
        ).detach().cpu().contiguous().clone()
        for state_name in parts[0]
        for field in parts[0][state_name]
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("Day 61 shard contains nonfinite tensors")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day61-source-state-kv-mediation-scan-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "batch_index": batch_index,
        "concept": batch[0]["concept"],
        "example_ids": [row["example_id"] for row in batch],
        "response_hashes": list(response_hashes),
        "state_names": sorted(parts[0]),
        "audits": list(audits),
        "tensor_sha256": sha256_file(tensor_path),
    })


def preflight(
    batch: Sequence[Mapping[str, Any]],
    runner: Any,
    source_capture: Any,
    mediation: Any,
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    pair_spec = contract["conditions"]["pairs"][batch[0]["concept"]]
    _pair, conditions = prepare_conditions(runner, batch, pair_spec)
    name = "irrelevant_trigger"
    partitions = partitions_for_conditions(
        runner, conditions, [row["prompt"] for row in batch], pair_spec
    )
    mask = informative_mask(partitions[name])
    candidates = tuple(contract["candidate_order"])
    captured = source_capture.run(conditions[name], candidates)
    natural = mediation.run(conditions[name], mask)
    k12_errors, margin_errors, kv_errors = [], [], []
    natural_kv, _ = flatten_aligned_kv(natural.kv_by_component, natural.kv_by_component)
    for candidate_id in candidates:
        replacement = aligned_source_replacement(
            captured[candidate_id], captured[candidate_id], mask, mask
        )
        result = mediation.run(
            conditions[name], mask, candidate_id=candidate_id, replacement=replacement
        )
        changed_kv, _ = flatten_aligned_kv(result.kv_by_component, natural.kv_by_component)
        k12_errors.append(float((result.k12 - natural.k12).abs().max()))
        margin_errors.append(float((result.mean_margins - natural.mean_margins).abs().max()))
        kv_errors.append(float((changed_kv - natural_kv).abs().max()))
    gates = contract["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_count": len(candidates) == int(gates["candidate_count"]),
        "same_state_k12": max(k12_errors) <= float(gates["same_state_k12_max_abs"]),
        "same_state_monitor": max(margin_errors) <= float(gates["same_state_monitor_margin_max_abs"]),
        "same_state_pre_rope_kv": max(kv_errors) <= float(gates["same_state_pre_rope_kv_max_abs"]),
        "finite": bool(
            torch.isfinite(natural.k12).all()
            and torch.isfinite(natural.mean_margins).all()
        ),
        "response_exact": torch.equal(natural.response_ids, conditions[name].response_ids)
        and torch.equal(natural.response_mask, conditions[name].response_mask),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "day61-candidate-blind-source-mediation-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_id": batch[0]["example_id"],
        "candidate_outcomes_generated": False,
        "same_state_k12_max_abs": max(k12_errors),
        "same_state_monitor_margin_max_abs": max(margin_errors),
        "same_state_pre_rope_kv_max_abs": max(kv_errors),
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    write_json_atomic(PREFLIGHT_PATH, report)
    if report["result"] != "pass":
        raise RuntimeError("Day 61 preflight failed")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        ROOT / "src/neural_chameleon/source_mediation.py",
    ):
        require_committed(path, commit)
    contract = expanded_contract()
    records = day60_records(day60_contract())
    batches = grouped_batches(records, 4)
    indexed = list(enumerate(batches))
    if args.batch_index:
        requested = set(args.batch_index)
        indexed = [value for value in indexed if value[0] in requested]
    runner = load_runner(contract, "chameleon")
    _probe_names, probes = load_probes()
    component_ids = tuple(contract["k12"]["component_ids"])
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    source_capture = FullSequenceSourceCaptureRunner(runner, monitor_layer=12)
    mediation = SourceMediationRunner(
        runner, probes, component_ids, monitor_layer=12
    )
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    if args.preflight_only:
        preflight(
            batches[0][:1], runner, source_capture, mediation, contract, commit
        )
        return
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Day 61 preflight is not exact and passing")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_index, batch in indexed:
        stem = f"{batch_index:03d}-{batch[0]['concept'].replace('/', '_')}"
        tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
        if metadata_path.exists() and tensor_path.exists():
            metadata = read_json(metadata_path)
            if (
                metadata.get("execution_commit") == commit
                and metadata.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and metadata.get("tensor_sha256") == sha256_file(tensor_path)
            ):
                continue
        parts, audits, hashes = [], [], []
        for row in batch:
            states, audit, response_hash = run_microbatch(
                [row],
                runner,
                realized,
                attention,
                source_capture,
                mediation,
                vector,
                component_ids,
                probes,
                contract,
            )
            parts.append(states)
            audits.append(audit)
            hashes.append(response_hash)
            gc.collect()
            torch.cuda.empty_cache()
        write_shard(batch_index, batch, parts, audits, hashes, commit)
        print(json.dumps({"batch_index": batch_index, "concept": batch[0]["concept"]}), flush=True)
    valid = [
        read_json(path)
        for path in sorted(SHARD_DIR.glob("*.json"))
        if read_json(path).get("execution_commit") == commit
        and read_json(path).get("contract_sha256") == sha256_file(CONTRACT_PATH)
    ]
    execution = {
        "schema_version": 1,
        "procedure": "day61-source-state-kv-mediation-scan-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "complete": len(valid) == len(batches),
        "shards": len(valid),
        "examples": sum(len(value["example_ids"]) for value in valid),
        "state_rows": sum(len(value["state_names"]) * len(value["example_ids"]) for value in valid),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    write_json_atomic(EXECUTION_PATH, execution)
    if not execution["complete"] and not args.batch_index:
        raise RuntimeError("Day 61 execution is incomplete")


if __name__ == "__main__":
    main()
