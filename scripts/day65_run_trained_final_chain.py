#!/usr/bin/env python3
"""Run the locked trained-concept semantic→K/V→K12→monitor final chain."""

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
from day45_run_prototype_population import mean_margins, require_committed, write_json_atomic  # noqa: E402
from day49_run_prompt_memory import make_job  # noqa: E402
from day57_run_confirm_trace_acquisition import chunked_vector_run, components_by_layer, load_runner  # noqa: E402
from day58_run_k12_context_and_pathway import orthogonal_replacement_group  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from day64_run_calibration_prototypes import full_prefix_mask, pooled_full_prefix_kv  # noqa: E402
from neural_chameleon import RealizedForwardRunner, VectorizedUpstreamRunner, prompt_qkv_factor_operation  # noqa: E402
from neural_chameleon.controller_actuator import build_source_mask_partition  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import AttentionStateCaptureRunner, source_replacements  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-64/frozen-trained-final-contract.json"
PROGRAM_PATH = ROOT / "results/day-64/frozen-trained-title-gate-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
FINAL_CAUSAL_PATH = ROOT / "data/splits/day64-v1/final-causal.LOCKED.jsonl"
FINAL_NEGATIVE_PATH = ROOT / "data/splits/day64-v1/final-negative.LOCKED.jsonl"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"
PREFLIGHT_DIR = ROOT / "results/day-65/preflight"
EXECUTION_DIR = ROOT / "results/day-65/execution"
SHARD_DIR = ROOT / "artifacts/trained-final-title-gate-v1/final-chain"
LAYERS = (9, 10, 11, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("chameleon", "exact_precursor"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--concept", action="append")
    parser.add_argument("--negative", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def response_hash(condition: Any) -> str:
    digest = hashlib.sha256()
    digest.update(condition.response_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(condition.response_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract, program = read_json(CONTRACT_PATH), read_json(PROGRAM_PATH)
    if contract["status"] != "frozen_after_calibration_before_any_locked_final_model_or_intervention_outcome":
        raise RuntimeError("final contract is not frozen")
    if contract["program_sha256"] != sha256_file(PROGRAM_PATH):
        raise RuntimeError("final parent program differs")
    for role, path in (("final_causal", FINAL_CAUSAL_PATH), ("final_negative", FINAL_NEGATIVE_PATH)):
        if contract["roles"][role]["sha256"] != sha256_file(path):
            raise RuntimeError(f"locked {role} differs")
    return contract, program


def partitions(
    runner: Any, conditions: Mapping[str, Any], prompts: Sequence[str], triggers: Sequence[str]
) -> dict[str, Any]:
    return {
        name: build_source_mask_partition(runner.tokenizer, conditions[name], prompts, trigger=trigger)
        for name, trigger in zip(triggers, triggers)
    }


def full_prefix_kv_replacements(
    source_name: str,
    target_name: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    source_partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    runner: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for layer, components in components_by_layer(component_ids).items():
        heads = tuple(int(component.head) for component in components)
        source, target = attention_states[source_name][layer], attention_states[target_name][layer]
        source_mask = full_prefix_mask(source_partitions[source_name])
        target_mask = full_prefix_mask(source_partitions[target_name])
        changed = prompt_qkv_factor_operation(
            source, target, heads, source_mask, target_mask, source_factors=("k", "v")
        )
        baseline = prompt_qkv_factor_operation(
            target, target, heads, target_mask, target_mask, source_factors=("k", "v")
        )
        target_values = selected_values(
            captures[target_name], [component.component_id for component in components], runner.layers
        ).float()
        for index, component in enumerate(components):
            result[component.component_id] = (
                target_values[:, :, index]
                + changed[:, :, int(component.head)]
                - baseline[:, :, int(component.head)]
            ).clone()
    return result


def compact(output: Any, index: int) -> dict[str, torch.Tensor]:
    return {
        "margins": output.mean_margins[index].float(),
        "k12": response_mean(output.k12[index].float(), output.response_mask),
        "activation_rms": output.activation_rms[index].float(),
    }


def run_single(
    row: Mapping[str, Any],
    concept: str,
    pair: Mapping[str, Any],
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    names_to_triggers = {
        "correct": pair["correct_trigger"],
        "irrelevant": pair["irrelevant_trigger"],
        "different": pair["different_trigger"],
    }
    conditions = prepare_conditions(runner, [row], tuple(names_to_triggers.values()))
    named_conditions = {name: conditions[trigger] for name, trigger in names_to_triggers.items()}
    prompts = [row["prompt"]]
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    captures = {name: realized.run(condition) for name, condition in named_conditions.items()}
    attention_states = {name: attention.run(condition, LAYERS) for name, condition in named_conditions.items()}
    source_partitions = {
        name: build_source_mask_partition(runner.tokenizer, named_conditions[name], prompts, trigger=trigger)
        for name, trigger in names_to_triggers.items()
    }
    states: dict[str, dict[str, torch.Tensor]] = {
        f"natural.{name}": {
            "margins": mean_margins(capture.monitor_residual, probes).T.float(),
            "k12": response_mean(
                selected_values(capture, component_ids, runner.layers).float(), capture.response_mask
            ),
            "kv": pooled_full_prefix_kv(attention_states[name], source_partitions[name], component_ids),
        }
        for name, capture in captures.items()
    }
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    audits: dict[str, Any] = {}
    direction_specs = (
        ("correct_to_irrelevant", "irrelevant", "correct"),
        ("irrelevant_to_correct", "correct", "irrelevant"),
    )
    for direction_index, (direction, target_name, donor_name) in enumerate(direction_specs):
        target, donor = captures[target_name], captures[donor_name]
        orthogonal, audit = orthogonal_replacement_group(
            target, donor, component_ids, runner, seed=seed + direction_index * 100
        )
        audits[direction] = audit
        target_replacements = {
            "identity_target": source_replacements(target, target, component_ids, runner.layers),
            "donor_kv_into_target": full_prefix_kv_replacements(
                donor_name, target_name, captures, attention_states, source_partitions, component_ids, runner
            ),
            "different_kv_into_target": full_prefix_kv_replacements(
                "different", target_name, captures, attention_states, source_partitions, component_ids, runner
            ),
            "matched_orthogonal_k12_into_target": orthogonal,
        }
        target_output = chunked_vector_run(
            vector,
            named_conditions[target_name],
            [make_job(name, target, replacement, runner) for name, replacement in target_replacements.items()],
            maximum_jobs_per_forward=1,
        )
        for index, name in enumerate(target_output.group_ids):
            states[f"{direction}.{name}"] = compact(target_output, index)

        donor_replacements = {
            "identity_donor": source_replacements(donor, donor, component_ids, runner.layers),
            "target_kv_into_donor": full_prefix_kv_replacements(
                target_name, donor_name, captures, attention_states, source_partitions, component_ids, runner
            ),
            "target_k12_into_donor": source_replacements(donor, target, component_ids, runner.layers),
            "target_kv_into_donor_plus_donor_k12_restore": source_replacements(
                donor, donor, component_ids, runner.layers
            ),
        }
        donor_output = chunked_vector_run(
            vector,
            named_conditions[donor_name],
            [make_job(name, donor, replacement, runner) for name, replacement in donor_replacements.items()],
            maximum_jobs_per_forward=1,
        )
        for index, name in enumerate(donor_output.group_ids):
            states[f"{direction}.{name}"] = compact(donor_output, index)
    hashes = {response_hash(condition) for condition in named_conditions.values()}
    if len(hashes) != 1 or runner.registered_hook_count() != 0:
        raise RuntimeError("response tokens differ or hooks leaked")
    return states, {"orthogonal": audits, "response_hash": hashes.pop(), "concept": concept}


def write_concept_shard(
    model: str,
    concept: str,
    rows: Sequence[Mapping[str, Any]],
    state_parts: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    audits: Sequence[Mapping[str, Any]],
    commit: str,
) -> None:
    if not state_parts or any(set(part) != set(state_parts[0]) for part in state_parts):
        raise RuntimeError("concept state sets differ")
    tensors = {
        f"{state}.{field}": torch.cat([part[state][field] for part in state_parts]).cpu().contiguous()
        for state in state_parts[0] for field in state_parts[0][state]
    }
    directory = SHARD_DIR / model
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = directory / f"{concept}.safetensors", directory / f"{concept}.json"
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day65-trained-final-chain-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "model": model,
        "concept": concept,
        "example_ids": [row["example_id"] for row in rows],
        "state_names": sorted(state_parts[0]),
        "orthogonal_audits": list(audits),
        "patched_component_count": 12,
        "source_operator": "full_monitoring_prefix_kv",
        "tail_regeneration": "natural_except_exact_declared_k12_raw_head_sites",
        "tensor_sha256": sha256_file(tensor_path),
    })


def run_negative(runner: Any, probes: Sequence[Any], commit: str) -> int:
    rows = read_jsonl(FINAL_NEGATIVE_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["content_sha256"], row)
    ordered = [unique[key] for key in sorted(unique)]
    parts = []
    for start in range(0, len(ordered), 4):
        batch = ordered[start:start + 4]
        condition = prepare_conditions(runner, batch, ("HTML",))["normal"]
        capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
        parts.append(mean_margins(capture.monitor_residual, probes).T.float())
    lookup = {row["content_sha256"]: index for index, row in enumerate(ordered)}
    tensor_path, metadata_path = SHARD_DIR / "chameleon" / "final-negative.safetensors", SHARD_DIR / "chameleon" / "final-negative.json"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file({"normal.margins": torch.cat(parts).cpu().contiguous()}, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day65-trained-final-negative-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "panel_sha256": sha256_file(FINAL_NEGATIVE_PATH),
        "unique_content_hashes_in_order": [row["content_sha256"] for row in ordered],
        "assignment_example_ids": [row["example_id"] for row in rows],
        "assignment_probe_concepts": [row["probe_concept"] for row in rows],
        "assignment_unique_indices": [lookup[row["content_sha256"]] for row in rows],
        "tensor_sha256": sha256_file(tensor_path),
    })
    return len(ordered)


def preflight(
    model: str, runner: Any, probes: Sequence[Any], component_ids: Sequence[str], contract: Mapping[str, Any], commit: str
) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    trigger = row["concept"]
    conditions = prepare_conditions(runner, [row], (trigger,))
    condition = conditions[trigger]
    capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12).run(condition, LAYERS)
    captures, attention_states = {"same": capture}, {"same": attention}
    source_partitions = {
        "same": build_source_mask_partition(runner.tokenizer, condition, [row["prompt"]], trigger=trigger)
    }
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    identity = make_job(
        "identity", capture, source_replacements(capture, capture, component_ids, runner.layers), runner
    )
    same_kv = make_job(
        "same_kv",
        capture,
        full_prefix_kv_replacements(
            "same", "same", captures, attention_states, source_partitions, component_ids, runner
        ),
        runner,
    )
    output = chunked_vector_run(vector, condition, (identity, same_kv), maximum_jobs_per_forward=1)
    natural_margins = mean_margins(capture.monitor_residual, probes).T.float()
    natural_k12 = response_mean(
        selected_values(capture, component_ids, runner.layers).float(), capture.response_mask
    )
    identity_k12 = response_mean(output.k12[0].float(), output.response_mask)
    same_kv_k12 = response_mean(output.k12[1].float(), output.response_mask)
    margin_error = float((output.mean_margins[0] - natural_margins).abs().max())
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "identity_k12_exact": float((identity_k12 - natural_k12).abs().max()) == 0.0,
        "identity_margin_within_gate": margin_error <= float(contract["implementation_gates"]["same_state_margin_max_abs"]),
        "same_kv_baseline_corrected_exact": float((same_kv_k12 - natural_k12).abs().max()) == 0.0,
        "finite": bool(torch.isfinite(output.k12).all() and torch.isfinite(output.mean_margins).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(PREFLIGHT_DIR / f"{model}.json", {
        "schema_version": 1,
        "procedure": "day65-candidate-blind-final-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "model": model,
        "candidate_final_outcomes_generated": False,
        "identity_margin_max_abs": margin_error,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"final preflight failed for {model}: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract, program = verify_contract()
    probe_names, probes = load_probes()
    if probe_names != contract["probe_names_in_order"]:
        raise RuntimeError("probe order differs")
    day57 = read_json(DAY57_PATH)
    component_ids = day57["k12"]["component_ids"]
    runner = load_runner(day57, args.model)
    if args.preflight_only:
        preflight(args.model, runner, probes, component_ids, contract, commit)
        return
    preflight_report = read_json(PREFLIGHT_DIR / f"{args.model}.json")
    if (
        preflight_report.get("result") != "pass"
        or preflight_report.get("execution_commit") != commit
        or preflight_report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("exact passing final preflight for this commit required")
    if args.negative and args.model != "chameleon":
        raise RuntimeError("final negatives run only for Chameleon")
    started = time.perf_counter()
    counts: dict[str, int] = {}
    if args.negative:
        counts["unique_final_negative_activations"] = run_negative(runner, probes, commit)
    selected = set(args.concept or contract["concepts_in_order"])
    unknown = selected - set(contract["concepts_in_order"])
    if unknown:
        raise RuntimeError(f"unfrozen concepts requested: {sorted(unknown)}")
    causal = read_jsonl(FINAL_CAUSAL_PATH)
    for concept_index, concept in enumerate(contract["concepts_in_order"]):
        if concept not in selected:
            continue
        rows = [row for row in causal if row["concept"] == concept]
        parts, audits = [], []
        for index, row in enumerate(rows):
            states, audit = run_single(
                row,
                concept,
                program["pairs"][concept],
                runner,
                probes,
                component_ids,
                65001 + concept_index * 1000 + index * 10,
            )
            parts.append(states)
            audits.append(audit)
            print(json.dumps({"model": args.model, "concept": concept, "completed": index + 1, "total": len(rows)}), flush=True)
        write_concept_shard(args.model, concept, rows, parts, audits, commit)
        counts[concept] = len(rows)
        gc.collect()
        torch.cuda.empty_cache()
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(EXECUTION_DIR / f"{args.model}.json", {
        "schema_version": 1,
        "procedure": "day65-trained-final-chain-execution-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "model": args.model,
        "counts_this_invocation": counts,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })


if __name__ == "__main__":
    main()
