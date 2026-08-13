#!/usr/bin/env python3
"""Run the untouched Day 63 semantic-to-K/V-to-K12-to-monitor causal chain."""

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

from day44_run_k12_pilot import selected_values  # noqa: E402
from day45_run_prototype_population import (  # noqa: E402
    mean_margins,
    require_committed,
    write_json_atomic,
)
from day49_run_prompt_memory import make_job  # noqa: E402
from day57_run_confirm_trace_acquisition import (  # noqa: E402
    chunked_vector_run,
    components_by_layer,
    load_runner,
)
from day58_run_k12_context_and_pathway import orthogonal_replacement_group  # noqa: E402
from day62_run_qualification_calibration import (  # noqa: E402
    informative_mask,
    load_probes,
    pooled_kv,
    prepare_conditions,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedUpstreamRunner,
    prompt_qkv_factor_operation,
)
from neural_chameleon.controller_actuator import build_source_mask_partition  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    source_replacements,
)


CONTRACT_PATH = ROOT / "results/day-63/frozen-final-causal-chain-contract.json"
PROGRAM_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
FINAL_CAUSAL_PATH = ROOT / "data/splits/day62-v1/final-causal.LOCKED.jsonl"
FINAL_NEGATIVE_PATH = ROOT / "data/splits/day62-v1/final-negative.LOCKED.jsonl"
VALIDATION_PATH = ROOT / "data/splits/day62-v1/probe-validation.jsonl"
PREFLIGHT_DIR = ROOT / "results/day-63/preflight"
EXECUTION_DIR = ROOT / "results/day-63/execution"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/final-causal-chain-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("chameleon", "exact_precursor"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--pair", action="append")
    parser.add_argument("--negative", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def response_hash(condition: Any) -> str:
    digest = hashlib.sha256()
    digest.update(condition.response_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(condition.response_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    contract, program = read_json(CONTRACT_PATH), read_json(PROGRAM_PATH)
    if contract["status"] != "frozen_before_any_locked_final_model_or_intervention_outcome":
        raise RuntimeError("final causal contract is not frozen")
    if contract["parents"]["program_contract_sha256"] != sha256_file(PROGRAM_PATH):
        raise RuntimeError("final parent program differs")
    if contract["parents"]["day57_contract_sha256"] != sha256_file(DAY57_PATH):
        raise RuntimeError("Day 57 parent differs")
    for role, path in (
        ("final_causal", FINAL_CAUSAL_PATH),
        ("final_negative", FINAL_NEGATIVE_PATH),
    ):
        spec = contract["panels"][role]
        if spec["parent_sha256"] != sha256_file(path):
            raise RuntimeError(f"locked {role} role differs")
    return contract, program


def selected_records(path: Path, selected_ids: Sequence[str]) -> list[dict[str, Any]]:
    lookup = {row["example_id"]: row for row in read_jsonl(path)}
    if set(selected_ids) - set(lookup):
        raise RuntimeError("a frozen final example is missing")
    return [lookup[example_id] for example_id in selected_ids]


def partitions(
    runner: Any,
    conditions: Mapping[str, Any],
    prompts: Sequence[str],
    triggers: Sequence[str],
) -> dict[str, Any]:
    return {
        trigger: build_source_mask_partition(
            runner.tokenizer, conditions[trigger], prompts, trigger=trigger
        )
        for trigger in triggers
    }


def kv_replacements(
    source_name: str,
    target_name: str,
    captures: Mapping[str, Any],
    attention_states: Mapping[str, Mapping[int, Any]],
    source_partitions: Mapping[str, Any],
    component_ids: Sequence[str],
    runner: Any,
) -> dict[str, torch.Tensor]:
    """Return baseline-corrected K/V-only recomputed K12 raw head outputs."""
    result: dict[str, torch.Tensor] = {}
    for layer, components in components_by_layer(component_ids).items():
        heads = tuple(int(component.head) for component in components)
        source, target = attention_states[source_name][layer], attention_states[target_name][layer]
        source_mask = informative_mask(source_partitions[source_name])
        target_mask = informative_mask(source_partitions[target_name])
        changed = prompt_qkv_factor_operation(
            source,
            target,
            heads,
            source_mask,
            target_mask,
            source_factors=("k", "v"),
        )
        baseline = prompt_qkv_factor_operation(
            target,
            target,
            heads,
            target_mask,
            target_mask,
            source_factors=("k", "v"),
        )
        target_values = selected_values(
            captures[target_name],
            [component.component_id for component in components],
            runner.layers,
        ).float()
        for index, component in enumerate(components):
            result[component.component_id] = (
                target_values[:, :, index]
                + changed[:, :, int(component.head)]
                - baseline[:, :, int(component.head)]
            ).clone()
    return result


def compact_output(output: Any, index: int) -> dict[str, torch.Tensor]:
    return {
        "margins": output.mean_margins[index].float(),
        "k12": response_mean(output.k12[index].float(), output.response_mask),
        "activation_rms": output.activation_rms[index].float(),
    }


def run_single(
    row: Mapping[str, Any],
    pair: Mapping[str, Any],
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    orthogonal_seed: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    a, b, irrelevant = pair["concept_a"], pair["concept_b"], pair["irrelevant_trigger"]
    triggers = (a, b, irrelevant)
    conditions = prepare_conditions(runner, [row], triggers)
    prompts = [row["prompt"]]
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    captures = {name: realized.run(conditions[name]) for name in triggers}
    attention_states = {
        name: attention.run(conditions[name], (9, 10, 11, 12)) for name in triggers
    }
    source_partitions = partitions(runner, conditions, prompts, triggers)
    states: dict[str, dict[str, torch.Tensor]] = {
        f"natural.{name}": {
            "margins": mean_margins(captures[name].monitor_residual, probes).T.float(),
            "k12": response_mean(
                selected_values(captures[name], component_ids, runner.layers).float(),
                captures[name].response_mask,
            ),
            "kv": pooled_kv(
                attention_states[name], source_partitions[name], component_ids
            ),
        }
        for name in triggers
    }
    vector = VectorizedUpstreamRunner(
        runner, probes, component_ids, monitor_layer=12
    )
    audits: dict[str, Any] = {}
    for direction_index, (direction, target_name, donor_name) in enumerate(
        (("a_to_b", a, b), ("b_to_a", b, a))
    ):
        target, donor = captures[target_name], captures[donor_name]
        orthogonal, audit = orthogonal_replacement_group(
            target,
            donor,
            component_ids,
            runner,
            seed=orthogonal_seed + direction_index * 100,
        )
        audits[direction] = audit
        target_replacements = {
            "identity_target": source_replacements(
                target, target, component_ids, runner.layers
            ),
            "donor_kv_into_target": kv_replacements(
                donor_name,
                target_name,
                captures,
                attention_states,
                source_partitions,
                component_ids,
                runner,
            ),
            "irrelevant_kv_into_target": kv_replacements(
                irrelevant,
                target_name,
                captures,
                attention_states,
                source_partitions,
                component_ids,
                runner,
            ),
            "matched_orthogonal_k12_into_target": orthogonal,
        }
        target_jobs = [
            make_job(name, target, replacement, runner)
            for name, replacement in target_replacements.items()
        ]
        target_output = chunked_vector_run(
            vector, conditions[target_name], target_jobs, maximum_jobs_per_forward=1
        )
        for index, name in enumerate(target_output.group_ids):
            states[f"{direction}.{name}"] = compact_output(target_output, index)

        donor_replacements = {
            "identity_donor": source_replacements(
                donor, donor, component_ids, runner.layers
            ),
            "target_kv_into_donor": kv_replacements(
                target_name,
                donor_name,
                captures,
                attention_states,
                source_partitions,
                component_ids,
                runner,
            ),
            "target_k12_into_donor": source_replacements(
                donor, target, component_ids, runner.layers
            ),
            # K/V affects this selected interface only. Replacing that interface
            # with exact donor K12 is the prospectively declared restoration.
            "target_kv_into_donor_plus_donor_k12_restore": source_replacements(
                donor, donor, component_ids, runner.layers
            ),
        }
        donor_jobs = [
            make_job(name, donor, replacement, runner)
            for name, replacement in donor_replacements.items()
        ]
        donor_output = chunked_vector_run(
            vector, conditions[donor_name], donor_jobs, maximum_jobs_per_forward=1
        )
        for index, name in enumerate(donor_output.group_ids):
            states[f"{direction}.{name}"] = compact_output(donor_output, index)

    hashes = {response_hash(condition) for condition in conditions.values()}
    if len(hashes) != 1 or runner.registered_hook_count() != 0:
        raise RuntimeError("final semantic contexts differ in response tokens or leaked hooks")
    return states, {"orthogonal": audits, "response_hash": hashes.pop()}


def write_pair_shard(
    model_name: str,
    pair_id: str,
    rows: Sequence[Mapping[str, Any]],
    state_parts: Sequence[Mapping[str, Mapping[str, torch.Tensor]]],
    audits: Sequence[Mapping[str, Any]],
    contract_hash: str,
    commit: str,
) -> None:
    if not state_parts or any(set(part) != set(state_parts[0]) for part in state_parts):
        raise RuntimeError("final pair state sets differ")
    tensors = {
        f"{state}.{field}": torch.cat(
            [part[state][field] for part in state_parts], dim=0
        ).detach().cpu().contiguous()
        for state in state_parts[0]
        for field in state_parts[0][state]
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("final causal shard is nonfinite")
    directory = SHARD_DIR / model_name
    directory.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = (
        directory / f"{pair_id}.safetensors",
        directory / f"{pair_id}.json",
    )
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(
        metadata_path,
        {
            "schema_version": 1,
            "procedure": "day63-final-causal-chain-v1",
            "execution_commit": commit,
            "contract_sha256": contract_hash,
            "model": model_name,
            "pair_id": pair_id,
            "example_ids": [row["example_id"] for row in rows],
            "state_names": sorted(state_parts[0]),
            "orthogonal_audits": list(audits),
            "patched_component_count": 12,
            "tail_regeneration": "natural_except_exact_declared_k12_raw_head_sites",
            "tensor_sha256": sha256_file(tensor_path),
        },
    )


def run_negative(
    runner: Any,
    probes: Sequence[Any],
    contract: Mapping[str, Any],
    commit: str,
) -> int:
    if contract["selected_concepts_in_order"] == []:
        raise RuntimeError("no selected final concepts")
    rows = selected_records(
        FINAL_NEGATIVE_PATH,
        contract["panels"]["final_negative"]["selected_example_ids"],
    )
    directory = SHARD_DIR / "chameleon"
    directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    for concept in contract["selected_concepts_in_order"]:
        values = [row for row in rows if row["probe_concept"] == concept]
        parts = []
        for start in range(0, len(values), 4):
            batch = values[start : start + 4]
            condition = prepare_conditions(runner, batch, (concept,))["normal"]
            capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
            parts.append(mean_margins(capture.monitor_residual, probes).T.float())
        tensor_path, metadata_path = (
            directory / f"negative-{concept}.safetensors",
            directory / f"negative-{concept}.json",
        )
        temporary = tensor_path.with_suffix(".safetensors.tmp")
        save_file({"normal.margins": torch.cat(parts).contiguous()}, temporary)
        os.replace(temporary, tensor_path)
        write_json_atomic(
            metadata_path,
            {
                "schema_version": 1,
                "procedure": "day63-untouched-final-negative-v1",
                "execution_commit": commit,
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "concept": concept,
                "example_ids": [row["example_id"] for row in values],
                "tensor_sha256": sha256_file(tensor_path),
            },
        )
        completed += len(values)
    return completed


def preflight(
    model_name: str,
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    commit: str,
) -> None:
    row = read_jsonl(VALIDATION_PATH)[0]
    triggers = ("academic", "casual")
    conditions = prepare_conditions(runner, [row], triggers)
    prompts = [row["prompt"]]
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    captures = {name: realized.run(conditions[name]) for name in triggers}
    attention_states = {
        name: attention.run(conditions[name], (9, 10, 11, 12)) for name in triggers
    }
    source_partitions = partitions(runner, conditions, prompts, triggers)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    identity = make_job(
        "identity",
        captures["academic"],
        source_replacements(
            captures["academic"], captures["academic"], component_ids, runner.layers
        ),
        runner,
    )
    same_kv = make_job(
        "same_kv",
        captures["academic"],
        kv_replacements(
            "academic",
            "academic",
            captures,
            attention_states,
            source_partitions,
            component_ids,
            runner,
        ),
        runner,
    )
    output = chunked_vector_run(vector, conditions["academic"], (identity, same_kv))
    natural_margins = mean_margins(
        captures["academic"].monitor_residual, probes
    ).T.float()
    natural_k12 = response_mean(
        selected_values(captures["academic"], component_ids, runner.layers).float(),
        captures["academic"].response_mask,
    )
    identity_k12 = response_mean(output.k12[0].float(), output.response_mask)
    same_kv_k12 = response_mean(output.k12[1].float(), output.response_mask)
    margin_error = float((output.mean_margins[0] - natural_margins).abs().max())
    checks = {
        "cuda": runner.device.type == "cuda",
        "response_tokens_exact": len({response_hash(value) for value in conditions.values()}) == 1,
        "identity_k12_exact": float((identity_k12 - natural_k12).abs().max()) == 0.0,
        "identity_margin_within_gate": margin_error
        <= float(contract["implementation_gates"]["same_state_margin_max_abs"]),
        "same_kv_baseline_corrected_exact": float(
            (same_kv_k12 - natural_k12).abs().max()
        )
        == 0.0,
        "finite": bool(
            torch.isfinite(output.k12).all()
            and torch.isfinite(output.mean_margins).all()
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = PREFLIGHT_DIR / f"{model_name}.json"
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "procedure": "day63-candidate-blind-final-chain-preflight-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "model": model_name,
            "candidate_outcomes_generated": False,
            "identity_margin_max_abs": margin_error,
            "checks": checks,
            "result": "pass" if all(checks.values()) else "fail",
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"final chain preflight failed for {model_name}: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract, program = load_contract()
    _probe_names, probes = load_probes(program)
    component_ids = contract["k12"]["component_ids"]
    runner = load_runner(read_json(DAY57_PATH), args.model)
    if args.preflight_only:
        preflight(args.model, runner, probes, component_ids, contract, commit)
        return
    preflight_path = PREFLIGHT_DIR / f"{args.model}.json"
    report = read_json(preflight_path)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("exact passing final preflight for this commit is required")
    if args.negative and args.model != "chameleon":
        raise RuntimeError("frozen final negatives are evaluated only in Chameleon")

    started = time.perf_counter()
    counts: dict[str, int] = {}
    if args.negative:
        counts["final_negative"] = run_negative(runner, probes, contract, commit)
    selected_pairs = set(args.pair or contract["selected_pairs_in_order"])
    unknown = selected_pairs - set(contract["selected_pairs_in_order"])
    if unknown:
        raise RuntimeError(f"unfrozen pair requested: {sorted(unknown)}")
    causal_rows = selected_records(
        FINAL_CAUSAL_PATH,
        contract["panels"]["final_causal"]["selected_example_ids"],
    )
    for pair_index, pair in enumerate(contract["selected_pair_specs"]):
        pair_id = pair["pair_id"]
        if pair_id not in selected_pairs:
            continue
        rows = [row for row in causal_rows if row["pair_id"] == pair_id]
        parts, audits = [], []
        for index, row in enumerate(rows):
            states, audit = run_single(
                row,
                pair,
                runner,
                probes,
                component_ids,
                int(contract["controls"]["orthogonal_seed_base"])
                + pair_index * 1000
                + index * 10,
            )
            parts.append(states)
            audits.append(audit)
            print(
                json.dumps(
                    {
                        "model": args.model,
                        "pair": pair_id,
                        "completed": index + 1,
                        "total": len(rows),
                    }
                ),
                flush=True,
            )
        write_pair_shard(
            args.model,
            pair_id,
            rows,
            parts,
            audits,
            sha256_file(CONTRACT_PATH),
            commit,
        )
        counts[pair_id] = len(rows)
        gc.collect()
        torch.cuda.empty_cache()
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        EXECUTION_DIR / f"{args.model}.json",
        {
            "schema_version": 1,
            "procedure": "day63-final-causal-chain-execution-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "model": args.model,
            "counts_this_invocation": counts,
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )


if __name__ == "__main__":
    main()
