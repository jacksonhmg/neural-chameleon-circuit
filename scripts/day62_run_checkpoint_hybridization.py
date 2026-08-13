#!/usr/bin/env python3
"""Run one frozen coarse Chameleon/precursor checkpoint hybridization pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import load_probes  # noqa: E402
from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from day62_run_qualification_calibration import natural_state, prepare_conditions  # noqa: E402
from neural_chameleon import RealizedForwardRunner  # noqa: E402
from neural_chameleon.post_gate1_interventions import AttentionStateCaptureRunner  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
DAY46_CONTRACT_PATH = ROOT / "results/day-46/frozen-selected-parameter-swap-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
DAY60_PANEL_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"
CHAMELEON_DIR = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PRECURSOR_DIR = ROOT / "artifacts/models/gemma-2-9b-it-abliterated"
PREFLIGHT_PATH = ROOT / "results/day-62/checkpoint-hybridization-preflight.json"
EXECUTION_PATH = ROOT / "results/day-62/checkpoint-hybridization-execution.json"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/checkpoint-hybridization-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


class CheckpointReader:
    """Keep sharded safetensor readers open for bounded live parameter copies."""

    def __init__(self, directory: Path, stack: ExitStack):
        index = read_json(directory / "model.safetensors.index.json")
        self.weight_map = index["weight_map"]
        self.readers = {
            filename: stack.enter_context(safe_open(directory / filename, framework="pt", device="cpu"))
            for filename in sorted(set(self.weight_map.values()))
        }

    def get(self, name: str) -> torch.Tensor:
        return self.readers[self.weight_map[name]].get_tensor(name)


def head_slice(parameter: torch.Tensor, projection: str, head: int, head_dim: int) -> torch.Tensor:
    return parameter[:, head * head_dim : (head + 1) * head_dim] if projection == "o_proj" else parameter[head * head_dim : (head + 1) * head_dim]


def copy_full(parameter: torch.Tensor, source: torch.Tensor) -> None:
    if parameter.shape != source.shape:
        raise RuntimeError("hybrid whole-parameter shape differs")
    parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def copy_heads(parameter: torch.Tensor, source: torch.Tensor, projection: str, heads: Sequence[int], head_dim: int) -> None:
    for head in heads:
        target = head_slice(parameter, projection, int(head), head_dim)
        replacement = head_slice(source, projection, int(head), head_dim)
        target.copy_(replacement.to(device=target.device, dtype=target.dtype))


def apply_group(runner: Any, reader: CheckpointReader, group: str, selected: Mapping[str, Any]) -> int:
    changed = 0
    with torch.no_grad():
        if group == "early_representation":
            for name, parameter in runner.model.named_parameters():
                if any(name.startswith(f"model.layers.{layer}.") for layer in range(9)):
                    copy_full(parameter, reader.get(name))
                    changed += 1
            return changed
        if group not in {"k12_kv_readout", "tail_attention_complement"}:
            raise ValueError(f"unknown hybrid group: {group}")
        for layer in (9, 10, 11, 12):
            attention = runner.layers[layer].self_attn
            selected_kv = set(int(value) for value in selected[str(layer)]["key_value"])
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parameter = getattr(attention, projection).weight
                name = f"model.layers.{layer}.self_attn.{projection}.weight"
                source = reader.get(name)
                if group == "k12_kv_readout":
                    heads = sorted(selected_kv) if projection in {"k_proj", "v_proj"} else []
                elif projection in {"q_proj", "o_proj"}:
                    heads = list(range(16))
                else:
                    heads = sorted(set(range(8)) - selected_kv)
                copy_heads(parameter, source, projection, heads, 256)
                changed += len(heads)
    return changed


def set_state(runner: Any, chameleon: CheckpointReader, precursor: CheckpointReader, state: str, selected: Mapping[str, Any]) -> int:
    # Restore the complete mutable union before constructing each exact hybrid.
    for group in ("early_representation", "k12_kv_readout", "tail_attention_complement"):
        apply_group(runner, chameleon, group, selected)
    groups = {
        "chameleon": (),
        "early_representation": ("early_representation",),
        "k12_kv_readout": ("k12_kv_readout",),
        "tail_attention_complement": ("tail_attention_complement",),
        "late_attention_union": ("k12_kv_readout", "tail_attention_complement"),
        "distributed_combination": ("early_representation", "k12_kv_readout", "tail_attention_complement"),
    }[state]
    return sum(apply_group(runner, precursor, group, selected) for group in groups)


def write_shard(concept: str, rows: Sequence[Mapping[str, Any]], states: Mapping[str, Mapping[str, torch.Tensor]], commit: str) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = SHARD_DIR / f"{concept}.safetensors", SHARD_DIR / f"{concept}.json"
    tensors = {f"{state}.{field}": value.detach().cpu().contiguous() for state, payload in states.items() for field, value in payload.items()}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("hybrid shard is nonfinite")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day62-timeboxed-checkpoint-hybridization-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "concept": concept,
        "example_ids": [row["example_id"] for row in rows],
        "state_names": sorted(states),
        "tensor_sha256": sha256_file(tensor_path),
    })


def preflight(runner: Any, probes: Sequence[Any], component_ids: Sequence[str], selected: Mapping[str, Any], contract: Mapping[str, Any], commit: str) -> None:
    row = json.loads(DAY60_PANEL_PATH.read_text().splitlines()[0])
    pair = read_json(DAY57_CONTRACT_PATH)["conditions"]["pairs"][row["concept"]]
    conditions = prepare_conditions(runner, [row], (pair["correct_trigger"], pair["irrelevant_trigger"]))
    realized, attention = RealizedForwardRunner(runner, monitor_layer=12), AttentionStateCaptureRunner(runner, monitor_layer=12)
    prompts = [row["prompt"]]
    first = natural_state(runner, realized, attention, probes, component_ids, conditions[pair["correct_trigger"]], prompts, pair["correct_trigger"])
    with ExitStack() as stack:
        chameleon, precursor = CheckpointReader(CHAMELEON_DIR, stack), CheckpointReader(PRECURSOR_DIR, stack)
        changed = set_state(runner, chameleon, precursor, "k12_kv_readout", selected)
        restored = set_state(runner, chameleon, precursor, "chameleon", selected)
    second = natural_state(runner, realized, attention, probes, component_ids, conditions[pair["correct_trigger"]], prompts, pair["correct_trigger"])
    errors = {field: float((first[field] - second[field]).abs().max()) for field in first}
    checks = {
        "cuda": runner.device.type == "cuda",
        "mutation_nonempty": changed > 0,
        "restore_scope_nonempty": restored == 0,
        "restore_exact": max(errors.values()) == 0.0,
        "finite": all(torch.isfinite(value).all() for value in second.values()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day62-candidate-blind-checkpoint-hybridization-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "candidate_outcomes_generated": False,
        "restore_endpoint_max_abs": errors,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"hybrid preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract, day57, day46 = read_json(CONTRACT_PATH), read_json(DAY57_CONTRACT_PATH), read_json(DAY46_CONTRACT_PATH)
    runner = load_runner(day57, "chameleon")
    _probe_names, probes = load_probes()
    component_ids = day57["k12"]["component_ids"]
    selected = day46["head_sets"]["selected"]
    if args.preflight_only:
        preflight(runner, probes, component_ids, selected, contract, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if report.get("result") != "pass" or report.get("execution_commit") != commit:
        raise RuntimeError("exact passing hybrid preflight is required")
    rows = [json.loads(line) for line in DAY60_PANEL_PATH.read_text().splitlines() if line]
    by_concept = {concept: [row for row in rows if row["concept"] == concept] for concept in sorted({row["concept"] for row in rows})}
    realized, attention = RealizedForwardRunner(runner, monitor_layer=12), AttentionStateCaptureRunner(runner, monitor_layer=12)
    states_order = contract["hybridization"]["states_in_order"]
    started = time.perf_counter()
    with ExitStack() as stack:
        chameleon, precursor = CheckpointReader(CHAMELEON_DIR, stack), CheckpointReader(PRECURSOR_DIR, stack)
        accumulated: dict[str, dict[str, Mapping[str, torch.Tensor]]] = {concept: {} for concept in by_concept}
        for state in states_order:
            set_state(runner, chameleon, precursor, state, selected)
            for concept, batch in by_concept.items():
                pair = day57["conditions"]["pairs"][concept]
                conditions = prepare_conditions(runner, batch, (pair["correct_trigger"], pair["irrelevant_trigger"]))
                prompts = [row["prompt"] for row in batch]
                for label, trigger in (("correct", pair["correct_trigger"]), ("irrelevant", pair["irrelevant_trigger"])):
                    accumulated[concept][f"{state}.{label}"] = natural_state(
                        runner, realized, attention, probes, component_ids, conditions[trigger], prompts, trigger
                    )
            print(json.dumps({"hybrid_state": state}), flush=True)
        set_state(runner, chameleon, precursor, "chameleon", selected)
    for concept, states in accumulated.items():
        write_shard(concept, by_concept[concept], states, commit)
    write_json_atomic(EXECUTION_PATH, {
        "schema_version": 1,
        "procedure": "day62-timeboxed-checkpoint-hybridization-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "complete": True,
        "states": len(states_order),
        "concepts": len(by_concept),
        "examples": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })


if __name__ == "__main__":
    main()
