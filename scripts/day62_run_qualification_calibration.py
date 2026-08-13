#!/usr/bin/env python3
"""Run frozen new-concept qualification and negative-only calibration."""

from __future__ import annotations

import argparse
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
from day45_run_prototype_population import mean_margins, require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from neural_chameleon import LinearProbe, RealizedForwardRunner  # noqa: E402
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.controller_actuator import SourceRegion, build_source_mask_partition  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import AttentionStateCaptureRunner, query_to_kv_head  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
PROBE_SUMMARY_PATH = ROOT / "results/day-62/new-probe-training-summary.json"
PROBE_DIR = ROOT / "artifacts/final-title-gate-v1/probes"
ROLE_PATHS = {
    "qualification": ROOT / "data/splits/day62-v1/qualification.jsonl",
    "calibration": ROOT / "data/splits/day62-v1/calibration.jsonl",
}
PREFLIGHT_PATH = ROOT / "results/day-62/qualification-calibration-preflight.json"
EXECUTION_PATH = ROOT / "results/day-62/qualification-calibration-execution.json"
SHARD_DIR = ROOT / "artifacts/final-title-gate-v1/qualification-calibration-shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage", choices=tuple(ROLE_PATHS), action="append")
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


def load_records(stage: str, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    path, spec = ROLE_PATHS[stage], contract["roles"][stage]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"{stage} role differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != int(spec["rows"]):
        raise RuntimeError(f"{stage} row count differs")
    return rows


def load_probes(contract: Mapping[str, Any]) -> tuple[list[str], tuple[LinearProbe, ...]]:
    summary = read_json(PROBE_SUMMARY_PATH)
    if summary.get("contract_sha256") != sha256_file(CONTRACT_PATH):
        raise RuntimeError("new probe summary contract differs")
    names = list(contract["concepts"])
    probes = []
    for name in names:
        path = PROBE_DIR / f"{name}_weights.pt"
        if sha256_file(path) != summary["concepts"][name]["probe_sha256"]:
            raise RuntimeError(f"new probe differs: {name}")
        probes.append(LinearProbe.load(path))
    return names, tuple(probes)


def prepare_conditions(runner: Any, rows: Sequence[Mapping[str, Any]], triggers: Sequence[str]) -> dict[str, Any]:
    prompts, responses = [row["prompt"] for row in rows], [row["response"] for row in rows]
    response_ids, response_mask = runner._tokenize_responses_once(responses)
    conditions = {"normal": runner._prepare_condition("normal", prompts, response_ids, response_mask)}
    for trigger in triggers:
        users = [runner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=prompt) for prompt in prompts]
        conditions[trigger] = runner._prepare_condition(trigger, users, response_ids, response_mask)
    reference = conditions["normal"]
    if any(
        not torch.equal(value.response_ids, reference.response_ids)
        or not torch.equal(value.response_mask, reference.response_mask)
        for value in conditions.values()
    ):
        raise RuntimeError("qualification conditions do not share response tensors")
    return conditions


def informative_mask(partition: Any) -> torch.Tensor:
    return partition.masks[SourceRegion.NAMED_CONCEPT].bool() | partition.masks[SourceRegion.TRIGGER_OTHER].bool()


def pooled_kv(
    attention_states: Mapping[int, Any],
    partition: Any,
    component_ids: Sequence[str],
) -> torch.Tensor:
    values = []
    mask = informative_mask(partition)
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        state = attention_states[component.layer]
        query_heads = state.raw_head_output.shape[2]
        kv_heads = state.values.shape[1]
        kv_head = query_to_kv_head(int(component.head), query_heads, kv_heads)
        rows = []
        for row in range(mask.shape[0]):
            indices = torch.nonzero(mask[row], as_tuple=False).flatten()
            rows.append(torch.stack((state.keys[row, kv_head, indices].float().mean(0), state.values[row, kv_head, indices].float().mean(0))))
        values.append(torch.stack(rows))
    return torch.stack(values, dim=1)


def natural_state(
    runner: Any,
    realized: RealizedForwardRunner,
    attention: AttentionStateCaptureRunner,
    probes: Sequence[LinearProbe],
    component_ids: Sequence[str],
    condition: Any,
    prompts: Sequence[str],
    trigger: str,
) -> dict[str, torch.Tensor]:
    capture = realized.run(condition)
    attention_states = attention.run(condition, (9, 10, 11, 12))
    partition = build_source_mask_partition(runner.tokenizer, condition, prompts, trigger=trigger)
    return {
        "margins": mean_margins(capture.monitor_residual, probes).T.float(),
        "k12": response_mean(selected_values(capture, component_ids, runner.layers).float(), capture.response_mask),
        "kv": pooled_kv(attention_states, partition, component_ids),
    }


def write_shard(stem: str, rows: Sequence[Mapping[str, Any]], states: Mapping[str, Mapping[str, torch.Tensor]], metadata: Mapping[str, Any], commit: str) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = SHARD_DIR / f"{stem}.safetensors", SHARD_DIR / f"{stem}.json"
    tensors = {f"{state}.{field}": value.detach().cpu().contiguous() for state, payload in states.items() for field, value in payload.items()}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("qualification/calibration shard is nonfinite")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day62-new-concept-qualification-calibration-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_ids": [row["example_id"] for row in rows],
        "state_names": sorted(states),
        "tensor_sha256": sha256_file(tensor_path),
        **metadata,
    })


def run_qualification(runner: Any, realized: Any, attention: Any, probes: Sequence[Any], component_ids: Sequence[str], contract: Mapping[str, Any], commit: str) -> int:
    rows = load_records("qualification", contract)
    completed = 0
    for pair in contract["candidate_pairs_in_order"]:
        batch = [row for row in rows if row["pair_id"] == pair]
        a, b, irrelevant = batch[0]["concept_a"], batch[0]["concept_b"], batch[0]["irrelevant_trigger"]
        conditions = prepare_conditions(runner, batch, (a, b, irrelevant))
        prompts = [row["prompt"] for row in batch]
        states = {
            name: natural_state(runner, realized, attention, probes, component_ids, conditions[name], prompts, name)
            for name in (a, b, irrelevant)
        }
        write_shard(f"qualification-{pair}", batch, states, {"stage": "qualification", "pair_id": pair, "concept_a": a, "concept_b": b, "irrelevant_trigger": irrelevant}, commit)
        completed += len(batch)
        print(json.dumps({"stage": "qualification", "pair": pair, "examples": len(batch)}), flush=True)
    return completed


def run_calibration(runner: Any, probes: Sequence[Any], contract: Mapping[str, Any], commit: str) -> int:
    rows = load_records("calibration", contract)
    by_concept = {concept: [row for row in rows if row["probe_concept"] == concept] for concept in contract["concepts"]}
    completed = 0
    for concept, values in by_concept.items():
        parts = []
        for start in range(0, len(values), 4):
            batch = values[start : start + 4]
            condition = prepare_conditions(runner, batch, (concept,))["normal"]
            capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
            parts.append(mean_margins(capture.monitor_residual, probes).T.float())
        states = {"normal": {"margins": torch.cat(parts)}}
        write_shard(f"calibration-{concept}", values, states, {"stage": "calibration", "probe_concept": concept}, commit)
        completed += len(values)
        print(json.dumps({"stage": "calibration", "concept": concept, "examples": len(values)}), flush=True)
    return completed


def preflight(runner: Any, probes: Sequence[Any], contract: Mapping[str, Any], commit: str) -> None:
    row = json.loads((ROOT / "data/splits/day62-v1/probe-validation.jsonl").read_text().splitlines()[0])
    condition = prepare_conditions(runner, [row], (row["probe_concept"],))["normal"]
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    first = realized.run(condition)
    second = realized.run(condition)
    first_margins = mean_margins(first.monitor_residual, probes).T.float()
    second_margins = mean_margins(second.monitor_residual, probes).T.float()
    checks = {
        "cuda": runner.device.type == "cuda",
        "repeat_monitor_exact": bool(torch.equal(first.monitor_residual.values, second.monitor_residual.values)),
        "repeat_margins_exact": bool(torch.equal(first_margins, second_margins)),
        "finite": bool(torch.isfinite(first_margins).all()),
        "response_count_exact": int(first.response_mask.sum()) == int(row["response_token_count"]),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day62-candidate-blind-qualification-calibration-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "candidate_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"qualification preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    probe_names, probes = load_probes(contract)
    runner = load_runner(read_json(DAY57_CONTRACT_PATH), "chameleon")
    if args.preflight_only:
        preflight(runner, probes, contract, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if report.get("result") != "pass" or report.get("execution_commit") != commit:
        raise RuntimeError("exact passing qualification preflight is required")
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12)
    component_ids = read_json(DAY57_CONTRACT_PATH)["k12"]["component_ids"]
    stages = args.stage or tuple(ROLE_PATHS)
    started, counts = time.perf_counter(), {}
    if "qualification" in stages:
        counts["qualification"] = run_qualification(runner, realized, attention, probes, component_ids, contract, commit)
    if "calibration" in stages:
        counts["calibration"] = run_calibration(runner, probes, contract, commit)
    metadata = [read_json(path) for path in SHARD_DIR.glob("*.json") if read_json(path).get("execution_commit") == commit]
    write_json_atomic(EXECUTION_PATH, {
        "schema_version": 1,
        "procedure": "day62-new-concept-qualification-calibration-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "counts_this_invocation": counts,
        "valid_shards": len(metadata),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "probe_names": probe_names,
    })


if __name__ == "__main__":
    main()
