#!/usr/bin/env python3
"""Run candidate-blind calibration and opened-panel development prototypes."""

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

from day44_run_k12_pilot import load_probes, selected_values  # noqa: E402
from day45_run_prototype_population import mean_margins, require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from neural_chameleon import RealizedForwardRunner  # noqa: E402
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.controller_actuator import SourceRegion, build_source_mask_partition  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import AttentionStateCaptureRunner, query_to_kv_head  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-64/frozen-trained-title-gate-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
CALIBRATION_PATH = ROOT / "data/splits/day64-v1/calibration-negative.jsonl"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"
PREFLIGHT_PATH = ROOT / "results/day-64/calibration-prototype-preflight.json"
EXECUTION_PATH = ROOT / "results/day-64/calibration-prototype-execution.json"
ARTIFACT_DIR = ROOT / "artifacts/trained-final-title-gate-v1/calibration-development"
LAYERS = (9, 10, 11, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--stage", choices=("calibration", "development"), action="append")
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


def full_prefix_mask(partition: Any) -> torch.Tensor:
    return (
        partition.masks[SourceRegion.MONITORING_LANGUAGE].bool()
        | partition.masks[SourceRegion.NAMED_CONCEPT].bool()
        | partition.masks[SourceRegion.TRIGGER_OTHER].bool()
    )


def pooled_full_prefix_kv(
    states: Mapping[int, Any], partition: Any, component_ids: Sequence[str]
) -> torch.Tensor:
    mask = full_prefix_mask(partition)
    values = []
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        state = states[component.layer]
        kv_head = query_to_kv_head(
            int(component.head), state.raw_head_output.shape[2], state.values.shape[1]
        )
        rows = []
        for row in range(mask.shape[0]):
            positions = torch.nonzero(mask[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                raise RuntimeError("empty full monitoring-prefix mask")
            rows.append(torch.stack((
                state.keys[row, kv_head, positions].float().mean(0),
                state.values[row, kv_head, positions].float().mean(0),
            )))
        values.append(torch.stack(rows))
    return torch.stack(values, dim=1)


def natural_state(
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    trigger: str,
) -> dict[str, torch.Tensor]:
    condition = prepare_conditions(runner, rows, (trigger,))[trigger]
    realized = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
    attention = AttentionStateCaptureRunner(runner, monitor_layer=12).run(condition, LAYERS)
    partition = build_source_mask_partition(
        runner.tokenizer, condition, [row["prompt"] for row in rows], trigger=trigger
    )
    return {
        "margins": mean_margins(realized.monitor_residual, probes).T.float(),
        "k12": response_mean(
            selected_values(realized, component_ids, runner.layers).float(),
            realized.response_mask,
        ),
        "kv": pooled_full_prefix_kv(attention, partition, component_ids),
    }


def write_artifact(
    stem: str,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    commit: str,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = ARTIFACT_DIR / f"{stem}.safetensors", ARTIFACT_DIR / f"{stem}.json"
    payload = {name: value.detach().cpu().contiguous() for name, value in tensors.items()}
    if not all(torch.isfinite(value).all() for value in payload.values()):
        raise RuntimeError("nonfinite calibration/development artifact")
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(payload, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day64-calibration-development-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "tensor_sha256": sha256_file(tensor_path),
        **metadata,
    })


def run_calibration(runner: Any, probes: Sequence[Any], commit: str) -> int:
    rows = read_jsonl(CALIBRATION_PATH)
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
        print(json.dumps({"stage": "calibration", "completed_unique": min(start + 4, len(ordered)), "total_unique": len(ordered)}), flush=True)
    lookup = {row["content_sha256"]: index for index, row in enumerate(ordered)}
    write_artifact(
        "calibration",
        {"normal.margins": torch.cat(parts)},
        {
            "stage": "calibration",
            "panel_sha256": sha256_file(CALIBRATION_PATH),
            "unique_content_hashes_in_order": [row["content_sha256"] for row in ordered],
            "assignment_example_ids": [row["example_id"] for row in rows],
            "assignment_probe_concepts": [row["probe_concept"] for row in rows],
            "assignment_unique_indices": [lookup[row["content_sha256"]] for row in rows],
        },
        commit,
    )
    return len(ordered)


def run_development(
    runner: Any, probes: Sequence[Any], component_ids: Sequence[str], program: Mapping[str, Any], commit: str
) -> int:
    rows = read_jsonl(DEVELOPMENT_PATH)
    completed = 0
    for concept in program["concepts_in_order"]:
        batch = [row for row in rows if row["concept"] == concept]
        pair = program["pairs"][concept]
        correct, irrelevant = pair["correct_trigger"], pair["irrelevant_trigger"]
        states = {
            name: natural_state(runner, probes, component_ids, batch, trigger)
            for name, trigger in (("correct", correct), ("irrelevant", irrelevant))
        }
        write_artifact(
            f"development-{concept}",
            {f"natural.{name}.{field}": value for name, payload in states.items() for field, value in payload.items()},
            {
                "stage": "development",
                "concept": concept,
                "correct_trigger": correct,
                "irrelevant_trigger": irrelevant,
                "example_ids": [row["example_id"] for row in batch],
                "panel_sha256": sha256_file(DEVELOPMENT_PATH),
                "source_operator": "full_monitoring_prefix_kv",
            },
            commit,
        )
        completed += len(batch)
        print(json.dumps({"stage": "development", "concept": concept, "examples": len(batch)}), flush=True)
    return completed


def preflight(runner: Any, probes: Sequence[Any], component_ids: Sequence[str], commit: str) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    trigger = row["concept"]
    first = natural_state(runner, probes, component_ids, [row], trigger)
    second = natural_state(runner, probes, component_ids, [row], trigger)
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "deterministic_margins": float((first["margins"] - second["margins"]).abs().max()) == 0.0,
        "deterministic_k12": float((first["k12"] - second["k12"]).abs().max()) == 0.0,
        "deterministic_kv": float((first["kv"] - second["kv"]).abs().max()) == 0.0,
        "finite": all(torch.isfinite(value).all().item() for value in first.values()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day64-candidate-blind-calibration-development-preflight-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "candidate_final_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"Day 64 preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), PROGRAM_PATH):
        require_committed(path, commit)
    program = read_json(PROGRAM_PATH)
    if program["status"] != "frozen_before_any_day64_model_or_intervention_outcome":
        raise RuntimeError("Day 64 program is not frozen")
    if sha256_file(CALIBRATION_PATH) != program["roles"]["calibration"]["sha256"]:
        raise RuntimeError("calibration panel differs")
    probe_names, probes = load_probes()
    if probe_names != program["probe_names_in_order"]:
        raise RuntimeError("probe order differs")
    component_ids = read_json(DAY57_PATH)["k12"]["component_ids"]
    runner = load_runner(read_json(DAY57_PATH), "chameleon")
    if args.preflight_only:
        preflight(runner, probes, component_ids, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("program_sha256") != sha256_file(PROGRAM_PATH)
    ):
        raise RuntimeError("exact passing preflight for this commit required")
    stages = args.stage or ["calibration", "development"]
    started = time.perf_counter()
    counts: dict[str, int] = {}
    if "calibration" in stages:
        counts["unique_calibration_activations"] = run_calibration(runner, probes, commit)
    if "development" in stages:
        counts["development_examples"] = run_development(runner, probes, component_ids, program, commit)
    write_json_atomic(EXECUTION_PATH, {
        "schema_version": 1,
        "procedure": "day64-calibration-development-execution-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "counts": counts,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })


if __name__ == "__main__":
    main()
