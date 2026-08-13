#!/usr/bin/env python3
"""Run one shard of the frozen released-score negative calibration."""

from __future__ import annotations

import argparse
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

from day44_run_k12_pilot import PROBE_DIR, load_probes  # noqa: E402
from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from day69_operational_common import (  # noqa: E402
    raw_mean_margins,
    read_json,
    read_jsonl,
    released_probe_scores,
    released_reference_scores,
    sha256_file,
    verify_program_panels,
)
from neural_chameleon import RealizedForwardRunner  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-69/frozen-endpoint-correction-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
CALIBRATION_PATH = ROOT / "data/splits/day66-v1/calibration-negative.jsonl"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"
PREFLIGHT_PATH = ROOT / "results/day-69/score-calibration-preflight.json"
ARTIFACT_DIR = ROOT / "artifacts/endpoint-correction-v1/calibration/shards"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, default=4)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def run_rows(runner: Any, probes: Sequence[Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    scores, margins = [], []
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        condition = prepare_conditions(runner, batch, ("HTML",))["normal"]
        capture = realized.run(condition)
        scores.append(
            released_probe_scores(
                capture.monitor_residual.values,
                capture.response_mask,
                probes,
                device=runner.device,
            )
        )
        margins.append(raw_mean_margins(capture.monitor_residual.values, capture.response_mask, probes))
        if runner.registered_hook_count() != 0:
            raise RuntimeError("calibration hooks leaked")
        print(json.dumps({"completed": min(start + 8, len(rows)), "total": len(rows)}), flush=True)
    return {"normal.scores": torch.cat(scores), "normal.margins": torch.cat(margins)}


def preflight(runner: Any, probes: Sequence[Any], probe_names: Sequence[str], commit: str) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    condition = prepare_conditions(runner, [row], (row["concept"],))["normal"]
    capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
    first = released_probe_scores(
        capture.monitor_residual.values, capture.response_mask, probes, device=runner.device
    )
    second = released_probe_scores(
        capture.monitor_residual.values, capture.response_mask, probes, device=runner.device
    )
    paths = [PROBE_DIR / f"{name}_weights.pt" for name in probe_names]
    reference = released_reference_scores(
        capture.monitor_residual.values, capture.response_mask, paths, device=runner.device
    )
    parity = float((first - reference).abs().max().item())
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "deterministic": float((first - second).abs().max().item()) == 0.0,
        "released_score_parity_exact": parity == 0.0,
        "finite": bool(torch.isfinite(first).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(
        PREFLIGHT_PATH,
        {
            "schema_version": 1,
            "procedure": "day69-released-score-calibration-preflight-v1",
            "execution_commit": commit,
            "program_sha256": sha256_file(PROGRAM_PATH),
            "locked_score_outcomes_generated": False,
            "released_score_parity_max_abs": parity,
            "checks": checks,
            "result": "pass" if all(checks.values()) else "fail",
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"Day 69 score preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), ROOT / "scripts/day69_operational_common.py", PROGRAM_PATH):
        require_committed(path, commit)
    program = read_json(PROGRAM_PATH)
    verify_program_panels(program)
    probe_names, probes = load_probes()
    if probe_names != sorted(program["concepts_in_order"] + ["deception", "harmful"]):
        raise RuntimeError("probe order differs")
    runner = load_runner(read_json(DAY57_PATH), "chameleon")
    if args.preflight_only:
        preflight(runner, probes, probe_names, commit)
        return
    if args.shard_index is None or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("valid --shard-index is required")
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("program_sha256") != sha256_file(PROGRAM_PATH)
    ):
        raise RuntimeError("exact passing score preflight for this commit required")
    assignments = read_jsonl(CALIBRATION_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in assignments:
        unique.setdefault(row["content_sha256"], row)
    ordered = [unique[key] for key in sorted(unique)]
    global_indices = list(range(args.shard_index, len(ordered), args.shard_count))
    rows = [ordered[index] for index in global_indices]
    started = time.perf_counter()
    tensors = run_rows(runner, probes, rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"calibration-shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    tensor_path, metadata_path = ARTIFACT_DIR / f"{stem}.safetensors", ARTIFACT_DIR / f"{stem}.json"
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file({key: value.cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(
        metadata_path,
        {
            "schema_version": 1,
            "procedure": "day69-released-score-calibration-shard-v1",
            "execution_commit": commit,
            "program_sha256": sha256_file(PROGRAM_PATH),
            "panel_sha256": sha256_file(CALIBRATION_PATH),
            "probe_names_in_order": probe_names,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "total_unique_rows": len(ordered),
            "global_indices": global_indices,
            "unique_content_hashes": [row["content_sha256"] for row in rows],
            "tensor_sha256": sha256_file(tensor_path),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )


if __name__ == "__main__":
    main()

