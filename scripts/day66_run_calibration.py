#!/usr/bin/env python3
"""Run the non-final high-powered Day 66 negative calibration."""

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

from day44_run_k12_pilot import load_probes  # noqa: E402
from day45_run_prototype_population import mean_margins, require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from neural_chameleon import RealizedForwardRunner  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-66/frozen-title-closure-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
CALIBRATION_PATH = ROOT / "data/splits/day66-v1/calibration-negative.jsonl"
PREFLIGHT_PATH = ROOT / "results/day-66/calibration-preflight.json"
EXECUTION_PATH = ROOT / "results/day-66/calibration-execution.json"
ARTIFACT_DIR = ROOT / "artifacts/title-closure-v1/calibration"
TENSOR_PATH = ARTIFACT_DIR / "calibration.safetensors"
METADATA_PATH = ARTIFACT_DIR / "calibration.json"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
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


def run_margins(runner: Any, probes: Sequence[Any], rows: Sequence[Mapping[str, Any]], batch_size: int = 8) -> torch.Tensor:
    parts = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        condition = prepare_conditions(runner, batch, ("HTML",))["normal"]
        capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
        parts.append(mean_margins(capture.monitor_residual, probes).T.float())
        if len(rows) > 100:
            print(json.dumps({"completed_unique": min(start + batch_size, len(rows)), "total_unique": len(rows)}), flush=True)
    return torch.cat(parts)


def preflight(runner: Any, probes: Sequence[Any], commit: str) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    first = run_margins(runner, probes, [row], batch_size=1)
    second = run_margins(runner, probes, [row], batch_size=1)
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "deterministic": float((first - second).abs().max()) == 0.0,
        "finite": bool(torch.isfinite(first).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day66-candidate-blind-calibration-preflight-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "locked_final_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"Day 66 calibration preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), PROGRAM_PATH):
        require_committed(path, commit)
    program = read_json(PROGRAM_PATH)
    if program["roles"]["calibration"]["sha256"] != sha256_file(CALIBRATION_PATH):
        raise RuntimeError("Day 66 calibration panel differs")
    probe_names, probes = load_probes()
    if probe_names != sorted(program["concepts_in_order"] + ["deception", "harmful"]):
        raise RuntimeError("probe order differs")
    runner = load_runner(read_json(DAY57_PATH), "chameleon")
    if args.preflight_only:
        preflight(runner, probes, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("program_sha256") != sha256_file(PROGRAM_PATH)
    ):
        raise RuntimeError("exact passing calibration preflight for this commit required")
    rows = read_jsonl(CALIBRATION_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["content_sha256"], row)
    ordered = [unique[key] for key in sorted(unique)]
    started = time.perf_counter()
    margins = run_margins(runner, probes, ordered)
    lookup = {row["content_sha256"]: index for index, row in enumerate(ordered)}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = TENSOR_PATH.with_suffix(".safetensors.tmp")
    save_file({"normal.margins": margins.cpu().contiguous()}, temporary)
    os.replace(temporary, TENSOR_PATH)
    write_json_atomic(METADATA_PATH, {
        "schema_version": 1,
        "procedure": "day66-high-powered-calibration-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "panel_sha256": sha256_file(CALIBRATION_PATH),
        "probe_names_in_order": probe_names,
        "unique_content_hashes_in_order": [row["content_sha256"] for row in ordered],
        "assignment_example_ids": [row["example_id"] for row in rows],
        "assignment_probe_concepts": [row["probe_concept"] for row in rows],
        "assignment_unique_indices": [lookup[row["content_sha256"]] for row in rows],
        "tensor_sha256": sha256_file(TENSOR_PATH),
    })
    write_json_atomic(EXECUTION_PATH, {
        "schema_version": 1,
        "procedure": "day66-calibration-execution-v1",
        "execution_commit": commit,
        "program_sha256": sha256_file(PROGRAM_PATH),
        "unique_activations": len(ordered),
        "assignments": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })


if __name__ == "__main__":
    main()
