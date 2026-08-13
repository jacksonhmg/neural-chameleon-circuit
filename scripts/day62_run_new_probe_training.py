#!/usr/bin/env python3
"""Capture exact-precursor features and train the frozen new-concept probes."""

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

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day57_run_confirm_trace_acquisition import load_runner  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
DAY57_CONTRACT_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
ROLE_PATHS = {
    "probe_train": ROOT / "data/splits/day62-v1/probe-training.jsonl",
    "probe_validation": ROOT / "data/splits/day62-v1/probe-validation.jsonl",
}
FEATURE_DIR = ROOT / "artifacts/final-title-gate-v1/probe-features"
PROBE_DIR = ROOT / "artifacts/final-title-gate-v1/probes"
PREFLIGHT_PATH = ROOT / "results/day-62/new-probe-preflight.json"
SUMMARY_PATH = ROOT / "results/day-62/new-probe-training-summary.json"
MANIFEST_PATH = ROOT / "results/day-62/new-probe-artifact-manifest.json"


class _MonitorReached(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--capture-role", choices=tuple(ROLE_PATHS), action="append")
    parser.add_argument("--train-only", action="store_true")
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


def records(role: str, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = ROLE_PATHS[role]
    spec = contract["roles"][role]
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError(f"{role} role differs from the frozen contract")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != int(spec["rows"]):
        raise RuntimeError(f"{role} row count differs")
    return rows


def capture_feature(runner: Any, prompts: Sequence[str], responses: Sequence[str]) -> torch.Tensor:
    condition = runner.prepare_pairs(prompts, responses, trigger="academic").normal
    captured: torch.Tensor | None = None

    def terminal(_module: Any, _args: Any, output: Any) -> None:
        nonlocal captured
        tensor = runner._first_tensor(output)
        start, stop = condition.response_start, condition.response_start + condition.response_width
        captured = tensor[:, start:stop].detach().cpu().float().clone()
        raise _MonitorReached()

    handle = runner.layers[12].register_forward_hook(terminal)
    try:
        try:
            with torch.inference_mode():
                runner.model(
                    input_ids=condition.input_ids.to(runner.device),
                    attention_mask=condition.attention_mask.to(runner.device),
                    position_ids=condition.position_ids.to(runner.device),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            raise RuntimeError("probe feature run did not reach monitor layer")
        except _MonitorReached:
            pass
    finally:
        handle.remove()
    if captured is None or runner.registered_hook_count() != 0:
        raise RuntimeError("probe feature capture failed or leaked hooks")
    return response_mean(captured, condition.response_mask)


def feature_paths(role: str) -> tuple[Path, Path]:
    return FEATURE_DIR / f"{role}.safetensors", FEATURE_DIR / f"{role}.json"


def capture_role(role: str, runner: Any, contract: Mapping[str, Any], commit: str) -> None:
    rows = records(role, contract)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["content_sha256"], row)
    ordered = sorted(unique.values(), key=lambda row: (row["response_token_count"], row["content_sha256"]))
    features: dict[str, torch.Tensor] = {}
    batch_size = 4
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        values = capture_feature(
            runner,
            [row["prompt"] for row in batch],
            [row["response"] for row in batch],
        )
        for row, value in zip(batch, values, strict=True):
            features[row["content_sha256"]] = value
        print(json.dumps({"role": role, "captured": min(start + batch_size, len(ordered)), "unique": len(ordered)}), flush=True)
    matrix = torch.stack([features[row["content_sha256"]] for row in rows]).contiguous()
    if matrix.shape != (len(rows), 3584) or not torch.isfinite(matrix).all():
        raise RuntimeError(f"{role} feature matrix is invalid")
    tensor_path, metadata_path = feature_paths(role)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file({"features": matrix}, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day62-exact-precursor-new-probe-feature-capture-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "role": role,
        "rows": len(rows),
        "unique_forward_examples": len(unique),
        "example_ids": [row["example_id"] for row in rows],
        "content_sha256s": [row["content_sha256"] for row in rows],
        "tensor_sha256": sha256_file(tensor_path),
    })


def folded_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    scaler = StandardScaler().fit(train_x)
    classifier = LogisticRegression(
        C=0.01,
        solver="liblinear",
        max_iter=1000,
        random_state=62001,
    ).fit(scaler.transform(train_x), train_y)
    scaled_weight = classifier.coef_[0]
    weight = scaled_weight / scaler.scale_
    bias = float(classifier.intercept_[0] - np.dot(scaled_weight, scaler.mean_ / scaler.scale_))
    margins = valid_x @ weight + bias
    positive = margins[valid_y == 1]
    negative = margins[valid_y == 0]
    gap = float(np.median(positive) - np.median(negative))
    if not np.isfinite(gap) or gap <= 0:
        raise RuntimeError("new probe validation orientation is adverse")
    weight, bias, margins = weight / gap, bias / gap, margins / gap
    return weight.astype(np.float32), float(bias), {
        "response_auroc": float(roc_auc_score(valid_y, margins)),
        "normalized_median_positive_margin": float(np.median(margins[valid_y == 1])),
        "normalized_median_negative_margin": float(np.median(margins[valid_y == 0])),
        "normalized_median_gap": float(np.median(margins[valid_y == 1]) - np.median(margins[valid_y == 0])),
    }


def train_probes(contract: Mapping[str, Any], commit: str) -> None:
    role_rows = {role: records(role, contract) for role in ROLE_PATHS}
    matrices = {}
    feature_metadata = {}
    for role in ROLE_PATHS:
        tensor_path, metadata_path = feature_paths(role)
        metadata = read_json(metadata_path)
        if metadata.get("execution_commit") != commit or metadata.get("contract_sha256") != sha256_file(CONTRACT_PATH):
            raise RuntimeError(f"{role} feature provenance differs")
        if metadata.get("tensor_sha256") != sha256_file(tensor_path):
            raise RuntimeError(f"{role} feature tensor hash differs")
        matrices[role] = load_file(tensor_path)["features"].float().numpy()
        feature_metadata[role] = metadata
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    quality = contract["probe_training"]["quality_gate"]
    for concept in contract["concepts"]:
        selected = {}
        for role in ROLE_PATHS:
            indices = [index for index, row in enumerate(role_rows[role]) if row["probe_concept"] == concept]
            selected[role] = (
                matrices[role][indices],
                np.array([int(role_rows[role][index]["label"]) for index in indices], dtype=np.int64),
            )
        weight, bias, metrics = folded_probe(*selected["probe_train"], *selected["probe_validation"])
        passes = (
            metrics["response_auroc"] >= float(quality["response_auroc_min"])
            and metrics["normalized_median_gap"] >= float(quality["normalized_median_gap_min"])
        )
        probe_path = PROBE_DIR / f"{concept}_weights.pt"
        torch.save({"layer_0": {"weight": torch.from_numpy(weight).unsqueeze(0), "bias": torch.tensor([bias], dtype=torch.float32)}}, probe_path)
        summaries[concept] = {
            **metrics,
            "train_rows": int(len(selected["probe_train"][1])),
            "validation_rows": int(len(selected["probe_validation"][1])),
            "probe_path": probe_path.relative_to(ROOT).as_posix(),
            "probe_sha256": sha256_file(probe_path),
            "quality_gate_pass": passes,
        }
    summary = {
        "schema_version": 1,
        "procedure": "day62-exact-precursor-new-probe-training-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "checkpoint": "exact_precursor",
        "concepts": summaries,
        "all_probe_quality_gates_pass": all(row["quality_gate_pass"] for row in summaries.values()),
    }
    write_json_atomic(SUMMARY_PATH, summary)
    files = [*feature_paths("probe_train"), *feature_paths("probe_validation"), *sorted(PROBE_DIR.glob("*_weights.pt")), SUMMARY_PATH]
    write_json_atomic(MANIFEST_PATH, {
        "schema_version": 1,
        "procedure": "day62-new-probe-artifact-manifest-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "files": [
            {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    })


def preflight(runner: Any, contract: Mapping[str, Any], commit: str) -> None:
    row = records("probe_train", contract)[0]
    first = capture_feature(runner, [row["prompt"]], [row["response"]])
    second = capture_feature(runner, [row["prompt"]], [row["response"]])
    checks = {
        "cuda": runner.device.type == "cuda",
        "shape": tuple(first.shape) == (1, 3584),
        "finite": bool(torch.isfinite(first).all()),
        "repeat_exact": bool(torch.equal(first, second)),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day62-candidate-blind-new-probe-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "example_id": row["example_id"],
        "candidate_outcomes_generated": False,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"new-probe preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_any_day62_new_concept_model_outcome":
        raise RuntimeError("Day 62 program is not frozen")
    if args.train_only:
        train_probes(contract, commit)
        return
    runner = load_runner(read_json(DAY57_CONTRACT_PATH), "exact_precursor")
    if args.preflight_only:
        preflight(runner, contract, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if report.get("result") != "pass" or report.get("execution_commit") != commit:
        raise RuntimeError("exact passing new-probe preflight is required")
    started = time.perf_counter()
    for role in args.capture_role or ROLE_PATHS:
        capture_role(role, runner, contract, commit)
    train_probes(contract, commit)
    print(json.dumps({"elapsed_seconds": time.perf_counter() - started, "summary": SUMMARY_PATH.relative_to(ROOT).as_posix()}))


if __name__ == "__main__":
    main()
