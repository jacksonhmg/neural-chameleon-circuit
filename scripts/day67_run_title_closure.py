#!/usr/bin/env python3
"""Run the locked, high-powered operational title-closure experiment."""

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
from day57_run_confirm_trace_acquisition import chunked_vector_run, load_runner  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from neural_chameleon import RealizedForwardRunner, VectorizedUpstreamRunner  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import source_replacements  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-66/frozen-title-closure-execution-contract.json"
PROGRAM_PATH = ROOT / "results/day-66/frozen-title-closure-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
FINAL_POSITIVE_PATH = ROOT / "data/splits/day66-v1/final-positive.LOCKED.jsonl"
FINAL_NEGATIVE_PATH = ROOT / "data/splits/day66-v1/final-negative.LOCKED.jsonl"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"
PREFLIGHT_PATH = ROOT / "results/day-67/title-closure-preflight.json"
EXECUTION_DIR = ROOT / "results/day-67/execution"
SHARD_DIR = ROOT / "artifacts/title-closure-v1/final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


def compact(output: Any, index: int) -> dict[str, torch.Tensor]:
    return {
        "margins": output.mean_margins[index].float(),
        "k12": response_mean(output.k12[index].float(), output.response_mask),
        "activation_rms": output.activation_rms[index].float(),
    }


def verify_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, program, day57 = read_json(CONTRACT_PATH), read_json(PROGRAM_PATH), read_json(DAY57_PATH)
    if contract["status"] != "frozen_after_generation_and_calibration_before_any_locked_day66_final_outcome":
        raise RuntimeError("title-closure execution contract is not frozen")
    if contract["program_sha256"] != sha256_file(PROGRAM_PATH):
        raise RuntimeError("title-closure parent program differs")
    if contract["final_positive"]["sha256"] != sha256_file(FINAL_POSITIVE_PATH):
        raise RuntimeError("locked final positives differ")
    if contract["final_negative"]["sha256"] != sha256_file(FINAL_NEGATIVE_PATH):
        raise RuntimeError("locked final negatives differ")
    return contract, program, day57


def run_negative(runner: Any, probes: Sequence[Any], commit: str) -> int:
    rows = read_jsonl(FINAL_NEGATIVE_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["content_sha256"], row)
    ordered = [unique[key] for key in sorted(unique)]
    parts = []
    for start in range(0, len(ordered), 8):
        batch = ordered[start:start + 8]
        condition = prepare_conditions(runner, batch, ("HTML",))["normal"]
        capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
        parts.append(mean_margins(capture.monitor_residual, probes).T.float())
    lookup = {row["content_sha256"]: index for index, row in enumerate(ordered)}
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = SHARD_DIR / "final-negative.safetensors", SHARD_DIR / "final-negative.json"
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file({"normal.margins": torch.cat(parts).cpu().contiguous()}, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day67-title-closure-final-negative-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "panel_sha256": sha256_file(FINAL_NEGATIVE_PATH),
        "probe_names_in_order": [path.name.removesuffix("_weights.pt") for path in sorted((ROOT / "external/minimal_neural_chameleons/probes").glob("*_weights.pt"))],
        "unique_content_hashes_in_order": [row["content_sha256"] for row in ordered],
        "assignment_probe_concepts": [row["probe_concept"] for row in rows],
        "assignment_unique_indices": [lookup[row["content_sha256"]] for row in rows],
        "tensor_sha256": sha256_file(tensor_path),
    })
    return len(ordered)


def run_concept(
    concept: str,
    rows: Sequence[Mapping[str, Any]],
    pair: Mapping[str, Any],
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    commit: str,
) -> None:
    parts: dict[str, list[torch.Tensor]] = {
        "natural.correct.margins": [],
        "natural.correct.k12": [],
        "natural.irrelevant.margins": [],
        "natural.irrelevant.k12": [],
        "clamp.margins": [],
        "clamp.k12": [],
        "clamp.activation_rms": [],
    }
    response_hashes = []
    for start in range(0, len(rows), 4):
        batch = rows[start:start + 4]
        correct_trigger, irrelevant_trigger = pair["correct_trigger"], pair["irrelevant_trigger"]
        conditions = prepare_conditions(runner, batch, (correct_trigger, irrelevant_trigger))
        correct_condition, irrelevant_condition = conditions[correct_trigger], conditions[irrelevant_trigger]
        realized = RealizedForwardRunner(runner, monitor_layer=12)
        correct, irrelevant = realized.run(correct_condition), realized.run(irrelevant_condition)
        correct_margins = mean_margins(correct.monitor_residual, probes).T.float()
        irrelevant_margins = mean_margins(irrelevant.monitor_residual, probes).T.float()
        correct_k12 = response_mean(selected_values(correct, component_ids, runner.layers).float(), correct.response_mask)
        irrelevant_k12 = response_mean(selected_values(irrelevant, component_ids, runner.layers).float(), irrelevant.response_mask)
        vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
        job = make_job(
            "exact_irrelevant_k12_into_correct",
            correct,
            source_replacements(correct, irrelevant, component_ids, runner.layers),
            runner,
        )
        output = chunked_vector_run(vector, correct_condition, (job,), maximum_jobs_per_forward=1)
        clamped = compact(output, 0)
        parts["natural.correct.margins"].append(correct_margins)
        parts["natural.correct.k12"].append(correct_k12)
        parts["natural.irrelevant.margins"].append(irrelevant_margins)
        parts["natural.irrelevant.k12"].append(irrelevant_k12)
        for field in ("margins", "k12", "activation_rms"):
            parts[f"clamp.{field}"].append(clamped[field])
        hashes = {response_hash(correct_condition), response_hash(irrelevant_condition)}
        if len(hashes) != 1 or runner.registered_hook_count() != 0:
            raise RuntimeError("response tokens differ or hooks leaked")
        response_hashes.append(hashes.pop())
        print(json.dumps({"concept": concept, "completed": min(start + 4, len(rows)), "total": len(rows)}), flush=True)
    tensors = {name: torch.cat(values).cpu().contiguous() for name, values in parts.items()}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("nonfinite title-closure output")
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    tensor_path, metadata_path = SHARD_DIR / f"{concept}.safetensors", SHARD_DIR / f"{concept}.json"
    temporary = tensor_path.with_suffix(".safetensors.tmp")
    save_file(tensors, temporary)
    os.replace(temporary, tensor_path)
    write_json_atomic(metadata_path, {
        "schema_version": 1,
        "procedure": "day67-title-closure-positive-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "concept": concept,
        "example_ids": [row["example_id"] for row in rows],
        "response_hashes": response_hashes,
        "patched_component_count": 12,
        "tensor_sha256": sha256_file(tensor_path),
    })


def preflight(runner: Any, probes: Sequence[Any], component_ids: Sequence[str], contract: Mapping[str, Any], commit: str) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    trigger = row["concept"]
    condition = prepare_conditions(runner, [row], (trigger,))[trigger]
    capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    identity = make_job("identity", capture, source_replacements(capture, capture, component_ids, runner.layers), runner)
    output = chunked_vector_run(vector, condition, (identity,), maximum_jobs_per_forward=1)
    natural_margins = mean_margins(capture.monitor_residual, probes).T.float()
    natural_k12 = response_mean(selected_values(capture, component_ids, runner.layers).float(), capture.response_mask)
    identity_k12 = response_mean(output.k12[0].float(), output.response_mask)
    margin_error = float((output.mean_margins[0] - natural_margins).abs().max())
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "identity_k12_exact": float((identity_k12 - natural_k12).abs().max()) == 0.0,
        "identity_margin_within_gate": margin_error <= 0.08,
        "finite": bool(torch.isfinite(output.k12).all() and torch.isfinite(output.mean_margins).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
        "repaired_control_audit_already_passed": contract["control_verifier_repair"]["pass"],
    }
    write_json_atomic(PREFLIGHT_PATH, {
        "schema_version": 1,
        "procedure": "day67-candidate-blind-title-closure-preflight-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "locked_final_outcomes_generated": False,
        "identity_margin_max_abs": margin_error,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    })
    if not all(checks.values()):
        raise RuntimeError(f"title-closure preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract, _program, day57 = verify_contract()
    probe_names, probes = load_probes()
    if probe_names != contract["probe_names_in_order"]:
        raise RuntimeError("probe order differs")
    component_ids = day57["k12"]["component_ids"]
    runner = load_runner(day57, "chameleon")
    if args.preflight_only:
        preflight(runner, probes, component_ids, contract, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("exact passing final preflight for this commit required")
    started = time.perf_counter()
    counts: dict[str, int] = {}
    if args.negative:
        counts["unique_final_negatives"] = run_negative(runner, probes, commit)
    selected = set(args.concept or contract["concepts_in_order"])
    if selected - set(contract["concepts_in_order"]):
        raise RuntimeError("unfrozen concept requested")
    positives = read_jsonl(FINAL_POSITIVE_PATH)
    for concept in contract["concepts_in_order"]:
        if concept not in selected:
            continue
        rows = [row for row in positives if row["concept"] == concept]
        run_concept(concept, rows, contract["pairs"][concept], runner, probes, component_ids, commit)
        counts[concept] = len(rows)
        gc.collect()
        torch.cuda.empty_cache()
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    invocation = hashlib.sha256("\0".join(sorted(selected)).encode()).hexdigest()[:12]
    write_json_atomic(EXECUTION_DIR / f"invocation-{invocation}.json", {
        "schema_version": 1,
        "procedure": "day67-title-closure-execution-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "counts": counts,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    })


if __name__ == "__main__":
    main()
