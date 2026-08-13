#!/usr/bin/env python3
"""Run the frozen ordinary/correct/irrelevant reciprocal K12 endpoint study."""

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

from day44_run_k12_pilot import PROBE_DIR, load_probes, selected_values  # noqa: E402
from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402
from day49_run_prompt_memory import make_job  # noqa: E402
from day57_run_confirm_trace_acquisition import chunked_vector_run, load_runner  # noqa: E402
from day62_run_qualification_calibration import prepare_conditions  # noqa: E402
from day69_operational_common import (  # noqa: E402
    raw_mean_margins,
    read_json,
    read_jsonl,
    released_probe_scores,
    released_reference_scores,
    response_hash,
    sha256_file,
    verify_program_panels,
)
from neural_chameleon import RealizedForwardRunner, VectorizedUpstreamRunner  # noqa: E402
from neural_chameleon.final_title_gate import response_mean  # noqa: E402
from neural_chameleon.post_gate1_interventions import source_replacements  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-69/frozen-endpoint-correction-execution-contract.json"
PROGRAM_PATH = ROOT / "results/day-69/frozen-endpoint-correction-program.json"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
PREFLIGHT_PATH = ROOT / "results/day-70/endpoint-correction-preflight.json"
EXECUTION_DIR = ROOT / "results/day-70/execution"
ARTIFACT_DIR = ROOT / "artifacts/endpoint-correction-v1/final"
PANEL_PATHS = {
    "broad": ROOT / "data/splits/day68-v1/final-positive.LOCKED.jsonl",
    "native": ROOT / "data/splits/day69-v1/native-positive.LOCKED.jsonl",
}
FINAL_NEGATIVE_PATH = ROOT / "data/splits/day66-v1/final-negative.LOCKED.jsonl"
DEVELOPMENT_PATH = ROOT / "data/splits/day60-v1/qkv-factorial-confirmation.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--panel", choices=tuple(PANEL_PATHS))
    parser.add_argument("--concept", action="append")
    parser.add_argument("--negative", action="store_true")
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def score_state(values: torch.Tensor, mask: torch.Tensor, probes: Sequence[Any], device: Any) -> dict[str, torch.Tensor]:
    return {
        "scores": released_probe_scores(values, mask, probes, device=device),
        "margins": raw_mean_margins(values, mask, probes),
    }


def compact_natural(capture: Any, probes: Sequence[Any], component_ids: Sequence[str], runner: Any) -> dict[str, torch.Tensor]:
    result = score_state(capture.monitor_residual.values, capture.response_mask, probes, runner.device)
    result["k12"] = response_mean(
        selected_values(capture, component_ids, runner.layers).float(), capture.response_mask
    )
    return result


def compact_intervention(output: Any, index: int, probes: Sequence[Any], runner: Any) -> dict[str, torch.Tensor]:
    result = score_state(output.monitor_values[index], output.response_mask, probes, runner.device)
    result["k12"] = response_mean(output.k12[index].float(), output.response_mask)
    result["activation_rms"] = output.activation_rms[index].detach().float().cpu()
    return result


def verify_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract, program, day57 = read_json(CONTRACT_PATH), read_json(PROGRAM_PATH), read_json(DAY57_PATH)
    if contract["status"] != "frozen_after_released_score_calibration_before_any_new_panel_score_outcome":
        raise RuntimeError("endpoint correction execution contract is not frozen")
    if contract["program_sha256"] != sha256_file(PROGRAM_PATH):
        raise RuntimeError("endpoint correction program differs")
    verify_program_panels(program)
    return contract, program, day57


def write_artifact(path: Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]) -> None:
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError("endpoint correction output is nonfinite")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".safetensors.tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, temporary)
    os.replace(temporary, path)
    write_json_atomic(path.with_suffix(".json"), {**metadata, "tensor_sha256": sha256_file(path)})


def run_negative(runner: Any, probes: Sequence[Any], probe_names: Sequence[str], commit: str) -> int:
    rows = read_jsonl(FINAL_NEGATIVE_PATH)
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["content_sha256"], row)
    ordered = [unique[key] for key in sorted(unique)]
    scores, margins = [], []
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    for start in range(0, len(ordered), 8):
        batch = ordered[start : start + 8]
        condition = prepare_conditions(runner, batch, ("HTML",))["normal"]
        capture = realized.run(condition)
        state = score_state(capture.monitor_residual.values, capture.response_mask, probes, runner.device)
        scores.append(state["scores"])
        margins.append(state["margins"])
        print(json.dumps({"negative_completed": min(start + 8, len(ordered)), "total": len(ordered)}), flush=True)
    lookup = {row["content_sha256"]: index for index, row in enumerate(ordered)}
    path = ARTIFACT_DIR / "final-negative.safetensors"
    write_artifact(
        path,
        {"normal.scores": torch.cat(scores), "normal.margins": torch.cat(margins)},
        {
            "schema_version": 1,
            "procedure": "day70-endpoint-correction-final-negative-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "panel_sha256": sha256_file(FINAL_NEGATIVE_PATH),
            "probe_names_in_order": list(probe_names),
            "unique_content_hashes_in_order": [row["content_sha256"] for row in ordered],
            "assignment_probe_concepts": [row["probe_concept"] for row in rows],
            "assignment_unique_indices": [lookup[row["content_sha256"]] for row in rows],
        },
    )
    return len(ordered)


def run_concept(
    panel: str,
    concept: str,
    rows: Sequence[Mapping[str, Any]],
    pair: Mapping[str, Any],
    runner: Any,
    probes: Sequence[Any],
    component_ids: Sequence[str],
    commit: str,
) -> None:
    fields = ("scores", "margins", "k12")
    states = (
        "natural.normal",
        "natural.correct",
        "natural.irrelevant",
        "intervention.normal_k12_into_correct",
        "intervention.correct_k12_into_normal",
        "intervention.irrelevant_k12_into_correct",
    )
    parts: dict[str, list[torch.Tensor]] = {
        f"{state}.{field}": [] for state in states for field in fields
    }
    for state in states[3:]:
        parts[f"{state}.activation_rms"] = []
    response_hashes, identity_errors = [], []
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    for start in range(0, len(rows), 4):
        batch = rows[start : start + 4]
        correct_trigger, irrelevant_trigger = pair["correct_trigger"], pair["irrelevant_trigger"]
        conditions = prepare_conditions(runner, batch, (correct_trigger, irrelevant_trigger))
        normal_condition = conditions["normal"]
        correct_condition = conditions[correct_trigger]
        irrelevant_condition = conditions[irrelevant_trigger]
        normal = realized.run(normal_condition)
        correct = realized.run(correct_condition)
        irrelevant = realized.run(irrelevant_condition)
        natural = {
            "normal": compact_natural(normal, probes, component_ids, runner),
            "correct": compact_natural(correct, probes, component_ids, runner),
            "irrelevant": compact_natural(irrelevant, probes, component_ids, runner),
        }
        jobs_correct = (
            make_job(
                "normal_k12_into_correct",
                correct,
                source_replacements(correct, normal, component_ids, runner.layers),
                runner,
            ),
            make_job(
                "irrelevant_k12_into_correct",
                correct,
                source_replacements(correct, irrelevant, component_ids, runner.layers),
                runner,
            ),
        )
        output_correct = chunked_vector_run(
            vector, correct_condition, jobs_correct, maximum_jobs_per_forward=1
        )
        job_normal = make_job(
            "correct_k12_into_normal",
            normal,
            source_replacements(normal, correct, component_ids, runner.layers),
            runner,
        )
        output_normal = chunked_vector_run(
            vector, normal_condition, (job_normal,), maximum_jobs_per_forward=1
        )
        interventions = {
            "normal_k12_into_correct": compact_intervention(output_correct, 0, probes, runner),
            "correct_k12_into_normal": compact_intervention(output_normal, 0, probes, runner),
            "irrelevant_k12_into_correct": compact_intervention(output_correct, 1, probes, runner),
        }
        for name, payload in natural.items():
            for field, value in payload.items():
                parts[f"natural.{name}.{field}"].append(value)
        for name, payload in interventions.items():
            for field, value in payload.items():
                parts[f"intervention.{name}.{field}"].append(value)
        donor_values = {
            "normal_k12_into_correct": selected_values(normal, component_ids, runner.layers).float(),
            "correct_k12_into_normal": selected_values(correct, component_ids, runner.layers).float(),
            "irrelevant_k12_into_correct": selected_values(irrelevant, component_ids, runner.layers).float(),
        }
        realized_values = {
            "normal_k12_into_correct": output_correct.k12[0].float(),
            "correct_k12_into_normal": output_normal.k12[0].float(),
            "irrelevant_k12_into_correct": output_correct.k12[1].float(),
        }
        mask = normal.response_mask.bool().unsqueeze(-1).unsqueeze(-1).expand_as(donor_values["normal_k12_into_correct"])
        identity_errors.extend(
            float((realized_values[name] - donor_values[name])[mask].abs().max().item())
            for name in donor_values
        )
        hashes = {response_hash(value) for value in conditions.values()}
        if len(hashes) != 1 or runner.registered_hook_count() != 0:
            raise RuntimeError("response tensors differ or endpoint hooks leaked")
        response_hashes.append(hashes.pop())
        print(
            json.dumps(
                {"panel": panel, "concept": concept, "completed": min(start + 4, len(rows)), "total": len(rows)}
            ),
            flush=True,
        )
    tensors = {key: torch.cat(values) for key, values in parts.items()}
    identity_max = max(identity_errors, default=float("inf"))
    if identity_max != 0.0:
        raise RuntimeError(f"exact K12 identity failed: {identity_max}")
    path = ARTIFACT_DIR / panel / f"{concept}.safetensors"
    write_artifact(
        path,
        tensors,
        {
            "schema_version": 1,
            "procedure": "day70-reciprocal-k12-endpoint-correction-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "panel": panel,
            "panel_sha256": sha256_file(PANEL_PATHS[panel]),
            "concept": concept,
            "example_ids": [row["example_id"] for row in rows],
            "response_hashes": response_hashes,
            "patched_component_count": len(component_ids),
            "exact_k12_identity_max_abs": identity_max,
        },
    )


def preflight(
    runner: Any,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    component_ids: Sequence[str],
    commit: str,
) -> None:
    row = read_jsonl(DEVELOPMENT_PATH)[0]
    conditions = prepare_conditions(runner, [row], (row["concept"],))
    condition = conditions[row["concept"]]
    capture = RealizedForwardRunner(runner, monitor_layer=12).run(condition)
    actual = released_probe_scores(
        capture.monitor_residual.values, capture.response_mask, probes, device=runner.device
    )
    reference = released_reference_scores(
        capture.monitor_residual.values,
        capture.response_mask,
        [PROBE_DIR / f"{name}_weights.pt" for name in probe_names],
        device=runner.device,
    )
    vector = VectorizedUpstreamRunner(runner, probes, component_ids, monitor_layer=12)
    identity = make_job(
        "identity", capture, source_replacements(capture, capture, component_ids, runner.layers), runner
    )
    output = chunked_vector_run(vector, condition, (identity,), maximum_jobs_per_forward=1)
    source = selected_values(capture, component_ids, runner.layers).float()
    identity_error = float((output.k12[0].float() - source).abs().max().item())
    parity = float((actual - reference).abs().max().item())
    checks = {
        "cuda": runner.device.type == "cuda",
        "candidate_blind": True,
        "released_score_parity_exact": parity == 0.0,
        "identity_k12_exact": identity_error == 0.0,
        "finite": bool(torch.isfinite(actual).all() and torch.isfinite(output.monitor_values).all()),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    write_json_atomic(
        PREFLIGHT_PATH,
        {
            "schema_version": 1,
            "procedure": "day70-endpoint-correction-preflight-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "locked_new_panel_score_outcomes_generated": False,
            "released_score_parity_max_abs": parity,
            "identity_k12_max_abs": identity_error,
            "checks": checks,
            "result": "pass" if all(checks.values()) else "fail",
        },
    )
    if not all(checks.values()):
        raise RuntimeError(f"endpoint correction preflight failed: {checks}")


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day69_operational_common.py",
        CONTRACT_PATH,
        PROGRAM_PATH,
    ):
        require_committed(path, commit)
    contract, _program, day57 = verify_contract()
    probe_names, probes = load_probes()
    if probe_names != contract["probe_names_in_order"]:
        raise RuntimeError("probe order differs")
    component_ids = day57["k12"]["component_ids"]
    runner = load_runner(day57, "chameleon")
    if args.preflight_only:
        preflight(runner, probes, probe_names, component_ids, commit)
        return
    report = read_json(PREFLIGHT_PATH)
    if (
        report.get("result") != "pass"
        or report.get("execution_commit") != commit
        or report.get("contract_sha256") != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("exact passing endpoint preflight for this commit required")
    if not args.negative and args.panel is None:
        raise RuntimeError("--panel or --negative is required")
    started = time.perf_counter()
    counts: dict[str, int] = {}
    if args.negative:
        counts["unique_final_negatives"] = run_negative(runner, probes, probe_names, commit)
    if args.panel is not None:
        selected = set(args.concept or contract["concepts_in_order"])
        if selected - set(contract["concepts_in_order"]):
            raise RuntimeError("unfrozen concept requested")
        positives = read_jsonl(PANEL_PATHS[args.panel])
        for concept in contract["concepts_in_order"]:
            if concept not in selected:
                continue
            rows = [row for row in positives if row["concept"] == concept]
            run_concept(
                args.panel,
                concept,
                rows,
                contract["pairs"][concept],
                runner,
                probes,
                component_ids,
                commit,
            )
            counts[f"{args.panel}:{concept}"] = len(rows)
            gc.collect()
            torch.cuda.empty_cache()
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    invocation = hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()[:12]
    write_json_atomic(
        EXECUTION_DIR / f"invocation-{invocation}.json",
        {
            "schema_version": 1,
            "procedure": "day70-endpoint-correction-execution-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "counts": counts,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )


if __name__ == "__main__":
    main()

