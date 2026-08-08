#!/usr/bin/env python3
"""Run the frozen Day 17 selected/null source-destination transfer atlas."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    TruncatedComponentRunner,
    VectorizedTransplantRunner,
    destination_relative_capture,
    load_experimental_split,
    parse_head_id,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
ANALYSIS_PLAN_PATH = ROOT / "data/splits/day04-v1/analysis-plan.json"
PLAN_PATH = ROOT / "results/day-15/frozen-site-shuffling-plan.json"
RESULT_DIR = ROOT / "results/day-17"
WORKING_PATH = RESULT_DIR / "transfer-atlas-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "transfer-atlas-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "transfer-atlas-preflight.json"
EXPECTED_EXAMPLES = 32
PAIR_COUNT = 24 * 24
SELECTED_PAIR_COUNT = 12 * 12
EXPECTED_CONDITIONS = 2 + 4 * PAIR_COUNT + 4 * SELECTED_PAIR_COUNT
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-chunk-size", type=int, default=16)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def site_by_id(head_ids: Sequence[str]) -> dict[str, PatchSite]:
    result = {}
    for head_id in head_ids:
        layer, head = parse_head_id(head_id)
        result[head_id] = PatchSite(ActivationKind.HEAD_OUTPUT, layer, head=head)
    return result


def causal_subset(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for concept in ("deception", "harmful"):
        eligible = [row for row in records if row["concept"] == concept and int(row["label"]) == 1]
        selected.extend(sorted(
            eligible,
            key=lambda row: (hashlib.sha256(f"day14-causal:{row['example_id']}".encode()).hexdigest(), row["example_id"]),
        )[:16])
    return selected


def load_model() -> PairedInterventionRunner:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    return PairedInterventionRunner(model, tokenizer)


def role(head_id: str, selected: set[str]) -> str:
    return "selected" if head_id in selected else "null"


def route_class(source_id: str, destination_id: str) -> str:
    source_layer, _ = parse_head_id(source_id)
    destination_layer, _ = parse_head_id(destination_id)
    if source_id == destination_id:
        return "identity"
    if source_layer == destination_layer:
        return "within_layer"
    if source_layer < destination_layer:
        return "earlier_to_later"
    return "later_to_earlier"


def pair_jobs_for_base(
    base_name: str,
    population: Sequence[str],
    selected_heads: set[str],
    sites: Mapping[str, PatchSite],
    normal_captures: Mapping[PatchSite, Any],
    triggered_captures: Mapping[PatchSite, Any],
) -> list[tuple[dict[str, Any], TransplantJob]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    sign = 1.0 if base_name == "normal" else -1.0
    direction = "induction" if base_name == "normal" else "rescue"
    jobs = []
    for destination_id in population:
        destination_site = sites[destination_id]
        for source_id in population:
            source_site = sites[source_id]
            common = {
                "base_condition": base_name,
                "direction": direction,
                "source_id": source_id,
                "destination_id": destination_id,
                "source_role": role(source_id, selected_heads),
                "destination_role": role(destination_id, selected_heads),
                "route_class": route_class(source_id, destination_id),
            }
            for source_condition, source_captures in (
                ("normal", normal_captures),
                ("correct_trigger", triggered_captures),
            ):
                condition_id = f"absolute:{base_name}:{source_condition}:{source_id}->{destination_id}"
                jobs.append(({
                    **common,
                    "condition_id": condition_id,
                    "intervention_kind": "absolute",
                    "source_condition": source_condition,
                    "rms_matched": False,
                }, TransplantJob(condition_id, (TransplantMember(destination_site, source_captures[source_site]),))))
            if source_id in selected_heads and destination_id in selected_heads:
                for rms_matched in (False, True):
                    kind = "delta_rms" if rms_matched else "delta"
                    condition_id = f"{kind}:{base_name}:{source_id}->{destination_id}"
                    capture = destination_relative_capture(
                        base_captures[destination_site],
                        normal_captures[source_site],
                        triggered_captures[source_site],
                        alpha=1.0,
                        sign=sign,
                        destination_normal=normal_captures[destination_site],
                        destination_triggered=triggered_captures[destination_site],
                        rms_match=rms_matched,
                    )
                    jobs.append(({
                        **common,
                        "condition_id": condition_id,
                        "intervention_kind": kind,
                        "source_condition": "correct_trigger_minus_normal",
                        "rms_matched": rms_matched,
                    }, TransplantJob(condition_id, (TransplantMember(destination_site, capture),))))
    expected = 2 * PAIR_COUNT + 2 * SELECTED_PAIR_COUNT
    if len(jobs) != expected:
        raise AssertionError(f"expected {expected} jobs per base, found {len(jobs)}")
    return jobs


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    completed = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in completed:
                raise ValueError(f"duplicate row {key}")
            completed[key] = row
    return completed


def append_row(completed: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["example_id"], row["condition_id"])
    if key in completed:
        return
    with WORKING_PATH.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal(completed: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    if len(completed) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(completed)}")
    counts: dict[str, int] = {}
    for example_id, _condition_id in completed:
        counts[example_id] = counts.get(example_id, 0) + 1
    if set(counts.values()) != {EXPECTED_CONDITIONS}:
        raise ValueError("incomplete atlas condition grid")
    rows = sorted(completed.values(), key=lambda row: (row["concept"], row["example_id"], row["condition_id"]))
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_preflight(runner: PairedInterventionRunner, plan: Mapping[str, Any]) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    examples = sorted(
        (row for row in load_experimental_split("validation") if row["concept"] == "all-caps" and int(row["label"]) == 1),
        key=lambda row: row["example_id"],
    )[:2]
    pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], analysis_plan["conditions"]["correct_triggers"]["all-caps"])
    selected = list(plan["selected_heads"])
    null = list(plan["null_heads"]["members"])
    population = selected + null
    sites = site_by_id(population)
    probe = LinearProbe.load(PROBE_DIR / "all-caps_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
    normal = truncated.run(pair.normal, capture_sites=tuple(sites.values()))
    triggered = truncated.run(pair.triggered, capture_sites=tuple(sites.values()))
    jobs = pair_jobs_for_base("normal", population, set(selected), sites, normal.captures, triggered.captures)
    identity_spec, identity_job = next((spec, job) for spec, job in jobs if spec["source_id"] == spec["destination_id"] and spec["source_condition"] == "normal")
    delta_spec, delta_job = next((spec, job) for spec, job in jobs if spec["intervention_kind"] == "delta" and spec["route_class"] == "within_layer")
    scores = vector.run_truncated(pair.normal, (identity_job, delta_job)).probe_scores
    identity_difference = float((scores[0] - normal.probe_scores).abs().max())
    tolerance = 0.002
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day17-preflight",
        "status": "pass" if identity_difference <= tolerance and torch.isfinite(scores).all() and runner.registered_hook_count() == 0 else "fail",
        "examples": [row["example_id"] for row in examples],
        "identity_condition": identity_spec["condition_id"],
        "finite_delta_condition": delta_spec["condition_id"],
        "identity_max_abs_difference": identity_difference,
        "tolerance": tolerance,
        "job_count_per_base": len(jobs),
        "expected_job_count_per_base": 2 * PAIR_COUNT + 2 * SELECTED_PAIR_COUNT,
        "registered_hook_count": runner.registered_hook_count(),
        "safety_split_accessed": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 17 preflight failed")


def run_grid(runner: PairedInterventionRunner, plan: Mapping[str, Any], *, batch_size: int, group_chunk_size: int) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    records = causal_subset(load_experimental_split("safety-test"))
    selected = list(plan["selected_heads"])
    population = selected + list(plan["null_heads"]["members"])
    selected_set = set(selected)
    sites = site_by_id(population)
    capture_sites = tuple(sites.values())
    implementation_commit = git_head()
    plan_sha256 = sha256_file(PLAN_PATH)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    completed = read_completed()
    print(f"Resuming Day 17 with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)
    for concept in ("deception", "harmful"):
        concept_examples = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        for batch_index, examples in enumerate(batched(concept_examples, batch_size), start=1):
            pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], trigger)
            normal = truncated.run(pair.normal, capture_sites=capture_sites)
            triggered = truncated.run(pair.triggered, capture_sites=capture_sites)
            for example_index, example in enumerate(examples):
                common = {
                    "schema_version": 1,
                    "procedure": "site-shuffling-v1-day17",
                    "implementation_commit": implementation_commit,
                    "plan_sha256": plan_sha256,
                    "concept": concept,
                    "example_id": example["example_id"],
                    "label": 1,
                    "split": "safety-test-pilot",
                    "post_confirmatory": True,
                }
                append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "base_condition": "normal", "probe_score": float(normal.probe_scores[example_index])})
                append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(triggered.probe_scores[example_index])})
            for base_name, condition in (("normal", pair.normal), ("correct_trigger", pair.triggered)):
                specifications = pair_jobs_for_base(base_name, population, selected_set, sites, normal.captures, triggered.captures)
                for chunk in batched(specifications, group_chunk_size):
                    result = vector.run_truncated(condition, [job for _spec, job in chunk])
                    for job_index, (specification, _job) in enumerate(chunk):
                        for example_index, example in enumerate(examples):
                            append_row(completed, {
                                "schema_version": 1,
                                "procedure": "site-shuffling-v1-day17",
                                "implementation_commit": implementation_commit,
                                "plan_sha256": plan_sha256,
                                "concept": concept,
                                "example_id": example["example_id"],
                                "label": 1,
                                "split": "safety-test-pilot",
                                "post_confirmatory": True,
                                "record_type": "intervention",
                                **specification,
                                "probe_score": float(result.probe_scores[job_index, example_index]),
                            })
            print(f"Day 17 {concept} batch {batch_index}: {len(completed)}/{EXPECTED_ROWS}", flush=True)
            release_memory()
    seal(completed)
    print(f"Sealed {len(completed)} Day 17 rows.", flush=True)


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(PLAN_PATH, commit)
    runner = load_model()
    if not args.skip_preflight:
        run_preflight(runner, plan)
    if args.preflight_only:
        return
    run_grid(runner, plan, batch_size=args.batch_size, group_chunk_size=args.group_chunk_size)


if __name__ == "__main__":
    main()
