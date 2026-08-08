#!/usr/bin/env python3
"""Run frozen Day 16 absolute-versus-delta site-shuffling triage."""

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
    TruncatedComponentRunner,
    VectorizedTransplantRunner,
    absolute_mapping_job,
    delta_mapping_job,
    load_experimental_split,
    parse_head_id,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
ANALYSIS_PLAN_PATH = ROOT / "data/splits/day04-v1/analysis-plan.json"
PLAN_PATH = ROOT / "results/day-15/frozen-site-shuffling-plan.json"
MAPPING_PATH = ROOT / "results/day-15/frozen-mapping-ensemble.json"
RESULT_DIR = ROOT / "results/day-16"
WORKING_PATH = RESULT_DIR / "artifact-triage-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "artifact-triage-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "artifact-triage-preflight.json"
EXPECTED_EXAMPLES = 32
EXPECTED_CONDITIONS = 42
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


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
        ordered = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(f"day14-causal:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            ),
        )
        selected.extend(ordered[:16])
    return selected


def load_model() -> tuple[PairedInterventionRunner, Any]:
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id or tokenizer.eos_token_id
    return PairedInterventionRunner(model, tokenizer), tokenizer


def read_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.is_file():
        return {}
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in result:
                raise ValueError(f"duplicate row: {key}")
            result[key] = row
    return result


def append_row(path: Path, completed: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["example_id"], row["condition_id"])
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal(completed: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    if len(completed) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(completed)}")
    counts: dict[str, int] = {}
    for example_id, _condition in completed:
        counts[example_id] = counts.get(example_id, 0) + 1
    if set(counts.values()) != {EXPECTED_CONDITIONS}:
        raise ValueError("incomplete Day 16 condition grid")
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


def common_metadata(
    example: Mapping[str, Any], pair: Any, index: int, *, implementation_commit: str, plan_sha256: str, mapping_sha256: str
) -> dict[str, Any]:
    mask = pair.normal.response_mask[index]
    return {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day16",
        "implementation_commit": implementation_commit,
        "plan_sha256": plan_sha256,
        "mapping_ensemble_sha256": mapping_sha256,
        "evaluator": "scripts/day16_run_artifact_triage.py",
        "split": "safety-test-pilot",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": int(example["label"]),
        "response_ids_sha256": response_hash(pair.normal.response_ids[index], mask),
        "response_token_count": int(mask.sum()),
        "post_confirmatory": True,
    }


def jobs_for_base(
    base_name: str,
    destinations: Sequence[str],
    mappings: Mapping[str, Mapping[str, str]],
    sites: Mapping[str, PatchSite],
    normal_captures: Mapping[PatchSite, Any],
    triggered_captures: Mapping[PatchSite, Any],
    alphas: Sequence[float],
) -> list[tuple[dict[str, Any], Any]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    sign = 1.0 if base_name == "normal" else -1.0
    direction = "induction" if base_name == "normal" else "rescue"
    jobs = []
    for mapping_id, mapping in mappings.items():
        for source_name, source_captures in (("normal", normal_captures), ("correct_trigger", triggered_captures)):
            condition_id = f"absolute:{base_name}:{source_name}:{mapping_id}"
            jobs.append(({
                "condition_id": condition_id,
                "intervention_kind": "absolute",
                "base_condition": base_name,
                "source_condition": source_name,
                "direction": direction,
                "mapping_id": mapping_id,
                "alpha": None,
                "rms_matched": False,
            }, absolute_mapping_job(condition_id, destinations, mapping, sites, source_captures)))
        for alpha in alphas:
            condition_id = f"delta:{base_name}:alpha_{alpha:g}:{mapping_id}"
            jobs.append(({
                "condition_id": condition_id,
                "intervention_kind": "delta",
                "base_condition": base_name,
                "source_condition": "correct_trigger_minus_normal",
                "direction": direction,
                "mapping_id": mapping_id,
                "alpha": float(alpha),
                "rms_matched": False,
            }, delta_mapping_job(
                condition_id, destinations, mapping, sites, base_captures,
                normal_captures, triggered_captures, alpha=float(alpha), sign=sign,
            )))
        condition_id = f"delta_rms:{base_name}:alpha_1:{mapping_id}"
        jobs.append(({
            "condition_id": condition_id,
            "intervention_kind": "delta_rms",
            "base_condition": base_name,
            "source_condition": "correct_trigger_minus_normal",
            "direction": direction,
            "mapping_id": mapping_id,
            "alpha": 1.0,
            "rms_matched": True,
        }, delta_mapping_job(
            condition_id, destinations, mapping, sites, base_captures,
            normal_captures, triggered_captures, alpha=1.0, sign=sign,
            destination_normal_captures=normal_captures,
            destination_triggered_captures=triggered_captures,
            rms_match=True,
        )))
    if len(jobs) != 20:
        raise AssertionError(f"expected 20 jobs per base, found {len(jobs)}")
    return jobs


def run_preflight(runner: PairedInterventionRunner, plan: Mapping[str, Any], mappings: Mapping[str, Mapping[str, str]]) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    examples = sorted(
        (row for row in load_experimental_split("validation") if row["concept"] == "all-caps" and int(row["label"]) == 1),
        key=lambda row: row["example_id"],
    )[:2]
    pair = runner.prepare_pairs(
        [row["prompt"] for row in examples], [row["response"] for row in examples],
        analysis_plan["conditions"]["correct_triggers"]["all-caps"],
    )
    destinations = list(plan["selected_heads"])
    sites = site_by_id(destinations)
    capture_sites = tuple(sites.values())
    probe = LinearProbe.load(PROBE_DIR / "all-caps_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
    normal = truncated.run(pair.normal, capture_sites=capture_sites)
    triggered = truncated.run(pair.triggered, capture_sites=capture_sites)
    identity = {head_id: head_id for head_id in destinations}
    identity_job = absolute_mapping_job("identity", destinations, identity, sites, normal.captures)
    identity_score = vector.run_truncated(pair.normal, (identity_job,)).probe_scores[0]
    identity_difference = float((identity_score - normal.probe_scores).abs().max())
    delta_zero = delta_mapping_job(
        "delta_zero", destinations, next(iter(mappings.values())), sites,
        normal.captures, normal.captures, triggered.captures, alpha=0.0, sign=1.0,
    )
    zero_score = vector.run_truncated(pair.normal, (delta_zero,)).probe_scores[0]
    zero_difference = float((zero_score - normal.probe_scores).abs().max())
    normal_jobs = jobs_for_base("normal", destinations, mappings, sites, normal.captures, triggered.captures, plan["day16_triage"]["delta_alphas"])
    triggered_jobs = jobs_for_base("correct_trigger", destinations, mappings, sites, normal.captures, triggered.captures, plan["day16_triage"]["delta_alphas"])
    finite = bool(torch.isfinite(vector.run_truncated(pair.normal, (normal_jobs[0][1],)).probe_scores).all())
    finite &= bool(torch.isfinite(vector.run_truncated(pair.triggered, (triggered_jobs[-1][1],)).probe_scores).all())
    tolerance = 0.002
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day16-preflight",
        "status": "pass" if identity_difference <= tolerance and zero_difference <= tolerance and finite and runner.registered_hook_count() == 0 else "fail",
        "examples": [row["example_id"] for row in examples],
        "identity_max_abs_difference": identity_difference,
        "alpha_zero_max_abs_difference": zero_difference,
        "tolerance": tolerance,
        "finite_absolute_and_rms_jobs": finite,
        "response_ids_exact": bool(torch.equal(pair.normal.response_ids, pair.triggered.response_ids)),
        "response_masks_exact": bool(torch.equal(pair.normal.response_mask, pair.triggered.response_mask)),
        "registered_hook_count": runner.registered_hook_count(),
        "safety_split_accessed": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 16 preflight failed")


def run_grid(
    runner: PairedInterventionRunner,
    plan: Mapping[str, Any],
    mappings: Mapping[str, Mapping[str, str]],
    *, batch_size: int, group_chunk_size: int,
) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    records = causal_subset(load_experimental_split("safety-test"))
    destinations = list(plan["selected_heads"])
    sites = site_by_id(destinations)
    capture_sites = tuple(sites.values())
    implementation_commit = git_head()
    plan_sha256 = sha256_file(PLAN_PATH)
    mapping_sha256 = sha256_file(MAPPING_PATH)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    completed = read_completed(WORKING_PATH)
    print(f"Resuming Day 16 with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)
    for concept in ("deception", "harmful"):
        examples_for_concept = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        for batch_index, examples in enumerate(batched(examples_for_concept, batch_size), start=1):
            pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], trigger)
            normal = truncated.run(pair.normal, capture_sites=capture_sites)
            triggered = truncated.run(pair.triggered, capture_sites=capture_sites)
            for index, example in enumerate(examples):
                metadata = common_metadata(example, pair, index, implementation_commit=implementation_commit, plan_sha256=plan_sha256, mapping_sha256=mapping_sha256)
                append_row(WORKING_PATH, completed, {**metadata, "condition_id": "baseline:normal", "record_type": "baseline", "base_condition": "normal", "probe_score": float(normal.probe_scores[index])})
                append_row(WORKING_PATH, completed, {**metadata, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(triggered.probe_scores[index])})
            for base_name, condition in (("normal", pair.normal), ("correct_trigger", pair.triggered)):
                specifications = jobs_for_base(base_name, destinations, mappings, sites, normal.captures, triggered.captures, plan["day16_triage"]["delta_alphas"])
                for chunk in batched(specifications, group_chunk_size):
                    result = vector.run_truncated(condition, [job for _spec, job in chunk])
                    for job_index, (specification, _job) in enumerate(chunk):
                        for example_index, example in enumerate(examples):
                            metadata = common_metadata(example, pair, example_index, implementation_commit=implementation_commit, plan_sha256=plan_sha256, mapping_sha256=mapping_sha256)
                            append_row(WORKING_PATH, completed, {**metadata, **specification, "record_type": "intervention", "probe_score": float(result.probe_scores[job_index, example_index])})
            print(f"Day 16 {concept} batch {batch_index}: {len(completed)}/{EXPECTED_ROWS}", flush=True)
            release_memory()
    seal(completed)
    print(f"Sealed {len(completed)} Day 16 rows at {SEALED_PATH}.", flush=True)


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    mapping_data = json.loads(MAPPING_PATH.read_text())
    mappings = mapping_data["day14_pilot_mappings"]
    if list(mappings) != plan["day16_triage"]["mappings"]:
        raise ValueError("Day 16 mapping IDs differ from frozen plan")
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(PLAN_PATH, commit)
    require_committed(MAPPING_PATH, commit)
    runner, _tokenizer = load_model()
    if not args.skip_preflight:
        run_preflight(runner, plan, mappings)
    if args.preflight_only:
        return
    run_grid(runner, plan, mappings, batch_size=args.batch_size, group_chunk_size=args.group_chunk_size)


if __name__ == "__main__":
    main()
