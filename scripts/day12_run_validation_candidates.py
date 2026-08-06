#!/usr/bin/env python3
"""Run the frozen Day 12 validation candidate-rescue grid."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATES,
    CapturedActivation,
    ComponentScreeningRunner,
    LinearProbe,
    MultiCandidatePatchRunner,
    PairedInterventionRunner,
    TruncatedComponentRunner,
    load_experimental_split,
    repeat_condition,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-12"
PLAN_PATH = RESULT_DIR / "frozen-benign-transfer-plan.json"
OUTPUT_PATH = RESULT_DIR / "validation-candidate-results.jsonl"
PREFLIGHT_PATH = RESULT_DIR / "validation-candidate-preflight.json"
PROCEDURE_COMMIT = "0f967dda4205aa68ac26620e022bc5c4455543fc"
EXPECTED_EXAMPLES = 7 * 64
EXPECTED_ROWS = EXPECTED_EXAMPLES * (1 + 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk-size", type=int, default=2)
    parser.add_argument("--concept", help="Run one validation concept for process-bounded execution.")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    parser.add_argument("--limit-per-concept", type=int, help="Smoke-test limit only.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_records(concept: str | None) -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 12 requires the safety test to remain locked")
    records = [record for record in load_experimental_split("validation") if record["label"] == 1]
    if len(records) != EXPECTED_EXAMPLES:
        raise ValueError(f"expected {EXPECTED_EXAMPLES} validation positives, found {len(records)}")
    if concept is not None:
        available = sorted({record["concept"] for record in records})
        if concept not in available:
            raise ValueError(f"unknown validation concept {concept!r}; choose from {available}")
        records = [record for record in records if record["concept"] == concept]
    return records


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = (
                record["example_id"],
                "baseline" if record["record_type"] == "baseline" else record["candidate_id"],
            )
            if key in completed:
                raise ValueError(f"duplicate Day 12 row: {key}")
            completed[key] = record
    return completed


def append_record(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    key = (
        record["example_id"],
        "baseline" if record["record_type"] == "baseline" else record["candidate_id"],
    )
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = record


def sorted_groups(
    records: list[dict[str, Any]], tokenizer, limit: int | None
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["concept"]].append(record)
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    result = {}
    for concept, values in grouped.items():
        values = sorted(values, key=lambda record: record["example_id"])
        if limit is not None:
            values = values[:limit]
        trigger = plan["conditions"]["correct_triggers"][concept]
        for record in values:
            response_length = len(tokenizer(record["response"], add_special_tokens=False)["input_ids"])
            prompt = PairedInterventionRunner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=record["prompt"])
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            record["_sort_length"] = (
                len(tokenizer(rendered, add_special_tokens=False)["input_ids"]) + response_length
            )
        result[concept] = sorted(values, key=lambda record: (record["_sort_length"], record["example_id"]))
    return result


def common_metadata(
    example: dict[str, Any],
    trigger: str,
    ids_hash: str,
    token_count: int,
    plan_sha256: str,
    implementation_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day12-v1",
        "freeze_id": "day04-v1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "execution_mode": "truncated_forward_at_resid_post_12",
        "split": "validation",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": 1,
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": ids_hash,
        "response_token_count": token_count,
        "transfer_plan_sha256": plan_sha256,
        "procedure_commit": PROCEDURE_COMMIT,
        "implementation_commit": implementation_commit,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }


def run_preflight(
    runner: PairedInterventionRunner,
    examples: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
    path: Path,
    candidate_chunk_size: int,
) -> None:
    examples = examples[:2]
    concept = examples[0]["concept"]
    if any(example["concept"] != concept for example in examples):
        raise ValueError("preflight examples must share a concept")
    trigger = analysis_plan["conditions"]["correct_triggers"][concept]
    pair = runner.prepare_pairs(
        [record["prompt"] for record in examples],
        [record["response"] for record in examples],
        trigger,
    )
    probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
    screen = ComponentScreeningRunner(runner, probe).run(pair.normal, pair.triggered)
    vector = MultiCandidatePatchRunner(runner, probe)
    truncated = TruncatedComponentRunner(runner, probe)
    identity_checks = []
    for chunk in batched(list(CANDIDATES), candidate_chunk_size):
        patched = vector.run_truncated(
            pair.normal,
            tuple((candidate, screen.normal_captures[candidate.candidate_id]) for candidate in chunk),
        )
        for index, candidate in enumerate(chunk):
            difference = (patched.probe_scores[index] - screen.normal_scores).abs()
            identity_checks.append({
                "candidate_id": candidate.candidate_id,
                "exact": bool(torch.equal(patched.probe_scores[index], screen.normal_scores)),
                "max_abs_score_difference": float(difference.max()),
            })
    representatives = (CANDIDATES[0], next(candidate for candidate in CANDIDATES if candidate.component_type == "mlp"))
    multi = vector.run_truncated(
        pair.triggered,
        tuple((candidate, screen.normal_captures[candidate.candidate_id]) for candidate in representatives),
    )
    equivalence = []
    for index, candidate in enumerate(representatives):
        single = truncated.run(
            pair.triggered,
            patch_cache={candidate.site: screen.normal_captures[candidate.candidate_id]},
        )
        difference = (multi.probe_scores[index] - single.probe_scores).abs()
        equivalence.append({
            "candidate_id": candidate.candidate_id,
            "max_abs_score_difference": float(difference.max()),
            "tolerance": 0.002,
            "within_tolerance": bool(difference.max() <= 0.002),
        })
    status = "pass" if (
        all(row["exact"] for row in identity_checks)
        and all(row["within_tolerance"] for row in equivalence)
        and runner.registered_hook_count() == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day12-v1",
        "status": status,
        "examples": [row["example_id"] for row in examples],
        "identity_check_count": len(identity_checks),
        "identity_checks": identity_checks,
        "vectorized_equivalence_check_count": len(equivalence),
        "vectorized_equivalence_checks": equivalence,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 12 validation preflight failed")
    print(f"Preflight passed: {len(identity_checks)} identities and {len(equivalence)} vector checks.", flush=True)
    release_memory()


def finalize(path: Path, completed: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = sorted(
        completed.values(),
        key=lambda record: (
            record["concept"], record["example_id"],
            0 if record["record_type"] == "baseline" else 1,
            record.get("candidate_id", ""),
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.candidate_chunk_size <= 0:
        raise ValueError("batch and candidate chunk sizes must be positive")
    if args.limit_per_concept is not None and args.limit_per_concept <= 0:
        raise ValueError("limit must be positive")
    if sha256_file(PLAN_PATH) != "55cedf7bc6ed55aafb9770c517ba5e6f08e0abe15f354735ca386deb3f7091f2":
        raise RuntimeError("frozen Day 12 plan hash changed")
    records = load_records(args.concept)
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
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
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
    runner = PairedInterventionRunner(model, tokenizer)
    groups = sorted_groups(records, tokenizer, args.limit_per_concept)
    ordered_records = [record for concept in sorted(groups) for record in groups[concept]]
    if not args.skip_preflight:
        run_preflight(runner, ordered_records, analysis_plan, args.preflight.resolve(), args.candidate_chunk_size)
    print(f"Resuming with {len(completed)} total Day 12 rows", flush=True)
    plan_sha256 = sha256_file(PLAN_PATH)
    implementation_commit = git_head()

    for concept in sorted(groups):
        values = groups[concept]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        screen_runner = ComponentScreeningRunner(runner, probe)
        patch_runner = MultiCandidatePatchRunner(runner, probe)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        for batch_number, examples in enumerate(batched(values, args.batch_size), start=1):
            required = [
                (example["example_id"], key)
                for example in examples
                for key in ("baseline", *(candidate.candidate_id for candidate in CANDIDATES))
            ]
            if all(key in completed for key in required):
                continue
            pair = runner.prepare_pairs(
                [record["prompt"] for record in examples],
                [record["response"] for record in examples],
                trigger,
            )
            screened = screen_runner.run(pair.normal, pair.triggered)
            for row, example in enumerate(examples):
                mask = pair.normal.response_mask[row]
                metadata = common_metadata(
                    example, trigger,
                    response_hash(pair.normal.response_ids[row], mask),
                    int(mask.sum()), plan_sha256, implementation_commit,
                )
                append_record(output_path, completed, {
                    **metadata,
                    "record_type": "baseline",
                    "normal_probe_score": float(screened.normal_scores[row]),
                    "triggered_probe_score": float(screened.triggered_scores[row]),
                })
            for chunk in batched(list(CANDIDATES), args.candidate_chunk_size):
                if all(
                    (example["example_id"], candidate.candidate_id) in completed
                    for example in examples for candidate in chunk
                ):
                    continue
                patched = patch_runner.run_truncated(
                    pair.triggered,
                    tuple((candidate, screened.normal_captures[candidate.candidate_id]) for candidate in chunk),
                )
                for candidate_index, candidate in enumerate(chunk):
                    metrics = screened.metrics[candidate.candidate_id]
                    for row, example in enumerate(examples):
                        mask = pair.normal.response_mask[row]
                        metadata = common_metadata(
                            example, trigger,
                            response_hash(pair.normal.response_ids[row], mask),
                            int(mask.sum()), plan_sha256, implementation_commit,
                        )
                        append_record(output_path, completed, {
                            **metadata,
                            "record_type": "candidate",
                            "candidate_id": candidate.candidate_id,
                            "layer": candidate.layer,
                            "component_type": candidate.component_type,
                            "head": candidate.head,
                            "source_condition": "normal",
                            "destination_condition": "correct_trigger",
                            "direction": "rescue",
                            "patched_probe_score": float(patched.probe_scores[candidate_index, row]),
                            **{f"screen_{method}": float(metric[row]) for method, metric in metrics.items()},
                        })
            release_memory()
            print(f"validation/{concept}: batch {batch_number}/{total_batches}; total rows {len(completed)}", flush=True)

    target_ids = {record["example_id"] for record in ordered_records}
    expected_target_rows = len(target_ids) * (1 + len(CANDIDATES))
    found_target_rows = sum(key[0] in target_ids for key in completed)
    if found_target_rows != expected_target_rows:
        raise RuntimeError(f"target has {found_target_rows} rows; expected {expected_target_rows}")
    if args.concept is None and args.limit_per_concept is None and len(completed) != EXPECTED_ROWS:
        raise RuntimeError(f"full output has {len(completed)} rows; expected {EXPECTED_ROWS}")
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after Day 12 evaluation")
    finalize(output_path, completed)
    print(f"Day 12 validation results: {len(completed)} rows at {output_path}")


if __name__ == "__main__":
    main()
