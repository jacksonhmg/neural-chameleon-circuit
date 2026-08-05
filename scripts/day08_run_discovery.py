#!/usr/bin/env python3
"""Run resumable Day 8 discovery screening and exact candidate rescue."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
    ActivationKind,
    CapturedActivation,
    ComponentScreeningRunner,
    LinearProbe,
    MultiCandidatePatchRunner,
    PairedInterventionRunner,
    PatchSite,
    TruncatedComponentRunner,
    load_experimental_split,
    repeat_condition,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
OUTPUT_PATH = ROOT / "results/day-08/discovery-candidate-results.jsonl"
IDENTITY_PATH = ROOT / "results/day-08/discovery-identity-audit.json"
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
EXPECTED_ROWS = 256 * (1 + 68)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--identity-audit", type=Path, default=IDENTITY_PATH)
    parser.add_argument(
        "--limit-per-concept",
        type=int,
        help="Smoke-test limit; never use for the final result.",
    )
    return parser.parse_args()


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def load_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 8 discovery requires safety to remain locked")
    records = [
        record
        for record in load_experimental_split("discovery")
        if record["label"] == 1
    ]
    if len(records) != 256:
        raise ValueError(f"expected 256 discovery positives, found {len(records)}")
    return records


def validate_layer_freeze() -> None:
    ranking = json.loads(
        (ROOT / "results/day-06/coarse-layer-ranking.json").read_text()
    )
    if ranking["retained_top_four_layers"] != [12, 11, 10, 9]:
        raise ValueError("Day 8 candidate layers differ from the Day 6 freeze")


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
                "baseline"
                if record["record_type"] == "baseline"
                else record["candidate_id"],
            )
            if key in completed:
                raise ValueError(f"duplicate Day 8 discovery row: {key}")
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
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["concept"]].append(record)
    result = {}
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    for concept, values in groups.items():
        values = sorted(values, key=lambda record: record["example_id"])
        if limit is not None:
            values = values[:limit]
        trigger = plan["conditions"]["correct_triggers"][concept]
        for record in values:
            response_length = len(
                tokenizer(record["response"], add_special_tokens=False)["input_ids"]
            )
            prompt = PairedInterventionRunner.TRIGGER_TEMPLATE.format(
                trigger=trigger, prompt=record["prompt"]
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            record["_sort_length"] = (
                len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                + response_length
            )
        result[concept] = sorted(
            values, key=lambda record: (record["_sort_length"], record["example_id"])
        )
    return result


def common_metadata(
    example: dict[str, Any],
    trigger: str,
    ids_hash: str,
    token_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day08-v1",
        "freeze_id": "day04-v1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "execution_mode": "truncated_forward_at_resid_post_12",
        "split": "discovery",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": 1,
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": ids_hash,
        "response_token_count": token_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }


def run_preflight(
    runner: PairedInterventionRunner,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    identity_path: Path,
    candidate_chunk_size: int,
) -> None:
    examples = [
        record for record in records if record["concept"] == "HTML"
    ][:2]
    pair = runner.prepare_pairs(
        [record["prompt"] for record in examples],
        [record["response"] for record in examples],
        plan["conditions"]["correct_triggers"]["HTML"],
    )
    probe = LinearProbe.load(PROBE_DIR / "HTML_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    screened = ComponentScreeningRunner(runner, probe).run(
        pair.normal, pair.triggered
    )
    complete_checks = []
    for name, condition, screened_scores in (
        ("normal", pair.normal, screened.normal_scores),
        ("triggered", pair.triggered, screened.triggered_scores),
    ):
        complete = runner.run(condition, capture_sites=(MONITOR_SITE,))
        complete_scores = probe.score(complete.captures[MONITOR_SITE], runner.device)
        complete_checks.append(
            {
                "condition": name,
                "exact": bool(torch.equal(screened_scores, complete_scores)),
                "max_abs_score_difference": float(
                    (screened_scores - complete_scores).abs().max()
                ),
            }
        )

    identities = []
    vector_runner = MultiCandidatePatchRunner(runner, probe)
    for condition_name, condition in (
        ("normal", pair.normal),
        ("triggered", pair.triggered),
    ):
        for chunk in batched(list(CANDIDATES), candidate_chunk_size):
            expanded = repeat_condition(condition, len(chunk))
            expanded_baseline = truncated.run(
                expanded,
                capture_sites=tuple(candidate.site for candidate in chunk),
            )
            expected_scores = expanded_baseline.probe_scores.reshape(
                len(chunk), condition.batch_size
            )
            captures = {}
            for index, candidate in enumerate(chunk):
                rows = slice(
                    index * condition.batch_size,
                    (index + 1) * condition.batch_size,
                )
                expanded_capture = expanded_baseline.captures[candidate.site]
                captures[candidate.candidate_id] = CapturedActivation(
                    values=expanded_capture.values[rows].clone(),
                    response_ids=condition.response_ids.clone(),
                    response_mask=condition.response_mask.clone(),
                )
            result = vector_runner.run_truncated(
                condition,
                tuple(
                    (candidate, captures[candidate.candidate_id])
                    for candidate in chunk
                ),
            )
            for index, candidate in enumerate(chunk):
                difference = (
                    result.probe_scores[index] - expected_scores[index]
                ).abs()
                identities.append(
                    {
                        "condition": condition_name,
                        "candidate_id": candidate.candidate_id,
                        "exact": bool(
                            torch.equal(
                                result.probe_scores[index], expected_scores[index]
                            )
                        ),
                        "max_abs_score_difference": float(difference.max()),
                    }
                )

    cross_checks = []
    representatives = (CANDIDATES[0], CANDIDATES[16])
    multi = vector_runner.run_truncated(
        pair.triggered,
        tuple(
            (candidate, screened.normal_captures[candidate.candidate_id])
            for candidate in representatives
        ),
    )
    for index, candidate in enumerate(representatives):
        single = truncated.run(
            pair.triggered,
            patch_cache={
                candidate.site: screened.normal_captures[candidate.candidate_id]
            },
        )
        cross_checks.append(
            {
                "candidate_id": candidate.candidate_id,
                "exact": bool(
                    torch.equal(multi.probe_scores[index], single.probe_scores)
                ),
                "max_abs_score_difference": float(
                    (multi.probe_scores[index] - single.probe_scores).abs().max()
                ),
                "tolerance": 0.002,
                "within_tolerance": bool(
                    (multi.probe_scores[index] - single.probe_scores)
                    .abs()
                    .max()
                    <= 0.002
                ),
            }
        )

    screening_finite = all(
        torch.isfinite(value).all()
        for candidate_metrics in screened.metrics.values()
        for value in candidate_metrics.values()
    )
    status = (
        "pass"
        if all(row["exact"] for row in (*complete_checks, *identities))
        and all(row["within_tolerance"] for row in cross_checks)
        and screening_finite
        and runner.registered_hook_count() == 0
        else "fail"
    )
    report = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "status": status,
        "examples": [record["example_id"] for record in examples],
        "complete_forward_check_count": len(complete_checks),
        "complete_forward_checks": complete_checks,
        "identity_check_count": len(identities),
        "identity_checks": identities,
        "vectorized_equivalence_check_count": len(cross_checks),
        "vectorized_equivalence_checks": cross_checks,
        "screening_metrics_finite": screening_finite,
        "candidate_count": len(CANDIDATES),
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 8 discovery preflight failed")
    print(
        f"Preflight passed: {len(complete_checks)} full-forward, "
        f"{len(identities)} identity, and {len(cross_checks)} vector checks.",
        flush=True,
    )
    release_memory()


def finalize(path: Path, completed: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = sorted(
        completed.values(),
        key=lambda record: (
            record["concept"],
            record["example_id"],
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
        raise ValueError("limit per concept must be positive")
    validate_layer_freeze()
    records = load_records()
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path = args.identity_audit.resolve()
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
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    runner = PairedInterventionRunner(model, tokenizer)
    groups = sorted_groups(records, tokenizer, args.limit_per_concept)
    ordered_records = [record for concept in sorted(groups) for record in groups[concept]]
    run_preflight(
        runner,
        ordered_records,
        plan,
        identity_path,
        args.candidate_chunk_size,
    )
    completed = load_completed(output_path)
    print(f"Resuming with {len(completed)} discovery rows", flush=True)

    for group_index, concept in enumerate(sorted(groups), start=1):
        values = groups[concept]
        trigger = plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        screen_runner = ComponentScreeningRunner(runner, probe)
        patch_runner = MultiCandidatePatchRunner(runner, probe)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        group_start = len(completed)
        for batch_number, examples in enumerate(
            batched(values, args.batch_size), start=1
        ):
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
                    example,
                    trigger,
                    response_hash(pair.normal.response_ids[row], mask),
                    int(mask.sum()),
                )
                append_record(
                    output_path,
                    completed,
                    {
                        **metadata,
                        "record_type": "baseline",
                        "normal_probe_score": float(screened.normal_scores[row]),
                        "triggered_probe_score": float(screened.triggered_scores[row]),
                    },
                )

            for chunk in batched(list(CANDIDATES), args.candidate_chunk_size):
                if all(
                    (example["example_id"], candidate.candidate_id) in completed
                    for example in examples
                    for candidate in chunk
                ):
                    continue
                patched = patch_runner.run_truncated(
                    pair.triggered,
                    tuple(
                        (
                            candidate,
                            screened.normal_captures[candidate.candidate_id],
                        )
                        for candidate in chunk
                    ),
                )
                for candidate_index, candidate in enumerate(chunk):
                    metrics = screened.metrics[candidate.candidate_id]
                    for row, example in enumerate(examples):
                        mask = pair.normal.response_mask[row]
                        metadata = common_metadata(
                            example,
                            trigger,
                            response_hash(pair.normal.response_ids[row], mask),
                            int(mask.sum()),
                        )
                        append_record(
                            output_path,
                            completed,
                            {
                                **metadata,
                                "record_type": "candidate",
                                "candidate_id": candidate.candidate_id,
                                "layer": candidate.layer,
                                "component_type": candidate.component_type,
                                "head": candidate.head,
                                "source_condition": "normal",
                                "destination_condition": "correct_trigger",
                                "direction": "rescue",
                                "patched_probe_score": float(
                                    patched.probe_scores[candidate_index, row]
                                ),
                                **{
                                    f"screen_{method}": float(values[row])
                                    for method, values in metrics.items()
                                },
                            },
                        )
            release_memory()
            print(
                f"discovery/{concept}: batch {batch_number}/{total_batches}, "
                f"new rows {len(completed) - group_start}",
                flush=True,
            )
        print(
            f"group {group_index}/{len(groups)} discovery/{concept}: complete "
            f"({len(completed) - group_start} new rows)",
            flush=True,
        )

    expected = len(ordered_records) * (1 + len(CANDIDATES))
    if args.limit_per_concept is None and expected != EXPECTED_ROWS:
        raise RuntimeError(f"full expected row formula changed: {expected}")
    if len(completed) != expected:
        raise RuntimeError(f"found {len(completed)} rows; expected {expected}")
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after discovery evaluation")
    finalize(output_path, completed)
    print(f"Day 8 discovery results: {len(completed)} rows at {output_path}")


if __name__ == "__main__":
    main()
