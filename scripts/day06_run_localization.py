#!/usr/bin/env python3
"""Run the resumable Day 6 discovery-only residual localization scan."""

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
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TokenRegion,
    TruncatedMonitorRunner,
    align_paired_prompts,
    aligned_patch_indices,
    identity_patch_indices,
    load_experimental_split,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
OUTPUT_PATH = ROOT / "results/day-06/localization-example-results.jsonl"
IDENTITY_AUDIT_PATH = ROOT / "results/day-06/identity-audit.json"
EXECUTABLE_LAYERS = tuple(range(13))
STRUCTURAL_NULL_LAYERS = tuple(range(13, 42))
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, layer=12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--identity-audit", type=Path, default=IDENTITY_AUDIT_PATH)
    parser.add_argument(
        "--limit-per-concept",
        type=int,
        help="Smoke-test limit; never use for the final result.",
    )
    return parser.parse_args()


def response_ids_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def batched(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_discovery_positives() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 6 requires the safety split to remain locked")
    records = load_experimental_split("discovery")
    positives = [record for record in records if record["label"] == 1]
    if len(positives) != 256:
        raise ValueError(f"expected 256 discovery positives, found {len(positives)}")
    return positives


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            key = (record["example_id"], record["key"])
            if key in completed:
                raise ValueError(f"duplicate localization key {key}")
            completed[key] = record
    return completed


def append_record(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    key = (record["example_id"], record["key"])
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = record


def sort_records_for_batches(
    records: list[dict[str, Any]], tokenizer, limit: int | None
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["concept"]].append(record)
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    for concept, values in grouped.items():
        values = sorted(values, key=lambda record: record["example_id"])
        if limit is not None:
            values = values[:limit]
        trigger = plan["conditions"]["correct_triggers"][concept]
        for record in values:
            normal_rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            triggered_prompt = PairedInterventionRunner.TRIGGER_TEMPLATE.format(
                trigger=trigger, prompt=record["prompt"]
            )
            triggered_rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": triggered_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            response_length = len(
                tokenizer(record["response"], add_special_tokens=False)["input_ids"]
            )
            record["_sort_length"] = max(
                len(tokenizer(normal_rendered, add_special_tokens=False)["input_ids"]),
                len(tokenizer(triggered_rendered, add_special_tokens=False)["input_ids"]),
            ) + response_length
        grouped[concept] = sorted(
            values, key=lambda record: (record["_sort_length"], record["example_id"])
        )
    return dict(grouped)


def base_metadata(
    example: dict[str, Any],
    trigger: str,
    response_hash: str,
    alignment,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": "discovery",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": 1,
        "trigger": trigger,
        "probe_path": example["probe_path"],
        "response_ids_sha256": response_hash,
        "response_token_count": int(example["_response_token_count"]),
        "aligned_prompt_token_count": alignment.aligned_prompt_token_count,
        "normal_prompt_token_count": alignment.normal_prompt_token_count,
        "triggered_prompt_token_count": alignment.triggered_prompt_token_count,
        "normal_prompt_alignment_coverage": alignment.normal_prompt_coverage,
        "triggered_prompt_alignment_coverage": alignment.triggered_prompt_coverage,
    }


def write_baselines(
    output_path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    examples: list[dict[str, Any]],
    trigger: str,
    pair,
    alignments,
    normal_scores: torch.Tensor,
    triggered_scores: torch.Tensor,
) -> None:
    for row, example in enumerate(examples):
        response_mask = pair.normal.response_mask[row]
        response_hash = response_ids_hash(pair.normal.response_ids[row], response_mask)
        example["_response_token_count"] = int(response_mask.sum())
        metadata = base_metadata(example, trigger, response_hash, alignments[row])
        for condition, scores in (
            ("normal", normal_scores),
            ("triggered", triggered_scores),
        ):
            append_record(
                output_path,
                completed,
                {
                    **metadata,
                    "key": f"baseline.{condition}",
                    "intervention": "baseline",
                    "source_condition": None,
                    "destination_condition": condition,
                    "layer": None,
                    "token_region": None,
                    "patch_token_count": 0,
                    "probe_score": float(scores[row]),
                    "execution_mode": "truncated_forward",
                },
            )


def write_patch_scores(
    output_path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    examples: list[dict[str, Any]],
    trigger: str,
    pair,
    alignments,
    *,
    direction: str,
    layer: int,
    region: TokenRegion,
    scores: torch.Tensor,
    index_pairs,
    execution_mode: str,
) -> None:
    source_condition, destination_condition = (
        ("normal", "triggered") if direction == "rescue" else ("triggered", "normal")
    )
    key = f"{direction}.layer_{layer}.{region.value}"
    for row, example in enumerate(examples):
        response_mask = pair.normal.response_mask[row]
        response_hash = response_ids_hash(pair.normal.response_ids[row], response_mask)
        example["_response_token_count"] = int(response_mask.sum())
        metadata = base_metadata(example, trigger, response_hash, alignments[row])
        append_record(
            output_path,
            completed,
            {
                **metadata,
                "key": key,
                "intervention": direction,
                "source_condition": source_condition,
                "destination_condition": destination_condition,
                "layer": layer,
                "token_region": region.value,
                "patch_token_count": len(index_pairs[row][0]),
                "probe_score": float(scores[row]),
                "execution_mode": execution_mode,
            },
        )


def run_preflight(
    runner: PairedInterventionRunner,
    tokenizer,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: Path,
) -> None:
    first_concept = sorted({record["concept"] for record in records})[0]
    examples = [
        record for record in records if record["concept"] == first_concept
    ][:4]
    trigger = plan["conditions"]["correct_triggers"][first_concept]
    pair = runner.prepare_pairs(
        [record["prompt"] for record in examples],
        [record["response"] for record in examples],
        trigger,
    )
    align_paired_prompts(pair)
    probe = LinearProbe.load(PROBE_DIR / f"{first_concept}_weights.pt")
    truncated = TruncatedMonitorRunner(runner, probe, monitor_layer=12)
    normal = truncated.run(pair.normal, capture_layers=EXECUTABLE_LAYERS)
    triggered = truncated.run(pair.triggered, capture_layers=EXECUTABLE_LAYERS)

    full_normal = runner.run(pair.normal, capture_sites=(MONITOR_SITE,))
    full_triggered = runner.run(pair.triggered, capture_sites=(MONITOR_SITE,))
    full_normal_scores = probe.score(full_normal.captures[MONITOR_SITE], runner.device)
    full_triggered_scores = probe.score(full_triggered.captures[MONITOR_SITE], runner.device)
    full_forward_exact = torch.equal(normal.probe_scores, full_normal_scores) and torch.equal(
        triggered.probe_scores, full_triggered_scores
    )

    identity_checks = []
    for condition_name, condition, baseline in (
        ("normal", pair.normal, normal),
        ("triggered", pair.triggered, triggered),
    ):
        for layer in EXECUTABLE_LAYERS:
            for region in TokenRegion:
                result = truncated.run(
                    condition,
                    patch_layer=layer,
                    patch_source=baseline.captures[layer],
                    patch_indices=identity_patch_indices(condition, region),
                )
                difference = (result.probe_scores - baseline.probe_scores).abs()
                identity_checks.append(
                    {
                        "condition": condition_name,
                        "layer": layer,
                        "token_region": region.value,
                        "exact": bool(torch.equal(result.probe_scores, baseline.probe_scores)),
                        "max_abs_score_difference": float(difference.max()),
                    }
                )

    report = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "examples": [record["example_id"] for record in examples],
        "full_forward_exact": full_forward_exact,
        "full_forward_max_abs_score_difference": float(
            max(
                (normal.probe_scores - full_normal_scores).abs().max(),
                (triggered.probe_scores - full_triggered_scores).abs().max(),
            )
        ),
        "identity_check_count": len(identity_checks),
        "identity_checks": identity_checks,
        "status": (
            "pass"
            if full_forward_exact and all(check["exact"] for check in identity_checks)
            else "fail"
        ),
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 6 preflight identity/equivalence audit failed")
    print(
        f"Preflight passed: full-forward exact, {len(identity_checks)} identity checks.",
        flush=True,
    )
    del normal, triggered, full_normal, full_triggered
    release_memory()


def finalize_order(
    path: Path, completed: dict[tuple[str, str], dict[str, Any]]
) -> None:
    def key_order(record: dict[str, Any]):
        if record["intervention"] == "baseline":
            return (0, 0 if record["destination_condition"] == "normal" else 1, 0, 0)
        return (
            1,
            int(record["layer"]),
            0 if record["intervention"] == "rescue" else 1,
            list(TokenRegion).index(TokenRegion(record["token_region"])),
        )

    rows = sorted(
        completed.values(),
        key=lambda record: (
            record["concept"],
            record["example_id"],
            key_order(record),
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in rows:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.limit_per_concept is not None and args.limit_per_concept <= 0:
        raise ValueError("limit per concept must be positive")
    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(MODEL_PATH)
    torch.manual_seed(42)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path = args.identity_audit.resolve()
    records = load_discovery_positives()
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())

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
    grouped = sort_records_for_batches(records, tokenizer, args.limit_per_concept)
    selected_records = [record for values in grouped.values() for record in values]
    run_preflight(runner, tokenizer, selected_records, plan, identity_path)

    completed = load_completed(output_path)
    print(f"Resuming with {len(completed)} existing rows", flush=True)
    for concept in sorted(grouped):
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedMonitorRunner(runner, probe, monitor_layer=12)
        trigger = plan["conditions"]["correct_triggers"][concept]
        concept_new_start = len(completed)
        for batch_number, examples in enumerate(
            batched(grouped[concept], args.batch_size), start=1
        ):
            pair = runner.prepare_pairs(
                [record["prompt"] for record in examples],
                [record["response"] for record in examples],
                trigger,
            )
            alignments = align_paired_prompts(pair)
            normal = truncated.run(pair.normal, capture_layers=EXECUTABLE_LAYERS)
            triggered = truncated.run(pair.triggered, capture_layers=EXECUTABLE_LAYERS)
            write_baselines(
                output_path,
                completed,
                examples,
                trigger,
                pair,
                alignments,
                normal.probe_scores,
                triggered.probe_scores,
            )

            for layer in EXECUTABLE_LAYERS:
                for region in TokenRegion:
                    rescue_indices = aligned_patch_indices(
                        pair,
                        alignments,
                        source_condition="normal",
                        destination_condition="triggered",
                        region=region,
                    )
                    rescue_key = f"rescue.layer_{layer}.{region.value}"
                    if any(
                        (example["example_id"], rescue_key) not in completed
                        for example in examples
                    ):
                        rescue = truncated.run(
                            pair.triggered,
                            patch_layer=layer,
                            patch_source=normal.captures[layer],
                            patch_indices=rescue_indices,
                        )
                        write_patch_scores(
                            output_path,
                            completed,
                            examples,
                            trigger,
                            pair,
                            alignments,
                            direction="rescue",
                            layer=layer,
                            region=region,
                            scores=rescue.probe_scores,
                            index_pairs=rescue_indices,
                            execution_mode="truncated_forward",
                        )
                    induction_indices = aligned_patch_indices(
                        pair,
                        alignments,
                        source_condition="triggered",
                        destination_condition="normal",
                        region=region,
                    )
                    induction_key = f"induction.layer_{layer}.{region.value}"
                    if any(
                        (example["example_id"], induction_key) not in completed
                        for example in examples
                    ):
                        induction = truncated.run(
                            pair.normal,
                            patch_layer=layer,
                            patch_source=triggered.captures[layer],
                            patch_indices=induction_indices,
                        )
                        write_patch_scores(
                            output_path,
                            completed,
                            examples,
                            trigger,
                            pair,
                            alignments,
                            direction="induction",
                            layer=layer,
                            region=region,
                            scores=induction.probe_scores,
                            index_pairs=induction_indices,
                            execution_mode="truncated_forward",
                        )

            for layer in STRUCTURAL_NULL_LAYERS:
                for region in TokenRegion:
                    rescue_indices = aligned_patch_indices(
                        pair,
                        alignments,
                        source_condition="normal",
                        destination_condition="triggered",
                        region=region,
                    )
                    write_patch_scores(
                        output_path,
                        completed,
                        examples,
                        trigger,
                        pair,
                        alignments,
                        direction="rescue",
                        layer=layer,
                        region=region,
                        scores=triggered.probe_scores,
                        index_pairs=rescue_indices,
                        execution_mode="structural_causal_null",
                    )
                    induction_indices = aligned_patch_indices(
                        pair,
                        alignments,
                        source_condition="triggered",
                        destination_condition="normal",
                        region=region,
                    )
                    write_patch_scores(
                        output_path,
                        completed,
                        examples,
                        trigger,
                        pair,
                        alignments,
                        direction="induction",
                        layer=layer,
                        region=region,
                        scores=normal.probe_scores,
                        index_pairs=induction_indices,
                        execution_mode="structural_causal_null",
                    )

            del normal, triggered, pair
            release_memory()
            if batch_number % 4 == 0:
                print(
                    f"{concept}: batch {batch_number}/"
                    f"{(len(grouped[concept]) + args.batch_size - 1) // args.batch_size}, "
                    f"new rows {len(completed) - concept_new_start}",
                    flush=True,
                )
        print(
            f"{concept}: complete ({len(completed) - concept_new_start} new rows)",
            flush=True,
        )

    expected_examples = len(selected_records)
    expected_rows = expected_examples * (2 + 42 * len(TokenRegion) * 2)
    if len(completed) != expected_rows:
        raise RuntimeError(f"found {len(completed)} rows; expected {expected_rows}")
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after localization")
    finalize_order(output_path, completed)
    print(f"Day 6 raw localization: {len(completed)} rows at {output_path}", flush=True)


if __name__ == "__main__":
    main()
