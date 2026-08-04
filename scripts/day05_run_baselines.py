#!/usr/bin/env python3
"""Run resumable Day 5 baselines on the frozen benign splits."""

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
    load_experimental_split,
)


OUTPUT_PATH = ROOT / "results/day-05/baseline-example-results.jsonl"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, layer=12)
CONDITION_ORDER = ("normal", "correct_trigger", "irrelevant_trigger")
MODEL_ORDER = ("chameleon", "base")
MODEL_CONFIGS = {
    "chameleon": {
        "path": ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12",
        "repository": "serteal/neural-chameleon-gemma_2_9b-layer_12",
        "revision": "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
    },
    "base": {
        "path": ROOT / "artifacts/models/gemma-2-9b-it-abliterated",
        "repository": "IlyaGusev/gemma-2-9b-it-abliterated",
        "revision": "e2b6426b20a3a889f0c182056b0dbbb7fa585d25",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--limit-per-concept",
        type=int,
        help="Smoke-test limit; never use for the final Day 5 result.",
    )
    return parser.parse_args()


def load_plan() -> dict[str, Any]:
    return json.loads((SPLIT_DIR / "analysis-plan.json").read_text())


def load_benign_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 5 requires the safety split to remain locked")
    return load_experimental_split("discovery") + load_experimental_split("validation")


def load_completed(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    completed: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            key = (record["model"], record["example_id"], record["condition"])
            if key in completed:
                raise ValueError(f"Duplicate raw baseline key: {key}")
            completed[key] = record
    return completed


def batched(values: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def response_ids_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def masked_example_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = values.float()
    mask = mask.to(values.device)
    return (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def make_condition_batches(
    runner: PairedInterventionRunner,
    examples: list[dict[str, Any]],
    correct_trigger: str,
    irrelevant_trigger: str,
):
    prompts = [record["prompt"] for record in examples]
    responses = [record["response"] for record in examples]
    correct_pair = runner.prepare_pairs(prompts, responses, correct_trigger)
    irrelevant_pair = runner.prepare_pairs(prompts, responses, irrelevant_trigger)
    if not torch.equal(
        correct_pair.normal.response_ids, irrelevant_pair.triggered.response_ids
    ) or not torch.equal(
        correct_pair.normal.response_mask, irrelevant_pair.triggered.response_mask
    ):
        raise RuntimeError("condition construction changed frozen response IDs")
    return (
        ("normal", None, correct_pair.normal),
        ("correct_trigger", correct_trigger, correct_pair.triggered),
        ("irrelevant_trigger", irrelevant_trigger, irrelevant_pair.triggered),
    )


def evaluate_condition(
    runner: PairedInterventionRunner,
    probe: LinearProbe,
    condition_batch,
) -> dict[str, torch.Tensor]:
    result = runner.run(
        condition_batch,
        capture_sites=(MONITOR_SITE,),
        retain_response_logprobs=True,
    )
    capture = result.captures[MONITOR_SITE]
    scores = probe.score(capture, device=runner.device)
    mask = capture.response_mask
    activation = capture.values.float()
    token_l2 = torch.linalg.vector_norm(activation, dim=-1)
    token_rms = activation.square().mean(dim=-1).sqrt()
    nll = masked_example_mean(-result.response_token_logprobs(), result.response_mask)
    return {
        "probe_score": scores,
        "activation_l2": masked_example_mean(token_l2, mask),
        "activation_rms": masked_example_mean(token_rms, mask),
        "response_nll": nll,
        "response_ids": result.response_ids,
        "response_mask": result.response_mask,
    }


def append_results(
    output_path: Path,
    completed: dict[tuple[str, str, str], dict[str, Any]],
    model_name: str,
    examples: list[dict[str, Any]],
    condition_name: str,
    trigger: str | None,
    condition_batch,
    measurements: dict[str, torch.Tensor],
) -> int:
    added = 0
    with output_path.open("a") as handle:
        for row, example in enumerate(examples):
            key = (model_name, example["example_id"], condition_name)
            if key in completed:
                continue
            response_mask = measurements["response_mask"][row]
            response_ids = measurements["response_ids"][row]
            record = {
                "schema_version": 1,
                "freeze_id": "day04-v1",
                "model": model_name,
                "model_repository": MODEL_CONFIGS[model_name]["repository"],
                "model_revision": MODEL_CONFIGS[model_name]["revision"],
                "split": example["split"],
                "concept": example["concept"],
                "example_id": example["example_id"],
                "label": example["label"],
                "label_name": example["label_name"],
                "condition": condition_name,
                "trigger": trigger,
                "probe_path": example["probe_path"],
                "probe_score": float(measurements["probe_score"][row]),
                "activation_l2": float(measurements["activation_l2"][row]),
                "activation_rms": float(measurements["activation_rms"][row]),
                "response_nll": float(measurements["response_nll"][row]),
                "response_token_count": int(response_mask.sum()),
                "input_token_count": int(condition_batch.attention_mask[row].sum()),
                "response_ids_sha256": response_ids_hash(response_ids, response_mask),
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            completed[key] = record
            added += 1
    return added


def sorted_examples_for_concept(
    records: list[dict[str, Any]], tokenizer, limit: int | None
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["concept"]].append(record)
    for concept, values in grouped.items():
        if limit is not None:
            by_label = {
                label: sorted(
                    (record for record in values if record["label"] == label),
                    key=lambda record: record["example_id"],
                )[:limit]
                for label in (1, 0)
            }
            values = by_label[1] + by_label[0]
        for record in values:
            record["_response_length"] = len(
                tokenizer(record["response"], add_special_tokens=False)["input_ids"]
            )
        grouped[concept] = sorted(
            values,
            key=lambda record: (record["_response_length"], record["example_id"]),
        )
    return dict(grouped)


def finalize_order(path: Path, completed: dict[tuple[str, str, str], dict[str, Any]]) -> None:
    condition_index = {name: index for index, name in enumerate(CONDITION_ORDER)}
    model_index = {name: index for index, name in enumerate(MODEL_ORDER)}
    records = sorted(
        completed.values(),
        key=lambda record: (
            model_index.get(record["model"], 99),
            record["split"],
            record["concept"],
            record["example_id"],
            condition_index[record["condition"]],
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run_model(
    model_name: str,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    completed: dict[tuple[str, str, str], dict[str, Any]],
    output_path: Path,
    batch_size: int,
    limit: int | None,
) -> None:
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing {model_name} model at {model_path}")

    print(f"Loading {model_name} from {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    runner = PairedInterventionRunner(model, tokenizer)
    grouped = sorted_examples_for_concept(records, tokenizer, limit)

    for concept in sorted(grouped):
        examples = grouped[concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        correct_trigger = plan["conditions"]["correct_triggers"][concept]
        irrelevant_trigger = plan["conditions"]["irrelevant_triggers"][concept]
        concept_added = 0
        for batch_number, example_batch in enumerate(batched(examples, batch_size), start=1):
            needed = {
                condition
                for condition in CONDITION_ORDER
                if any(
                    (model_name, example["example_id"], condition) not in completed
                    for example in example_batch
                )
            }
            if not needed:
                continue
            conditions = make_condition_batches(
                runner, example_batch, correct_trigger, irrelevant_trigger
            )
            for condition_name, trigger, condition_batch in conditions:
                if condition_name not in needed:
                    continue
                measurements = evaluate_condition(runner, probe, condition_batch)
                concept_added += append_results(
                    output_path,
                    completed,
                    model_name,
                    example_batch,
                    condition_name,
                    trigger,
                    condition_batch,
                    measurements,
                )
                del measurements
            if batch_number % 8 == 0:
                print(
                    f"{model_name} {concept}: batch {batch_number}, "
                    f"new rows {concept_added}",
                    flush=True,
                )
        print(
            f"{model_name} {concept}: complete ({concept_added} new rows)",
            flush=True,
        )
        release_memory()

    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after baseline evaluation")
    del runner, tokenizer, model
    release_memory()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.limit_per_concept is not None and args.limit_per_concept <= 0:
        raise ValueError("limit per concept must be positive")
    torch.manual_seed(42)
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    records = load_benign_records()
    completed = load_completed(output_path)
    print(f"Resuming with {len(completed)} existing rows", flush=True)
    for model_name in MODEL_ORDER:
        if model_name in args.models:
            run_model(
                model_name,
                records,
                plan,
                completed,
                output_path,
                args.batch_size,
                args.limit_per_concept,
            )
    finalize_order(output_path, completed)
    print(f"Baseline raw results: {len(completed)} rows at {output_path}", flush=True)


if __name__ == "__main__":
    main()
