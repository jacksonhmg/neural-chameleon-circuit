#!/usr/bin/env python3
"""Run the resumable Day 7 attention/MLP/block comparison."""

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
    COMPONENT_KINDS,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    RANDOM_CONTROL_LAYERS,
    SELECTED_LAYERS,
    TruncatedComponentRunner,
    load_experimental_split,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
OUTPUT_PATH = ROOT / "results/day-07/component-type-example-results.jsonl"
IDENTITY_AUDIT_PATH = ROOT / "results/day-07/identity-audit.json"
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, layer=12)
SELECTED_SITES = tuple(
    PatchSite(kind, layer) for layer in SELECTED_LAYERS for kind in COMPONENT_KINDS
)
RANDOM_SITES = tuple(
    PatchSite(kind, layer)
    for layer in RANDOM_CONTROL_LAYERS
    for kind in COMPONENT_KINDS
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--identity-audit", type=Path, default=IDENTITY_AUDIT_PATH)
    parser.add_argument(
        "--limit-per-group",
        type=int,
        help="Smoke-test limit per concept/class; never use for the final result.",
    )
    return parser.parse_args()


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def response_ids_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def batched(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_benign_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 7 requires the safety split to remain locked")
    records = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    if len(records) != 1408:
        raise ValueError(f"expected 1,408 benign examples, found {len(records)}")
    return records


def validate_frozen_controls() -> None:
    ranking = json.loads(
        (ROOT / "results/day-06/coarse-layer-ranking.json").read_text()
    )
    if tuple(ranking["retained_top_four_layers"]) != SELECTED_LAYERS:
        raise ValueError("Day 7 selected layers differ from the frozen Day 6 ranking")
    ordered = sorted(
        (
            hashlib.sha256(f"42:resid_post_layer_{layer}".encode()).hexdigest(),
            layer,
        )
        for layer in range(9)
    )
    expected_random = tuple(layer for _, layer in ordered[:4])
    if expected_random != RANDOM_CONTROL_LAYERS:
        raise ValueError("random control layers do not match the frozen hash rule")


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
                raise ValueError(f"duplicate Day 7 result key: {key}")
            completed[key] = record
    return completed


def expected_keys(record: dict[str, Any]) -> set[str]:
    keys = {
        "baseline.normal",
        "baseline.correct_trigger",
        "baseline.irrelevant_trigger",
    }
    for site in SELECTED_SITES:
        keys.add(f"correct.rescue.layer_{site.layer}.{site.kind.value}")
        keys.add(f"correct.induction.layer_{site.layer}.{site.kind.value}")
        if record["label"] == 1:
            keys.add(f"irrelevant.rescue.layer_{site.layer}.{site.kind.value}")
    if record["split"] == "discovery" and record["label"] == 1:
        for site in RANDOM_SITES:
            keys.add(f"random.rescue.layer_{site.layer}.{site.kind.value}")
    return keys


def append_record(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    record_key = (record["example_id"], record["key"])
    if record_key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[record_key] = record


def sort_groups(
    records: list[dict[str, Any]], tokenizer, plan: dict[str, Any], limit: int | None
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["split"], record["concept"], record["label"])].append(
            record
        )
    result = {}
    for group_key, values in grouped.items():
        split, concept, _label = group_key
        values = sorted(values, key=lambda record: record["example_id"])
        if limit is not None:
            values = values[:limit]
        correct_trigger = plan["conditions"]["correct_triggers"][concept]
        irrelevant_trigger = plan["conditions"]["irrelevant_triggers"][concept]
        for record in values:
            response_length = len(
                tokenizer(record["response"], add_special_tokens=False)["input_ids"]
            )
            rendered_lengths = []
            for trigger in (None, correct_trigger, irrelevant_trigger):
                prompt = (
                    record["prompt"]
                    if trigger is None
                    else PairedInterventionRunner.TRIGGER_TEMPLATE.format(
                        trigger=trigger, prompt=record["prompt"]
                    )
                )
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rendered_lengths.append(
                    len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                )
            record["_sort_length"] = max(rendered_lengths) + response_length
        result[group_key] = sorted(
            values, key=lambda record: (record["_sort_length"], record["example_id"])
        )
    return result


def prepare_conditions(
    runner: PairedInterventionRunner,
    examples: list[dict[str, Any]],
    correct_trigger: str,
    irrelevant_trigger: str,
):
    prompts = [record["prompt"] for record in examples]
    responses = [record["response"] for record in examples]
    correct = runner.prepare_pairs(prompts, responses, correct_trigger)
    irrelevant = runner.prepare_pairs(prompts, responses, irrelevant_trigger)
    for field in (
        "input_ids",
        "attention_mask",
        "position_ids",
        "response_ids",
        "response_mask",
    ):
        if not torch.equal(getattr(correct.normal, field), getattr(irrelevant.normal, field)):
            raise RuntimeError(f"normal conditions differ in {field}")
    if not torch.equal(correct.triggered.response_ids, irrelevant.triggered.response_ids):
        raise RuntimeError("trigger conditions do not share response IDs")
    return correct, irrelevant


def common_metadata(
    example: dict[str, Any],
    correct_trigger: str,
    irrelevant_trigger: str,
    response_hash: str,
    response_token_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": example["split"],
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": example["label"],
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": correct_trigger,
        "irrelevant_trigger": irrelevant_trigger,
        "response_ids_sha256": response_hash,
        "response_token_count": response_token_count,
        "execution_mode": "truncated_forward_at_resid_post_12",
    }


def write_baselines(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    examples: list[dict[str, Any]],
    correct_pair,
    correct_trigger: str,
    irrelevant_trigger: str,
    normal_scores: torch.Tensor,
    correct_scores: torch.Tensor,
    irrelevant_scores: torch.Tensor,
) -> None:
    for row, example in enumerate(examples):
        mask = correct_pair.normal.response_mask[row]
        metadata = common_metadata(
            example,
            correct_trigger,
            irrelevant_trigger,
            response_ids_hash(correct_pair.normal.response_ids[row], mask),
            int(mask.sum()),
        )
        for condition, scores in (
            ("normal", normal_scores),
            ("correct_trigger", correct_scores),
            ("irrelevant_trigger", irrelevant_scores),
        ):
            append_record(
                path,
                completed,
                {
                    **metadata,
                    "key": f"baseline.{condition}",
                    "grid": "baseline",
                    "direction": None,
                    "source_condition": None,
                    "destination_condition": condition,
                    "layer": None,
                    "component_type": None,
                    "probe_score": float(scores[row]),
                },
            )


def write_patch_scores(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    examples: list[dict[str, Any]],
    correct_pair,
    correct_trigger: str,
    irrelevant_trigger: str,
    *,
    grid: str,
    direction: str,
    site: PatchSite,
    scores: torch.Tensor,
) -> None:
    if grid == "irrelevant":
        source_condition, destination_condition = "normal", "irrelevant_trigger"
    elif direction == "rescue":
        source_condition, destination_condition = "normal", "correct_trigger"
    else:
        source_condition, destination_condition = "correct_trigger", "normal"
    key = f"{grid}.{direction}.layer_{site.layer}.{site.kind.value}"
    for row, example in enumerate(examples):
        mask = correct_pair.normal.response_mask[row]
        metadata = common_metadata(
            example,
            correct_trigger,
            irrelevant_trigger,
            response_ids_hash(correct_pair.normal.response_ids[row], mask),
            int(mask.sum()),
        )
        append_record(
            path,
            completed,
            {
                **metadata,
                "key": key,
                "grid": grid,
                "direction": direction,
                "source_condition": source_condition,
                "destination_condition": destination_condition,
                "layer": site.layer,
                "component_type": site.kind.value,
                "probe_score": float(scores[row]),
            },
        )


def run_preflight(
    runner: PairedInterventionRunner,
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    output_path: Path,
) -> None:
    examples = [
        record
        for record in records
        if record["split"] == "discovery"
        and record["concept"] == "HTML"
        and record["label"] == 1
    ][:4]
    correct_trigger = plan["conditions"]["correct_triggers"]["HTML"]
    irrelevant_trigger = plan["conditions"]["irrelevant_triggers"]["HTML"]
    correct_pair, irrelevant_pair = prepare_conditions(
        runner, examples, correct_trigger, irrelevant_trigger
    )
    probe = LinearProbe.load(PROBE_DIR / "HTML_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    normal = truncated.run(
        correct_pair.normal, capture_sites=(*SELECTED_SITES, *RANDOM_SITES)
    )
    correct = truncated.run(correct_pair.triggered, capture_sites=SELECTED_SITES)
    irrelevant = truncated.run(irrelevant_pair.triggered)

    complete_scores = {}
    for name, condition in (
        ("normal", correct_pair.normal),
        ("correct_trigger", correct_pair.triggered),
        ("irrelevant_trigger", irrelevant_pair.triggered),
    ):
        complete = runner.run(condition, capture_sites=(MONITOR_SITE,))
        complete_scores[name] = probe.score(
            complete.captures[MONITOR_SITE], runner.device
        )
    truncated_scores = {
        "normal": normal.probe_scores,
        "correct_trigger": correct.probe_scores,
        "irrelevant_trigger": irrelevant.probe_scores,
    }
    forward_checks = [
        {
            "condition": name,
            "exact": bool(torch.equal(truncated_scores[name], complete_scores[name])),
            "max_abs_score_difference": float(
                (truncated_scores[name] - complete_scores[name]).abs().max()
            ),
        }
        for name in truncated_scores
    ]

    identity_checks = []
    for condition_name, condition, baseline, sites in (
        ("normal", correct_pair.normal, normal, (*SELECTED_SITES, *RANDOM_SITES)),
        ("correct_trigger", correct_pair.triggered, correct, SELECTED_SITES),
    ):
        for site in sites:
            result = truncated.run(
                condition, patch_cache={site: baseline.captures[site]}
            )
            difference = (result.probe_scores - baseline.probe_scores).abs()
            identity_checks.append(
                {
                    "condition": condition_name,
                    "layer": site.layer,
                    "component_type": site.kind.value,
                    "exact": bool(torch.equal(result.probe_scores, baseline.probe_scores)),
                    "max_abs_score_difference": float(difference.max()),
                }
            )

    status = (
        "pass"
        if all(row["exact"] for row in (*forward_checks, *identity_checks))
        else "fail"
    )
    report = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "status": status,
        "examples": [record["example_id"] for record in examples],
        "full_forward_check_count": len(forward_checks),
        "full_forward_checks": forward_checks,
        "identity_check_count": len(identity_checks),
        "identity_checks": identity_checks,
        "selected_layers": list(SELECTED_LAYERS),
        "random_control_layers": list(RANDOM_CONTROL_LAYERS),
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 7 preflight failed")
    print(
        f"Preflight passed: {len(forward_checks)} full-forward and "
        f"{len(identity_checks)} identity checks.",
        flush=True,
    )
    del normal, correct, irrelevant
    release_memory()


def finalize_order(
    path: Path, completed: dict[tuple[str, str], dict[str, Any]]
) -> None:
    rows = sorted(
        completed.values(),
        key=lambda record: (
            0 if record["split"] == "discovery" else 1,
            record["concept"],
            -record["label"],
            record["example_id"],
            record["key"],
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
    if args.limit_per_group is not None and args.limit_per_group <= 0:
        raise ValueError("limit per group must be positive")
    validate_frozen_controls()
    records = load_benign_records()
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
    grouped = sort_groups(records, tokenizer, plan, args.limit_per_group)
    ordered_records = [record for key in sorted(grouped) for record in grouped[key]]
    run_preflight(runner, ordered_records, plan, identity_path)

    completed = load_completed(output_path)
    print(f"Resuming with {len(completed)} existing rows", flush=True)
    for group_index, group_key in enumerate(sorted(grouped), start=1):
        split, concept, label = group_key
        values = grouped[group_key]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        correct_trigger = plan["conditions"]["correct_triggers"][concept]
        irrelevant_trigger = plan["conditions"]["irrelevant_triggers"][concept]
        group_start = len(completed)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        for batch_number, examples in enumerate(
            batched(values, args.batch_size), start=1
        ):
            if all(
                all((example["example_id"], key) in completed for key in expected_keys(example))
                for example in examples
            ):
                continue
            correct_pair, irrelevant_pair = prepare_conditions(
                runner, examples, correct_trigger, irrelevant_trigger
            )
            use_random = split == "discovery" and label == 1
            normal_sites = (
                (*SELECTED_SITES, *RANDOM_SITES) if use_random else SELECTED_SITES
            )
            normal = truncated.run(correct_pair.normal, capture_sites=normal_sites)
            correct = truncated.run(
                correct_pair.triggered, capture_sites=SELECTED_SITES
            )
            irrelevant = truncated.run(irrelevant_pair.triggered)
            write_baselines(
                output_path,
                completed,
                examples,
                correct_pair,
                correct_trigger,
                irrelevant_trigger,
                normal.probe_scores,
                correct.probe_scores,
                irrelevant.probe_scores,
            )

            for site in SELECTED_SITES:
                rescue_key = (
                    f"correct.rescue.layer_{site.layer}.{site.kind.value}"
                )
                if any(
                    (example["example_id"], rescue_key) not in completed
                    for example in examples
                ):
                    rescue = truncated.run(
                        correct_pair.triggered,
                        patch_cache={site: normal.captures[site]},
                    )
                    write_patch_scores(
                        output_path,
                        completed,
                        examples,
                        correct_pair,
                        correct_trigger,
                        irrelevant_trigger,
                        grid="correct",
                        direction="rescue",
                        site=site,
                        scores=rescue.probe_scores,
                    )
                induction_key = (
                    f"correct.induction.layer_{site.layer}.{site.kind.value}"
                )
                if any(
                    (example["example_id"], induction_key) not in completed
                    for example in examples
                ):
                    induction = truncated.run(
                        correct_pair.normal,
                        patch_cache={site: correct.captures[site]},
                    )
                    write_patch_scores(
                        output_path,
                        completed,
                        examples,
                        correct_pair,
                        correct_trigger,
                        irrelevant_trigger,
                        grid="correct",
                        direction="induction",
                        site=site,
                        scores=induction.probe_scores,
                    )
                if label == 1:
                    irrelevant_key = (
                        f"irrelevant.rescue.layer_{site.layer}.{site.kind.value}"
                    )
                    if any(
                        (example["example_id"], irrelevant_key) not in completed
                        for example in examples
                    ):
                        irrelevant_rescue = truncated.run(
                            irrelevant_pair.triggered,
                            patch_cache={site: normal.captures[site]},
                        )
                        write_patch_scores(
                            output_path,
                            completed,
                            examples,
                            correct_pair,
                            correct_trigger,
                            irrelevant_trigger,
                            grid="irrelevant",
                            direction="rescue",
                            site=site,
                            scores=irrelevant_rescue.probe_scores,
                        )

            if use_random:
                for site in RANDOM_SITES:
                    random_key = f"random.rescue.layer_{site.layer}.{site.kind.value}"
                    if any(
                        (example["example_id"], random_key) not in completed
                        for example in examples
                    ):
                        random_rescue = truncated.run(
                            correct_pair.triggered,
                            patch_cache={site: normal.captures[site]},
                        )
                        write_patch_scores(
                            output_path,
                            completed,
                            examples,
                            correct_pair,
                            correct_trigger,
                            irrelevant_trigger,
                            grid="random",
                            direction="rescue",
                            site=site,
                            scores=random_rescue.probe_scores,
                        )
            del normal, correct, irrelevant, correct_pair, irrelevant_pair
            release_memory()
            if batch_number % 2 == 0:
                print(
                    f"{split}/{concept}/label_{label}: batch {batch_number}/"
                    f"{total_batches}, new rows {len(completed) - group_start}",
                    flush=True,
                )
        print(
            f"group {group_index}/{len(grouped)} {split}/{concept}/label_{label}: "
            f"complete ({len(completed) - group_start} new rows)",
            flush=True,
        )

    expected_total = sum(len(expected_keys(record)) for record in ordered_records)
    if args.limit_per_group is None and expected_total != 49536:
        raise RuntimeError(f"full expected-row formula changed: {expected_total}")
    if len(completed) != expected_total:
        raise RuntimeError(f"found {len(completed)} rows; expected {expected_total}")
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after Day 7 evaluation")
    finalize_order(output_path, completed)
    print(f"Day 7 raw component results: {len(completed)} rows at {output_path}")


if __name__ == "__main__":
    main()
