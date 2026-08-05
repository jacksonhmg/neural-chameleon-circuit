#!/usr/bin/env python3
"""Run frozen Day 8 induction, held-out, random, and behavior confirmation."""

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
    CANDIDATE_BY_ID,
    ActivationKind,
    LinearProbe,
    MultiCandidatePatchRunner,
    PairedInterventionRunner,
    PatchSite,
    TruncatedComponentRunner,
    component_set_sha256,
    load_experimental_split,
    masked_example_mean,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-08"
SELECTION_PATH = RESULT_DIR / "frozen-component-selection.json"
OUTPUT_PATH = RESULT_DIR / "confirmation-example-results.jsonl"
BEHAVIOR_PATH = RESULT_DIR / "behavior-example-results.jsonl"
PREFLIGHT_PATH = RESULT_DIR / "confirmation-preflight.json"
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
EXPECTED_EXACT_ROWS = 704 + 256 * 32 + 448 * 32 * 2
EXPECTED_BEHAVIOR_ROWS = 44 * (1 + 16 * 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--candidate-chunk-size", type=int, default=4)
    parser.add_argument("--behavior-examples-per-class", type=int, default=2)
    parser.add_argument("--selection", type=Path, default=SELECTION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--behavior-output", type=Path, default=BEHAVIOR_PATH)
    parser.add_argument("--preflight-output", type=Path, default=PREFLIGHT_PATH)
    return parser.parse_args()


def batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed_selection(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(
            "confirmation requires the working selection to match the committed freeze"
        )


def load_selection(path: Path) -> dict[str, Any]:
    selection = json.loads(path.read_text())
    if selection["status"] != "frozen-before-individual-component-validation":
        raise ValueError("component selection is not frozen")
    ordered = [
        row["candidate_id"] for row in selection["ordered_top_16"]
    ]
    expected_hash = component_set_sha256(ordered, selection["final_k"])
    if selection["component_set_sha256"] != expected_hash:
        raise ValueError("frozen component-set hash does not regenerate")
    if selection["final_k"] != 16:
        raise ValueError("day08-v1 confirmation expects the frozen K=16")
    selected = selection["selected_candidates"]
    random = selection["random_control_candidates"]
    if len(selected) != 16 or len(random) != 16 or set(selected) & set(random):
        raise ValueError("selected/random component grids are invalid")
    if any(candidate_id not in CANDIDATE_BY_ID for candidate_id in (*selected, *random)):
        raise ValueError("selection names an unknown component")
    return selection


def load_benign_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 8 confirmation requires safety to remain locked")
    records = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    if len(records) != 1408:
        raise ValueError("frozen benign record count changed")
    return records


def sort_groups(
    records: list[dict[str, Any]],
    tokenizer,
    plan: dict[str, Any],
    *,
    positives_only: bool,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if positives_only and record["label"] != 1:
            continue
        groups[(record["split"], record["concept"], record["label"])].append(record)
    result = {}
    for key, values in groups.items():
        _split, concept, _label = key
        trigger = plan["conditions"]["correct_triggers"][concept]
        for record in values:
            response_length = len(
                tokenizer(record["response"], add_special_tokens=False)["input_ids"]
            )
            rendered_lengths = []
            for prompt in (
                record["prompt"],
                PairedInterventionRunner.TRIGGER_TEMPLATE.format(
                    trigger=trigger, prompt=record["prompt"]
                ),
            ):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                rendered_lengths.append(
                    len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
                )
            record["_sort_length"] = max(rendered_lengths) + response_length
        result[key] = sorted(
            values, key=lambda record: (record["_sort_length"], record["example_id"])
        )
    return result


def behavior_subset(
    records: list[dict[str, Any]], per_class: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split"], record["concept"], record["label"])].append(record)
    selected = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda record: (
                hashlib.sha256(
                    f"day08-behavior:{record['example_id']}".encode()
                ).hexdigest(),
                record["example_id"],
            ),
        )
        selected.extend(ordered[:per_class])
    return selected


def load_completed(
    path: Path, key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = tuple(record.get(field) for field in key_fields)
            if key in completed:
                raise ValueError(f"duplicate row at {path}: {key}")
            completed[key] = record
    return completed


def append_record(
    path: Path,
    completed: dict[tuple[Any, ...], dict[str, Any]],
    key_fields: tuple[str, ...],
    record: dict[str, Any],
) -> None:
    key = tuple(record.get(field) for field in key_fields)
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = record


def condition_metadata(
    example: dict[str, Any],
    trigger: str,
    selection_commit: str,
    component_hash: str,
    ids_hash: str,
    token_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day08-v1",
        "freeze_id": "day04-v1",
        "selection_commit": selection_commit,
        "component_set_sha256": component_hash,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": example["split"],
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": example["label"],
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
    examples: list[dict[str, Any]],
    plan: dict[str, Any],
    selected_ids: list[str],
    output: Path,
) -> None:
    examples = examples[:2]
    concept = examples[0]["concept"]
    pair = runner.prepare_pairs(
        [record["prompt"] for record in examples],
        [record["response"] for record in examples],
        plan["conditions"]["correct_triggers"][concept],
    )
    probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
    candidates = [CANDIDATE_BY_ID[selected_ids[0]], CANDIDATE_BY_ID[selected_ids[-1]]]
    sites = tuple(candidate.site for candidate in candidates)
    normal = runner.run(
        pair.normal,
        capture_sites=sites,
        retain_response_logprobs=True,
    )
    multi = MultiCandidatePatchRunner(runner, probe).run_full(
        pair.triggered,
        tuple((candidate, normal.captures[candidate.site]) for candidate in candidates),
    )
    checks = []
    for index, candidate in enumerate(candidates):
        single = runner.run(
            pair.triggered,
            patch_cache={candidate.site: normal.captures[candidate.site]},
            retain_response_logprobs=True,
        )
        expected = masked_example_mean(
            -single.response_token_logprobs(), single.response_mask
        )
        difference = (multi.response_nll[index] - expected).abs()
        checks.append(
            {
                "candidate_id": candidate.candidate_id,
                "exact": bool(torch.equal(multi.response_nll[index], expected)),
                "max_abs_nll_difference": float(difference.max()),
            }
        )
    status = "pass" if all(row["exact"] for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "status": status,
        "examples": [record["example_id"] for record in examples],
        "vectorized_full_forward_check_count": len(checks),
        "vectorized_full_forward_checks": checks,
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 8 confirmation preflight failed")
    print(f"Confirmation preflight passed: {len(checks)} exact NLL checks.", flush=True)


def finalize(
    path: Path,
    completed: dict[tuple[Any, ...], dict[str, Any]],
    sort_fields: tuple[str, ...],
) -> None:
    rows = sorted(
        completed.values(),
        key=lambda record: tuple(
            "" if record.get(field) is None else record.get(field)
            for field in sort_fields
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
    if args.behavior_examples_per_class != 2:
        raise ValueError("day08-v1 freezes two behavior examples per class and concept")
    selection_path = args.selection.resolve()
    selection = load_selection(selection_path)
    selection_commit = git_head()
    require_committed_selection(selection_path, selection_commit)
    records = load_benign_records()
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    selected_ids = selection["selected_candidates"]
    random_ids = selection["random_control_candidates"]
    all_ids = selected_ids + random_ids
    all_candidates = [CANDIDATE_BY_ID[candidate_id] for candidate_id in all_ids]
    selected_candidates = [
        CANDIDATE_BY_ID[candidate_id] for candidate_id in selected_ids
    ]

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

    discovery_preflight = sorted(
        (
            record
            for record in records
            if record["split"] == "discovery"
            and record["concept"] == "HTML"
            and record["label"] == 1
        ),
        key=lambda record: record["example_id"],
    )
    run_preflight(
        runner,
        discovery_preflight,
        plan,
        selected_ids,
        args.preflight_output.resolve(),
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exact_key_fields = ("example_id", "record_type", "candidate_id", "direction")
    exact_completed = load_completed(output_path, exact_key_fields)
    positive_groups = sort_groups(
        records, tokenizer, plan, positives_only=True
    )
    print(f"Resuming with {len(exact_completed)} confirmation rows", flush=True)
    for group_index, group_key in enumerate(sorted(positive_groups), start=1):
        split, concept, _label = group_key
        values = positive_groups[group_key]
        trigger = plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = MultiCandidatePatchRunner(runner, probe)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        group_start = len(exact_completed)
        directions = ("induction",) if split == "discovery" else ("rescue", "induction")
        for batch_number, examples in enumerate(
            batched(values, args.batch_size), start=1
        ):
            pair = runner.prepare_pairs(
                [record["prompt"] for record in examples],
                [record["response"] for record in examples],
                trigger,
            )
            sites = tuple(candidate.site for candidate in all_candidates)
            normal = truncated.run(pair.normal, capture_sites=sites)
            triggered = truncated.run(pair.triggered, capture_sites=sites)
            for row, example in enumerate(examples):
                mask = pair.normal.response_mask[row]
                metadata = condition_metadata(
                    example,
                    trigger,
                    selection_commit,
                    selection["component_set_sha256"],
                    response_hash(pair.normal.response_ids[row], mask),
                    int(mask.sum()),
                )
                append_record(
                    output_path,
                    exact_completed,
                    exact_key_fields,
                    {
                        **metadata,
                        "record_type": "baseline",
                        "candidate_id": None,
                        "direction": None,
                        "normal_probe_score": float(normal.probe_scores[row]),
                        "triggered_probe_score": float(triggered.probe_scores[row]),
                        "execution_mode": "truncated_forward_at_resid_post_12",
                    },
                )
            for direction in directions:
                destination = pair.triggered if direction == "rescue" else pair.normal
                source = normal.captures if direction == "rescue" else triggered.captures
                for chunk in batched(all_candidates, args.candidate_chunk_size):
                    if all(
                        (
                            example["example_id"],
                            "patch",
                            candidate.candidate_id,
                            direction,
                        )
                        in exact_completed
                        for example in examples
                        for candidate in chunk
                    ):
                        continue
                    result = vector.run_truncated(
                        destination,
                        tuple(
                            (candidate, source[candidate.site])
                            for candidate in chunk
                        ),
                    )
                    for candidate_index, candidate in enumerate(chunk):
                        role = (
                            "selected"
                            if candidate.candidate_id in selected_ids
                            else "random_control"
                        )
                        for row, example in enumerate(examples):
                            mask = pair.normal.response_mask[row]
                            metadata = condition_metadata(
                                example,
                                trigger,
                                selection_commit,
                                selection["component_set_sha256"],
                                response_hash(pair.normal.response_ids[row], mask),
                                int(mask.sum()),
                            )
                            append_record(
                                output_path,
                                exact_completed,
                                exact_key_fields,
                                {
                                    **metadata,
                                    "record_type": "patch",
                                    "candidate_id": candidate.candidate_id,
                                    "candidate_role": role,
                                    "layer": candidate.layer,
                                    "component_type": candidate.component_type,
                                    "head": candidate.head,
                                    "direction": direction,
                                    "source_condition": (
                                        "normal"
                                        if direction == "rescue"
                                        else "correct_trigger"
                                    ),
                                    "destination_condition": (
                                        "correct_trigger"
                                        if direction == "rescue"
                                        else "normal"
                                    ),
                                    "patched_probe_score": float(
                                        result.probe_scores[candidate_index, row]
                                    ),
                                    "execution_mode": "truncated_forward_at_resid_post_12",
                                },
                            )
            release_memory()
            print(
                f"{split}/{concept}: batch {batch_number}/{total_batches}, "
                f"new rows {len(exact_completed) - group_start}",
                flush=True,
            )
        print(
            f"group {group_index}/{len(positive_groups)} {split}/{concept}: "
            f"complete ({len(exact_completed) - group_start} new rows)",
            flush=True,
        )

    if len(exact_completed) != EXPECTED_EXACT_ROWS:
        raise RuntimeError(
            f"found {len(exact_completed)} exact rows; expected {EXPECTED_EXACT_ROWS}"
        )
    finalize(
        output_path,
        exact_completed,
        ("split", "concept", "example_id", "record_type", "candidate_id", "direction"),
    )

    behavior_path = args.behavior_output.resolve()
    behavior_key_fields = ("example_id", "record_type", "candidate_id", "direction")
    behavior_completed = load_completed(behavior_path, behavior_key_fields)
    subset = behavior_subset(records, args.behavior_examples_per_class)
    behavior_groups = sort_groups(
        subset, tokenizer, plan, positives_only=False
    )
    print(
        f"Resuming with {len(behavior_completed)} behavior rows "
        f"on {len(subset)} frozen examples",
        flush=True,
    )
    for group_index, group_key in enumerate(sorted(behavior_groups), start=1):
        split, concept, label = group_key
        values = behavior_groups[group_key]
        trigger = plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = MultiCandidatePatchRunner(runner, probe)
        pair = runner.prepare_pairs(
            [record["prompt"] for record in values],
            [record["response"] for record in values],
            trigger,
        )
        sites = tuple(candidate.site for candidate in selected_candidates)
        normal = runner.run(
            pair.normal,
            capture_sites=sites,
            retain_response_logprobs=True,
        )
        triggered = runner.run(
            pair.triggered,
            capture_sites=sites,
            retain_response_logprobs=True,
        )
        normal_nll = masked_example_mean(
            -normal.response_token_logprobs(), normal.response_mask
        )
        triggered_nll = masked_example_mean(
            -triggered.response_token_logprobs(), triggered.response_mask
        )
        for row, example in enumerate(values):
            mask = pair.normal.response_mask[row]
            metadata = condition_metadata(
                example,
                trigger,
                selection_commit,
                selection["component_set_sha256"],
                response_hash(pair.normal.response_ids[row], mask),
                int(mask.sum()),
            )
            append_record(
                behavior_path,
                behavior_completed,
                behavior_key_fields,
                {
                    **metadata,
                    "record_type": "baseline",
                    "candidate_id": None,
                    "direction": None,
                    "normal_response_nll": float(normal_nll[row]),
                    "triggered_response_nll": float(triggered_nll[row]),
                    "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                },
            )
        for direction, destination, source in (
            ("rescue", pair.triggered, normal.captures),
            ("induction", pair.normal, triggered.captures),
        ):
            for chunk in batched(selected_candidates, 2):
                result = vector.run_full(
                    destination,
                    tuple(
                        (candidate, source[candidate.site])
                        for candidate in chunk
                    ),
                )
                for candidate_index, candidate in enumerate(chunk):
                    for row, example in enumerate(values):
                        mask = pair.normal.response_mask[row]
                        metadata = condition_metadata(
                            example,
                            trigger,
                            selection_commit,
                            selection["component_set_sha256"],
                            response_hash(pair.normal.response_ids[row], mask),
                            int(mask.sum()),
                        )
                        append_record(
                            behavior_path,
                            behavior_completed,
                            behavior_key_fields,
                            {
                                **metadata,
                                "record_type": "patch",
                                "candidate_id": candidate.candidate_id,
                                "candidate_role": "selected",
                                "layer": candidate.layer,
                                "component_type": candidate.component_type,
                                "head": candidate.head,
                                "direction": direction,
                                "destination_condition": (
                                    "correct_trigger"
                                    if direction == "rescue"
                                    else "normal"
                                ),
                                "patched_response_nll": float(
                                    result.response_nll[candidate_index, row]
                                ),
                                "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                            },
                        )
        release_memory()
        print(
            f"behavior group {group_index}/{len(behavior_groups)} "
            f"{split}/{concept}/label_{label}: complete",
            flush=True,
        )

    if len(subset) != 44 or len(behavior_completed) != EXPECTED_BEHAVIOR_ROWS:
        raise RuntimeError(
            f"behavior grid mismatch: examples={len(subset)}, "
            f"rows={len(behavior_completed)}"
        )
    finalize(
        behavior_path,
        behavior_completed,
        ("split", "concept", "label", "example_id", "record_type", "candidate_id", "direction"),
    )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remained registered after confirmation")
    print(
        f"Day 8 confirmation: {len(exact_completed)} exact rows and "
        f"{len(behavior_completed)} behavior rows."
    )


if __name__ == "__main__":
    main()
