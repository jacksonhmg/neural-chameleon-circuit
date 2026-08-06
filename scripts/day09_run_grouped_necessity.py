#!/usr/bin/env python3
"""Run the frozen Day 9 grouped rescue and bounded behavior grids."""

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
    GroupPatchJob,
    GroupedComponentPatchRunner,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TruncatedComponentRunner,
    group_specifications,
    load_experimental_split,
    masked_example_mean,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-09"
PLAN_PATH = RESULT_DIR / "frozen-group-plan.json"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
OUTPUT_PATH = RESULT_DIR / "grouped-example-results.jsonl"
BEHAVIOR_PATH = RESULT_DIR / "grouped-behavior-results.jsonl"
PREFLIGHT_PATH = RESULT_DIR / "grouped-preflight.json"
MONITOR_SITE = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
EXPECTED_SCORE_ROWS = 1408 * 14
EXPECTED_BEHAVIOR_ROWS = 44 * 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=2)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def latest_file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
    ).strip()


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from its frozen commit {commit}")


def load_frozen_inputs(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    plan = json.loads(plan_path.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    if plan.get("status") != "frozen-before-grouped-results":
        raise ValueError("Day 9 group plan is not frozen")
    if plan.get("procedure") != "day09-v1" or plan.get("freeze_id") != "day04-v1":
        raise ValueError("unexpected Day 9 procedure or data freeze")
    if plan["source_component_set_sha256"] != selection["component_set_sha256"]:
        raise ValueError("Day 9 plan does not name the frozen Day 8 component set")
    for size in (1, 2, 4, 8, 16):
        if plan["selected_prefixes"][str(size)] != selection["selected_candidates"][:size]:
            raise ValueError(f"selected K={size} prefix differs from Day 8")
        if plan["random_prefixes"][str(size)] != selection["random_control_candidates"][:size]:
            raise ValueError(f"random K={size} prefix differs from Day 8")
    selected = set(selection["selected_candidates"])
    expected_control = sorted(
        (
            hashlib.sha256(f"42:{candidate_id}".encode()).hexdigest(),
            candidate_id,
        )
        for candidate_id in CANDIDATE_BY_ID
        if candidate_id not in selected and CANDIDATE_BY_ID[candidate_id].layer != 11
    )[:17]
    if plan["block_control"]["candidate_ids"] != [item for _digest, item in expected_control]:
        raise ValueError("Day 9 outside-layer control differs from its frozen hash rule")
    selection_commit = latest_file_commit(SELECTION_PATH)
    procedure_commit = latest_file_commit(plan_path)
    require_committed_file(SELECTION_PATH, selection_commit)
    require_committed_file(plan_path, procedure_commit)
    return plan, selection, selection_commit, procedure_commit


def load_benign_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 9 requires the safety split to remain locked")
    records = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    if len(records) != 1408:
        raise ValueError(f"expected 1,408 frozen benign records, found {len(records)}")
    return records


def sort_groups(
    records: list[dict[str, Any]], tokenizer, analysis_plan: dict[str, Any]
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split"], record["concept"], record["label"])].append(record)
    ordered = {}
    for key, values in groups.items():
        _split, concept, _label = key
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
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
        ordered[key] = sorted(
            values, key=lambda record: (record["_sort_length"], record["example_id"])
        )
    return ordered


def behavior_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        selected.extend(ordered[:2])
    if len(selected) != 44:
        raise ValueError("frozen Day 9 behavior subset must contain 44 examples")
    return selected


def load_completed(path: Path) -> dict[tuple[str, str | None], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = (row["example_id"], row.get("group_id"))
            if key in completed:
                raise ValueError(f"duplicate Day 9 row key: {key}")
            completed[key] = row
    return completed


def append_record(
    path: Path,
    completed: dict[tuple[str, str | None], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    key = (record["example_id"], record.get("group_id"))
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = record


def finalize(
    path: Path, completed: dict[tuple[str, str | None], dict[str, Any]]
) -> None:
    rows = sorted(
        completed.values(),
        key=lambda row: (
            row["split"],
            row["concept"],
            row["label"],
            row["example_id"],
            "" if row.get("group_id") is None else row["group_id"],
        ),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def group_hash(candidate_ids: list[str]) -> str:
    payload = json.dumps(candidate_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def metadata(
    example: dict[str, Any],
    trigger: str,
    response_ids_sha256: str,
    response_token_count: int,
    *,
    plan_sha256: str,
    selection_commit: str,
    procedure_commit: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day09-v1",
        "freeze_id": "day04-v1",
        "selection_commit": selection_commit,
        "procedure_commit": procedure_commit,
        "group_plan_sha256": plan_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": example["split"],
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": example["label"],
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": response_ids_sha256,
        "response_token_count": response_token_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }


def component_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        specification
        for specification in group_specifications(plan)
        if specification["group_role"] != "positive_control"
    ]


def make_job(
    specification: dict[str, Any], captures
) -> GroupPatchJob:
    return GroupPatchJob(
        specification["group_id"],
        tuple(
            (CANDIDATE_BY_ID[candidate_id], captures[CANDIDATE_BY_ID[candidate_id].site])
            for candidate_id in specification["candidate_ids"]
        ),
    )


def run_preflight(
    runner: PairedInterventionRunner,
    examples: list[dict[str, Any]],
    analysis_plan: dict[str, Any],
    plan: dict[str, Any],
    output: Path,
) -> None:
    examples = examples[:2]
    concept = examples[0]["concept"]
    pair = runner.prepare_pairs(
        [record["prompt"] for record in examples],
        [record["response"] for record in examples],
        analysis_plan["conditions"]["correct_triggers"][concept],
    )
    probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    grouped = GroupedComponentPatchRunner(runner, probe, monitor_layer=12)
    specifications = component_specs(plan)
    component_sites = tuple(
        dict.fromkeys(
            CANDIDATE_BY_ID[candidate_id].site
            for specification in specifications
            for candidate_id in specification["candidate_ids"]
        )
    )
    capture_sites = (*component_sites, MONITOR_SITE)
    normal = truncated.run(pair.normal, capture_sites=capture_sites)
    triggered = truncated.run(pair.triggered, capture_sites=capture_sites)

    identity_checks = []
    for condition_name, condition, baseline in (
        ("normal", pair.normal, normal),
        ("correct_trigger", pair.triggered, triggered),
    ):
        for specification in specifications:
            result = grouped.run_truncated(
                condition, [make_job(specification, baseline.captures)]
            )
            exact = torch.equal(result.probe_scores[0], baseline.probe_scores)
            identity_checks.append(
                {
                    "condition": condition_name,
                    "group_id": specification["group_id"],
                    "exact": exact,
                    "max_abs_score_difference": float(
                        (result.probe_scores[0] - baseline.probe_scores).abs().max()
                    ),
                }
            )
        block_identity = truncated.run(
            condition,
            patch_cache={MONITOR_SITE: baseline.captures[MONITOR_SITE]},
        )
        identity_checks.append(
            {
                "condition": condition_name,
                "group_id": plan["positive_control"]["group_id"],
                "exact": torch.equal(block_identity.probe_scores, baseline.probe_scores),
                "max_abs_score_difference": float(
                    (block_identity.probe_scores - baseline.probe_scores).abs().max()
                ),
            }
        )

    comparison_specs = [
        next(row for row in specifications if row["group_id"] == group_id)
        for group_id in ("selected_k2", "all_layer11_components")
    ]
    vector = grouped.run_truncated(
        pair.triggered,
        [make_job(specification, normal.captures) for specification in comparison_specs],
    )
    vector_score_checks = []
    for index, specification in enumerate(comparison_specs):
        independent = grouped.run_truncated(
            pair.triggered, [make_job(specification, normal.captures)]
        )
        difference = (
            vector.probe_scores[index] - independent.probe_scores[0]
        ).abs().max()
        vector_score_checks.append(
            {
                "group_id": specification["group_id"],
                "max_abs_score_difference": float(difference),
                "tolerance": 0.002,
                "within_tolerance": bool(difference <= 0.002),
            }
        )

    vector_full = grouped.run_full(
        pair.triggered,
        [make_job(specification, normal.captures) for specification in comparison_specs],
    )
    vector_nll_checks = []
    for index, specification in enumerate(comparison_specs):
        independent = grouped.run_full(
            pair.triggered, [make_job(specification, normal.captures)]
        )
        difference = (
            vector_full.response_nll[index] - independent.response_nll[0]
        ).abs().max()
        vector_nll_checks.append(
            {
                "group_id": specification["group_id"],
                "max_abs_nll_difference": float(difference),
                "tolerance": 0.02,
                "within_tolerance": bool(difference <= 0.02),
            }
        )

    order_spec = next(row for row in specifications if row["group_id"] == "selected_k16")
    forward_job = make_job(order_spec, normal.captures)
    reverse_job = GroupPatchJob(forward_job.group_id, tuple(reversed(forward_job.members)))
    forward_score = grouped.run_truncated(pair.triggered, [forward_job]).probe_scores
    reverse_score = grouped.run_truncated(pair.triggered, [reverse_job]).probe_scores
    order_check = {
        "group_id": order_spec["group_id"],
        "exact": torch.equal(forward_score, reverse_score),
        "max_abs_score_difference": float((forward_score - reverse_score).abs().max()),
    }

    hook_count = runner.registered_hook_count()
    status = "pass" if (
        len(identity_checks) == 26
        and all(row["exact"] for row in identity_checks)
        and all(row["within_tolerance"] for row in vector_score_checks)
        and all(row["within_tolerance"] for row in vector_nll_checks)
        and order_check["exact"]
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day09-v1",
        "status": status,
        "examples": [record["example_id"] for record in examples],
        "same_shape_identity_check_count": len(identity_checks),
        "same_shape_identity_checks": identity_checks,
        "vectorized_probe_score_checks": vector_score_checks,
        "vectorized_response_nll_checks": vector_nll_checks,
        "group_member_order_check": order_check,
        "registered_hook_count_after_checks": hook_count,
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 9 real-checkpoint preflight failed")
    print("Day 9 preflight passed: 26 identities plus vector and order checks.", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or not 1 <= args.group_chunk_size <= 2:
        raise ValueError("batch size must be positive and group chunk size must be one or two")
    plan_path = args.plan.resolve()
    plan, selection, selection_commit, procedure_commit = load_frozen_inputs(plan_path)
    plan_sha256 = sha256_file(plan_path)
    records = load_benign_records()
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    specifications = component_specs(plan)
    all_group_specs = group_specifications(plan)
    positive_spec = next(
        row for row in all_group_specs if row["group_role"] == "positive_control"
    )
    component_sites = tuple(
        dict.fromkeys(
            CANDIDATE_BY_ID[candidate_id].site
            for specification in specifications
            for candidate_id in specification["candidate_ids"]
        )
    )
    capture_sites = (*component_sites, MONITOR_SITE)

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

    preflight_examples = sorted(
        (
            row
            for row in records
            if row["split"] == "discovery"
            and row["concept"] == "HTML"
            and row["label"] == 1
        ),
        key=lambda row: row["example_id"],
    )
    run_preflight(
        runner,
        preflight_examples,
        analysis_plan,
        plan,
        args.preflight_output.resolve(),
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    groups = sort_groups(records, tokenizer, analysis_plan)
    print(f"Resuming with {len(completed)} grouped score rows", flush=True)
    for group_index, group_key in enumerate(sorted(groups), start=1):
        split, concept, label = group_key
        values = groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        grouped = GroupedComponentPatchRunner(runner, probe, monitor_layer=12)
        group_start = len(completed)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        for batch_number, examples in enumerate(batched(values, args.batch_size), start=1):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal = truncated.run(pair.normal, capture_sites=capture_sites)
            triggered = truncated.run(pair.triggered)
            for row_index, example in enumerate(examples):
                mask = pair.normal.response_mask[row_index]
                common = metadata(
                    example,
                    trigger,
                    response_hash(pair.normal.response_ids[row_index], mask),
                    int(mask.sum()),
                    plan_sha256=plan_sha256,
                    selection_commit=selection_commit,
                    procedure_commit=procedure_commit,
                )
                append_record(
                    output_path,
                    completed,
                    {
                        **common,
                        "record_type": "baseline",
                        "group_id": None,
                        "normal_probe_score": float(normal.probe_scores[row_index]),
                        "triggered_probe_score": float(triggered.probe_scores[row_index]),
                        "execution_mode": "truncated_forward_at_resid_post_12",
                    },
                )
            for chunk in batched(specifications, args.group_chunk_size):
                if all(
                    (example["example_id"], specification["group_id"]) in completed
                    for example in examples
                    for specification in chunk
                ):
                    continue
                jobs = [make_job(specification, normal.captures) for specification in chunk]
                result = grouped.run_truncated(pair.triggered, jobs)
                for job_index, specification in enumerate(chunk):
                    for row_index, example in enumerate(examples):
                        mask = pair.normal.response_mask[row_index]
                        common = metadata(
                            example,
                            trigger,
                            response_hash(pair.normal.response_ids[row_index], mask),
                            int(mask.sum()),
                            plan_sha256=plan_sha256,
                            selection_commit=selection_commit,
                            procedure_commit=procedure_commit,
                        )
                        append_record(
                            output_path,
                            completed,
                            {
                                **common,
                                "record_type": "group_patch",
                                **specification,
                                "candidate_ids_sha256": group_hash(specification["candidate_ids"]),
                                "direction": "rescue",
                                "source_condition": "normal",
                                "destination_condition": "correct_trigger",
                                "patched_probe_score": float(
                                    result.probe_scores[job_index, row_index]
                                ),
                                "execution_mode": "truncated_forward_at_resid_post_12",
                            },
                        )
            block = truncated.run(
                pair.triggered,
                patch_cache={MONITOR_SITE: normal.captures[MONITOR_SITE]},
            )
            for row_index, example in enumerate(examples):
                mask = pair.normal.response_mask[row_index]
                common = metadata(
                    example,
                    trigger,
                    response_hash(pair.normal.response_ids[row_index], mask),
                    int(mask.sum()),
                    plan_sha256=plan_sha256,
                    selection_commit=selection_commit,
                    procedure_commit=procedure_commit,
                )
                append_record(
                    output_path,
                    completed,
                    {
                        **common,
                        "record_type": "group_patch",
                        **positive_spec,
                        "candidate_ids_sha256": group_hash([]),
                        "patch_site": "resid_post_layer_12",
                        "direction": "rescue",
                        "source_condition": "normal",
                        "destination_condition": "correct_trigger",
                        "patched_probe_score": float(block.probe_scores[row_index]),
                        "execution_mode": "truncated_forward_at_resid_post_12",
                    },
                )
            release_memory()
            print(
                f"{split}/{concept}/label_{label}: batch {batch_number}/{total_batches}, "
                f"new rows {len(completed) - group_start}",
                flush=True,
            )
        print(
            f"group {group_index}/{len(groups)} complete: {split}/{concept}/label_{label}",
            flush=True,
        )
    if len(completed) != EXPECTED_SCORE_ROWS:
        raise RuntimeError(f"found {len(completed)} score rows; expected {EXPECTED_SCORE_ROWS}")
    finalize(output_path, completed)

    behavior_path = args.behavior_output.resolve()
    behavior_completed = load_completed(behavior_path)
    subset = behavior_subset(records)
    behavior_groups = sort_groups(subset, tokenizer, analysis_plan)
    print(f"Resuming with {len(behavior_completed)} grouped behavior rows", flush=True)
    for group_index, group_key in enumerate(sorted(behavior_groups), start=1):
        split, concept, label = group_key
        examples = behavior_groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        grouped = GroupedComponentPatchRunner(runner, probe, monitor_layer=12)
        pair = runner.prepare_pairs(
            [row["prompt"] for row in examples],
            [row["response"] for row in examples],
            trigger,
        )
        normal = runner.run(
            pair.normal, capture_sites=capture_sites, retain_response_logprobs=True
        )
        triggered = runner.run(pair.triggered, retain_response_logprobs=True)
        normal_nll = masked_example_mean(
            -normal.response_token_logprobs(), normal.response_mask
        )
        triggered_nll = masked_example_mean(
            -triggered.response_token_logprobs(), triggered.response_mask
        )
        for row_index, example in enumerate(examples):
            mask = pair.normal.response_mask[row_index]
            common = metadata(
                example,
                trigger,
                response_hash(pair.normal.response_ids[row_index], mask),
                int(mask.sum()),
                plan_sha256=plan_sha256,
                selection_commit=selection_commit,
                procedure_commit=procedure_commit,
            )
            append_record(
                behavior_path,
                behavior_completed,
                {
                    **common,
                    "record_type": "baseline",
                    "group_id": None,
                    "normal_response_nll": float(normal_nll[row_index]),
                    "triggered_response_nll": float(triggered_nll[row_index]),
                    "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                },
            )
        for chunk in batched(specifications, args.group_chunk_size):
            jobs = [make_job(specification, normal.captures) for specification in chunk]
            result = grouped.run_full(pair.triggered, jobs)
            for job_index, specification in enumerate(chunk):
                for row_index, example in enumerate(examples):
                    mask = pair.normal.response_mask[row_index]
                    common = metadata(
                        example,
                        trigger,
                        response_hash(pair.normal.response_ids[row_index], mask),
                        int(mask.sum()),
                        plan_sha256=plan_sha256,
                        selection_commit=selection_commit,
                        procedure_commit=procedure_commit,
                    )
                    append_record(
                        behavior_path,
                        behavior_completed,
                        {
                            **common,
                            "record_type": "group_patch",
                            **specification,
                            "candidate_ids_sha256": group_hash(specification["candidate_ids"]),
                            "direction": "rescue",
                            "source_condition": "normal",
                            "destination_condition": "correct_trigger",
                            "patched_response_nll": float(
                                result.response_nll[job_index, row_index]
                            ),
                            "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                        },
                    )
        block = runner.run(
            pair.triggered,
            patch_cache={MONITOR_SITE: normal.captures[MONITOR_SITE]},
            retain_response_logprobs=True,
        )
        block_nll = masked_example_mean(
            -block.response_token_logprobs(), block.response_mask
        )
        for row_index, example in enumerate(examples):
            mask = pair.normal.response_mask[row_index]
            common = metadata(
                example,
                trigger,
                response_hash(pair.normal.response_ids[row_index], mask),
                int(mask.sum()),
                plan_sha256=plan_sha256,
                selection_commit=selection_commit,
                procedure_commit=procedure_commit,
            )
            append_record(
                behavior_path,
                behavior_completed,
                {
                    **common,
                    "record_type": "group_patch",
                    **positive_spec,
                    "candidate_ids_sha256": group_hash([]),
                    "patch_site": "resid_post_layer_12",
                    "direction": "rescue",
                    "source_condition": "normal",
                    "destination_condition": "correct_trigger",
                    "patched_response_nll": float(block_nll[row_index]),
                    "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                },
            )
        release_memory()
        print(
            f"behavior {group_index}/{len(behavior_groups)} complete: "
            f"{split}/{concept}/label_{label}",
            flush=True,
        )
    if len(behavior_completed) != EXPECTED_BEHAVIOR_ROWS:
        raise RuntimeError(
            f"found {len(behavior_completed)} behavior rows; expected {EXPECTED_BEHAVIOR_ROWS}"
        )
    finalize(behavior_path, behavior_completed)
    if runner.registered_hook_count() != 0:
        raise RuntimeError("model hooks leaked after Day 9 execution")
    print(
        f"Day 9 model execution complete: {len(completed)} score rows and "
        f"{len(behavior_completed)} behavior rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
