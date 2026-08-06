#!/usr/bin/env python3
"""Run the frozen Day 10 natural-transplant and dose-response grids."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    TruncatedComponentRunner,
    VectorizedTransplantRunner,
    group_activation_norms,
    interpolate_capture,
    load_experimental_split,
    masked_example_mean,
    sufficiency_specifications,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-10"
PLAN_PATH = RESULT_DIR / "frozen-sufficiency-plan.json"
DAY09_PLAN_PATH = ROOT / "results/day-09/frozen-group-plan.json"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
EXACT_PATH = RESULT_DIR / "sufficiency-example-results.jsonl"
DOSE_PATH = RESULT_DIR / "dose-response-results.jsonl"
BEHAVIOR_PATH = RESULT_DIR / "sufficiency-behavior-results.jsonl"
PREFLIGHT_PATH = RESULT_DIR / "sufficiency-preflight.json"
EXPECTED_EXACT_ROWS = 1408 * 14
EXPECTED_DOSE_ROWS = 176 * 6 * 3
EXPECTED_BEHAVIOR_ROWS = 44 * 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--job-chunk-size", type=int, default=2)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--exact-output", type=Path, default=EXACT_PATH)
    parser.add_argument("--dose-output", type=Path, default=DOSE_PATH)
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


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from frozen commit {commit}")


def load_frozen_inputs(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    plan = json.loads(plan_path.read_text())
    day09_plan = json.loads(DAY09_PLAN_PATH.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    if plan.get("status") != "frozen-before-sufficiency-results":
        raise ValueError("Day 10 sufficiency plan is not frozen")
    if plan.get("procedure") != "day10-v1" or plan.get("freeze_id") != "day04-v1":
        raise ValueError("unexpected Day 10 procedure or split freeze")
    if plan["source_component_set_sha256"] != selection["component_set_sha256"]:
        raise ValueError("Day 10 plan does not name the Day 8 component set")
    if plan["source_day09_plan_sha256"] != sha256_file(DAY09_PLAN_PATH):
        raise ValueError("Day 10 plan does not name the exact Day 9 plan")
    specifications = sufficiency_specifications(plan)
    selected_k16 = next(row for row in specifications if row["group_id"] == "selected_k16")
    random_k16 = next(row for row in specifications if row["group_id"] == "random_k16")
    if selected_k16["candidate_ids"] != selection["selected_candidates"]:
        raise ValueError("Day 10 selected K16 differs from Day 8")
    if random_k16["candidate_ids"] != selection["random_control_candidates"]:
        raise ValueError("Day 10 random K16 differs from Day 8")
    for rank in range(1, 5):
        selected = next(row for row in specifications if row["group_id"] == f"selected_single_rank{rank}")
        random = next(row for row in specifications if row["group_id"] == f"random_single_rank{rank}")
        if selected["candidate_ids"] != [selection["selected_candidates"][rank - 1]]:
            raise ValueError("strong individual order differs from Day 8")
        if random["candidate_ids"] != [selection["random_control_candidates"][rank - 1]]:
            raise ValueError("random individual order differs from Day 8")
    commits = {
        "selection_commit": latest_file_commit(SELECTION_PATH),
        "day09_procedure_commit": latest_file_commit(DAY09_PLAN_PATH),
        "day10_procedure_commit": latest_file_commit(plan_path),
        "implementation_commit": git_head(),
    }
    require_committed_file(SELECTION_PATH, commits["selection_commit"])
    require_committed_file(DAY09_PLAN_PATH, commits["day09_procedure_commit"])
    require_committed_file(plan_path, commits["day10_procedure_commit"])
    return plan, selection, commits


def load_benign_records() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 10 requires the safety split to remain locked")
    records = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    if len(records) != 1408:
        raise ValueError(f"expected 1,408 benign records, found {len(records)}")
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
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[(record["split"], record["concept"], record["label"])].append(record)
    selected = []
    for key in sorted(cells):
        ordered = sorted(
            cells[key],
            key=lambda record: (
                hashlib.sha256(f"day08-behavior:{record['example_id']}".encode()).hexdigest(),
                record["example_id"],
            ),
        )
        selected.extend(ordered[:2])
    if len(selected) != 44:
        raise ValueError("Day 10 behavior subset must contain 44 examples")
    return selected


def dose_subset(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    concepts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["label"] == 1:
            concepts[(record["split"], record["concept"])].append(record)
    selected = []
    for key in sorted(concepts):
        ordered = sorted(
            concepts[key],
            key=lambda record: (
                hashlib.sha256(f"day10-dose:{record['example_id']}".encode()).hexdigest(),
                record["example_id"],
            ),
        )
        selected.extend(ordered[:16])
    if len(selected) != 176:
        raise ValueError("Day 10 dose subset must contain 176 positive examples")
    return selected


def residual_site(label: str) -> PatchSite:
    mapping = {
        "resid_post_layer_08": PatchSite(ActivationKind.BLOCK_OUTPUT, 8),
        "resid_post_layer_12": PatchSite(ActivationKind.BLOCK_OUTPUT, 12),
    }
    try:
        return mapping[label]
    except KeyError as error:
        raise ValueError(f"unsupported frozen residual site: {label}") from error


def specification_sites(specification: Mapping[str, Any]) -> tuple[PatchSite, ...]:
    sites = [CANDIDATE_BY_ID[item].site for item in specification["candidate_ids"]]
    sites.extend(residual_site(item) for item in specification["residual_sites"])
    if not sites or len(sites) != len(set(sites)):
        raise ValueError(f"invalid transplant site set for {specification['group_id']}")
    return tuple(sites)


def make_job(
    specification: Mapping[str, Any], captures: Mapping[PatchSite, Any]
) -> TransplantJob:
    return TransplantJob(
        specification["group_id"],
        tuple(
            TransplantMember(site, captures[site])
            for site in specification_sites(specification)
        ),
    )


def group_hash(specification: Mapping[str, Any]) -> str:
    payload = {
        "candidate_ids": specification["candidate_ids"],
        "residual_sites": specification["residual_sites"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_completed(
    path: Path, key_fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = tuple(row.get(field) for field in key_fields)
            if key in completed:
                raise ValueError(f"duplicate Day 10 row key: {key}")
            completed[key] = row
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


def finalize(
    path: Path,
    completed: dict[tuple[Any, ...], dict[str, Any]],
    sort_fields: tuple[str, ...],
) -> None:
    rows = sorted(
        completed.values(),
        key=lambda row: tuple("" if row.get(field) is None else row.get(field) for field in sort_fields),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def metadata(
    example: Mapping[str, Any],
    trigger: str,
    ids_sha256: str,
    token_count: int,
    *,
    plan_sha256: str,
    commits: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day10-v1",
        "freeze_id": "day04-v1",
        **commits,
        "sufficiency_plan_sha256": plan_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": example["split"],
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": example["label"],
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": ids_sha256,
        "response_token_count": token_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }


def norm_fields(
    specification: Mapping[str, Any],
    normal_captures: Mapping[PatchSite, Any],
    triggered_captures: Mapping[PatchSite, Any],
    mixed_captures: Mapping[PatchSite, Any],
) -> dict[str, torch.Tensor]:
    sites = specification_sites(specification)
    return group_activation_norms(
        sites, normal_captures, triggered_captures, mixed_captures
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
        [row["prompt"] for row in examples],
        [row["response"] for row in examples],
        analysis_plan["conditions"]["correct_triggers"][concept],
    )
    probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
    specifications = sufficiency_specifications(plan)
    capture_sites = tuple(
        dict.fromkeys(
            site for specification in specifications for site in specification_sites(specification)
        )
    )
    normal = truncated.run(pair.normal, capture_sites=capture_sites)
    triggered = truncated.run(pair.triggered, capture_sites=capture_sites)

    identity_checks = []
    for condition_name, condition, baseline in (
        ("normal", pair.normal, normal),
        ("correct_trigger", pair.triggered, triggered),
    ):
        for specification in specifications:
            result = vector.run_truncated(
                condition, [make_job(specification, baseline.captures)]
            )
            difference = (result.probe_scores[0] - baseline.probe_scores).abs().max()
            identity_checks.append(
                {
                    "condition": condition_name,
                    "group_id": specification["group_id"],
                    "exact": torch.equal(result.probe_scores[0], baseline.probe_scores),
                    "max_abs_score_difference": float(difference),
                }
            )

    comparison_specs = [
        next(row for row in specifications if row["group_id"] == group_id)
        for group_id in ("selected_k16", "selected_k16_plus_resid_post_layer08")
    ]
    vector_scores = vector.run_truncated(
        pair.normal,
        [make_job(specification, triggered.captures) for specification in comparison_specs],
    )
    vector_score_checks = []
    for index, specification in enumerate(comparison_specs):
        independent = vector.run_truncated(
            pair.normal, [make_job(specification, triggered.captures)]
        )
        difference = (vector_scores.probe_scores[index] - independent.probe_scores[0]).abs().max()
        vector_score_checks.append(
            {
                "group_id": specification["group_id"],
                "max_abs_score_difference": float(difference),
                "tolerance": 0.002,
                "within_tolerance": bool(difference <= 0.002),
            }
        )

    vector_full = vector.run_full(
        pair.normal,
        [make_job(specification, triggered.captures) for specification in comparison_specs],
    )
    vector_nll_checks = []
    for index, specification in enumerate(comparison_specs):
        independent = vector.run_full(
            pair.normal, [make_job(specification, triggered.captures)]
        )
        difference = (vector_full.response_nll[index] - independent.response_nll[0]).abs().max()
        vector_nll_checks.append(
            {
                "group_id": specification["group_id"],
                "max_abs_nll_difference": float(difference),
                "tolerance": 0.02,
                "within_tolerance": bool(difference <= 0.02),
            }
        )

    endpoint_checks = []
    norm_checks = []
    for specification in comparison_specs:
        sites = specification_sites(specification)
        alpha_zero = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 0)
            for site in sites
        }
        alpha_one = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 1)
            for site in sites
        }
        alpha_half = {
            site: interpolate_capture(normal.captures[site], triggered.captures[site], 0.5)
            for site in sites
        }
        zero = vector.run_truncated(pair.normal, [make_job(specification, alpha_zero)])
        exact = vector.run_truncated(pair.normal, [make_job(specification, triggered.captures)])
        one = vector.run_truncated(pair.normal, [make_job(specification, alpha_one)])
        endpoint_checks.extend(
            [
                {
                    "group_id": specification["group_id"],
                    "alpha": 0.0,
                    "exact": torch.equal(zero.probe_scores[0], normal.probe_scores),
                    "max_abs_score_difference": float((zero.probe_scores[0] - normal.probe_scores).abs().max()),
                },
                {
                    "group_id": specification["group_id"],
                    "alpha": 1.0,
                    "exact": torch.equal(one.probe_scores[0], exact.probe_scores[0]),
                    "max_abs_score_difference": float((one.probe_scores[0] - exact.probe_scores[0]).abs().max()),
                },
            ]
        )
        norms = group_activation_norms(
            sites, normal.captures, triggered.captures, alpha_half
        )
        norm_checks.append(
            {
                "group_id": specification["group_id"],
                "alpha": 0.5,
                "maximum_bound_ratio": float(norms["bound_ratio_max"].max()),
                "bound": 1.001,
                "within_bound": bool(norms["bound_ratio_max"].max() <= 1.001),
            }
        )

    order_spec = comparison_specs[1]
    forward = make_job(order_spec, triggered.captures)
    reverse = TransplantJob(forward.group_id, tuple(reversed(forward.members)))
    forward_score = vector.run_truncated(pair.normal, [forward]).probe_scores
    reverse_score = vector.run_truncated(pair.normal, [reverse]).probe_scores
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
        and len(endpoint_checks) == 4
        and all(row["exact"] for row in endpoint_checks)
        and all(row["within_bound"] for row in norm_checks)
        and order_check["exact"]
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day10-v1",
        "status": status,
        "examples": [row["example_id"] for row in examples],
        "same_shape_identity_check_count": len(identity_checks),
        "same_shape_identity_checks": identity_checks,
        "vectorized_probe_score_checks": vector_score_checks,
        "vectorized_response_nll_checks": vector_nll_checks,
        "interpolation_endpoint_checks": endpoint_checks,
        "activation_norm_checks": norm_checks,
        "group_member_order_check": order_check,
        "registered_hook_count_after_checks": hook_count,
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 10 real-checkpoint preflight failed")
    print("Day 10 preflight passed: identities, endpoints, vectors, norms, and order.", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or not 1 <= args.job_chunk_size <= 2:
        raise ValueError("batch size must be positive and job chunk size must be one or two")
    plan_path = args.plan.resolve()
    plan, _selection, commits = load_frozen_inputs(plan_path)
    plan_sha256 = sha256_file(plan_path)
    records = load_benign_records()
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    specifications = sufficiency_specifications(plan)
    specification_by_id = {row["group_id"]: row for row in specifications}
    capture_sites = tuple(
        dict.fromkeys(
            site for specification in specifications for site in specification_sites(specification)
        )
    )

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

    preflight_examples = sorted(
        (
            row for row in records
            if row["split"] == "discovery" and row["concept"] == "HTML" and row["label"] == 1
        ),
        key=lambda row: row["example_id"],
    )
    run_preflight(
        runner, preflight_examples, analysis_plan, plan, args.preflight_output.resolve()
    )

    exact_path = args.exact_output.resolve()
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    exact_key_fields = ("example_id", "group_id")
    exact_completed = load_completed(exact_path, exact_key_fields)
    groups = sort_groups(records, tokenizer, analysis_plan)
    print(f"Resuming with {len(exact_completed)} exact sufficiency rows", flush=True)
    for group_index, group_key in enumerate(sorted(groups), start=1):
        split, concept, label = group_key
        values = groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        group_start = len(exact_completed)
        total_batches = (len(values) + args.batch_size - 1) // args.batch_size
        for batch_number, examples in enumerate(batched(values, args.batch_size), start=1):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal = truncated.run(pair.normal, capture_sites=capture_sites)
            triggered = truncated.run(pair.triggered, capture_sites=capture_sites)
            for row_index, example in enumerate(examples):
                mask = pair.normal.response_mask[row_index]
                common = metadata(
                    example,
                    trigger,
                    response_hash(pair.normal.response_ids[row_index], mask),
                    int(mask.sum()),
                    plan_sha256=plan_sha256,
                    commits=commits,
                )
                append_record(
                    exact_path,
                    exact_completed,
                    exact_key_fields,
                    {
                        **common,
                        "record_type": "baseline",
                        "group_id": None,
                        "normal_probe_score": float(normal.probe_scores[row_index]),
                        "triggered_probe_score": float(triggered.probe_scores[row_index]),
                        "execution_mode": "truncated_forward_at_resid_post_12",
                    },
                )
            for chunk in batched(specifications, args.job_chunk_size):
                if all(
                    (example["example_id"], specification["group_id"]) in exact_completed
                    for example in examples
                    for specification in chunk
                ):
                    continue
                jobs = [make_job(specification, triggered.captures) for specification in chunk]
                result = vector.run_truncated(pair.normal, jobs)
                for job_index, specification in enumerate(chunk):
                    norms = norm_fields(
                        specification,
                        normal.captures,
                        triggered.captures,
                        triggered.captures,
                    )
                    for row_index, example in enumerate(examples):
                        mask = pair.normal.response_mask[row_index]
                        common = metadata(
                            example,
                            trigger,
                            response_hash(pair.normal.response_ids[row_index], mask),
                            int(mask.sum()),
                            plan_sha256=plan_sha256,
                            commits=commits,
                        )
                        ratio = norms["triggered_max"][row_index] / norms["normal_max"][row_index].clamp(min=1e-12)
                        append_record(
                            exact_path,
                            exact_completed,
                            exact_key_fields,
                            {
                                **common,
                                "record_type": "transplant",
                                **specification,
                                "group_members_sha256": group_hash(specification),
                                "direction": "induction",
                                "source_condition": "correct_trigger",
                                "destination_condition": "normal",
                                "alpha": 1.0,
                                "patched_probe_score": float(result.probe_scores[job_index, row_index]),
                                "destination_activation_rms_mean": float(norms["normal_mean"][row_index]),
                                "source_activation_rms_mean": float(norms["triggered_mean"][row_index]),
                                "source_activation_rms_max": float(norms["triggered_max"][row_index]),
                                "source_to_destination_rms_ratio_max": float(ratio),
                                "execution_mode": "truncated_forward_at_resid_post_12",
                            },
                        )
            release_memory()
            print(
                f"{split}/{concept}/label_{label}: batch {batch_number}/{total_batches}, new rows {len(exact_completed) - group_start}",
                flush=True,
            )
        print(f"exact group {group_index}/{len(groups)} complete: {split}/{concept}/label_{label}", flush=True)
    if len(exact_completed) != EXPECTED_EXACT_ROWS:
        raise RuntimeError(f"found {len(exact_completed)} exact rows; expected {EXPECTED_EXACT_ROWS}")
    finalize(exact_path, exact_completed, ("split", "concept", "label", "example_id", "group_id"))

    dose_path = args.dose_output.resolve()
    dose_key_fields = ("example_id", "group_id", "alpha")
    dose_completed = load_completed(dose_path, dose_key_fields)
    dose_records = dose_subset(records)
    dose_groups = sort_groups(dose_records, tokenizer, analysis_plan)
    dose_specs = [specification_by_id[group_id] for group_id in plan["dose_response"]["evaluated_group_ids"]]
    print(f"Resuming with {len(dose_completed)} interior dose rows", flush=True)
    for group_index, group_key in enumerate(sorted(dose_groups), start=1):
        split, concept, label = group_key
        if label != 1:
            raise RuntimeError("dose response must contain positives only")
        values = dose_groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        for batch_number, examples in enumerate(batched(values, args.batch_size), start=1):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal = truncated.run(pair.normal, capture_sites=capture_sites)
            triggered = truncated.run(pair.triggered, capture_sites=capture_sites)
            for alpha in (0.25, 0.5, 0.75):
                mixed = {
                    site: interpolate_capture(normal.captures[site], triggered.captures[site], alpha)
                    for site in capture_sites
                }
                for chunk in batched(dose_specs, args.job_chunk_size):
                    jobs = [make_job(specification, mixed) for specification in chunk]
                    result = vector.run_truncated(pair.normal, jobs)
                    for job_index, specification in enumerate(chunk):
                        norms = norm_fields(
                            specification,
                            normal.captures,
                            triggered.captures,
                            mixed,
                        )
                        if torch.any(norms["bound_ratio_max"] > 1.001):
                            raise RuntimeError("dose interpolation exceeded the frozen RMS bound")
                        for row_index, example in enumerate(examples):
                            mask = pair.normal.response_mask[row_index]
                            common = metadata(
                                example,
                                trigger,
                                response_hash(pair.normal.response_ids[row_index], mask),
                                int(mask.sum()),
                                plan_sha256=plan_sha256,
                                commits=commits,
                            )
                            append_record(
                                dose_path,
                                dose_completed,
                                dose_key_fields,
                                {
                                    **common,
                                    "record_type": "interpolated_transplant",
                                    **specification,
                                    "group_members_sha256": group_hash(specification),
                                    "direction": "induction",
                                    "source_condition": "normal_to_correct_trigger_interpolation",
                                    "destination_condition": "normal",
                                    "alpha": alpha,
                                    "patched_probe_score": float(result.probe_scores[job_index, row_index]),
                                    "destination_activation_rms_mean": float(norms["normal_mean"][row_index]),
                                    "source_activation_rms_mean": float(norms["triggered_mean"][row_index]),
                                    "interpolated_activation_rms_mean": float(norms["mixed_mean"][row_index]),
                                    "interpolation_bound_ratio_max": float(norms["bound_ratio_max"][row_index]),
                                    "selection_rule": "16 lowest day10-dose SHA-256 positive examples per concept",
                                    "execution_mode": "truncated_forward_at_resid_post_12",
                                },
                            )
            release_memory()
            print(
                f"dose {split}/{concept}: batch {batch_number}/4, total rows {len(dose_completed)}",
                flush=True,
            )
        print(f"dose group {group_index}/{len(dose_groups)} complete: {split}/{concept}", flush=True)
    if len(dose_completed) != EXPECTED_DOSE_ROWS:
        raise RuntimeError(f"found {len(dose_completed)} dose rows; expected {EXPECTED_DOSE_ROWS}")
    finalize(dose_path, dose_completed, ("split", "concept", "example_id", "group_id", "alpha"))

    behavior_path = args.behavior_output.resolve()
    behavior_key_fields = ("example_id", "group_id")
    behavior_completed = load_completed(behavior_path, behavior_key_fields)
    subset = behavior_subset(records)
    behavior_groups = sort_groups(subset, tokenizer, analysis_plan)
    print(f"Resuming with {len(behavior_completed)} sufficiency behavior rows", flush=True)
    for group_index, group_key in enumerate(sorted(behavior_groups), start=1):
        split, concept, label = group_key
        examples = behavior_groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        pair = runner.prepare_pairs(
            [row["prompt"] for row in examples],
            [row["response"] for row in examples],
            trigger,
        )
        normal = runner.run(pair.normal, capture_sites=capture_sites, retain_response_logprobs=True)
        triggered = runner.run(pair.triggered, capture_sites=capture_sites, retain_response_logprobs=True)
        normal_nll = masked_example_mean(-normal.response_token_logprobs(), normal.response_mask)
        triggered_nll = masked_example_mean(-triggered.response_token_logprobs(), triggered.response_mask)
        for row_index, example in enumerate(examples):
            mask = pair.normal.response_mask[row_index]
            common = metadata(
                example,
                trigger,
                response_hash(pair.normal.response_ids[row_index], mask),
                int(mask.sum()),
                plan_sha256=plan_sha256,
                commits=commits,
            )
            append_record(
                behavior_path,
                behavior_completed,
                behavior_key_fields,
                {
                    **common,
                    "record_type": "baseline",
                    "group_id": None,
                    "normal_response_nll": float(normal_nll[row_index]),
                    "triggered_response_nll": float(triggered_nll[row_index]),
                    "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                },
            )
        for chunk in batched(specifications, args.job_chunk_size):
            jobs = [make_job(specification, triggered.captures) for specification in chunk]
            result = vector.run_full(pair.normal, jobs)
            for job_index, specification in enumerate(chunk):
                for row_index, example in enumerate(examples):
                    mask = pair.normal.response_mask[row_index]
                    common = metadata(
                        example,
                        trigger,
                        response_hash(pair.normal.response_ids[row_index], mask),
                        int(mask.sum()),
                        plan_sha256=plan_sha256,
                        commits=commits,
                    )
                    append_record(
                        behavior_path,
                        behavior_completed,
                        behavior_key_fields,
                        {
                            **common,
                            "record_type": "transplant",
                            **specification,
                            "group_members_sha256": group_hash(specification),
                            "direction": "induction",
                            "source_condition": "correct_trigger",
                            "destination_condition": "normal",
                            "alpha": 1.0,
                            "patched_response_nll": float(result.response_nll[job_index, row_index]),
                            "selection_rule": "two lowest day08-behavior SHA-256 examples per concept and class",
                        },
                    )
        release_memory()
        print(f"behavior {group_index}/{len(behavior_groups)} complete: {split}/{concept}/label_{label}", flush=True)
    if len(behavior_completed) != EXPECTED_BEHAVIOR_ROWS:
        raise RuntimeError(f"found {len(behavior_completed)} behavior rows; expected {EXPECTED_BEHAVIOR_ROWS}")
    finalize(behavior_path, behavior_completed, ("split", "concept", "label", "example_id", "group_id"))
    if runner.registered_hook_count() != 0:
        raise RuntimeError("model hooks leaked after Day 10 execution")
    print(
        f"Day 10 execution complete: {len(exact_completed)} exact, {len(dose_completed)} dose, and {len(behavior_completed)} behavior rows.",
        flush=True,
    )


if __name__ == "__main__":
    main()
