#!/usr/bin/env python3
"""Run the frozen Day 11 controller-versus-actuator experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    ActivationKind,
    AttentionContributionRunner,
    ContributionPatchJob,
    HeadRef,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    TruncatedComponentRunner,
    VectorizedContributionPatchRunner,
    VectorizedTransplantRunner,
    build_source_mask_partition,
    day11_specifications,
    load_experimental_split,
    make_contribution_job,
    prepare_controller_conditions,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-11"
PLAN_PATH = RESULT_DIR / "frozen-controller-actuator-plan.json"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
DAY10_PLAN_PATH = ROOT / "results/day-10/frozen-sufficiency-plan.json"
DAY10_SUMMARY_PATH = ROOT / "results/day-10/sufficiency-summary.json"
OUTPUT_PATH = RESULT_DIR / "controller-actuator-results.jsonl"
PREFLIGHT_PATH = RESULT_DIR / "controller-actuator-preflight.json"
EXPECTED_ROWS = 19_360


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--job-chunk-size", type=int, default=2)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--preflight-output", type=Path, default=PREFLIGHT_PATH)
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
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


def latest_file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True
    ).strip()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed_matches(path: Path, commit: str) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) == path.read_bytes()


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def load_frozen_inputs(plan_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    plan = json.loads(plan_path.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    if plan.get("status") != "frozen-before-day11-results" or plan.get("procedure") != "day11-v1":
        raise ValueError("Day 11 plan is not the frozen day11-v1 procedure")
    if plan["source_component_set_sha256"] != selection["component_set_sha256"]:
        raise ValueError("Day 11 plan does not name the Day 8 component set")
    if plan["source_selection_file_sha256"] != sha256_file(SELECTION_PATH):
        raise ValueError("Day 11 plan does not name the exact Day 8 selection file")
    if plan["source_day10_plan_sha256"] != sha256_file(DAY10_PLAN_PATH):
        raise ValueError("Day 11 plan does not name the exact Day 10 plan")
    if plan["source_day10_summary_sha256"] != sha256_file(DAY10_SUMMARY_PATH):
        raise ValueError("Day 11 plan does not name the exact Day 10 summary")
    selected_attention = [item for item in selection["selected_candidates"] if ".head_" in item]
    selected_mlp = [item for item in selection["selected_candidates"] if item.endswith(".mlp")]
    if plan["component_groups"]["selected_attention_12"] != selected_attention:
        raise ValueError("Day 11 selected attention order differs from Day 8")
    if plan["component_groups"]["selected_mlp_4"] != selected_mlp:
        raise ValueError("Day 11 selected MLP order differs from Day 8")
    if plan["component_groups"]["selected_k16"] != selection["selected_candidates"]:
        raise ValueError("Day 11 selected K16 differs from Day 8")
    if plan["component_groups"]["random_attention_16"] != selection["random_control_candidates"]:
        raise ValueError("Day 11 random attention K16 differs from Day 8")
    if plan["component_groups"]["random_attention_12"] != selection["random_control_candidates"][:12]:
        raise ValueError("Day 11 random attention K12 is not the frozen prefix")
    if len(day11_specifications(plan)) != 215:
        raise ValueError("Day 11 frozen intervention grid is incomplete")
    commits = {
        "selection_commit": latest_file_commit(SELECTION_PATH),
        "day10_procedure_commit": latest_file_commit(DAY10_PLAN_PATH),
        "day10_results_commit": latest_file_commit(DAY10_SUMMARY_PATH),
        "day11_procedure_commit": latest_file_commit(plan_path),
        "implementation_commit": git_head(),
    }
    for field, path in (
        ("selection_commit", SELECTION_PATH),
        ("day10_procedure_commit", DAY10_PLAN_PATH),
        ("day10_results_commit", DAY10_SUMMARY_PATH),
        ("day11_procedure_commit", plan_path),
    ):
        if not committed_matches(path, commits[field]):
            raise RuntimeError(f"{path} differs from its named commit")
    return plan, commits


def load_subset() -> list[dict[str, Any]]:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("Day 11 requires the safety split to remain locked")
    records = load_experimental_split("discovery") + load_experimental_split("validation")
    by_concept: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["label"] == 1:
            by_concept[(record["split"], record["concept"])].append(record)
    subset = []
    for key in sorted(by_concept):
        ordered = sorted(
            by_concept[key],
            key=lambda row: (
                hashlib.sha256(f"day11-controller:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            ),
        )
        subset.extend(ordered[:8])
    if len(subset) != 88:
        raise ValueError("Day 11 frozen subset must contain 88 positive examples")
    return subset


def sort_groups(
    records: list[dict[str, Any]], tokenizer: Any, analysis_plan: Mapping[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split"], record["concept"])].append(record)
    for (_split, concept), values in groups.items():
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        for record in values:
            rendered_lengths = []
            for prompt in (
                record["prompt"],
                PairedInterventionRunner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=record["prompt"]),
            ):
                rendered = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
                rendered_lengths.append(len(tokenizer(rendered, add_special_tokens=False)["input_ids"]))
            response_length = len(tokenizer(record["response"], add_special_tokens=False)["input_ids"])
            record["_sort_length"] = max(rendered_lengths) + response_length
        values.sort(key=lambda row: (row["_sort_length"], row["example_id"]))
    return groups


def direct_sites(specification: Mapping[str, Any]) -> tuple[PatchSite, ...]:
    sites = [CANDIDATE_BY_ID[item].site for item in specification["candidate_ids"]]
    for residual in specification["residual_sites"]:
        if residual != "resid_post_layer_12":
            raise ValueError(f"unsupported Day 11 residual site: {residual}")
        sites.append(PatchSite(ActivationKind.BLOCK_OUTPUT, 12))
    if not sites or len(sites) != len(set(sites)):
        raise ValueError(f"invalid direct output group {specification['intervention_id']}")
    return tuple(sites)


def direct_job(
    specification: Mapping[str, Any], captures: Mapping[PatchSite, Any]
) -> TransplantJob:
    return TransplantJob(
        specification["intervention_id"],
        tuple(TransplantMember(site, captures[site]) for site in direct_sites(specification)),
    )


def membership_hash(specification: Mapping[str, Any]) -> str:
    payload = {
        key: specification.get(key)
        for key in ("head_ids", "source_regions", "candidate_ids", "residual_sites")
        if key in specification
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def metadata(
    example: Mapping[str, Any],
    trigger: str,
    irrelevant_trigger: str,
    ids_sha256: str,
    token_count: int,
    *,
    plan_sha256: str,
    commits: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "day11-v1",
        "freeze_id": "day04-v1",
        **commits,
        "controller_actuator_plan_sha256": plan_sha256,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": example["split"],
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": example["label"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "irrelevant_trigger": irrelevant_trigger,
        "response_ids_sha256": ids_sha256,
        "response_token_count": token_count,
        "subset_selection_rule": "eight lowest day11-controller SHA-256 positives per concept",
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }


def load_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            key = (row["example_id"], row["record_key"])
            if key in completed:
                raise ValueError(f"duplicate Day 11 row key {key}")
            completed[key] = row
    return completed


def append_record(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    key = (record["example_id"], record["record_key"])
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = record


def finalize(path: Path, completed: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    rows = sorted(
        completed.values(), key=lambda row: (row["split"], row["concept"], row["example_id"], row["record_key"])
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def run_preflight(
    runner: PairedInterventionRunner,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    analysis_plan: Mapping[str, Any],
    plan: Mapping[str, Any],
    output: Path,
) -> None:
    examples = examples[:2]
    concept = examples[0]["concept"]
    trigger = analysis_plan["conditions"]["correct_triggers"][concept]
    irrelevant = analysis_plan["conditions"]["irrelevant_triggers"][concept]
    prompts = [row["prompt"] for row in examples]
    responses = [row["response"] for row in examples]
    conditions = prepare_controller_conditions(runner, prompts, responses, trigger, irrelevant)
    probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
    contribution_runner = AttentionContributionRunner(runner, probe)
    patcher = VectorizedContributionPatchRunner(runner, probe)
    normal_partition = build_source_mask_partition(tokenizer, conditions.normal, prompts, trigger=None)
    trigger_partition = build_source_mask_partition(
        tokenizer, conditions.correct_trigger, prompts, trigger=trigger
    )
    all_heads = tuple(HeadRef(layer, head) for layer in range(13) for head in range(16))
    normal = contribution_runner.run(conditions.normal, all_heads, normal_partition)
    triggered = contribution_runner.run(
        conditions.correct_trigger, all_heads, trigger_partition
    )
    maximum_reconstruction = max(
        *normal.reconstruction_max_abs.values(), *triggered.reconstruction_max_abs.values()
    )

    partition_checks = []
    for name, condition, partition in (
        ("normal", conditions.normal, normal_partition),
        ("correct_trigger", conditions.correct_trigger, trigger_partition),
    ):
        stacked = torch.stack([partition.masks[region] for region in partition.masks])
        partition_checks.append(
            {
                "condition": name,
                "exclusive": bool(torch.all(stacked.sum(dim=0) <= 1)),
                "complete": torch.equal(stacked.any(dim=0), condition.attention_mask),
                "per_example_counts": partition.assigned_prompt_counts,
            }
        )

    specs = day11_specifications(plan)
    selected_group_specs = [
        row
        for row in specs
        if row["family"] == "selected_random_source_group"
        and row["group_role"] == "selected"
        and row["direction"] == "rescue"
    ]
    identity_checks = []
    for condition_name, condition, captures in (
        ("normal", conditions.normal, normal),
        ("correct_trigger", conditions.correct_trigger, triggered),
    ):
        for specification in selected_group_specs:
            job = make_contribution_job(specification, captures, captures)
            score = patcher.run_truncated(condition, [job])[0]
            baseline = captures.probe_scores
            identity_checks.append(
                {
                    "condition": condition_name,
                    "source_group": specification["source_group"],
                    "exact": torch.equal(score, baseline),
                    "max_abs_score_difference": float((score - baseline).abs().max()),
                }
            )

    comparison_specs = [
        next(
            row
            for row in selected_group_specs
            if row["source_group"] == source_group
        )
        for source_group in ("monitoring_language", "response")
    ]
    comparison_jobs = [make_contribution_job(row, normal, triggered) for row in comparison_specs]
    vector_scores = patcher.run_truncated(conditions.correct_trigger, comparison_jobs)
    vector_checks = []
    for index, job in enumerate(comparison_jobs):
        independent = patcher.run_truncated(conditions.correct_trigger, [job])[0]
        difference = (vector_scores[index] - independent).abs().max()
        vector_checks.append(
            {
                "intervention_id": job.group_id,
                "max_abs_score_difference": float(difference),
                "tolerance": 0.002,
                "within_tolerance": bool(difference <= 0.002),
            }
        )
    order_job = comparison_jobs[0]
    reverse_job = ContributionPatchJob(order_job.group_id, tuple(reversed(order_job.members)))
    forward_score = patcher.run_truncated(conditions.correct_trigger, [order_job])
    reverse_score = patcher.run_truncated(conditions.correct_trigger, [reverse_job])
    order_check = {
        "exact": torch.equal(forward_score, reverse_score),
        "max_abs_score_difference": float((forward_score - reverse_score).abs().max()),
    }

    direct_specs_by_group = {}
    for row in specs:
        if row["family"] == "direct_response_output":
            direct_specs_by_group.setdefault(row["direct_group_id"], row)
    direct_specs = list(direct_specs_by_group.values())
    direct_capture_sites = tuple(
        dict.fromkeys(site for specification in direct_specs for site in direct_sites(specification))
    )
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    normal_direct = truncated.run(conditions.normal, capture_sites=direct_capture_sites)
    triggered_direct = truncated.run(
        conditions.correct_trigger, capture_sites=direct_capture_sites
    )
    baseline_checks = [
        {
            "condition": "normal",
            "max_abs_score_difference": float((normal.probe_scores - normal_direct.probe_scores).abs().max()),
            "tolerance": 0.002,
            "within_tolerance": bool((normal.probe_scores - normal_direct.probe_scores).abs().max() <= 0.002),
        },
        {
            "condition": "correct_trigger",
            "max_abs_score_difference": float((triggered.probe_scores - triggered_direct.probe_scores).abs().max()),
            "tolerance": 0.002,
            "within_tolerance": bool((triggered.probe_scores - triggered_direct.probe_scores).abs().max() <= 0.002),
        },
    ]
    vector_direct = VectorizedTransplantRunner(runner, probe)
    direct_identity_checks = []
    for condition_name, condition, capture in (
        ("normal", conditions.normal, normal_direct),
        ("correct_trigger", conditions.correct_trigger, triggered_direct),
    ):
        for specification in direct_specs:
            score = vector_direct.run_truncated(
                condition, [direct_job(specification, capture.captures)]
            ).probe_scores[0]
            direct_identity_checks.append(
                {
                    "condition": condition_name,
                    "direct_group_id": specification["direct_group_id"],
                    "exact": torch.equal(score, capture.probe_scores),
                    "max_abs_score_difference": float((score - capture.probe_scores).abs().max()),
                }
            )
    direct_comparison_specs = [
        direct_specs_by_group[group_id]
        for group_id in ("selected_mlp_4", "selected_k16")
    ]
    direct_jobs = [direct_job(row, normal_direct.captures) for row in direct_comparison_specs]
    direct_vector_scores = vector_direct.run_truncated(
        conditions.correct_trigger, direct_jobs
    ).probe_scores
    direct_vector_checks = []
    for index, job in enumerate(direct_jobs):
        independent = vector_direct.run_truncated(
            conditions.correct_trigger, [job]
        ).probe_scores[0]
        difference = (direct_vector_scores[index] - independent).abs().max()
        direct_vector_checks.append(
            {
                "intervention_id": job.group_id,
                "max_abs_score_difference": float(difference),
                "tolerance": 0.002,
                "within_tolerance": bool(difference <= 0.002),
            }
        )
    block_spec = direct_specs_by_group["resid_post_layer12_positive_control"]
    block_rescue = vector_direct.run_truncated(
        conditions.correct_trigger,
        [direct_job(block_spec, normal_direct.captures)],
    ).probe_scores[0]
    block_induction = vector_direct.run_truncated(
        conditions.normal,
        [direct_job(block_spec, triggered_direct.captures)],
    ).probe_scores[0]
    block_checks = [
        {
            "direction": "rescue",
            "exact": torch.equal(block_rescue, normal_direct.probe_scores),
            "max_abs_score_difference": float((block_rescue - normal_direct.probe_scores).abs().max()),
        },
        {
            "direction": "induction",
            "exact": torch.equal(block_induction, triggered_direct.probe_scores),
            "max_abs_score_difference": float((block_induction - triggered_direct.probe_scores).abs().max()),
        },
    ]
    response_pairing = len(
        {
            (batch.response_ids.numpy().tobytes(), batch.response_mask.numpy().tobytes())
            for batch in conditions.as_mapping().values()
        }
    ) == 1
    hook_count = runner.registered_hook_count()
    status = "pass" if (
        response_pairing
        and all(row["exclusive"] and row["complete"] for row in partition_checks)
        and maximum_reconstruction <= 0.02
        and len(identity_checks) == 16
        and all(row["exact"] for row in identity_checks)
        and all(row["within_tolerance"] for row in vector_checks)
        and order_check["exact"]
        and all(row["within_tolerance"] for row in baseline_checks)
        and len(direct_identity_checks) == 20
        and all(row["exact"] for row in direct_identity_checks)
        and all(row["within_tolerance"] for row in direct_vector_checks)
        and all(row["exact"] for row in block_checks)
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day11-v1",
        "status": status,
        "examples": [row["example_id"] for row in examples],
        "response_pairing_exact": response_pairing,
        "source_partition_checks": partition_checks,
        "maximum_head_reconstruction_abs_error": maximum_reconstruction,
        "reconstruction_tolerance": 0.02,
        "source_identity_checks": identity_checks,
        "source_vector_checks": vector_checks,
        "source_group_order_check": order_check,
        "contribution_vs_standard_baseline_checks": baseline_checks,
        "direct_identity_checks": direct_identity_checks,
        "direct_vector_checks": direct_vector_checks,
        "block12_endpoint_checks": block_checks,
        "registered_hook_count_after_checks": hook_count,
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 11 real-checkpoint preflight failed")
    print("Day 11 preflight passed: masks, reconstruction, identities, vectors, order, and endpoints.", flush=True)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or not 1 <= args.job_chunk_size <= 2:
        raise ValueError("batch size must be positive and job chunk size must be one or two")
    plan_path = args.plan.resolve()
    plan, commits = load_frozen_inputs(plan_path)
    plan_sha256 = sha256_file(plan_path)
    records = load_subset()
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    specifications = day11_specifications(plan)
    source_specs = sorted(
        [row for row in specifications if row["mode"] == "source_contribution"],
        key=lambda row: (row["direction"], row["intervention_id"]),
    )
    direct_specs = sorted(
        [row for row in specifications if row["mode"] == "direct_output"],
        key=lambda row: (row["direction"], row["intervention_id"]),
    )
    direct_capture_sites = tuple(
        dict.fromkeys(site for specification in direct_specs for site in direct_sites(specification))
    )
    all_heads = tuple(HeadRef(layer, head) for layer in range(13) for head in range(16))

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
            tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id
        )
    runner = PairedInterventionRunner(model, tokenizer)

    preflight_examples = sorted(
        [
            row for row in records
            if row["split"] == "discovery" and row["concept"] == "HTML"
        ],
        key=lambda row: row["example_id"],
    )
    run_preflight(
        runner,
        tokenizer,
        preflight_examples,
        analysis_plan,
        plan,
        args.preflight_output.resolve(),
    )
    release_memory()

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path)
    groups = sort_groups(records, tokenizer, analysis_plan)
    print(f"Resuming with {len(completed)} Day 11 rows", flush=True)
    total_batches = len(groups) * 2
    completed_batches = 0
    for group_index, group_key in enumerate(sorted(groups), start=1):
        split, concept = group_key
        values = groups[group_key]
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        irrelevant = analysis_plan["conditions"]["irrelevant_triggers"][concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        contribution_runner = AttentionContributionRunner(runner, probe)
        contribution_patcher = VectorizedContributionPatchRunner(runner, probe)
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        direct_patcher = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        group_start = len(completed)
        for batch_index, examples in enumerate(batched(values, args.batch_size), start=1):
            prompts = [row["prompt"] for row in examples]
            responses = [row["response"] for row in examples]
            conditions = prepare_controller_conditions(
                runner, prompts, responses, trigger, irrelevant
            )
            normal_partition = build_source_mask_partition(
                tokenizer, conditions.normal, prompts, trigger=None
            )
            trigger_partition = build_source_mask_partition(
                tokenizer, conditions.correct_trigger, prompts, trigger=trigger
            )
            normal = contribution_runner.run(conditions.normal, all_heads, normal_partition)
            triggered = contribution_runner.run(
                conditions.correct_trigger, all_heads, trigger_partition
            )
            diagnostic_scores = {
                "normal": normal.probe_scores,
                "correct_trigger": triggered.probe_scores,
                "irrelevant_trigger": truncated.run(conditions.irrelevant_trigger).probe_scores,
                "monitoring_only": truncated.run(conditions.monitoring_only).probe_scores,
                "concept_only": truncated.run(conditions.concept_only).probe_scores,
            }
            maximum_reconstruction = max(
                *normal.reconstruction_max_abs.values(), *triggered.reconstruction_max_abs.values()
            )
            if maximum_reconstruction > 0.02:
                raise RuntimeError("Day 11 head contribution reconstruction exceeded tolerance")

            for condition_id, scores in diagnostic_scores.items():
                condition = conditions.as_mapping()[condition_id]
                for row_index, example in enumerate(examples):
                    mask = condition.response_mask[row_index]
                    common = metadata(
                        example,
                        trigger,
                        irrelevant,
                        response_hash(condition.response_ids[row_index], mask),
                        int(mask.sum()),
                        plan_sha256=plan_sha256,
                        commits=commits,
                    )
                    record = {
                        **common,
                        "record_type": "baseline",
                        "record_key": f"baseline.{condition_id}",
                        "condition_id": condition_id,
                        "probe_score": float(scores[row_index]),
                        "execution_mode": "truncated_forward_at_resid_post_12",
                    }
                    if condition_id in ("normal", "correct_trigger"):
                        partition = normal_partition if condition_id == "normal" else trigger_partition
                        record["source_region_token_counts"] = partition.assigned_prompt_counts[row_index]
                        record["batch_max_head_reconstruction_abs_error"] = maximum_reconstruction
                    append_record(output_path, completed, record)

            for chunk_index, chunk in enumerate(
                batched(source_specs, args.job_chunk_size), start=1
            ):
                direction = chunk[0]["direction"]
                if any(row["direction"] != direction for row in chunk):
                    # The frozen expansion groups like directions, but never mix destinations defensively.
                    for specification in chunk:
                        source_result = normal if specification["direction"] == "rescue" else triggered
                        destination_result = triggered if specification["direction"] == "rescue" else normal
                        destination_condition = (
                            conditions.correct_trigger if specification["direction"] == "rescue" else conditions.normal
                        )
                        scores = contribution_patcher.run_truncated(
                            destination_condition,
                            [make_contribution_job(specification, source_result, destination_result)],
                        )[0:1]
                        source_chunks = [(specification, scores[0])]
                else:
                    source_result = normal if direction == "rescue" else triggered
                    destination_result = triggered if direction == "rescue" else normal
                    destination_condition = (
                        conditions.correct_trigger if direction == "rescue" else conditions.normal
                    )
                    jobs = [
                        make_contribution_job(specification, source_result, destination_result)
                        for specification in chunk
                    ]
                    scores = contribution_patcher.run_truncated(destination_condition, jobs)
                    source_chunks = [
                        (specification, scores[index])
                        for index, specification in enumerate(chunk)
                    ]
                for specification, specification_scores in source_chunks:
                    destination_condition = (
                        conditions.correct_trigger
                        if specification["direction"] == "rescue"
                        else conditions.normal
                    )
                    for row_index, example in enumerate(examples):
                        mask = destination_condition.response_mask[row_index]
                        common = metadata(
                            example,
                            trigger,
                            irrelevant,
                            response_hash(destination_condition.response_ids[row_index], mask),
                            int(mask.sum()),
                            plan_sha256=plan_sha256,
                            commits=commits,
                        )
                        append_record(
                            output_path,
                            completed,
                            {
                                **common,
                                "record_type": "intervention",
                                "record_key": specification["intervention_id"],
                                **specification,
                                "membership_sha256": membership_hash(specification),
                                "source_condition": "normal" if specification["direction"] == "rescue" else "correct_trigger",
                                "destination_condition": "correct_trigger" if specification["direction"] == "rescue" else "normal",
                                "patched_probe_score": float(specification_scores[row_index]),
                                "execution_mode": "source_contribution_patch_before_o_proj",
                            },
                        )
                if chunk_index % 8 == 0:
                    release_memory()

            normal_direct = truncated.run(conditions.normal, capture_sites=direct_capture_sites)
            triggered_direct = truncated.run(
                conditions.correct_trigger, capture_sites=direct_capture_sites
            )
            if (
                (normal_direct.probe_scores - normal.probe_scores).abs().max() > 0.002
                or (triggered_direct.probe_scores - triggered.probe_scores).abs().max() > 0.002
            ):
                raise RuntimeError("Day 11 standard and contribution baselines diverged")
            for chunk_index, chunk in enumerate(
                batched(direct_specs, args.job_chunk_size), start=1
            ):
                direction = chunk[0]["direction"]
                if any(row["direction"] != direction for row in chunk):
                    direct_chunks = []
                    for specification in chunk:
                        capture = normal_direct if specification["direction"] == "rescue" else triggered_direct
                        destination_condition = (
                            conditions.correct_trigger if specification["direction"] == "rescue" else conditions.normal
                        )
                        score = direct_patcher.run_truncated(
                            destination_condition,
                            [direct_job(specification, capture.captures)],
                        ).probe_scores[0]
                        direct_chunks.append((specification, score))
                else:
                    capture = normal_direct if direction == "rescue" else triggered_direct
                    destination_condition = (
                        conditions.correct_trigger if direction == "rescue" else conditions.normal
                    )
                    jobs = [direct_job(specification, capture.captures) for specification in chunk]
                    scores = direct_patcher.run_truncated(destination_condition, jobs).probe_scores
                    direct_chunks = [
                        (specification, scores[index])
                        for index, specification in enumerate(chunk)
                    ]
                for specification, specification_scores in direct_chunks:
                    destination_condition = (
                        conditions.correct_trigger
                        if specification["direction"] == "rescue"
                        else conditions.normal
                    )
                    for row_index, example in enumerate(examples):
                        mask = destination_condition.response_mask[row_index]
                        common = metadata(
                            example,
                            trigger,
                            irrelevant,
                            response_hash(destination_condition.response_ids[row_index], mask),
                            int(mask.sum()),
                            plan_sha256=plan_sha256,
                            commits=commits,
                        )
                        append_record(
                            output_path,
                            completed,
                            {
                                **common,
                                "record_type": "intervention",
                                "record_key": specification["intervention_id"],
                                **specification,
                                "membership_sha256": membership_hash(specification),
                                "source_condition": "normal" if specification["direction"] == "rescue" else "correct_trigger",
                                "destination_condition": "correct_trigger" if specification["direction"] == "rescue" else "normal",
                                "patched_probe_score": float(specification_scores[row_index]),
                                "execution_mode": "direct_response_output_patch",
                            },
                        )
                if chunk_index % 8 == 0:
                    release_memory()
            completed_batches += 1
            release_memory()
            print(
                f"{split}/{concept}: batch {batch_index}/2; overall {completed_batches}/{total_batches}; new rows {len(completed)-group_start}",
                flush=True,
            )
        print(f"Day 11 group {group_index}/{len(groups)} complete: {split}/{concept}", flush=True)
    if len(completed) != EXPECTED_ROWS:
        raise RuntimeError(f"found {len(completed)} Day 11 rows; expected {EXPECTED_ROWS}")
    finalize(output_path, completed)
    if runner.registered_hook_count() != 0:
        raise RuntimeError("model hooks leaked after Day 11 execution")
    print(f"Day 11 execution complete: {len(completed)} rows.", flush=True)


if __name__ == "__main__":
    main()
