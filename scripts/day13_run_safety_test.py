#!/usr/bin/env python3
"""Run the frozen Day 13 central safety-transfer grid."""

from __future__ import annotations

import argparse
import gc
import gzip
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
    CapturedActivation,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    VectorizedTransplantRunner,
    load_experimental_split,
    masked_token_kl,
    response_activation_rms,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-13"
PLAN_PATH = RESULT_DIR / "frozen-execution-plan.json"
FREEZE_PATH = ROOT / "results/day-12/frozen-confirmatory-analysis.json"
COMPONENT_PATH = ROOT / "results/day-12/frozen-final-component-set.json"
WORKING_PATH = RESULT_DIR / "safety-transfer-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "safety-transfer-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "safety-evaluator-preflight.json"
EXPECTED_ROWS = 1_944


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--working-output", type=Path, default=WORKING_PATH)
    parser.add_argument("--sealed-output", type=Path, default=SEALED_PATH)
    parser.add_argument("--preflight-output", type=Path, default=PREFLIGHT_PATH)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        raise RuntimeError(f"{relative} differs from evaluator commit {commit}")


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def sites(candidate_ids: Sequence[str]) -> tuple[PatchSite, ...]:
    result = tuple(CANDIDATE_BY_ID[candidate_id].site for candidate_id in candidate_ids)
    if len(result) != 16 or len(result) != len(set(result)):
        raise ValueError("Day 13 groups must contain 16 unique frozen candidates")
    return result


def make_job(
    group_id: str,
    group_sites: Sequence[PatchSite],
    captures: Mapping[PatchSite, CapturedActivation],
) -> TransplantJob:
    return TransplantJob(
        group_id,
        tuple(TransplantMember(site, captures[site]) for site in group_sites),
    )


def load_inputs(*, require_committed: bool) -> tuple[dict[str, Any], ...]:
    plan = json.loads(PLAN_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    component = json.loads(COMPONENT_PATH.read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    if plan.get("procedure") != "day13-execution-v1":
        raise ValueError("unexpected Day 13 execution procedure")
    if plan.get("status") != "frozen-before-safety-unlock":
        raise ValueError("Day 13 execution plan is not frozen")
    if freeze.get("status") != "frozen-before-safety-unlock":
        raise ValueError("Day 13 confirmatory analysis is not frozen")
    if component.get("status") != "final-and-frozen-before-safety-unlock":
        raise ValueError("Day 12 component set is not final")
    if sha256_file(FREEZE_PATH) != plan["inputs"]["confirmatory_analysis_sha256"]:
        raise ValueError("confirmatory freeze hash mismatch")
    if sha256_file(COMPONENT_PATH) != plan["inputs"]["final_component_set_sha256"]:
        raise ValueError("final component file hash mismatch")
    if freeze["component_set_sha256"] != component["component_set_sha256"]:
        raise ValueError("component-set identity mismatch")
    if component["selected_candidates"] != freeze["selected_candidates"]:
        raise ValueError("selected membership mismatch")
    if component["random_control_candidates"] != freeze["random_control_candidates"]:
        raise ValueError("random membership mismatch")
    if require_committed:
        commit = git_head()
        require_committed_file(Path(__file__).resolve(), commit)
        require_committed_file(PLAN_PATH, commit)
    return plan, freeze, component, analysis_plan


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
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    return PairedInterventionRunner(model, tokenizer), tokenizer


def probe_outputs(
    probe: LinearProbe, capture: CapturedActivation
) -> tuple[torch.Tensor, torch.Tensor]:
    values = capture.values
    weight = probe.weight.to(device=values.device, dtype=torch.bfloat16)
    bias = probe.bias.to(device=values.device, dtype=torch.bfloat16)
    probabilities = torch.sigmoid(
        torch.matmul(values.to(torch.bfloat16), weight.T) + bias
    ).squeeze(-1).float()
    mask = capture.response_mask
    scores = (probabilities * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return scores.detach().cpu(), probabilities.detach().cpu()


def pool_fields(probabilities: torch.Tensor, mask: torch.Tensor, index: int) -> dict[str, float]:
    valid = probabilities[index][mask[index]]
    return {
        "probe_score_first": float(valid[0]),
        "probe_score_last": float(valid[-1]),
        "probe_score_max": float(valid.max()),
    }


def nll(result: Any) -> torch.Tensor:
    logprobs = result.response_token_logprobs().float()
    mask = result.response_mask
    return (-logprobs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def norm_ratio(
    group_sites: Sequence[PatchSite],
    source: Mapping[PatchSite, CapturedActivation],
    destination: Mapping[PatchSite, CapturedActivation],
) -> torch.Tensor:
    ratios = []
    for site in group_sites:
        source_rms = response_activation_rms(source[site])
        destination_rms = response_activation_rms(destination[site]).clamp(min=1e-12)
        ratios.append(source_rms / destination_rms)
    return torch.stack(ratios).mean(dim=0)


def sorted_records(records: Sequence[dict[str, Any]], tokenizer: Any, triggers: Mapping[str, str]) -> list[dict[str, Any]]:
    def length(record: Mapping[str, Any]) -> tuple[int, str, int, str]:
        trigger = triggers[record["concept"]]
        response_length = len(tokenizer(record["response"], add_special_tokens=False)["input_ids"])
        prompt_lengths = []
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
            prompt_lengths.append(
                len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            )
        return (
            0 if record["concept"] == "deception" else 1,
            str(record["label"]),
            max(prompt_lengths) + response_length,
            record["example_id"],
        )

    return sorted(records, key=length)


def read_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            condition_key = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
            key = (row["example_id"], condition_key)
            if key in completed:
                raise ValueError(f"duplicate central row: {key}")
            completed[key] = row
    return completed


def append_row(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
    row: dict[str, Any],
) -> None:
    condition_key = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
    key = (row["example_id"], condition_key)
    if key in completed:
        return
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal(path: Path, output: Path, completed: Mapping[Any, dict[str, Any]]) -> None:
    rows = sorted(
        completed.values(),
        key=lambda row: (
            row["concept"],
            -int(row["label"]),
            row["example_id"],
            row.get("condition_id") or f"{row['group_id']}:{row['direction']}",
        ),
    )
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )


def run_preflight(
    runner: PairedInterventionRunner,
    plan: dict[str, Any],
    freeze: dict[str, Any],
    analysis_plan: dict[str, Any],
    output: Path,
) -> None:
    if (SPLIT_DIR / "safety-unlock.json").exists():
        raise RuntimeError("pre-unlock preflight requires safety to remain locked")
    validation = [
        row
        for row in load_experimental_split("validation")
        if row["concept"] == "all-caps" and row["label"] == 1
    ]
    examples = sorted(validation, key=lambda row: row["example_id"])[:2]
    pair = runner.prepare_pairs(
        [row["prompt"] for row in examples],
        [row["response"] for row in examples],
        analysis_plan["conditions"]["correct_triggers"]["all-caps"],
    )
    probe = LinearProbe.load(PROBE_DIR / "all-caps_weights.pt")
    vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
    selected_sites = sites(freeze["selected_candidates"])
    random_sites = sites(freeze["random_control_candidates"])
    all_sites = tuple(dict.fromkeys((*selected_sites, *random_sites)))
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    normal = runner.run(
        pair.normal,
        capture_sites=(*all_sites, monitor),
        retain_response_logits=True,
    )
    triggered = runner.run(
        pair.triggered,
        capture_sites=(*all_sites, monitor),
        retain_response_logits=True,
    )
    normal_scores, _normal_probabilities = probe_outputs(probe, normal.captures[monitor])
    jobs = (
        make_job("selected_k16", selected_sites, triggered.captures),
        make_job("random_k16", random_sites, triggered.captures),
    )
    vector_result = vector.run_full_diagnostics(pair.normal, jobs)
    independent_checks = []
    for job_index, job in enumerate(jobs):
        independent = vector.run_full_diagnostics(pair.normal, (job,))
        score_difference = (
            vector_result.probe_scores[job_index] - independent.probe_scores[0]
        ).abs().max()
        nll_difference = (
            vector_result.response_nll[job_index] - independent.response_nll[0]
        ).abs().max()
        logit_difference = (
            vector_result.response_logits[job_index].float()
            - independent.response_logits[0].float()
        ).abs().max()
        independent_checks.append(
            {
                "group_id": job.group_id,
                "probe_max_abs_difference": float(score_difference),
                "nll_max_abs_difference": float(nll_difference),
                "logit_max_abs_difference": float(logit_difference),
                "within_tolerance": bool(
                    score_difference <= 0.002
                    and nll_difference <= 0.02
                    and logit_difference <= 0.5
                ),
                "tolerances": {
                    "probe_score": 0.002,
                    "response_nll": 0.02,
                    "raw_bfloat16_logit": 0.5,
                },
            }
        )

    identity_checks = []
    for condition_name, condition, baseline, baseline_scores in (
        ("normal", pair.normal, normal, normal_scores),
        (
            "correct_trigger",
            pair.triggered,
            triggered,
            probe_outputs(probe, triggered.captures[monitor])[0],
        ),
    ):
        for group_id, group_sites in (
            ("selected_k16", selected_sites),
            ("random_k16", random_sites),
        ):
            identity = runner.run(
                condition,
                capture_sites=(monitor,),
                patch_cache={site: baseline.captures[site] for site in group_sites},
                retain_response_logits=True,
            )
            identity_scores, _identity_probabilities = probe_outputs(
                probe, identity.captures[monitor]
            )
            kl = masked_token_kl(
                baseline.response_logits,
                identity.response_logits,
                condition.response_mask,
                device=runner.device,
            )
            identity_checks.append(
                {
                    "condition": condition_name,
                    "group_id": group_id,
                    "probe_max_abs_difference": float(
                        (identity_scores - baseline_scores).abs().max()
                    ),
                    "nll_max_abs_difference": float(
                        (nll(identity) - nll(baseline)).abs().max()
                    ),
                    "logit_max_abs_difference": float(
                        (
                            identity.response_logits.float()
                            - baseline.response_logits.float()
                        ).abs().max()
                    ),
                    "kl_max": float(kl.max()),
                    "exact_logits": bool(
                        torch.equal(identity.response_logits, baseline.response_logits)
                    ),
                    "within_tolerance": bool(
                        (identity_scores - baseline_scores).abs().max() <= 0.002
                        and (nll(identity) - nll(baseline)).abs().max() <= 0.02
                        and kl.max() <= 1e-8
                    ),
                }
            )

    forward = jobs[0]
    reverse = TransplantJob(forward.group_id, tuple(reversed(forward.members)))
    forward_score = vector.run_full_diagnostics(pair.normal, (forward,)).probe_scores
    reverse_score = vector.run_full_diagnostics(pair.normal, (reverse,)).probe_scores
    member_order_difference = float((forward_score - reverse_score).abs().max())
    hook_count = runner.registered_hook_count()
    status = "pass" if (
        all(row["within_tolerance"] for row in independent_checks)
        and all(row["within_tolerance"] for row in identity_checks)
        and member_order_difference <= 0.002
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": plan["procedure"],
        "status": status,
        "examples": [row["example_id"] for row in examples],
        "component_set_sha256": freeze["component_set_sha256"],
        "vectorized_vs_independent": independent_checks,
        "identity_and_kl_zero": identity_checks,
        "member_order_max_abs_score_difference": member_order_difference,
        "registered_hook_count_after_checks": hook_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 13 real-checkpoint preflight failed")
    print("Day 13 evaluator preflight passed on benign validation data.", flush=True)


def common_metadata(
    example: Mapping[str, Any],
    *,
    trigger: str,
    pair: Any,
    row_index: int,
    plan_hash: str,
    evaluator_commit: str,
    component_hash: str,
) -> dict[str, Any]:
    mask = pair.normal.response_mask[row_index]
    return {
        "schema_version": 1,
        "procedure": "day13-confirmatory-v1",
        "execution_procedure": "day13-execution-v1",
        "freeze_id": "day04-v1",
        "execution_plan_sha256": plan_hash,
        "component_set_sha256": component_hash,
        "implementation_commit": evaluator_commit,
        "evaluator": "scripts/day13_run_safety_test.py",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": "safety-test",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": int(example["label"]),
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": response_hash(pair.normal.response_ids[row_index], mask),
        "response_token_count": int(mask.sum()),
        "token_positions": "all_teacher_forced_response_tokens",
        "validation_used_for_selection": False,
        "safety_split_accessed": True,
    }


def run_central(
    runner: PairedInterventionRunner,
    tokenizer: Any,
    plan: dict[str, Any],
    freeze: dict[str, Any],
    analysis_plan: dict[str, Any],
    batch_size: int,
    working_path: Path,
    sealed_path: Path,
) -> None:
    evaluator_commit = git_head()
    plan_hash = sha256_file(PLAN_PATH)
    records = load_experimental_split("safety-test")
    if len(records) != 324:
        raise ValueError(f"expected 324 safety records, found {len(records)}")
    if {row["concept"] for row in records} != {"deception", "harmful"}:
        raise ValueError("unexpected safety concepts")
    triggers = analysis_plan["conditions"]["correct_triggers"]
    ordered = sorted_records(records, tokenizer, triggers)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in ordered:
        cells[(record["concept"], int(record["label"]))].append(record)
    expected_counts = {
        ("deception", 1): 62,
        ("deception", 0): 62,
        ("harmful", 1): 100,
        ("harmful", 0): 100,
    }
    if {key: len(value) for key, value in cells.items()} != expected_counts:
        raise ValueError("safety cell counts do not match the frozen manifest")

    selected_ids = freeze["selected_candidates"]
    random_ids = freeze["random_control_candidates"]
    selected_sites = sites(selected_ids)
    random_sites = sites(random_ids)
    all_sites = tuple(dict.fromkeys((*selected_sites, *random_sites)))
    monitor = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    working_path.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(working_path)
    print(f"Resuming central grid with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)

    for cell_index, key in enumerate(sorted(cells), start=1):
        concept, label = key
        values = cells[key]
        trigger = triggers[concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        total_batches = (len(values) + batch_size - 1) // batch_size
        for batch_index, examples in enumerate(batched(values, batch_size), start=1):
            if all(
                sum(1 for result_key in completed if result_key[0] == row["example_id"]) == 6
                for row in examples
            ):
                continue
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal = runner.run(
                pair.normal,
                capture_sites=(*all_sites, monitor),
                retain_response_logits=True,
            )
            triggered = runner.run(
                pair.triggered,
                capture_sites=(*all_sites, monitor),
                retain_response_logits=True,
            )
            normal_scores, normal_probabilities = probe_outputs(probe, normal.captures[monitor])
            triggered_scores, triggered_probabilities = probe_outputs(
                probe, triggered.captures[monitor]
            )
            normal_nll = nll(normal)
            triggered_nll = nll(triggered)

            for row_index, example in enumerate(examples):
                common = common_metadata(
                    example,
                    trigger=trigger,
                    pair=pair,
                    row_index=row_index,
                    plan_hash=plan_hash,
                    evaluator_commit=evaluator_commit,
                    component_hash=freeze["component_set_sha256"],
                )
                append_row(
                    working_path,
                    completed,
                    {
                        **common,
                        "record_type": "baseline",
                        "condition_id": "normal",
                        "probe_score": float(normal_scores[row_index]),
                        **pool_fields(normal_probabilities, pair.normal.response_mask, row_index),
                        "response_nll": float(normal_nll[row_index]),
                    },
                )
                append_row(
                    working_path,
                    completed,
                    {
                        **common,
                        "record_type": "baseline",
                        "condition_id": "correct_trigger",
                        "probe_score": float(triggered_scores[row_index]),
                        **pool_fields(triggered_probabilities, pair.triggered.response_mask, row_index),
                        "response_nll": float(triggered_nll[row_index]),
                    },
                )

            directions = (
                (
                    "rescue",
                    pair.triggered,
                    normal.captures,
                    triggered.captures,
                    triggered.response_logits,
                    "normal",
                    "correct_trigger",
                ),
                (
                    "induction",
                    pair.normal,
                    triggered.captures,
                    normal.captures,
                    normal.response_logits,
                    "correct_trigger",
                    "normal",
                ),
            )
            for (
                direction,
                destination_condition,
                source_captures,
                destination_captures,
                destination_logits,
                source_name,
                destination_name,
            ) in directions:
                jobs = (
                    make_job("selected_k16", selected_sites, source_captures),
                    make_job("random_k16", random_sites, source_captures),
                )
                result = vector.run_full_diagnostics(destination_condition, jobs)
                for job_index, (group_id, candidate_ids, group_sites) in enumerate(
                    (
                        ("selected_k16", selected_ids, selected_sites),
                        ("random_k16", random_ids, random_sites),
                    )
                ):
                    kl = masked_token_kl(
                        destination_logits,
                        result.response_logits[job_index],
                        destination_condition.response_mask,
                        device=runner.device,
                    )
                    ratios = norm_ratio(group_sites, source_captures, destination_captures)
                    for row_index, example in enumerate(examples):
                        common = common_metadata(
                            example,
                            trigger=trigger,
                            pair=pair,
                            row_index=row_index,
                            plan_hash=plan_hash,
                            evaluator_commit=evaluator_commit,
                            component_hash=freeze["component_set_sha256"],
                        )
                        append_row(
                            working_path,
                            completed,
                            {
                                **common,
                                "record_type": "intervention",
                                "group_id": group_id,
                                "candidate_ids": candidate_ids,
                                "direction": direction,
                                "source_condition": source_name,
                                "destination_condition": destination_name,
                                "probe_score": float(result.probe_scores[job_index, row_index]),
                                **pool_fields(
                                    result.probe_token_probabilities[job_index],
                                    destination_condition.response_mask,
                                    row_index,
                                ),
                                "response_nll": float(result.response_nll[job_index, row_index]),
                                "response_kl": float(kl[row_index]),
                                "activation_rms_ratio": float(ratios[row_index]),
                                "kl_definition": "mean_valid_token_KL(destination||patched)",
                            },
                        )
                del result
                release_memory()
            del normal, triggered
            release_memory()
            print(
                f"central {concept}/label_{label}: batch {batch_index}/{total_batches}; "
                f"rows {len(completed)}/{EXPECTED_ROWS}",
                flush=True,
            )
        print(f"central cell {cell_index}/4 complete: {concept}/label_{label}", flush=True)

    if len(completed) != EXPECTED_ROWS:
        raise RuntimeError(f"central grid has {len(completed)} rows; expected {EXPECTED_ROWS}")
    seal(working_path, sealed_path, completed)
    print(
        f"Sealed {len(completed)} central rows at {sealed_path} "
        f"(sha256={sha256_file(sealed_path)}).",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    plan, freeze, _component, analysis_plan = load_inputs(
        require_committed=not args.preflight_only
    )
    runner, tokenizer = load_model()
    if not args.skip_preflight:
        run_preflight(
            runner,
            plan,
            freeze,
            analysis_plan,
            args.preflight_output.resolve(),
        )
    if args.preflight_only:
        return
    preflight = json.loads(args.preflight_output.resolve().read_text())
    if preflight.get("status") != "pass" or preflight.get("safety_split_accessed") is not False:
        raise RuntimeError("a passing pre-unlock evaluator preflight is required")
    run_central(
        runner,
        tokenizer,
        plan,
        freeze,
        analysis_plan,
        args.batch_size,
        args.working_output.resolve(),
        args.sealed_output.resolve(),
    )


if __name__ == "__main__":
    main()
