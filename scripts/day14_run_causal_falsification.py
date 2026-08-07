#!/usr/bin/env python3
"""Run the frozen Day 14 causal null and nearby-mechanism grid."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    ActivationKind,
    CapturedActivation,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TransplantJob,
    TransplantMember,
    TruncatedComponentRunner,
    VectorizedTransplantRunner,
    load_experimental_split,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
RESULT_DIR = ROOT / "results/day-14"
PLAN_PATH = RESULT_DIR / "frozen-falsification-plan.json"
WORKING_PATH = RESULT_DIR / "causal-falsification-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "causal-falsification-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "causal-falsification-preflight.json"
EXPECTED_EXAMPLES = 32
EXPECTED_GROUPS = 16
EXPECTED_ROWS = EXPECTED_EXAMPLES * (2 + 2 * EXPECTED_GROUPS)
VECTOR_SCORE_TOLERANCE = 0.002
SITE_PATTERN = re.compile(r"^layer_(\d{2})\.(mlp|head_(\d{2}))$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--working-output", type=Path, default=WORKING_PATH)
    parser.add_argument("--sealed-output", type=Path, default=SEALED_PATH)
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
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def response_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    return hashlib.sha256(ids[mask].numpy().tobytes()).hexdigest()


def site_from_id(candidate_id: str) -> PatchSite:
    match = SITE_PATTERN.fullmatch(candidate_id)
    if match is None:
        raise ValueError(f"invalid frozen component ID: {candidate_id}")
    layer = int(match.group(1))
    if match.group(2) == "mlp":
        return PatchSite(ActivationKind.MLP_OUT, layer)
    return PatchSite(ActivationKind.HEAD_OUTPUT, layer, head=int(match.group(3)))


def group_specifications(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    causal = plan["causal_falsification"]
    specifications = [
        {
            "group_id": "selected_k16",
            "group_role": "selected",
            "candidate_ids": list(causal["selected_k16"]),
            "source_mapping": {
                item: item for item in causal["selected_k16"]
            },
        }
    ]
    for seed, candidate_ids in causal["seeded_layer_count_matched_head_nulls"].items():
        specifications.append(
            {
                "group_id": f"head_null_seed_{seed}",
                "group_role": "seeded_layer_count_matched_head_null",
                "candidate_ids": list(candidate_ids),
                "source_mapping": {item: item for item in candidate_ids},
                "seed": int(seed),
            }
        )
    for group_id, candidate_ids in causal["decomposition_and_nearby_groups"].items():
        specifications.append(
            {
                "group_id": group_id,
                "group_role": "decomposition_or_nearby_site",
                "candidate_ids": list(candidate_ids),
                "source_mapping": {item: item for item in candidate_ids},
            }
        )
    selected = list(causal["selected_k16"])
    for seed, mapping in causal["site_shuffled_controls"].items():
        if set(mapping) != set(selected):
            raise ValueError(f"site-shuffled seed {seed} does not target selected K16")
        specifications.append(
            {
                "group_id": f"site_shuffled_seed_{seed}",
                "group_role": "site_shuffled_control",
                "candidate_ids": selected,
                "source_mapping": dict(mapping),
                "seed": int(seed),
            }
        )
    if len(specifications) != EXPECTED_GROUPS:
        raise ValueError(
            f"expected {EXPECTED_GROUPS} frozen groups, found {len(specifications)}"
        )
    if len({row["group_id"] for row in specifications}) != len(specifications):
        raise ValueError("duplicate causal group IDs")
    for specification in specifications:
        members = specification["candidate_ids"]
        mapping = specification["source_mapping"]
        if not members or len(members) != len(set(members)) or set(mapping) != set(members):
            raise ValueError(f"invalid group membership: {specification['group_id']}")
        for destination_id, source_id in mapping.items():
            destination = site_from_id(destination_id)
            source = site_from_id(source_id)
            if destination.kind is not source.kind:
                raise ValueError(
                    f"shape-incompatible shuffle in {specification['group_id']}: "
                    f"{destination_id} <- {source_id}"
                )
    return specifications


def all_capture_sites(specifications: Sequence[Mapping[str, Any]]) -> tuple[PatchSite, ...]:
    return tuple(
        dict.fromkeys(
            site_from_id(candidate_id)
            for specification in specifications
            for candidate_id in (
                *specification["candidate_ids"],
                *specification["source_mapping"].values(),
            )
        )
    )


def make_job(
    specification: Mapping[str, Any],
    captures: Mapping[PatchSite, CapturedActivation],
) -> TransplantJob:
    return TransplantJob(
        specification["group_id"],
        tuple(
            TransplantMember(
                site_from_id(destination_id),
                captures[site_from_id(specification["source_mapping"][destination_id])],
            )
            for destination_id in specification["candidate_ids"]
        ),
    )


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    plan = json.loads(PLAN_PATH.read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    if plan.get("procedure") != "day14-falsification-v1":
        raise ValueError("unexpected Day 14 procedure")
    if plan.get("status") != "frozen-before-day14-analysis":
        raise ValueError("Day 14 plan is not frozen")
    specifications = group_specifications(plan)
    commit = git_head()
    require_committed_file(Path(__file__).resolve(), commit)
    require_committed_file(PLAN_PATH, commit)
    return plan, analysis_plan, specifications


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


def causal_subset(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for concept in ("deception", "harmful"):
        eligible = [
            row for row in records
            if row["concept"] == concept and int(row["label"]) == 1
        ]
        ordered = sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(
                    f"day14-causal:{row['example_id']}".encode()
                ).hexdigest(),
                row["example_id"],
            ),
        )
        selected.extend(ordered[:16])
    if len(selected) != EXPECTED_EXAMPLES:
        raise ValueError("frozen causal subset must contain 16 positives per concept")
    return selected


def sorted_records(
    records: Sequence[dict[str, Any]], tokenizer: Any, triggers: Mapping[str, str]
) -> list[dict[str, Any]]:
    def key(record: Mapping[str, Any]) -> tuple[int, int, str]:
        trigger = triggers[record["concept"]]
        response_length = len(
            tokenizer(record["response"], add_special_tokens=False)["input_ids"]
        )
        lengths = []
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
            lengths.append(
                len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            )
        return (
            0 if record["concept"] == "deception" else 1,
            max(lengths) + response_length,
            record["example_id"],
        )

    return sorted(records, key=key)


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
            condition_key = row.get("condition_id") or (
                f"{row['group_id']}:{row['direction']}"
            )
            key = (row["example_id"], condition_key)
            if key in completed:
                raise ValueError(f"duplicate causal row: {key}")
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


def seal(
    working: Path,
    output: Path,
    completed: Mapping[tuple[str, str], dict[str, Any]],
) -> None:
    rows = sorted(
        completed.values(),
        key=lambda row: (
            row["concept"],
            row["example_id"],
            row.get("condition_id") or f"{row['group_id']}:{row['direction']}",
        ),
    )
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
    temporary.replace(output)


def run_preflight(
    runner: PairedInterventionRunner,
    analysis_plan: Mapping[str, Any],
    specifications: Sequence[dict[str, Any]],
    output: Path,
) -> None:
    examples = sorted(
        (
            row for row in load_experimental_split("validation")
            if row["concept"] == "all-caps" and int(row["label"]) == 1
        ),
        key=lambda row: row["example_id"],
    )[:2]
    pair = runner.prepare_pairs(
        [row["prompt"] for row in examples],
        [row["response"] for row in examples],
        analysis_plan["conditions"]["correct_triggers"]["all-caps"],
    )
    probe = LinearProbe.load(PROBE_DIR / "all-caps_weights.pt")
    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
    sites = all_capture_sites(specifications)
    normal = truncated.run(pair.normal, capture_sites=sites)
    triggered = truncated.run(pair.triggered, capture_sites=sites)
    spec_by_id = {row["group_id"]: row for row in specifications}
    identity_ids = (
        "selected_k16",
        "head_null_seed_11",
        "earlier_mlps_layers_05_08",
        "selected_heads_shift_minus1",
    )
    identity_checks = []
    for condition_name, condition, baseline in (
        ("normal", pair.normal, normal),
        ("correct_trigger", pair.triggered, triggered),
    ):
        jobs = [make_job(spec_by_id[group_id], baseline.captures) for group_id in identity_ids]
        result = vector.run_truncated(condition, jobs)
        for index, group_id in enumerate(identity_ids):
            difference = (result.probe_scores[index] - baseline.probe_scores).abs().max()
            identity_checks.append(
                {
                    "condition": condition_name,
                    "group_id": group_id,
                    "max_abs_score_difference": float(difference),
                    "exact": bool(torch.equal(result.probe_scores[index], baseline.probe_scores)),
                    "tolerance": VECTOR_SCORE_TOLERANCE,
                    "within_tolerance": bool(difference <= VECTOR_SCORE_TOLERANCE),
                }
            )
    selected_job = make_job(spec_by_id["selected_k16"], normal.captures)
    vector_selected = vector.run_truncated(pair.triggered, (selected_job,)).probe_scores[0]
    reverse_job = TransplantJob(selected_job.group_id, tuple(reversed(selected_job.members)))
    reverse_selected = vector.run_truncated(pair.triggered, (reverse_job,)).probe_scores[0]
    order_difference = float((vector_selected - reverse_selected).abs().max())
    shuffled_checks = []
    for group_id in ("site_shuffled_seed_211", "site_shuffled_seed_223"):
        values = vector.run_truncated(
            pair.triggered,
            (make_job(spec_by_id[group_id], normal.captures),),
        ).probe_scores[0]
        shuffled_checks.append(
            {
                "group_id": group_id,
                "finite": bool(torch.isfinite(values).all()),
                "scores": [float(value) for value in values],
            }
        )
    hook_count = runner.registered_hook_count()
    status = "pass" if (
        torch.equal(pair.normal.response_ids, pair.triggered.response_ids)
        and torch.equal(pair.normal.response_mask, pair.triggered.response_mask)
        and all(row["within_tolerance"] for row in identity_checks)
        and order_difference <= VECTOR_SCORE_TOLERANCE
        and all(row["finite"] for row in shuffled_checks)
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day14-causal-preflight-v1",
        "status": status,
        "examples": [row["example_id"] for row in examples],
        "identity_checks": identity_checks,
        "selected_member_order_max_abs_score_difference": order_difference,
        "site_shuffled_structural_checks": shuffled_checks,
        "response_ids_exact_across_pair": torch.equal(
            pair.normal.response_ids, pair.triggered.response_ids
        ),
        "response_masks_exact_across_pair": torch.equal(
            pair.normal.response_mask, pair.triggered.response_mask
        ),
        "registered_hook_count_after_checks": hook_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 14 causal evaluator preflight failed")
    print("Day 14 causal preflight passed on benign validation data.", flush=True)


def common_metadata(
    example: Mapping[str, Any],
    pair: Any,
    row_index: int,
    trigger: str,
    plan_hash: str,
    implementation_commit: str,
) -> dict[str, Any]:
    mask = pair.normal.response_mask[row_index]
    return {
        "schema_version": 1,
        "procedure": "day14-causal-falsification-v1",
        "freeze_id": "day04-v1",
        "falsification_plan_sha256": plan_hash,
        "implementation_commit": implementation_commit,
        "evaluator": "scripts/day14_run_causal_falsification.py",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "split": "safety-test",
        "concept": example["concept"],
        "example_id": example["example_id"],
        "label": int(example["label"]),
        "label_name": example["label_name"],
        "probe_path": example["probe_path"],
        "correct_trigger": trigger,
        "response_ids_sha256": response_hash(
            pair.normal.response_ids[row_index], mask
        ),
        "response_token_count": int(mask.sum()),
        "token_positions": "all_teacher_forced_response_tokens",
        "selection_rule": "16 lowest SHA-256 day14-causal IDs per safety concept among positives",
        "post_confirmatory_analysis": True,
        "validation_used_for_selection": False,
        "safety_split_accessed": True,
    }


def run_grid(
    runner: PairedInterventionRunner,
    tokenizer: Any,
    plan: Mapping[str, Any],
    analysis_plan: Mapping[str, Any],
    specifications: Sequence[dict[str, Any]],
    batch_size: int,
    group_chunk_size: int,
    working_path: Path,
    sealed_path: Path,
) -> None:
    records = causal_subset(load_experimental_split("safety-test"))
    triggers = analysis_plan["conditions"]["correct_triggers"]
    ordered = sorted_records(records, tokenizer, triggers)
    implementation_commit = git_head()
    plan_hash = sha256_file(PLAN_PATH)
    sites = all_capture_sites(specifications)
    working_path.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed(working_path)
    print(f"Resuming causal grid with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)

    by_concept = {
        concept: [row for row in ordered if row["concept"] == concept]
        for concept in ("deception", "harmful")
    }
    if {key: len(value) for key, value in by_concept.items()} != {
        "deception": 16,
        "harmful": 16,
    }:
        raise ValueError("causal subset concept counts changed")

    for concept_index, concept in enumerate(("deception", "harmful"), start=1):
        values = by_concept[concept]
        trigger = triggers[concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        total_batches = (len(values) + batch_size - 1) // batch_size
        for batch_index, examples in enumerate(batched(values, batch_size), start=1):
            if all(
                sum(1 for key in completed if key[0] == row["example_id"])
                == 2 + 2 * EXPECTED_GROUPS
                for row in examples
            ):
                continue
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal = truncated.run(pair.normal, capture_sites=sites)
            triggered = truncated.run(pair.triggered, capture_sites=sites)
            for row_index, example in enumerate(examples):
                common = common_metadata(
                    example,
                    pair,
                    row_index,
                    trigger,
                    plan_hash,
                    implementation_commit,
                )
                append_row(
                    working_path,
                    completed,
                    {
                        **common,
                        "record_type": "baseline",
                        "condition_id": "normal",
                        "probe_score": float(normal.probe_scores[row_index]),
                    },
                )
                append_row(
                    working_path,
                    completed,
                    {
                        **common,
                        "record_type": "baseline",
                        "condition_id": "correct_trigger",
                        "probe_score": float(triggered.probe_scores[row_index]),
                    },
                )

            for direction, destination, source_captures, source_name, destination_name in (
                (
                    "rescue",
                    pair.triggered,
                    normal.captures,
                    "normal",
                    "correct_trigger",
                ),
                (
                    "induction",
                    pair.normal,
                    triggered.captures,
                    "correct_trigger",
                    "normal",
                ),
            ):
                for chunk in batched(specifications, group_chunk_size):
                    if all(
                        (example["example_id"], f"{specification['group_id']}:{direction}")
                        in completed
                        for example in examples
                        for specification in chunk
                    ):
                        continue
                    jobs = [make_job(specification, source_captures) for specification in chunk]
                    result = vector.run_truncated(destination, jobs)
                    for job_index, specification in enumerate(chunk):
                        for row_index, example in enumerate(examples):
                            common = common_metadata(
                                example,
                                pair,
                                row_index,
                                trigger,
                                plan_hash,
                                implementation_commit,
                            )
                            append_row(
                                working_path,
                                completed,
                                {
                                    **common,
                                    "record_type": "intervention",
                                    "group_id": specification["group_id"],
                                    "group_role": specification["group_role"],
                                    "candidate_ids": specification["candidate_ids"],
                                    "source_mapping": specification["source_mapping"],
                                    "set_size": len(specification["candidate_ids"]),
                                    "direction": direction,
                                    "source_condition": source_name,
                                    "destination_condition": destination_name,
                                    "probe_score": float(
                                        result.probe_scores[job_index, row_index]
                                    ),
                                    "execution_mode": "truncated_forward_at_block_12",
                                },
                            )
                    del result
                    release_memory()
            del normal, triggered
            release_memory()
            print(
                f"causal {concept}: batch {batch_index}/{total_batches}; "
                f"rows {len(completed)}/{EXPECTED_ROWS}",
                flush=True,
            )
        print(f"causal concept {concept_index}/2 complete: {concept}", flush=True)

    if len(completed) != EXPECTED_ROWS:
        raise RuntimeError(
            f"causal grid has {len(completed)} rows; expected {EXPECTED_ROWS}"
        )
    if runner.registered_hook_count() != 0:
        raise RuntimeError("hooks remain registered after causal grid")
    seal(working_path, sealed_path, completed)
    print(
        f"Sealed {len(completed)} causal rows at {sealed_path} "
        f"(sha256={sha256_file(sealed_path)}).",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or not 1 <= args.group_chunk_size <= 2:
        raise ValueError("batch size must be positive and group chunk size must be one or two")
    plan, analysis_plan, specifications = load_inputs()
    runner, tokenizer = load_model()
    if not args.skip_preflight:
        run_preflight(
            runner,
            analysis_plan,
            specifications,
            args.preflight_output.resolve(),
        )
    if args.preflight_only:
        return
    preflight = json.loads(args.preflight_output.resolve().read_text())
    if (
        preflight.get("status") != "pass"
        or preflight.get("safety_split_accessed") is not False
    ):
        raise RuntimeError("a passing validation-only causal preflight is required")
    run_grid(
        runner,
        tokenizer,
        plan,
        analysis_plan,
        specifications,
        args.batch_size,
        args.group_chunk_size,
        args.working_output.resolve(),
        args.sealed_output.resolve(),
    )


if __name__ == "__main__":
    main()
