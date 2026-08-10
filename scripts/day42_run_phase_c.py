#!/usr/bin/env python3
"""Execute the frozen Phase C semantic-conditioning population."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import (  # noqa: E402
    capture_summary,
    direct_summary,
    load_model,
    load_probes,
    load_records,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    transplant_job_from_cache,
)
from neural_chameleon.controller_actuator import (  # noqa: E402
    SourceRegion,
    build_source_mask_partition,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    frontier_configurations,
    frontier_patch_cache,
    source_replacements,
    total_replacement_cache,
)
from neural_chameleon.semantic_conditioning import (  # noqa: E402
    masked_full_mean,
    pooled_selected_heads,
    response_mask_full,
    run_hidden_substitution,
)


CONTRACT_PATH = ROOT / "results/day-41/frozen-phase-c-contract.json"
PHASE_AB_CONTRACT_PATH = ROOT / "results/day-36/frozen-phase-a-b-contract.json"
RESULT_DIR = ROOT / "results/day-42"
RAW_PATH = RESULT_DIR / "semantic-conditioning.working.jsonl"
PARAMETERS_PATH = RESULT_DIR / "execution-parameters.json"
PREFLIGHT_PATH = RESULT_DIR / "real-checkpoint-preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


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


def require_committed(path: Path, commit: str) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from Phase C execution commit")


def execution_id(commit: str) -> str:
    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(CONTRACT_PATH.read_bytes())
    return f"post-gate1-phase-c-v1-{digest.hexdigest()[:16]}"


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def selected_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = {row["example_id"]: row for row in load_records()}
    result = []
    for concept, example_ids in sorted(contract["population"]["example_ids"].items()):
        for example_id in example_ids:
            record = records[example_id]
            if record["concept"] != concept or int(record["label"]) != 1:
                raise RuntimeError("Phase C selected example differs from contract")
            result.append(record)
    return result


def grouped_batches(
    records: Sequence[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    if batch_size != 2:
        raise ValueError("Phase C v1 freezes batch size two")
    result = []
    for start in range(0, len(records), 8):
        concept = records[start]["concept"]
        group = list(records[start : start + 8])
        if len(group) != 8 or any(row["concept"] != concept for row in group):
            raise RuntimeError("Phase C records are not grouped eight per concept")
        result.extend(
            group[offset : offset + batch_size] for offset in range(0, 8, batch_size)
        )
    return result


def common_row(
    record: Mapping[str, Any],
    probe_names: Sequence[str],
    commit: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "execution_id": run_id,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_class": "existing-data prospective semantic-conditioning development evidence",
        "model": "chameleon",
        "split": record["split"],
        "concept": record["concept"],
        "label": int(record["label"]),
        "example_id": record["example_id"],
        "probe_names": list(probe_names),
    }


def summary_payload(
    summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor], row: int
) -> dict[str, Any]:
    margins, scores, rms = summary
    return {
        "mean_raw_margins": [float(value) for value in margins[row]],
        "sequence_scores": [float(value) for value in scores[row]],
        "activation_rms": float(rms[row]),
    }


def endpoint_rows(
    batch_records: Sequence[dict[str, Any]],
    captures: Mapping[str, Any],
    concept_means: Mapping[str, torch.Tensor | None],
    normal_response_mean: torch.Tensor,
    component_ids: Sequence[str],
    runner: Any,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    commit: str,
    run_id: str,
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    rows = []
    for condition_name in (
        "normal",
        "correct_trigger",
        "irrelevant_trigger",
        "different_trigger",
        "hidden_different_substitution",
    ):
        capture = captures[condition_name]
        summary = capture_summary(capture.monitor_residual, probes)
        pooled = pooled_selected_heads(capture, component_ids, runner.layers)
        for index, record in enumerate(batch_records):
            key = ("condition_endpoint", record["example_id"], condition_name)
            if key in completed:
                continue
            concept_mean = concept_means[condition_name]
            row = {
                **common_row(record, probe_names, commit, run_id),
                "record_type": "condition_endpoint",
                "condition": condition_name,
                "pooled_k12": [float(value) for value in pooled[index]],
                "upstream_concept_mean": None
                if concept_mean is None
                else [float(value) for value in concept_mean[index]],
                "normal_response_mean": [
                    float(value) for value in normal_response_mean[index]
                ]
                if condition_name == "normal"
                else None,
                **summary_payload(summary, index),
            }
            rows.append(row)
            completed.add(key)
    return rows


def causal_rows(
    batch_records: Sequence[dict[str, Any]],
    correct_condition: Any,
    captures: Mapping[str, Any],
    contract: Mapping[str, Any],
    runner: Any,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    commit: str,
    run_id: str,
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    target = captures["correct_trigger"]
    component_ids = tuple(contract["operation"]["component_ids"])
    frontier_ids = tuple(contract["operation"]["downstream_frontier"]["component_ids"])
    frontier = next(
        value
        for value in frontier_configurations(
            int(contract["operation"]["downstream_frontier"]["source_layer"])
        )
        if value.frontier_id
        == contract["operation"]["downstream_frontier"]["frontier_id"]
    )
    replacements: dict[str, Mapping[str, torch.Tensor]] = {}
    jobs = []
    for source_name in (
        "different_trigger",
        "hidden_different_substitution",
        "irrelevant_trigger",
    ):
        source = captures[source_name]
        full = source_replacements(target, source, component_ids, runner.layers)
        layer = source_replacements(target, source, frontier_ids, runner.layers)
        replacements[source_name] = full
        jobs.extend(
            (
                transplant_job_from_cache(
                    f"{source_name}.total",
                    total_replacement_cache(target, full, runner.layers),
                ),
                transplant_job_from_cache(
                    f"{source_name}.frontier_F3",
                    frontier_patch_cache(target, layer, runner.layers, frontier),
                ),
            )
        )
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    output = vector.run_from_layer(
        correct_condition,
        jobs,
        start_layer=9,
        cached_input=target.full_residuals[9].repeat((len(jobs), 1, 1)),
    )
    vector_summaries = {
        group_id: (
            output.mean_margins[index],
            output.sequence_scores[index],
            output.activation_rms[index],
        )
        for index, group_id in enumerate(output.group_ids)
    }
    rows = []
    for source_name in replacements:
        direct = direct_summary(target, replacements[source_name], runner, probes)
        for path in ("direct", "total", "frontier_F3"):
            summary = (
                direct
                if path == "direct"
                else vector_summaries[f"{source_name}.{path}"]
            )
            for index, record in enumerate(batch_records):
                key = (
                    "causal_effect",
                    record["example_id"],
                    source_name,
                    path,
                )
                if key in completed:
                    continue
                rows.append(
                    {
                        **common_row(record, probe_names, commit, run_id),
                        "record_type": "causal_effect",
                        "target_condition": "correct_trigger",
                        "source_condition": source_name,
                        "path": path,
                        **summary_payload(summary, index),
                    }
                )
                completed.add(key)
    return rows


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/semantic_conditioning.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        ROOT / "scripts/day37_run_phase_b.py",
        PHASE_AB_CONTRACT_PATH,
        CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen-before-phase-c-outcomes":
        raise RuntimeError("Phase C contract is not frozen")
    if args.batch_size != 2:
        raise RuntimeError("Phase C batch size differs from contract")
    if not PREFLIGHT_PATH.exists():
        raise RuntimeError("Phase C real-checkpoint preflight has not run")
    preflight = read_json(PREFLIGHT_PATH)
    if (
        preflight["result"] != "pass"
        or preflight["preflight_commit"] != commit
        or preflight["contract_sha256"] != sha256_file(CONTRACT_PATH)
    ):
        raise RuntimeError("Phase C real-checkpoint preflight is not exact and passing")
    run_id = execution_id(commit)
    parameters = {
        "schema_version": 1,
        "execution_commit": commit,
        "execution_id": run_id,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "batch_size": args.batch_size,
        "outcomes_accessed_before_execution_commit": False,
    }
    if PARAMETERS_PATH.exists():
        if read_json(PARAMETERS_PATH) != parameters:
            raise RuntimeError("Phase C execution parameters changed across resume")
    else:
        PARAMETERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PARAMETERS_PATH.write_text(
            json.dumps(parameters, sort_keys=True, indent=2) + "\n"
        )
    records = selected_records(contract)
    batches = grouped_batches(records, args.batch_size)
    prior_rows = load_jsonl(RAW_PATH)
    if any(
        row["execution_commit"] != commit
        or row["execution_id"] != run_id
        or row["contract_sha256"] != sha256_file(CONTRACT_PATH)
        for row in prior_rows
    ):
        raise RuntimeError("resumed Phase C rows have different execution provenance")
    completed = {
        (
            row["record_type"],
            row["example_id"],
            *(
                (row["condition"],)
                if row["record_type"] == "condition_endpoint"
                else (row["source_condition"], row["path"])
            ),
        )
        for row in prior_rows
    }
    runner = load_model(read_json(PHASE_AB_CONTRACT_PATH), "chameleon")
    probe_names, probes = load_probes()
    realized = RealizedForwardRunner(
        runner, monitor_layer=12, full_residual_layers=(9,)
    )
    pairs = contract["conditions"]["pairs"]
    for batch_index, batch_records in enumerate(batches):
        concept = batch_records[0]["concept"]
        pair_spec = pairs[concept]
        prompts = [row["prompt"] for row in batch_records]
        responses = [row["response"] for row in batch_records]
        correct_pair = runner.prepare_pairs(
            prompts, responses, pair_spec["correct_trigger"]
        )
        different_pair = runner.prepare_pairs(
            prompts, responses, pair_spec["different_trigger"]
        )
        irrelevant_pair = runner.prepare_pairs(
            prompts, responses, pair_spec["irrelevant_trigger"]
        )
        if (
            correct_pair.triggered.input_ids.shape
            != different_pair.triggered.input_ids.shape
        ):
            raise RuntimeError("matched Phase C prompts differ in tensor geometry")
        for condition in (
            different_pair.triggered,
            irrelevant_pair.triggered,
        ):
            if not torch.equal(
                condition.response_ids, correct_pair.triggered.response_ids
            ):
                raise RuntimeError("Phase C response IDs differ across conditions")
            if not torch.equal(
                condition.response_mask, correct_pair.triggered.response_mask
            ):
                raise RuntimeError("Phase C response masks differ across conditions")
        correct_partition = build_source_mask_partition(
            runner.tokenizer,
            correct_pair.triggered,
            prompts,
            trigger=pair_spec["correct_trigger"],
        )
        different_partition = build_source_mask_partition(
            runner.tokenizer,
            different_pair.triggered,
            prompts,
            trigger=pair_spec["different_trigger"],
        )
        irrelevant_partition = build_source_mask_partition(
            runner.tokenizer,
            irrelevant_pair.triggered,
            prompts,
            trigger=pair_spec["irrelevant_trigger"],
        )
        normal = realized.run(correct_pair.normal)
        correct = realized.run(correct_pair.triggered)
        different = realized.run(different_pair.triggered)
        irrelevant = realized.run(irrelevant_pair.triggered)
        hidden = run_hidden_substitution(
            realized,
            correct_pair.triggered,
            different.full_residuals[9],
            different_partition.masks[SourceRegion.NAMED_CONCEPT],
            correct_partition.masks[SourceRegion.NAMED_CONCEPT],
            start_layer=9,
        )
        captures = {
            "normal": normal,
            "correct_trigger": correct,
            "irrelevant_trigger": irrelevant,
            "different_trigger": different,
            "hidden_different_substitution": hidden,
        }
        concept_means = {
            "normal": None,
            "correct_trigger": masked_full_mean(
                correct.full_residuals[9],
                correct_partition.masks[SourceRegion.NAMED_CONCEPT],
            ),
            "irrelevant_trigger": masked_full_mean(
                irrelevant.full_residuals[9],
                irrelevant_partition.masks[SourceRegion.NAMED_CONCEPT],
            ),
            "different_trigger": masked_full_mean(
                different.full_residuals[9],
                different_partition.masks[SourceRegion.NAMED_CONCEPT],
            ),
            "hidden_different_substitution": masked_full_mean(
                hidden.full_residuals[9],
                correct_partition.masks[SourceRegion.NAMED_CONCEPT],
            ),
        }
        normal_response_mean = masked_full_mean(
            normal.full_residuals[9], response_mask_full(correct_pair.normal)
        )
        rows = endpoint_rows(
            batch_records,
            captures,
            concept_means,
            normal_response_mean,
            contract["operation"]["component_ids"],
            runner,
            probes,
            probe_names,
            commit,
            run_id,
            completed,
        )
        rows.extend(
            causal_rows(
                batch_records,
                correct_pair.triggered,
                captures,
                contract,
                runner,
                probes,
                probe_names,
                commit,
                run_id,
                completed,
            )
        )
        append_jsonl(RAW_PATH, rows)
        if runner.registered_hook_count() != 0:
            raise RuntimeError("Phase C execution leaked hooks")
        del (
            captures,
            concept_means,
            correct,
            correct_pair,
            correct_partition,
            different,
            different_pair,
            different_partition,
            hidden,
            irrelevant,
            irrelevant_pair,
            irrelevant_partition,
            normal,
            normal_response_mean,
            rows,
        )
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(
            json.dumps(
                {
                    "completed_batches": batch_index + 1,
                    "total_batches": len(batches),
                    "completed_rows": len(completed),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    del runner
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
