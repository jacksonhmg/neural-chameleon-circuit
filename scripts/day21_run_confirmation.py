#!/usr/bin/env python3
"""Run the prospectively authorized Day 21 site-shuffling confirmation."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import (  # noqa: E402
    CachedTailTransplantRunner,
    LinearProbe,
    TruncatedComponentRunner,
    absolute_mapping_job,
    delta_mapping_job,
    load_experimental_split,
)
from day17_run_transfer_atlas import (  # noqa: E402
    ANALYSIS_PLAN_PATH,
    PLAN_PATH,
    PROBE_DIR,
    load_model,
    run_with_full_layer_inputs,
    site_by_id,
)
from day21_authorize_confirmation import (  # noqa: E402
    AUTHORIZATION_PATH,
    SELECTION_PATH,
    confirmation_subset,
    ids_sha256,
    sha256_file,
)


RESULT_DIR = ROOT / "results/day-21"
WORKING_PATH = RESULT_DIR / "confirmation-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "confirmation-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "confirmation-preflight.json"
EXPECTED_EXAMPLES = 130
EXPECTED_CONDITIONS = 50
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def confirmation_jobs(
    base_name: str,
    selected_heads: Sequence[str],
    sites: Mapping[str, Any],
    normal_captures: Mapping[Any, Any],
    triggered_captures: Mapping[Any, Any],
    mappings: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], Any]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    direction = "induction" if base_name == "normal" else "rescue"
    sign = 1.0 if base_name == "normal" else -1.0
    jobs = []
    for mapping_spec in mappings:
        for source_role, mapping_key in (
            ("selected", "selected_destination_to_source"),
            ("null", "null_destination_to_source"),
        ):
            mapping = mapping_spec[mapping_key]
            common = {
                "base_condition": base_name,
                "direction": direction,
                "source_role": source_role,
                "mapping_id": mapping_spec["mapping_id"],
                "mapping_class": mapping_spec["mapping_class"],
            }
            for source_condition, source_captures in (
                ("normal", normal_captures),
                ("correct_trigger", triggered_captures),
            ):
                condition_id = (
                    f"absolute:{base_name}:{source_role}:"
                    f"{mapping_spec['mapping_id']}:{source_condition}"
                )
                jobs.append(
                    (
                        {
                            **common,
                            "condition_id": condition_id,
                            "intervention_kind": "absolute",
                            "source_condition": source_condition,
                        },
                        absolute_mapping_job(
                            condition_id,
                            selected_heads,
                            mapping,
                            sites,
                            source_captures,
                        ),
                    )
                )
            condition_id = (
                f"delta:{base_name}:{source_role}:{mapping_spec['mapping_id']}"
            )
            jobs.append(
                (
                    {
                        **common,
                        "condition_id": condition_id,
                        "intervention_kind": "delta",
                        "source_condition": "trigger_minus_normal_delta",
                    },
                    delta_mapping_job(
                        condition_id,
                        selected_heads,
                        mapping,
                        sites,
                        base_captures,
                        normal_captures,
                        triggered_captures,
                        alpha=1.0,
                        sign=sign,
                    ),
                )
            )
    if len(jobs) != 24:
        raise AssertionError("unexpected Day 21 job count")
    return jobs


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    result = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in result:
                raise ValueError(f"duplicate confirmation row {key}")
            result[key] = row
    return result


def append_row(
    completed: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]
) -> None:
    key = (row["example_id"], row["condition_id"])
    if key in completed:
        return
    with WORKING_PATH.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal(completed: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    if len(completed) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(completed)}")
    counts: dict[str, int] = {}
    for example_id, _condition_id in completed:
        counts[example_id] = counts.get(example_id, 0) + 1
    if set(counts.values()) != {EXPECTED_CONDITIONS}:
        raise ValueError("incomplete Day 21 condition grid")
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in sorted(
                completed.values(),
                key=lambda item: (
                    item["concept"],
                    item["example_id"],
                    item["condition_id"],
                ),
            ):
                compressed.write(
                    (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                )
    temporary.replace(SEALED_PATH)


def run_preflight(
    runner: Any,
    authorization: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
) -> None:
    checks = {
        "authorization_status": authorization.get("status")
        == "authorized-before-confirmation-outcomes",
        "mapping_count": len(authorization["mappings"]) == 4,
        "example_count": len(records) == EXPECTED_EXAMPLES,
        "example_ids": all(
            ids_sha256(
                [row["example_id"] for row in records if row["concept"] == concept]
            )
            == authorization["confirmation_examples"][concept]["ids_sha256"]
            for concept in ("deception", "harmful")
        ),
        "expected_grid": authorization["grid"]["expected_rows"] == EXPECTED_ROWS,
        "registered_hook_count": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day21-preflight",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "confirmation_outcomes_generated_during_preflight": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 21 preflight failed")


def run_grid(
    runner: Any,
    authorization: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    group_chunk_size: int,
) -> None:
    plan = json.loads(PLAN_PATH.read_text())
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    selected_heads = list(authorization["selected_heads"])
    population = selected_heads + list(plan["null_heads"]["members"])
    sites = site_by_id(population)
    capture_sites = tuple(sites.values())
    completed = read_completed()
    implementation_commit = git_head()
    authorization_sha256 = sha256_file(AUTHORIZATION_PATH)
    print(f"Resuming Day 21 with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)
    for concept in ("deception", "harmful"):
        concept_examples = [row for row in records if row["concept"] == concept]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        tail = CachedTailTransplantRunner(runner, probe, monitor_layer=12)
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        for batch_index, examples in enumerate(
            batched(concept_examples, batch_size), start=1
        ):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in examples],
                [row["response"] for row in examples],
                trigger,
            )
            normal, normal_inputs = run_with_full_layer_inputs(
                runner, truncated, pair.normal, capture_sites, layers=(9,)
            )
            triggered, triggered_inputs = run_with_full_layer_inputs(
                runner, truncated, pair.triggered, capture_sites, layers=(9,)
            )
            for example_index, example in enumerate(examples):
                common = {
                    "schema_version": 1,
                    "procedure": "site-shuffling-v1-day21-confirmation",
                    "implementation_commit": implementation_commit,
                    "authorization_sha256": authorization_sha256,
                    "concept": concept,
                    "example_id": example["example_id"],
                    "label": 1,
                    "split": "remaining-positive-safety-confirmation",
                    "prospective_site_shuffle_confirmation": True,
                }
                append_row(
                    completed,
                    {
                        **common,
                        "condition_id": "baseline:normal",
                        "record_type": "baseline",
                        "base_condition": "normal",
                        "probe_score": float(normal.probe_scores[example_index]),
                    },
                )
                append_row(
                    completed,
                    {
                        **common,
                        "condition_id": "baseline:correct_trigger",
                        "record_type": "baseline",
                        "base_condition": "correct_trigger",
                        "probe_score": float(triggered.probe_scores[example_index]),
                    },
                )
            for base_name, condition, cached_input in (
                ("normal", pair.normal, normal_inputs[9]),
                ("correct_trigger", pair.triggered, triggered_inputs[9]),
            ):
                specifications = confirmation_jobs(
                    base_name,
                    selected_heads,
                    sites,
                    normal.captures,
                    triggered.captures,
                    authorization["mappings"],
                )
                for chunk in batched(specifications, group_chunk_size):
                    pending = [
                        item
                        for item in chunk
                        if any(
                            (example["example_id"], item[0]["condition_id"])
                            not in completed
                            for example in examples
                        )
                    ]
                    if not pending:
                        continue
                    result = tail.run_truncated_from_layer(
                        condition,
                        [job for _specification, job in pending],
                        start_layer=9,
                        cached_input=cached_input,
                    )
                    for job_index, (specification, _job) in enumerate(pending):
                        for example_index, example in enumerate(examples):
                            append_row(
                                completed,
                                {
                                    "schema_version": 1,
                                    "procedure": "site-shuffling-v1-day21-confirmation",
                                    "implementation_commit": implementation_commit,
                                    "authorization_sha256": authorization_sha256,
                                    "concept": concept,
                                    "example_id": example["example_id"],
                                    "label": 1,
                                    "split": "remaining-positive-safety-confirmation",
                                    "prospective_site_shuffle_confirmation": True,
                                    "record_type": "intervention",
                                    **specification,
                                    "probe_score": float(
                                        result.probe_scores[job_index, example_index]
                                    ),
                                },
                            )
            print(
                f"Day 21 {concept} batch {batch_index}: {len(completed)}/{EXPECTED_ROWS}",
                flush=True,
            )
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    seal(completed)
    print(f"Sealed {len(completed)} Day 21 rows.", flush=True)


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        AUTHORIZATION_PATH,
        PLAN_PATH,
        SELECTION_PATH,
    ):
        require_committed(path, commit)
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    records = confirmation_subset(load_experimental_split("safety-test"))
    runner = load_model()
    if not args.skip_preflight:
        run_preflight(runner, authorization, records)
    if args.preflight_only:
        return
    run_grid(
        runner,
        authorization,
        records,
        batch_size=args.batch_size,
        group_chunk_size=args.group_chunk_size,
    )


if __name__ == "__main__":
    main()
