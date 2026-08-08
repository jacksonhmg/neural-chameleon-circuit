#!/usr/bin/env python3
"""Run the frozen benign within- and cross-layer permutation ensembles."""

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


MAPPING_PATH = ROOT / "results/day-15/frozen-mapping-ensemble.json"
RESULT_DIR = ROOT / "results/day-19"
WORKING_PATH = RESULT_DIR / "permutation-ensemble-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "permutation-ensemble-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "permutation-ensemble-preflight.json"
EXPECTED_EXAMPLES = 176
EXPECTED_CONDITIONS = 2 + 2 * 64
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def hash_subset(records: Sequence[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    selected = []
    for concept in sorted({row["concept"] for row in records}):
        eligible = [row for row in records if row["concept"] == concept and int(row["label"]) == 1]
        selected.extend(sorted(
            eligible,
            key=lambda row: (hashlib.sha256(f"{prefix}:{row['example_id']}".encode()).hexdigest(), row["example_id"]),
        )[:16])
    return selected


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    result = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in result:
                raise ValueError(f"duplicate row {key}")
            result[key] = row
    return result


def append_row(completed: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
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
    rows = sorted(completed.values(), key=lambda row: (row["split_role"], row["concept"], row["example_id"], row["condition_id"]))
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def ensemble_jobs(
    base_name: str,
    selected: Sequence[str],
    sites: Mapping[str, Any],
    normal_captures: Mapping[Any, Any],
    triggered_captures: Mapping[Any, Any],
    mappings: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], Any]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    sign = 1.0 if base_name == "normal" else -1.0
    direction = "induction" if base_name == "normal" else "rescue"
    jobs = []
    for mapping_spec in mappings:
        mapping_id = mapping_spec["mapping_id"]
        condition_id = f"delta:{base_name}:{mapping_id}"
        jobs.append(({
            "condition_id": condition_id,
            "base_condition": base_name,
            "direction": direction,
            "mapping_id": mapping_id,
            "mapping_class": mapping_spec["mapping_class"],
        }, delta_mapping_job(
            condition_id,
            selected,
            mapping_spec["destination_to_source"],
            sites,
            base_captures,
            normal_captures,
            triggered_captures,
            alpha=1.0,
            sign=sign,
        )))
    return jobs


def run_preflight(runner: Any, plan: Mapping[str, Any], mappings: Sequence[Mapping[str, Any]]) -> None:
    checks = {
        "mapping_count": len(mappings) == 64,
        "mapping_ids_unique": len({row["mapping_id"] for row in mappings}) == 64,
        "class_counts": {key: sum(row["mapping_class"] == key for row in mappings) for key in ("within_layer", "cross_layer")} == {"within_layer": 32, "cross_layer": 32},
        "all_bijections": all(set(row["destination_to_source"]) == set(plan["selected_heads"]) and set(row["destination_to_source"].values()) == set(plan["selected_heads"]) for row in mappings),
        "registered_hook_count": runner.registered_hook_count() == 0,
    }
    report = {"schema_version": 1, "procedure": "site-shuffling-v1-day19-preflight", "status": "pass" if all(checks.values()) else "fail", "checks": checks, "safety_split_accessed": False}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 19 preflight failed")


def run_grid(runner: Any, plan: Mapping[str, Any], mappings: Sequence[Mapping[str, Any]], *, batch_size: int, group_chunk_size: int) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    roles = [
        ("development", hash_subset(load_experimental_split("discovery"), "site-shuffle-development")),
        ("validation", hash_subset(load_experimental_split("validation"), "site-shuffle-validation")),
    ]
    if sum(len(records) for _role, records in roles) != EXPECTED_EXAMPLES:
        raise ValueError("Day 19 benign subset count changed")
    selected = list(plan["selected_heads"])
    sites = site_by_id(selected)
    capture_sites = tuple(sites.values())
    completed = read_completed()
    implementation_commit = git_head()
    print(f"Resuming Day 19 with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)
    for split_role, records in roles:
        for concept in sorted({row["concept"] for row in records}):
            concept_examples = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
            probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
            truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
            tail = CachedTailTransplantRunner(runner, probe, monitor_layer=12)
            trigger = analysis_plan["conditions"]["correct_triggers"][concept]
            for batch_index, examples in enumerate(batched(concept_examples, batch_size), start=1):
                pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], trigger)
                normal, normal_inputs = run_with_full_layer_inputs(runner, truncated, pair.normal, capture_sites, layers=(9,))
                triggered, triggered_inputs = run_with_full_layer_inputs(runner, truncated, pair.triggered, capture_sites, layers=(9,))
                for example_index, example in enumerate(examples):
                    common = {"schema_version": 1, "procedure": "site-shuffling-v1-day19", "implementation_commit": implementation_commit, "split_role": split_role, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True}
                    append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "base_condition": "normal", "probe_score": float(normal.probe_scores[example_index])})
                    append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(triggered.probe_scores[example_index])})
                for base_name, condition, cached_input in (("normal", pair.normal, normal_inputs[9]), ("correct_trigger", pair.triggered, triggered_inputs[9])):
                    specifications = ensemble_jobs(base_name, selected, sites, normal.captures, triggered.captures, mappings)
                    for chunk in batched(specifications, group_chunk_size):
                        result = tail.run_truncated_from_layer(condition, [job for _spec, job in chunk], start_layer=9, cached_input=cached_input)
                        for job_index, (specification, _job) in enumerate(chunk):
                            for example_index, example in enumerate(examples):
                                append_row(completed, {"schema_version": 1, "procedure": "site-shuffling-v1-day19", "implementation_commit": implementation_commit, "split_role": split_role, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True, "record_type": "intervention", **specification, "probe_score": float(result.probe_scores[job_index, example_index])})
                print(f"Day 19 {split_role} {concept} batch {batch_index}: {len(completed)}/{EXPECTED_ROWS}", flush=True)
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    seal(completed)


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    mapping_data = json.loads(MAPPING_PATH.read_text())
    mappings = mapping_data["ensemble"]
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(PLAN_PATH, commit)
    require_committed(MAPPING_PATH, commit)
    runner = load_model()
    if not args.skip_preflight:
        run_preflight(runner, plan, mappings)
    if args.preflight_only:
        return
    run_grid(runner, plan, mappings, batch_size=args.batch_size, group_chunk_size=args.group_chunk_size)


if __name__ == "__main__":
    main()
