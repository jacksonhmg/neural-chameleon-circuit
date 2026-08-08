#!/usr/bin/env python3
"""Run frozen K1/K2/K4/K8/K12 composition for benign-selected mappings."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import CachedTailTransplantRunner, LinearProbe, TruncatedComponentRunner, delta_mapping_job, load_experimental_split  # noqa: E402
from day17_run_transfer_atlas import ANALYSIS_PLAN_PATH, PLAN_PATH, PROBE_DIR, load_model, run_with_full_layer_inputs, site_by_id  # noqa: E402
from day19_run_permutation_ensemble import batched, hash_subset  # noqa: E402


RESULT_DIR = ROOT / "results/day-19"
SELECTION_PATH = RESULT_DIR / "benign-selected-mappings.json"
WORKING_PATH = RESULT_DIR / "composition-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "composition-results.jsonl.gz"
EXPECTED_EXAMPLES = 176
EXPECTED_CONDITIONS = 2 + 2 * 4 * 5
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    result = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            result[(row["example_id"], row["condition_id"])] = row
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
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in sorted(completed.values(), key=lambda row: (row["split_role"], row["concept"], row["example_id"], row["condition_id"])):
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def main() -> None:
    args = parse_args()
    selection = json.loads(SELECTION_PATH.read_text())
    if selection.get("status") != "pass" or len(selection["selected_mappings"]) != 4:
        raise ValueError("benign mapping selection is not complete")
    plan = json.loads(PLAN_PATH.read_text())
    selected_heads = sorted(plan["selected_heads"])
    sites = site_by_id(selected_heads)
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    roles = [("development", hash_subset(load_experimental_split("discovery"), "site-shuffle-development")), ("validation", hash_subset(load_experimental_split("validation"), "site-shuffle-validation"))]
    runner = load_model()
    completed = read_completed()
    implementation_commit = git_head()
    for split_role, records in roles:
        for concept in sorted({row["concept"] for row in records}):
            concept_examples = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
            probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
            truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
            tail = CachedTailTransplantRunner(runner, probe, monitor_layer=12)
            for examples in batched(concept_examples, args.batch_size):
                pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], analysis_plan["conditions"]["correct_triggers"][concept])
                normal, normal_inputs = run_with_full_layer_inputs(runner, truncated, pair.normal, tuple(sites.values()), layers=(9,))
                triggered, triggered_inputs = run_with_full_layer_inputs(runner, truncated, pair.triggered, tuple(sites.values()), layers=(9,))
                for index, example in enumerate(examples):
                    common = {"schema_version": 1, "procedure": "site-shuffling-v1-day19-composition", "implementation_commit": implementation_commit, "split_role": split_role, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True}
                    append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "probe_score": float(normal.probe_scores[index])})
                    append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "probe_score": float(triggered.probe_scores[index])})
                for base_name, condition, cached_input, base_captures, sign, direction in (("normal", pair.normal, normal_inputs[9], normal.captures, 1.0, "induction"), ("correct_trigger", pair.triggered, triggered_inputs[9], triggered.captures, -1.0, "rescue")):
                    specifications = []
                    for mapping in selection["selected_mappings"]:
                        for size in (1, 2, 4, 8, 12):
                            destinations = selected_heads[:size]
                            subset_mapping = {destination: mapping["destination_to_source"][destination] for destination in destinations}
                            condition_id = f"delta:{base_name}:{mapping['mapping_id']}:k{size}"
                            specifications.append(({"condition_id": condition_id, "mapping_id": mapping["mapping_id"], "mapping_class": mapping["mapping_class"], "size": size, "direction": direction, "base_condition": base_name}, delta_mapping_job(condition_id, destinations, subset_mapping, sites, base_captures, normal.captures, triggered.captures, alpha=1.0, sign=sign)))
                    for chunk in batched(specifications, args.group_chunk_size):
                        result = tail.run_truncated_from_layer(condition, [job for _spec, job in chunk], start_layer=9, cached_input=cached_input)
                        for job_index, (specification, _job) in enumerate(chunk):
                            for index, example in enumerate(examples):
                                append_row(completed, {"schema_version": 1, "procedure": "site-shuffling-v1-day19-composition", "implementation_commit": implementation_commit, "split_role": split_role, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True, "record_type": "intervention", **specification, "probe_score": float(result.probe_scores[job_index, index])})
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    seal(completed)


if __name__ == "__main__":
    main()
