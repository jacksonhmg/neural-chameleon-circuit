#!/usr/bin/env python3
"""Run frozen Phase B attention stages in fresh, ordered child processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import (  # noqa: E402
    ATTENTION_EVALUATION_CORRECTION_V9_PATH,
    ATTENTION_PATH,
    CONTRACT_PATH,
    OPERATIONS,
    PARAMETERS_PATH,
    SELECTION_PATH,
    attention_population_batches,
    attention_sites,
    git_head,
    load_jsonl,
    load_records,
    read_json,
    require_committed,
)


RUNNER_PATH = ROOT / "scripts/day37_run_phase_b.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("attention-discovery", "attention-eval"), required=True
    )
    return parser.parse_args()


def stage_population(
    stage: str, records: list[dict[str, Any]], contract: dict[str, Any]
) -> tuple[list[dict[str, Any]], tuple[tuple[str, tuple[str, ...]], ...], str]:
    if stage == "attention-discovery":
        population = [
            row
            for row in records
            if row["split"] == "discovery" and int(row["label"]) == 1
        ]
        sites = attention_sites(contract["component_sets"]["layer_groups"])
        return population, sites, "discovery"
    selection = read_json(SELECTION_PATH)
    population = [
        row
        for row in records
        if int(row["label"]) == 0
        or (row["split"] != "discovery" and int(row["label"]) == 1)
    ]
    site_id = selection["selected_attention_site"]["site_id"]
    site = next(
        value
        for value in attention_sites(contract["component_sets"]["layer_groups"])
        if value[0] == site_id
    )
    return population, (site,), "heldout_or_negative"


def completion_counts() -> Counter[tuple[str, str, str]]:
    return Counter(
        (row["model"], row["evaluation_scope"], row["example_id"])
        for row in load_jsonl(ATTENTION_PATH)
    )


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        RUNNER_PATH,
        ATTENTION_EVALUATION_CORRECTION_V9_PATH,
        SELECTION_PATH,
    ):
        require_committed(path, commit)
    correction = read_json(ATTENTION_EVALUATION_CORRECTION_V9_PATH)
    if correction["status"] != "frozen-before-v9-attention-evaluation-outcomes":
        raise RuntimeError("V9 process-isolation contract is not frozen")
    if not PARAMETERS_PATH.exists():
        raise RuntimeError("initialize V9 execution parameters before attention shards")
    parameters = read_json(PARAMETERS_PATH)
    if parameters.get("correction_runtime_commit") not in {
        commit,
        correction["superseded_v8_evaluation_attempt"]["runtime_commit"],
    }:
        raise RuntimeError("attention shard driver differs from the bound runtime")
    contract = read_json(CONTRACT_PATH)
    records = load_records()
    population, sites, scope = stage_population(args.stage, records, contract)
    batches = attention_population_batches(population, batch_size=2)
    shard_field = (
        "discovery_process_shard_batch_count"
        if args.stage == "attention-discovery"
        else "evaluation_process_shard_batch_count"
    )
    shard_count = int(correction["correction"][shard_field])
    expected_per_example = len(sites) * len(OPERATIONS) * 2 * 2
    models = ("chameleon",) if args.stage == "attention-discovery" else (
        "chameleon",
        "precursor",
    )
    environment = os.environ.copy()
    environment["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = correction["correction"][
        "mps_high_watermark_ratio"
    ]
    for model in models:
        for start in range(0, len(batches), shard_count):
            shard = batches[start : start + shard_count]
            counts = completion_counts()
            if all(
                counts[(model, scope, row["example_id"])] == expected_per_example
                for batch in shard
                for row in batch
            ):
                continue
            command = (
                sys.executable,
                str(RUNNER_PATH),
                "--stage",
                args.stage,
                "--models",
                model,
                "--batch-size",
                "2",
                "--job-chunk-size",
                "32",
                "--attention-job-chunk-size",
                "32",
                "--attention-metadata-block-size",
                "32",
                "--attention-shard-start",
                str(start),
                "--attention-shard-count",
                str(shard_count),
            )
            subprocess.run(command, cwd=ROOT, env=environment, check=True)
            counts = completion_counts()
            if not all(
                counts[(model, scope, row["example_id"])] == expected_per_example
                for batch in shard
                for row in batch
            ):
                raise RuntimeError("attention child did not complete its exact shard")
            print(
                json.dumps(
                    {
                        "model": model,
                        "scope": scope,
                        "completed_batches": min(start + shard_count, len(batches)),
                        "total_batches": len(batches),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
