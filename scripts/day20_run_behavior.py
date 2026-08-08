#!/usr/bin/env python3
"""Run frozen full-model NLL/KL diagnostics for key shuffled interventions."""

from __future__ import annotations

import argparse
import gzip
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

from neural_chameleon import (  # noqa: E402
    ActivationKind,
    LinearProbe,
    PatchSite,
    TruncatedComponentRunner,
    VectorizedTransplantRunner,
    absolute_mapping_job,
    delta_mapping_job,
    load_experimental_split,
    masked_token_kl,
)
from day17_run_transfer_atlas import ANALYSIS_PLAN_PATH, PLAN_PATH, PROBE_DIR, causal_subset, load_model, site_by_id  # noqa: E402


MAPPING_PATH = ROOT / "results/day-15/frozen-mapping-ensemble.json"
RESULT_DIR = ROOT / "results/day-20"
WORKING_PATH = RESULT_DIR / "behavior-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "behavior-results.jsonl.gz"
EXPECTED_ROWS = 32 * 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def response_nll(result: Any) -> torch.Tensor:
    logprobs = result.response_token_logprobs()
    mask = result.response_mask
    return (-logprobs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


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


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    pilot_mappings = json.loads(MAPPING_PATH.read_text())["day14_pilot_mappings"]
    selected = list(plan["selected_heads"])
    sites = site_by_id(selected)
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    identity = {head_id: head_id for head_id in selected}
    records = causal_subset(load_experimental_split("safety-test"))
    runner = load_model()
    completed = read_completed()
    implementation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for concept in ("deception", "harmful"):
        examples_for_concept = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        for start in range(0, len(examples_for_concept), args.batch_size):
            examples = examples_for_concept[start : start + args.batch_size]
            pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], analysis_plan["conditions"]["correct_triggers"][concept])
            normal = runner.run(pair.normal, capture_sites=(*sites.values(), monitor_site), retain_response_logits=True)
            triggered = runner.run(pair.triggered, capture_sites=(*sites.values(), monitor_site), retain_response_logits=True)
            normal_nll = response_nll(normal)
            triggered_nll = response_nll(triggered)
            if normal.response_logits is None or triggered.response_logits is None:
                raise RuntimeError("baseline logits missing")
            triggered_kl = masked_token_kl(normal.response_logits, triggered.response_logits, pair.normal.response_mask, device=runner.device)
            jobs = [
                absolute_mapping_job("selected_k12_identity", selected, identity, sites, triggered.captures),
                absolute_mapping_job("day14_seed_211_raw", selected, pilot_mappings["day14_seed_211_heads"], sites, triggered.captures),
                absolute_mapping_job("day14_seed_223_raw", selected, pilot_mappings["day14_seed_223_heads"], sites, triggered.captures),
                delta_mapping_job("day14_seed_211_delta", selected, pilot_mappings["day14_seed_211_heads"], sites, normal.captures, normal.captures, triggered.captures, alpha=1.0, sign=1.0),
                delta_mapping_job("day14_seed_223_delta", selected, pilot_mappings["day14_seed_223_heads"], sites, normal.captures, normal.captures, triggered.captures, alpha=1.0, sign=1.0),
            ]
            diagnostics = vector.run_full_diagnostics(pair.normal, jobs)
            for index, example in enumerate(examples):
                common = {"schema_version": 1, "procedure": "site-shuffling-v1-day20-behavior", "implementation_commit": implementation_commit, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True}
                append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "probe_score": float(probe.score(normal.captures[monitor_site])[index]), "response_nll": float(normal_nll[index]), "kl_from_normal": 0.0})
                append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "probe_score": float(probe.score(triggered.captures[monitor_site])[index]), "response_nll": float(triggered_nll[index]), "kl_from_normal": float(triggered_kl[index])})
            for job_index, job in enumerate(jobs):
                kl = masked_token_kl(normal.response_logits, diagnostics.response_logits[job_index], pair.normal.response_mask, device=runner.device)
                for index, example in enumerate(examples):
                    append_row(completed, {"schema_version": 1, "procedure": "site-shuffling-v1-day20-behavior", "implementation_commit": implementation_commit, "concept": concept, "example_id": example["example_id"], "label": 1, "post_confirmatory": True, "condition_id": job.group_id, "record_type": "intervention", "probe_score": float(diagnostics.probe_scores[job_index, index]), "response_nll": float(diagnostics.response_nll[job_index, index]), "kl_from_normal": float(kl[index])})
    if len(completed) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} behavior rows, found {len(completed)}")
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in sorted(completed.values(), key=lambda row: (row["concept"], row["example_id"], row["condition_id"])):
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


if __name__ == "__main__":
    main()
