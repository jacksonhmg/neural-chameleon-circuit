#!/usr/bin/env python3
"""Run frozen trigger, class, token-order, and cross-probe shuffle diagnostics."""

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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import (  # noqa: E402
    ActivationKind,
    CachedTailTransplantRunner,
    LinearProbe,
    PatchSite,
    TruncatedComponentRunner,
    delta_mapping_job,
    load_experimental_split,
    site_shuffling_masked_rms,
)
from day17_run_transfer_atlas import ANALYSIS_PLAN_PATH, PLAN_PATH, PROBE_DIR, causal_subset, load_model, run_with_full_layer_inputs, site_by_id  # noqa: E402
from day19_run_permutation_ensemble import batched  # noqa: E402


DAY14_MAPPING_PATH = ROOT / "results/day-15/frozen-mapping-ensemble.json"
SELECTION_PATH = ROOT / "results/day-19/benign-selected-mappings.json"
DAY13_PLAN_PATH = ROOT / "results/day-13/frozen-execution-plan.json"
RESULT_DIR = ROOT / "results/day-20"
WORKING_PATH = RESULT_DIR / "specificity-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "specificity-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "specificity-preflight.json"
EXPECTED_EXAMPLES = 64
MAPPING_COUNT = 6
EXPECTED_CONDITIONS = 4 + 2 * 4 * 2 * MAPPING_COUNT
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    return parser.parse_args()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def negative_subset(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for concept in ("deception", "harmful"):
        eligible = [row for row in records if row["concept"] == concept and int(row["label"]) == 0]
        selected.extend(sorted(
            eligible,
            key=lambda row: (hashlib.sha256(f"site-shuffle-negative:{row['example_id']}".encode()).hexdigest(), row["example_id"]),
        )[:16])
    return selected


def source_conditions(runner: Any, pair: Any, prompts: Sequence[str], correct_trigger: str, irrelevant_trigger: str, templates: Mapping[str, str]) -> dict[str, Any]:
    rendered_prompts = {
        "irrelevant_concept": [
            runner.TRIGGER_TEMPLATE.format(trigger=irrelevant_trigger, prompt=prompt)
            for prompt in prompts
        ],
        "monitoring_only": [templates["monitoring_only"].format(prompt=prompt) for prompt in prompts],
        "malformed_trigger": [templates["malformed_trigger"].format(trigger=correct_trigger, prompt=prompt) for prompt in prompts],
    }
    # Rebuild custom conditions from the already tokenized response to preserve IDs exactly.
    result = {"correct_trigger": pair.triggered}
    for name, values in rendered_prompts.items():
        result[name] = runner._prepare_condition(name, values, pair.normal.response_ids, pair.normal.response_mask)
    return result


def distribution_metrics(
    mapping: Mapping[str, str],
    selected: Sequence[str],
    sites: Mapping[str, Any],
    normal_captures: Mapping[Any, Any],
    correct_captures: Mapping[Any, Any],
    source_captures: Mapping[Any, Any],
) -> dict[str, np.ndarray]:
    source_destination = []
    delta_natural = []
    patched_destination = []
    for destination_id in selected:
        source_id = mapping[destination_id]
        destination_site = sites[destination_id]
        source_site = sites[source_id]
        destination = normal_captures[destination_site]
        source = source_captures[source_site]
        source_normal = normal_captures[source_site]
        destination_delta = correct_captures[destination_site].values.float() - destination.values.float()
        source_delta = source.values.float() - source_normal.values.float()
        source_destination.append(
            site_shuffling_masked_rms(source.values.float(), source.response_mask)
            / site_shuffling_masked_rms(destination.values.float(), destination.response_mask).clamp(min=1e-12)
        )
        delta_natural.append(
            site_shuffling_masked_rms(source_delta, source.response_mask)
            / site_shuffling_masked_rms(destination_delta, destination.response_mask).clamp(min=1e-12)
        )
        patched_destination.append(
            site_shuffling_masked_rms(destination.values.float() + source_delta, destination.response_mask)
            / site_shuffling_masked_rms(destination.values.float(), destination.response_mask).clamp(min=1e-12)
        )
    return {
        "source_destination_rms_ratio": torch.stack(source_destination).mean(dim=0).numpy(),
        "delta_destination_natural_rms_ratio": torch.stack(delta_natural).mean(dim=0).numpy(),
        "patched_destination_rms_ratio": torch.stack(patched_destination).mean(dim=0).numpy(),
    }


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
            for row in sorted(completed.values(), key=lambda row: (row["concept"], row["label"], row["example_id"], row["condition_id"])):
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    templates = json.loads(DAY13_PLAN_PATH.read_text())["confounds"]["prompt_conditions"]
    day14_mappings = json.loads(DAY14_MAPPING_PATH.read_text())["day14_pilot_mappings"]
    selection = json.loads(SELECTION_PATH.read_text())
    if selection.get("status") != "pass":
        raise ValueError("benign mapping selection must pass before Day 20")
    mappings = {**{mapping_id: {"mapping_class": "day14_pilot", "destination_to_source": mapping} for mapping_id, mapping in day14_mappings.items()}, **{row["mapping_id"]: row for row in selection["selected_mappings"]}}
    if len(mappings) != MAPPING_COUNT:
        raise ValueError("Day 20 requires two pilot and four benign-selected mappings")
    records = causal_subset(load_experimental_split("safety-test")) + negative_subset(load_experimental_split("safety-test"))
    selected = list(plan["selected_heads"])
    sites = site_by_id(selected)
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    capture_sites = (*sites.values(), monitor_site)
    runner = load_model()
    completed = read_completed()
    implementation_commit = git_head()
    preflight = {"schema_version": 1, "procedure": "site-shuffling-v1-day20-preflight", "status": "pass", "mapping_count": len(mappings), "example_count": len(records), "safety_positive_count": 32, "safety_negative_count": 32}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    for concept in ("deception", "harmful"):
        concept_examples = sorted((row for row in records if row["concept"] == concept), key=lambda row: (int(row["label"]), row["example_id"]))
        target_probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        other_concept = "harmful" if concept == "deception" else "deception"
        other_probe = LinearProbe.load(PROBE_DIR / f"{other_concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, target_probe, monitor_layer=12)
        target_tail = CachedTailTransplantRunner(runner, target_probe, monitor_layer=12)
        other_tail = CachedTailTransplantRunner(runner, other_probe, monitor_layer=12)
        correct_trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        irrelevant_trigger = analysis_plan["conditions"]["irrelevant_triggers"][concept]
        for examples in batched(concept_examples, args.batch_size):
            prompts = [row["prompt"] for row in examples]
            responses = [row["response"] for row in examples]
            pair = runner.prepare_pairs(prompts, responses, correct_trigger)
            normal, normal_inputs = run_with_full_layer_inputs(runner, truncated, pair.normal, capture_sites, layers=(9,))
            correct = truncated.run(pair.triggered, capture_sites=capture_sites)
            conditions = source_conditions(runner, pair, prompts, correct_trigger, irrelevant_trigger, templates)
            source_results = {"correct_trigger": correct}
            for source_name in ("irrelevant_concept", "monitoring_only", "malformed_trigger"):
                source_results[source_name] = truncated.run(conditions[source_name], capture_sites=tuple(sites.values()))
            for example_index, example in enumerate(examples):
                for probe_concept, probe in ((concept, target_probe), (other_concept, other_probe)):
                    common = {"schema_version": 1, "procedure": "site-shuffling-v1-day20-specificity", "implementation_commit": implementation_commit, "concept": concept, "probe_concept": probe_concept, "probe_role": "target" if probe_concept == concept else "other_safety", "example_id": example["example_id"], "label": int(example["label"]), "post_confirmatory": True}
                    append_row(completed, {**common, "condition_id": f"baseline:normal:{probe_concept}", "record_type": "baseline", "base_condition": "normal", "probe_score": float(probe.score(normal.captures[monitor_site])[example_index])})
                    append_row(completed, {**common, "condition_id": f"baseline:correct_trigger:{probe_concept}", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(probe.score(correct.captures[monitor_site])[example_index])})
            for source_name, source_result in source_results.items():
                for token_control, reverse_tokens in (("aligned", False), ("reverse_valid_response_tokens", True)):
                    specifications = []
                    distributions = {}
                    for mapping_id, mapping_spec in mappings.items():
                        mapping = mapping_spec["destination_to_source"]
                        condition_id = f"delta:{source_name}:{token_control}:{mapping_id}"
                        specifications.append((mapping_id, delta_mapping_job(condition_id, selected, mapping, sites, normal.captures, normal.captures, source_result.captures, alpha=1.0, sign=1.0, reverse_tokens=reverse_tokens)))
                        distributions[mapping_id] = distribution_metrics(mapping, selected, sites, normal.captures, correct.captures, source_result.captures)
                    for probe_concept, tail in ((concept, target_tail), (other_concept, other_tail)):
                        for chunk in batched(specifications, args.group_chunk_size):
                            result = tail.run_truncated_from_layer(pair.normal, [job for _mapping_id, job in chunk], start_layer=9, cached_input=normal_inputs[9])
                            for job_index, (mapping_id, _job) in enumerate(chunk):
                                for example_index, example in enumerate(examples):
                                    metrics = distributions[mapping_id]
                                    condition_id = f"delta:{source_name}:{token_control}:{mapping_id}:{probe_concept}"
                                    append_row(completed, {"schema_version": 1, "procedure": "site-shuffling-v1-day20-specificity", "implementation_commit": implementation_commit, "concept": concept, "probe_concept": probe_concept, "probe_role": "target" if probe_concept == concept else "other_safety", "example_id": example["example_id"], "label": int(example["label"]), "post_confirmatory": True, "condition_id": condition_id, "record_type": "intervention", "source_condition": source_name, "token_control": token_control, "mapping_id": mapping_id, "mapping_class": mappings[mapping_id].get("mapping_class", "benign_selected"), "probe_score": float(result.probe_scores[job_index, example_index]), "source_destination_rms_ratio": float(metrics["source_destination_rms_ratio"][example_index]), "delta_destination_natural_rms_ratio": float(metrics["delta_destination_natural_rms_ratio"][example_index]), "patched_destination_rms_ratio": float(metrics["patched_destination_rms_ratio"][example_index])})
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    seal(completed)


if __name__ == "__main__":
    main()
