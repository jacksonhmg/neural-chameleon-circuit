#!/usr/bin/env python3
"""Run the frozen Day 14 analysis-only falsification suite."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import load_experimental_split  # noqa: E402
from neural_chameleon.safety_transfer import summarize_safety_transfer  # noqa: E402


RESULT_DIR = ROOT / "results/day-14"
PLAN_PATH = RESULT_DIR / "frozen-falsification-plan.json"
CENTRAL_PATH = ROOT / "results/day-13/safety-transfer-results.jsonl.gz"
CONFOUND_SUMMARY_PATH = ROOT / "results/day-13/confound-summary.json"
COMPONENT_PATH = ROOT / "results/day-12/frozen-final-component-set.json"
DISCOVERY_PATH = ROOT / "results/day-08/discovery-candidate-results.jsonl.gz"
OUTPUT_PATH = RESULT_DIR / "analysis-only-summary.json"
OUTLIER_CSV = RESULT_DIR / "outlier-robustness.csv"
THRESHOLD_CSV = RESULT_DIR / "threshold-robustness.csv"
SEED_CSV = RESULT_DIR / "bootstrap-seed-robustness.csv"
LEAKAGE_PATH = RESULT_DIR / "leakage-audit.json"
MULTIPLICITY_PATH = RESULT_DIR / "multiple-comparison-audit.json"
CONCEPTS = ("deception", "harmful")
DIRECTIONS = ("rescue", "induction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


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


def index_central(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        condition = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
        key = (row["example_id"], condition)
        if key in indexed:
            raise ValueError(f"duplicate central key {key}")
        indexed[key] = row
    return indexed


def ids_for(
    index: Mapping[tuple[str, str], dict[str, Any]], concept: str, label: int
) -> list[str]:
    return sorted(
        example_id
        for example_id, condition in index
        if condition == "normal"
        and index[(example_id, "normal")]["concept"] == concept
        and int(index[(example_id, "normal")]["label"]) == label
    )


def arrays_for(
    index: Mapping[tuple[str, str], dict[str, Any]],
    concept: str,
    label: int,
    field: str = "probe_score",
) -> tuple[list[str], dict[str, np.ndarray]]:
    ids = ids_for(index, concept, label)
    conditions = (
        "normal",
        "correct_trigger",
        "selected_k16:rescue",
        "selected_k16:induction",
        "random_k16:rescue",
        "random_k16:induction",
    )
    return ids, {
        condition: np.asarray(
            [index[(example_id, condition)][field] for example_id in ids], dtype=float
        )
        for condition in conditions
    }


def numerator(arrays: Mapping[str, np.ndarray], group: str, direction: str) -> np.ndarray:
    patched = arrays[f"{group}:{direction}"]
    if direction == "rescue":
        return patched - arrays["correct_trigger"]
    return arrays["normal"] - patched


def denominator(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    return arrays["normal"] - arrays["correct_trigger"]


def ratio(values: np.ndarray, gaps: np.ndarray, keep: np.ndarray | None = None) -> float:
    if keep is not None:
        values = values[keep]
        gaps = gaps[keep]
    denominator_mean = float(gaps.mean())
    if denominator_mean <= 0:
        raise ValueError("non-positive suppression denominator")
    return float(values.mean() / denominator_mean)


def trimmed_mean(values: np.ndarray, fraction: float) -> float:
    count = int(math.floor(len(values) * fraction))
    ordered = np.sort(values)
    if 2 * count >= len(values):
        raise ValueError("trim fraction removes every value")
    return float(ordered[count : len(values) - count].mean())


def median_of_means(
    example_ids: Sequence[str], values: np.ndarray, gaps: np.ndarray, concept: str, groups: int
) -> float:
    order = sorted(
        range(len(example_ids)),
        key=lambda index: hashlib.sha256(
            f"day14-mom:{concept}:{example_ids[index]}".encode()
        ).hexdigest(),
    )
    buckets = [[] for _ in range(groups)]
    for rank, index in enumerate(order):
        buckets[rank % groups].append(index)
    estimates = []
    for bucket in buckets:
        selected = np.asarray(bucket, dtype=int)
        estimates.append(ratio(values[selected], gaps[selected]))
    return float(np.median(estimates))


def run_bootstrap_seeds(
    central: Sequence[dict[str, Any]],
    component: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    rows = []
    support_by_seed = {}
    for seed in plan["analysis_only_falsifications"]["bootstrap"]["seeds"]:
        summary = summarize_safety_transfer(
            central,
            selected_candidates=component["selected_candidates"],
            random_candidates=component["random_control_candidates"],
            replicates=plan["analysis_only_falsifications"]["bootstrap"]["replicates"],
            seed=seed,
        )
        support_by_seed[str(seed)] = summary["safety_transfer_supported_by_concept"]
        for concept in summary["concepts"]:
            lookup = {
                (cell["group_id"], cell["direction"], cell["label"]): cell
                for cell in concept["cells"]
            }
            contrasts = {
                row["direction"]: row["fraction_difference"]
                for row in concept["selected_random_contrasts"]
            }
            for direction in DIRECTIONS:
                selected = lookup[("selected_k16", direction, 1)]["fraction"]
                contrast = contrasts[direction]
                rows.append(
                    {
                        "seed": seed,
                        "concept": concept["concept"],
                        "direction": direction,
                        "selected_estimate": selected["estimate"],
                        "selected_ci_low": selected["ci_low"],
                        "selected_ci_high": selected["ci_high"],
                        "contrast_estimate": contrast["estimate"],
                        "contrast_ci_low": contrast["ci_low"],
                        "contrast_ci_high": contrast["ci_high"],
                        "concept_support": concept["safety_transfer_supported"],
                    }
                )
    with SEED_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "seeds": support_by_seed,
        "all_concepts_supported_all_seeds": all(
            all(concepts.values()) for concepts in support_by_seed.values()
        ),
        "csv_sha256": sha256_file(SEED_CSV),
    }


def run_outlier_analysis(
    index: Mapping[tuple[str, str], dict[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    gate_values = []
    settings = plan["analysis_only_falsifications"]["outlier_dependence"]
    summaries = []
    for concept in CONCEPTS:
        example_ids, arrays = arrays_for(index, concept, 1)
        gaps = denominator(arrays)
        for direction in DIRECTIONS:
            selected_values = numerator(arrays, "selected_k16", direction)
            random_values = numerator(arrays, "random_k16", direction)
            estimands = {
                "selected": selected_values,
                "selected_minus_random": selected_values - random_values,
            }
            for estimand, values in estimands.items():
                full = ratio(values, gaps)
                loo = []
                for deleted in range(len(example_ids)):
                    keep = np.ones(len(example_ids), dtype=bool)
                    keep[deleted] = False
                    loo.append(ratio(values, gaps, keep))
                rows.append(
                    {
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "method": "full",
                        "fraction": 0,
                        "estimate": full,
                        "minimum": full,
                        "maximum": full,
                        "removed_examples": 0,
                    }
                )
                rows.append(
                    {
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "method": "leave_one_out",
                        "fraction": 1 / len(example_ids),
                        "estimate": float(np.mean(loo)),
                        "minimum": float(np.min(loo)),
                        "maximum": float(np.max(loo)),
                        "removed_examples": 1,
                    }
                )
                for fraction in settings["worst_case_deletion_fractions"]:
                    count = int(math.ceil(len(example_ids) * fraction))
                    remove = np.argsort(values)[-count:]
                    keep = np.ones(len(example_ids), dtype=bool)
                    keep[remove] = False
                    estimate = ratio(values, gaps, keep)
                    rows.append(
                        {
                            "concept": concept,
                            "direction": direction,
                            "estimand": estimand,
                            "method": "worst_case_favorable_deletion",
                            "fraction": fraction,
                            "estimate": estimate,
                            "minimum": estimate,
                            "maximum": estimate,
                            "removed_examples": count,
                        }
                    )
                    if fraction == 0.2:
                        gate_values.append(estimate)
                for fraction in settings["trimmed_mean_fractions"]:
                    estimate = trimmed_mean(values, fraction) / trimmed_mean(gaps, fraction)
                    rows.append(
                        {
                            "concept": concept,
                            "direction": direction,
                            "estimand": estimand,
                            "method": "symmetric_trimmed_mean",
                            "fraction": fraction,
                            "estimate": estimate,
                            "minimum": estimate,
                            "maximum": estimate,
                            "removed_examples": 2 * int(math.floor(len(values) * fraction)),
                        }
                    )
                mom = median_of_means(
                    example_ids,
                    values,
                    gaps,
                    concept,
                    settings["median_of_means_groups"],
                )
                rows.append(
                    {
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "method": "median_of_means",
                        "fraction": 0,
                        "estimate": mom,
                        "minimum": mom,
                        "maximum": mom,
                        "removed_examples": 0,
                    }
                )
                summaries.append(
                    {
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "full": full,
                        "leave_one_out_minimum": float(np.min(loo)),
                        "leave_one_out_maximum": float(np.max(loo)),
                        "median_of_means": mom,
                    }
                )
                gate_values.append(float(np.min(loo)))
    with OUTLIER_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "summaries": summaries,
        "robustness_gate_passed": all(value > 0 for value in gate_values),
        "minimum_gate_value": min(gate_values),
        "csv_sha256": sha256_file(OUTLIER_CSV),
    }


def higher_quantile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), quantile, method="higher"))


def rate(values: np.ndarray, threshold: float) -> float:
    return float(np.mean(values > threshold))


def run_threshold_analysis(
    index: Mapping[tuple[str, str], dict[str, Any]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    concept_summaries = []
    threshold_plan = plan["analysis_only_falsifications"]["thresholds"]
    for concept in CONCEPTS:
        _positive_ids, positive = arrays_for(index, concept, 1)
        _negative_ids, negative = arrays_for(index, concept, 0)
        thresholds = [
            ("absolute", value, float(value))
            for value in threshold_plan["absolute_probe_thresholds"]
        ]
        for target in threshold_plan["normal_negative_fpr_targets"]:
            thresholds.append(
                (
                    "normal_negative_fpr",
                    target,
                    higher_quantile(negative["normal"], 1 - target),
                )
            )
        valid_direction_tests = 0
        intended_direction_tests = 0
        for threshold_type, target, threshold in thresholds:
            condition_rates = {}
            for condition in positive:
                condition_rates[condition] = {
                    "tpr": rate(positive[condition], threshold),
                    "fpr": rate(negative[condition], threshold),
                }
            gap = condition_rates["normal"]["tpr"] - condition_rates["correct_trigger"]["tpr"]
            causal = {}
            for group in ("selected_k16", "random_k16"):
                for direction in DIRECTIONS:
                    patched = condition_rates[f"{group}:{direction}"]["tpr"]
                    if gap > 0:
                        value = (
                            patched - condition_rates["correct_trigger"]["tpr"]
                            if direction == "rescue"
                            else condition_rates["normal"]["tpr"] - patched
                        ) / gap
                    else:
                        value = None
                    causal[(group, direction)] = value
            for direction in DIRECTIONS:
                selected = causal[("selected_k16", direction)]
                random = causal[("random_k16", direction)]
                if selected is not None and random is not None:
                    valid_direction_tests += 1
                    intended_direction_tests += int(selected > 0 and selected > random)
            rows.append(
                {
                    "concept": concept,
                    "threshold_type": threshold_type,
                    "target": target,
                    "threshold": threshold,
                    "normal_tpr": condition_rates["normal"]["tpr"],
                    "normal_fpr": condition_rates["normal"]["fpr"],
                    "triggered_tpr": condition_rates["correct_trigger"]["tpr"],
                    "triggered_fpr": condition_rates["correct_trigger"]["fpr"],
                    "selected_rescue_tpr": condition_rates["selected_k16:rescue"]["tpr"],
                    "selected_induction_tpr": condition_rates["selected_k16:induction"]["tpr"],
                    "random_rescue_tpr": condition_rates["random_k16:rescue"]["tpr"],
                    "random_induction_tpr": condition_rates["random_k16:induction"]["tpr"],
                    "selected_rescue_fraction": causal[("selected_k16", "rescue")],
                    "selected_induction_fraction": causal[("selected_k16", "induction")],
                    "random_rescue_fraction": causal[("random_k16", "rescue")],
                    "random_induction_fraction": causal[("random_k16", "induction")],
                }
            )
        concept_summaries.append(
            {
                "concept": concept,
                "valid_direction_tests": valid_direction_tests,
                "selected_positive_and_above_random": intended_direction_tests,
                "fraction_in_intended_direction": intended_direction_tests / valid_direction_tests,
            }
        )
    with THRESHOLD_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "concepts": concept_summaries,
        "all_valid_threshold_tests_intended_direction": all(
            row["valid_direction_tests"] == row["selected_positive_and_above_random"]
            for row in concept_summaries
        ),
        "csv_sha256": sha256_file(THRESHOLD_CSV),
    }


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split()).strip()


def word_ngrams(record: Mapping[str, Any], size: int = 5) -> set[tuple[str, ...]]:
    normalized = normalize_text(f"{record['prompt']} {record['response']}")
    tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def cross_phase_duplicates(
    phases: Mapping[str, Sequence[dict[str, Any]]], key_function
) -> list[dict[str, Any]]:
    ownership: dict[Any, list[tuple[str, str]]] = defaultdict(list)
    for phase, records in phases.items():
        for record in records:
            ownership[key_function(record)].append((phase, record["example_id"]))
    duplicates = []
    for key, owners in ownership.items():
        if len({phase for phase, _example_id in owners}) > 1:
            duplicates.append({"key": str(key), "owners": owners})
    return sorted(duplicates, key=lambda row: row["key"])


def near_duplicates_between(
    left_name: str,
    left: Sequence[dict[str, Any]],
    right_name: str,
    right: Sequence[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    right_sets = {record["example_id"]: word_ngrams(record) for record in right}
    postings: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for example_id, grams in right_sets.items():
        for gram in grams:
            postings[gram].append(example_id)
    matches = []
    for record in left:
        left_set = word_ngrams(record)
        intersections: Counter[str] = Counter()
        for gram in left_set:
            intersections.update(postings.get(gram, ()))
        for right_id, intersection in intersections.items():
            union = len(left_set) + len(right_sets[right_id]) - intersection
            score = intersection / union if union else 1.0
            if score >= threshold:
                matches.append(
                    {
                        "left_phase": left_name,
                        "left_example_id": record["example_id"],
                        "right_phase": right_name,
                        "right_example_id": right_id,
                        "word_fivegram_jaccard": score,
                    }
                )
    return sorted(
        matches,
        key=lambda row: (-row["word_fivegram_jaccard"], row["left_example_id"], row["right_example_id"]),
    )


def run_leakage_audit(plan: Mapping[str, Any]) -> dict[str, Any]:
    phases = {
        "discovery": load_experimental_split("discovery"),
        "validation": load_experimental_split("validation"),
        "safety-test": load_experimental_split("safety-test"),
    }
    leakage_plan = plan["analysis_only_falsifications"]["data_leakage"]
    exact = {
        "example_id": cross_phase_duplicates(phases, lambda row: row["example_id"]),
        "content_sha256": cross_phase_duplicates(phases, lambda row: row["content_sha256"]),
        "normalized_prompt_response_sha256": cross_phase_duplicates(
            phases,
            lambda row: hashlib.sha256(
                normalize_text(f"{row['prompt']}\n{row['response']}").encode()
            ).hexdigest(),
        ),
        "source_path_and_record_index": cross_phase_duplicates(
            phases,
            lambda row: (row["source"]["path"], row["source"]["record_index"]),
        ),
    }
    near = []
    phase_names = list(phases)
    for left_index, left_name in enumerate(phase_names):
        for right_name in phase_names[left_index + 1 :]:
            near.extend(
                near_duplicates_between(
                    left_name,
                    phases[left_name],
                    right_name,
                    phases[right_name],
                    leakage_plan["near_duplicate_review_threshold"],
                )
            )
    concepts = {
        phase: sorted({record["concept"] for record in records})
        for phase, records in phases.items()
    }
    trigger_occurrence = {}
    for phase, records in phases.items():
        trigger_occurrence[phase] = {
            "prompt": sum(
                normalize_text(record["trigger_concept"]) in normalize_text(record["prompt"])
                for record in records
            ),
            "response": sum(
                normalize_text(record["trigger_concept"]) in normalize_text(record["response"])
                for record in records
            ),
            "records": len(records),
        }
    major_failure = any(exact.values())
    report = {
        "schema_version": 1,
        "procedure": "day14-leakage-audit-v1",
        "status": "fail" if major_failure else "pass",
        "phase_counts": {phase: len(records) for phase, records in phases.items()},
        "concepts": concepts,
        "concept_sets_disjoint": len(set(concepts["discovery"]) & set(concepts["validation"])) == 0
        and len(set(concepts["discovery"]) & set(concepts["safety-test"])) == 0
        and len(set(concepts["validation"]) & set(concepts["safety-test"])) == 0,
        "exact_cross_phase_duplicates": exact,
        "major_exact_leakage_detected": major_failure,
        "near_duplicate_review_threshold": leakage_plan["near_duplicate_review_threshold"],
        "near_duplicate_flags": near,
        "near_duplicate_flag_count": len(near),
        "trigger_literal_occurrence": trigger_occurrence,
        "limitations": [
            "This audit can detect split overlap and textual near-duplicates, not unknown pretraining exposure.",
            "Near-duplicate flags are descriptive and require contextual review; they are not automatically leakage.",
        ],
    }
    LEAKAGE_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_multiplicity_audit(plan: Mapping[str, Any], component: Mapping[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(DISCOVERY_PATH)
    baselines = {
        row["example_id"]: row
        for row in rows
        if row["record_type"] == "baseline" and int(row["label"]) == 1
    }
    candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["record_type"] == "candidate" and int(row["label"]) == 1:
            candidates[row["candidate_id"]][row["example_id"]] = row
    candidate_ids = sorted(candidates)
    example_ids = sorted(baselines)
    expected_candidates = plan["analysis_only_falsifications"]["multiple_comparisons"]["discovery_candidates"]
    if len(candidate_ids) != expected_candidates:
        raise ValueError(f"expected {expected_candidates} discovery candidates, found {len(candidate_ids)}")
    concept_denominators = {}
    for concept in sorted({row["concept"] for row in baselines.values()}):
        concept_rows = [row for row in baselines.values() if row["concept"] == concept]
        concept_denominators[concept] = float(
            np.mean([row["normal_probe_score"] for row in concept_rows])
            - np.mean([row["triggered_probe_score"] for row in concept_rows])
        )
    values = np.empty((len(candidate_ids), len(example_ids)), dtype=np.float64)
    for candidate_index, candidate_id in enumerate(candidate_ids):
        if set(candidates[candidate_id]) != set(example_ids):
            raise ValueError(f"incomplete discovery candidate grid for {candidate_id}")
        for example_index, example_id in enumerate(example_ids):
            baseline = baselines[example_id]
            patched = candidates[candidate_id][example_id]
            values[candidate_index, example_index] = (
                patched["patched_probe_score"] - baseline["triggered_probe_score"]
            ) / concept_denominators[baseline["concept"]]
    scale = np.sqrt(np.mean(values.square(), axis=1)).clip(min=1e-12) / math.sqrt(len(example_ids))
    observed = np.abs(values.mean(axis=1) / scale)
    settings = plan["analysis_only_falsifications"]["multiple_comparisons"]
    rng = np.random.default_rng(settings["seed"])
    exceed_unadjusted = np.zeros(len(candidate_ids), dtype=np.int64)
    exceed_max = np.zeros(len(candidate_ids), dtype=np.int64)
    replicates = 20000
    chunk_size = 500
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(example_ids)))
        statistics = np.abs((signs @ values.T) / len(example_ids) / scale)
        exceed_unadjusted += np.sum(statistics >= observed[None, :], axis=0)
        maxima = statistics.max(axis=1)
        exceed_max += np.sum(maxima[:, None] >= observed[None, :], axis=0)
    unadjusted = (exceed_unadjusted + 1) / (replicates + 1)
    adjusted = (exceed_max + 1) / (replicates + 1)
    candidate_results = []
    selected_set = set(component["selected_candidates"])
    for index, candidate_id in enumerate(candidate_ids):
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "selected_k16_member": candidate_id in selected_set,
                "mean_normalized_rescue": float(values[index].mean()),
                "studentized_abs_statistic": float(observed[index]),
                "unadjusted_sign_flip_p": float(unadjusted[index]),
                "max_t_familywise_p": float(adjusted[index]),
                "survives_familywise_0_05": bool(adjusted[index] <= 0.05),
            }
        )
    selected_results = [row for row in candidate_results if row["selected_k16_member"]]
    report = {
        "schema_version": 1,
        "procedure": "day14-max-t-v1",
        "status": "pass",
        "candidates": len(candidate_ids),
        "positive_examples": len(example_ids),
        "permutations": replicates,
        "seed": settings["seed"],
        "statistical_unit": "example-level sign flip; exploratory because components were selected on these data",
        "candidate_results": candidate_results,
        "selected_members_surviving_familywise_0_05": sum(
            row["survives_familywise_0_05"] for row in selected_results
        ),
        "selected_member_count": len(selected_results),
        "safety_test_multiplicity_protection": "The K16 identity and four-interval safety gate were frozen before any safety score was computed; discovery max-T results do not convert discovery selection into confirmatory evidence.",
        "limitations": [
            "Example-level sign flips treat examples as exchangeable and do not fully model concept clustering.",
            "This audits the exact causal rescue screen across 68 candidates, not every exploratory visualization or screening heuristic.",
        ],
    }
    MULTIPLICITY_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    commit = git_head()
    require_committed_file(Path(__file__).resolve(), commit)
    require_committed_file(PLAN_PATH, commit)
    plan = json.loads(PLAN_PATH.read_text())
    component = json.loads(COMPONENT_PATH.read_text())
    if plan["status"] != "frozen-before-day14-analysis":
        raise ValueError("Day 14 plan is not frozen")
    expected_hashes = plan["inputs"]
    actual_hashes = {
        "day13_central_raw_sha256": sha256_file(CENTRAL_PATH),
        "day13_central_summary_sha256": sha256_file(ROOT / "results/day-13/safety-transfer-summary.json"),
        "day13_confound_summary_sha256": sha256_file(CONFOUND_SUMMARY_PATH),
        "day13_audit_sha256": sha256_file(ROOT / "results/day-13/day13-audit.json"),
        "final_component_set_sha256": sha256_file(COMPONENT_PATH),
        "safety_split_sha256": sha256_file(ROOT / "data/splits/day04-v1/safety-test.LOCKED.jsonl"),
        "day08_discovery_raw_sha256": sha256_file(DISCOVERY_PATH),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("a frozen Day 14 input hash changed")
    central = load_jsonl(CENTRAL_PATH)
    if len(central) != 1944:
        raise ValueError("central Day 13 row count changed")
    index = index_central(central)
    seed_results = run_bootstrap_seeds(central, component, plan)
    outlier_results = run_outlier_analysis(index, plan)
    threshold_results = run_threshold_analysis(index, plan)
    leakage_results = run_leakage_audit(plan)
    multiplicity_results = run_multiplicity_audit(plan, component)
    confounds = json.loads(CONFOUND_SUMMARY_PATH.read_text())
    pooling = {
        concept["concept"]: {
            pool["pooling"]: pool["transfer_supported_descriptively"]
            for pool in concept["pools"]
        }
        for concept in confounds["alternative_probe_pooling"]
    }
    summary = {
        "schema_version": 1,
        "procedure": "day14-analysis-only-v1",
        "implementation_commit": commit,
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "input_hashes_verified": True,
        "bootstrap_seed_robustness": seed_results,
        "outlier_robustness": outlier_results,
        "threshold_robustness": threshold_results,
        "alternative_pooling_support": pooling,
        "all_pooling_rules_supported": all(all(values.values()) for values in pooling.values()),
        "leakage_audit": {
            "status": leakage_results["status"],
            "major_exact_leakage_detected": leakage_results["major_exact_leakage_detected"],
            "near_duplicate_flag_count": leakage_results["near_duplicate_flag_count"],
            "sha256": sha256_file(LEAKAGE_PATH),
        },
        "multiple_comparison_audit": {
            "selected_members_surviving_familywise_0_05": multiplicity_results["selected_members_surviving_familywise_0_05"],
            "selected_member_count": multiplicity_results["selected_member_count"],
            "sha256": sha256_file(MULTIPLICITY_PATH),
        },
        "analysis_only_falsification_gate_passed": bool(
            seed_results["all_concepts_supported_all_seeds"]
            and outlier_results["robustness_gate_passed"]
            and all(all(values.values()) for values in pooling.values())
            and not leakage_results["major_exact_leakage_detected"]
        ),
        "threshold_results_are_secondary": True,
        "epistemic_status": "post-confirmatory falsification",
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        "Day 14 analysis-only falsification complete: "
        f"gate={summary['analysis_only_falsification_gate_passed']}, "
        f"near_duplicate_flags={leakage_results['near_duplicate_flag_count']}, "
        f"FWER-selected={multiplicity_results['selected_members_surviving_familywise_0_05']}/"
        f"{multiplicity_results['selected_member_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
