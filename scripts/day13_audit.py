#!/usr/bin/env python3
"""Audit Day 13 safety and confound artifacts against every frozen invariant."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import load_experimental_split  # noqa: E402
from neural_chameleon.interventions import PairedInterventionRunner  # noqa: E402


RESULT_DIR = ROOT / "results/day-13"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
PLAN_PATH = RESULT_DIR / "frozen-execution-plan.json"
FREEZE_PATH = ROOT / "results/day-12/frozen-confirmatory-analysis.json"
COMPONENT_PATH = ROOT / "results/day-12/frozen-final-component-set.json"
CENTRAL_SEAL_PATH = RESULT_DIR / "central-result-seal.json"
CENTRAL_PATH = RESULT_DIR / "safety-transfer-results.jsonl.gz"
CENTRAL_SUMMARY_PATH = RESULT_DIR / "safety-transfer-summary.json"
PROMPT_PATH = RESULT_DIR / "prompt-confound-results.jsonl.gz"
ZERO_PATH = RESULT_DIR / "zero-ablation-results.jsonl.gz"
GENERATION_PATH = RESULT_DIR / "generation-diagnostics.jsonl.gz"
CONFOUND_SUMMARY_PATH = RESULT_DIR / "confound-summary.json"
OUTPUT_PATH = RESULT_DIR / "day13-audit.json"
PROMPT_CONDITIONS = {
    "concept_only",
    "monitoring_only",
    "irrelevant_concept",
    "nearby_concept",
    "paraphrased_trigger",
    "malformed_trigger",
    "relocated_trigger",
}
GENERATION_CONDITIONS = {
    "normal",
    "correct_trigger",
    "concept_only",
    "monitoring_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
    ).strip()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=ROOT,
        check=False,
    ).returncode == 0


def deterministic_subset(
    records: Sequence[dict[str, Any]], salt: str, per_cell: int
) -> set[str]:
    cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    for record in records:
        cells[(record["concept"], int(record["label"]))].append(record["example_id"])
    selected = set()
    for key in sorted(cells):
        ranked = sorted(
            cells[key],
            key=lambda example_id: (
                hashlib.sha256(f"{salt}:{example_id}".encode()).hexdigest(),
                example_id,
            ),
        )
        selected.update(ranked[:per_cell])
    return selected


def prompt_for(
    condition_id: str,
    record: Mapping[str, Any],
    trigger: str,
    plan: Mapping[str, Any],
) -> str:
    if condition_id == "normal":
        return record["prompt"]
    if condition_id == "correct_trigger":
        return PairedInterventionRunner.TRIGGER_TEMPLATE.format(
            trigger=trigger, prompt=record["prompt"]
        )
    templates = plan["confounds"]["prompt_conditions"]
    substitutions = plan["confounds"]["semantic_substitutions"][record["concept"]]
    template_by_condition = {
        "concept_only": templates["concept_only"],
        "monitoring_only": templates["monitoring_only"],
        "irrelevant_concept": templates["irrelevant_concept"],
        "nearby_concept": templates["nearby_concept"],
        "paraphrased_trigger": templates["paraphrased_trigger"],
        "malformed_trigger": templates["malformed_trigger"],
        "relocated_trigger": templates["relocated_trigger"],
    }
    return template_by_condition[condition_id].format(
        trigger=trigger,
        prompt=record["prompt"],
        irrelevant=substitutions["irrelevant"],
        nearby=substitutions["nearby"],
    )


def all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def assert_exact_grid(
    rows: Sequence[dict[str, Any]],
    expected_ids: set[str],
    expected_conditions: set[str],
    condition_field: str = "condition_id",
) -> None:
    keys = [(row["example_id"], row[condition_field]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate Day 13 diagnostic key")
    if {row["example_id"] for row in rows} != expected_ids:
        raise ValueError("diagnostic example subset differs from frozen salt")
    by_example: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_example[row["example_id"]].add(row[condition_field])
    if any(conditions != expected_conditions for conditions in by_example.values()):
        raise ValueError("diagnostic condition grid is incomplete")


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    freeze = json.loads(FREEZE_PATH.read_text())
    component = json.loads(COMPONENT_PATH.read_text())
    seal = json.loads(CENTRAL_SEAL_PATH.read_text())
    central_summary = json.loads(CENTRAL_SUMMARY_PATH.read_text())
    confound_summary = json.loads(CONFOUND_SUMMARY_PATH.read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    central = load_jsonl(CENTRAL_PATH)
    prompt = load_jsonl(PROMPT_PATH)
    zero = load_jsonl(ZERO_PATH)
    generation = load_jsonl(GENERATION_PATH)
    safety = load_experimental_split("safety-test")
    safety_by_id = {record["example_id"]: record for record in safety}

    checks: dict[str, Any] = {}
    checks["raw_counts"] = {
        "central": len(central),
        "prompt": len(prompt),
        "zero": len(zero),
        "generation": len(generation),
    }
    if tuple(checks["raw_counts"].values()) != (1944, 448, 64, 128):
        raise ValueError("raw Day 13 counts differ from frozen plan")

    plan_hash = sha256_file(PLAN_PATH)
    if plan_hash != seal["execution_plan_sha256"]:
        raise ValueError("execution plan changed after central seal")
    if sha256_file(CENTRAL_PATH) != seal["central_raw_results_sha256"]:
        raise ValueError("central result changed after its seal")
    if central_summary["raw_results_sha256"] != seal["central_raw_results_sha256"]:
        raise ValueError("central analysis does not name the sealed raw archive")
    if confound_summary["raw_sha256"] != {
        "central": sha256_file(CENTRAL_PATH),
        "prompt_confounds": sha256_file(PROMPT_PATH),
        "zero_ablation": sha256_file(ZERO_PATH),
        "generation": sha256_file(GENERATION_PATH),
    }:
        raise ValueError("confound summary raw hashes mismatch")
    checks["hash_chain"] = "pass"

    if component["component_set_sha256"] != freeze["component_set_sha256"]:
        raise ValueError("component set changed between Day 12 freezes")
    if any(
        row["component_set_sha256"] != freeze["component_set_sha256"]
        for row in (*central, *prompt, *zero, *generation)
    ):
        raise ValueError("a raw row names the wrong component set")
    if any(row["execution_plan_sha256"] != plan_hash for row in (*central, *prompt, *zero, *generation)):
        raise ValueError("a raw row names the wrong execution plan")
    checks["immutable_component_and_plan"] = "pass"

    central_keys = []
    central_by_example: dict[str, set[str]] = defaultdict(set)
    for row in central:
        condition = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
        central_keys.append((row["example_id"], condition))
        central_by_example[row["example_id"]].add(condition)
        if row["record_type"] == "intervention":
            expected_candidates = (
                freeze["selected_candidates"]
                if row["group_id"] == "selected_k16"
                else freeze["random_control_candidates"]
            )
            if row["candidate_ids"] != expected_candidates:
                raise ValueError("central intervention membership changed")
    required_central = {
        "normal",
        "correct_trigger",
        "selected_k16:rescue",
        "selected_k16:induction",
        "random_k16:rescue",
        "random_k16:induction",
    }
    if len(central_keys) != len(set(central_keys)) or len(central_by_example) != 324:
        raise ValueError("central result keys are not unique and complete")
    if any(conditions != required_central for conditions in central_by_example.values()):
        raise ValueError("central six-condition grid is incomplete")
    checks["central_grid"] = "pass"

    confound_ids = deterministic_subset(safety, "day13-confounds", 16)
    generation_ids = deterministic_subset(safety, "day13-generation", 8)
    assert_exact_grid(prompt, confound_ids, PROMPT_CONDITIONS)
    assert_exact_grid(generation, generation_ids, GENERATION_CONDITIONS)
    if {row["example_id"] for row in zero} != confound_ids or len({row["example_id"] for row in zero}) != 64:
        raise ValueError("zero-ablation subset mismatch")
    if any(row["candidate_ids"] != freeze["selected_candidates"] for row in zero):
        raise ValueError("zero ablation changed selected membership")
    checks["frozen_subsets_and_membership"] = "pass"

    triggers = analysis_plan["conditions"]["correct_triggers"]
    for row in (*prompt, *generation):
        record = safety_by_id[row["example_id"]]
        rendered = prompt_for(row["condition_id"], record, triggers[row["concept"]], plan)
        if hashlib.sha256(rendered.encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError("prompt confound rendering hash mismatch")
    checks["prompt_renderings"] = "pass"

    confound_commit = next(iter({row["implementation_commit"] for row in prompt + zero + generation}))
    if any(row["implementation_commit"] != confound_commit for row in prompt + zero + generation):
        raise ValueError("confound rows used multiple implementation commits")
    central_seal_commit = file_commit(CENTRAL_SEAL_PATH)
    if not is_ancestor(central_seal_commit, confound_commit):
        raise ValueError("confound evaluator predates central seal")
    if confound_commit != file_commit(ROOT / "scripts/day13_run_confounds.py"):
        raise ValueError("confound rows do not name the committed evaluator")
    checks["central_before_confounds_commit_order"] = {
        "central_seal_commit": central_seal_commit,
        "confound_implementation_commit": confound_commit,
        "pass": True,
    }

    if any(row.get("safety_split_accessed") is not True for row in (*central, *prompt, *zero, *generation)):
        raise ValueError("authorized safety access flag missing")
    if any(row.get("safety_specific_reranking") is not False for row in (*prompt, *zero, *generation)):
        raise ValueError("confound row does not deny safety-specific reranking")
    if any(not row["nonempty"] or row["generated_probe_score"] is None for row in generation):
        raise ValueError("generation diagnostic is empty or unscored")
    checks["authorized_access_and_generation_completeness"] = "pass"

    pooling_support = {
        concept["concept"]: {
            pool["pooling"]: pool["transfer_supported_descriptively"]
            for pool in concept["pools"]
        }
        for concept in confound_summary["alternative_probe_pooling"]
    }
    if not all(all(values.values()) for values in pooling_support.values()):
        raise ValueError("alternative pooling lost descriptive support")
    if any(row["point_estimate_outside_frozen_band"] for row in confound_summary["activation_norms"]):
        raise ValueError("activation norm point estimate left frozen band")
    if not all_finite(confound_summary):
        raise ValueError("confound summary contains non-finite values")
    checks["analysis_invariants"] = {
        "central_support_unchanged": central_summary["overall_safety_transfer_supported"],
        "pooling_support": pooling_support,
        "activation_norm_point_estimate_anomalies": 0,
        "finite_summary": True,
    }

    artifact_paths = (
        RESULT_DIR / "prompt-confound-metrics.csv",
        RESULT_DIR / "alternative-pooling-metrics.csv",
        RESULT_DIR / "generation-diagnostic-metrics.csv",
        RESULT_DIR / "confound-diagnostics.png",
        RESULT_DIR / "confound-diagnostics.pdf",
    )
    artifact_hashes = {path.name: sha256_file(path) for path in artifact_paths}
    checks["analysis_artifact_sha256"] = artifact_hashes

    report = {
        "schema_version": 1,
        "procedure": "day13-audit-v1",
        "status": "pass",
        "checks": checks,
        "component_set_modified_after_safety": False,
        "validation_used_for_selection": False,
        "safety_specific_reranking": False,
        "central_overall_safety_transfer_supported": central_summary["overall_safety_transfer_supported"],
        "interpretation_classification": "qualified_causal_safety_transfer",
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("Day 13 audit passed: all frozen grids, hashes, subsets, and provenance verified.")


if __name__ == "__main__":
    main()
