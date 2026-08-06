#!/usr/bin/env python3
"""Independently audit the complete Day 12 benign-transfer package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import CANDIDATES, classify_mechanism, load_experimental_split  # noqa: E402


RESULT_DIR = ROOT / "results/day-12"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_ROWS = 30_912
EXPECTED_EXAMPLES = 448


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "benign-transfer-audit.json")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def committed_matches(path: Path, commit: str) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) == path.read_bytes()


def csv_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output = args.output.resolve()
    paths = {
        "raw": result_dir / "validation-candidate-results.jsonl.gz",
        "preflight": result_dir / "validation-candidate-preflight.json",
        "summary": result_dir / "benign-transfer-summary.json",
        "manifest": result_dir / "day12-artifacts.json",
        "plan": result_dir / "frozen-benign-transfer-plan.json",
        "final_set": result_dir / "frozen-final-component-set.json",
        "confirmatory": result_dir / "frozen-confirmatory-analysis.json",
    }
    records = load_jsonl(paths["raw"])
    preflight = json.loads(paths["preflight"].read_text())
    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    plan = json.loads(paths["plan"].read_text())
    final_set = json.loads(paths["final_set"].read_text())
    confirmatory = json.loads(paths["confirmatory"].read_text())
    checks = []

    checks.append(check(len(records) == EXPECTED_ROWS, "raw_record_count", f"Raw archive contains {len(records)} rows."))
    keys = [
        (row["example_id"], "baseline" if row["record_type"] == "baseline" else row["candidate_id"])
        for row in records
    ]
    checks.append(check(len(keys) == len(set(keys)), "unique_record_keys", "Every example/condition key is unique."))
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_example[row["example_id"]].append(row)
    candidates = {candidate.candidate_id for candidate in CANDIDATES}
    complete = all(
        len(rows) == 69
        and sum(row["record_type"] == "baseline" for row in rows) == 1
        and {row["candidate_id"] for row in rows if row["record_type"] == "candidate"} == candidates
        for rows in by_example.values()
    )
    checks.append(check(len(by_example) == EXPECTED_EXAMPLES and complete, "complete_validation_grid", f"Found {len(by_example)} examples with one baseline and 68 candidates each."))

    frozen = [row for row in load_experimental_split("validation") if row["label"] == 1]
    frozen_by_id = {row["example_id"]: row for row in frozen}
    checks.append(check(set(by_example) == set(frozen_by_id), "exact_frozen_validation_positives", "The raw grid covers exactly all 448 frozen validation positives."))
    metadata_ok = all(
        row["split"] == "validation"
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == 1
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["procedure"] == "day12-v1"
        and row["freeze_id"] == "day04-v1"
        and row["model_repository"] == plan["model"]["repository"]
        and row["model_revision"] == plan["model"]["revision"]
        and row["transfer_plan_sha256"] == sha256_file(paths["plan"])
        and row["procedure_commit"] == "0f967dda4205aa68ac26620e022bc5c4455543fc"
        and row["validation_used_for_selection"] is False
        and row["safety_split_accessed"] is False
        for row in records
    )
    checks.append(check(metadata_ok, "pinned_metadata_and_safety_isolation", "Every row matches the frozen validation split, model, procedure, and safety flags."))

    response_pairing = all(
        len({row["response_ids_sha256"] for row in rows}) == 1
        and len({row["response_token_count"] for row in rows}) == 1
        for rows in by_example.values()
    )
    checks.append(check(response_pairing, "exact_response_pairing", "All 69 conditions share one response token sequence per example."))
    finite = all(
        math.isfinite(float(row["normal_probe_score"] if row["record_type"] == "baseline" else row["patched_probe_score"]))
        and math.isfinite(float(row["triggered_probe_score"] if row["record_type"] == "baseline" else row["patched_probe_score"]))
        for row in records
    )
    checks.append(check(finite, "finite_probe_scores", "Every recorded probe score is finite."))

    implementation_commits = {row["implementation_commit"] for row in records}
    implementation_ok = len(implementation_commits) == 1
    if implementation_ok:
        implementation_commit = next(iter(implementation_commits))
        implementation_ok = committed_matches(ROOT / "scripts/day12_run_validation_candidates.py", implementation_commit)
    checks.append(check(implementation_ok, "committed_evaluator_provenance", f"Raw rows name one committed evaluator: {sorted(implementation_commits)}."))
    checks.append(check(
        preflight["status"] == "pass"
        and preflight["identity_check_count"] == 68
        and preflight["vectorized_equivalence_check_count"] == 2
        and all(row["exact"] for row in preflight["identity_checks"])
        and all(row["within_tolerance"] for row in preflight["vectorized_equivalence_checks"]),
        "real_checkpoint_preflight", "The real checkpoint passed 68 identities and two vectorized equivalence checks.",
    ))

    summary_ok = (
        summary["validation_record_count"] == EXPECTED_ROWS
        and summary["validation_raw_sha256"] == sha256_file(paths["raw"])
        and summary["transfer_plan_sha256"] == sha256_file(paths["plan"])
        and summary["component_set_sha256"] == plan["component_set"]["component_set_sha256"]
        and summary["final_component_set"] == plan["component_set"]["selected_candidates"]
        and summary["validation_used_for_selection"] is False
        and summary["safety_split_accessed"] is False
    )
    checks.append(check(summary_ok, "summary_provenance_and_final_set", "Summary hashes the raw inputs and preserves the exact frozen K=16 set."))
    checks.append(check(
        final_set["status"] == "final-and-frozen-before-safety-unlock"
        and final_set["selected_candidates"] == plan["component_set"]["selected_candidates"]
        and final_set["component_set_sha256"] == plan["component_set"]["component_set_sha256"]
        and final_set["frozen_after_benign_result_sha256"] == sha256_file(paths["summary"])
        and final_set["validation_used_for_selection"] is False
        and final_set["safety_split_accessed"] is False,
        "final_component_set_freeze", "The final K=16 set exactly preserves the discovery selection and hashes the benign result.",
    ))
    checks.append(check(
        confirmatory["status"] == "frozen-before-safety-unlock"
        and confirmatory["component_set_file_sha256"] == sha256_file(paths["final_set"])
        and confirmatory["analysis_script_sha256"] == sha256_file(ROOT / confirmatory["analysis_script"])
        and confirmatory["analysis_module_sha256"] == sha256_file(ROOT / confirmatory["analysis_module"])
        and confirmatory["safety_split"]["sha256"] == "48d6818db1946919c1a49c373170f701219df472b669900d39cde834e0d595ec"
        and confirmatory["selected_candidates"] == final_set["selected_candidates"]
        and confirmatory["random_control_candidates"] == final_set["random_control_candidates"]
        and confirmatory["safety_split_accessed"] is False,
        "confirmatory_analysis_freeze", "The executable safety analysis, component membership, metrics, and locked split hash are frozen.",
    ))
    candidate = summary["candidate_transfer"]
    checks.append(check(
        candidate["candidate_count"] == 68
        and len(candidate["concepts"]) == 11
        and all(row["n_positive"] == 64 for row in candidate["concepts"])
        and len(candidate["pairwise_ranking_agreement"]) == 55,
        "complete_candidate_analysis", "Analysis covers 68 candidates, 11 concepts, 64 positives per concept, and all 55 concept pairs.",
    ))
    checks.append(check(
        all(row["suppression_denominator"]["ci_low"] > 0 for row in candidate["concepts"]),
        "positive_suppression_denominators", "Every concept has a strictly positive bootstrap suppression denominator.",
    ))
    checks.append(check(
        len(summary["direct_transfer"]["concepts"]) == 11
        and len(summary["candidate_transfer"]["selected_component_matrix"]) == 16,
        "transfer_matrix_dimensions", "The transfer package contains 11 direct rows and 16 frozen component columns.",
    ))
    evidence = summary["evidence"]
    expected_classification = classify_mechanism(
        direct_transfer_supported=evidence["direct_transfer_supported"],
        sharing_supported=evidence["rank_sharing_supported"],
        high_sharing_supported=evidence["high_rank_sharing_supported"],
        sparse_k4_supported=summary["sparse_k4_test"]["supported"],
        shared_actuator_supported=evidence["shared_downstream_actuator_supported"],
        trigger_reader_specificity_supported=evidence["trigger_reader_source_specificity_supported"],
    )
    checks.append(check(summary["mechanistic_classification"] == expected_classification, "frozen_classification_rule", f"Classification is {expected_classification}."))

    table_counts = {
        "benign-cross-concept-transfer-matrix.csv": 11,
        "concept-ranking-agreement.csv": 55,
        "candidate-macro-rankings.csv": 204,
        "trigger-reader-source-roles.csv": 132,
    }
    checks.append(check(
        all(csv_count(result_dir / name) == expected for name, expected in table_counts.items()),
        "analysis_table_dimensions", f"CSV row counts match {table_counts}.",
    ))
    images = [result_dir / "benign-cross-concept-transfer.png", result_dir / "component-ranking-agreement.png"]
    checks.append(check(
        all(Image.open(path).size == (4800, 2700) for path in images),
        "publication_figure_dimensions", "Both PNG figures are 4,800 by 2,700 pixels.",
    ))
    manifest_ok = all(
        (result_dir / name).is_file()
        and entry["sha256"] == sha256_file(result_dir / name)
        and entry["bytes"] == (result_dir / name).stat().st_size
        for name, entry in manifest["files"].items()
    )
    checks.append(check(manifest_ok, "artifact_manifest", f"All {len(manifest['files'])} manifested artifacts match size and SHA-256."))
    checks.append(check(
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and (SPLIT_DIR / "safety-test.LOCKED.jsonl").exists()
        and summary["safety_split_accessed"] is False,
        "safety_remained_locked", "No unlock authorization exists and the locked safety dataset was not loaded.",
    ))

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day12-v1",
        "status": status,
        "check_count": len(checks),
        "checks": checks,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        failed = [row["name"] for row in checks if row["status"] == "fail"]
        raise RuntimeError(f"Day 12 audit failed: {failed}")
    print(f"Day 12 audit passed: {len(checks)} checks.")


if __name__ == "__main__":
    main()
