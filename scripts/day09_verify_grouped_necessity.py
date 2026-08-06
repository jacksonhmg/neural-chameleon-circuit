#!/usr/bin/env python3
"""Independently audit the complete Day 9 grouped intervention package."""

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

from neural_chameleon import (  # noqa: E402
    CANDIDATE_BY_ID,
    group_specifications,
    load_experimental_split,
    summarize_grouped_necessity,
)


RESULT_DIR = ROOT / "results/day-09"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
EXPECTED_SCORE_ROWS = 19712
EXPECTED_BEHAVIOR_ROWS = 616
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "grouped-necessity-audit.json"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows = []
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "pass" if condition else "fail",
        "detail": detail,
    }


def latest_file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT,
        text=True,
    ).strip()


def committed_matches(path: Path, commit: str) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    return committed == path.read_bytes()


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    score_path = result_dir / "grouped-example-results.jsonl.gz"
    behavior_path = result_dir / "grouped-behavior-results.jsonl.gz"
    plan_path = result_dir / "frozen-group-plan.json"
    summary_path = result_dir / "grouped-necessity-summary.json"
    preflight_path = result_dir / "grouped-preflight.json"
    manifest_path = result_dir / "day09-artifacts.json"
    scores = load_jsonl(score_path)
    behavior = load_jsonl(behavior_path)
    plan = json.loads(plan_path.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    summary = json.loads(summary_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    specifications = group_specifications(plan)
    specification_by_id = {row["group_id"]: row for row in specifications}
    expected_group_ids = set(specification_by_id)
    checks: list[dict[str, str]] = []

    checks.append(
        check(
            len(scores) == EXPECTED_SCORE_ROWS
            and len(behavior) == EXPECTED_BEHAVIOR_ROWS,
            "raw_record_counts",
            f"Score and behavior counts are {len(scores)} and {len(behavior)}.",
        )
    )
    score_keys = [(row["example_id"], row.get("group_id")) for row in scores]
    behavior_keys = [(row["example_id"], row.get("group_id")) for row in behavior]
    checks.append(
        check(
            len(score_keys) == len(set(score_keys))
            and len(behavior_keys) == len(set(behavior_keys)),
            "unique_raw_keys",
            "Every raw row has a unique example/group key.",
        )
    )

    frozen = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    frozen_by_id = {row["example_id"]: row for row in frozen}
    by_score_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_score_example[row["example_id"]].append(row)
    complete_score_grid = (
        set(by_score_example) == set(frozen_by_id)
        and all(
            {row.get("group_id") for row in rows} == {None, *expected_group_ids}
            and len(rows) == 14
            for rows in by_score_example.values()
        )
    )
    checks.append(
        check(
            complete_score_grid,
            "complete_all_benign_score_grid",
            "All 1,408 frozen benign examples have one baseline and all 13 interventions.",
        )
    )

    expected_behavior_ids = set()
    frozen_cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in frozen:
        frozen_cells[(row["split"], row["concept"], row["label"])].append(row)
    for rows in frozen_cells.values():
        ordered = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    f"day08-behavior:{row['example_id']}".encode()
                ).hexdigest(),
                row["example_id"],
            ),
        )
        expected_behavior_ids.update(row["example_id"] for row in ordered[:2])
    by_behavior_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        by_behavior_example[row["example_id"]].append(row)
    complete_behavior_grid = (
        set(by_behavior_example) == expected_behavior_ids
        and len(expected_behavior_ids) == 44
        and all(
            {row.get("group_id") for row in rows} == {None, *expected_group_ids}
            and len(rows) == 14
            for rows in by_behavior_example.values()
        )
    )
    checks.append(
        check(
            complete_behavior_grid,
            "complete_frozen_behavior_grid",
            "Exactly the two frozen examples in each concept/class cell have all 13 interventions.",
        )
    )

    metadata_ok = all(
        row["split"] == frozen_by_id[row["example_id"]]["split"]
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == frozen_by_id[row["example_id"]]["label"]
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["correct_trigger"]
        == analysis_plan["conditions"]["correct_triggers"][row["concept"]]
        for row in (*scores, *behavior)
    )
    checks.append(
        check(
            metadata_ok,
            "frozen_metadata_and_triggers",
            "Every row matches its frozen split, concept, label, probe, and correct trigger.",
        )
    )

    plan_sha256 = sha256_file(plan_path)
    model_ok = all(
        row["procedure"] == "day09-v1"
        and row["freeze_id"] == "day04-v1"
        and row["model_repository"] == EXPECTED_REPOSITORY
        and row["model_revision"] == EXPECTED_REVISION
        and row["group_plan_sha256"] == plan_sha256
        and row["validation_used_for_selection"] is False
        and row["safety_split_accessed"] is False
        for row in (*scores, *behavior)
    )
    checks.append(
        check(
            model_ok,
            "pinned_model_procedure_and_plan",
            "All rows name the frozen model, revision, Day 9 procedure, and exact group plan.",
        )
    )

    response_pairing = all(
        len({row["response_ids_sha256"] for row in rows}) == 1
        and len({row["response_token_count"] for row in rows}) == 1
        for rows in (*by_score_example.values(), *by_behavior_example.values())
    )
    checks.append(
        check(
            response_pairing,
            "exact_response_pairing",
            "Every example reuses one response-token hash and count across conditions.",
        )
    )

    finite_scores = all(
        all(
            math.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1
            for field in (
                ("normal_probe_score", "triggered_probe_score")
                if row["record_type"] == "baseline"
                else ("patched_probe_score",)
            )
        )
        for row in scores
    )
    finite_behavior = all(
        all(
            math.isfinite(float(row[field]))
            for field in (
                ("normal_response_nll", "triggered_response_nll")
                if row["record_type"] == "baseline"
                else ("patched_response_nll",)
            )
        )
        for row in behavior
    )
    checks.append(
        check(
            finite_scores and finite_behavior,
            "finite_scores_and_nll",
            "All probabilities are finite and bounded; all fixed-continuation NLL values are finite.",
        )
    )

    group_rows = [row for row in (*scores, *behavior) if row["record_type"] != "baseline"]
    group_membership_ok = all(
        row["group_id"] in specification_by_id
        and row["group_role"] == specification_by_id[row["group_id"]]["group_role"]
        and row["set_size"] == specification_by_id[row["group_id"]]["set_size"]
        and row["candidate_ids"] == specification_by_id[row["group_id"]]["candidate_ids"]
        and row["candidate_ids_sha256"]
        == hashlib.sha256(
            json.dumps(row["candidate_ids"], separators=(",", ":")).encode()
        ).hexdigest()
        for row in group_rows
    )
    checks.append(
        check(
            group_membership_ok,
            "exact_frozen_group_membership",
            "Every patch row reproduces its frozen group role, size, ordered membership, and hash.",
        )
    )
    checks.append(
        check(
            all(
                row["direction"] == "rescue"
                and row["source_condition"] == "normal"
                and row["destination_condition"] == "correct_trigger"
                for row in group_rows
            ),
            "causal_direction_mapping",
            "All Day 9 patches are normal-to-correct-trigger rescue interventions.",
        )
    )

    selection_commit = latest_file_commit(SELECTION_PATH)
    procedure_commit = latest_file_commit(plan_path)
    named_selection_commits = {row["selection_commit"] for row in (*scores, *behavior)}
    named_procedure_commits = {row["procedure_commit"] for row in (*scores, *behavior)}
    committed_ok = (
        named_selection_commits == {selection_commit}
        and named_procedure_commits == {procedure_commit}
        and committed_matches(SELECTION_PATH, selection_commit)
        and committed_matches(plan_path, procedure_commit)
    )
    checks.append(
        check(
            committed_ok,
            "committed_frozen_inputs",
            "Rows name byte-identical committed Day 8 selection and Day 9 procedure inputs.",
        )
    )

    identity_checks = preflight["same_shape_identity_checks"]
    preflight_ok = (
        preflight["status"] == "pass"
        and preflight["same_shape_identity_check_count"] == 26
        and len(identity_checks) == 26
        and all(row["exact"] for row in identity_checks)
        and all(row["within_tolerance"] for row in preflight["vectorized_probe_score_checks"])
        and all(row["within_tolerance"] for row in preflight["vectorized_response_nll_checks"])
        and preflight["group_member_order_check"]["exact"]
        and preflight["registered_hook_count_after_checks"] == 0
    )
    checks.append(
        check(
            preflight_ok,
            "real_checkpoint_preflight",
            "All 26 identities, vector score/NLL tolerances, order invariance, and hook cleanup passed.",
        )
    )

    regenerated = summarize_grouped_necessity(
        scores,
        behavior,
        plan,
        replicates=summary["bootstrap"]["replicates"],
        seed=summary["bootstrap"]["seed"],
    )
    summary_matches = all(summary[key] == value for key, value in regenerated.items())
    checks.append(
        check(
            summary_matches,
            "deterministic_summary_regeneration",
            "The checked-in grouped summary exactly regenerates from raw archives and the frozen plan.",
        )
    )
    denominators = [
        concept["positive_suppression_denominator"] for concept in summary["concepts"]
    ]
    checks.append(
        check(
            len(denominators) == 11
            and all(metric["estimate"] > 0 and metric["ci_low"] > 0 for metric in denominators),
            "stable_positive_suppression_denominators",
            "All 11 positive suppression denominators and their bootstrap lower bounds are above zero.",
        )
    )

    curve_lookup = {
        (row["scope"], row["set_size"]): row["relative_to_k16"]["estimate"]
        for row in summary["selected_curve"]
    }
    compact_size = next(
        (
            size
            for size in (1, 2, 4)
            if all(curve_lookup[(scope, size)] >= 0.8 for scope in ("discovery", "validation"))
        ),
        None,
    )
    if compact_size is not None:
        expected_classification, expected_size = "compact", compact_size
    elif all(curve_lookup[(scope, 8)] >= 0.8 for scope in ("discovery", "validation")):
        expected_classification, expected_size = "moderately_distributed", 8
    else:
        expected_classification, expected_size = "highly_distributed", 16
    observed_classification = summary["distribution_classification"]
    checks.append(
        check(
            observed_classification["classification"] == expected_classification
            and observed_classification["first_size_reaching_rule"] == expected_size,
            "frozen_compactness_rule",
            f"The frozen 80% rule independently yields {expected_classification} at K={expected_size}.",
        )
    )

    positive_control_id = plan["positive_control"]["group_id"]
    exact_positive_control = all(
        row["patched_probe_score"]
        == next(
            baseline["normal_probe_score"]
            for baseline in by_score_example[row["example_id"]]
            if baseline["record_type"] == "baseline"
        )
        for row in scores
        if row.get("group_id") == positive_control_id
    )
    checks.append(
        check(
            exact_positive_control,
            "exact_block12_positive_control",
            "Complete block-12 output rescue exactly reproduces the normal probe score for every example.",
        )
    )

    nested_prefixes = all(
        plan[key][str(smaller)] == plan[key][str(larger)][:smaller]
        for key in ("selected_prefixes", "random_prefixes")
        for smaller, larger in ((1, 2), (2, 4), (4, 8), (8, 16))
    )
    selected_exact = all(
        plan["selected_prefixes"][str(size)] == selection["selected_candidates"][:size]
        for size in (1, 2, 4, 8, 16)
    )
    random_exact = all(
        plan["random_prefixes"][str(size)] == selection["random_control_candidates"][:size]
        for size in (1, 2, 4, 8, 16)
    )
    control_candidates = plan["block_control"]["candidate_ids"]
    checks.append(
        check(
            nested_prefixes
            and selected_exact
            and random_exact
            and len(control_candidates) == 17
            and not (set(control_candidates) & set(selection["selected_candidates"]))
            and all(CANDIDATE_BY_ID[item].layer != 11 for item in control_candidates),
            "nested_prefixes_and_control_isolation",
            "Selected/random prefixes exactly follow Day 8; the 17-site control is nonselected and outside layer 11.",
        )
    )

    expected_table_counts = {
        "grouped-necessity-macro.csv": 78,
        "grouped-necessity-concepts.csv": 286,
        "selected-random-contrasts.csv": 30,
        "selected-completeness-curve.csv": 10,
        "grouped-behavior-summary.csv": 78,
    }
    observed_table_counts = {
        name: csv_row_count(result_dir / name) for name in expected_table_counts
    }
    checks.append(
        check(
            observed_table_counts == expected_table_counts,
            "summary_table_shapes",
            f"CSV row counts are {observed_table_counts}.",
        )
    )

    figure_names = (
        "necessity-completeness-curves.png",
        "grouped-controls-behavior.png",
    )
    figure_sizes = {
        name: Image.open(result_dir / name).size for name in figure_names
    }
    checks.append(
        check(
            all(size == (4800, 2700) for size in figure_sizes.values()),
            "publication_figure_dimensions",
            f"PNG figure sizes are {figure_sizes}.",
        )
    )

    manifest_ok = all(
        (result_dir / name).is_file()
        and metadata["sha256"] == sha256_file(result_dir / name)
        and metadata["bytes"] == (result_dir / name).stat().st_size
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            manifest_ok and len(manifest["files"]) == 14,
            "artifact_manifest",
            "All 14 pre-audit artifacts exist and match their recorded byte counts and SHA-256 hashes.",
        )
    )

    safety_locked = not (SPLIT_DIR / "safety-unlock.json").exists()
    safety_metadata = (
        summary["safety_split_accessed"] is False
        and manifest["safety_split_accessed"] is False
        and preflight["safety_split_accessed"] is False
    )
    checks.append(
        check(
            safety_locked and safety_metadata,
            "safety_split_isolation",
            "The safety unlock is absent and every Day 9 artifact records no safety access.",
        )
    )

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day09-v1",
        "status": status,
        "check_count": len(checks),
        "checks": checks,
        "inputs": {
            score_path.name: sha256_file(score_path),
            behavior_path.name: sha256_file(behavior_path),
            plan_path.name: sha256_file(plan_path),
            summary_path.name: sha256_file(summary_path),
            preflight_path.name: sha256_file(preflight_path),
            manifest_path.name: sha256_file(manifest_path),
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 9 audit: {status} ({len(checks)} checks).")
    if status != "pass":
        failed = ", ".join(row["name"] for row in checks if row["status"] == "fail")
        raise SystemExit(f"Day 9 audit failed: {failed}")


if __name__ == "__main__":
    main()
