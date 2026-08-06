#!/usr/bin/env python3
"""Independently audit the complete Day 11 controller-actuator package."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import day11_specifications, load_experimental_split, summarize_controller_actuator  # noqa: E402


RESULT_DIR = ROOT / "results/day-11"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
DAY10_PLAN_PATH = ROOT / "results/day-10/frozen-sufficiency-plan.json"
DAY10_SUMMARY_PATH = ROOT / "results/day-10/sufficiency-summary.json"
EXPECTED_ROWS = 19_360
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "controller-actuator-audit.json")
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


def latest_file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True
    ).strip()


def committed_matches(path: Path, commit: str) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) == path.read_bytes()


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def membership_hash(specification: Mapping[str, Any]) -> str:
    payload = {
        key: specification.get(key)
        for key in ("head_ids", "source_regions", "candidate_ids", "residual_sites")
        if key in specification
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    paths = {
        "raw": result_dir / "controller-actuator-results.jsonl.gz",
        "plan": result_dir / "frozen-controller-actuator-plan.json",
        "preflight": result_dir / "controller-actuator-preflight.json",
        "summary": result_dir / "controller-actuator-summary.json",
        "manifest": result_dir / "day11-artifacts.json",
    }
    records = load_jsonl(paths["raw"])
    plan = json.loads(paths["plan"].read_text())
    preflight = json.loads(paths["preflight"].read_text())
    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    specifications = day11_specifications(plan)
    spec_by_id = {row["intervention_id"]: row for row in specifications}
    expected_interventions = set(spec_by_id)
    expected_conditions = set(plan["baseline_conditions"])
    checks: list[dict[str, str]] = []

    checks.append(check(
        len(records) == EXPECTED_ROWS, "raw_record_count",
        f"Raw archive contains {len(records)} rows.",
    ))
    raw_keys = [(row["example_id"], row["record_key"]) for row in records]
    checks.append(check(
        len(raw_keys) == len(set(raw_keys)), "unique_raw_keys",
        "Every example/record key is unique.",
    ))

    frozen = load_experimental_split("discovery") + load_experimental_split("validation")
    frozen_by_id = {row["example_id"]: row for row in frozen}
    positive_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frozen:
        if row["label"] == 1:
            positive_cells[(row["split"], row["concept"])].append(row)
    expected_ids = set()
    for cell in positive_cells.values():
        ordered = sorted(
            cell,
            key=lambda row: (
                hashlib.sha256(f"day11-controller:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            ),
        )
        expected_ids.update(row["example_id"] for row in ordered[:8])
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_example[row["example_id"]].append(row)
    checks.append(check(
        len(expected_ids) == 88 and set(by_example) == expected_ids,
        "frozen_positive_subset",
        "The archive uses exactly the eight frozen SHA-selected positives per benign concept.",
    ))
    complete_grid = all(
        len(rows) == 220
        and {row["condition_id"] for row in rows if row["record_type"] == "baseline"} == expected_conditions
        and {row["intervention_id"] for row in rows if row["record_type"] == "intervention"} == expected_interventions
        for rows in by_example.values()
    )
    checks.append(check(
        complete_grid, "complete_220_condition_grid",
        "Every one of 88 examples has five baselines and all 215 interventions.",
    ))
    family_counts = Counter(row.get("family", "baseline") for row in records)
    expected_family_counts = {
        "baseline": 88 * 5,
        "layer_source_scan": 88 * 91,
        "individual_selected_head": 88 * 72,
        "selected_random_source_group": 88 * 32,
        "direct_response_output": 88 * 20,
    }
    checks.append(check(
        dict(family_counts) == expected_family_counts,
        "frozen_family_counts", f"Raw family counts are {dict(family_counts)}.",
    ))

    metadata_ok = all(
        row["split"] == frozen_by_id[row["example_id"]]["split"]
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == 1
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["correct_trigger"] == analysis_plan["conditions"]["correct_triggers"][row["concept"]]
        and row["irrelevant_trigger"] == analysis_plan["conditions"]["irrelevant_triggers"][row["concept"]]
        for row in records
    )
    checks.append(check(
        metadata_ok, "frozen_metadata_and_triggers",
        "Every row matches its frozen example, label, probe, correct trigger, and irrelevant trigger.",
    ))
    plan_sha256 = sha256_file(paths["plan"])
    checks.append(check(
        all(
            row["procedure"] == "day11-v1"
            and row["freeze_id"] == "day04-v1"
            and row["model_repository"] == EXPECTED_REPOSITORY
            and row["model_revision"] == EXPECTED_REVISION
            and row["controller_actuator_plan_sha256"] == plan_sha256
            and row["validation_used_for_selection"] is False
            and row["safety_split_accessed"] is False
            for row in records
        ),
        "pinned_model_procedure_and_plan",
        "All rows name the pinned checkpoint, procedure, and exact Day 11 plan.",
    ))
    checks.append(check(
        all(
            len({row["response_ids_sha256"] for row in rows}) == 1
            and len({row["response_token_count"] for row in rows}) == 1
            for rows in by_example.values()
        ),
        "exact_response_pairing",
        "All 220 conditions reuse one response-token sequence per example.",
    ))
    finite = all(
        math.isfinite(float(row["probe_score"] if row["record_type"] == "baseline" else row["patched_probe_score"]))
        and 0 <= float(row["probe_score"] if row["record_type"] == "baseline" else row["patched_probe_score"]) <= 1
        for row in records
    )
    checks.append(check(finite, "finite_bounded_probe_scores", "Every raw probe score is finite and in [0, 1]."))

    intervention_rows = [row for row in records if row["record_type"] == "intervention"]
    specification_ok = all(
        row["intervention_id"] in spec_by_id
        and all(row.get(key) == value for key, value in spec_by_id[row["intervention_id"]].items())
        and row["membership_sha256"] == membership_hash(spec_by_id[row["intervention_id"]])
        for row in intervention_rows
    )
    checks.append(check(
        specification_ok, "exact_frozen_intervention_membership",
        "Every intervention exactly reproduces its frozen definition and ordered membership hash.",
    ))
    direction_ok = all(
        row["source_condition"] == ("normal" if row["direction"] == "rescue" else "correct_trigger")
        and row["destination_condition"] == ("correct_trigger" if row["direction"] == "rescue" else "normal")
        for row in intervention_rows
    )
    checks.append(check(
        direction_ok, "causal_direction_mapping",
        "Rescue and induction rows use the frozen source/destination mapping.",
    ))

    expected_commits = {
        "selection_commit": latest_file_commit(SELECTION_PATH),
        "day10_procedure_commit": latest_file_commit(DAY10_PLAN_PATH),
        "day10_results_commit": latest_file_commit(DAY10_SUMMARY_PATH),
        "day11_procedure_commit": latest_file_commit(paths["plan"]),
        "implementation_commit": latest_file_commit(ROOT / "scripts/day11_run_controller_actuator.py"),
    }
    committed_ok = all(
        {row[field] for row in records} == {commit} for field, commit in expected_commits.items()
    ) and all(
        committed_matches(path, expected_commits[field])
        for field, path in (
            ("selection_commit", SELECTION_PATH),
            ("day10_procedure_commit", DAY10_PLAN_PATH),
            ("day10_results_commit", DAY10_SUMMARY_PATH),
            ("day11_procedure_commit", paths["plan"]),
            ("implementation_commit", ROOT / "scripts/day11_run_controller_actuator.py"),
        )
    )
    checks.append(check(
        committed_ok, "committed_inputs_and_implementation",
        "Rows name byte-identical committed inputs, Day 11 procedure, and evaluator.",
    ))

    preflight_ok = (
        preflight["status"] == "pass"
        and preflight["response_pairing_exact"]
        and len(preflight["source_partition_checks"]) == 2
        and all(row["exclusive"] and row["complete"] for row in preflight["source_partition_checks"])
        and preflight["maximum_head_reconstruction_abs_error"] <= preflight["reconstruction_tolerance"] == 0.02
        and len(preflight["source_identity_checks"]) == 16
        and all(row["exact"] for row in preflight["source_identity_checks"])
        and all(row["within_tolerance"] for row in preflight["source_vector_checks"])
        and preflight["source_group_order_check"]["exact"]
        and all(row["within_tolerance"] for row in preflight["contribution_vs_standard_baseline_checks"])
        and len(preflight["direct_identity_checks"]) == 20
        and all(row["exact"] for row in preflight["direct_identity_checks"])
        and all(row["within_tolerance"] for row in preflight["direct_vector_checks"])
        and len(preflight["block12_endpoint_checks"]) == 2
        and all(row["exact"] for row in preflight["block12_endpoint_checks"])
        and preflight["registered_hook_count_after_checks"] == 0
    )
    checks.append(check(
        preflight_ok, "real_checkpoint_preflight",
        "Masks, reconstruction, identities, vectors, order, direct patches, endpoints, and hook cleanup pass.",
    ))
    token_partition_metadata = all(
        all(row["source_region_token_counts"][key] == 0 for key in ("monitoring_language", "named_concept", "trigger_other"))
        for row in records if row.get("condition_id") == "normal"
    ) and all(
        row["source_region_token_counts"]["monitoring_language"] > 0
        and row["source_region_token_counts"]["named_concept"] > 0
        and row["source_region_token_counts"]["original_prompt"] > 0
        and row["source_region_token_counts"]["response"] > 0
        for row in records if row.get("condition_id") == "correct_trigger"
    )
    checks.append(check(
        token_partition_metadata, "raw_source_partition_metadata",
        "Normal trigger-only counts are zero and every full trigger has monitoring, concept, prompt, and response tokens.",
    ))

    regenerated = summarize_controller_actuator(
        records, plan, replicates=summary["bootstrap"]["replicates"], seed=summary["bootstrap"]["seed"]
    )
    checks.append(check(
        all(summary[key] == value for key, value in regenerated.items()),
        "deterministic_summary_regeneration",
        "The checked-in summary exactly regenerates from raw rows and the frozen plan.",
    ))
    denominators = [row["positive_suppression_denominator"] for row in summary["concepts"]]
    checks.append(check(
        len(denominators) == 11
        and all(metric["estimate"] > 0 and metric["ci_low"] > 0 for metric in denominators),
        "stable_positive_denominators",
        "All 11 concept denominators and bootstrap lower bounds are positive.",
    ))
    full_trigger = [
        row for row in summary["baseline_condition_macro"] if row["condition_id"] == "correct_trigger"
    ]
    checks.append(check(
        len(full_trigger) == 3
        and all(row["suppression_fraction"] == {"estimate": 1.0, "ci_low": 1.0, "ci_high": 1.0} for row in full_trigger),
        "exact_full_trigger_baseline_fraction",
        "The correct-trigger baseline is exactly one by the frozen normalization.",
    ))

    macro = {(row["scope"], row["intervention_id"]): row for row in summary["macro"]}
    contrasts = {
        (row["scope"], row["source_group"], row["direction"]): row
        for row in summary["selected_random_contrasts"]
    }
    def source_rule(source_group: str) -> bool:
        return all(
            macro[(scope, f"group.selected_attention_12.{source_group}.rescue")]["fraction"]["ci_low"] > 0
            and contrasts[(scope, source_group, "rescue")]["fraction_difference"]["ci_low"] > 0
            for scope in ("discovery", "validation")
        )
    stage_support = {
        "monitoring_recognition_supported": source_rule("monitoring_language"),
        "concept_identification_supported": source_rule("named_concept"),
        "response_processing_supported": source_rule("response"),
        "mlp_actuator_supported": all(
            macro[(scope, f"direct.selected_mlp_4.{direction}")]["fraction"]["ci_low"] > 0
            for scope in ("discovery", "validation") for direction in ("rescue", "induction")
        ),
    }
    expected_count = sum(stage_support.values())
    expected_overall = expected_count == 4
    expected_classification = (
        "supported_controller_actuator_decomposition" if expected_overall else
        "partially_supported_controller_actuator_decomposition" if expected_count >= 2 else
        "controller_actuator_decomposition_not_supported"
    )
    observed_evidence = summary["controller_actuator_evidence"]
    evidence_ok = (
        all(observed_evidence[key] == value for key, value in stage_support.items())
        and observed_evidence["supported_stage_count"] == expected_count
        and observed_evidence["overall_supported"] == expected_overall
        and observed_evidence["classification"] == expected_classification
    )
    checks.append(check(
        evidence_ok, "frozen_controller_actuator_rules",
        f"The four frozen stage rules independently yield {expected_classification}.",
    ))

    block_ok = all(
        row["patched_probe_score"] == next(
            baseline["probe_score"] for baseline in by_example[row["example_id"]]
            if baseline.get("condition_id") == ("normal" if row["direction"] == "rescue" else "correct_trigger")
        )
        for row in intervention_rows
        if row.get("direct_group_id") == "resid_post_layer12_positive_control"
    )
    checks.append(check(
        block_ok, "exact_block12_endpoints",
        "Every block-12 rescue/induction exactly reproduces its natural source baseline score.",
    ))
    component_structure = (
        plan["component_groups"]["selected_k16"] == selection["selected_candidates"]
        and plan["component_groups"]["random_attention_16"] == selection["random_control_candidates"]
        and plan["component_groups"]["random_attention_12"] == selection["random_control_candidates"][:12]
        and set(plan["component_groups"]["selected_attention_12"])
        | set(plan["component_groups"]["selected_mlp_4"])
        == set(selection["selected_candidates"])
    )
    checks.append(check(
        component_structure, "frozen_component_structure",
        "Selected attention/MLP groups and random prefixes exactly reproduce the Day 8 freeze.",
    ))
    layer_structure = all(
        len(row["head_ids"]) == 16
        and {int(item.split(".head_")[1]) for item in row["head_ids"]} == set(range(16))
        and len({item.split(".")[0] for item in row["head_ids"]}) == 1
        for row in specifications if row["family"] == "layer_source_scan"
    )
    checks.append(check(
        layer_structure, "complete_layer_head_scans",
        "Every layer/source scan contains all 16 query heads at exactly one frozen layer.",
    ))

    expected_tables = {
        "baseline-condition-summary.csv": 15,
        "controller-actuator-macro.csv": 645,
        "selected-random-source-contrasts.csv": 48,
        "layer-source-curves.csv": 273,
        "individual-head-source.csv": 144,
        "direct-output-summary.csv": 60,
        "controller-actuator-concepts.csv": 2365,
    }
    table_counts = {name: csv_row_count(result_dir / name) for name in expected_tables}
    checks.append(check(
        table_counts == expected_tables, "summary_table_shapes",
        f"CSV row counts are {table_counts}.",
    ))
    figure_names = (
        "layer-source-contributions.png", "selected-head-source-roles.png",
        "provisional-controller-actuator-mechanism.png",
    )
    figure_sizes = {name: Image.open(result_dir / name).size for name in figure_names}
    checks.append(check(
        all(size == (4800, 2700) for size in figure_sizes.values()),
        "publication_figure_dimensions", f"PNG dimensions are {figure_sizes}.",
    ))
    manifest_ok = len(manifest["files"]) == 17 and all(
        (result_dir / name).is_file()
        and metadata["sha256"] == sha256_file(result_dir / name)
        and metadata["bytes"] == (result_dir / name).stat().st_size
        for name, metadata in manifest["files"].items()
    )
    checks.append(check(
        manifest_ok, "artifact_manifest",
        "All 17 pre-audit artifacts match their recorded byte sizes and SHA-256 hashes.",
    ))
    checks.append(check(
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and summary["safety_split_accessed"] is False
        and manifest["safety_split_accessed"] is False
        and preflight["safety_split_accessed"] is False,
        "safety_split_isolation",
        "The safety unlock is absent and all Day 11 artifacts record no safety access.",
    ))

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1, "procedure": "day11-v1", "status": status,
        "check_count": len(checks), "checks": checks,
        "inputs": {path.name: sha256_file(path) for path in paths.values()},
        "validation_used_for_selection": False, "safety_split_accessed": False,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 11 audit: {status} ({len(checks)} checks).")
    if status != "pass":
        failed = ", ".join(row["name"] for row in checks if row["status"] == "fail")
        raise SystemExit(f"Day 11 audit failed: {failed}")


if __name__ == "__main__":
    main()
