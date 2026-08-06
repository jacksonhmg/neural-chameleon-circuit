#!/usr/bin/env python3
"""Independently audit the complete Day 10 sufficiency package."""

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

from neural_chameleon import load_experimental_split, sufficiency_specifications, summarize_sufficiency  # noqa: E402


RESULT_DIR = ROOT / "results/day-10"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
SELECTION_PATH = ROOT / "results/day-08/frozen-component-selection.json"
DAY09_PLAN_PATH = ROOT / "results/day-09/frozen-group-plan.json"
EXPECTED_COUNTS = (19_712, 3_168, 616)
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "sufficiency-audit.json")
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


def result(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def latest_file_commit(path: Path) -> str:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT, text=True
    ).strip()


def committed_matches(path: Path, commit: str) -> bool:
    relative = path.resolve().relative_to(ROOT).as_posix()
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    ) == path.read_bytes()


def csv_row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def selected_ids(rows: list[dict[str, Any]], salt: str, count: int) -> set[str]:
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{salt}:{row['example_id']}".encode()).hexdigest(),
            row["example_id"],
        ),
    )
    return {row["example_id"] for row in ordered[:count]}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    paths = {
        "exact": result_dir / "sufficiency-example-results.jsonl.gz",
        "dose": result_dir / "dose-response-results.jsonl.gz",
        "behavior": result_dir / "sufficiency-behavior-results.jsonl.gz",
        "plan": result_dir / "frozen-sufficiency-plan.json",
        "preflight": result_dir / "sufficiency-preflight.json",
        "summary": result_dir / "sufficiency-summary.json",
        "manifest": result_dir / "day10-artifacts.json",
    }
    exact = load_jsonl(paths["exact"])
    dose = load_jsonl(paths["dose"])
    behavior = load_jsonl(paths["behavior"])
    plan = json.loads(paths["plan"].read_text())
    preflight = json.loads(paths["preflight"].read_text())
    summary = json.loads(paths["summary"].read_text())
    manifest = json.loads(paths["manifest"].read_text())
    analysis_plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    specifications = sufficiency_specifications(plan)
    spec_by_id = {row["group_id"]: row for row in specifications}
    group_ids = set(spec_by_id)
    dose_group_ids = set(plan["dose_response"]["evaluated_group_ids"])
    checks: list[dict[str, str]] = []

    counts = (len(exact), len(dose), len(behavior))
    checks.append(result(
        counts == EXPECTED_COUNTS, "raw_record_counts",
        f"Exact, dose, and behavior counts are {counts}.",
    ))
    keys = (
        [(row["example_id"], row.get("group_id")) for row in exact],
        [(row["example_id"], row["group_id"], float(row["alpha"])) for row in dose],
        [(row["example_id"], row.get("group_id")) for row in behavior],
    )
    checks.append(result(
        all(len(values) == len(set(values)) for values in keys),
        "unique_raw_keys", "Every raw record has a unique key.",
    ))

    frozen = load_experimental_split("discovery") + load_experimental_split("validation")
    frozen_by_id = {row["example_id"]: row for row in frozen}
    by_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in exact:
        by_exact[row["example_id"]].append(row)
    checks.append(result(
        set(by_exact) == set(frozen_by_id) and all(
            len(rows) == 14 and {row.get("group_id") for row in rows} == {None, *group_ids}
            for rows in by_exact.values()
        ),
        "complete_all_benign_exact_grid",
        "All 1,408 benign examples contain one baseline and all 13 exact transplants.",
    ))

    positive_cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    behavior_cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in frozen:
        behavior_cells[(row["split"], row["concept"], row["label"])].append(row)
        if row["label"] == 1:
            positive_cells[(row["split"], row["concept"])].append(row)
    expected_dose_ids: set[str] = set()
    for rows in positive_cells.values():
        expected_dose_ids.update(selected_ids(rows, "day10-dose", 16))
    by_dose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dose:
        by_dose[row["example_id"]].append(row)
    dose_keys = {
        (group_id, alpha)
        for group_id in dose_group_ids for alpha in (0.25, 0.5, 0.75)
    }
    checks.append(result(
        len(expected_dose_ids) == 176
        and set(by_dose) == expected_dose_ids
        and all(
            len(rows) == 18
            and {(row["group_id"], float(row["alpha"])) for row in rows} == dose_keys
            for rows in by_dose.values()
        ),
        "complete_frozen_dose_grid",
        "The frozen 176-example positive subset has six groups at three interior alphas.",
    ))

    expected_behavior_ids: set[str] = set()
    for rows in behavior_cells.values():
        expected_behavior_ids.update(selected_ids(rows, "day08-behavior", 2))
    by_behavior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        by_behavior[row["example_id"]].append(row)
    checks.append(result(
        len(expected_behavior_ids) == 44
        and set(by_behavior) == expected_behavior_ids
        and all(
            len(rows) == 14 and {row.get("group_id") for row in rows} == {None, *group_ids}
            for rows in by_behavior.values()
        ),
        "complete_frozen_behavior_grid",
        "The frozen 44-example behavior subset has one baseline and all 13 transplants.",
    ))

    all_rows = (*exact, *dose, *behavior)
    checks.append(result(
        all(
            row["split"] == frozen_by_id[row["example_id"]]["split"]
            and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
            and row["label"] == frozen_by_id[row["example_id"]]["label"]
            and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
            and row["correct_trigger"] == analysis_plan["conditions"]["correct_triggers"][row["concept"]]
            for row in all_rows
        ),
        "frozen_metadata_and_triggers",
        "All records match frozen metadata, probe paths, and correct triggers.",
    ))
    plan_sha256 = sha256_file(paths["plan"])
    checks.append(result(
        all(
            row["procedure"] == "day10-v1"
            and row["freeze_id"] == "day04-v1"
            and row["model_repository"] == EXPECTED_REPOSITORY
            and row["model_revision"] == EXPECTED_REVISION
            and row["sufficiency_plan_sha256"] == plan_sha256
            and row["validation_used_for_selection"] is False
            and row["safety_split_accessed"] is False
            for row in all_rows
        ),
        "pinned_model_procedure_and_plan",
        "Every record names the frozen checkpoint, procedure, and plan.",
    ))
    checks.append(result(
        all(
            len({row["response_ids_sha256"] for row in rows}) == 1
            and len({row["response_token_count"] for row in rows}) == 1
            for rows in (*by_exact.values(), *by_dose.values(), *by_behavior.values())
        ),
        "exact_response_pairing",
        "Every example reuses one response-token sequence across conditions.",
    ))
    finite = all(
        all(
            math.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1
            for field in (("normal_probe_score", "triggered_probe_score") if row["record_type"] == "baseline" else ("patched_probe_score",))
        ) for row in exact
    ) and all(
        math.isfinite(float(row["patched_probe_score"])) and 0 <= float(row["patched_probe_score"]) <= 1
        for row in dose
    ) and all(
        all(
            math.isfinite(float(row[field]))
            for field in (("normal_response_nll", "triggered_response_nll") if row["record_type"] == "baseline" else ("patched_response_nll",))
        ) for row in behavior
    )
    checks.append(result(finite, "finite_scores_and_nll", "All probabilities are bounded and every metric is finite."))

    transplant_rows = [row for row in all_rows if row["record_type"] != "baseline"]
    checks.append(result(
        all(
            row["group_id"] in spec_by_id
            and row["group_role"] == spec_by_id[row["group_id"]]["group_role"]
            and row["set_size"] == spec_by_id[row["group_id"]]["set_size"]
            and row["candidate_ids"] == spec_by_id[row["group_id"]]["candidate_ids"]
            and row["residual_sites"] == spec_by_id[row["group_id"]]["residual_sites"]
            and row["group_members_sha256"] == hashlib.sha256(json.dumps(
                {"candidate_ids": row["candidate_ids"], "residual_sites": row["residual_sites"]},
                sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            for row in transplant_rows
        ),
        "exact_frozen_group_membership",
        "Every transplant reproduces its frozen ordered membership and hash.",
    ))
    checks.append(result(
        all(
            row["direction"] == "induction" and row["destination_condition"] == "normal"
            and row["source_condition"] == (
                "normal_to_correct_trigger_interpolation"
                if row["record_type"] == "interpolated_transplant" else "correct_trigger"
            )
            for row in transplant_rows
        ),
        "causal_direction_mapping",
        "All natural activations are mapped from correct-trigger source to normal destination.",
    ))

    expected_commits = {
        "selection_commit": latest_file_commit(SELECTION_PATH),
        "day09_procedure_commit": latest_file_commit(DAY09_PLAN_PATH),
        "day10_procedure_commit": latest_file_commit(paths["plan"]),
        "implementation_commit": latest_file_commit(ROOT / "scripts/day10_run_sufficiency.py"),
    }
    committed_ok = all(
        {row[field] for row in all_rows} == {commit}
        for field, commit in expected_commits.items()
    ) and all(
        committed_matches(path, expected_commits[field])
        for field, path in (
            ("selection_commit", SELECTION_PATH),
            ("day09_procedure_commit", DAY09_PLAN_PATH),
            ("day10_procedure_commit", paths["plan"]),
            ("implementation_commit", ROOT / "scripts/day10_run_sufficiency.py"),
        )
    )
    checks.append(result(
        committed_ok, "committed_inputs_and_implementation",
        "Records name byte-identical committed selection, plans, and evaluator.",
    ))

    preflight_ok = (
        preflight["status"] == "pass"
        and preflight["same_shape_identity_check_count"] == 26
        and len(preflight["same_shape_identity_checks"]) == 26
        and all(row["exact"] for row in preflight["same_shape_identity_checks"])
        and all(row["within_tolerance"] for row in preflight["vectorized_probe_score_checks"])
        and all(row["within_tolerance"] for row in preflight["vectorized_response_nll_checks"])
        and len(preflight["interpolation_endpoint_checks"]) == 4
        and all(row["exact"] for row in preflight["interpolation_endpoint_checks"])
        and len(preflight["activation_norm_checks"]) == 2
        and all(row["within_bound"] for row in preflight["activation_norm_checks"])
        and preflight["group_member_order_check"]["exact"]
        and preflight["registered_hook_count_after_checks"] == 0
    )
    checks.append(result(
        preflight_ok, "real_checkpoint_preflight",
        "Identities, vector checks, endpoints, norm bounds, order, and hook cleanup all pass.",
    ))

    regenerated = summarize_sufficiency(
        exact, dose, behavior, plan,
        replicates=summary["bootstrap"]["replicates"], seed=summary["bootstrap"]["seed"],
    )
    checks.append(result(
        all(summary[key] == value for key, value in regenerated.items()),
        "deterministic_summary_regeneration",
        "The summary exactly regenerates from raw archives and the frozen plan.",
    ))
    denominators = [row["positive_suppression_denominator"] for row in summary["concepts"]]
    denominators += [row["positive_suppression_denominator"] for row in summary["dose_response"]["concepts"]]
    checks.append(result(
        len(denominators) == 22
        and all(metric["estimate"] > 0 and metric["ci_low"] > 0 for metric in denominators),
        "stable_positive_suppression_denominators",
        "All full-grid and dose-subset denominators have positive points and lower bounds.",
    ))

    macro = {(row["scope"], row["group_id"], row["label"]): row for row in summary["macro"]}
    contrasts = {
        (row["scope"], row["label"]): row
        for row in summary["selected_random_contrasts"] if row["comparison"] == "k16"
    }
    supported = all(
        macro[(scope, "selected_k16", 1)]["fraction"]["ci_low"] > 0
        and contrasts[(scope, 1)]["fraction_difference"]["ci_low"] > 0
        for scope in ("discovery", "validation")
    )
    classification = "not_supported"
    if supported:
        classification = "near_complete_sufficiency" if all(
            macro[(scope, "selected_k16", 1)]["fraction"]["estimate"] >= 0.9
            for scope in ("discovery", "validation")
        ) else "partial_sufficiency"
    evidence = summary["sufficiency_evidence"]
    checks.append(result(
        evidence["supported"] == supported and evidence["classification"] == classification,
        "frozen_sufficiency_rule",
        f"The preregistered CI and 90% rules independently yield {classification}.",
    ))
    monotonic = {
        (row["scope"], row["group_id"]): row
        for row in summary["dose_response"]["monotonicity"]
    }
    dose_supported = all(
        monotonic[(scope, "selected_k16")]["nondecreasing"]
        and all(later >= earlier for earlier, later in zip(
            monotonic[(scope, "selected_k16")]["point_estimates"],
            monotonic[(scope, "selected_k16")]["point_estimates"][1:],
        ))
        for scope in ("discovery", "validation")
    )
    checks.append(result(
        dose_supported == summary["dose_response"]["selected_k16_dose_response_supported"]
        == evidence["selected_k16_dose_response_supported"],
        "frozen_dose_response_rule",
        f"The selected-K16 five-alpha monotonicity rule independently yields {dose_supported}.",
    ))

    block_rows = [row for row in exact if row.get("group_id") == "resid_post_layer12_positive_control"]
    checks.append(result(
        len(block_rows) == 1408 and all(
            row["patched_probe_score"] == next(
                base["triggered_probe_score"] for base in by_exact[row["example_id"]]
                if base["record_type"] == "baseline"
            ) for row in block_rows
        ),
        "exact_block12_positive_control",
        "Complete block-12 output exactly reproduces the triggered probe score.",
    ))
    checks.append(result(
        all(float(row["interpolation_bound_ratio_max"]) <= 1.001 for row in dose)
        and summary["activation_norms"]["dose_interpolation_bound_ratio_max"] <= 1.001,
        "activation_interpolation_norm_bound",
        "Every interior interpolation obeys the frozen 1.001 RMS bound.",
    ))
    combined = spec_by_id["selected_k16_plus_resid_post_layer08"]
    context_ok = (
        combined["candidate_ids"] == spec_by_id["selected_k16"]["candidate_ids"]
        and combined["residual_sites"] == ["resid_post_layer_08"]
        and spec_by_id["resid_post_layer08_context"]["candidate_ids"] == []
        and spec_by_id["resid_post_layer08_context"]["residual_sites"] == ["resid_post_layer_08"]
    )
    checks.append(result(
        context_ok, "surrounding_residual_context_structure",
        "The context comparison adds only entry residual layer 8 to the identical selected K16 set.",
    ))

    expected_tables = {
        "sufficiency-macro.csv": 78,
        "sufficiency-concepts.csv": 286,
        "sufficiency-control-contrasts.csv": 30,
        "residual-context-increment.csv": 6,
        "dose-response-summary.csv": 90,
        "sufficiency-behavior-summary.csv": 78,
    }
    table_counts = {name: csv_row_count(result_dir / name) for name in expected_tables}
    checks.append(result(
        table_counts == expected_tables, "summary_table_shapes",
        f"CSV row counts are {table_counts}.",
    ))
    figure_names = ("sufficiency-overview.png", "sufficiency-dose-response.png")
    figure_sizes = {name: Image.open(result_dir / name).size for name in figure_names}
    checks.append(result(
        all(size == (4800, 2700) for size in figure_sizes.values()),
        "publication_figure_dimensions", f"PNG dimensions are {figure_sizes}.",
    ))
    manifest_ok = len(manifest["files"]) == 16 and all(
        (result_dir / name).is_file()
        and metadata["sha256"] == sha256_file(result_dir / name)
        and metadata["bytes"] == (result_dir / name).stat().st_size
        for name, metadata in manifest["files"].items()
    )
    checks.append(result(
        manifest_ok, "artifact_manifest",
        "All 16 pre-audit artifacts match their recorded sizes and hashes.",
    ))
    checks.append(result(
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and summary["safety_split_accessed"] is False
        and manifest["safety_split_accessed"] is False
        and preflight["safety_split_accessed"] is False,
        "safety_split_isolation",
        "The safety unlock is absent and all Day 10 artifacts record no safety access.",
    ))

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day10-v1",
        "status": status,
        "check_count": len(checks),
        "checks": checks,
        "inputs": {path.name: sha256_file(path) for path in paths.values()},
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 10 audit: {status} ({len(checks)} checks).")
    if status != "pass":
        failed = ", ".join(row["name"] for row in checks if row["status"] == "fail")
        raise SystemExit(f"Day 10 audit failed: {failed}")


if __name__ == "__main__":
    main()
