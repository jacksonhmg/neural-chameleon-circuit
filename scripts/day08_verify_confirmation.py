#!/usr/bin/env python3
"""Independently audit the complete Day 8 component package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    CANDIDATES,
    CANDIDATE_BY_ID,
    component_set_sha256,
    load_experimental_split,
    summarize_component_confirmation,
)


RESULT_DIR = ROOT / "results/day-08"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_EXACT_ROWS = 37568
EXPECTED_BEHAVIOR_ROWS = 1452
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "component-confirmation-audit.json"
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


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    discovery_path = result_dir / "discovery-candidate-results.jsonl.gz"
    confirmation_path = result_dir / "confirmation-example-results.jsonl.gz"
    behavior_path = result_dir / "behavior-example-results.jsonl.gz"
    discovery = load_jsonl(discovery_path)
    confirmation = load_jsonl(confirmation_path)
    behavior = load_jsonl(behavior_path)
    selection_path = result_dir / "frozen-component-selection.json"
    selection = json.loads(selection_path.read_text())
    summary = json.loads((result_dir / "component-confirmation-summary.json").read_text())
    discovery_summary = json.loads(
        (result_dir / "discovery-candidate-summary.json").read_text()
    )
    discovery_audit = json.loads(
        (result_dir / "discovery-selection-audit.json").read_text()
    )
    identity = json.loads((result_dir / "discovery-identity-audit.json").read_text())
    preflight = json.loads((result_dir / "confirmation-preflight.json").read_text())
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    checks: list[dict[str, str]] = []

    checks.append(
        check(
            len(discovery) == 17664
            and len(confirmation) == EXPECTED_EXACT_ROWS
            and len(behavior) == EXPECTED_BEHAVIOR_ROWS,
            "raw_record_counts",
            f"Discovery, confirmation, and behavior counts are {len(discovery)}, "
            f"{len(confirmation)}, and {len(behavior)}.",
        )
    )
    discovery_keys = [
        (row["example_id"], row["record_type"], row.get("candidate_id"))
        for row in discovery
    ]
    confirmation_keys = [
        (
            row["example_id"],
            row["record_type"],
            row.get("candidate_id"),
            row.get("direction"),
        )
        for row in confirmation
    ]
    behavior_keys = [
        (
            row["example_id"],
            row["record_type"],
            row.get("candidate_id"),
            row.get("direction"),
        )
        for row in behavior
    ]
    checks.append(
        check(
            len(discovery_keys) == len(set(discovery_keys))
            and len(confirmation_keys) == len(set(confirmation_keys))
            and len(behavior_keys) == len(set(behavior_keys)),
            "unique_raw_keys",
            "Every discovery, confirmation, and behavior row has a unique scoped key.",
        )
    )

    frozen = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    frozen_by_id = {row["example_id"]: row for row in frozen}
    positive_ids = {
        example_id for example_id, row in frozen_by_id.items() if row["label"] == 1
    }
    discovery_positive_ids = {
        example_id
        for example_id in positive_ids
        if frozen_by_id[example_id]["split"] == "discovery"
    }
    confirmation_ids = {row["example_id"] for row in confirmation}
    checks.append(
        check(
            confirmation_ids == positive_ids
            and {row["example_id"] for row in discovery} == discovery_positive_ids,
            "exact_positive_example_coverage",
            "The exact grids contain every and only frozen discovery/validation positives in scope.",
        )
    )

    selected = selection["selected_candidates"]
    random_controls = selection["random_control_candidates"]
    all_ids = set(selected) | set(random_controls)
    by_exact_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in confirmation:
        by_exact_example[row["example_id"]].append(row)
    exact_complete = True
    for example_id, rows in by_exact_example.items():
        split = frozen_by_id[example_id]["split"]
        observed = {
            (row["record_type"], row.get("candidate_id"), row.get("direction"))
            for row in rows
        }
        expected = {("baseline", None, None)}
        directions = ("induction",) if split == "discovery" else ("rescue", "induction")
        expected.update(
            ("patch", candidate_id, direction)
            for candidate_id in all_ids
            for direction in directions
        )
        exact_complete &= observed == expected
    checks.append(
        check(
            exact_complete,
            "complete_confirmation_grids",
            "Each discovery positive has baseline+32 induction rows; each validation positive has baseline+64 rescue/induction rows.",
        )
    )

    by_behavior_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in behavior:
        by_behavior_example[row["example_id"]].append(row)
    behavior_complete = all(
        {
            (row["record_type"], row.get("candidate_id"), row.get("direction"))
            for row in rows
        }
        == {("baseline", None, None)}
        | {
            ("patch", candidate_id, direction)
            for candidate_id in selected
            for direction in ("rescue", "induction")
        }
        for rows in by_behavior_example.values()
    )
    expected_behavior_ids = set()
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in frozen:
        groups[(record["split"], record["concept"], record["label"])].append(record)
    for records in groups.values():
        ordered = sorted(
            records,
            key=lambda record: (
                hashlib.sha256(
                    f"day08-behavior:{record['example_id']}".encode()
                ).hexdigest(),
                record["example_id"],
            ),
        )
        expected_behavior_ids.update(row["example_id"] for row in ordered[:2])
    checks.append(
        check(
            behavior_complete
            and set(by_behavior_example) == expected_behavior_ids
            and len(expected_behavior_ids) == 44,
            "complete_frozen_behavior_grid",
            "The behavior grid uses exactly the two preregistered examples per concept/class and all 16×2 patches.",
        )
    )

    frozen_metadata = all(
        row["split"] == frozen_by_id[row["example_id"]]["split"]
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == frozen_by_id[row["example_id"]]["label"]
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["correct_trigger"]
        == plan["conditions"]["correct_triggers"][row["concept"]]
        for row in (*confirmation, *behavior)
    )
    checks.append(
        check(
            frozen_metadata,
            "frozen_metadata_and_triggers",
            "Split, concept, label, probe, and correct trigger match the frozen records.",
        )
    )
    model_metadata = all(
        row["freeze_id"] == "day04-v1"
        and row["procedure"] == "day08-v1"
        and row["model_repository"] == EXPECTED_REPOSITORY
        and row["model_revision"] == EXPECTED_REVISION
        and row["component_set_sha256"] == selection["component_set_sha256"]
        for row in (*confirmation, *behavior)
    )
    checks.append(
        check(
            model_metadata,
            "pinned_model_procedure_and_component_set",
            "Every confirmation row names the frozen model, revision, procedure, and component-set hash.",
        )
    )

    response_pairing = all(
        len({row["response_ids_sha256"] for row in rows}) == 1
        and len({row["response_token_count"] for row in rows}) == 1
        for rows in (*by_exact_example.values(), *by_behavior_example.values())
    )
    checks.append(
        check(
            response_pairing,
            "exact_response_pairing",
            "All conditions for each example reuse one response-token hash and count.",
        )
    )

    finite_exact = all(
        math.isfinite(float(row[field])) and 0 <= float(row[field]) <= 1
        for row in confirmation
        for field in (
            ("normal_probe_score", "triggered_probe_score")
            if row["record_type"] == "baseline"
            else ("patched_probe_score",)
        )
    )
    finite_behavior = all(
        math.isfinite(float(row[field]))
        for row in behavior
        for field in (
            ("normal_response_nll", "triggered_response_nll")
            if row["record_type"] == "baseline"
            else ("patched_response_nll",)
        )
    )
    checks.append(
        check(
            finite_exact and finite_behavior,
            "finite_scores_and_nll",
            "All probe probabilities and fixed-continuation NLL values are finite and probabilities are bounded.",
        )
    )

    direction_mapping = all(
        row["record_type"] == "baseline"
        or (
            row["source_condition"] == "normal"
            and row["destination_condition"] == "correct_trigger"
            if row["direction"] == "rescue"
            else row["source_condition"] == "correct_trigger"
            and row["destination_condition"] == "normal"
        )
        for row in confirmation
    ) and all(
        row["record_type"] == "baseline"
        or row["destination_condition"]
        == ("correct_trigger" if row["direction"] == "rescue" else "normal")
        for row in behavior
    )
    checks.append(
        check(
            direction_mapping,
            "causal_direction_mapping",
            "Rescue and induction source/destination conditions are correctly oriented.",
        )
    )

    role_scope = all(
        row["record_type"] == "baseline"
        or (
            row["candidate_id"] in all_ids
            and row["candidate_role"]
            == ("selected" if row["candidate_id"] in selected else "random_control")
            and row["layer"] == CANDIDATE_BY_ID[row["candidate_id"]].layer
            and row["component_type"]
            == CANDIDATE_BY_ID[row["candidate_id"]].component_type
            and row["head"] == CANDIDATE_BY_ID[row["candidate_id"]].head
        )
        for row in confirmation
    ) and all(
        row["record_type"] == "baseline"
        or row["candidate_id"] in selected
        and row["candidate_role"] == "selected"
        for row in behavior
    )
    checks.append(
        check(
            role_scope,
            "frozen_candidate_roles_and_sites",
            "Every patch uses a committed selected/control identity and its exact frozen site metadata.",
        )
    )

    ordered = [row["candidate_id"] for row in selection["ordered_top_16"]]
    outside = [
        candidate.candidate_id
        for candidate in CANDIDATES
        if candidate.candidate_id not in ordered
    ]
    expected_random = sorted(
        outside,
        key=lambda candidate_id: (
            hashlib.sha256(f"42:{candidate_id}".encode()).hexdigest(),
            candidate_id,
        ),
    )[:16]
    selection_valid = (
        ordered == selected
        and selection["final_k"] == 16
        and selection["component_set_sha256"] == component_set_sha256(ordered, 16)
        and random_controls == expected_random
        and selection["discovery_summary_sha256"]
        == sha256_file(result_dir / "discovery-candidate-summary.json")
        and discovery_summary["selected_candidates"] == selected
        and discovery_summary["random_control_candidates"] == random_controls
        and discovery_summary["validation_used_for_selection"] is False
    )
    checks.append(
        check(
            selection_valid,
            "selection_and_control_regeneration",
            "K, ordered identities, component hash, discovery-summary hash, and deterministic controls regenerate.",
        )
    )

    commits = {row["selection_commit"] for row in (*confirmation, *behavior)}
    committed_selection = False
    if len(commits) == 1:
        commit = next(iter(commits))
        try:
            committed = subprocess.check_output(
                ["git", "show", f"{commit}:results/day-08/frozen-component-selection.json"],
                cwd=ROOT,
            )
            committed_selection = committed == selection_path.read_bytes()
        except subprocess.CalledProcessError:
            committed_selection = False
    checks.append(
        check(
            committed_selection,
            "selection_committed_before_confirmation",
            f"Every confirmation row names one commit containing the byte-identical frozen selection: {sorted(commits)}.",
        )
    )

    controls_valid = (
        discovery_audit["status"] == "pass"
        and identity["status"] == "pass"
        and identity["full_forward_check_count"] == 2
        and identity["identity_check_count"] == 136
        and identity["vectorized_check_count"] == 2
        and preflight["status"] == "pass"
        and preflight["vectorized_full_forward_check_count"] == 2
        and all(row["within_tolerance"] for row in preflight["vectorized_full_forward_checks"])
    )
    checks.append(
        check(
            controls_valid,
            "preflight_identity_and_discovery_audits",
            "Discovery audit, two full forwards, 136 identities, vector checks, and confirmation NLL preflight all pass.",
        )
    )

    regenerated = summarize_component_confirmation(
        discovery,
        confirmation,
        behavior,
        selection,
        replicates=10000,
        seed=42,
    )
    core_keys = (
        "schema_version",
        "procedure",
        "bootstrap",
        "selected_candidates",
        "random_control_candidates",
        "component_set_sha256",
        "exact",
        "behavior",
        "validation_used_for_selection",
        "safety_split_accessed",
    )
    checks.append(
        check(
            regenerated == {key: summary[key] for key in core_keys},
            "deterministic_summary_regeneration",
            "All exact, consistency, control, behavior, contrast, and interval values regenerate exactly.",
        )
    )

    stable_denominators = all(
        row["suppression_denominator"]["ci_low"] > 0
        for row in summary["exact"]["concepts"]
    )
    summary_counts = (
        len(summary["exact"]["concepts"]),
        len(summary["exact"]["macro"]),
        len(summary["exact"]["role_aggregates"]),
        len(summary["exact"]["role_contrasts"]),
        len(summary["exact"]["same_layer_controls"]),
        len(summary["behavior"]["concept_class_cells"]),
        len(summary["behavior"]["macro"]),
    )
    checks.append(
        check(
            stable_denominators and summary_counts == (11, 128, 8, 4, 16, 704, 192),
            "summary_scope_and_denominator_coverage",
            f"All denominators are stable and summary cell counts are {summary_counts}.",
        )
    )

    same_layer_valid = all(
        row["control_candidate_count"]
        == 17
        - sum(CANDIDATE_BY_ID[item].layer == row["layer"] for item in selected)
        and not set(row["control_candidates"]) & set(selected)
        and all(CANDIDATE_BY_ID[item].layer == row["layer"] for item in row["control_candidates"])
        for row in summary["exact"]["same_layer_controls"]
    )
    checks.append(
        check(
            same_layer_valid,
            "complete_nonselected_same_layer_controls",
            "Every selected component is compared with every nonselected candidate at its layer.",
        )
    )

    raw_hashes_valid = (
        summary["discovery_results_sha256"] == sha256_file(discovery_path)
        and summary["confirmation_results_sha256"] == sha256_file(confirmation_path)
        and summary["behavior_results_sha256"] == sha256_file(behavior_path)
        and summary["discovery_record_count"] == len(discovery)
        and summary["confirmation_record_count"] == len(confirmation)
        and summary["behavior_record_count"] == len(behavior)
    )
    checks.append(
        check(
            raw_hashes_valid,
            "summary_raw_hashes_and_counts",
            "Summary hashes and counts match all three deterministic raw archives.",
        )
    )

    manifest = json.loads((result_dir / "day08-artifacts.json").read_text())
    artifact_integrity = all(
        (result_dir / name).stat().st_size == metadata["bytes"]
        and sha256_file(result_dir / name) == metadata["sha256"]
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            artifact_integrity,
            "artifact_manifest_integrity",
            "Raw archives, freezes, audits, summaries, tables, and figures match their manifest hashes.",
        )
    )

    figures_valid = True
    figure_sizes = {
        "discovery-candidate-ranking": (4800, 2700),
        "component-confirmation-overview": (4800, 2700),
        "component-controls-behavior": (4800, 2700),
    }
    for stem, size in figure_sizes.items():
        with Image.open(result_dir / f"{stem}.png") as image:
            figures_valid &= image.size == size and image.mode == "RGBA"
        figures_valid &= (result_dir / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
    checks.append(
        check(
            figures_valid,
            "publication_figure_files",
            "All three 300-DPI figures are 4,800×2,700 RGBA with companion PDFs.",
        )
    )

    isolation = (
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and selection["validation_used_for_selection"] is False
        and selection["safety_split_accessed"] is False
        and summary["validation_used_for_selection"] is False
        and summary["safety_split_accessed"] is False
        and all(row["validation_used_for_selection"] is False for row in (*confirmation, *behavior))
        and all(row["safety_split_accessed"] is False for row in (*confirmation, *behavior))
    )
    checks.append(
        check(
            isolation,
            "validation_selection_and_safety_isolation",
            "Validation did not alter selection, and safety remains locked and unaccessed.",
        )
    )

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "status": status,
        "check_count": len(checks),
        "checks": checks,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 8 confirmation audit: {status} ({len(checks)} checks).")
    if status != "pass":
        failed = [row["name"] for row in checks if row["status"] == "fail"]
        raise SystemExit(f"Day 8 confirmation audit failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
