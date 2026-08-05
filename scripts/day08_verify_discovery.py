#!/usr/bin/env python3
"""Independently audit the Day 8 discovery selection freeze."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
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
    summarize_discovery_candidates,
)


RESULT_DIR = ROOT / "results/day-08"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_ROWS = 17664
EXPECTED_MODEL = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "discovery-selection-audit.json"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    raw_path = result_dir / "discovery-candidate-results.jsonl.gz"
    rows = load_rows(raw_path)
    summary = json.loads((result_dir / "discovery-candidate-summary.json").read_text())
    selection = json.loads((result_dir / "frozen-component-selection.json").read_text())
    identity = json.loads((result_dir / "discovery-identity-audit.json").read_text())
    checks = []

    checks.append(
        check(
            len(rows) == EXPECTED_ROWS,
            "raw_record_count",
            f"Found {len(rows)} rows; expected {EXPECTED_ROWS}.",
        )
    )
    raw_keys = [
        (
            row["example_id"],
            "baseline" if row["record_type"] == "baseline" else row["candidate_id"],
        )
        for row in rows
    ]
    checks.append(
        check(
            len(raw_keys) == len(set(raw_keys)),
            "unique_raw_keys",
            "Every example/baseline-or-candidate key is unique.",
        )
    )

    frozen = [
        record
        for record in load_experimental_split("discovery")
        if record["label"] == 1
    ]
    frozen_by_id = {record["example_id"]: record for record in frozen}
    observed_ids = {row["example_id"] for row in rows}
    checks.append(
        check(
            observed_ids == set(frozen_by_id) and len(frozen_by_id) == 256,
            "exact_discovery_positive_coverage",
            "Rows contain every and only the 256 frozen discovery positives.",
        )
    )

    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[row["example_id"]].append(row)
    complete_grid = all(
        len(example_rows) == 69
        and sum(row["record_type"] == "baseline" for row in example_rows) == 1
        and {
            row["candidate_id"]
            for row in example_rows
            if row["record_type"] == "candidate"
        }
        == set(CANDIDATE_BY_ID)
        for example_rows in by_example.values()
    )
    checks.append(
        check(
            complete_grid,
            "complete_68_candidate_grid",
            "Each example has one baseline and all 68 exact candidate rows.",
        )
    )

    metadata_valid = all(
        row["procedure"] == "day08-v1"
        and row["freeze_id"] == "day04-v1"
        and row["model_repository"] == EXPECTED_MODEL
        and row["model_revision"] == EXPECTED_REVISION
        and row["execution_mode"] == "truncated_forward_at_resid_post_12"
        and row["split"] == "discovery"
        and row["label"] == 1
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["validation_used_for_selection"] is False
        and row["safety_split_accessed"] is False
        for row in rows
    )
    checks.append(
        check(
            metadata_valid,
            "frozen_metadata_and_isolation",
            "Every row matches the pinned model, frozen example, execution, and split role.",
        )
    )

    response_pairing = all(
        len({row["response_ids_sha256"] for row in example_rows}) == 1
        and len({row["response_token_count"] for row in example_rows}) == 1
        for example_rows in by_example.values()
    )
    checks.append(
        check(
            response_pairing,
            "exact_response_pairing",
            "All conditions and candidates for an example share response tokens.",
        )
    )

    candidate_rows = [row for row in rows if row["record_type"] == "candidate"]
    site_valid = all(
        row["candidate_id"] in CANDIDATE_BY_ID
        and row["layer"] == CANDIDATE_BY_ID[row["candidate_id"]].layer
        and row["component_type"]
        == CANDIDATE_BY_ID[row["candidate_id"]].component_type
        and row["head"] == CANDIDATE_BY_ID[row["candidate_id"]].head
        and row["source_condition"] == "normal"
        and row["destination_condition"] == "correct_trigger"
        and row["direction"] == "rescue"
        for row in candidate_rows
    )
    checks.append(
        check(
            site_valid,
            "candidate_site_and_direction_scope",
            "All candidate IDs, sites, and rescue directions match the 68-site freeze.",
        )
    )

    finite = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key.endswith("probe_score") or key.startswith("screen_")
    )
    checks.append(
        check(
            finite,
            "finite_scores_and_screens",
            "All baseline, exact-patch, and screening values are finite.",
        )
    )

    layer_ranking = json.loads(
        (ROOT / "results/day-06/coarse-layer-ranking.json").read_text()
    )
    checks.append(
        check(
            layer_ranking["retained_top_four_layers"] == [12, 11, 10, 9]
            and len(CANDIDATES) == 68,
            "frozen_layer_and_candidate_universe",
            "The candidate universe is exactly 68 sites at the Day 6 layers.",
        )
    )

    identity_valid = (
        identity["status"] == "pass"
        and identity["complete_forward_check_count"] == 2
        and identity["identity_check_count"] == 136
        and identity["vectorized_equivalence_check_count"] == 2
        and identity["screening_metrics_finite"] is True
        and all(
            row["exact"]
            for group in (
                identity["complete_forward_checks"],
                identity["identity_checks"],
                identity["vectorized_equivalence_checks"],
            )
            for row in group
        )
    )
    checks.append(
        check(
            identity_valid,
            "real_checkpoint_preflight",
            "Full forwards, 136 identities, two vector checks, and screens pass.",
        )
    )

    regenerated = summarize_discovery_candidates(rows, replicates=10000, seed=42)
    summary_core = {
        key: summary[key]
        for key in (
            "schema_version",
            "procedure",
            "bootstrap",
            "eligible_layers",
            "candidate_count",
            "concepts",
            "candidates",
            "eligible_candidate_count",
            "ordered_eligible_candidates",
            "frozen_top_16",
            "nested_set_sizes",
            "final_k",
            "selected_candidates",
            "random_control_candidates",
            "component_set_sha256",
            "screening_evaluation",
            "validation_used_for_selection",
            "safety_split_accessed",
        )
    }
    checks.append(
        check(
            regenerated == summary_core,
            "deterministic_summary_regeneration",
            "All exact effects, screens, intervals, ranks, and controls regenerate exactly.",
        )
    )

    stable_denominators = all(
        concept["suppression_denominator"]["ci_low"] > 0
        for concept in summary["concepts"]
    )
    checks.append(
        check(
            stable_denominators,
            "stable_discovery_denominators",
            "All four discovery suppression intervals are strictly positive.",
        )
    )

    selected = summary["selected_candidates"]
    random = summary["random_control_candidates"]
    expected_random = sorted(
        (
            candidate.candidate_id
            for candidate in CANDIDATES
            if candidate.candidate_id not in summary["frozen_top_16"]
        ),
        key=lambda candidate_id: (
            hashlib.sha256(f"42:{candidate_id}".encode()).hexdigest(),
            candidate_id,
        ),
    )[: summary["final_k"]]
    selection_valid = (
        summary["final_k"] == 16
        and len(selected) == len(set(selected)) == 16
        and len(random) == len(set(random)) == 16
        and not set(selected) & set(random)
        and random == expected_random
        and selection["selected_candidates"] == selected
        and selection["random_control_candidates"] == random
        and selection["component_set_sha256"]
        == component_set_sha256(summary["frozen_top_16"], 16)
        and selection["validation_used_for_selection"] is False
        and selection["safety_split_accessed"] is False
    )
    checks.append(
        check(
            selection_valid,
            "frozen_selection_and_random_controls",
            "Top 16, K, component hash, and distinct hash-selected controls reproduce.",
        )
    )

    gate_valid = all(
        next(
            row
            for row in summary["candidates"]
            if row["candidate_id"] == candidate_id
        )["shared_candidate_gate"]
        for candidate_id in summary["frozen_top_16"]
    )
    checks.append(
        check(
            gate_valid,
            "shared_candidate_gate",
            "Every frozen top-16 candidate is positive on at least three discovery concepts.",
        )
    )

    manifest = json.loads((result_dir / "discovery-artifacts.json").read_text())
    artifact_valid = all(
        (result_dir / name).stat().st_size == metadata["bytes"]
        and sha256_file(result_dir / name) == metadata["sha256"]
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            artifact_valid,
            "discovery_artifact_integrity",
            "Raw, identity, summary, ranking, selection, and figures match hashes.",
        )
    )

    with Image.open(result_dir / "discovery-candidate-ranking.png") as image:
        figure_valid = image.size == (4800, 2700) and image.mode == "RGBA"
    figure_valid &= (
        result_dir / "discovery-candidate-ranking.pdf"
    ).read_bytes().startswith(b"%PDF-")
    checks.append(
        check(
            figure_valid,
            "discovery_figure_files",
            "The 300-DPI 4,800 by 2,700 RGBA figure and PDF are valid.",
        )
    )

    isolation = (
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and summary["validation_used_for_selection"] is False
        and summary["safety_split_accessed"] is False
        and identity["validation_split_accessed"] is False
        and identity["safety_split_accessed"] is False
    )
    checks.append(
        check(
            isolation,
            "validation_and_safety_isolation",
            "No individual-component validation or safety data entered selection.",
        )
    )

    completion = (
        len(candidate_rows) == 256 * 68
        and summary["candidate_count"] == 68
        and summary["final_k"] == 16
        and stable_denominators
        and selection_valid
    )
    checks.append(
        check(
            completion,
            "discovery_selection_gate",
            "All 68 candidates are exactly tested and the frozen K=16 set is complete.",
        )
    )

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day08-v1",
        "verified_on": "2026-08-05",
        "stage": "discovery-selection-freeze",
        "status": status,
        "raw_record_count": len(rows),
        "check_count": len(checks),
        "checks": checks,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 8 discovery audit: {status} ({len(checks)} checks).")
    if status != "pass":
        for row in checks:
            if row["status"] == "fail":
                print(f"FAIL: {row['name']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
