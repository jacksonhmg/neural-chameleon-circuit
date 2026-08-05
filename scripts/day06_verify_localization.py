#!/usr/bin/env python3
"""Independently audit the complete Day 6 localization result package."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import TokenRegion, load_experimental_split, summarize_localization  # noqa: E402


RESULT_DIR = ROOT / "results/day-06"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
EXPECTED_ROWS = 256 * (2 + 42 * 3 * 2)
EXPECTED_KEYS_PER_EXAMPLE = 2 + 42 * 3 * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "localization-audit.json"
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    raw_path = result_dir / "localization-example-results.jsonl.gz"
    rows = load_jsonl(raw_path)
    summary = json.loads((result_dir / "localization-summary.json").read_text())
    ranking = json.loads((result_dir / "coarse-layer-ranking.json").read_text())
    identity = json.loads((result_dir / "identity-audit.json").read_text())
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    checks: list[dict[str, str]] = []

    checks.append(
        check(
            len(rows) == EXPECTED_ROWS,
            "raw_record_count",
            f"Found {len(rows)} rows; expected {EXPECTED_ROWS}.",
        )
    )
    raw_keys = [(row["example_id"], row["key"]) for row in rows]
    checks.append(
        check(
            len(raw_keys) == len(set(raw_keys)),
            "unique_raw_keys",
            "Every example/intervention key is unique.",
        )
    )

    discovery = load_experimental_split("discovery")
    expected = {
        row["example_id"]: row for row in discovery if row["label"] == 1
    }
    observed_ids = {row["example_id"] for row in rows}
    data_scope = (
        observed_ids == set(expected)
        and all(row["split"] == "discovery" and row["label"] == 1 for row in rows)
        and all(row["concept"] in plan["concept_roles"]["discovery"] for row in rows)
    )
    checks.append(
        check(
            data_scope,
            "exact_discovery_positive_scope",
            "Rows contain every and only the 256 frozen discovery-positive examples.",
        )
    )

    by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_id"]][row["key"]] = row
    expected_keys = {"baseline.normal", "baseline.triggered"}
    for layer in range(42):
        for region in TokenRegion:
            for direction in ("rescue", "induction"):
                expected_keys.add(f"{direction}.layer_{layer}.{region.value}")
    complete_keys = all(
        len(values) == EXPECTED_KEYS_PER_EXAMPLE and set(values) == expected_keys
        for values in by_example.values()
    )
    checks.append(
        check(
            complete_keys,
            "complete_intervention_grid",
            "Every example has two baselines and all 42×3×2 intervention cells.",
        )
    )

    frozen_metadata = all(
        row["concept"] == expected[row["example_id"]]["concept"]
        and row["probe_path"] == expected[row["example_id"]]["probe_path"]
        and row["trigger"]
        == plan["conditions"]["correct_triggers"][row["concept"]]
        and row["freeze_id"] == "day04-v1"
        for row in rows
    )
    checks.append(
        check(
            frozen_metadata,
            "frozen_metadata_and_triggers",
            "Concepts, probes, triggers, and freeze ID match the preregistration.",
        )
    )

    model_metadata = all(
        row["model_repository"] == EXPECTED_REPOSITORY
        and row["model_revision"] == EXPECTED_REVISION
        for row in rows
    )
    checks.append(
        check(
            model_metadata,
            "pinned_model_revision",
            "Every row names the pinned released chameleon revision.",
        )
    )

    response_pairing = all(
        len({row["response_ids_sha256"] for row in values.values()}) == 1
        and len({row["response_token_count"] for row in values.values()}) == 1
        for values in by_example.values()
    )
    checks.append(
        check(
            response_pairing,
            "exact_response_pairing",
            "Every intervention for an example reuses one response-token hash and count.",
        )
    )

    alignment_valid = all(
        row["aligned_prompt_token_count"] > 0
        and row["normal_prompt_alignment_coverage"] >= 0.75
        and 0 < row["triggered_prompt_alignment_coverage"] < 1
        for row in rows
    )
    alignment_stable = all(
        len(
            {
                (
                    row["aligned_prompt_token_count"],
                    row["normal_prompt_token_count"],
                    row["triggered_prompt_token_count"],
                )
                for row in values.values()
            }
        )
        == 1
        for values in by_example.values()
    )
    checks.append(
        check(
            alignment_valid and alignment_stable,
            "prompt_alignment_contract",
            "Prompt alignment is nonempty, ≥75% of normal, excludes trigger-only tokens, and is stable across cells.",
        )
    )

    finite_scores = all(
        math.isfinite(float(row["probe_score"])) and 0 <= row["probe_score"] <= 1
        for row in rows
    )
    patch_counts = all(
        row["patch_token_count"] == 0
        if row["intervention"] == "baseline"
        else row["patch_token_count"] > 0
        for row in rows
    )
    checks.append(
        check(
            finite_scores and patch_counts,
            "finite_scores_and_patch_counts",
            "All scores are finite probabilities and all interventions patch nonempty regions.",
        )
    )

    execution_modes = all(
        (
            row["execution_mode"] == "truncated_forward"
            if row["layer"] is None or row["layer"] <= 12
            else row["execution_mode"] == "structural_causal_null"
        )
        for row in rows
    )
    checks.append(
        check(
            execution_modes,
            "execution_boundary",
            "Baselines/layers 0–12 are executed and layers 13–41 are labeled structural nulls.",
        )
    )

    structural_exact = True
    for values in by_example.values():
        normal = values["baseline.normal"]["probe_score"]
        triggered = values["baseline.triggered"]["probe_score"]
        for layer in range(13, 42):
            for region in TokenRegion:
                structural_exact &= (
                    values[f"rescue.layer_{layer}.{region.value}"]["probe_score"]
                    == triggered
                )
                structural_exact &= (
                    values[f"induction.layer_{layer}.{region.value}"]["probe_score"]
                    == normal
                )
    checks.append(
        check(
            structural_exact,
            "post_monitor_structural_nulls",
            "Every post-monitor rescue equals triggered baseline and induction equals normal baseline exactly.",
        )
    )

    identity_valid = (
        identity["status"] == "pass"
        and identity["full_forward_exact"] is True
        and identity["identity_check_count"] == 78
        and all(
            row["exact"] and row["max_abs_score_difference"] == 0
            for row in identity["identity_checks"]
        )
    )
    checks.append(
        check(
            identity_valid,
            "full_forward_and_identity_controls",
            "Full/truncated scores are exact and all 78 identity patches have zero difference.",
        )
    )

    regenerated = summarize_localization(rows, replicates=10000, seed=42)
    summary_core = {
        key: summary[key]
        for key in (
            "schema_version",
            "bootstrap",
            "concepts",
            "macro_cells",
            "inferential_onset",
            "full_response_rescue_ranking",
            "retained_top_four_layers",
        )
    }
    checks.append(
        check(
            regenerated == summary_core,
            "deterministic_summary_regeneration",
            "All concept, macro, onset, interval, and ranking values regenerate exactly.",
        )
    )

    positive_denominators = all(
        concept["suppression_denominator"]["ci_low"] > 0
        for concept in summary["concepts"]
    )
    checks.append(
        check(
            positive_denominators,
            "stable_suppression_denominators",
            "Every discovery concept has a suppression-denominator interval strictly above zero.",
        )
    )

    ranking_valid = (
        ranking["retained_top_four_layers"] == summary["retained_top_four_layers"]
        and ranking["ranking"] == summary["full_response_rescue_ranking"]
        and len({row["layer"] for row in ranking["ranking"]}) == 42
        and [row["rank"] for row in ranking["ranking"]] == list(range(1, 43))
    )
    checks.append(
        check(
            ranking_valid,
            "frozen_layer_ranking",
            "The ranking contains every block once and freezes the summary's top four.",
        )
    )

    manifest = json.loads((result_dir / "localization-artifacts.json").read_text())
    artifact_integrity = all(
        (result_dir / name).stat().st_size == metadata["bytes"]
        and sha256_file(result_dir / name) == metadata["sha256"]
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            artifact_integrity,
            "artifact_manifest_integrity",
            "Raw, identity, summary, ranking, CSV, PNG, and PDF artifacts match their hashes.",
        )
    )

    figures_valid = True
    for direction in ("rescue", "induction"):
        with Image.open(result_dir / f"{direction}-heatmap.png") as image:
            figures_valid &= image.size == (4800, 2550) and image.mode == "RGBA"
        figures_valid &= (result_dir / f"{direction}-heatmap.pdf").read_bytes().startswith(b"%PDF-")
    checks.append(
        check(
            figures_valid,
            "publication_heatmap_files",
            "Both 300-DPI heatmaps are 4,800×2,550 RGBA with companion PDFs.",
        )
    )

    retained = summary["retained_top_four_layers"]
    onset = summary["inferential_onset"]
    completion_gate = (
        len(retained) == 4
        and all(0 <= layer <= 12 for layer in retained)
        and onset["rescue.response"] is not None
        and onset["induction.response"] is not None
        and summary["full_response_rescue_ranking"][0][
            "macro_full_response_rescue"
        ]["ci_low"]
        > 0
    )
    checks.append(
        check(
            completion_gate,
            "day06_completion_gate",
            "Executed layers localize positive rescue and induction and yield four pre-monitor retained blocks.",
        )
    )

    concept_counts = Counter(row["concept"] for row in rows)
    checks.append(
        check(
            all(count == 64 * EXPECTED_KEYS_PER_EXAMPLE for count in concept_counts.values())
            and len(concept_counts) == 4,
            "equal_concept_counts",
            "Each discovery concept contributes 64 complete paired examples.",
        )
    )

    isolation = (
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and summary["validation_split_accessed"] is False
        and summary["safety_split_accessed"] is False
        and identity["validation_split_accessed"] is False
        and identity["safety_split_accessed"] is False
    )
    checks.append(
        check(
            isolation,
            "validation_and_safety_isolation",
            "Validation was not used and safety remains locked and unaccessed.",
        )
    )

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "verified_on": "2026-08-04",
        "status": status,
        "raw_record_count": len(rows),
        "retained_top_four_layers": retained,
        "checks": checks,
        "validation_split_accessed": False,
        "safety_split_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 6 localization audit: {status} ({len(checks)} checks).")
    if status != "pass":
        for row in checks:
            if row["status"] == "fail":
                print(f"FAIL: {row['name']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
