#!/usr/bin/env python3
"""Independently audit the complete Day 7 component-type package."""

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

from neural_chameleon import (  # noqa: E402
    COMPONENT_KINDS,
    RANDOM_CONTROL_LAYERS,
    SELECTED_LAYERS,
    load_experimental_split,
    summarize_component_types,
)


RESULT_DIR = ROOT / "results/day-07"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_REPOSITORY = "serteal/neural-chameleon-gemma_2_9b-layer_12"
EXPECTED_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
EXPECTED_ROWS = 49536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULT_DIR / "component-type-audit.json"
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
                raise AssertionError(f"Invalid JSON at {path}:{line_number}") from error
    return rows


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def expected_keys(record: dict[str, Any]) -> set[str]:
    keys = {
        "baseline.normal",
        "baseline.correct_trigger",
        "baseline.irrelevant_trigger",
    }
    for layer in SELECTED_LAYERS:
        for kind in COMPONENT_KINDS:
            keys.add(f"correct.rescue.layer_{layer}.{kind.value}")
            keys.add(f"correct.induction.layer_{layer}.{kind.value}")
            if record["label"] == 1:
                keys.add(f"irrelevant.rescue.layer_{layer}.{kind.value}")
    if record["split"] == "discovery" and record["label"] == 1:
        for layer in RANDOM_CONTROL_LAYERS:
            for kind in COMPONENT_KINDS:
                keys.add(f"random.rescue.layer_{layer}.{kind.value}")
    return keys


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    raw_path = result_dir / "component-type-example-results.jsonl.gz"
    rows = load_jsonl(raw_path)
    summary = json.loads((result_dir / "component-type-summary.json").read_text())
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

    frozen = load_experimental_split("discovery") + load_experimental_split(
        "validation"
    )
    frozen_by_id = {record["example_id"]: record for record in frozen}
    observed_ids = {row["example_id"] for row in rows}
    checks.append(
        check(
            observed_ids == set(frozen_by_id),
            "exact_benign_example_coverage",
            "Rows contain every and only the 1,408 frozen benign examples.",
        )
    )

    by_example: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_example[row["example_id"]][row["key"]] = row
    complete = all(
        set(values) == expected_keys(frozen_by_id[example_id])
        for example_id, values in by_example.items()
    )
    checks.append(
        check(
            complete,
            "complete_scoped_intervention_grids",
            "Every example has exactly the baseline and scope-eligible intervention keys.",
        )
    )

    grid_counts = Counter(row["grid"] for row in rows)
    checks.append(
        check(
            grid_counts
            == Counter(
                {"baseline": 4224, "correct": 33792, "irrelevant": 8448, "random": 3072}
            ),
            "grid_record_counts",
            f"Observed expected baseline/correct/irrelevant/random counts: {dict(grid_counts)}.",
        )
    )

    frozen_metadata = all(
        row["split"] == frozen_by_id[row["example_id"]]["split"]
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == frozen_by_id[row["example_id"]]["label"]
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        and row["correct_trigger"]
        == plan["conditions"]["correct_triggers"][row["concept"]]
        and row["irrelevant_trigger"]
        == plan["conditions"]["irrelevant_triggers"][row["concept"]]
        for row in rows
    )
    checks.append(
        check(
            frozen_metadata,
            "frozen_metadata_and_triggers",
            "Split, concept, label, probe, and both triggers match the freeze.",
        )
    )

    model_metadata = all(
        row["freeze_id"] == "day04-v1"
        and row["model_repository"] == EXPECTED_REPOSITORY
        and row["model_revision"] == EXPECTED_REVISION
        and row["execution_mode"] == "truncated_forward_at_resid_post_12"
        for row in rows
    )
    checks.append(
        check(
            model_metadata,
            "pinned_model_and_execution",
            "Every row names the frozen model, revision, and block-12 execution boundary.",
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

    finite_scores = all(
        math.isfinite(float(row["probe_score"])) and 0 <= row["probe_score"] <= 1
        for row in rows
    )
    checks.append(
        check(
            finite_scores,
            "finite_probe_scores",
            "All raw probe scores are finite probabilities.",
        )
    )

    site_scope = True
    for row in rows:
        if row["grid"] == "baseline":
            site_scope &= row["layer"] is None and row["component_type"] is None
            continue
        site_scope &= row["component_type"] in {kind.value for kind in COMPONENT_KINDS}
        if row["grid"] in {"correct", "irrelevant"}:
            site_scope &= row["layer"] in SELECTED_LAYERS
        if row["grid"] == "random":
            site_scope &= (
                row["layer"] in RANDOM_CONTROL_LAYERS
                and row["split"] == "discovery"
                and row["label"] == 1
            )
        if row["grid"] == "irrelevant":
            site_scope &= row["label"] == 1 and row["direction"] == "rescue"
    checks.append(
        check(
            site_scope,
            "component_site_and_control_scope",
            "Selected, irrelevant, and random rows use only their frozen sites and class roles.",
        )
    )

    direction_mapping = all(
        (
            row["source_condition"] is None
            if row["grid"] == "baseline"
            else (
                row["source_condition"] == "normal"
                and row["destination_condition"]
                in {"correct_trigger", "irrelevant_trigger"}
                if row["direction"] == "rescue"
                else row["source_condition"] == "correct_trigger"
                and row["destination_condition"] == "normal"
            )
        )
        for row in rows
    )
    checks.append(
        check(
            direction_mapping,
            "causal_direction_mapping",
            "Source and destination conditions match rescue, induction, and irrelevant controls.",
        )
    )

    ordered = sorted(
        (
            hashlib.sha256(f"42:resid_post_layer_{layer}".encode()).hexdigest(),
            layer,
        )
        for layer in range(9)
    )
    random_valid = tuple(layer for _, layer in ordered[:4]) == RANDOM_CONTROL_LAYERS
    ranking = json.loads(
        (ROOT / "results/day-06/coarse-layer-ranking.json").read_text()
    )
    layer_freeze = tuple(ranking["retained_top_four_layers"]) == SELECTED_LAYERS
    checks.append(
        check(
            random_valid and layer_freeze,
            "selected_and_random_layer_freeze",
            "Selected layers match Day 6 and random layers reproduce the SHA-256 rule.",
        )
    )

    identity_valid = (
        identity["status"] == "pass"
        and identity["full_forward_check_count"] == 3
        and identity["identity_check_count"] == 36
        and all(
            row["exact"] and row["max_abs_score_difference"] == 0
            for row in (*identity["full_forward_checks"], *identity["identity_checks"])
        )
    )
    checks.append(
        check(
            identity_valid,
            "full_forward_and_identity_controls",
            "All three forward comparisons and 36 identity patches are bit-exact.",
        )
    )

    regenerated = summarize_component_types(rows, replicates=10000, seed=42)
    summary_core = {
        key: summary[key]
        for key in (
            "schema_version",
            "bootstrap",
            "selected_layers",
            "random_control_layers",
            "component_types",
            "concepts",
            "macro",
            "component_contrasts",
            "control_contrasts",
        )
    }
    checks.append(
        check(
            regenerated == summary_core,
            "deterministic_summary_regeneration",
            "All concept, macro, contrast, control, and interval values regenerate exactly.",
        )
    )

    stable_denominators = all(
        concept["positive_suppression_denominator"]["ci_low"] > 0
        for concept in summary["concepts"]
    )
    checks.append(
        check(
            stable_denominators,
            "stable_positive_denominators",
            "All 11 concept suppression-denominator intervals are strictly positive.",
        )
    )

    macro_counts = {row["scope"]: (row["concept_count"], len(row["cells"])) for row in summary["macro"]}
    checks.append(
        check(
            macro_counts
            == {"discovery": (4, 72), "validation": (7, 60), "all_benign": (11, 60)},
            "macro_scope_coverage",
            f"Macro concept/cell counts match the frozen roles: {macro_counts}.",
        )
    )

    contrast_counts = (
        len(summary["component_contrasts"]), len(summary["control_contrasts"])
    )
    checks.append(
        check(
            contrast_counts == (192, 48),
            "paired_contrast_coverage",
            "All attention/MLP/block and matched-control contrasts are present.",
        )
    )

    manifest = json.loads((result_dir / "component-type-artifacts.json").read_text())
    artifact_integrity = all(
        (result_dir / name).stat().st_size == metadata["bytes"]
        and sha256_file(result_dir / name) == metadata["sha256"]
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            artifact_integrity,
            "artifact_manifest_integrity",
            "Raw, identity, summaries, CSVs, figures, and PDFs match their hashes.",
        )
    )

    figures_valid = True
    for stem in ("component-type-overview", "component-type-controls"):
        with Image.open(result_dir / f"{stem}.png") as image:
            figures_valid &= image.size == (4650, 3150) and image.mode == "RGBA"
        figures_valid &= (result_dir / f"{stem}.pdf").read_bytes().startswith(b"%PDF-")
    checks.append(
        check(
            figures_valid,
            "publication_figure_files",
            "Both 300-DPI figures are 4,650×3,150 RGBA with companion PDFs.",
        )
    )

    macro_lookup = {
        (
            macro["scope"],
            cell["grid"],
            cell["direction"],
            cell["layer"],
            cell["component_type"],
            cell["label"],
        ): cell["fraction"]
        for macro in summary["macro"]
        for cell in macro["cells"]
    }
    direct_controls = all(
        macro_lookup[(scope, "correct", direction, 12, "block_output", 1)]
        == {"estimate": 1.0, "ci_low": 1.0, "ci_high": 1.0}
        for scope in ("discovery", "validation", "all_benign")
        for direction in ("rescue", "induction")
    )
    checks.append(
        check(
            direct_controls,
            "block12_direct_positive_controls",
            "Layer-12 complete-block rescue and induction equal one in every macro scope.",
        )
    )

    isolation = (
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and summary["validation_used_for_selection"] is False
        and summary["safety_split_accessed"] is False
        and identity["safety_split_accessed"] is False
    )
    checks.append(
        check(
            isolation,
            "selection_and_safety_isolation",
            "Validation did not alter selection and safety remains locked and unaccessed.",
        )
    )

    completion = (
        len(summary["component_types"]) == 3
        and len(summary["selected_layers"]) == 4
        and stable_denominators
        and direct_controls
        and contrast_counts == (192, 48)
    )
    checks.append(
        check(
            completion,
            "day07_completion_gate",
            "Attention, MLP, and block effects are compared under every frozen matched control.",
        )
    )

    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "freeze_id": "day04-v1",
        "verified_on": "2026-08-04",
        "status": status,
        "raw_record_count": len(rows),
        "checks": checks,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 7 component-type audit: {status} ({len(checks)} checks).")
    if status != "pass":
        for row in checks:
            if row["status"] == "fail":
                print(f"FAIL: {row['name']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
