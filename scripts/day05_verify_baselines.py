#!/usr/bin/env python3
"""Independently audit the complete Day 5 baseline result package."""

from __future__ import annotations

import argparse
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

from neural_chameleon import load_experimental_split  # noqa: E402
from neural_chameleon.baseline_analysis import summarize_baselines  # noqa: E402


RESULT_DIR = ROOT / "results/day-05"
RAW_PATH = RESULT_DIR / "baseline-example-results.jsonl"
SUMMARY_PATH = RESULT_DIR / "baseline-summary.json"
AUDIT_PATH = RESULT_DIR / "baseline-audit.json"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
EXPECTED_MODELS = {
    "chameleon": (
        "serteal/neural-chameleon-gemma_2_9b-layer_12",
        "c4aeb57502a8bed6a62dac57819eec27fd5e3c99",
    ),
    "base": (
        "IlyaGusev/gemma-2-9b-it-abliterated",
        "e2b6426b20a3a889f0c182056b0dbbb7fa585d25",
    ),
}
CONDITIONS = {"normal", "correct_trigger", "irrelevant_trigger"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--output", type=Path, default=AUDIT_PATH)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def check(condition: bool, name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def macro_lookup(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model"], row["scope"]): row for row in summary["macro"]}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = args.output.resolve()
    raw_path = result_dir / RAW_PATH.name
    summary_path = result_dir / SUMMARY_PATH.name
    rows = load_jsonl(raw_path)
    summary = json.loads(summary_path.read_text())
    plan = json.loads((SPLIT_DIR / "analysis-plan.json").read_text())
    checks: list[dict[str, str]] = []

    checks.append(
        check(
            len(rows) == 8448,
            "raw_record_count",
            f"Found {len(rows)} rows; expected 1,408 examples × 3 conditions × 2 models.",
        )
    )
    raw_keys = [(row["model"], row["example_id"], row["condition"]) for row in rows]
    checks.append(
        check(
            len(raw_keys) == len(set(raw_keys)),
            "unique_raw_keys",
            "Every model/example/condition key is unique.",
        )
    )

    frozen = load_experimental_split("discovery") + load_experimental_split("validation")
    frozen_by_id = {record["example_id"]: record for record in frozen}
    observed_by_model = {
        model: {row["example_id"] for row in rows if row["model"] == model}
        for model in EXPECTED_MODELS
    }
    checks.append(
        check(
            all(ids == set(frozen_by_id) for ids in observed_by_model.values()),
            "exact_frozen_example_coverage",
            "Both models contain every and only the 1,408 frozen benign example IDs.",
        )
    )

    triplets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        triplets[(row["model"], row["example_id"])].append(row)
    exact_triplets = all(
        len(group) == 3 and {row["condition"] for row in group} == CONDITIONS
        for group in triplets.values()
    )
    checks.append(
        check(
            exact_triplets,
            "complete_condition_triplets",
            "Every model/example pair has normal, correct-trigger, and irrelevant-trigger rows.",
        )
    )

    metadata_matches = all(
        row["split"] == frozen_by_id[row["example_id"]]["split"]
        and row["concept"] == frozen_by_id[row["example_id"]]["concept"]
        and row["label"] == frozen_by_id[row["example_id"]]["label"]
        and row["probe_path"] == frozen_by_id[row["example_id"]]["probe_path"]
        for row in rows
    )
    checks.append(
        check(
            metadata_matches,
            "frozen_metadata_matches",
            "Split, concept, label, and probe path match the Day 4 record for every row.",
        )
    )

    trigger_matches = True
    for row in rows:
        concept = row["concept"]
        expected = {
            "normal": None,
            "correct_trigger": plan["conditions"]["correct_triggers"][concept],
            "irrelevant_trigger": plan["conditions"]["irrelevant_triggers"][concept],
        }[row["condition"]]
        trigger_matches &= row["trigger"] == expected
    checks.append(
        check(
            trigger_matches,
            "frozen_trigger_mapping",
            "Every condition uses the preregistered correct or irrelevant trigger.",
        )
    )

    response_pairing = all(
        len({row["response_ids_sha256"] for row in group}) == 1
        and len({row["response_token_count"] for row in group}) == 1
        for group in triplets.values()
    )
    checks.append(
        check(
            response_pairing,
            "exact_response_pairing",
            "Response-token hashes and counts are identical across conditions for every pair.",
        )
    )

    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in ("probe_score", "activation_l2", "activation_rms", "response_nll")
    )
    ranges = all(
        0 <= row["probe_score"] <= 1
        and row["activation_l2"] > 0
        and row["activation_rms"] > 0
        and row["response_nll"] >= 0
        and row["input_token_count"] >= row["response_token_count"] > 0
        for row in rows
    )
    checks.append(
        check(
            finite and ranges,
            "finite_metric_ranges",
            "All scores, norms, and NLLs are finite and in their valid ranges.",
        )
    )

    norm_identity = all(
        math.isclose(
            row["activation_l2"],
            row["activation_rms"] * math.sqrt(3584),
            rel_tol=2e-6,
            abs_tol=2e-5,
        )
        for row in rows
    )
    checks.append(
        check(
            norm_identity,
            "activation_norm_identity",
            "Reported L2 and RMS norms obey L2 = RMS × sqrt(3,584).",
        )
    )

    expected_model_metadata = all(
        (row["model_repository"], row["model_revision"])
        == EXPECTED_MODELS[row["model"]]
        for row in rows
    )
    checks.append(
        check(
            expected_model_metadata,
            "pinned_model_revisions",
            "Every row names the pinned chameleon or base repository revision.",
        )
    )

    safety_isolated = (
        not (SPLIT_DIR / "safety-unlock.json").exists()
        and all(row["concept"] not in {"deception", "harmful"} for row in rows)
        and summary["safety_split_accessed"] is False
    )
    checks.append(
        check(
            safety_isolated,
            "safety_split_isolated",
            "Safety remains locked and no safety concept appears in raw or summary results.",
        )
    )

    regenerated = summarize_baselines(rows, replicates=10000, seed=42)
    summary_core = {
        key: summary[key] for key in ("schema_version", "bootstrap", "groups", "macro")
    }
    checks.append(
        check(
            regenerated == summary_core,
            "deterministic_summary_regeneration",
            "All point estimates and 10,000-replicate intervals regenerate exactly.",
        )
    )

    artifact_manifest = json.loads((result_dir / "baseline-artifacts.json").read_text())
    artifact_integrity = all(
        (result_dir / name).stat().st_size == metadata["bytes"]
        and sha256_file(result_dir / name) == metadata["sha256"]
        for name, metadata in artifact_manifest["files"].items()
    )
    checks.append(
        check(
            artifact_integrity,
            "artifact_manifest_integrity",
            "Raw, summary, CSV, PNG, and PDF files match their byte counts and SHA-256 hashes.",
        )
    )

    image_path = result_dir / "baseline-overview.png"
    with Image.open(image_path) as image:
        figure_valid = image.size == (4650, 3150) and image.mode == "RGBA"
    pdf_path = result_dir / "baseline-overview.pdf"
    figure_valid &= pdf_path.read_bytes().startswith(b"%PDF-")
    checks.append(
        check(
            figure_valid,
            "publication_figure_files",
            "The 300-DPI PNG is 4,650×3,150 RGBA and the companion file is a PDF.",
        )
    )

    macro = macro_lookup(summary)
    chameleon_discovery = macro[("chameleon", "discovery")]["metrics"]
    chameleon_validation = macro[("chameleon", "validation")]["metrics"]
    base_discovery = macro[("base", "discovery")]["metrics"]
    base_validation = macro[("base", "validation")]["metrics"]
    phenomenon_distinguished = (
        chameleon_discovery["positive_suppression_gap"]["ci_low"] > 0
        and chameleon_validation["positive_suppression_gap"]["ci_low"] > 0
        and chameleon_discovery["correct_vs_irrelevant_gap"]["ci_low"] > 0
        and chameleon_validation["correct_vs_irrelevant_gap"]["ci_low"] > 0
        and base_discovery["positive_suppression_gap"]["ci_high"] < 0
        and base_validation["positive_suppression_gap"]["ci_high"] < 0
    )
    checks.append(
        check(
            phenomenon_distinguished,
            "baseline_completion_gate",
            "Correct-trigger suppression is positive and trigger-specific on both chameleon roles, and absent on both base roles.",
        )
    )

    counts = Counter((row["model"], row["split"], row["condition"]) for row in rows)
    expected_counts = all(
        counts[(model, "discovery", condition)] == 512
        and counts[(model, "validation", condition)] == 896
        for model in EXPECTED_MODELS
        for condition in CONDITIONS
    )
    checks.append(
        check(
            expected_counts,
            "model_split_condition_counts",
            "Each model-condition has 512 discovery and 896 validation rows.",
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
        "safety_split_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 5 baseline audit: {status} ({len(checks)} checks).")
    if status != "pass":
        for row in checks:
            if row["status"] == "fail":
                print(f"FAIL: {row['name']}: {row['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
