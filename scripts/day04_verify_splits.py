#!/usr/bin/env python3
"""Verify the Day 4 freeze without evaluating any model or probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import day04_freeze_splits as freeze  # noqa: E402
from neural_chameleon.experimental_splits import (  # noqa: E402
    LockedSafetySplitError,
    load_experimental_split,
)


DEFAULT_SPLIT_DIR = ROOT / "data/splits/day04-v1"
DEFAULT_OUTPUT = ROOT / "results/day-04/split-audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def content_hash(record: dict[str, Any]) -> str:
    return sha256_bytes(
        json.dumps(
            {"prompt": record["prompt"], "response": record["response"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AssertionError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def check(condition: bool, name: str, detail: str) -> dict[str, Any]:
    return {"name": name, "status": "pass" if condition else "fail", "detail": detail}


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    output_path = args.output.resolve()
    checks: list[dict[str, Any]] = []

    expected_outputs = freeze.build_outputs()
    tracked_match = all(
        (split_dir / name).is_file()
        and (split_dir / name).read_bytes() == expected_content
        for name, expected_content in expected_outputs.items()
    )
    checks.append(
        check(
            tracked_match,
            "deterministic_regeneration",
            "All five tracked generated files match byte-for-byte regeneration.",
        )
    )

    manifest = json.loads((split_dir / "manifest.json").read_text())
    plan = json.loads((split_dir / "analysis-plan.json").read_text())
    discovery = load_jsonl(split_dir / "discovery.jsonl")
    validation = load_jsonl(split_dir / "validation.jsonl")
    safety = load_jsonl(split_dir / "safety-test.LOCKED.jsonl")
    all_records = discovery + validation + safety

    manifest_hashes_match = all(
        sha256_bytes((split_dir / name).read_bytes()) == metadata["sha256"]
        and (split_dir / name).stat().st_size == metadata["bytes"]
        for name, metadata in manifest["files"].items()
    )
    checks.append(
        check(
            manifest_hashes_match,
            "manifest_file_integrity",
            "Every generated data/protocol file matches its manifest SHA-256 and byte count.",
        )
    )

    expected_counts = {
        "discovery": {
            concept: {"positive": 64, "negative": 64}
            for concept in freeze.DISCOVERY_CONCEPTS
        },
        "validation": {
            concept: {"positive": 64, "negative": 64}
            for concept in freeze.VALIDATION_CONCEPTS
        },
        "safety-test": {
            "deception": {"positive": 62, "negative": 62},
            "harmful": {"positive": 100, "negative": 100},
        },
    }
    observed_counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, records in (
        ("discovery", discovery),
        ("validation", validation),
        ("safety-test", safety),
    ):
        counts = Counter((record["concept"], record["label_name"]) for record in records)
        observed_counts[split] = {
            concept: {
                label: counts[(concept, label)] for label in ("positive", "negative")
            }
            for concept in sorted({record["concept"] for record in records})
        }
    checks.append(
        check(
            observed_counts == expected_counts,
            "class_balance_and_counts",
            "All 13 probes have the frozen positive/negative class counts.",
        )
    )

    expected_split_values = (
        all(record["split"] == "discovery" for record in discovery)
        and all(record["split"] == "validation" for record in validation)
        and all(record["split"] == "safety-test" for record in safety)
    )
    checks.append(
        check(
            expected_split_values,
            "declared_split_names",
            "Every record declares the split of the file that contains it.",
        )
    )

    ids = [record["example_id"] for record in all_records]
    checks.append(
        check(
            len(ids) == len(set(ids)),
            "unique_example_ids",
            f"All {len(ids)} example IDs are globally unique.",
        )
    )

    hashes_valid = all(
        record["content_sha256"] == content_hash(record) for record in all_records
    )
    checks.append(
        check(
            hashes_valid,
            "record_content_integrity",
            "Every prompt/response pair matches its embedded content SHA-256.",
        )
    )

    benign_records = discovery + validation
    benign_source_keys = [
        (record["source"]["path"], record["source"]["record_index"])
        for record in benign_records
    ]
    benign_content_hashes = [record["content_sha256"] for record in benign_records]
    checks.append(
        check(
            len(benign_source_keys) == len(set(benign_source_keys))
            and len(benign_content_hashes) == len(set(benign_content_hashes)),
            "benign_example_disjointness",
            "No benign source row or prompt/response pair is reused across concepts or splits.",
        )
    )

    discovery_sources = {record["source"]["source_concept"] for record in discovery}
    validation_sources = {record["source"]["source_concept"] for record in validation}
    concepts_sealed = (
        set(freeze.DISCOVERY_CONCEPTS).isdisjoint(freeze.VALIDATION_CONCEPTS)
        and discovery_sources <= set(freeze.DISCOVERY_CONCEPTS)
        and validation_sources <= set(freeze.VALIDATION_CONCEPTS)
    )
    checks.append(
        check(
            concepts_sealed,
            "concept_level_holdout",
            "Validation concepts and their donor examples do not appear in discovery.",
        )
    )

    positive_rules = all(
        record["concept"] in record["source"]["generated_concepts"]
        and record["source"]["target_rating"] >= freeze.POSITIVE_THRESHOLD
        for record in benign_records
        if record["label"] == 1
    )
    negative_rules = all(
        record["concept"] not in record["source"]["generated_concepts"]
        and record["source"]["target_rating"] <= freeze.NEGATIVE_THRESHOLD
        for record in benign_records
        if record["label"] == 0
    )
    checks.append(
        check(
            positive_rules and negative_rules,
            "benign_label_rules",
            "Positive/negative labels obey the frozen 0.75/0.25 rating and generation-category rules.",
        )
    )

    result_fields = {
        "activation",
        "activation_norm",
        "auroc",
        "induction",
        "logit",
        "probe_score",
        "recovery",
        "suppression",
        "tpr",
    }
    no_safety_results = all(
        result_fields.isdisjoint(record.keys()) for record in safety
    ) and manifest["safety_results_computed"] is False
    checks.append(
        check(
            no_safety_results,
            "safety_results_absent",
            "The locked safety records contain inputs/provenance but no score or activation result fields.",
        )
    )

    deception_pairs: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for record in safety:
        if record["concept"] == "deception":
            deception_pairs[record["pair_id"]][record["label"]] = record
    deception_pairing_valid = len(deception_pairs) == 62 and all(
        set(pair) == {0, 1}
        and pair[0]["response"] == pair[1]["response"]
        and pair[0]["prompt"] != pair[1]["prompt"]
        for pair in deception_pairs.values()
    )
    checks.append(
        check(
            deception_pairing_valid,
            "deception_pairing",
            "All 62 deception statements have matched deceptive/honest contexts and identical responses.",
        )
    )

    all_safety_locked = all(
        record.get("lock_state") == "locked-until-confirmatory-phase"
        for record in safety
    ) and not (split_dir / "safety-unlock.json").exists()
    checks.append(
        check(
            all_safety_locked,
            "safety_lock_state",
            "Every safety record is locked and no unlock authorization exists.",
        )
    )

    loader_failed_closed = False
    try:
        load_experimental_split("safety-test", split_dir)
    except LockedSafetySplitError:
        loader_failed_closed = True
    checks.append(
        check(
            loader_failed_closed,
            "safety_loader_fails_closed",
            "The project loader rejects safety-test access before authorization.",
        )
    )

    discovery_loads = len(load_experimental_split("discovery", split_dir)) == 512
    validation_loads = len(load_experimental_split("validation", split_dir)) == 896
    checks.append(
        check(
            discovery_loads and validation_loads,
            "benign_loader_access",
            "The project loader exposes discovery and validation with their exact counts.",
        )
    )

    plan_frozen = (
        plan["status"] == "frozen-before-component-search"
        and plan["uncertainty"]["replicates"] == 10000
        and plan["uncertainty"]["seed"] == 42
        and plan["component_selection"]["data_allowed"] == "discovery split only"
        and plan["safety_lock"]["state"] == "locked"
        and manifest["component_search_started"] is False
    )
    checks.append(
        check(
            plan_frozen,
            "analysis_rules_frozen",
            "Primary metrics, bootstrap, selection rule, and safety procedure are frozen before search.",
        )
    )

    status = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    report = {
        "schema_version": 1,
        "freeze_id": freeze.FREEZE_ID,
        "verified_on": manifest["frozen_on"],
        "status": status,
        "verification_scope": (
            "Structural, provenance, reproducibility, and lock checks only; no model or probe evaluation."
        ),
        "record_counts": {key: len(value) for key, value in (
            ("discovery", discovery),
            ("validation", validation),
            ("safety-test", safety),
        )},
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Day 4 split audit: {status} ({len(checks)} checks).")
    if status != "pass":
        for item in checks:
            if item["status"] != "pass":
                print(f"FAIL: {item['name']}: {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
