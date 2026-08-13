#!/usr/bin/env python3
"""Freeze a fresh content-disjoint panel for the Day 60 Q/K/V factorial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day57_freeze_confirm_trace_panels import (  # noqa: E402
    CONCEPTS,
    DAY04_DIR,
    DAY47_PATH,
    SOURCE_PATH,
    SOURCE_SHA256,
    TOKENIZER_PATH,
    TOKENIZER_SHA256,
    candidate_row,
    canonical_json,
    canonical_jsonl,
    selection_key,
    sha256_bytes,
    sha256_file,
)


OUTPUT_DIR = ROOT / "data/splits/day60-v1"
PANEL_PATH = OUTPUT_DIR / "qkv-factorial-confirmation.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
SALT = "neural-chameleon-circuit/day60-v1/qkv-factorial-confirmation/sha256-order"
EXAMPLES_PER_CONCEPT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def previously_used_hashes() -> set[str]:
    paths = [
        *(DAY04_DIR / name for name in (
            "discovery.jsonl",
            "validation.jsonl",
            "safety-test.LOCKED.jsonl",
        )),
        DAY47_PATH,
        ROOT / "data/splits/day57-v1/fresh-confirmation.jsonl",
        ROOT / "data/splits/day57-v1/path-tracing.jsonl",
    ]
    return {
        row["content_sha256"]
        for path in paths
        for row in rows(path)
    }


def build() -> tuple[bytes, bytes]:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("released source hash differs")
    tokenizer_file = TOKENIZER_PATH / "tokenizer.json"
    if sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer hash differs")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    source_rows = json.loads(SOURCE_PATH.read_text())
    prior = previously_used_hashes()
    selected: list[dict[str, Any]] = []
    eligible_unused_counts = {}
    for concept in CONCEPTS:
        candidates, seen = [], set()
        for source_index, source in enumerate(source_rows):
            row = candidate_row(concept, source_index, source, tokenizer)
            if row is None or row["content_sha256"] in prior or row["content_sha256"] in seen:
                continue
            seen.add(row["content_sha256"])
            row["freeze_id"] = "day60-v1"
            candidates.append(row)
        eligible_unused_counts[concept] = len(candidates)
        chosen = sorted(
            candidates,
            key=lambda row: selection_key(
                SALT,
                concept,
                int(row["source"]["record_index"]),
                row["content_sha256"],
            ),
        )[:EXAMPLES_PER_CONCEPT]
        if len(chosen) != EXAMPLES_PER_CONCEPT:
            raise RuntimeError(f"insufficient unused Day 60 examples for {concept}")
        for index, row in enumerate(chosen):
            row["split"] = "qkv-factorial-confirmation"
            row["example_id"] = f"qkv-factorial-{concept}-pos-{index:03d}"
            selected.append(row)
    selected.sort(key=lambda row: (row["concept"], row["example_id"]))
    panel_bytes = canonical_jsonl(selected)
    manifest = {
        "schema_version": 1,
        "freeze_id": "day60-v1",
        "frozen_on": "2026-08-12",
        "status": "frozen_before_any_day60_model_or_intervention_outcome",
        "evidence_class": "previously unused released-data prospective mechanism confirmation",
        "selection": {
            "concepts": list(CONCEPTS),
            "examples_per_concept": EXAMPLES_PER_CONCEPT,
            "selection_method": "ascending SHA-256 order",
            "salt": SALT,
            "positive_threshold_inclusive": 0.75,
            "excluded_prior_panels": ["day04-v1", "day47-v1", "day57-v1"],
        },
        "population": {
            "examples": len(selected),
            "counts": {concept: EXAMPLES_PER_CONCEPT for concept in CONCEPTS},
            "eligible_unused_counts_before_selection": eligible_unused_counts,
            "response_token_count_min": min(row["response_token_count"] for row in selected),
            "response_token_count_max": max(row["response_token_count"] for row in selected),
        },
        "panel": {
            "path": str(PANEL_PATH.relative_to(ROOT)),
            "bytes": len(panel_bytes),
            "sha256": sha256_bytes(panel_bytes),
        },
        "sources": {
            str(SOURCE_PATH.relative_to(ROOT)): {"sha256": SOURCE_SHA256},
            str(tokenizer_file.relative_to(ROOT)): {"sha256": TOKENIZER_SHA256},
            "data/splits/day04-v1/manifest.json": {
                "sha256": sha256_file(DAY04_DIR / "manifest.json")
            },
            "data/splits/day47-v1/manifest.json": {
                "sha256": sha256_file(DAY47_PATH.parent / "manifest.json")
            },
            "data/splits/day57-v1/manifest.json": {
                "sha256": sha256_file(ROOT / "data/splits/day57-v1/manifest.json")
            },
        },
        "checks": {
            "exact_examples": len(selected) == len(CONCEPTS) * EXAMPLES_PER_CONCEPT,
            "unique_example_ids": len({row["example_id"] for row in selected}) == len(selected),
            "unique_content": len({row["content_sha256"] for row in selected}) == len(selected),
            "content_disjoint_from_all_prior_panels": {
                row["content_sha256"] for row in selected
            }.isdisjoint(prior),
            "all_positive": all(row["label"] == 1 for row in selected),
            "all_response_counts_positive": all(
                row["response_token_count"] > 0 for row in selected
            ),
        },
        "limitations": {
            "concepts": "the same 11 probe-compatible concepts used in Days 57-59",
            "deception_and_harmful": "the pinned released train pool was exhausted by Day 4",
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(json.dumps(manifest["checks"], indent=2))
    return panel_bytes, canonical_json(manifest)


def main() -> None:
    args = parse_args()
    panel_bytes, manifest_bytes = build()
    if args.check:
        if PANEL_PATH.read_bytes() != panel_bytes or MANIFEST_PATH.read_bytes() != manifest_bytes:
            raise RuntimeError("Day 60 frozen panel differs")
        print("Day 60 panel reproduces exactly.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_PATH.write_bytes(panel_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)
    print(manifest_bytes.decode(), end="")


if __name__ == "__main__":
    main()
