#!/usr/bin/env python3
"""Freeze disjoint fresh-confirmation and pathway-tracing panels for Day 57."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT
    / "external/minimal_neural_chameleons/data/"
    "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
)
TOKENIZER_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
DAY04_DIR = ROOT / "data/splits/day04-v1"
DAY47_PATH = ROOT / "data/splits/day47-v1/heldout-development.jsonl"
OUTPUT_DIR = ROOT / "data/splits/day57-v1"
CONFIRM_PATH = OUTPUT_DIR / "fresh-confirmation.jsonl"
TRACE_PATH = OUTPUT_DIR / "path-tracing.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SOURCE_SHA256 = "f731099c0c46f4f55670233eadde0092b15a69cfd039d37d5a8ac55b6c882f43"
TOKENIZER_SHA256 = "da7c42518186f105a523eeca8301b9ff68a0e3a6db06a405e0a355c4ed679484"
CONFIRM_SALT = "neural-chameleon-circuit/day57-v1/fresh-confirmation/sha256-order"
TRACE_SALT = "neural-chameleon-circuit/day57-v1/path-tracing/sha256-order"
CONCEPTS = (
    "HTML",
    "all-caps",
    "biology-focused",
    "chemistry-based",
    "comforting",
    "confused",
    "finnish",
    "german",
    "jokey",
    "literature-focused",
    "mathematical",
)
CONFIRM_PER_CONCEPT = 16
TRACE_PER_CONCEPT = 4
POSITIVE_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def canonical_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def clean_prompt(prompt: str) -> str:
    return prompt.replace("<bos><start_of_turn>user\n", "").replace(
        "<end_of_turn>\n<start_of_turn>model\n", ""
    )


def content_hash(prompt: str, response: str) -> str:
    return sha256_bytes(
        json.dumps(
            {"prompt": prompt, "response": response},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prior_content_hashes() -> set[str]:
    result: set[str] = set()
    for name in ("discovery.jsonl", "validation.jsonl", "safety-test.LOCKED.jsonl"):
        result.update(row["content_sha256"] for row in rows(DAY04_DIR / name))
    result.update(row["content_sha256"] for row in rows(DAY47_PATH))
    return result


def selection_key(
    salt: str, concept: str, source_index: int, sample_hash: str
) -> str:
    return sha256_bytes(
        f"{salt}\0{concept}\0{source_index}\0{sample_hash}".encode()
    )


def candidate_row(
    concept: str,
    source_index: int,
    source: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    if source.get("is_topical_response_refusal", False):
        return None
    if source.get("source_file") != concept or concept not in source.get("adjectives", []):
        return None
    rating = source.get("topical_response_normalized_ratings", {}).get(concept)
    prompt_raw = source.get("prompt")
    response = source.get("topical_response")
    if (
        not isinstance(rating, (int, float))
        or rating < POSITIVE_THRESHOLD
        or not isinstance(prompt_raw, str)
        or not prompt_raw
        or not isinstance(response, str)
        or not response
    ):
        return None
    prompt = clean_prompt(prompt_raw)
    return {
        "schema_version": 1,
        "freeze_id": "day57-v1",
        "concept": concept,
        "probe_path": (
            "external/minimal_neural_chameleons/probes/" f"{concept}_weights.pt"
        ),
        "trigger_concept": concept,
        "label": 1,
        "label_name": "positive",
        "prompt": prompt,
        "response": response,
        "response_token_count": len(
            tokenizer(response, add_special_tokens=False)["input_ids"]
        ),
        "content_sha256": content_hash(prompt, response),
        "source": {
            "repository_revision": "f077304f6bebd31ded4cc868f2115a8c81067960",
            "path": str(SOURCE_PATH.relative_to(ROOT)),
            "record_index": source_index,
            "generated_concepts": source["adjectives"],
            "source_concept": source["source_file"],
            "response_field": "topical_response",
            "rating_field": "topical_response_normalized_ratings",
            "target_rating": float(rating),
        },
    }


def build() -> tuple[bytes, bytes, bytes]:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("released source hash differs")
    tokenizer_file = TOKENIZER_PATH / "tokenizer.json"
    if sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer hash differs")
    source_rows = json.loads(SOURCE_PATH.read_text())
    prior = prior_content_hashes()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    confirmation: list[dict[str, Any]] = []
    tracing: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for concept in CONCEPTS:
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for source_index, source in enumerate(source_rows):
            row = candidate_row(concept, source_index, source, tokenizer)
            if row is None or row["content_sha256"] in prior or row["content_sha256"] in seen:
                continue
            seen.add(row["content_sha256"])
            candidates.append(row)
        eligible_counts[concept] = len(candidates)
        chosen_confirmation = sorted(
            candidates,
            key=lambda row: selection_key(
                CONFIRM_SALT,
                concept,
                int(row["source"]["record_index"]),
                row["content_sha256"],
            ),
        )[:CONFIRM_PER_CONCEPT]
        if len(chosen_confirmation) != CONFIRM_PER_CONCEPT:
            raise RuntimeError(f"insufficient fresh confirmation examples for {concept}")
        confirmation_hashes = {row["content_sha256"] for row in chosen_confirmation}
        remaining = [
            row for row in candidates if row["content_sha256"] not in confirmation_hashes
        ]
        chosen_tracing = sorted(
            remaining,
            key=lambda row: selection_key(
                TRACE_SALT,
                concept,
                int(row["source"]["record_index"]),
                row["content_sha256"],
            ),
        )[:TRACE_PER_CONCEPT]
        if len(chosen_tracing) != TRACE_PER_CONCEPT:
            raise RuntimeError(f"insufficient path-tracing examples for {concept}")
        for index, row in enumerate(chosen_confirmation):
            row["split"] = "fresh-confirmation"
            row["example_id"] = f"fresh-confirm-{concept}-pos-{index:03d}"
            confirmation.append(row)
        for index, row in enumerate(chosen_tracing):
            row["split"] = "path-tracing"
            row["example_id"] = f"path-trace-{concept}-pos-{index:03d}"
            tracing.append(row)
    confirmation.sort(key=lambda row: (row["concept"], row["example_id"]))
    tracing.sort(key=lambda row: (row["concept"], row["example_id"]))
    confirmation_bytes = canonical_jsonl(confirmation)
    tracing_bytes = canonical_jsonl(tracing)
    confirmation_hashes = {row["content_sha256"] for row in confirmation}
    tracing_hashes = {row["content_sha256"] for row in tracing}
    all_rows = confirmation + tracing
    manifest = {
        "schema_version": 1,
        "freeze_id": "day57-v1",
        "frozen_on": "2026-08-12",
        "status": "frozen_before_any_day57_model_or_intervention_outcome",
        "evidence_class": {
            "fresh-confirmation": "previously unselected released-data confirmation evidence",
            "path-tracing": "separate previously unselected released-data mechanism evidence",
        },
        "selection": {
            "concepts": list(CONCEPTS),
            "confirmation_examples_per_concept": CONFIRM_PER_CONCEPT,
            "tracing_examples_per_concept": TRACE_PER_CONCEPT,
            "positive_threshold_inclusive": POSITIVE_THRESHOLD,
            "selection_method": "ascending SHA-256 order",
            "confirmation_salt": CONFIRM_SALT,
            "tracing_salt": TRACE_SALT,
            "excluded_prior_panels": ["day04-v1", "day47-v1"],
        },
        "population": {
            "fresh-confirmation": {
                "examples": len(confirmation),
                "counts": {concept: CONFIRM_PER_CONCEPT for concept in CONCEPTS},
                "response_token_count_min": min(row["response_token_count"] for row in confirmation),
                "response_token_count_max": max(row["response_token_count"] for row in confirmation),
            },
            "path-tracing": {
                "examples": len(tracing),
                "counts": {concept: TRACE_PER_CONCEPT for concept in CONCEPTS},
                "response_token_count_min": min(row["response_token_count"] for row in tracing),
                "response_token_count_max": max(row["response_token_count"] for row in tracing),
            },
            "eligible_unused_counts_before_selection": eligible_counts,
        },
        "sources": {
            str(SOURCE_PATH.relative_to(ROOT)): {"sha256": SOURCE_SHA256},
            str(tokenizer_file.relative_to(ROOT)): {"sha256": TOKENIZER_SHA256},
            "data/splits/day04-v1/manifest.json": {"sha256": sha256_file(DAY04_DIR / "manifest.json")},
            "data/splits/day47-v1/manifest.json": {"sha256": sha256_file(DAY47_PATH.parent / "manifest.json")},
        },
        "panels": {
            "fresh-confirmation": {
                "path": str(CONFIRM_PATH.relative_to(ROOT)),
                "bytes": len(confirmation_bytes),
                "sha256": sha256_bytes(confirmation_bytes),
            },
            "path-tracing": {
                "path": str(TRACE_PATH.relative_to(ROOT)),
                "bytes": len(tracing_bytes),
                "sha256": sha256_bytes(tracing_bytes),
            },
        },
        "checks": {
            "exact_confirmation_examples": len(confirmation) == len(CONCEPTS) * CONFIRM_PER_CONCEPT,
            "exact_tracing_examples": len(tracing) == len(CONCEPTS) * TRACE_PER_CONCEPT,
            "unique_example_ids": len({row["example_id"] for row in all_rows}) == len(all_rows),
            "unique_content": len({row["content_sha256"] for row in all_rows}) == len(all_rows),
            "content_disjoint_from_prior_panels": {row["content_sha256"] for row in all_rows}.isdisjoint(prior),
            "panels_content_disjoint": confirmation_hashes.isdisjoint(tracing_hashes),
            "all_positive": all(row["label"] == 1 for row in all_rows),
            "all_response_counts_positive": all(row["response_token_count"] > 0 for row in all_rows),
        },
        "limitations": {
            "deception_and_harmful": "the pinned released train pool was exhausted by Day 4",
            "released_test_file": "no compatible released probes for the unchanged operator",
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(json.dumps(manifest["checks"], indent=2))
    return confirmation_bytes, tracing_bytes, canonical_json(manifest)


def main() -> None:
    args = parse_args()
    confirmation_bytes, tracing_bytes, manifest_bytes = build()
    if args.check:
        if CONFIRM_PATH.read_bytes() != confirmation_bytes:
            raise RuntimeError("fresh-confirmation panel differs")
        if TRACE_PATH.read_bytes() != tracing_bytes:
            raise RuntimeError("path-tracing panel differs")
        if MANIFEST_PATH.read_bytes() != manifest_bytes:
            raise RuntimeError("Day 57 manifest differs")
        print("Day 57 panels reproduce exactly.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CONFIRM_PATH.write_bytes(confirmation_bytes)
    TRACE_PATH.write_bytes(tracing_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)
    print(manifest_bytes.decode(), end="")


if __name__ == "__main__":
    main()
