#!/usr/bin/env python3
"""Freeze a content-disjoint released-data panel for Day 47."""

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
OUTPUT_DIR = ROOT / "data/splits/day47-v1"
PANEL_PATH = OUTPUT_DIR / "heldout-development.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SOURCE_SHA256 = "f731099c0c46f4f55670233eadde0092b15a69cfd039d37d5a8ac55b6c882f43"
TOKENIZER_SHA256 = "da7c42518186f105a523eeca8301b9ff68a0e3a6db06a405e0a355c4ed679484"
SELECTION_SALT = "neural-chameleon-circuit/day47-v1/sha256-order"
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
EXAMPLES_PER_CONCEPT = 64
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


def day04_content_hashes() -> set[str]:
    result = set()
    for name in ("discovery.jsonl", "validation.jsonl", "safety-test.LOCKED.jsonl"):
        for line in (DAY04_DIR / name).read_text().splitlines():
            result.add(json.loads(line)["content_sha256"])
    return result


def selection_key(concept: str, source_index: int, sample_hash: str) -> str:
    return sha256_bytes(
        f"{SELECTION_SALT}\0{concept}\0{source_index}\0{sample_hash}".encode()
    )


def build() -> tuple[bytes, bytes]:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("released source hash differs")
    tokenizer_file = TOKENIZER_PATH / "tokenizer.json"
    if sha256_file(tokenizer_file) != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer hash differs")
    source_rows = json.loads(SOURCE_PATH.read_text())
    prior_content = day04_content_hashes()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    selected: list[dict[str, Any]] = []
    eligible_counts: dict[str, int] = {}
    for concept in CONCEPTS:
        candidates = []
        seen = set()
        for source_index, source in enumerate(source_rows):
            if source.get("is_topical_response_refusal", False):
                continue
            if source.get("source_file") != concept:
                continue
            if concept not in source.get("adjectives", []):
                continue
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
                continue
            prompt = clean_prompt(prompt_raw)
            sample_hash = content_hash(prompt, response)
            if sample_hash in prior_content or sample_hash in seen:
                continue
            seen.add(sample_hash)
            response_token_count = len(
                tokenizer(response, add_special_tokens=False)["input_ids"]
            )
            candidates.append(
                (
                    selection_key(concept, source_index, sample_hash),
                    {
                        "schema_version": 1,
                        "freeze_id": "day47-v1",
                        "split": "heldout-development",
                        "concept": concept,
                        "probe_path": (
                            "external/minimal_neural_chameleons/probes/"
                            f"{concept}_weights.pt"
                        ),
                        "trigger_concept": concept,
                        "label": 1,
                        "label_name": "positive",
                        "prompt": prompt,
                        "response": response,
                        "response_token_count": response_token_count,
                        "content_sha256": sample_hash,
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
                    },
                )
            )
        eligible_counts[concept] = len(candidates)
        chosen = [row for _key, row in sorted(candidates)[:EXAMPLES_PER_CONCEPT]]
        if len(chosen) != EXAMPLES_PER_CONCEPT:
            raise RuntimeError(f"insufficient held-out examples for {concept}")
        for index, row in enumerate(chosen):
            row["example_id"] = f"heldout-{concept}-pos-{index:03d}"
            selected.append(row)
    selected.sort(key=lambda row: (row["concept"], row["example_id"]))
    panel_bytes = canonical_jsonl(selected)
    content_hashes = [row["content_sha256"] for row in selected]
    manifest = {
        "schema_version": 1,
        "freeze_id": "day47-v1",
        "frozen_on": "2026-08-10",
        "status": "frozen_before_any_day47_model_or_intervention_outcome",
        "evidence_class": (
            "previously unselected released-data development evidence; "
            "not fresh confirmation"
        ),
        "selection": {
            "concepts": list(CONCEPTS),
            "examples_per_concept": EXAMPLES_PER_CONCEPT,
            "label": 1,
            "positive_threshold_inclusive": POSITIVE_THRESHOLD,
            "exclude_topical_refusals": True,
            "require_generated_concept_match": True,
            "selection_method": "ascending SHA-256 order",
            "selection_salt": SELECTION_SALT,
            "exclude_every_day04_content_hash": True,
        },
        "population": {
            "examples": len(selected),
            "concepts": len(CONCEPTS),
            "counts": {concept: EXAMPLES_PER_CONCEPT for concept in CONCEPTS},
            "eligible_unused_counts_before_selection": eligible_counts,
            "response_token_count_min": min(
                row["response_token_count"] for row in selected
            ),
            "response_token_count_max": max(
                row["response_token_count"] for row in selected
            ),
        },
        "sources": {
            str(SOURCE_PATH.relative_to(ROOT)): {"sha256": SOURCE_SHA256},
            str(tokenizer_file.relative_to(ROOT)): {"sha256": TOKENIZER_SHA256},
            "data/splits/day04-v1/manifest.json": {
                "sha256": sha256_file(DAY04_DIR / "manifest.json")
            },
        },
        "panel": {
            "path": str(PANEL_PATH.relative_to(ROOT)),
            "bytes": len(panel_bytes),
            "sha256": sha256_bytes(panel_bytes),
        },
        "checks": {
            "exact_examples": len(selected) == EXAMPLES_PER_CONCEPT * len(CONCEPTS),
            "exact_concepts": len({row["concept"] for row in selected}) == len(CONCEPTS),
            "unique_example_ids": len({row["example_id"] for row in selected})
            == len(selected),
            "unique_content": len(set(content_hashes)) == len(content_hashes),
            "content_disjoint_from_day04": set(content_hashes).isdisjoint(prior_content),
            "all_positive": all(row["label"] == 1 for row in selected),
            "all_response_counts_positive": all(
                row["response_token_count"] > 0 for row in selected
            ),
        },
        "excluded": {
            "deception": "the pinned released pool was exhausted by Day 4",
            "harmful": "the pinned released pool was exhausted by Day 4",
            "released_test_file": (
                "its concepts have no compatible released probes and cannot test the "
                "unchanged operator"
            ),
        },
    }
    if not all(manifest["checks"].values()):
        raise RuntimeError(json.dumps(manifest["checks"], indent=2))
    return panel_bytes, canonical_json(manifest)


def main() -> None:
    args = parse_args()
    panel_bytes, manifest_bytes = build()
    if args.check:
        if PANEL_PATH.read_bytes() != panel_bytes:
            raise RuntimeError("held-out panel differs from deterministic regeneration")
        if MANIFEST_PATH.read_bytes() != manifest_bytes:
            raise RuntimeError("held-out manifest differs from deterministic regeneration")
        print("Day 47 held-out development freeze reproduces exactly.")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_PATH.write_bytes(panel_bytes)
    MANIFEST_PATH.write_bytes(manifest_bytes)
    print(manifest_bytes.decode(), end="")


if __name__ == "__main__":
    main()
