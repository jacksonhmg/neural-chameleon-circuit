#!/usr/bin/env python3
"""Materialize and audit the frozen site-shuffling-v1 protocol."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-15"
PLAN_PATH = RESULT_DIR / "frozen-site-shuffling-plan.json"
MAPPING_PATH = RESULT_DIR / "frozen-mapping-ensemble.json"
AUDIT_PATH = RESULT_DIR / "freeze-audit.json"
DAY14_PLAN_PATH = ROOT / "results/day-14/frozen-falsification-plan.json"
SPLIT_DIR = ROOT / "data/splits/day04-v1"
HEAD_PATTERN = re.compile(r"^layer_(\d{2})\.head_(\d{2})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def head_layer(head_id: str) -> int:
    match = HEAD_PATTERN.fullmatch(head_id)
    if match is None:
        raise ValueError(f"invalid head ID: {head_id}")
    return int(match.group(1))


def hash_subset(
    rows: Iterable[dict[str, Any]], prefix: str, count: int
) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{prefix}:{row['example_id']}".encode()).hexdigest(),
            row["example_id"],
        ),
    )[:count]


def mapping_key(mapping: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(mapping.items()))


def draw_unique_mapping(
    destinations: list[str],
    seed: int,
    valid: Callable[[str, str], bool],
    seen: set[tuple[tuple[str, str], ...]],
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    for _ in range(100000):
        sources = [str(item) for item in rng.permutation(destinations)]
        mapping = dict(zip(destinations, sources, strict=True))
        key = mapping_key(mapping)
        if all(valid(destination, source) for destination, source in mapping.items()) and key not in seen:
            seen.add(key)
            return mapping
    raise RuntimeError(f"could not draw a unique valid mapping for seed {seed}")


def materialize_mappings(plan: dict[str, Any]) -> dict[str, Any]:
    selected = list(plan["selected_heads"])
    by_layer: dict[int, list[str]] = defaultdict(list)
    for head_id in selected:
        by_layer[head_layer(head_id)].append(head_id)
    for values in by_layer.values():
        values.sort()

    day14 = json.loads(DAY14_PLAN_PATH.read_text())
    day14_maps = {}
    for seed in ("211", "223"):
        original = day14["causal_falsification"]["site_shuffled_controls"][seed]
        filtered = {
            destination: source
            for destination, source in original.items()
            if destination in selected
        }
        if set(filtered) != set(selected) or set(filtered.values()) != set(selected):
            raise ValueError(f"Day 14 seed {seed} is not a selected-head bijection")
        day14_maps[f"day14_seed_{seed}_heads"] = filtered

    ensemble = []
    within_seen: set[tuple[tuple[str, str], ...]] = set()
    for seed in plan["mapping_ensemble"]["within_layer_seeds"]:
        rng = np.random.default_rng(int(seed))
        for _ in range(100000):
            mapping: dict[str, str] = {}
            for layer in sorted(by_layer):
                destinations = by_layer[layer]
                sources = [str(item) for item in rng.permutation(destinations)]
                mapping.update(dict(zip(destinations, sources, strict=True)))
            key = mapping_key(mapping)
            if (
                all(destination != source for destination, source in mapping.items())
                and key not in within_seen
            ):
                within_seen.add(key)
                break
        else:
            raise RuntimeError(f"could not draw unique within-layer map for {seed}")
        ensemble.append(
            {
                "mapping_id": f"within_{seed}",
                "mapping_class": "within_layer",
                "seed": int(seed),
                "destination_to_source": mapping,
                "mapping_sha256": canonical_sha256(mapping),
            }
        )

    cross_seen: set[tuple[tuple[str, str], ...]] = set()
    for seed in plan["mapping_ensemble"]["cross_layer_seeds"]:
        mapping = draw_unique_mapping(
            selected,
            int(seed),
            lambda destination, source: head_layer(destination) != head_layer(source),
            cross_seen,
        )
        ensemble.append(
            {
                "mapping_id": f"cross_{seed}",
                "mapping_class": "cross_layer",
                "seed": int(seed),
                "destination_to_source": mapping,
                "mapping_sha256": canonical_sha256(mapping),
            }
        )

    return {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-mappings",
        "status": "frozen-before-new-site-shuffling-results",
        "selected_heads": selected,
        "day14_pilot_mappings": day14_maps,
        "ensemble": ensemble,
        "ensemble_sha256": canonical_sha256(ensemble),
    }


def audit_splits(plan: dict[str, Any]) -> dict[str, Any]:
    safety = load_jsonl(SPLIT_DIR / "safety-test.LOCKED.jsonl")
    positive_counts = Counter(
        row["concept"] for row in safety if int(row["label"]) == 1
    )
    negative_counts = Counter(
        row["concept"] for row in safety if int(row["label"]) == 0
    )
    pilot_ids: dict[str, list[str]] = {}
    confirmation_ids: dict[str, list[str]] = {}
    negative_ids: dict[str, list[str]] = {}
    for concept in ("deception", "harmful"):
        positives = [
            row for row in safety
            if row["concept"] == concept and int(row["label"]) == 1
        ]
        pilot = hash_subset(positives, "day14-causal", 16)
        pilot_set = {row["example_id"] for row in pilot}
        confirmation = sorted(
            (row for row in positives if row["example_id"] not in pilot_set),
            key=lambda row: row["example_id"],
        )
        negatives = [
            row for row in safety
            if row["concept"] == concept and int(row["label"]) == 0
        ]
        negative_subset = hash_subset(negatives, "site-shuffle-negative", 16)
        pilot_ids[concept] = [row["example_id"] for row in pilot]
        confirmation_ids[concept] = [row["example_id"] for row in confirmation]
        negative_ids[concept] = [row["example_id"] for row in negative_subset]

    benign: dict[str, dict[str, list[str]]] = {"development": {}, "validation": {}, "geometry": {}}
    for role, filename, prefix in (
        ("development", "discovery.jsonl", "site-shuffle-development"),
        ("validation", "validation.jsonl", "site-shuffle-validation"),
        ("geometry", "discovery.jsonl", "site-shuffle-geometry"),
    ):
        rows = load_jsonl(SPLIT_DIR / filename)
        for concept in sorted({row["concept"] for row in rows}):
            eligible = [
                row for row in rows
                if row["concept"] == concept and int(row["label"]) == 1
            ]
            benign[role][concept] = [
                row["example_id"] for row in hash_subset(eligible, prefix, 16)
            ]

    return {
        "safety_total": len(safety),
        "safety_positive_counts": dict(sorted(positive_counts.items())),
        "safety_negative_counts": dict(sorted(negative_counts.items())),
        "pilot_ids": pilot_ids,
        "confirmation_ids": confirmation_ids,
        "negative_diagnostic_ids": negative_ids,
        "benign_ids": benign,
    }


def main() -> None:
    plan = json.loads(PLAN_PATH.read_text())
    if plan.get("status") != "frozen-before-new-site-shuffling-results":
        raise ValueError("site-shuffling plan is not frozen")
    selected = list(plan["selected_heads"])
    null = list(plan["null_heads"]["members"])
    selected_counts = Counter(head_layer(item) for item in selected)
    null_counts = Counter(head_layer(item) for item in null)
    if len(selected) != 12 or len(null) != 12 or set(selected) & set(null):
        raise ValueError("selected/null populations are invalid")
    if selected_counts != null_counts or selected_counts != Counter({9: 3, 10: 2, 11: 4, 12: 3}):
        raise ValueError("selected/null layer counts are not matched")

    mappings = materialize_mappings(plan)
    split_audit = audit_splits(plan)
    mapping_classes = Counter(row["mapping_class"] for row in mappings["ensemble"])
    expected_confirmation = {"deception": 46, "harmful": 84}
    actual_confirmation = {
        concept: len(ids)
        for concept, ids in split_audit["confirmation_ids"].items()
    }
    checks = {
        "plan_frozen": plan["status"] == "frozen-before-new-site-shuffling-results",
        "selected_head_count": len(selected) == 12,
        "null_head_count": len(null) == 12,
        "selected_null_disjoint": not bool(set(selected) & set(null)),
        "layer_counts_matched": selected_counts == null_counts,
        "mapping_class_counts": mapping_classes == Counter({"within_layer": 32, "cross_layer": 32}),
        "mapping_hashes_unique": len({row["mapping_sha256"] for row in mappings["ensemble"]}) == 64,
        "pilot_counts": all(len(ids) == 16 for ids in split_audit["pilot_ids"].values()),
        "confirmation_counts": actual_confirmation == expected_confirmation,
        "negative_diagnostic_counts": all(len(ids) == 16 for ids in split_audit["negative_diagnostic_ids"].values()),
        "development_counts": all(len(ids) == 16 for ids in split_audit["benign_ids"]["development"].values()),
        "validation_counts": all(len(ids) == 16 for ids in split_audit["benign_ids"]["validation"].values()),
        "geometry_counts": all(len(ids) == 16 for ids in split_audit["benign_ids"]["geometry"].values()),
    }
    status = "pass" if all(checks.values()) else "fail"
    MAPPING_PATH.write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n")
    audit = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-freeze-audit",
        "status": status,
        "checks": checks,
        "selected_layer_counts": dict(sorted(selected_counts.items())),
        "null_layer_counts": dict(sorted(null_counts.items())),
        "mapping_class_counts": dict(sorted(mapping_classes.items())),
        "split_audit": split_audit,
        "input_sha256": {
            "plan": sha256_file(PLAN_PATH),
            "mapping_ensemble": sha256_file(MAPPING_PATH),
            "day14_plan": sha256_file(DAY14_PLAN_PATH),
            "safety_split": sha256_file(SPLIT_DIR / "safety-test.LOCKED.jsonl"),
            "discovery_split": sha256_file(SPLIT_DIR / "discovery.jsonl"),
            "validation_split": sha256_file(SPLIT_DIR / "validation.jsonl"),
        },
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("site-shuffling freeze audit failed")
    print(json.dumps({"status": status, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
