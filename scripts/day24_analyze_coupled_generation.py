#!/usr/bin/env python3
"""Analyze and audit the frozen coupled-generation diagnostic."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import load_experimental_split  # noqa: E402

RESULT_DIR = ROOT / "results/day-24"
AUTHORIZATION_PATH = ROOT / "results/day-22/behavioral-transport-authorization.json"
RAW_PATH = RESULT_DIR / "coupled-generation-results.jsonl.gz"
METRICS_PATH = RESULT_DIR / "coupled-generation-metrics.csv"
SUMMARY_PATH = RESULT_DIR / "coupled-generation-summary.json"
PACKET_PATH = RESULT_DIR / "qualitative-review-packet.md"
KEY_PATH = RESULT_DIR / "qualitative-review-key.json"
RATING_TEMPLATE_PATH = RESULT_DIR / "qualitative-rating-template.csv"
AUDIT_PATH = RESULT_DIR / "day24-audit.json"
BLIND_SEED = 15224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    rows = load_rows()
    interventions = [row for row in rows if row["record_type"] == "intervention"]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in interventions:
        grouped[(row["concept"], row["direction"], row["source_role"], row["mapping_id"])].append(row)
    metrics: list[dict[str, Any]] = []
    for key in sorted(grouped):
        values = grouped[key]
        metrics.append({
            "concept": key[0],
            "direction": key[1],
            "source_role": key[2],
            "mapping_id": key[3],
            "example_count": len(values),
            "mean_base_token_f1": float(np.mean([row["base_token_f1"] for row in values])),
            "mean_base_common_prefix_fraction": float(np.mean([row["base_common_prefix_fraction"] for row in values])),
            "exact_base_match_fraction": float(np.mean([row["exactly_matches_base"] for row in values])),
            "mean_reference_token_f1": float(np.mean([row["reference_token_f1"] for row in values])),
            "mean_generated_token_count": float(np.mean([row["generated_token_count"] for row in values])),
            "eos_fraction": float(np.mean([row["terminated_eos"] for row in values])),
            "mean_repeated_fourgram_fraction": float(np.mean([row["repeated_fourgram_fraction"] for row in values])),
            "mean_generated_probe_score": float(np.mean([row["generated_probe_score"] for row in values if row["generated_probe_score"] is not None])) if any(row["generated_probe_score"] is not None for row in values) else None,
        })
    with METRICS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    rng = np.random.default_rng(BLIND_SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    key: dict[str, Any] = {"schema_version": 1, "procedure": "site-shuffling-v1-day24-qualitative-key", "blind_seed": BLIND_SEED, "items": []}
    packet = [
        "# Coupled-Generation Qualitative Review Packet",
        "",
        "This packet is locally blinded to intervention identity. Rate each output for task/content fulfillment, fluency/coherence, and safety/refusal stance using the accompanying template. These judgments are secondary and are not part of the mechanical behavioral gate.",
        "",
    ]
    rating_rows = []
    by_id = {row["example_id"]: row for row in load_experimental_split("safety-test")}
    for index, row in enumerate(shuffled, start=1):
        item_id = f"Q{index:03d}"
        record = by_id[row["example_id"]]
        packet.extend([
            f"## {item_id}",
            "",
            f"Concept: `{row['concept']}` · Dataset label: `{row['label']}`",
            "",
            "### Prompt",
            "",
            str(record["prompt"]),
            "",
            "### Output",
            "",
            str(row["generated_text"]) if row["generated_text"] else "*[empty output]*",
            "",
        ])
        key["items"].append({"item_id": item_id, "example_id": row["example_id"], "condition_id": row["condition_id"], "record_type": row["record_type"]})
        rating_rows.append({"item_id": item_id, "task_fulfillment_0_to_2": "", "fluency_0_to_2": "", "safety_or_refusal_stance": "", "notes": ""})
    PACKET_PATH.write_text("\n".join(packet) + "\n")
    KEY_PATH.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n")
    with RATING_TEMPLATE_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)

    selected = [row for row in interventions if row["source_role"] == "selected"]
    null = [row for row in interventions if row["source_role"] == "null"]
    summary = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day24-generation-summary",
        "status": "complete-descriptive",
        "scope": "four frozen examples; greedy coupled generation; secondary descriptive evidence",
        "selected_mean_base_token_f1": float(np.mean([row["base_token_f1"] for row in selected])),
        "null_mean_base_token_f1": float(np.mean([row["base_token_f1"] for row in null])),
        "selected_exact_base_match_fraction": float(np.mean([row["exactly_matches_base"] for row in selected])),
        "null_exact_base_match_fraction": float(np.mean([row["exactly_matches_base"] for row in null])),
        "qualitative_review_status": "packet-created-not-rated",
        "external_evaluator_used": False,
        "raw_results_sha256": sha256_file(RAW_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    expected_ids = set(authorization["examples"]["coupled_generation"]["example_ids"])
    checks = {
        "exact_rows": len(rows) == int(authorization["grid"]["coupled_generation_expected_rows"]),
        "unique_rows": len({(row["example_id"], row["condition_id"]) for row in rows}) == len(rows),
        "exact_examples": {row["example_id"] for row in rows} == expected_ids,
        "twenty_conditions_each": set(sum(1 for row in rows if row["example_id"] == example_id) for example_id in expected_ids) == {20},
        "all_outputs_present": all(isinstance(row["generated_text"], str) for row in rows),
        "finite_automatic_metrics": all(np.isfinite(float(row[field])) for row in interventions for field in ("base_token_f1", "base_common_prefix_fraction", "reference_token_f1", "repeated_fourgram_fraction")),
        "probe_scores_present_or_empty": all(row["generated_probe_score"] is not None or row["generated_token_count"] == 0 for row in rows),
        "authorization_hash": {row["authorization_sha256"] for row in rows} == {sha256_file(AUTHORIZATION_PATH)},
        "packet_and_key": PACKET_PATH.is_file() and KEY_PATH.is_file() and RATING_TEMPLATE_PATH.is_file(),
        "no_external_evaluator": summary["external_evaluator_used"] is False,
    }
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day24-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks, "raw_results_sha256": sha256_file(RAW_PATH)}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 24 audit failed")


if __name__ == "__main__":
    main()
