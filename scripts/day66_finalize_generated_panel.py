#!/usr/bin/env python3
"""Qualify precursor rewrites and lock the positive title-closure panel."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402
from day66_freeze_title_closure_program import (  # noqa: E402
    CONCEPTS,
    FINAL_POSITIVES,
    OUTPUT_DIR as SPLIT_DIR,
    PROGRAM_PATH,
    STYLE_SPECS,
    canonical_jsonl,
    qualification,
    sha256_file,
)


GENERATED_DIR = ROOT / "artifacts/title-closure-v1/generated-rewrites"
FINAL_POSITIVE_PATH = SPLIT_DIR / "final-positive.LOCKED.jsonl"
SUMMARY_PATH = ROOT / "results/day-66/generated-positive-qualification-summary.json"
CONTRACT_PATH = ROOT / "results/day-66/frozen-title-closure-final-contract.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    program = read_json(PROGRAM_PATH)
    program_hash = sha256_file(PROGRAM_PATH)
    selected: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    artifact_hashes = {}
    for concept in CONCEPTS:
        path = GENERATED_DIR / f"{concept}.jsonl"
        metadata_path = GENERATED_DIR / f"{concept}.metadata.json"
        metadata = read_json(metadata_path)
        if (
            metadata["program_sha256"] != program_hash
            or metadata["output_sha256"] != sha256_file(path)
            or metadata["concept"] != concept
        ):
            raise RuntimeError(f"generated rewrite provenance differs: {concept}")
        rows = read_jsonl(path)
        if [row["seed_order"] for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"generated seed order differs: {concept}")
        qualified = []
        seen = set()
        all_audits = []
        for row in rows:
            audit = qualification(concept, row["response"])
            all_audits.append({"seed_order": row["seed_order"], **audit})
            if audit["pass"] and row["content_sha256"] not in seen:
                seen.add(row["content_sha256"])
                qualified.append((row, audit))
        chosen = qualified[:FINAL_POSITIVES]
        if len(chosen) != FINAL_POSITIVES:
            raise RuntimeError(f"title closure stops: only {len(chosen)} qualified {concept} rewrites")
        for position, (value, audit) in enumerate(chosen):
            row = dict(value)
            row.update({
                "role": "final_positive",
                "example_id": f"closure-final-positive-{concept}-{position:04d}",
                "qualification": audit,
                "style_spec_sha256": hashlib.sha256(
                    json.dumps(STYLE_SPECS[concept], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            })
            selected.append(row)
        details[concept] = {
            "generated": len(rows),
            "qualified": len(qualified),
            "selected": len(chosen),
            "first_selected_seed_order": chosen[0][0]["seed_order"],
            "last_selected_seed_order": chosen[-1][0]["seed_order"],
            "failure_reasons": {
                "length": sum(not row["length_pass"] for row in all_audits),
                "style": sum(not row["style_pass"] for row in all_audits),
                "meta_language": sum(not row["meta_language_absent"] for row in all_audits),
            },
        }
        artifact_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        artifact_hashes[metadata_path.relative_to(ROOT).as_posix()] = sha256_file(metadata_path)
    value = canonical_jsonl(selected)
    FINAL_POSITIVE_PATH.write_bytes(value)
    summary = {
        "schema_version": 1,
        "procedure": "day66-generated-positive-qualification-v1",
        "program_sha256": program_hash,
        "result": "pass",
        "selection_or_rule_changes": False,
        "concepts": details,
        "final_positive": {
            "path": FINAL_POSITIVE_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(FINAL_POSITIVE_PATH),
            "rows": len(selected),
            "unique_content_hashes": len({row["content_sha256"] for row in selected}),
        },
        "generation_artifact_hashes": artifact_hashes,
    }
    if summary["final_positive"]["unique_content_hashes"] != len(selected):
        raise RuntimeError("generated positive responses are not globally unique")
    write_json_atomic(SUMMARY_PATH, summary)
    contract = {
        "schema_version": 1,
        "procedure": "day66-frozen-title-closure-final-contract-v1",
        "status": "frozen_after_generation_before_any_day66_chameleon_outcome",
        "program_sha256": program_hash,
        "qualification_summary_sha256": sha256_file(SUMMARY_PATH),
        "final_positive": summary["final_positive"],
        "generation_artifact_hashes": artifact_hashes,
        "concepts_in_order": program["concepts_in_order"],
        "roles": program["roles"],
        "calibration": program["calibration"],
        "final_evaluation": program["final_evaluation"],
        "closure_gates": program["closure_gates"],
        "control_verifier_repair": program["control_verifier_repair"],
        "uncertainty": program["uncertainty"],
        "parents": program["parents"],
        "stop_rules": program["stop_rules"],
    }
    write_json_atomic(CONTRACT_PATH, contract)
    print(json.dumps({
        "result": "pass",
        "selected_rows": len(selected),
        "final_positive_sha256": summary["final_positive"]["sha256"],
        "final_contract_sha256": sha256_file(CONTRACT_PATH),
        "qualified_counts": {concept: details[concept]["qualified"] for concept in CONCEPTS},
    }, indent=2))


if __name__ == "__main__":
    main()
