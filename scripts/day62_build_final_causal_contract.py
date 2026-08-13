#!/usr/bin/env python3
"""Freeze the selected untouched panel and final causal-chain analysis contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import require_committed, write_json_atomic  # noqa: E402


PROGRAM_PATH = ROOT / "results/day-62/frozen-final-title-gate-program-contract.json"
PROBE_SUMMARY_PATH = ROOT / "results/day-62/new-probe-training-summary.json"
QUALIFICATION_PATH = ROOT / "results/day-62/qualification-calibration-summary.json"
PROTOTYPE_PATH = ROOT / "artifacts/final-title-gate-v1/qualification-prototypes.safetensors"
DAY57_PATH = ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
FINAL_CAUSAL_PATH = ROOT / "data/splits/day62-v1/final-causal.LOCKED.jsonl"
FINAL_NEGATIVE_PATH = ROOT / "data/splits/day62-v1/final-negative.LOCKED.jsonl"
OUTPUT_PATH = ROOT / "results/day-63/frozen-final-causal-chain-contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def ensure_hash(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} differs from its frozen parent")


def selected_rows(
    rows: list[dict[str, Any]], selected_pairs: list[str], selected_concepts: list[str]
) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("pair_id") in selected_pairs
        or row.get("probe_concept") in selected_concepts
    ]


def row_spec(rows: list[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    return {
        "parent_path": path.relative_to(ROOT).as_posix(),
        "parent_sha256": sha256_file(path),
        "selected_rows": len(rows),
        "selected_example_ids": [row["example_id"] for row in rows],
        "selected_content_sha256s": [row["content_sha256"] for row in rows],
    }


def main() -> None:
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        PROGRAM_PATH,
        PROBE_SUMMARY_PATH,
        QUALIFICATION_PATH,
    ):
        require_committed(path, commit)
    program = read_json(PROGRAM_PATH)
    probe_summary = read_json(PROBE_SUMMARY_PATH)
    qualification = read_json(QUALIFICATION_PATH)
    day57 = read_json(DAY57_PATH)
    if program["status"] != "frozen_before_any_day62_new_concept_model_outcome":
        raise RuntimeError("Day 62 parent contract has the wrong status")
    if not probe_summary["all_probe_quality_gates_pass"]:
        raise RuntimeError("not all prospectively frozen probes qualified")
    if not qualification["qualification_continue_gate_pass"]:
        raise RuntimeError("fewer than two prospectively ordered pairs qualified")
    ensure_hash(PROTOTYPE_PATH, qualification["prototype_sha256"], "prototype tensor")
    ensure_hash(
        FINAL_CAUSAL_PATH,
        program["roles"]["final_causal"]["sha256"],
        "locked final causal role",
    )
    ensure_hash(
        FINAL_NEGATIVE_PATH,
        program["roles"]["final_negative"]["sha256"],
        "locked final negative role",
    )

    selected_pairs = list(qualification["selected_pairs"])
    pair_lookup = {
        row["pair_id"]: row
        for row in read_json(ROOT / "data/splits/day62-v1/manifest.json")[
            "pairs_in_order"
        ]
    }
    selected_concepts = [
        concept
        for pair in selected_pairs
        for concept in (
            pair_lookup[pair]["concept_a"],
            pair_lookup[pair]["concept_b"],
        )
    ]
    if len(set(selected_concepts)) != len(selected_concepts):
        raise RuntimeError("selected final concepts overlap")
    causal_rows = selected_rows(
        read_jsonl(FINAL_CAUSAL_PATH), selected_pairs, selected_concepts
    )
    negative_rows = selected_rows(
        read_jsonl(FINAL_NEGATIVE_PATH), selected_pairs, selected_concepts
    )
    if len(causal_rows) != 16 * len(selected_pairs):
        raise RuntimeError("selected final causal panel has the wrong size")
    if len(negative_rows) != 32 * len(selected_concepts):
        raise RuntimeError("selected final negative panel has the wrong size")

    contract = {
        "schema_version": 1,
        "procedure": "day63-frozen-final-causal-chain-v1",
        "status": "frozen_before_any_locked_final_model_or_intervention_outcome",
        "frozen_on": "2026-08-12",
        "target_title": program["target_title"],
        "execution_commit": commit,
        "parents": {
            "program_contract_sha256": sha256_file(PROGRAM_PATH),
            "probe_summary_sha256": sha256_file(PROBE_SUMMARY_PATH),
            "qualification_summary_sha256": sha256_file(QUALIFICATION_PATH),
            "qualification_prototype_sha256": sha256_file(PROTOTYPE_PATH),
            "day57_contract_sha256": sha256_file(DAY57_PATH),
        },
        "selected_pairs_in_order": selected_pairs,
        "selected_pair_specs": [pair_lookup[pair] for pair in selected_pairs],
        "selected_concepts_in_order": selected_concepts,
        "probe_names_in_order": list(program["concepts"]),
        "probe_sha256s": {
            name: probe_summary["concepts"][name]["probe_sha256"]
            for name in program["concepts"]
        },
        "panels": {
            "final_causal": row_spec(causal_rows, FINAL_CAUSAL_PATH),
            "final_negative": row_spec(negative_rows, FINAL_NEGATIVE_PATH),
            "qualification_examples_per_pair": program["qualification"][
                "examples_per_pair"
            ],
            "calibration_negatives_per_probe": 128,
        },
        "predictions": {
            "prototype_path": PROTOTYPE_PATH.relative_to(ROOT).as_posix(),
            "prototype_sha256": sha256_file(PROTOTYPE_PATH),
            "coordinates": program["qualification"][
                "frozen_prediction_coordinates"
            ],
        },
        "calibration_thresholds": {
            concept: qualification["calibration_thresholds"][concept]
            for concept in selected_concepts
        },
        "models": day57["models"],
        "k12": day57["k12"],
        "semantic_matrix": {
            "directions_each_pair": ["a_to_b", "b_to_a"],
            "target_context_jobs_in_order": [
                "identity_target",
                "donor_kv_into_target",
                "irrelevant_kv_into_target",
                "matched_orthogonal_k12_into_target",
            ],
            "donor_context_jobs_in_order": [
                "identity_donor",
                "target_kv_into_donor",
                "target_k12_into_donor",
                "target_kv_into_donor_plus_donor_k12_restore",
            ],
            "semantic_swap": "natural trigger A versus natural trigger B on exact shared response tokens",
            "kv_operation": "source-side named-concept plus trigger-other pre-RoPE K and V substitution in selected K12 heads; response queries remain target-natural",
            "k12_clamp": "replace all 12 selected raw K12 response-head outputs with exact aligned natural target-context values",
            "restoration": "restore all 12 exact natural donor-context K12 outputs while donor semantic context and naturally recomputed 52-head tail remain active",
            "tail_policy": "only the declared 12 K12 raw-head sites are patched; all other heads, MLPs, residual paths, normalization, and layer-12 monitor values regenerate naturally",
        },
        "controls": {
            "irrelevant_trigger": "third prospectively assigned concept",
            "matched_orthogonal_k12": "per-head signed-permutation rotation preserving each target-to-donor head-delta norm",
            "orthogonal_seed_base": 63001,
            "identity_jobs_required": True,
            "identical_matrix_in_precursor": True,
        },
        "gates": program["final_gates"],
        "reductions": {
            "unit": "individual held-out example; medians over the complete selected panel",
            "directions": "every selected pair and both preregistered directions required",
            "complete_probe_vector": "all six prospectively trained new-concept probes",
            "operating_rule": "strict margin greater than frozen calibration threshold",
            "uncertainty": {
                "secondary_only": True,
                "method": "deterministic example bootstrap within each pair or concept",
                "draws": 10000,
                "seed": 63002,
                "interval": [0.025, 0.975],
            },
            "no_post_final_tuning": True,
        },
        "implementation_gates": program["implementation_gates"],
        "budget": program["budget"],
    }
    write_json_atomic(OUTPUT_PATH, contract)
    print(
        json.dumps(
            {
                "output": OUTPUT_PATH.relative_to(ROOT).as_posix(),
                "selected_pairs": selected_pairs,
                "selected_concepts": selected_concepts,
                "final_causal_examples": len(causal_rows),
                "final_negative_examples": len(negative_rows),
                "sha256": sha256_file(OUTPUT_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
