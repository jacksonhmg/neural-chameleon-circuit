#!/usr/bin/env python3
"""Materialize the prospective Day 21 authorization without model outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import load_experimental_split, parse_head_id  # noqa: E402
from day17_run_transfer_atlas import PLAN_PATH, causal_subset  # noqa: E402


RESULT_DIR = ROOT / "results/day-21"
AUTHORIZATION_PATH = RESULT_DIR / "confirmation-authorization.json"
SELECTION_PATH = ROOT / "results/day-19/benign-selected-mappings.json"
NEGATIVE_METRICS_PATH = ROOT / "results/day-20/specificity-metrics.csv"
SAFETY_SPLIT_PATH = ROOT / "data/splits/day04-v1/safety-test.LOCKED.jsonl"
OUTCOME_PATHS = (
    RESULT_DIR / "confirmation-results.working.jsonl",
    RESULT_DIR / "confirmation-results.jsonl.gz",
    RESULT_DIR / "confirmation-cells.csv",
    RESULT_DIR / "confirmation-gate.json",
)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(example_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(example_ids) + "\n").encode()).hexdigest()


def confirmation_subset(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    pilot_ids = {row["example_id"] for row in causal_subset(records)}
    return sorted(
        (
            row
            for row in records
            if int(row["label"]) == 1 and row["example_id"] not in pilot_ids
        ),
        key=lambda row: (row["concept"], row["example_id"]),
    )


def build_null_partner(plan: Mapping[str, Any]) -> dict[str, str]:
    selected = list(plan["selected_heads"])
    null = list(plan["null_heads"]["members"])
    result: dict[str, str] = {}
    for layer in sorted({parse_head_id(head_id)[0] for head_id in selected}):
        selected_layer = sorted(
            head_id for head_id in selected if parse_head_id(head_id)[0] == layer
        )
        null_layer = sorted(
            head_id for head_id in null if parse_head_id(head_id)[0] == layer
        )
        if len(selected_layer) != len(null_layer):
            raise ValueError(f"selected/null count mismatch at layer {layer}")
        result.update(dict(zip(selected_layer, null_layer, strict=True)))
    if set(result) != set(selected) or len(set(result.values())) != len(null):
        raise ValueError("null partner map is not a complete layer-matched bijection")
    return result


def null_source_mapping(
    selected_mapping: Mapping[str, str], null_partner: Mapping[str, str]
) -> dict[str, str]:
    return {
        destination_id: null_partner[source_id]
        for destination_id, source_id in selected_mapping.items()
    }


def main() -> None:
    if any(path.exists() for path in OUTCOME_PATHS):
        raise RuntimeError("refusing late authorization: Day 21 outcome artifact exists")
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        PLAN_PATH,
        SELECTION_PATH,
        NEGATIVE_METRICS_PATH,
        SAFETY_SPLIT_PATH,
    ):
        require_committed(path, commit)

    plan = json.loads(PLAN_PATH.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    if selection.get("status") != "pass" or len(selection["selected_mappings"]) != 4:
        raise ValueError("exactly four passing benign-selected mappings are required")
    expected_ids = {
        "within_15015",
        "within_15004",
        "cross_15130",
        "cross_15122",
    }
    selected_ids = {row["mapping_id"] for row in selection["selected_mappings"]}
    if selected_ids != expected_ids:
        raise ValueError(f"unexpected benign-selected mapping IDs: {selected_ids}")

    records = confirmation_subset(load_experimental_split("safety-test"))
    ids_by_concept = {
        concept: [row["example_id"] for row in records if row["concept"] == concept]
        for concept in ("deception", "harmful")
    }
    if {concept: len(ids) for concept, ids in ids_by_concept.items()} != {
        "deception": 46,
        "harmful": 84,
    }:
        raise ValueError("confirmation subset count changed")

    null_partner = build_null_partner(plan)
    mappings = []
    for row in selection["selected_mappings"]:
        mapping = dict(row["destination_to_source"])
        mappings.append(
            {
                "mapping_id": row["mapping_id"],
                "mapping_class": row["mapping_class"],
                "selection_development_score": row["development_score"],
                "selection_validation_score": row["validation_score"],
                "mapping_sha256": row["mapping_sha256"],
                "selected_destination_to_source": mapping,
                "null_destination_to_source": null_source_mapping(
                    mapping, null_partner
                ),
            }
        )

    authorization = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day21-authorization",
        "status": "authorized-before-confirmation-outcomes",
        "authorized_on": "2026-08-07",
        "implementation_commit": commit,
        "outcome_artifacts_absent_at_authorization": True,
        "selection_used_safety_outcomes": False,
        "publication_scope": "local commits only; no push, tag, release, draft transmission, author contact, or external message",
        "inputs": {
            "site_shuffling_plan": {
                "path": PLAN_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(PLAN_PATH),
            },
            "benign_selection": {
                "path": SELECTION_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(SELECTION_PATH),
            },
            "safety_split": {
                "path": SAFETY_SPLIT_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(SAFETY_SPLIT_PATH),
            },
            "negative_specificity_reference": {
                "path": NEGATIVE_METRICS_PATH.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(NEGATIVE_METRICS_PATH),
                "rule": "reuse the already completed Day 20 fixed negative subset; do not regenerate or rerank controls after confirmation",
            },
        },
        "confirmation_examples": {
            concept: {
                "count": len(ids),
                "ids_sha256": ids_sha256(ids),
                "example_ids": ids,
            }
            for concept, ids in ids_by_concept.items()
        },
        "selected_heads": list(plan["selected_heads"]),
        "null_partner_by_selected_source": null_partner,
        "mappings": mappings,
        "grid": {
            "base_conditions": ["normal", "correct_trigger"],
            "source_roles": ["selected", "null"],
            "absolute_source_conditions": ["normal", "correct_trigger"],
            "delta_condition": "destination-relative raw trigger-minus-normal source delta; add in normal base and subtract in correct-trigger base",
            "baseline_conditions_per_example": 2,
            "conditions_per_mapping_per_base": 6,
            "conditions_per_example": 50,
            "expected_examples": 130,
            "expected_rows": 6500,
        },
        "inference": {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 15210,
            "unit": "paired example within concept",
            "interval": "95 percent percentile bootstrap",
            "mapping_population": "unweighted mean over the four benign-selected mappings before example bootstrap",
            "primary_estimands": ["conditional", "delta"],
            "primary_comparator": "route-matched layer-matched null-source mean",
            "secondary_estimands": ["absolute", "same_condition_mismatch"],
        },
        "mechanical_gate": {
            "portable_support": "for every concept, direction, and primary estimand, both the selected-source population lower 95% bound and selected-minus-null lower 95% bound exceed zero",
            "qualified_support": "portable_support fails, but every selected-source population point estimate for both primary estimands and both directions is positive",
            "generic_disruption": "portable_support and qualified_support fail, every selected-source absolute point estimate is positive, and no corrected primary lower bound exceeds zero",
            "rejected": "all other patterns, including a reversed corrected primary point estimate",
            "precedence": [
                "portable_support",
                "qualified_support",
                "generic_disruption",
                "rejected",
            ],
            "diagnostic_boundary": "Day 20 behavior and manifold diagnostics qualify prose interpretation but cannot replace or alter this causal gate",
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    AUTHORIZATION_PATH.write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Authorized {sum(len(ids) for ids in ids_by_concept.values())} examples and 6500 rows at {AUTHORIZATION_PATH.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
