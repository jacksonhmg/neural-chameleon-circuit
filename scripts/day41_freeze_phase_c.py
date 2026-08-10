#!/usr/bin/env python3
"""Materialize the outcome-free Phase C semantic-conditioning contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import load_records  # noqa: E402


OUTPUT_PATH = ROOT / "results/day-41/frozen-phase-c-contract.json"
PHASE_AB_CONTRACT_PATH = ROOT / "results/day-36/frozen-phase-a-b-contract.json"
PHASE_AB_GATE_PATH = ROOT / "results/day-40/mechanism-signature-summary.json"
PHASE_AB_AUDIT_PATH = ROOT / "results/day-40/phase-a-b-audit.json"
SELECTION_PATH = ROOT / "results/day-39/development-selection.json"
ANALYSIS_PLAN_PATH = ROOT / "data/splits/day04-v1/analysis-plan.json"
AUTHORIZATION_PATH = ROOT / "decision-log/0033-authorize-post-gate1-phase-a-b-execution.md"
TOKENIZER_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"

DIFFERENT_CONCEPTS = {
    "HTML": "deception",
    "deception": "confused",
    "confused": "german",
    "german": "mathematical",
    "mathematical": "HTML",
    "comforting": "finnish",
    "finnish": "harmful",
    "harmful": "jokey",
    "jokey": "comforting",
    "all-caps": "biology-focused",
    "biology-focused": "chemistry-based",
    "chemistry-based": "literature-focused",
    "literature-focused": "all-caps",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def selected_examples() -> dict[str, list[str]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_records():
        if int(record["label"]) == 1:
            groups[record["concept"]].append(record)
    if set(groups) != set(DIFFERENT_CONCEPTS):
        raise RuntimeError("Phase C concept panel differs from the frozen 13 concepts")
    selected: dict[str, list[str]] = {}
    for concept, records in sorted(groups.items()):
        ordered = sorted(
            records,
            key=lambda row: hashlib.sha256(
                f"phase-c-semantic:{row['example_id']}".encode()
            ).hexdigest(),
        )
        selected[concept] = [row["example_id"] for row in ordered[:8]]
        if len(selected[concept]) != 8:
            raise RuntimeError(f"Phase C requires eight positives for {concept}")
    if len({value for values in selected.values() for value in values}) != 104:
        raise RuntimeError("Phase C selected example IDs are not unique")
    return selected


def prompt_pair_audit(
    selected: Mapping[str, list[str]], plan: Mapping[str, Any]
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    records = {row["example_id"]: row for row in load_records()}
    triggers = plan["conditions"]["correct_triggers"]
    template = plan["conditions"]["correct_trigger_template"]
    digest = hashlib.sha256()
    token_counts: dict[str, int] = {}
    for concept, example_ids in sorted(selected.items()):
        different = DIFFERENT_CONCEPTS[concept]
        correct_trigger = triggers[concept]
        different_trigger = triggers[different]
        for example_id in example_ids:
            prompt = records[example_id]["prompt"]
            correct = template.format(trigger=correct_trigger, prompt=prompt)
            alternative = template.format(trigger=different_trigger, prompt=prompt)
            rendered_correct = tokenizer.apply_chat_template(
                [{"role": "user", "content": correct}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rendered_alternative = tokenizer.apply_chat_template(
                [{"role": "user", "content": alternative}],
                tokenize=False,
                add_generation_prompt=True,
            )
            correct_ids = tokenizer(
                rendered_correct, add_special_tokens=False
            ).input_ids
            alternative_ids = tokenizer(
                rendered_alternative, add_special_tokens=False
            ).input_ids
            if len(correct_ids) != len(alternative_ids):
                raise RuntimeError(f"unmatched prompt token count for {example_id}")
            token_counts[example_id] = len(correct_ids)
            digest.update(
                json.dumps(
                    {
                        "example_id": example_id,
                        "correct_ids": correct_ids,
                        "different_ids": alternative_ids,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            )
    return {
        "all_104_rendered_prompt_pairs_token_count_matched": True,
        "minimum_rendered_prompt_tokens": min(token_counts.values()),
        "maximum_rendered_prompt_tokens": max(token_counts.values()),
        "ordered_pair_token_ids_sha256": digest.hexdigest(),
    }


def main() -> None:
    gate = read_json(PHASE_AB_GATE_PATH)
    audit = read_json(PHASE_AB_AUDIT_PATH)
    if (
        gate["gates"]["scientific_continue_gate"] != "pass"
        or audit["result"] != "pass"
    ):
        raise RuntimeError("Phase C cannot be frozen without passing Phase A-B")
    plan = read_json(ANALYSIS_PLAN_PATH)
    selected = selected_examples()
    pair_audit = prompt_pair_audit(selected, plan)
    triggers = plan["conditions"]["correct_triggers"]
    irrelevant = plan["conditions"]["irrelevant_triggers"]
    pairs = {
        concept: {
            "correct_concept": concept,
            "correct_trigger": triggers[concept],
            "different_concept": different,
            "different_trigger": triggers[different],
            "irrelevant_trigger": irrelevant[concept],
        }
        for concept, different in sorted(DIFFERENT_CONCEPTS.items())
    }
    value = {
        "schema_version": 1,
        "status": "frozen-before-phase-c-outcomes",
        "procedure": "post-Gate-1 Phase C semantic conditioning v1",
        "frozen_at_commit": git_head(),
        "authorization": {
            "decision": str(AUTHORIZATION_PATH.relative_to(ROOT)),
            "sha256": sha256_file(AUTHORIZATION_PATH),
            "scope": "conditionally authorized local Phase C execution after a passing Phase A-B gate and committed Phase C contract",
        },
        "prerequisites": {
            "phase_a_b_contract_sha256": sha256_file(PHASE_AB_CONTRACT_PATH),
            "phase_a_b_gate_sha256": sha256_file(PHASE_AB_GATE_PATH),
            "phase_a_b_audit_sha256": sha256_file(PHASE_AB_AUDIT_PATH),
            "development_selection_sha256": sha256_file(SELECTION_PATH),
            "phase_a_b_continue_gate": "pass",
            "sealed_complete_writer_gate_1": "fail",
        },
        "evidence_role": {
            "class": "existing-data prospective semantic-conditioning development evidence",
            "fresh_confirmation": False,
            "all_13_concepts_and_104_examples_are_development_only": True,
        },
        "model": read_json(PHASE_AB_CONTRACT_PATH)["models"]["chameleon"],
        "dtype": "bfloat16",
        "attention_implementation": "eager",
        "monitor_layer": 12,
        "operation": {
            "mechanism_class": "direct_reconfiguration",
            "primary_site": "all 12 frozen selected pre-o_proj response-head slices",
            "component_ids": read_json(PHASE_AB_CONTRACT_PATH)["component_sets"][
                "k12_ordered"
            ],
            "primary_effect": "complete standardized 13-probe vector from direct-path full-K12 source replacement into the correct-trigger target",
            "total_effect": "ordinary full-model full-K12 source replacement into the correct-trigger target",
            "downstream_frontier": {
                "source_layer": 11,
                "frontier_id": "F3",
                "component_ids": read_json(PHASE_AB_CONTRACT_PATH)[
                    "component_sets"
                ]["layer_groups"]["layer_11"],
            },
            "attention_source_diagnostic": {
                "site_id": "group.selected_heads.layer_10",
                "operation": "concept_span_ov",
                "claim_role": "descriptive secondary diagnostic; not the central operation",
            },
        },
        "upstream_boundary": {
            "site": "layer_08.block_output",
            "interpretation": "residual after layer 8 and strictly before the earliest selected layer-9 head",
            "source_positions": "tokenizer-aligned named-concept prompt span only",
            "pooling_for_prediction": "arithmetic mean over named-concept tokens in original order after alignment audit",
            "hidden_substitution": "replace correct-trigger named-concept-span residual rows with aligned different-trigger residual rows; leave every other row unchanged",
            "response_state_baseline": "arithmetic mean of valid normal-condition response residuals at layer_08.block_output",
            "expansion_or_boundary_search": "prohibited in v1; failure rejects concept-span mediation",
        },
        "conditions": {
            "order": [
                "normal",
                "correct_trigger",
                "irrelevant_trigger",
                "different_trigger",
                "hidden_different_substitution",
            ],
            "correct_trigger_template": plan["conditions"][
                "correct_trigger_template"
            ],
            "teacher_forcing": "reuse exact response-token IDs and masks across every condition",
            "pairs": pairs,
            "pair_audit": pair_audit,
            "irrelevant_policy": "retain the Day 4 frozen irrelevant trigger even when its token length differs; it is a separate specificity control, not the matched natural-text benchmark",
        },
        "population": {
            "concept_count": 13,
            "positive_examples_per_concept": 8,
            "total_examples": 104,
            "selection": "ascending SHA-256 of phase-c-semantic:{example_id} within concept and positive label",
            "example_ids": selected,
            "replacement_or_exclusion": "prohibited after this freeze",
        },
        "expected_execution_matrix": {
            "condition_endpoint_rows": 520,
            "source_conditions": [
                "different_trigger",
                "hidden_different_substitution",
                "irrelevant_trigger",
            ],
            "paths": ["direct", "total", "frontier_F3"],
            "causal_effect_rows": 936,
            "total_rows": 1456,
        },
        "features_and_prediction": {
            "target": "standardized 13-probe direct-effect vector of different-trigger full-K12 replacement into correct-trigger target",
            "upstream_feature": "different-minus-correct layer-8 named-concept-span mean residual",
            "normal_feature": "normal layer-8 valid-response mean residual",
            "projection": {
                "kind": "fixed Rademacher Johnson-Lindenstrauss projection divided by sqrt(64)",
                "dimension_per_feature_block": 64,
                "seed": 41001,
            },
            "full_predictor": "ridge on concatenated projected upstream and normal feature blocks",
            "baseline_predictor": "ridge on projected normal feature block only",
            "ridge_lambda": 10.0,
            "standardization": "fit feature means and positive standard deviations on training concepts only; target is not standardized beyond the frozen probe scaling",
            "evaluation": "13 leave-one-concept-out folds; no concept-specific coefficient; report every concept",
        },
        "metrics": {
            "probe_standardization": "results/day-33/intermediate-prediction-summary.json probe_standardization in released probe order",
            "vector_recovery": "1 - sum squared error / max(sum squared natural-different effect, 1e-6), computed per example then equal-concept averaged",
            "writer_scaling": "coordinatewise standard deviation of normal pooled K12 states across all 104 examples with 1e-6 floor; normal-only and therefore outcome-independent",
            "irrelevant_ratio": "equal-concept mean L2 norm of irrelevant direct-effect vector divided by different-trigger direct-effect norm with 1e-6 floor",
            "predictive_snmse": "sum squared prediction error divided by sum squared target around the training-fold target mean, accumulated on held-out examples and equal-concept averaged",
            "bootstrap": "10,000 deterministic concept bootstrap replicates; seed 41002; one-sided percentile bounds for gates and two-sided intervals for reporting",
        },
        "scientific_continue_gate": {
            "precondition": "complete implementation audit and exact real-checkpoint preflight pass",
            "upstream_prediction": {
                "full_predictor_one_sided_95_upper_snmse_at_most": 0.9,
                "full_minus_baseline_one_sided_95_upper_at_most": -0.1,
                "concepts_with_full_snmse_better_than_baseline_at_least": 9,
            },
            "hidden_substitution": {
                "direct_probe_vector_recovery_one_sided_95_lower_at_least": 0.5,
                "total_probe_vector_recovery_one_sided_95_lower_at_least": 0.5,
                "frontier_probe_vector_recovery_one_sided_95_lower_at_least": 0.5,
                "pooled_k12_pattern_recovery_one_sided_95_lower_at_least": 0.5,
            },
            "specificity": {
                "irrelevant_to_different_direct_effect_ratio_one_sided_95_upper_at_most": 0.75
            },
            "stability": "hidden direct-effect recovery is positive in at least 9 of 13 concepts; all concepts reported",
            "pass": "all clauses pass conjunctively",
            "pass_consequence": "fresh-confirmation contract and disjoint data-role manifest drafting permitted",
            "fail_consequence": "stop before fresh confirmation and retain the Phase A-B acquired direct-reconfiguration result",
            "prohibited_adjustments": [
                "change an example, concept pair, upstream boundary, feature projection, ridge lambda, metric, threshold, or control after any Phase C outcome",
                "search another boundary after a hidden-substitution failure",
                "promote the attention-source diagnostic to the central operation",
                "describe Phase C as fresh confirmation",
            ],
        },
        "implementation_gate": {
            "requirements": [
                "all selected IDs, pairs, trigger strings, response IDs, masks, and matched prompt token counts reproduce the contract",
                "same-condition upstream substitution is identity within the inherited numerical tolerances",
                "full K12 same-condition replacement is identity on direct, total, and frontier paths",
                "every expected condition and causal row is present exactly once",
                "all hooks are removed after every batch and all values are finite",
                "two reducer runs from the committed analysis implementation are byte-identical",
            ],
            "failure": "blocks interpretation and is neither a Phase C scientific pass nor fail",
        },
        "required_outputs": [
            "phase-c-audit.json",
            "semantic-conditioning-summary.json",
            "phase-c-artifact-manifest.json",
        ],
        "no_release_boundary": "no push, tag, release, submission, upload, external message, or author contact is authorized",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
