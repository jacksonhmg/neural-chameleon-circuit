#!/usr/bin/env python3
"""Run the bounded real-checkpoint preflight for Gate 1 machinery."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    ComponentEffectRunner,
    LinearProbe,
    PairedInterventionRunner,
    RealizedForwardRunner,
    audit_realized_forward,
    diagnostic_examples_from_batch,
    load_experimental_split,
    probe_sequence_scores,
    probe_token_margins,
    validate_diagnostic_examples,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
PLAN_PATH = ROOT / "results/day-31/frozen-acquired-writer-plan.json"
OUTPUT_PATH = ROOT / "results/day-32/real-checkpoint-preflight.json"


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


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def load_model() -> tuple[PairedInterventionRunner, Any]:
    torch.manual_seed(31003)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    return PairedInterventionRunner(model, tokenizer), tokenizer


def main() -> None:
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(ROOT / "src/neural_chameleon/causal_mechanisms.py", commit)
    require_committed(ROOT / "src/neural_chameleon/mechanism_prediction.py", commit)
    require_committed(PLAN_PATH, commit)
    plan = json.loads(PLAN_PATH.read_text())
    tolerances = plan["realized_forward_accounting"]["primary_tolerances"]

    records = sorted(
        (
            row
            for row in load_experimental_split("validation")
            if row["concept"] == "all-caps" and int(row["label"]) == 1
        ),
        key=lambda row: (len(row["response"]), row["example_id"]),
    )[:2]
    if len(records) != 2:
        raise RuntimeError(
            "real-checkpoint preflight requires two validation positives"
        )

    runner, _tokenizer = load_model()
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        trigger="all-caps",
    )
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    normal = realized.run(pair.normal)
    triggered = realized.run(pair.triggered)
    probe_paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    probes = tuple(LinearProbe.load(path) for path in probe_paths)
    if len(probes) != 13:
        raise RuntimeError(f"expected 13 released probes, found {len(probes)}")
    normal_audit = audit_realized_forward(normal, runner.layers, probes)
    triggered_audit = audit_realized_forward(triggered, runner.layers, probes)

    k12 = plan["component_sets"]["k12_ordered"]
    identity = ComponentEffectRunner(runner, monitor_layer=12).run(
        pair.normal, normal, normal, k12
    )
    baseline_margins = probe_token_margins(normal.monitor_residual, probes)
    total_margins = probe_token_margins(identity.total, probes)
    direct_margins = probe_token_margins(identity.direct_path, probes)
    baseline_scores = probe_sequence_scores(baseline_margins, normal.response_mask)
    total_scores = probe_sequence_scores(total_margins, normal.response_mask)
    direct_scores = probe_sequence_scores(direct_margins, normal.response_mask)
    identity_hidden_total = float(
        (identity.total.values.float() - normal.monitor_residual.values.float())
        .abs()[normal.response_mask]
        .max()
    )
    identity_hidden_direct = float(
        (identity.direct_path.values.float() - normal.monitor_residual.values.float())
        .abs()[normal.response_mask]
        .max()
    )
    identity_score_total = float((total_scores - baseline_scores).abs().max())
    identity_score_direct = float((direct_scores - baseline_scores).abs().max())

    diagnostic_examples = diagnostic_examples_from_batch(
        [row["example_id"] for row in records],
        [row["concept"] for row in records],
        normal,
        triggered,
        k12,
        runner.layers,
        normal_state_layer=8,
    )
    leakage_audit = validate_diagnostic_examples(diagnostic_examples, k12)
    accounting = {
        "normal": normal_audit.to_dict(),
        "correct_trigger": triggered_audit.to_dict(),
    }
    max_hidden = max(value["hidden_max_abs_error"] for value in accounting.values())
    max_allocation = max(
        value["attention_allocation_max_abs_error"] for value in accounting.values()
    )
    max_margin = max(
        value["probe_margin_max_abs_error"] for value in accounting.values()
    )
    max_score = max(
        value["sequence_score_max_abs_error"] for value in accounting.values()
    )
    checks = {
        "paired_response_ids_equal": torch.equal(
            pair.normal.response_ids, pair.triggered.response_ids
        ),
        "paired_response_masks_equal": torch.equal(
            pair.normal.response_mask, pair.triggered.response_mask
        ),
        "all_13_probes_loaded": len(probes) == 13,
        "hidden_accounting_within_tolerance": max_hidden
        <= tolerances["max_hidden_absolute_error"],
        "head_allocation_within_tolerance": max_allocation
        <= tolerances["max_attention_allocation_absolute_error"],
        "probe_margin_within_tolerance": max_margin
        <= tolerances["max_probe_margin_absolute_error"],
        "sequence_score_within_tolerance": max_score
        <= tolerances["max_sequence_score_absolute_error"],
        "k12_total_identity_within_tolerance": identity_score_total
        <= plan["causal_effects"]["identity_tolerances"]["max_probe_score_change"],
        "k12_direct_identity_within_tolerance": identity_score_direct
        <= plan["causal_effects"]["identity_tolerances"]["max_probe_score_change"],
        "k12_total_identity_hidden_within_accounting_tolerance": identity_hidden_total
        <= tolerances["max_hidden_absolute_error"],
        "k12_direct_identity_hidden_within_accounting_tolerance": identity_hidden_direct
        <= tolerances["max_hidden_absolute_error"],
        "diagnostic_feature_contract_exact": leakage_audit["feature_fields"]
        == [
            "observed_k12_delta",
            "normal_resid_post_8",
            "response_relative_position",
        ],
        "diagnostic_forbidden_fields_absent": not leakage_audit[
            "forbidden_feature_fields_present"
        ],
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    result = {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-real-checkpoint-preflight",
        "implementation_commit": commit,
        "plan_sha256": sha256_file(PLAN_PATH),
        "model_revision": plan["models"]["chameleon"]["revision"],
        "example_ids": [row["example_id"] for row in records],
        "response_token_counts": [
            example.token_count for example in diagnostic_examples
        ],
        "accounting": accounting,
        "identity": {
            "k12_total_hidden_max_abs": identity_hidden_total,
            "k12_direct_hidden_max_abs": identity_hidden_direct,
            "k12_total_probe_score_max_abs": identity_score_total,
            "k12_direct_probe_score_max_abs": identity_score_direct,
        },
        "leakage_audit": leakage_audit,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
        "scientific_outcome_boundary": "implementation preflight only; no acquisition or prediction gate statistic computed",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["result"] != "pass":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
