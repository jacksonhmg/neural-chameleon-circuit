#!/usr/bin/env python3
"""Independently audit layer indexing and hook placement on the real checkpoint."""

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
    ActivationKind,
    CapturedActivation,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    TruncatedComponentRunner,
    load_experimental_split,
)


MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_PATH = ROOT / "external/minimal_neural_chameleons/probes/all-caps_weights.pt"
PLAN_PATH = ROOT / "results/day-14/frozen-falsification-plan.json"
OUTPUT_PATH = ROOT / "results/day-14/machinery-audit.json"


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


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} versus {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max())


def masked_max_abs(
    left: torch.Tensor,
    right: torch.Tensor,
    response_mask: torch.Tensor,
) -> float:
    """Compare only real response tokens, excluding batch-padding slots."""
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {tuple(left.shape)} versus {tuple(right.shape)}")
    mask = response_mask.bool().to(left.device)
    if mask.shape != left.shape[:2]:
        raise ValueError(
            f"mask mismatch: {tuple(mask.shape)} versus {tuple(left.shape[:2])}"
        )
    return float((left.float() - right.float()).abs()[mask].max())


def response_values(tensor: torch.Tensor, condition: Any) -> torch.Tensor:
    start = condition.response_start
    stop = start + condition.response_width
    return tensor[:, start:stop, :].detach().cpu().clone()


def nll(result: Any) -> torch.Tensor:
    logprobs = result.response_token_logprobs().float()
    mask = result.response_mask
    return (-logprobs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def load_model() -> tuple[PairedInterventionRunner, Any]:
    torch.manual_seed(42)
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
    require_committed_file(Path(__file__).resolve(), commit)
    require_committed_file(PLAN_PATH, commit)
    plan = json.loads(PLAN_PATH.read_text())
    tolerances = plan["machinery_audit"]
    examples = sorted(
        (
            row
            for row in load_experimental_split("validation")
            if row["concept"] == "all-caps" and int(row["label"]) == 1
        ),
        key=lambda row: row["example_id"],
    )[:2]
    if len(examples) != 2:
        raise ValueError("expected two all-caps validation examples")
    runner, _tokenizer = load_model()
    pair = runner.prepare_pairs(
        [row["prompt"] for row in examples],
        [row["response"] for row in examples],
        "all-caps",
    )
    if not torch.equal(pair.normal.response_ids, pair.triggered.response_ids):
        raise RuntimeError("paired response IDs differ")
    if not torch.equal(pair.normal.response_mask, pair.triggered.response_mask):
        raise RuntimeError("paired response masks differ")
    probe = LinearProbe.load(PROBE_PATH)
    resid_pre_12 = PatchSite(ActivationKind.RESID_PRE, 12)
    attention_12 = PatchSite(ActivationKind.ATTN_OUT, 12)
    mlp_12 = PatchSite(ActivationKind.MLP_OUT, 12)
    block_12 = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    resid_pre_13 = PatchSite(ActivationKind.RESID_PRE, 13)
    head_sites = tuple(
        PatchSite(ActivationKind.HEAD_OUTPUT, 12, head=head)
        for head in range(16)
    )
    sites = (
        resid_pre_12,
        attention_12,
        mlp_12,
        block_12,
        resid_pre_13,
        *head_sites,
    )

    raw: dict[str, torch.Tensor] = {}
    handles = []
    condition = pair.normal

    def capture_output(name: str):
        def hook(_module: Any, _args: tuple[Any, ...], output: Any):
            raw[name] = response_values(runner._first_tensor(output), condition)

        return hook

    def capture_input(name: str):
        def hook(_module: Any, args: tuple[Any, ...]):
            raw[name] = response_values(runner._first_tensor(args), condition)

        return hook

    handles.extend(
        [
            runner.layers[12].register_forward_hook(capture_output("block_12_output")),
            runner.layers[13].register_forward_pre_hook(capture_input("resid_pre_13_input")),
            runner.layers[12].post_attention_layernorm.register_forward_hook(
                capture_output("post_attention_layernorm_output")
            ),
            runner.layers[12].post_feedforward_layernorm.register_forward_hook(
                capture_output("post_feedforward_layernorm_output")
            ),
            runner.layers[12].self_attn.o_proj.register_forward_pre_hook(
                capture_input("o_proj_input")
            ),
        ]
    )
    try:
        audited = runner.run(condition, capture_sites=sites)
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(raw) != {
        "block_12_output",
        "resid_pre_13_input",
        "post_attention_layernorm_output",
        "post_feedforward_layernorm_output",
        "o_proj_input",
    }:
        raise RuntimeError("one or more independent raw hooks did not fire")

    placement_checks = {
        "runner_block12_vs_raw_layer12_output": max_abs(
            audited.captures[block_12].values, raw["block_12_output"]
        ),
        "runner_residpre13_vs_raw_layer13_input": max_abs(
            audited.captures[resid_pre_13].values, raw["resid_pre_13_input"]
        ),
        "block12_output_vs_residpre13_input": max_abs(
            raw["block_12_output"], raw["resid_pre_13_input"]
        ),
        "runner_attention_vs_raw_module": max_abs(
            audited.captures[attention_12].values,
            raw["post_attention_layernorm_output"],
        ),
        "runner_mlp_vs_raw_module": max_abs(
            audited.captures[mlp_12].values,
            raw["post_feedforward_layernorm_output"],
        ),
    }
    reconstructed_heads = torch.stack(
        [audited.captures[site].values for site in head_sites], dim=2
    ).reshape_as(raw["o_proj_input"])
    placement_checks["all_heads_reconstruct_raw_o_proj_input"] = max_abs(
        reconstructed_heads, raw["o_proj_input"]
    )

    with torch.inference_mode():
        direct = runner.model(
            input_ids=condition.input_ids.to(runner.device),
            attention_mask=condition.attention_mask.to(runner.device),
            position_ids=condition.position_ids.to(runner.device),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            logits_to_keep=1,
        )
    if direct.hidden_states is None or len(direct.hidden_states) < 14:
        raise RuntimeError("model did not return hidden_states[13]")
    hidden_13 = response_values(direct.hidden_states[13], condition)
    placement_checks["block12_output_vs_hidden_states_13"] = max_abs(
        audited.captures[block_12].values, hidden_13
    )
    hidden_capture = CapturedActivation(
        values=hidden_13,
        response_ids=condition.response_ids.clone(),
        response_mask=condition.response_mask.clone(),
    )
    full_scores = probe.score(audited.captures[block_12])
    hidden_scores = probe.score(hidden_capture)
    placement_checks["probe_score_runner_vs_hidden_states_13"] = max_abs(
        full_scores, hidden_scores
    )
    del direct

    truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
    truncated_result = truncated.run(condition)
    placement_checks["full_vs_truncated_probe_score"] = max_abs(
        full_scores, truncated_result.probe_scores
    )

    baseline = runner.run(
        condition,
        capture_sites=(resid_pre_12, attention_12, mlp_12, head_sites[0], block_12),
        retain_response_logprobs=True,
    )
    baseline_scores = probe.score(baseline.captures[block_12])
    identity_checks = []
    for site in (resid_pre_12, attention_12, mlp_12, head_sites[0], block_12):
        patched = runner.run(
            condition,
            capture_sites=(block_12,),
            patch_cache={site: baseline.captures[site]},
            retain_response_logprobs=True,
        )
        score_difference = max_abs(probe.score(patched.captures[block_12]), baseline_scores)
        nll_difference = max_abs(nll(patched), nll(baseline))
        identity_checks.append(
            {
                "site": site.label(),
                "score_max_abs_difference": score_difference,
                "nll_max_abs_difference": nll_difference,
                "exact_monitor_activation": torch.equal(
                    patched.captures[block_12].values,
                    baseline.captures[block_12].values,
                ),
                "within_tolerance": score_difference <= tolerances["score_tolerance"]
                and nll_difference <= tolerances["response_nll_tolerance"],
            }
        )

    normal_block = runner.run(pair.normal, capture_sites=(block_12, resid_pre_13))
    triggered_block = runner.run(pair.triggered, capture_sites=(block_12, resid_pre_13))
    replacement = runner.run(
        pair.triggered,
        capture_sites=(block_12,),
        patch_cache={block_12: normal_block.captures[block_12]},
    )
    response_mask = normal_block.captures[block_12].response_mask
    replacement_checks = {
        "monitor_all_slots_equal_normal": torch.equal(
            replacement.captures[block_12].values,
            normal_block.captures[block_12].values,
        ),
        "monitor_valid_response_activation_equals_normal": torch.equal(
            replacement.captures[block_12].values[response_mask.bool()],
            normal_block.captures[block_12].values[response_mask.bool()],
        ),
        "monitor_valid_response_activation_max_abs_difference_from_normal": masked_max_abs(
            replacement.captures[block_12].values,
            normal_block.captures[block_12].values,
            response_mask,
        ),
        "monitor_score_max_abs_difference_from_normal": max_abs(
            probe.score(replacement.captures[block_12]),
            probe.score(normal_block.captures[block_12]),
        ),
    }
    post_monitor_patch = runner.run(
        pair.triggered,
        capture_sites=(block_12,),
        patch_cache={resid_pre_13: normal_block.captures[resid_pre_13]},
    )
    replacement_checks["post_monitor_patch_leaves_block12_exact"] = torch.equal(
        post_monitor_patch.captures[block_12].values,
        triggered_block.captures[block_12].values,
    )
    replacement_checks["post_monitor_patch_score_max_abs_difference"] = max_abs(
        probe.score(post_monitor_patch.captures[block_12]),
        probe.score(triggered_block.captures[block_12]),
    )

    hook_count = runner.registered_hook_count()
    placement_pass = all(
        value <= tolerances["activation_tolerance"]
        for key, value in placement_checks.items()
        if "probe_score" not in key
    ) and all(
        value <= tolerances["score_tolerance"]
        for key, value in placement_checks.items()
        if "probe_score" in key
    )
    identity_pass = all(row["within_tolerance"] for row in identity_checks)
    replacement_pass = bool(
        replacement_checks["monitor_valid_response_activation_equals_normal"]
        and replacement_checks[
            "monitor_valid_response_activation_max_abs_difference_from_normal"
        ]
        <= tolerances["activation_tolerance"]
        and replacement_checks["monitor_score_max_abs_difference_from_normal"]
        <= tolerances["score_tolerance"]
        and replacement_checks["post_monitor_patch_leaves_block12_exact"]
        and replacement_checks["post_monitor_patch_score_max_abs_difference"]
        <= tolerances["score_tolerance"]
    )
    status = "pass" if (
        placement_pass
        and identity_pass
        and replacement_pass
        and torch.equal(pair.normal.response_ids, pair.triggered.response_ids)
        and torch.equal(pair.normal.response_mask, pair.triggered.response_mask)
        and hook_count == 0
    ) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day14-machinery-audit-v2",
        "status": status,
        "implementation_commit": commit,
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "examples": [row["example_id"] for row in examples],
        "model_decoder_layer_count": len(runner.layers),
        "hidden_state_index_for_post_block_12": 13,
        "response_ids_exact_across_pair": torch.equal(
            pair.normal.response_ids, pair.triggered.response_ids
        ),
        "response_masks_exact_across_pair": torch.equal(
            pair.normal.response_mask, pair.triggered.response_mask
        ),
        "placement_max_abs_differences": placement_checks,
        "identity_checks": identity_checks,
        "replacement_checks": replacement_checks,
        "registered_hook_count_after_audit": hook_count,
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        raise RuntimeError("Day 14 machinery audit failed")
    print(
        "Day 14 machinery audit passed: raw hooks, hidden-state index, identities, "
        "replacement, structural null, and cleanup verified.",
        flush=True,
    )


if __name__ == "__main__":
    main()
