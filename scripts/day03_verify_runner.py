#!/usr/bin/env python3
"""Run the Day 3 intervention correctness checks on the released 9B checkpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.interventions import (  # noqa: E402
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    induction_fraction,
    recovery_fraction,
)


MODEL_REPO = "serteal/neural-chameleon-gemma_2_9b-layer_12"
MODEL_REVISION = "c4aeb57502a8bed6a62dac57819eec27fd5e3c99"
MINIMAL_REPO_REVISION = "f077304f6bebd31ded4cc868f2115a8c81067960"
FULL_REPO_REVISION = "261e6547f2042385c95c47ad25860aacedb6aa73"
BATCH_SCORE_MAX_ABS_DIFFERENCE = 1e-3
BATCH_ACTIVATION_MAX_RELATIVE_L2 = 1e-2
BATCH_ACTIVATION_MIN_COSINE = 0.9999
BATCH_LOGIT_MAX_RELATIVE_L2 = 2e-2
BATCH_LOGIT_MIN_COSINE = 0.9995


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12",
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=ROOT / "external/minimal_neural_chameleons/probes/german_weights.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/day-03/checkpoint-verification.json",
    )
    return parser.parse_args()


def release_unused_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def response_ids_hash(response_ids: torch.Tensor, response_mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for row_ids, row_mask in zip(response_ids, response_mask):
        digest.update(row_ids[row_mask].numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def max_valid_difference(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> float:
    expanded_mask = mask
    while expanded_mask.ndim < left.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(left)
    if not expanded_mask.any():
        return 0.0
    return float((left.float() - right.float()).abs()[expanded_mask].max())


def mean(values: torch.Tensor) -> float:
    return float(values.float().mean())


def path_reference(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def tensor_agreement(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    left_float = left.float().reshape(-1)
    right_float = right.float().reshape(-1)
    difference = left_float - right_float
    denominator = float(right_float.norm())
    return {
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "root_mean_square_difference": float(difference.square().mean().sqrt()),
        "relative_l2_difference": (
            float(difference.norm()) / denominator if denominator else 0.0
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                left_float.unsqueeze(0), right_float.unsqueeze(0)
            )[0]
        ),
        "reference_max_abs_value": float(right_float.abs().max()),
        "reference_root_mean_square": float(right_float.square().mean().sqrt()),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    model_dir = args.model_dir.resolve()
    probe_path = args.probe.resolve()
    output_path = args.output.resolve()

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )

    runner = PairedInterventionRunner(model, tokenizer)
    probe = LinearProbe.load(probe_path)
    prompts = [
        "Was ist die Hauptstadt von Deutschland?",
        "Guten Morgen! Wie heißt du?",
    ]
    responses = [
        "Die Hauptstadt von Deutschland ist Berlin.",
        "Guten Morgen! Ich bin ein KI-Assistent.",
    ]
    paired = runner.prepare_pairs(prompts, responses, trigger="german")

    all_sites = (
        PatchSite(ActivationKind.RESID_PRE, 12),
        PatchSite(ActivationKind.ATTN_OUT, 12),
        PatchSite(ActivationKind.MLP_OUT, 12),
        PatchSite(ActivationKind.BLOCK_OUTPUT, 12),
        PatchSite(ActivationKind.HEAD_OUTPUT, 12, head=0),
    )
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)

    normal = runner.run(
        paired.normal,
        capture_sites=all_sites,
        retain_response_logits=True,
    )
    normal_scores = probe.score(normal.captures[monitor_site], device=runner.device)
    print(f"normal mean: {mean(normal_scores):.6f}", flush=True)
    release_unused_memory()

    triggered = runner.run(paired.triggered, capture_sites=(monitor_site,))
    triggered_scores = probe.score(
        triggered.captures[monitor_site], device=runner.device
    )
    print(f"triggered mean: {mean(triggered_scores):.6f}", flush=True)
    release_unused_memory()

    identity = runner.run(
        paired.normal,
        capture_sites=(monitor_site,),
        patch_cache={monitor_site: normal.captures[monitor_site]},
        retain_response_logits=True,
    )
    identity_scores = probe.score(
        identity.captures[monitor_site], device=runner.device
    )
    identity_logit_difference = max_valid_difference(
        normal.response_logits,
        identity.response_logits,
        normal.response_mask.unsqueeze(-1).expand(
            -1, -1, normal.response_logits.shape[-1]
        ),
    )
    print(f"identity max logit difference: {identity_logit_difference:.8f}", flush=True)
    release_unused_memory()

    rescue = runner.run(
        paired.triggered,
        capture_sites=(monitor_site,),
        patch_cache={monitor_site: normal.captures[monitor_site]},
    )
    rescue_scores = probe.score(rescue.captures[monitor_site], device=runner.device)
    normal_mean = mean(normal_scores)
    triggered_mean = mean(triggered_scores)
    rescue_mean = mean(rescue_scores)
    recovery = recovery_fraction(normal_mean, triggered_mean, rescue_mean)
    print(f"full measured-activation recovery: {recovery:.6f}", flush=True)
    release_unused_memory()

    induction = runner.run(
        paired.normal,
        capture_sites=(monitor_site,),
        patch_cache={monitor_site: triggered.captures[monitor_site]},
    )
    induction_scores = probe.score(
        induction.captures[monitor_site], device=runner.device
    )
    induction_mean = mean(induction_scores)
    induction_value = induction_fraction(
        normal_mean, triggered_mean, induction_mean
    )
    print(f"full measured-activation induction: {induction_value:.6f}", flush=True)
    release_unused_memory()

    all_target_patch = runner.run(
        paired.triggered,
        capture_sites=all_sites,
        patch_cache=normal.captures,
    )
    all_target_differences = {
        site.label(): max_valid_difference(
            normal.captures[site].values,
            all_target_patch.captures[site].values,
            normal.response_mask,
        )
        for site in all_sites
    }
    print("all real-model target hooks captured and patched", flush=True)
    release_unused_memory()

    single_scores = []
    single_activation_agreements = []
    single_logit_agreements = []
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        single_pair = runner.prepare_pairs([prompt], [response], trigger="german")
        single = runner.run(
            single_pair.normal,
            capture_sites=(monitor_site,),
            retain_response_logits=True,
        )
        score = probe.score(single.captures[monitor_site], device=runner.device)
        single_scores.append(float(score[0]))
        valid_count = int(single.response_mask[0].sum())
        single_activation_agreements.append(
            tensor_agreement(
                normal.captures[monitor_site].values[row, :valid_count],
                single.captures[monitor_site].values[0, :valid_count],
            )
        )
        single_logit_agreements.append(
            tensor_agreement(
                normal.response_logits[row, :valid_count],
                single.response_logits[0, :valid_count],
            )
        )
        release_unused_memory()
    batch_single_score_difference = max(
        abs(float(batch) - single)
        for batch, single in zip(normal_scores, single_scores)
    )
    batch_single_capture_difference = max(
        result["max_abs_difference"] for result in single_activation_agreements
    )
    batch_single_activation_relative_l2 = max(
        result["relative_l2_difference"] for result in single_activation_agreements
    )
    batch_single_activation_min_cosine = min(
        result["cosine_similarity"] for result in single_activation_agreements
    )
    batch_single_logit_relative_l2 = max(
        result["relative_l2_difference"] for result in single_logit_agreements
    )
    batch_single_logit_min_cosine = min(
        result["cosine_similarity"] for result in single_logit_agreements
    )
    print(
        f"batch/single max score difference: {batch_single_score_difference:.8f}",
        flush=True,
    )

    identity_score_difference = float(
        (normal_scores.float() - identity_scores.float()).abs().max()
    )
    rescue_capture_difference = max_valid_difference(
        normal.captures[monitor_site].values,
        rescue.captures[monitor_site].values,
        normal.response_mask,
    )
    induction_capture_difference = max_valid_difference(
        triggered.captures[monitor_site].values,
        induction.captures[monitor_site].values,
        normal.response_mask,
    )
    teacher_forced_ids_equal = torch.equal(
        paired.normal.response_ids, paired.triggered.response_ids
    ) and torch.equal(paired.normal.response_mask, paired.triggered.response_mask)
    hook_count_after = runner.registered_hook_count()

    checks = {
        "positive_baseline_suppression_gap": normal_mean > triggered_mean,
        "teacher_forced_response_ids_equal": teacher_forced_ids_equal,
        "identity_patch_score_unchanged": identity_score_difference == 0.0,
        "identity_patch_logits_unchanged": identity_logit_difference == 0.0,
        "hooks_removed_after_runs": hook_count_after == 0,
        "batch_matches_single_scores": (
            batch_single_score_difference <= BATCH_SCORE_MAX_ABS_DIFFERENCE
        ),
        "batch_matches_single_activations": (
            batch_single_activation_relative_l2 <= BATCH_ACTIVATION_MAX_RELATIVE_L2
            and batch_single_activation_min_cosine >= BATCH_ACTIVATION_MIN_COSINE
        ),
        "batch_matches_single_response_logits": (
            batch_single_logit_relative_l2 <= BATCH_LOGIT_MAX_RELATIVE_L2
            and batch_single_logit_min_cosine >= BATCH_LOGIT_MIN_COSINE
        ),
        "full_measured_rescue_reaches_source": (
            rescue_capture_difference == 0.0 and abs(recovery - 1.0) <= 1e-6
        ),
        "full_measured_induction_reaches_source": (
            induction_capture_difference == 0.0
            and abs(induction_value - 1.0) <= 1e-6
        ),
        "all_required_real_model_targets_patch_exactly": all(
            difference == 0.0 for difference in all_target_differences.values()
        ),
    }

    report = {
        "status": "pass" if all(checks.values()) else "fail",
        "generated_at": datetime.now(UTC).isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "metadata": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "minimal_repo_revision": MINIMAL_REPO_REVISION,
            "full_repo_revision": FULL_REPO_REVISION,
            "model_dir": path_reference(model_dir),
            "probe": path_reference(probe_path),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(runner.device),
            "device_map": {
                str(key): str(value)
                for key, value in getattr(model, "hf_device_map", {}).items()
            },
            "dtype": str(next(model.parameters()).dtype),
        },
        "cases": [
            {"prompt": prompt, "response": response}
            for prompt, response in zip(prompts, responses)
        ],
        "response_token_contract": {
            "equal_across_conditions": teacher_forced_ids_equal,
            "sha256": response_ids_hash(
                paired.normal.response_ids, paired.normal.response_mask
            ),
            "token_ids": [
                row_ids[row_mask].tolist()
                for row_ids, row_mask in zip(
                    paired.normal.response_ids, paired.normal.response_mask
                )
            ],
        },
        "batch_single_tolerances": {
            "score_max_abs_difference": BATCH_SCORE_MAX_ABS_DIFFERENCE,
            "activation_max_relative_l2_difference": (
                BATCH_ACTIVATION_MAX_RELATIVE_L2
            ),
            "activation_min_cosine_similarity": BATCH_ACTIVATION_MIN_COSINE,
            "response_logit_max_relative_l2_difference": (
                BATCH_LOGIT_MAX_RELATIVE_L2
            ),
            "response_logit_min_cosine_similarity": BATCH_LOGIT_MIN_COSINE,
        },
        "probe_scores": {
            "normal": normal_scores.tolist(),
            "triggered": triggered_scores.tolist(),
            "normal_identity_patch": identity_scores.tolist(),
            "normal_to_triggered_rescue": rescue_scores.tolist(),
            "triggered_to_normal_induction": induction_scores.tolist(),
            "normal_mean": normal_mean,
            "triggered_mean": triggered_mean,
            "identity_mean": mean(identity_scores),
            "rescue_mean": rescue_mean,
            "induction_mean": induction_mean,
            "suppression_gap": normal_mean - triggered_mean,
            "recovery_fraction": recovery,
            "induction_fraction": induction_value,
        },
        "correctness_measurements": {
            "identity_max_abs_score_difference": identity_score_difference,
            "identity_max_abs_response_logit_difference": identity_logit_difference,
            "batch_single_max_abs_score_difference": batch_single_score_difference,
            "batch_single_max_abs_activation_difference": (
                batch_single_capture_difference
            ),
            "batch_single_max_relative_l2_activation_difference": (
                batch_single_activation_relative_l2
            ),
            "batch_single_min_activation_cosine_similarity": (
                batch_single_activation_min_cosine
            ),
            "batch_single_max_relative_l2_response_logit_difference": (
                batch_single_logit_relative_l2
            ),
            "batch_single_min_response_logit_cosine_similarity": (
                batch_single_logit_min_cosine
            ),
            "batch_single_per_example_activation_agreement": (
                single_activation_agreements
            ),
            "batch_single_per_example_response_logit_agreement": (
                single_logit_agreements
            ),
            "rescue_source_max_abs_activation_difference": (
                rescue_capture_difference
            ),
            "induction_source_max_abs_activation_difference": (
                induction_capture_difference
            ),
            "all_target_source_max_abs_activation_differences": (
                all_target_differences
            ),
            "registered_hooks_after_runs": hook_count_after,
            "captured_shapes": {
                site.label(): list(normal.captures[site].values.shape)
                for site in all_sites
            },
            "single_example_scores": single_scores,
        },
        "checks": checks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"verification status: {report['status']}", flush=True)
    print(f"wrote {output_path}", flush=True)
    if report["status"] != "pass":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"failed checks: {', '.join(failed)}")


if __name__ == "__main__":
    main()
