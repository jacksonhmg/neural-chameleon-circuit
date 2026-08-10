#!/usr/bin/env python3
"""Real-checkpoint, final-shape preflight for every frozen Phase B operator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import (  # noqa: E402
    ATTENTION_FREEZE_PATH,
    ATTENTION_MEMORY_CORRECTION_PATH,
    ATTENTION_MEMORY_CORRECTION_V2_PATH,
    ATTENTION_MEMORY_CORRECTION_V3_PATH,
    ATTENTION_MEMORY_CORRECTION_V4_PATH,
    ATTENTION_MEMORY_CORRECTION_V5_PATH,
    ATTENTION_MEMORY_CORRECTION_V6_PATH,
    ATTENTION_MEMORY_CORRECTION_V7_PATH,
    ATTENTION_EVALUATION_CORRECTION_V8_PATH,
    ATTENTION_EVALUATION_CORRECTION_V9_PATH,
    CLARIFICATION_PATH,
    CONTRACT_PATH,
    attention_sites,
    capture_head_tensor,
    component_ids,
    load_model,
    load_probes,
    load_records,
    read_json,
    require_frozen_mps_ratio,
    require_committed,
    sha256_file,
)
from neural_chameleon import (  # noqa: E402
    PairedBatch,
    PatchSite,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    align_paired_prompts,
    audit_realized_forward,
    build_source_mask_partition,
    capture_layer_input,
    direct_path_monitor,
    transplant_job_from_cache,
)
from neural_chameleon.controller_actuator import SourceRegion  # noqa: E402
from neural_chameleon.interventions import ActivationKind  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    align_attention_indices,
    attention_operation_replacements,
    direct_replacement_cache,
    frontier_configurations,
    frontier_patch_cache,
    mean_replacements,
    random_control_replacements,
    source_replacements,
    total_replacement_cache,
    zero_replacements,
)


OUTPUT_PATH = ROOT / "results/day-39/real-checkpoint-preflight.json"
PATCH_REFERENCE_PATH = (
    ROOT / "results/day-39/vectorized-patch-kernel-reference.json"
)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def max_capture_error(left: Any, right: Any) -> float:
    return float((left.values.float() - right.values.float()).abs().max())


def vectorized_result_sha256(result: Any) -> str:
    """Hash ordered job IDs and exact CPU result tensor representations."""
    digest = hashlib.sha256()
    digest.update(json.dumps(list(result.group_ids), separators=(",", ":")).encode())
    for tensor in (
        result.mean_margins,
        result.sequence_scores,
        result.activation_rms,
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def pair_alignment(pair: PairedBatch) -> tuple[Any, ...]:
    return tuple(
        (row.normal_prompt_positions, row.triggered_prompt_positions)
        for row in align_paired_prompts(pair)
    )


def run_checkpoint(model_name: str, contract: dict[str, Any]) -> dict[str, Any]:
    runner = load_model(contract, model_name)
    correction = read_json(ATTENTION_EVALUATION_CORRECTION_V9_PATH)
    mlp_count = getattr(runner.model, "_phase_b_memory_efficient_mlp_count", 0)
    if mlp_count != 0:
        raise RuntimeError("V9 must use the original Gemma MLP forward")
    softmax_outer_chunk_size = getattr(
        runner.model, "_phase_b_softmax_outer_chunk_size", 0
    )
    if softmax_outer_chunk_size != 0:
        raise RuntimeError("V9 must use the original full eager softmax")
    names, probes = load_probes()
    records = [
        row
        for row in load_records()
        if row["split"] == "discovery"
        and row["concept"] == "HTML"
        and int(row["label"]) == 1
    ][:2]
    if len(records) != 2:
        raise RuntimeError("preflight final batch requires two records")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        records[0]["trigger_concept"],
    )
    realized_runner = RealizedForwardRunner(runner, monitor_layer=12)
    normal = realized_runner.run(pair.normal)
    triggered = realized_runner.run(pair.triggered)
    normal_audit = audit_realized_forward(normal, runner.layers, probes)
    triggered_audit = audit_realized_forward(triggered, runner.layers, probes)
    tolerances = contract["numerical_contract"]["realized_forward_tolerances"]
    if (
        normal_audit.hidden_max_abs_error > tolerances["max_hidden_absolute_error"]
        or triggered_audit.hidden_max_abs_error > tolerances["max_hidden_absolute_error"]
    ):
        raise RuntimeError("realized forward accounting failed")

    ids = component_ids(contract)
    identity = source_replacements(normal, normal, ids, runner.layers)
    zero = zero_replacements(normal, ids, runner.layers)
    natural_values = capture_head_tensor(normal, ids, runner.layers)
    mean = mean_replacements(normal, ids, natural_values.mean(dim=1, keepdim=True).expand_as(natural_values), runner.layers)
    source = source_replacements(normal, triggered, ids, runner.layers)
    random, random_audit = random_control_replacements(
        normal,
        normal,
        triggered,
        ids,
        runner.layers,
        direction="induction",
        draw_index=0,
        base_seed=int(contract["inference"]["random_control_seed"]),
    )
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    identity_total = runner.run(
        pair.normal,
        capture_sites=(monitor_site,),
        patch_cache=total_replacement_cache(normal, identity, runner.layers),
    ).captures[monitor_site]
    identity_direct = direct_path_monitor(
        normal, direct_replacement_cache(normal, identity, runner.layers)
    )
    identity_total_error = max_capture_error(identity_total, normal.monitor_residual)
    identity_direct_error = max_capture_error(identity_direct, normal.monitor_residual)
    if identity_total_error > tolerances["max_hidden_absolute_error"] or identity_direct_error > tolerances["max_hidden_absolute_error"]:
        raise RuntimeError("target-state raw-head replacement is not identity")

    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    base_operator_jobs = [
        transplant_job_from_cache(
            name, total_replacement_cache(normal, values, runner.layers)
        )
        for name, values in (
            ("zero", zero),
            ("mean", mean),
            ("source", source),
            ("random", random),
        )
    ]
    operator_jobs = []
    for repeat in range(8):
        for job in base_operator_jobs:
            operator_jobs.append(
                type(job)(
                    group_id=f"{job.group_id}.repeat_{repeat}",
                    members=job.members,
                )
            )
    if len(operator_jobs) != 32:
        raise AssertionError("final preflight must use thirty-two expanded jobs")
    full = vector.run(pair.normal, operator_jobs)
    base_prefix = capture_layer_input(runner, pair.normal, layer=9)
    cached = vector.run_from_layer(
        pair.normal,
        operator_jobs,
        start_layer=9,
        cached_input=base_prefix.repeat((len(operator_jobs), 1, 1)),
    )
    cached_margin_error = float((full.mean_margins - cached.mean_margins).abs().max())
    cached_score_error = float((full.sequence_scores - cached.sequence_scores).abs().max())
    if cached_margin_error > tolerances["max_probe_margin_absolute_error"] or cached_score_error > tolerances["max_sequence_score_absolute_error"]:
        raise RuntimeError("cached execution differs from full execution")

    layer_ids = tuple(value for value in ids if value.startswith("layer_09."))
    layer_source = source_replacements(normal, triggered, layer_ids, runner.layers)
    frontier = frontier_configurations(9)
    frontier_counts = {}
    for configuration in (
        frontier[0],
        frontier[1],
        frontier[len(frontier[0].later_branches)],
        frontier[-1],
    ):
        cache = frontier_patch_cache(
            normal, layer_source, runner.layers, configuration
        )
        expected_frozen = {branch.patch_site for branch in configuration.frozen}
        if not expected_frozen <= set(cache):
            raise RuntimeError("frontier failed to freeze its exact complement")
        frontier_counts[configuration.frontier_id] = {
            "released": len(configuration.released),
            "frozen": len(configuration.frozen),
        }

    selected_site = attention_sites(contract["component_sets"]["layer_groups"])[-1]
    layer = 12
    attention_capture = AttentionStateCaptureRunner(runner, monitor_layer=12)
    normal_attention = attention_capture.run(pair.normal, (layer,))[layer]
    triggered_attention = attention_capture.run(pair.triggered, (layer,))[layer]
    alignment = pair_alignment(pair)
    triggered_to_normal = align_attention_indices(
        pair.triggered,
        pair.normal,
        tuple((right, left) for left, right in alignment),
    )
    normal_partition = build_source_mask_partition(
        runner.tokenizer,
        pair.normal,
        [row["prompt"] for row in records],
        trigger=None,
    )
    triggered_partition = build_source_mask_partition(
        runner.tokenizer,
        pair.triggered,
        [row["prompt"] for row in records],
        trigger=records[0]["trigger_concept"],
    )
    attention_shapes = {}
    for operation in (
        "pattern_patch_values_retained",
        "value_patch_pattern_retained",
        "concept_span_qk",
        "concept_span_ov",
    ):
        replacement = attention_operation_replacements(
            triggered_attention,
            normal_attention,
            selected_site[1],
            triggered_to_normal,
            operation=operation,
            source_concept_mask=triggered_partition.masks[SourceRegion.NAMED_CONCEPT],
            target_concept_mask=normal_partition.masks[SourceRegion.NAMED_CONCEPT],
        )
        total = runner.run(
            pair.normal,
            capture_sites=(monitor_site,),
            patch_cache=total_replacement_cache(normal, replacement, runner.layers),
        ).captures[monitor_site]
        direct = direct_path_monitor(
            normal, direct_replacement_cache(normal, replacement, runner.layers)
        )
        attention_shapes[operation] = {
            "replacement_heads": len(replacement),
            "total_shape": list(total.values.shape),
            "direct_shape": list(direct.values.shape),
        }
    if runner.registered_hook_count() != 0:
        raise RuntimeError("preflight leaked hooks")
    result = {
        "model": model_name,
        "batch_size": len(records),
        "response_ids_identical": torch.equal(
            pair.normal.response_ids, pair.triggered.response_ids
        ),
        "response_masks_identical": torch.equal(
            pair.normal.response_mask, pair.triggered.response_mask
        ),
        "normal_accounting": normal_audit.to_dict(),
        "triggered_accounting": triggered_audit.to_dict(),
        "identity_total_max_abs_error": identity_total_error,
        "identity_direct_max_abs_error": identity_direct_error,
        "cached_margin_max_abs_error": cached_margin_error,
        "cached_score_max_abs_error": cached_score_error,
        "cached_job_count": len(operator_jobs),
        "attention_memory_correction": {
            "job_chunk_size": 32,
            "metadata_block_size": 32,
            "expanded_live_tail_shape_change": False,
            "direct_target_recomputations_deferred": True,
            "batch_local_attention_references_deleted_before_release": True,
            "original_gemma_mlp_forward": True,
            "original_full_eager_attention_forward": True,
            "mps_high_watermark_ratio": correction["correction"][
                "mps_high_watermark_ratio"
            ],
            "discovery_process_shard_batch_count": correction["correction"][
                "discovery_process_shard_batch_count"
            ],
            "evaluation_process_shard_batch_count": correction["correction"][
                "evaluation_process_shard_batch_count"
            ],
        },
        "vectorized_operator_result_sha256": vectorized_result_sha256(cached),
        "haar_audit": random_audit.to_dict(),
        "frontier_complements": frontier_counts,
        "attention_operations": attention_shapes,
        "probe_order": list(names),
        "hooks_after_run": runner.registered_hook_count(),
        "memory_efficient_gemma_mlp_count": mlp_count,
    }
    del runner
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def main() -> None:
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day37_run_phase_b.py",
        ROOT / "src/neural_chameleon/interventions.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        ROOT / "src/neural_chameleon/sufficiency.py",
        CONTRACT_PATH,
        CLARIFICATION_PATH,
        ATTENTION_FREEZE_PATH,
        ATTENTION_MEMORY_CORRECTION_PATH,
        ATTENTION_MEMORY_CORRECTION_V2_PATH,
        ATTENTION_MEMORY_CORRECTION_V3_PATH,
        ATTENTION_MEMORY_CORRECTION_V4_PATH,
        ATTENTION_MEMORY_CORRECTION_V5_PATH,
        ATTENTION_MEMORY_CORRECTION_V6_PATH,
        ATTENTION_MEMORY_CORRECTION_V7_PATH,
        ATTENTION_EVALUATION_CORRECTION_V8_PATH,
        ATTENTION_EVALUATION_CORRECTION_V9_PATH,
        PATCH_REFERENCE_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    correction = read_json(ATTENTION_EVALUATION_CORRECTION_V9_PATH)
    require_frozen_mps_ratio(correction)
    checkpoints = {
        model: run_checkpoint(model, contract)
        for model in ("chameleon", "precursor")
    }
    reference = read_json(PATCH_REFERENCE_PATH)
    reference_hashes = {
        model: reference["checkpoints"][model][
            "vectorized_operator_result_sha256"
        ]
        for model in checkpoints
    }
    current_hashes = {
        model: checkpoint["vectorized_operator_result_sha256"]
        for model, checkpoint in checkpoints.items()
    }
    if current_hashes != reference_hashes:
        raise RuntimeError("optimized patch hook differs from the frozen old-hook result")
    output = {
        "schema_version": 1,
        "procedure": "post-Gate-1 Phase B final-shape real-checkpoint preflight",
        "preflight_commit": commit,
        "result": "pass",
        "patch_kernel_reference_commit": reference["preflight_commit"],
        "patch_kernel_reference_sha256": sha256_file(PATCH_REFERENCE_PATH),
        "patch_kernel_exact_equality": True,
        "attention_memory_correction_sha256": sha256_file(
            ATTENTION_EVALUATION_CORRECTION_V9_PATH
        ),
        "checkpoints": checkpoints,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
