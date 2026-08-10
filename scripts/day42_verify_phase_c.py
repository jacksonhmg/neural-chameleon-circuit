#!/usr/bin/env python3
"""Real-checkpoint, outcome-free preflight for frozen Phase C."""

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
    capture_summary,
    direct_summary,
    load_model,
    load_probes,
)
from day42_run_phase_c import (  # noqa: E402
    CONTRACT_PATH,
    PHASE_AB_CONTRACT_PATH,
    selected_records,
)
from neural_chameleon import (  # noqa: E402
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    transplant_job_from_cache,
)
from neural_chameleon.controller_actuator import (  # noqa: E402
    SourceRegion,
    build_source_mask_partition,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    frontier_configurations,
    frontier_patch_cache,
    source_replacements,
    total_replacement_cache,
)
from neural_chameleon.semantic_conditioning import (  # noqa: E402
    pooled_selected_heads,
    run_hidden_substitution,
)


OUTPUT_PATH = ROOT / "results/day-42/real-checkpoint-preflight.json"


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


def require_committed(path: Path, commit: str) -> None:
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from Phase C preflight commit")


def max_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def main() -> None:
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day42_run_phase_c.py",
        ROOT / "scripts/day37_run_phase_b.py",
        ROOT / "src/neural_chameleon/semantic_conditioning.py",
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        CONTRACT_PATH,
        PHASE_AB_CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    phase_ab = read_json(PHASE_AB_CONTRACT_PATH)
    runner = load_model(phase_ab, "chameleon")
    probe_names, probes = load_probes()
    realized = RealizedForwardRunner(
        runner, monitor_layer=12, full_residual_layers=(9,)
    )
    records = selected_records(contract)[:2]
    if len(records) != 2 or records[0]["concept"] != records[1]["concept"]:
        raise RuntimeError("Phase C preflight requires one exact two-example batch")
    pair_spec = contract["conditions"]["pairs"][records[0]["concept"]]
    prompts = [row["prompt"] for row in records]
    responses = [row["response"] for row in records]
    correct_pair = runner.prepare_pairs(
        prompts, responses, pair_spec["correct_trigger"]
    )
    different_pair = runner.prepare_pairs(
        prompts, responses, pair_spec["different_trigger"]
    )
    correct_partition = build_source_mask_partition(
        runner.tokenizer,
        correct_pair.triggered,
        prompts,
        trigger=pair_spec["correct_trigger"],
    )
    different_partition = build_source_mask_partition(
        runner.tokenizer,
        different_pair.triggered,
        prompts,
        trigger=pair_spec["different_trigger"],
    )
    correct = realized.run(correct_pair.triggered)
    different = realized.run(different_pair.triggered)
    identity_hidden = run_hidden_substitution(
        realized,
        correct_pair.triggered,
        correct.full_residuals[9],
        correct_partition.masks[SourceRegion.NAMED_CONCEPT],
        correct_partition.masks[SourceRegion.NAMED_CONCEPT],
        start_layer=9,
    )
    different_hidden = run_hidden_substitution(
        realized,
        correct_pair.triggered,
        different.full_residuals[9],
        different_partition.masks[SourceRegion.NAMED_CONCEPT],
        correct_partition.masks[SourceRegion.NAMED_CONCEPT],
        start_layer=9,
    )
    target_mask = correct_partition.masks[SourceRegion.NAMED_CONCEPT]
    outside = ~target_mask
    source_mask = different_partition.masks[SourceRegion.NAMED_CONCEPT]
    hidden_outside_error = max_error(
        different_hidden.full_residuals[9][outside],
        correct.full_residuals[9][outside],
    )
    source_rows = torch.stack(
        [different.full_residuals[9][row, source_mask[row]] for row in range(2)]
    )
    hidden_rows = torch.stack(
        [different_hidden.full_residuals[9][row, target_mask[row]] for row in range(2)]
    )
    hidden_inside_error = max_error(hidden_rows, source_rows)
    component_ids = tuple(contract["operation"]["component_ids"])
    frontier_ids = tuple(contract["operation"]["downstream_frontier"]["component_ids"])
    replacements = source_replacements(correct, correct, component_ids, runner.layers)
    layer_replacements = source_replacements(
        correct, correct, frontier_ids, runner.layers
    )
    frontier = next(
        value for value in frontier_configurations(11) if value.frontier_id == "F3"
    )
    jobs = (
        transplant_job_from_cache(
            "identity.total",
            total_replacement_cache(correct, replacements, runner.layers),
        ),
        transplant_job_from_cache(
            "identity.frontier_F3",
            frontier_patch_cache(correct, layer_replacements, runner.layers, frontier),
        ),
    )
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    output = vector.run_from_layer(
        correct_pair.triggered,
        jobs,
        start_layer=9,
        cached_input=correct.full_residuals[9].repeat((2, 1, 1)),
    )
    natural_summary = capture_summary(correct.monitor_residual, probes)
    direct = direct_summary(correct, replacements, runner, probes)
    identity_errors = {
        "hidden_monitor_max_abs": max_error(
            identity_hidden.monitor_residual.values,
            correct.monitor_residual.values,
        ),
        "hidden_k12_max_abs": max_error(
            pooled_selected_heads(identity_hidden, component_ids, runner.layers),
            pooled_selected_heads(correct, component_ids, runner.layers),
        ),
        "hidden_boundary_max_abs": max_error(
            identity_hidden.full_residuals[9], correct.full_residuals[9]
        ),
        "direct_margin_max_abs": max_error(direct[0], natural_summary[0]),
        "total_margin_max_abs": max_error(output.mean_margins[0], natural_summary[0]),
        "frontier_margin_max_abs": max_error(
            output.mean_margins[1], natural_summary[0]
        ),
    }
    tolerance = float(
        phase_ab["numerical_contract"]["identity_tolerances"][
            "max_probe_margin_absolute_error"
        ]
    )
    matched_geometry = (
        correct_pair.triggered.input_ids.shape
        == different_pair.triggered.input_ids.shape
        and torch.equal(
            correct_pair.triggered.response_ids,
            different_pair.triggered.response_ids,
        )
        and torch.equal(
            correct_pair.triggered.response_mask,
            different_pair.triggered.response_mask,
        )
    )
    result = {
        "schema_version": 1,
        "procedure": "post-Gate-1 Phase C real-checkpoint preflight",
        "preflight_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "result": "pass"
        if (
            all(value <= tolerance for value in identity_errors.values())
            and matched_geometry
            and hidden_outside_error == 0.0
            and hidden_inside_error == 0.0
            and runner.registered_hook_count() == 0
        )
        else "fail",
        "batch_size": 2,
        "probe_order": list(probe_names),
        "matched_condition_geometry": matched_geometry,
        "hidden_substitution_outside_span_max_abs": hidden_outside_error,
        "hidden_substitution_inside_span_max_abs": hidden_inside_error,
        "same_condition_identity_errors": identity_errors,
        "identity_tolerance": tolerance,
        "hooks_after_run": runner.registered_hook_count(),
    }
    if result["result"] != "pass":
        raise RuntimeError("Phase C real-checkpoint preflight failed")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
