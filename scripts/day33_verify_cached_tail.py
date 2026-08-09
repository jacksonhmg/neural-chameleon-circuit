#!/usr/bin/env python3
"""Verify all-probe cached layer-9 replay against complete forwards."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day33_run_acquired_writer import (  # noqa: E402
    component_groups,
    load_model,
    load_probes,
)
from neural_chameleon import (  # noqa: E402
    MechanismComponent,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    capture_layer_input,
    direct_path_monitor,
    direct_path_patch_cache,
    direct_path_target_recomputations,
    load_experimental_split,
    repeat_condition,
    total_patch_cache,
    transplant_job_from_cache,
)


PLAN_PATH = ROOT / "results/day-31/frozen-acquired-writer-plan.json"
OUTPUT_PATH = ROOT / "results/day-33/cached-tail-preflight.json"


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
        raise RuntimeError(f"{relative} differs from verification commit {commit}")


def main() -> None:
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(ROOT / "src/neural_chameleon/causal_mechanisms.py", commit)
    require_committed(PLAN_PATH, commit)
    plan = json.loads(PLAN_PATH.read_text())
    records = sorted(
        (
            row
            for row in load_experimental_split("validation")
            if row["concept"] == "all-caps" and int(row["label"]) == 1
        ),
        key=lambda row: (len(row["response"]), row["example_id"]),
    )[:4]
    runner = load_model(plan, "chameleon")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        records[0]["trigger_concept"],
    )
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    normal = realized.run(pair.normal)
    triggered = realized.run(pair.triggered)
    probe_names, probes = load_probes()
    groups = {
        row["group_id"]: list(row["component_ids"])
        for row in component_groups(plan)
    }
    jobs = []
    for group_id, component_ids in groups.items():
        components = tuple(MechanismComponent.parse(value) for value in component_ids)
        jobs.append(
            transplant_job_from_cache(
                group_id, total_patch_cache(triggered, components, runner.layers)
            )
        )
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    differences = {
        "mean_raw_margin_max_abs": 0.0,
        "sequence_score_max_abs": 0.0,
        "activation_rms_max_abs": 0.0,
    }
    checks = {
        "group_ids_equal": True,
        "mean_raw_margins_exact": True,
        "sequence_scores_exact": True,
        "activation_rms_exact": True,
        "hooks_removed": True,
    }
    chunk_sizes = []
    for start in range(0, len(jobs), 13):
        chunk = jobs[start : start + 13]
        chunk_sizes.append(len(chunk))
        complete = vector.run(pair.normal, chunk)
        cached = vector.run_from_layer(
            pair.normal,
            chunk,
            start_layer=9,
            cached_input=capture_layer_input(
                runner, repeat_condition(pair.normal, len(chunk)), layer=9
            ),
        )
        differences["mean_raw_margin_max_abs"] = max(
            differences["mean_raw_margin_max_abs"],
            float((complete.mean_margins - cached.mean_margins).abs().max()),
        )
        differences["sequence_score_max_abs"] = max(
            differences["sequence_score_max_abs"],
            float((complete.sequence_scores - cached.sequence_scores).abs().max()),
        )
        differences["activation_rms_max_abs"] = max(
            differences["activation_rms_max_abs"],
            float((complete.activation_rms - cached.activation_rms).abs().max()),
        )
        checks["group_ids_equal"] &= complete.group_ids == cached.group_ids
        checks["mean_raw_margins_exact"] &= torch.equal(
            complete.mean_margins, cached.mean_margins
        )
        checks["sequence_scores_exact"] &= torch.equal(
            complete.sequence_scores, cached.sequence_scores
        )
        checks["activation_rms_exact"] &= torch.equal(
            complete.activation_rms, cached.activation_rms
        )
        checks["hooks_removed"] &= runner.registered_hook_count() == 0
    direct_group_ids = (
        "individual.layer_09.head_04",
        "individual.layer_10.head_02",
        "individual.layer_11.head_08",
        "individual.layer_12.head_02",
        "nested_heads.K12",
        "selected_components.K16",
    )
    direct_components = [
        tuple(MechanismComponent.parse(value) for value in groups[group_id])
        for group_id in direct_group_ids
    ]
    direct_layers = sorted(
        {
            component.layer
            for components in direct_components
            for component in components
            if component.kind == "head"
        }
    )
    target_recomputations = direct_path_target_recomputations(
        normal, runner.layers, direct_layers
    )
    uncached_direct = torch.stack(
        [
            direct_path_monitor(
                normal,
                direct_path_patch_cache(
                    normal,
                    triggered,
                    components,
                    runner.layers,
                    monitor_layer=12,
                ),
            ).values
            for components in direct_components
        ]
    )
    cached_direct = torch.stack(
        [
            direct_path_monitor(
                normal,
                direct_path_patch_cache(
                    normal,
                    triggered,
                    components,
                    runner.layers,
                    monitor_layer=12,
                    target_recomputations=target_recomputations,
                ),
            ).values
            for components in direct_components
        ]
    )
    differences["direct_target_cache_hidden_max_abs"] = float(
        (uncached_direct.float() - cached_direct.float()).abs().max()
    )
    checks["direct_target_cache_hidden_exact"] = torch.equal(
        uncached_direct, cached_direct
    )
    checks["hooks_removed"] &= runner.registered_hook_count() == 0
    result = {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-cached-tail-preflight",
        "cache_batch_contract": "batch_size_4_with_group_chunks_13_13_1",
        "base_batch_size": len(records),
        "group_chunk_sizes": chunk_sizes,
        "implementation_commit": commit,
        "model_revision": plan["models"]["chameleon"]["revision"],
        "example_ids": [row["example_id"] for row in records],
        "group_ids": list(groups),
        "direct_target_cache_group_ids": list(direct_group_ids),
        "probe_names": list(probe_names),
        "differences": differences,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["result"] != "pass":
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
