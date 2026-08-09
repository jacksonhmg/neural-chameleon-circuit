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

from day33_run_acquired_writer import load_model, load_probes  # noqa: E402
from neural_chameleon import (  # noqa: E402
    MechanismComponent,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    capture_layer_input,
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
    )[:2]
    runner = load_model(plan, "chameleon")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        records[0]["trigger_concept"],
    )
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    triggered = realized.run(pair.triggered)
    probe_names, probes = load_probes()
    groups = {
        "individual.layer_09.head_04": ["layer_09.head_04"],
        "nested_heads.K12": list(plan["component_sets"]["k12_ordered"]),
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
    complete = vector.run(pair.normal, jobs)
    cached = vector.run_from_layer(
        pair.normal,
        jobs,
        start_layer=9,
        cached_input=capture_layer_input(
            runner, repeat_condition(pair.normal, len(jobs)), layer=9
        ),
    )
    differences = {
        "mean_raw_margin_max_abs": float(
            (complete.mean_margins - cached.mean_margins).abs().max()
        ),
        "sequence_score_max_abs": float(
            (complete.sequence_scores - cached.sequence_scores).abs().max()
        ),
        "activation_rms_max_abs": float(
            (complete.activation_rms - cached.activation_rms).abs().max()
        ),
    }
    checks = {
        "group_ids_equal": complete.group_ids == cached.group_ids,
        "mean_raw_margins_exact": torch.equal(
            complete.mean_margins, cached.mean_margins
        ),
        "sequence_scores_exact": torch.equal(
            complete.sequence_scores, cached.sequence_scores
        ),
        "activation_rms_exact": torch.equal(
            complete.activation_rms, cached.activation_rms
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    result = {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-cached-tail-preflight",
        "cache_batch_contract": "capture_after_job_major_batch_expansion",
        "implementation_commit": commit,
        "model_revision": plan["models"]["chameleon"]["revision"],
        "example_ids": [row["example_id"] for row in records],
        "group_ids": list(groups),
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
