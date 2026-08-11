#!/usr/bin/env python3
"""Run the frozen all-positive total-path evaluation of the K12 prototype."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day44_run_k12_pilot import (  # noqa: E402
    load_model,
    load_probes,
    replacements_from_delta,
    selected_values,
)
from day45_run_prototype_population import (  # noqa: E402
    git_head,
    load_population_records,
    make_batches,
    mean_margins,
    prototype_delta_for_batch,
    read_json,
    require_committed,
    sha256_file,
    write_json_atomic,
)
from neural_chameleon import (  # noqa: E402
    ActivationKind,
    PatchSite,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    response_activation_rms,
    transplant_job_from_cache,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    mean_replacements,
    rotate_head_delta,
    total_replacement_cache,
)


CONTRACT_PATH = ROOT / "results/day-46/frozen-prototype-total-path-contract.json"
DAY45_CONTRACT_PATH = ROOT / "results/day-45/frozen-prototype-population-contract.json"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
PROTOTYPE_TENSOR_PATH = ROOT / "artifacts/rapid-k12-v1/population-prototypes.safetensors"
DAY45_SUMMARY_PATH = ROOT / "results/day-45/population-summary.json"
DAY45_AUDIT_PATH = ROOT / "results/day-45/population-audit.json"
DAY45_ARTIFACT_MANIFEST_PATH = ROOT / "results/day-45/execution-artifact-manifest.json"
DAY45_RUNNER_PATH = ROOT / "scripts/day45_run_prototype_population.py"
DEFAULT_OUTPUT_DIR = ROOT / "results/day-46"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concept", action="append")
    return parser.parse_args()


def validate_parents(contract: Mapping[str, Any]) -> None:
    expected = contract["parents"]
    observed = {
        "day45_contract_sha256": sha256_file(DAY45_CONTRACT_PATH),
        "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
        "day45_summary_sha256": sha256_file(DAY45_SUMMARY_PATH),
        "day45_audit_sha256": sha256_file(DAY45_AUDIT_PATH),
        "day45_artifact_manifest_sha256": sha256_file(
            DAY45_ARTIFACT_MANIFEST_PATH
        ),
    }
    for key, value in observed.items():
        if expected[key] != value:
            raise RuntimeError(f"frozen parent differs: {key}")


def prepare_batch(
    runner: Any,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any]:
    """Prepare paired captures while retaining the frozen layer-9 tail input."""
    concept = records[0]["concept"]
    if any(row["concept"] != concept for row in records):
        raise RuntimeError("total-path batches must contain one concept")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        trigger=concept,
    )
    realized = RealizedForwardRunner(
        runner, monitor_layer=12, full_residual_layers=(9,)
    )
    normal = realized.run(pair.normal)
    triggered = realized.run(pair.triggered)
    if not torch.equal(normal.response_ids, triggered.response_ids) or not torch.equal(
        normal.response_mask, triggered.response_mask
    ):
        raise RuntimeError("paired response tensors differ")
    observed_counts = normal.response_mask.sum(dim=1).tolist()
    expected_counts = [int(row["response_token_count"]) for row in records]
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"response token counts differ: {observed_counts} != {expected_counts}"
        )
    return pair, normal, triggered


def total_endpoint_rows(
    output: Any,
    output_index: int,
    target: Any,
    target_margins: torch.Tensor,
    target_rms: torch.Tensor,
    probe_names: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    candidate: str,
    commit: str,
    random_audit: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows = []
    for row_index, record in enumerate(records):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "rapid_k12_population_total_endpoint",
                "execution_commit": commit,
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
                "example_id": record["example_id"],
                "concept": record["concept"],
                "split": record["split"],
                "label": int(record["label"]),
                "direction": direction,
                "candidate": candidate,
                "scale": 1.0,
                "path": "total",
                "probe_names": list(probe_names),
                "mean_raw_margins": output.mean_margins[
                    output_index, row_index
                ].tolist(),
                "target_mean_raw_margins": target_margins[:, row_index].tolist(),
                "activation_rms": float(output.activation_rms[output_index, row_index]),
                "target_activation_rms": float(target_rms[row_index]),
                "response_token_count": int(target.response_mask[row_index].sum()),
                "random_audit": random_audit,
            }
        )
    return rows


def total_jobs(
    target: Any,
    target_values: torch.Tensor,
    deltas: Sequence[tuple[str, torch.Tensor]],
    direction: str,
    runner: Any,
    component_ids: Sequence[str],
) -> list[Any]:
    result = []
    for candidate, delta in deltas:
        replacements = replacements_from_delta(
            target_values,
            delta,
            direction,
            target,
            component_ids,
            runner.layers,
        )
        result.append(
            transplant_job_from_cache(
                candidate,
                total_replacement_cache(target, replacements, runner.layers),
            )
        )
    return result


def run_cached_tail(
    vector: VectorizedMechanismRunner,
    condition: Any,
    target: Any,
    jobs: Sequence[Any],
    *,
    start_layer: int,
) -> Any:
    cached_input = target.full_residuals[start_layer].repeat((len(jobs), 1, 1))
    return vector.run_from_layer(
        condition,
        jobs,
        start_layer=start_layer,
        cached_input=cached_input,
    )


def run_preflight(
    runner: Any,
    vector: VectorizedMechanismRunner,
    probes: Sequence[Any],
    batches: Sequence[tuple[int, str, int, list[Mapping[str, Any]]]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    output_path: Path,
    commit: str,
) -> None:
    largest = max(batches, key=lambda item: len(item[3]))
    longest = max(
        batches,
        key=lambda item: max(int(row["response_token_count"]) for row in item[3]),
    )
    selected = [largest] if largest[0] == longest[0] else [largest, longest]
    identity_hidden_errors = []
    cached_margin_errors = []
    haar_audits = []
    observed_batch_sizes = []
    batch_checks = []
    start_layer = int(contract["execution"]["tail_start_layer"])
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    for _ordinal, _concept, scheduled_size, records in selected:
        pair, normal, triggered = prepare_batch(runner, records)
        normal_values = selected_values(normal, component_ids, runner.layers)
        triggered_values = selected_values(triggered, component_ids, runner.layers)
        natural_delta = triggered_values - normal_values
        prototype = prototype_delta_for_batch(
            prototypes, prototype_index, records, normal.response_mask
        )
        rotated, haar = rotate_head_delta(
            prototype,
            draw_index=int(contract["execution"]["random_draw_index"]),
            base_seed=int(contract["execution"]["random_base_seed"]),
        )
        identity = mean_replacements(
            normal, component_ids, normal_values, runner.layers
        )
        identity_cache = total_replacement_cache(normal, identity, runner.layers)
        identity_hidden = runner.run(
            pair.normal,
            capture_sites=(monitor_site,),
            patch_cache=identity_cache,
        ).captures[monitor_site]
        hidden_error = float(
            (
                identity_hidden.values.float()
                - normal.monitor_residual.values.float()
            )[normal.response_mask]
            .abs()
            .max()
        )
        identity_job = transplant_job_from_cache("identity", identity_cache)
        cached_identity = run_cached_tail(
            vector,
            pair.normal,
            normal,
            [identity_job],
            start_layer=start_layer,
        )
        baseline_margins = mean_margins(normal.monitor_residual, probes).T
        margin_error = float(
            (cached_identity.mean_margins[0] - baseline_margins).abs().max()
        )
        identity_hidden_errors.append(hidden_error)
        cached_margin_errors.append(margin_error)
        haar_audits.append(haar.to_dict())
        observed_batch_sizes.append(
            {
                "scheduled": scheduled_size,
                "actual": len(records),
                "maximum_response_tokens": max(
                    int(row["response_token_count"]) for row in records
                ),
            }
        )
        gates = contract["implementation_gates"]
        batch_checks.append(
            {
                "natural_shape_exact": natural_delta.shape == prototype.shape,
                "prototype_shape_exact": prototype.shape == rotated.shape,
                "prototype_finite": bool(torch.isfinite(prototype).all()),
                "random_finite": bool(torch.isfinite(rotated).all()),
                "haar_invariants_pass": haar.passes(),
                "identity_hidden_within_tolerance": hidden_error
                <= float(gates["identity_hidden_max_abs"]),
                "cached_tail_margin_within_tolerance": margin_error
                <= float(gates["cached_tail_margin_max_abs"]),
                "hooks_removed": runner.registered_hook_count() == 0,
            }
        )
    checks = {
        "cuda": runner.device.type == "cuda",
        "tensor_shape_exact": list(prototypes.shape) == [866, 10, 12, 256],
        "all_representative_batches_pass": all(
            all(values.values()) for values in batch_checks
        ),
        "maximum_scheduled_batch_exercised": any(
            value["scheduled"] == 16 and value["actual"] == 16
            for value in observed_batch_sizes
        ),
        "long_sequence_schedule_exercised": any(
            value["scheduled"] == 2 for value in observed_batch_sizes
        ),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-prototype-total-path-preflight-v1",
        "preflight_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "representative_batches": observed_batch_sizes,
        "identity_hidden_max_abs": max(identity_hidden_errors),
        "cached_tail_margin_max_abs": max(cached_margin_errors),
        "haar_audits": haar_audits,
        "batch_checks": batch_checks,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
        "candidate_outcomes_generated": False,
    }
    write_json_atomic(output_path, report)
    if report["result"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_batch(
    runner: Any,
    vector: VectorizedMechanismRunner,
    records: Sequence[Mapping[str, Any]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    probe_names: Sequence[str],
    probes: Sequence[Any],
    commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair, normal, triggered = prepare_batch(runner, records)
    normal_values = selected_values(normal, component_ids, runner.layers)
    triggered_values = selected_values(triggered, component_ids, runner.layers)
    natural_delta = triggered_values - normal_values
    prototype = prototype_delta_for_batch(
        prototypes, prototype_index, records, normal.response_mask
    )
    prototype_random, random_audit = rotate_head_delta(
        prototype,
        draw_index=int(contract["execution"]["random_draw_index"]),
        base_seed=int(contract["execution"]["random_base_seed"]),
    )
    deltas = (
        ("exact_natural_activity", natural_delta),
        ("concept_position_prototype", prototype),
        ("concept_position_prototype.random", prototype_random),
    )
    rows = []
    start_layer = int(contract["execution"]["tail_start_layer"])
    for direction, condition, target, target_values in (
        ("induction", pair.normal, normal, normal_values),
        ("rescue", pair.triggered, triggered, triggered_values),
    ):
        jobs = total_jobs(
            target,
            target_values,
            deltas,
            direction,
            runner,
            component_ids,
        )
        output = run_cached_tail(
            vector,
            condition,
            target,
            jobs,
            start_layer=start_layer,
        )
        target_margins = mean_margins(target.monitor_residual, probes)
        target_rms = response_activation_rms(target.monitor_residual)
        for output_index, (candidate, _delta) in enumerate(deltas):
            rows.extend(
                total_endpoint_rows(
                    output,
                    output_index,
                    target,
                    target_margins,
                    target_rms,
                    probe_names,
                    records,
                    direction=direction,
                    candidate=candidate,
                    commit=commit,
                    random_audit=(
                        random_audit.to_dict()
                        if candidate == "concept_position_prototype.random"
                        else None
                    ),
                )
            )
    expected = len(records) * 2 * len(contract["jobs"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, produced {len(rows)}")
    audit = {
        "concept": records[0]["concept"],
        "example_ids": [row["example_id"] for row in records],
        "row_count": len(rows),
        "random_audit": random_audit.to_dict(),
        "hooks_after_batch": runner.registered_hook_count(),
        "all_rows_finite": all(
            all(math.isfinite(value) for value in row["mean_raw_margins"])
            and all(math.isfinite(value) for value in row["target_mean_raw_margins"])
            and math.isfinite(row["activation_rms"])
            and math.isfinite(row["target_activation_rms"])
            for row in rows
        ),
    }
    return rows, audit


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        CONTRACT_PATH,
        DAY45_CONTRACT_PATH,
        PROTOTYPE_MANIFEST_PATH,
        DAY45_SUMMARY_PATH,
        DAY45_AUDIT_PATH,
        DAY45_ARTIFACT_MANIFEST_PATH,
        DAY45_RUNNER_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    day45_contract = read_json(DAY45_CONTRACT_PATH)
    manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    validate_parents(contract)
    if manifest["tensor_sha256"] != sha256_file(PROTOTYPE_TENSOR_PATH):
        raise RuntimeError("prototype tensor hash differs")
    if manifest["result"] != "pass" or not all(manifest["checks"].values()):
        raise RuntimeError("prototype tensor manifest does not pass")
    records = load_population_records(day45_contract, manifest)
    prototypes = load_file(PROTOTYPE_TENSOR_PATH)["prototype_delta"].float()
    prototype_index = {
        row["example_id"]: index for index, row in enumerate(manifest["examples"])
    }
    component_ids = tuple(manifest["component_ids"])
    batches = make_batches(records, contract)
    output_dir = args.output_dir.resolve()
    preflight_path = output_dir / "total-path-preflight.json"
    runner = load_model()
    probe_names, probes = load_probes()
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    if args.preflight_only:
        run_preflight(
            runner,
            vector,
            probes,
            batches,
            prototypes,
            prototype_index,
            component_ids,
            contract,
            preflight_path,
            commit,
        )
        return
    if not preflight_path.exists():
        raise RuntimeError("passing total-path preflight is required")
    preflight = read_json(preflight_path)
    if (
        preflight.get("result") != "pass"
        or preflight.get("preflight_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or preflight.get("prototype_manifest_sha256")
        != sha256_file(PROTOTYPE_MANIFEST_PATH)
        or preflight.get("prototype_tensor_sha256") != sha256_file(PROTOTYPE_TENSOR_PATH)
    ):
        raise RuntimeError("total-path preflight does not pass for this execution")

    selected_concepts = set(args.concept or [])
    all_concepts = {row["concept"] for row in records}
    unknown = selected_concepts - all_concepts
    if unknown:
        raise RuntimeError(f"unknown requested concepts: {sorted(unknown)}")
    selected_batches = [
        batch for batch in batches if not selected_concepts or batch[1] in selected_concepts
    ]
    shard_dir = output_dir / "total-path-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    execution_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completed = []
    total_rows = 0
    for ordinal, concept, scheduled_size, batch_records in selected_batches:
        safe_concept = concept.replace("/", "-")
        shard_path = shard_dir / f"{ordinal:04d}-{safe_concept}.json"
        expected_rows = len(batch_records) * 2 * len(contract["jobs"])
        if shard_path.exists():
            existing = read_json(shard_path)
            if (
                existing.get("execution_commit") == commit
                and existing.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and existing.get("prototype_manifest_sha256")
                == sha256_file(PROTOTYPE_MANIFEST_PATH)
                and existing.get("audit", {}).get("row_count") == expected_rows
            ):
                completed.append(str(shard_path.relative_to(ROOT)))
                total_rows += expected_rows
                print(f"Skipping complete shard {ordinal:04d} for {concept}.", flush=True)
                continue
            raise RuntimeError(f"existing shard has incompatible provenance: {shard_path}")
        rows, audit = run_batch(
            runner,
            vector,
            batch_records,
            prototypes,
            prototype_index,
            component_ids,
            contract,
            probe_names,
            probes,
            commit,
        )
        shard = {
            "schema_version": 1,
            "procedure": "rapid-k12-prototype-total-path-shard-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
            "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
            "batch_ordinal": ordinal,
            "scheduled_batch_size": scheduled_size,
            "concept": concept,
            "rows": rows,
            "audit": audit,
        }
        write_json_atomic(shard_path, shard)
        completed.append(str(shard_path.relative_to(ROOT)))
        total_rows += len(rows)
        print(
            f"Completed total shard {ordinal + 1}/{len(batches)} {concept}: "
            f"{len(batch_records)} examples, {len(rows)} rows",
            flush=True,
        )
    torch.cuda.synchronize()
    full_population = not selected_concepts
    expected_total = (
        int(contract["implementation_gates"]["exact_row_count"])
        if full_population
        else sum(len(batch[3]) * 2 * len(contract["jobs"]) for batch in selected_batches)
    )
    if total_rows != expected_total:
        raise RuntimeError(f"execution row count {total_rows} != {expected_total}")
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-prototype-total-path-execution-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "full_population": full_population,
        "concepts_completed": sorted({batch[1] for batch in selected_batches}),
        "batch_count": len(selected_batches),
        "row_count": total_rows,
        "shards": completed,
        "elapsed_seconds": time.perf_counter() - execution_started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "hooks_after_execution": runner.registered_hook_count(),
        "result": "complete",
    }
    write_json_atomic(output_dir / "total-path-execution.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
