#!/usr/bin/env python3
"""Run the frozen all-positive population evaluation of the K12 prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import response_deciles  # noqa: E402
from day44_run_k12_pilot import (  # noqa: E402
    direct_endpoint,
    load_model,
    load_probes,
    replacements_from_delta,
    selected_values,
)
from neural_chameleon import (  # noqa: E402
    PairedInterventionRunner,
    RealizedForwardRunner,
    direct_path_monitor,
    load_experimental_split,
    probe_token_margins,
    response_activation_rms,
)
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    direct_replacement_cache,
    direct_target_recomputations,
    mean_replacements,
    rotate_head_delta,
)


CONTRACT_PATH = ROOT / "results/day-45/frozen-prototype-population-contract.json"
PROTOTYPE_MANIFEST_PATH = ROOT / "results/day-45/population-prototype-tensors.json"
PROTOTYPE_TENSOR_PATH = ROOT / "artifacts/rapid-k12-v1/population-prototypes.safetensors"
DAY44_RUNNER_PATH = ROOT / "scripts/day44_run_k12_pilot.py"
DAY44_SPEC_PATH = ROOT / "results/day-44/frozen-pilot-operator-spec.json"
DAY44_SUMMARY_PATH = ROOT / "results/day-44/pilot-summary.json"
DAY44_AUDIT_PATH = ROOT / "results/day-44/pilot-audit.json"
CAPTURE_MANIFEST_PATH = ROOT / "results/day-33/execution-artifact-manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "results/day-45"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--concept", action="append")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


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
        raise RuntimeError(f"{relative} differs from execution commit {commit}")


def validate_parents(contract: Mapping[str, Any]) -> None:
    expected = contract["parents"]
    observed = {
        "pilot_operator_spec_sha256": sha256_file(DAY44_SPEC_PATH),
        "pilot_summary_sha256": sha256_file(DAY44_SUMMARY_PATH),
        "pilot_audit_sha256": sha256_file(DAY44_AUDIT_PATH),
        "capture_manifest_sha256": sha256_file(CAPTURE_MANIFEST_PATH),
    }
    for key, value in observed.items():
        if expected[key] != value:
            raise RuntimeError(f"frozen parent differs: {key}")


def load_population_records(
    contract: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_id = {}
    for split in contract["population"]["splits"]:
        for row in load_experimental_split(split):
            by_id[row["example_id"]] = row
    records = []
    frozen_examples = manifest["examples"]
    for frozen in frozen_examples:
        row = by_id.get(frozen["example_id"])
        if row is None:
            raise RuntimeError(f"missing population example {frozen['example_id']}")
        if (
            row["concept"] != frozen["concept"]
            or row["split"] != frozen["split"]
            or int(row["label"]) != 1
        ):
            raise RuntimeError(f"population identity differs for {frozen['example_id']}")
        records.append({**row, "response_token_count": frozen["response_token_count"]})
    expected = int(contract["population"]["examples"])
    if len(records) != expected or len({row["example_id"] for row in records}) != expected:
        raise RuntimeError("population records are incomplete or duplicated")
    if len({row["concept"] for row in records}) != int(
        contract["population"]["concepts"]
    ):
        raise RuntimeError("population concept count differs")
    return records


def make_batches(
    records: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[tuple[int, str, int, list[Mapping[str, Any]]]]:
    result = []
    ordinal = 0
    lower = -1
    for concept in sorted({row["concept"] for row in records}):
        concept_records = [row for row in records if row["concept"] == concept]
        lower = -1
        for schedule in contract["execution"]["batch_schedule"]:
            maximum = int(schedule["maximum_response_tokens"])
            batch_size = int(schedule["batch_size"])
            bucket = sorted(
                (
                    row
                    for row in concept_records
                    if lower < int(row["response_token_count"]) <= maximum
                ),
                key=lambda row: (int(row["response_token_count"]), row["example_id"]),
            )
            for start in range(0, len(bucket), batch_size):
                result.append((ordinal, concept, batch_size, bucket[start : start + batch_size]))
                ordinal += 1
            lower = maximum
    if sum(len(batch) for _ordinal, _concept, _size, batch in result) != len(records):
        raise RuntimeError("batch schedule did not cover the exact population")
    return result


def prototype_delta_for_batch(
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    records: Sequence[Mapping[str, Any]],
    response_mask: torch.Tensor,
) -> torch.Tensor:
    deciles = response_deciles(response_mask)
    result = torch.zeros(
        (*response_mask.shape, prototypes.shape[2], prototypes.shape[3]),
        dtype=torch.float32,
    )
    for row_index, record in enumerate(records):
        source = prototypes[prototype_index[record["example_id"]]]
        valid = response_mask[row_index].bool()
        result[row_index, valid] = source[deciles[row_index, valid]]
    return result


def prepare_batch(
    runner: PairedInterventionRunner,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any]:
    concept = records[0]["concept"]
    if any(row["concept"] != concept for row in records):
        raise RuntimeError("population batches must contain one concept")
    pair = runner.prepare_pairs(
        [row["prompt"] for row in records],
        [row["response"] for row in records],
        trigger=concept,
    )
    realized = RealizedForwardRunner(runner, monitor_layer=12)
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


def mean_margins(capture: Any, probes: Sequence[Any]) -> torch.Tensor:
    margins = probe_token_margins(capture, probes)
    mask = capture.response_mask.unsqueeze(0)
    return (margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)


def endpoint_rows(
    output: Any,
    target: Any,
    probes: Sequence[Any],
    probe_names: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    candidate: str,
    commit: str,
    contract_sha256: str,
    prototype_manifest_sha256: str,
    random_audit: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    output_margins = mean_margins(output, probes)
    target_margins = mean_margins(target.monitor_residual, probes)
    output_rms = response_activation_rms(output)
    target_rms = response_activation_rms(target.monitor_residual)
    rows = []
    for row_index, record in enumerate(records):
        rows.append(
            {
                "schema_version": 1,
                "record_type": "rapid_k12_population_direct_endpoint",
                "execution_commit": commit,
                "contract_sha256": contract_sha256,
                "prototype_manifest_sha256": prototype_manifest_sha256,
                "example_id": record["example_id"],
                "concept": record["concept"],
                "split": record["split"],
                "label": int(record["label"]),
                "direction": direction,
                "candidate": candidate,
                "scale": 1.0,
                "path": "direct",
                "probe_names": list(probe_names),
                "mean_raw_margins": output_margins[:, row_index].tolist(),
                "target_mean_raw_margins": target_margins[:, row_index].tolist(),
                "activation_rms": float(output_rms[row_index]),
                "target_activation_rms": float(target_rms[row_index]),
                "response_token_count": int(output.response_mask[row_index].sum()),
                "random_audit": random_audit,
            }
        )
    return rows


def identity_error(
    runner: PairedInterventionRunner,
    target: Any,
    target_values: torch.Tensor,
    component_ids: Sequence[str],
) -> float:
    replacements = mean_replacements(target, component_ids, target_values, runner.layers)
    recomputations = direct_target_recomputations(target, replacements, runner.layers)
    cache = direct_replacement_cache(
        target, replacements, runner.layers, target_recomputations=recomputations
    )
    output = direct_path_monitor(target, cache)
    return float(
        (
            output.values.float() - target.monitor_residual.values.float()
        )[target.response_mask]
        .abs()
        .max()
    )


def run_preflight(
    runner: PairedInterventionRunner,
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
    identity_errors = []
    haar_audits = []
    observed_batch_sizes = []
    all_checks = []
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
        error = identity_error(runner, normal, normal_values, component_ids)
        identity_errors.append(error)
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
        all_checks.append(
            {
                "response_ids_exact": torch.equal(
                    pair.normal.response_ids, pair.triggered.response_ids
                ),
                "response_masks_exact": torch.equal(
                    pair.normal.response_mask, pair.triggered.response_mask
                ),
                "natural_shape_exact": natural_delta.shape == prototype.shape,
                "prototype_shape_exact": prototype.shape == rotated.shape,
                "prototype_finite": bool(torch.isfinite(prototype).all()),
                "random_finite": bool(torch.isfinite(rotated).all()),
                "haar_invariants_pass": haar.passes(),
                "identity_within_tolerance": error
                <= float(contract["implementation_gates"]["identity_hidden_max_abs"]),
                "hooks_removed": runner.registered_hook_count() == 0,
            }
        )
    checks = {
        "cuda": runner.device.type == "cuda",
        "tensor_shape_exact": list(prototypes.shape) == [866, 10, 12, 256],
        "all_representative_batches_pass": all(
            all(batch_checks.values()) for batch_checks in all_checks
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
        "procedure": "rapid-k12-prototype-population-preflight-v1",
        "preflight_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "prototype_manifest_sha256": sha256_file(PROTOTYPE_MANIFEST_PATH),
        "prototype_tensor_sha256": sha256_file(PROTOTYPE_TENSOR_PATH),
        "representative_batches": observed_batch_sizes,
        "identity_hidden_max_abs": max(identity_errors),
        "haar_audits": haar_audits,
        "batch_checks": all_checks,
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
        "candidate_outcomes_generated": False,
    }
    write_json_atomic(output_path, report)
    if report["result"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_batch(
    runner: PairedInterventionRunner,
    records: Sequence[Mapping[str, Any]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    contract: Mapping[str, Any],
    probe_names: Sequence[str],
    probes: Sequence[Any],
    commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _pair, normal, triggered = prepare_batch(runner, records)
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
    rows = []
    for direction, target, target_values in (
        ("induction", normal, normal_values),
        ("rescue", triggered, triggered_values),
    ):
        identity_replacements = mean_replacements(
            target, component_ids, target_values, runner.layers
        )
        recomputations = direct_target_recomputations(
            target, identity_replacements, runner.layers
        )
        for candidate, delta, audit in (
            ("exact_natural_activity", natural_delta, None),
            ("concept_position_prototype", prototype, None),
            ("concept_position_prototype.random", prototype_random, random_audit.to_dict()),
        ):
            replacements = replacements_from_delta(
                target_values,
                delta,
                direction,
                target,
                component_ids,
                runner.layers,
            )
            output = direct_endpoint(target, replacements, runner, recomputations)
            rows.extend(
                endpoint_rows(
                    output,
                    target,
                    probes,
                    probe_names,
                    records,
                    direction=direction,
                    candidate=candidate,
                    commit=commit,
                    contract_sha256=sha256_file(CONTRACT_PATH),
                    prototype_manifest_sha256=sha256_file(PROTOTYPE_MANIFEST_PATH),
                    random_audit=audit,
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
        PROTOTYPE_MANIFEST_PATH,
        DAY44_RUNNER_PATH,
        DAY44_SPEC_PATH,
        DAY44_SUMMARY_PATH,
        DAY44_AUDIT_PATH,
        CAPTURE_MANIFEST_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    manifest = read_json(PROTOTYPE_MANIFEST_PATH)
    validate_parents(contract)
    if manifest["contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("prototype manifest contract hash differs")
    if manifest["tensor_sha256"] != sha256_file(PROTOTYPE_TENSOR_PATH):
        raise RuntimeError("prototype tensor hash differs")
    if manifest["result"] != "pass" or not all(manifest["checks"].values()):
        raise RuntimeError("prototype tensor manifest does not pass")
    records = load_population_records(contract, manifest)
    prototypes = load_file(PROTOTYPE_TENSOR_PATH)["prototype_delta"].float()
    prototype_index = {
        row["example_id"]: index for index, row in enumerate(manifest["examples"])
    }
    component_ids = tuple(manifest["component_ids"])
    batches = make_batches(records, contract)
    output_dir = args.output_dir.resolve()
    preflight_path = output_dir / "population-preflight.json"
    runner = load_model()
    probe_names, probes = load_probes()
    if args.preflight_only:
        run_preflight(
            runner,
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
        raise RuntimeError("passing population preflight is required")
    preflight = read_json(preflight_path)
    if (
        preflight.get("result") != "pass"
        or preflight.get("preflight_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or preflight.get("prototype_manifest_sha256")
        != sha256_file(PROTOTYPE_MANIFEST_PATH)
        or preflight.get("prototype_tensor_sha256") != sha256_file(PROTOTYPE_TENSOR_PATH)
    ):
        raise RuntimeError("population preflight does not pass for this execution")

    selected_concepts = set(args.concept or [])
    all_concepts = {row["concept"] for row in records}
    unknown = selected_concepts - all_concepts
    if unknown:
        raise RuntimeError(f"unknown requested concepts: {sorted(unknown)}")
    selected_batches = [
        batch for batch in batches if not selected_concepts or batch[1] in selected_concepts
    ]
    shard_dir = output_dir / "population-shards"
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
            "procedure": "rapid-k12-prototype-population-shard-v1",
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
            f"Completed shard {ordinal + 1}/{len(batches)} {concept}: "
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
        "procedure": "rapid-k12-prototype-population-execution-v1",
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
    write_json_atomic(output_dir / "population-execution.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
