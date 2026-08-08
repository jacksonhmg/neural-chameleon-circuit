#!/usr/bin/env python3
"""Run frozen benign geometry capture and aligned selected-head transport."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import (  # noqa: E402
    CachedTailTransplantRunner,
    LinearProbe,
    TransplantJob,
    TransplantMember,
    TruncatedComponentRunner,
    destination_relative_capture,
    load_experimental_split,
    parse_head_id,
)
from day17_run_transfer_atlas import (  # noqa: E402
    ANALYSIS_PLAN_PATH,
    MODEL_PATH,
    PLAN_PATH,
    PROBE_DIR,
    load_model,
    run_with_full_layer_inputs,
    site_by_id,
)


RESULT_DIR = ROOT / "results/day-18"
WORKING_PATH = RESULT_DIR / "geometry-transfer-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "geometry-transfer-results.jsonl.gz"
GEOMETRY_PATH = RESULT_DIR / "geometry-captures.npz"
PROJECTION_PATH = RESULT_DIR / "projection-features.npz"
PREFLIGHT_PATH = RESULT_DIR / "geometry-preflight.json"
EXPECTED_EXAMPLES = 64
EXPECTED_CONDITIONS = 2 + 2 * 3 * 12 * 12
EXPECTED_ROWS = EXPECTED_EXAMPLES * EXPECTED_CONDITIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-chunk-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT) != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_subset(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for concept in sorted({row["concept"] for row in records}):
        eligible = [row for row in records if row["concept"] == concept and int(row["label"]) == 1]
        selected.extend(sorted(
            eligible,
            key=lambda row: (
                hashlib.sha256(f"site-shuffle-geometry:{row['example_id']}".encode()).hexdigest(),
                row["example_id"],
            ),
        )[:16])
    if len(selected) != EXPECTED_EXAMPLES:
        raise ValueError("expected 16 positives for each of four discovery concepts")
    return selected


def output_projection_data(runner: Any, head_ids: Sequence[str]) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str], torch.Tensor], np.ndarray]:
    matrices: dict[str, torch.Tensor] = {}
    for head_id in head_ids:
        layer, head = parse_head_id(head_id)
        attention = runner.layers[layer].self_attn
        head_dim = runner._head_dim(attention)
        weight = attention.o_proj.weight.detach().float().cpu()
        matrices[head_id] = weight[:, head * head_dim : (head + 1) * head_dim].contiguous()
    pinverses = {head_id: torch.linalg.pinv(matrix) for head_id, matrix in matrices.items()}
    transforms = {
        (destination_id, source_id): pinverses[destination_id] @ matrices[source_id]
        for destination_id in head_ids
        for source_id in head_ids
    }
    projection_cosine = np.empty((len(head_ids), len(head_ids)), dtype=np.float32)
    flattened = [matrices[head_id].flatten() for head_id in head_ids]
    for destination_index, destination in enumerate(flattened):
        for source_index, source in enumerate(flattened):
            projection_cosine[destination_index, source_index] = float(
                torch.nn.functional.cosine_similarity(destination, source, dim=0)
            )
    return matrices, transforms, projection_cosine


def output_projection_matrices(
    runner: Any, head_ids: Sequence[str]
) -> dict[str, torch.Tensor]:
    matrices = {}
    for head_id in head_ids:
        layer, head = parse_head_id(head_id)
        attention = runner.layers[layer].self_attn
        head_dim = runner._head_dim(attention)
        weight = attention.o_proj.weight.detach().float().cpu()
        matrices[head_id] = weight[
            :, head * head_dim : (head + 1) * head_dim
        ].contiguous()
    return matrices


def geometry_jobs_for_base(
    base_name: str,
    selected: Sequence[str],
    sites: Mapping[str, Any],
    normal_captures: Mapping[Any, Any],
    triggered_captures: Mapping[Any, Any],
    transforms: Mapping[tuple[str, str], torch.Tensor],
) -> list[tuple[dict[str, Any], TransplantJob]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    sign = 1.0 if base_name == "normal" else -1.0
    direction = "induction" if base_name == "normal" else "rescue"
    jobs = []
    for destination_id in selected:
        destination_site = sites[destination_id]
        for source_id in selected:
            source_site = sites[source_id]
            source_layer, _ = parse_head_id(source_id)
            destination_layer, _ = parse_head_id(destination_id)
            route = "identity" if source_id == destination_id else "within_layer" if source_layer == destination_layer else "earlier_to_later" if source_layer < destination_layer else "later_to_earlier"
            for transport in ("raw", "rms", "aligned"):
                transform = transforms[(destination_id, source_id)] if transport == "aligned" else None
                rms_match = transport == "rms"
                condition_id = f"{transport}:{base_name}:{source_id}->{destination_id}"
                capture = destination_relative_capture(
                    base_captures[destination_site],
                    normal_captures[source_site],
                    triggered_captures[source_site],
                    alpha=1.0,
                    sign=sign,
                    destination_normal=normal_captures[destination_site],
                    destination_triggered=triggered_captures[destination_site],
                    rms_match=rms_match,
                    transform=transform,
                )
                jobs.append(({
                    "condition_id": condition_id,
                    "base_condition": base_name,
                    "direction": direction,
                    "transport": transport,
                    "source_id": source_id,
                    "destination_id": destination_id,
                    "route_class": route,
                }, TransplantJob(condition_id, (TransplantMember(destination_site, capture),))))
    if len(jobs) != 3 * 12 * 12:
        raise AssertionError("unexpected Day 18 job count")
    return jobs


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    result = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in result:
                raise ValueError(f"duplicate row {key}")
            result[key] = row
    return result


def append_row(completed: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["example_id"], row["condition_id"])
    if key in completed:
        return
    with WORKING_PATH.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    completed[key] = row


def seal(completed: Mapping[tuple[str, str], dict[str, Any]]) -> None:
    if len(completed) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, found {len(completed)}")
    rows = sorted(completed.values(), key=lambda row: (row["concept"], row["example_id"], row["condition_id"]))
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def run_preflight(runner: Any, plan: Mapping[str, Any], transforms: Mapping[tuple[str, str], torch.Tensor]) -> None:
    selected = list(plan["selected_heads"])
    finite = all(torch.isfinite(value).all() for value in transforms.values())
    identity_errors = []
    identity = torch.eye(next(iter(transforms.values())).shape[0])
    for head_id in selected:
        identity_errors.append(float((transforms[(head_id, head_id)] - identity).abs().max()))
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day18-preflight",
        "status": "pass" if finite and max(identity_errors) <= 0.01 and runner.registered_hook_count() == 0 else "fail",
        "transform_count": len(transforms),
        "all_transforms_finite": finite,
        "maximum_identity_transform_error": max(identity_errors),
        "identity_tolerance": 0.01,
        "registered_hook_count": runner.registered_hook_count(),
        "safety_split_accessed": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 18 preflight failed")


def run_grid(
    runner: Any,
    plan: Mapping[str, Any],
    matrices: Mapping[str, torch.Tensor],
    transforms: Mapping[tuple[str, str], torch.Tensor],
    projection_cosine: np.ndarray,
    *,
    batch_size: int,
    group_chunk_size: int,
) -> None:
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    records = development_subset(load_experimental_split("discovery"))
    selected = list(plan["selected_heads"])
    population = selected + list(plan["null_heads"]["members"])
    sites = site_by_id(population)
    capture_sites = tuple(sites.values())
    completed = read_completed()
    implementation_commit = git_head()
    plan_sha256 = sha256_file(PLAN_PATH)
    geometry_example_ids: list[str] = []
    geometry_concepts: list[str] = []
    raw_deltas: list[np.ndarray] = []
    residual_deltas: list[np.ndarray] = []
    delta_rms_values: list[np.ndarray] = []
    print(f"Resuming Day 18 with {len(completed)}/{EXPECTED_ROWS} rows.", flush=True)
    for concept in sorted({row["concept"] for row in records}):
        examples_for_concept = sorted((row for row in records if row["concept"] == concept), key=lambda row: row["example_id"])
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        truncated = TruncatedComponentRunner(runner, probe, monitor_layer=12)
        tail = CachedTailTransplantRunner(runner, probe, monitor_layer=12)
        trigger = analysis_plan["conditions"]["correct_triggers"][concept]
        for batch_index, examples in enumerate(batched(examples_for_concept, batch_size), start=1):
            pair = runner.prepare_pairs([row["prompt"] for row in examples], [row["response"] for row in examples], trigger)
            normal, normal_inputs = run_with_full_layer_inputs(runner, truncated, pair.normal, capture_sites)
            triggered, triggered_inputs = run_with_full_layer_inputs(runner, truncated, pair.triggered, capture_sites)
            for example_index, example in enumerate(examples):
                geometry_example_ids.append(example["example_id"])
                geometry_concepts.append(concept)
                example_raw = []
                example_residual = []
                example_rms = []
                mask = pair.normal.response_mask[example_index]
                for head_id in population:
                    site = sites[head_id]
                    delta = triggered.captures[site].values[example_index].float() - normal.captures[site].values[example_index].float()
                    valid = delta[mask]
                    mean_delta = valid.mean(dim=0)
                    example_raw.append(mean_delta.numpy())
                    example_residual.append((matrices[head_id] @ mean_delta).numpy())
                    example_rms.append(float(torch.sqrt(valid.square().mean())))
                raw_deltas.append(np.stack(example_raw))
                residual_deltas.append(np.stack(example_residual))
                delta_rms_values.append(np.asarray(example_rms, dtype=np.float32))
                common = {
                    "schema_version": 1,
                    "procedure": "site-shuffling-v1-day18",
                    "implementation_commit": implementation_commit,
                    "plan_sha256": plan_sha256,
                    "concept": concept,
                    "example_id": example["example_id"],
                    "label": 1,
                    "split": "discovery-geometry-development",
                    "post_confirmatory": True,
                }
                append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "base_condition": "normal", "probe_score": float(normal.probe_scores[example_index])})
                append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(triggered.probe_scores[example_index])})
            for base_name, condition, cached_inputs in (("normal", pair.normal, normal_inputs), ("correct_trigger", pair.triggered, triggered_inputs)):
                specifications = geometry_jobs_for_base(base_name, selected, sites, normal.captures, triggered.captures, transforms)
                for destination_layer in (9, 10, 11, 12):
                    layer_specs = [item for item in specifications if parse_head_id(item[0]["destination_id"])[0] == destination_layer]
                    for chunk in batched(layer_specs, group_chunk_size):
                        pending = [item for item in chunk if any((example["example_id"], item[0]["condition_id"]) not in completed for example in examples)]
                        if not pending:
                            continue
                        result = tail.run_truncated_from_layer(condition, [job for _spec, job in pending], start_layer=destination_layer, cached_input=cached_inputs[destination_layer])
                        for job_index, (specification, _job) in enumerate(pending):
                            for example_index, example in enumerate(examples):
                                append_row(completed, {
                                    "schema_version": 1,
                                    "procedure": "site-shuffling-v1-day18",
                                    "implementation_commit": implementation_commit,
                                    "plan_sha256": plan_sha256,
                                    "concept": concept,
                                    "example_id": example["example_id"],
                                    "label": 1,
                                    "split": "discovery-geometry-development",
                                    "post_confirmatory": True,
                                    "record_type": "intervention",
                                    **specification,
                                    "probe_score": float(result.probe_scores[job_index, example_index]),
                                })
            print(f"Day 18 {concept} batch {batch_index}: {len(completed)}/{EXPECTED_ROWS}", flush=True)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    seal(completed)
    np.savez_compressed(
        GEOMETRY_PATH,
        example_ids=np.asarray(geometry_example_ids),
        concepts=np.asarray(geometry_concepts),
        head_ids=np.asarray(population),
        raw_deltas=np.stack(raw_deltas),
        residual_deltas=np.stack(residual_deltas),
        delta_rms=np.stack(delta_rms_values),
    )
    np.savez_compressed(
        PROJECTION_PATH,
        head_ids=np.asarray(selected),
        projection_cosine=projection_cosine,
    )
    print(f"Sealed {len(completed)} Day 18 rows and geometry arrays.", flush=True)


def main() -> None:
    args = parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(PLAN_PATH, commit)
    runner = load_model()
    matrices, transforms, projection_cosine = output_projection_data(runner, plan["selected_heads"])
    # Null-head matrices are needed for the descriptive 24-head geometry archive.
    matrices.update(
        output_projection_matrices(runner, plan["null_heads"]["members"])
    )
    if not args.skip_preflight:
        run_preflight(runner, plan, transforms)
    if args.preflight_only:
        return
    run_grid(runner, plan, matrices, transforms, projection_cosine, batch_size=args.batch_size, group_chunk_size=args.group_chunk_size)


if __name__ == "__main__":
    main()
