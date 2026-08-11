#!/usr/bin/env python3
"""Run the frozen 26-example direct-path K12 operator pilot on CUDA."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day37_run_phase_b import response_deciles  # noqa: E402
from neural_chameleon import (  # noqa: E402
    LinearProbe,
    PairedInterventionRunner,
    RealizedForwardRunner,
    direct_path_monitor,
    load_experimental_split,
    probe_token_margins,
    reconstruct_residual_after,
    response_activation_rms,
)
from neural_chameleon.causal_mechanisms import MechanismComponent  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    captured_head,
    direct_replacement_cache,
    direct_target_recomputations,
    mean_replacements,
    rotate_head_delta,
)


CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
SELECTION_PATH = ROOT / "results/day-44/pilot-selection.json"
SPEC_PATH = ROOT / "results/day-44/frozen-pilot-operator-spec.json"
OPERATOR_MANIFEST_PATH = ROOT / "results/day-44/pilot-operator-tensors.json"
OPERATOR_TENSOR_PATH = ROOT / "results/day-44/pilot-operator-tensors.safetensors"
MODEL_PATH = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
DEFAULT_OUTPUT_DIR = ROOT / "results/day-44"


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_model() -> PairedInterventionRunner:
    torch.manual_seed(44001)
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
    runner = PairedInterventionRunner(model, tokenizer)
    if runner.device.type != "cuda":
        raise RuntimeError(f"Day 44 pilot requires CUDA, found {runner.device}")
    return runner


def load_probes() -> tuple[list[str], tuple[LinearProbe, ...]]:
    paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    names = [path.name.removesuffix("_weights.pt") for path in paths]
    probes = tuple(LinearProbe.load(path) for path in paths)
    if len(probes) != 13:
        raise RuntimeError(f"expected 13 probes, found {len(probes)}")
    return names, probes


def load_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {}
    for split in ("discovery", "validation", "safety-test"):
        for row in load_experimental_split(split):
            by_id[row["example_id"]] = row
    records = []
    for frozen in contract["pilot"]["examples"]:
        row = by_id.get(frozen["example_id"])
        if row is None:
            raise RuntimeError(f"missing pilot example {frozen['example_id']}")
        if (
            row["concept"] != frozen["concept"]
            or int(row["label"]) != 1
            or row["split"] != frozen.get("split", row["split"])
        ):
            raise RuntimeError(f"pilot identity differs for {frozen['example_id']}")
        records.append(row)
    if len(records) != 26 or len({row["example_id"] for row in records}) != 26:
        raise RuntimeError("pilot records are incomplete or duplicated")
    return records


def selected_values(
    capture: Any, component_ids: Sequence[str], layers: Sequence[torch.nn.Module]
) -> torch.Tensor:
    return torch.stack(
        [
            captured_head(capture, MechanismComponent.parse(value), layers).values.float()
            for value in component_ids
        ],
        dim=2,
    )


def selected_o_concat(
    runner: PairedInterventionRunner, component_ids: Sequence[str]
) -> torch.Tensor:
    slices = []
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        attention = runner.layers[component.layer].self_attn
        head_dim = runner._head_dim(attention)
        start = int(component.head) * head_dim
        slices.append(attention.o_proj.weight[:, start : start + head_dim].float())
    return torch.cat(slices, dim=1)


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
    for row, record in enumerate(records):
        source = prototypes[prototype_index[record["example_id"]]]
        valid = response_mask[row].bool()
        result[row, valid] = source[deciles[row, valid]]
    return result


def tangential_delta(
    natural_delta: torch.Tensor,
    normal_state: torch.Tensor,
    response_mask: torch.Tensor,
    o_concat: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    device = o_concat.device
    shape = natural_delta.shape
    flat_delta = natural_delta.reshape(*shape[:2], -1).to(device=device)
    state = normal_state.to(device=device, dtype=torch.float32)
    covector = torch.matmul(state, o_concat)
    numerator = (covector * flat_delta).sum(dim=-1, keepdim=True)
    denominator = covector.square().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    changed = flat_delta - covector * (numerator / denominator)
    valid = response_mask.to(device).unsqueeze(-1)
    changed = torch.where(valid, changed, torch.zeros_like(changed))
    relative_dot = torch.divide(
        (covector * changed).sum(dim=-1).abs(),
        (
            torch.linalg.vector_norm(covector, dim=-1)
            * torch.linalg.vector_norm(changed, dim=-1)
        ).clamp(min=1e-12),
    )
    max_relative = float(relative_dot[response_mask.to(device).bool()].max().item())
    return changed.reshape(shape).cpu(), max_relative


def replacements_from_delta(
    target_values: torch.Tensor,
    delta: torch.Tensor,
    direction: str,
    target: Any,
    component_ids: Sequence[str],
    layers: Sequence[torch.nn.Module],
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    if direction == "induction":
        replacement = target_values + scale * delta
    elif direction == "rescue":
        replacement = target_values - scale * delta
    else:
        raise ValueError(f"unknown direction {direction}")
    return mean_replacements(target, component_ids, replacement, layers)


def direct_endpoint(
    target: Any,
    replacements: Mapping[str, torch.Tensor],
    runner: PairedInterventionRunner,
    target_recomputations: Mapping[int, torch.Tensor],
) -> Any:
    cache = direct_replacement_cache(
        target,
        replacements,
        runner.layers,
        target_recomputations=target_recomputations,
    )
    return direct_path_monitor(target, cache)


def mean_margins(capture: Any, probes: Sequence[LinearProbe]) -> torch.Tensor:
    margins = probe_token_margins(capture, probes)
    mask = capture.response_mask.unsqueeze(0)
    return (margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)


def endpoint_rows(
    output: Any,
    target: Any,
    probes: Sequence[LinearProbe],
    probe_names: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    candidate: str,
    scale: float,
    commit: str,
    contract_sha256: str,
    spec_sha256: str,
    tangency_relative_dot: float | None,
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
                "record_type": "rapid_k12_pilot_direct_endpoint",
                "execution_commit": commit,
                "contract_sha256": contract_sha256,
                "operator_spec_sha256": spec_sha256,
                "example_id": record["example_id"],
                "concept": record["concept"],
                "split": record["split"],
                "label": int(record["label"]),
                "direction": direction,
                "candidate": candidate,
                "scale": scale,
                "path": "direct",
                "probe_names": list(probe_names),
                "mean_raw_margins": output_margins[:, row_index].tolist(),
                "target_mean_raw_margins": target_margins[:, row_index].tolist(),
                "activation_rms": float(output_rms[row_index]),
                "target_activation_rms": float(target_rms[row_index]),
                "response_token_count": int(output.response_mask[row_index].sum()),
                "tangency_relative_dot_max": tangency_relative_dot,
                "random_audit": random_audit,
            }
        )
    return rows


def prepare_batch(
    runner: PairedInterventionRunner,
    records: Sequence[Mapping[str, Any]],
) -> tuple[Any, Any, Any, Any]:
    concept = records[0]["concept"]
    if any(row["concept"] != concept for row in records):
        raise RuntimeError("pilot batches must contain one concept")
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
    return pair, normal, triggered, reconstruct_residual_after(normal, 8).float()


def run_preflight(
    runner: PairedInterventionRunner,
    records: Sequence[Mapping[str, Any]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    spec: Mapping[str, Any],
    output_path: Path,
    commit: str,
) -> None:
    batch = records[:2]
    pair, normal, triggered, normal_state = prepare_batch(runner, batch)
    normal_values = selected_values(normal, component_ids, runner.layers)
    triggered_values = selected_values(triggered, component_ids, runner.layers)
    natural_delta = triggered_values - normal_values
    prototype = prototype_delta_for_batch(
        prototypes, prototype_index, batch, normal.response_mask
    )
    o_concat = selected_o_concat(runner, component_ids)
    tangential, tangency_error = tangential_delta(
        natural_delta, normal_state, normal.response_mask, o_concat
    )
    rotated, random_audit = rotate_head_delta(
        tangential,
        draw_index=0,
        base_seed=int(spec["operators"]["matched_random"]["base_seed"]),
    )
    identity_replacements = mean_replacements(
        normal, component_ids, normal_values, runner.layers
    )
    recomputations = direct_target_recomputations(
        normal, identity_replacements, runner.layers
    )
    identity = direct_endpoint(
        normal, identity_replacements, runner, recomputations
    )
    identity_error = float(
        (
            identity.values.float() - normal.monitor_residual.values.float()
        )[normal.response_mask]
        .abs()
        .max()
    )
    gates = spec["implementation_gates"]
    checks = {
        "cuda": runner.device.type == "cuda",
        "response_ids_exact": torch.equal(pair.normal.response_ids, pair.triggered.response_ids),
        "response_masks_exact": torch.equal(
            pair.normal.response_mask, pair.triggered.response_mask
        ),
        "prototype_shape_exact": prototype.shape == natural_delta.shape,
        "prototype_finite": bool(torch.isfinite(prototype).all()),
        "tangential_shape_exact": tangential.shape == natural_delta.shape,
        "tangential_finite": bool(torch.isfinite(tangential).all()),
        "tangency_within_tolerance": tangency_error
        <= float(gates["tangency_relative_dot_max"]),
        "random_shape_exact": rotated.shape == tangential.shape,
        "haar_invariants_pass": random_audit.passes(),
        "identity_within_tolerance": identity_error
        <= float(gates["identity_hidden_max_abs"]),
        "hooks_removed": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-pilot-real-checkpoint-preflight-v1",
        "preflight_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "operator_spec_sha256": sha256_file(SPEC_PATH),
        "operator_tensor_sha256": sha256_file(OPERATOR_TENSOR_PATH),
        "example_ids": [row["example_id"] for row in batch],
        "identity_hidden_max_abs": identity_error,
        "tangency_relative_dot_max": tangency_error,
        "haar_audit": random_audit.to_dict(),
        "checks": checks,
        "result": "pass" if all(checks.values()) else "fail",
        "candidate_outcomes_generated": False,
    }
    write_json_atomic(output_path, report)
    if report["result"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def run_concept(
    runner: PairedInterventionRunner,
    records: Sequence[Mapping[str, Any]],
    prototypes: torch.Tensor,
    prototype_index: Mapping[str, int],
    component_ids: Sequence[str],
    spec: Mapping[str, Any],
    probe_names: Sequence[str],
    probes: Sequence[LinearProbe],
    commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _pair, normal, triggered, normal_state = prepare_batch(runner, records)
    normal_values = selected_values(normal, component_ids, runner.layers)
    triggered_values = selected_values(triggered, component_ids, runner.layers)
    natural_delta = triggered_values - normal_values
    prototype = prototype_delta_for_batch(
        prototypes, prototype_index, records, normal.response_mask
    )
    o_concat = selected_o_concat(runner, component_ids)
    tangential, tangency_error = tangential_delta(
        natural_delta, normal_state, normal.response_mask, o_concat
    )
    base_seed = int(spec["operators"]["matched_random"]["base_seed"])
    prototype_random, prototype_random_audit = rotate_head_delta(
        prototype, draw_index=0, base_seed=base_seed
    )
    tangential_random, tangential_random_audit = rotate_head_delta(
        tangential, draw_index=0, base_seed=base_seed
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
        jobs: list[tuple[str, torch.Tensor, float, float | None, Mapping[str, Any] | None]] = [
            ("exact_natural_activity", natural_delta, 1.0, None, None),
            ("concept_position_prototype", prototype, 1.0, None, None),
            (
                "concept_position_prototype.random",
                prototype_random,
                1.0,
                None,
                prototype_random_audit.to_dict(),
            ),
            (
                "tangential_actual_activity",
                tangential,
                1.0,
                tangency_error,
                None,
            ),
            (
                "tangential_actual_activity.random",
                tangential_random,
                1.0,
                None,
                tangential_random_audit.to_dict(),
            ),
        ]
        jobs.extend(
            (
                f"tangential_actual_activity.scale_{scale:.1f}",
                tangential,
                float(scale),
                tangency_error,
                None,
            )
            for scale in spec["dose_scales"]
        )
        for candidate, delta, scale, tangency, random_audit in jobs:
            replacements = replacements_from_delta(
                target_values,
                delta,
                direction,
                target,
                component_ids,
                runner.layers,
                scale=scale,
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
                    scale=scale,
                    commit=commit,
                    contract_sha256=sha256_file(CONTRACT_PATH),
                    spec_sha256=sha256_file(SPEC_PATH),
                    tangency_relative_dot=tangency,
                    random_audit=random_audit,
                )
            )
    expected = len(records) * 2 * len(spec["jobs_per_direction"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} rows, produced {len(rows)}")
    audit = {
        "concept": records[0]["concept"],
        "example_ids": [row["example_id"] for row in records],
        "row_count": len(rows),
        "tangency_relative_dot_max": tangency_error,
        "prototype_random_audit": prototype_random_audit.to_dict(),
        "tangential_random_audit": tangential_random_audit.to_dict(),
        "hooks_after_batch": runner.registered_hook_count(),
        "all_rows_finite": all(
            all(math.isfinite(value) for value in row["mean_raw_margins"])
            and math.isfinite(row["activation_rms"])
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
        SELECTION_PATH,
        SPEC_PATH,
        OPERATOR_MANIFEST_PATH,
        OPERATOR_TENSOR_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    selection = read_json(SELECTION_PATH)
    spec = read_json(SPEC_PATH)
    operator_manifest = read_json(OPERATOR_MANIFEST_PATH)
    if spec["parent_contract_sha256"] != sha256_file(CONTRACT_PATH):
        raise RuntimeError("operator spec parent hash differs")
    if spec["pilot_selection_sha256"] != sha256_file(SELECTION_PATH):
        raise RuntimeError("operator spec selection hash differs")
    if operator_manifest["tensor_sha256"] != sha256_file(OPERATOR_TENSOR_PATH):
        raise RuntimeError("prototype tensor hash differs")
    if selection["promoted"] != ["tangential_actual_activity"]:
        raise RuntimeError("pilot selection differs")
    records = load_records(contract)
    prototypes = load_file(OPERATOR_TENSOR_PATH)["prototype_delta"].float()
    prototype_index = {
        example_id: index
        for index, example_id in enumerate(operator_manifest["example_ids"])
    }
    component_ids = tuple(contract["component_set"])
    output_dir = args.output_dir.resolve()
    preflight_path = output_dir / "pilot-preflight.json"
    runner = load_model()
    probe_names, probes = load_probes()
    if args.preflight_only:
        run_preflight(
            runner,
            records,
            prototypes,
            prototype_index,
            component_ids,
            spec,
            preflight_path,
            commit,
        )
        return
    if not preflight_path.exists():
        raise RuntimeError("passing pilot preflight is required")
    preflight = read_json(preflight_path)
    if (
        preflight.get("result") != "pass"
        or preflight.get("preflight_commit") != commit
        or preflight.get("contract_sha256") != sha256_file(CONTRACT_PATH)
        or preflight.get("operator_spec_sha256") != sha256_file(SPEC_PATH)
    ):
        raise RuntimeError("pilot preflight does not pass for this execution")

    selected_concepts = set(args.concept or [])
    unknown = selected_concepts - {row["concept"] for row in records}
    if unknown:
        raise RuntimeError(f"unknown requested concepts: {sorted(unknown)}")
    ordered_concepts = []
    for row in records:
        if row["concept"] not in ordered_concepts:
            ordered_concepts.append(row["concept"])
    if selected_concepts:
        ordered_concepts = [
            concept for concept in ordered_concepts if concept in selected_concepts
        ]
    shard_dir = output_dir / "pilot-shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    execution_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    completed = []
    for concept_index, concept in enumerate(ordered_concepts):
        concept_records = [row for row in records if row["concept"] == concept]
        if len(concept_records) != 2:
            raise RuntimeError(f"concept {concept} does not have exactly two pilot records")
        safe_concept = concept.replace("/", "-")
        shard_path = shard_dir / f"{concept_index:02d}-{safe_concept}.json"
        if shard_path.exists():
            existing = read_json(shard_path)
            if (
                existing.get("execution_commit") == commit
                and existing.get("contract_sha256") == sha256_file(CONTRACT_PATH)
                and existing.get("operator_spec_sha256") == sha256_file(SPEC_PATH)
                and existing.get("audit", {}).get("row_count")
                == len(concept_records) * 2 * len(spec["jobs_per_direction"])
            ):
                completed.append(str(shard_path.relative_to(ROOT)))
                print(f"Skipping complete shard for {concept}.", flush=True)
                continue
            raise RuntimeError(f"existing shard has incompatible provenance: {shard_path}")
        rows, audit = run_concept(
            runner,
            concept_records,
            prototypes,
            prototype_index,
            component_ids,
            spec,
            probe_names,
            probes,
            commit,
        )
        shard = {
            "schema_version": 1,
            "procedure": "rapid-k12-direct-pilot-shard-v1",
            "execution_commit": commit,
            "contract_sha256": sha256_file(CONTRACT_PATH),
            "operator_spec_sha256": sha256_file(SPEC_PATH),
            "operator_tensor_sha256": sha256_file(OPERATOR_TENSOR_PATH),
            "concept": concept,
            "rows": rows,
            "audit": audit,
        }
        write_json_atomic(shard_path, shard)
        completed.append(str(shard_path.relative_to(ROOT)))
        print(
            f"Completed {concept}: {len(rows)} rows, tangency {audit['tangency_relative_dot_max']:.3e}",
            flush=True,
        )
    torch.cuda.synchronize()
    report = {
        "schema_version": 1,
        "procedure": "rapid-k12-direct-pilot-execution-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "operator_spec_sha256": sha256_file(SPEC_PATH),
        "operator_tensor_sha256": sha256_file(OPERATOR_TENSOR_PATH),
        "concepts_completed": ordered_concepts,
        "shards": completed,
        "elapsed_seconds": time.perf_counter() - execution_started,
        "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "hooks_after_execution": runner.registered_hook_count(),
        "result": "complete",
    }
    write_json_atomic(output_dir / "pilot-execution.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
