#!/usr/bin/env python3
"""Run the resumable frozen Phase B causal-operation matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    LinearProbe,
    MechanismComponent,
    PairedBatch,
    PairedInterventionRunner,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    align_paired_prompts,
    build_source_mask_partition,
    capture_layer_input,
    direct_path_target_recomputations,
    direct_path_monitor,
    load_experimental_split,
    probe_sequence_scores,
    probe_token_margins,
    repeat_condition,
    transplant_job_from_cache,
)
from neural_chameleon.controller_actuator import SourceRegion  # noqa: E402
from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    AttentionStateCaptureRunner,
    align_attention_indices,
    attention_operation_replacements,
    attention_sites,
    captured_head,
    direct_replacement_cache,
    frontier_configurations,
    frontier_patch_cache,
    mean_replacements,
    random_control_replacements,
    source_replacements,
    total_replacement_cache,
    zero_replacements,
)


CONTRACT_PATH = ROOT / "results/day-36/frozen-phase-a-b-contract.json"
CLARIFICATION_PATH = (
    ROOT / "results/day-37/frozen-phase-b-row-accounting-clarification.json"
)
ATTENTION_FREEZE_PATH = (
    ROOT / "results/day-37/frozen-attention-operator-implementation.json"
)
RESULT_DIR = ROOT / "results/day-39"
ARTIFACT_DIR = ROOT / "artifacts/post-gate1-phase-b-v1"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
NATURAL_PATH = RESULT_DIR / "natural-endpoints.working.jsonl"
ABSOLUTE_PATH = RESULT_DIR / "absolute-effects.working.jsonl"
RANDOM_PATH = RESULT_DIR / "random-effects.working.jsonl"
FRONTIER_PATH = RESULT_DIR / "frontier-effects.working.jsonl"
ATTENTION_PATH = RESULT_DIR / "attention-effects.working.jsonl"
SELECTION_PATH = RESULT_DIR / "development-selection.json"
PARAMETERS_PATH = RESULT_DIR / "execution-parameters.json"
OPERATIONS = (
    "pattern_patch_values_retained",
    "value_patch_pattern_retained",
    "concept_span_qk",
    "concept_span_ov",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "means",
            "absolute-random",
            "frontier-discovery",
            "attention-discovery",
            "select",
            "frontier-eval",
            "attention-eval",
        ),
        required=True,
    )
    parser.add_argument(
        "--models", nargs="+", choices=("chameleon", "precursor"), default=None
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--job-chunk-size", type=int, default=16)
    parser.add_argument("--limit-per-cell", type=int)
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
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from execution commit {commit}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def execution_id(commit: str) -> str:
    digest = hashlib.sha256()
    digest.update(commit.encode())
    digest.update(CONTRACT_PATH.read_bytes())
    digest.update(CLARIFICATION_PATH.read_bytes())
    digest.update(ATTENTION_FREEZE_PATH.read_bytes())
    return f"post-gate1-phase-b-v1-{digest.hexdigest()[:16]}"


def load_records(limit_per_cell: int | None = None) -> list[dict[str, Any]]:
    records = (
        load_experimental_split("discovery")
        + load_experimental_split("validation")
        + load_experimental_split("safety-test")
    )
    records = sorted(
        records,
        key=lambda row: (
            row["split"],
            row["concept"],
            int(row["label"]),
            row["example_id"],
        ),
    )
    if limit_per_cell is None:
        return records
    if limit_per_cell <= 0:
        raise ValueError("limit-per-cell must be positive")
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    selected = []
    for record in records:
        key = (record["split"], record["concept"], int(record["label"]))
        if counts[key] < limit_per_cell:
            selected.append(record)
            counts[key] += 1
    return selected


def group_records(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[
            (record["concept"], record["trigger_concept"], int(record["label"]))
        ].append(record)
    return [groups[key] for key in sorted(groups)]


def model_config(contract: Mapping[str, Any], model_name: str) -> Mapping[str, Any]:
    return contract["models"][
        "chameleon" if model_name == "chameleon" else "exact_precursor"
    ]


def load_model(
    contract: Mapping[str, Any], model_name: str
) -> PairedInterventionRunner:
    config = model_config(contract, model_name)
    path = ROOT / config["local_path"]
    torch.manual_seed(int(contract["inference"]["implementation_audit_seed"]))
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=contract["models"]["attention_implementation"],
        local_files_only=True,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.eos_token_id
        )
    return PairedInterventionRunner(model, tokenizer)


def load_probes() -> tuple[tuple[str, ...], tuple[LinearProbe, ...]]:
    paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    names = tuple(path.name.removesuffix("_weights.pt") for path in paths)
    probes = tuple(LinearProbe.load(path) for path in paths)
    if len(probes) != 13:
        raise ValueError("expected thirteen released probes")
    return names, probes


def load_probe_scale() -> Tensor:
    summary = read_json(ROOT / "results/day-33/intermediate-prediction-summary.json")
    scale = torch.tensor(summary["probe_standardization"], dtype=torch.float32)
    if scale.shape != (13,) or torch.any(scale <= 0):
        raise ValueError("sealed probe standardization is invalid")
    return scale


def component_ids(contract: Mapping[str, Any], family: str = "selected") -> tuple[str, ...]:
    key = (
        "k12_ordered"
        if family == "selected"
        else "layer_count_matched_nonselected_heads"
    )
    return tuple(contract["component_sets"][key])


def components_by_layer(ids: Sequence[str]) -> dict[int, tuple[str, ...]]:
    result: dict[int, list[str]] = defaultdict(list)
    for component_id in ids:
        component = MechanismComponent.parse(component_id)
        result[component.layer].append(component_id)
    return {layer: tuple(values) for layer, values in sorted(result.items())}


def capture_head_tensor(
    capture: Any, ids: Sequence[str], layers: Sequence[torch.nn.Module]
) -> Tensor:
    return torch.stack(
        [
            captured_head(capture, MechanismComponent.parse(value), layers).values.float()
            for value in ids
        ],
        dim=2,
    )


def response_deciles(mask: Tensor) -> Tensor:
    result = torch.full(mask.shape, -1, dtype=torch.long)
    for row in range(mask.shape[0]):
        count = int(mask[row].sum())
        positions = torch.arange(mask.shape[1])
        deciles = torch.clamp((10 * positions) // max(count - 1, 1), max=9)
        result[row, :count] = deciles[:count]
    return result


def mean_stats_path(model_name: str) -> Path:
    return ARTIFACT_DIR / f"{model_name}-normal-mean-stats.pt"


def accumulate_mean_stats(
    records: Sequence[dict[str, Any]], values: Tensor, response_mask: Tensor
) -> dict[str, Any]:
    """Return batch increments for concept/label and discovery-global cells."""
    deciles = response_deciles(response_mask)
    increments: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row, record in enumerate(records):
        for decile in range(10):
            selected = deciles[row] == decile
            if not selected.any():
                continue
            cells = [
                ("concept", record["concept"], int(record["label"]), decile)
            ]
            if record["split"] == "discovery":
                cells.append(("discovery_global", int(record["label"]), decile))
            for cell in cells:
                value_sum = values[row, selected].double().sum(dim=0)
                value_count = int(selected.sum())
                if cell not in increments:
                    increments[cell] = {"sum": value_sum, "count": value_count}
                else:
                    increments[cell]["sum"] += value_sum
                    increments[cell]["count"] += value_count
    return increments


def merge_mean_stats(
    destination: dict[tuple[Any, ...], dict[str, Any]],
    source: Mapping[tuple[Any, ...], Mapping[str, Any]],
) -> None:
    for key, value in source.items():
        if key not in destination:
            destination[key] = {
                "sum": value["sum"].clone(),
                "count": int(value["count"]),
            }
        else:
            destination[key]["sum"] += value["sum"]
            destination[key]["count"] += int(value["count"])


def run_means(
    model_name: str,
    contract: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    commit: str,
    run_id: str,
) -> None:
    path = mean_stats_path(model_name)
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact["execution_commit"] != commit or artifact["execution_id"] != run_id:
            raise RuntimeError("mean-stat artifact has a different execution identity")
        return
    runner = load_model(contract, model_name)
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    ids = component_ids(contract)
    stats: dict[tuple[Any, ...], dict[str, Any]] = {}
    observed = []
    for group in group_records(records):
        for batch_records in batched(group, batch_size):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in batch_records],
                [row["response"] for row in batch_records],
                batch_records[0]["trigger_concept"],
            )
            normal = realized.run(pair.normal)
            values = capture_head_tensor(normal, ids, runner.layers)
            merge_mean_stats(
                stats,
                accumulate_mean_stats(batch_records, values, normal.response_mask),
            )
            observed.extend(row["example_id"] for row in batch_records)
            if runner.registered_hook_count() != 0:
                raise RuntimeError("mean capture leaked hooks")
            release_memory()
    if len(observed) != len(records) or len(set(observed)) != len(records):
        raise RuntimeError("mean capture population is incomplete or duplicated")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": 1,
            "execution_commit": commit,
            "execution_id": run_id,
            "model": model_name,
            "component_ids": ids,
            "record_count": len(records),
            "stats": stats,
        },
        temporary,
    )
    temporary.replace(path)
    del runner
    release_memory()


def loo_mean_tensor(
    records: Sequence[dict[str, Any]],
    normal_values: Tensor,
    response_mask: Tensor,
    artifact: Mapping[str, Any],
) -> Tensor:
    stats = artifact["stats"]
    deciles = response_deciles(response_mask)
    result = torch.zeros_like(normal_values)
    for row, record in enumerate(records):
        for decile in range(10):
            selected = deciles[row] == decile
            if not selected.any():
                continue
            key = ("concept", record["concept"], int(record["label"]), decile)
            total = stats[key]["sum"] - normal_values[row, selected].double().sum(dim=0)
            count = int(stats[key]["count"]) - int(selected.sum())
            if count <= 0:
                fallback = ("discovery_global", int(record["label"]), decile)
                total = stats[fallback]["sum"]
                count = int(stats[fallback]["count"])
            result[row, selected] = (total / count).float()
    return result


def capture_summary(
    capture: Any, probes: Sequence[LinearProbe]
) -> tuple[Tensor, Tensor, Tensor]:
    margins = probe_token_margins(capture, probes)
    mask = capture.response_mask.unsqueeze(0)
    mean_margins = ((margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)).T
    scores = probe_sequence_scores(margins, capture.response_mask).T
    token_rms = capture.values.float().square().mean(dim=-1).sqrt()
    rms = (token_rms * capture.response_mask).sum(dim=1) / capture.response_mask.sum(
        dim=1
    ).clamp(min=1)
    return mean_margins, scores, rms


def natural_summary(
    capture: Any, probes: Sequence[LinearProbe]
) -> tuple[Tensor, Tensor, Tensor]:
    return capture_summary(capture.monitor_residual, probes)


def common_row(
    model_name: str,
    record: Mapping[str, Any],
    probe_names: Sequence[str],
    commit: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "execution_id": run_id,
        "model": model_name,
        "split": record["split"],
        "concept": record["concept"],
        "trigger_concept": record["trigger_concept"],
        "label": int(record["label"]),
        "example_id": record["example_id"],
        "probe_names": list(probe_names),
    }


def effect_payload(
    margins: Tensor, scores: Tensor, rms: Tensor, row: int
) -> dict[str, Any]:
    return {
        "mean_raw_margins": [float(value) for value in margins[row]],
        "sequence_scores": [float(value) for value in scores[row]],
        "activation_rms": float(rms[row]),
    }


def write_natural(
    model_name: str,
    records: Sequence[dict[str, Any]],
    conditions: Sequence[tuple[str, Any]],
    probe_names: Sequence[str],
    probes: Sequence[LinearProbe],
    commit: str,
    run_id: str,
) -> None:
    completed = {
        (row["model"], row["example_id"], row["condition"])
        for row in load_jsonl(NATURAL_PATH)
    }
    rows = []
    for condition, capture in conditions:
        margins, scores, rms = natural_summary(capture, probes)
        for index, record in enumerate(records):
            key = (model_name, record["example_id"], condition)
            if key in completed:
                continue
            rows.append(
                {
                    **common_row(
                        model_name, record, probe_names, commit, run_id
                    ),
                    "record_type": "natural_endpoint",
                    "condition": condition,
                    "response_token_count": int(capture.response_mask[index].sum()),
                    **effect_payload(margins, scores, rms, index),
                }
            )
            completed.add(key)
    append_jsonl(NATURAL_PATH, rows)


def run_total_jobs(
    condition: Any,
    jobs: Sequence[Any],
    vector: VectorizedMechanismRunner,
    *,
    chunk_size: int,
    start_layer: int,
) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
    result = {}
    prefix_cache: dict[int, Tensor] = {}
    for chunk in batched(tuple(jobs), chunk_size):
        if len(chunk) not in prefix_cache:
            prefix_cache[len(chunk)] = capture_layer_input(
                vector.runner,
                repeat_condition(condition, len(chunk)),
                layer=start_layer,
            )
        output = vector.run_from_layer(
            condition,
            chunk,
            start_layer=start_layer,
            cached_input=prefix_cache[len(chunk)],
        )
        for index, group_id in enumerate(output.group_ids):
            result[group_id] = (
                output.mean_margins[index],
                output.sequence_scores[index],
                output.activation_rms[index],
            )
    return result


def direct_summary(
    target: Any,
    replacements: Mapping[str, Tensor],
    runner: PairedInterventionRunner,
    probes: Sequence[LinearProbe],
    *,
    target_recomputations: Mapping[int, Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    patched = direct_path_monitor(
        target,
        direct_replacement_cache(
            target,
            replacements,
            runner.layers,
            target_recomputations=target_recomputations,
        ),
    )
    return capture_summary(patched, probes)


def absolute_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return row["model"], row["example_id"], row["operator"], row["path"]


def random_key(row: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        row["model"],
        row["example_id"],
        int(row["draw_index"]),
        row["direction"],
        row["path"],
    )


def run_absolute_random(
    model_name: str,
    contract: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    chunk_size: int,
    commit: str,
    run_id: str,
) -> None:
    runner = load_model(contract, model_name)
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    names, probes = load_probes()
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    ids = component_ids(contract)
    mean_artifact = torch.load(
        mean_stats_path(model_name), map_location="cpu", weights_only=False
    )
    completed_absolute = {absolute_key(row) for row in load_jsonl(ABSOLUTE_PATH)}
    completed_random = {random_key(row) for row in load_jsonl(RANDOM_PATH)}
    for group in group_records(records):
        for batch_records in batched(group, batch_size):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in batch_records],
                [row["response"] for row in batch_records],
                batch_records[0]["trigger_concept"],
            )
            normal = realized.run(pair.normal)
            triggered = realized.run(pair.triggered)
            write_natural(
                model_name,
                batch_records,
                (("normal", normal), ("correct_trigger", triggered)),
                names,
                probes,
                commit,
                run_id,
            )
            normal_values = capture_head_tensor(normal, ids, runner.layers)
            mean_values = loo_mean_tensor(
                batch_records, normal_values, normal.response_mask, mean_artifact
            )
            replacements = {
                "N_zero": (pair.normal, normal, zero_replacements(normal, ids, runner.layers)),
                "T_zero": (
                    pair.triggered,
                    triggered,
                    zero_replacements(triggered, ids, runner.layers),
                ),
                "N_from_T": (
                    pair.normal,
                    normal,
                    source_replacements(normal, triggered, ids, runner.layers),
                ),
                "T_from_N": (
                    pair.triggered,
                    triggered,
                    source_replacements(triggered, normal, ids, runner.layers),
                ),
                "N_normal_concept_mean": (
                    pair.normal,
                    normal,
                    mean_replacements(normal, ids, mean_values, runner.layers),
                ),
                "T_normal_concept_mean": (
                    pair.triggered,
                    triggered,
                    mean_replacements(triggered, ids, mean_values, runner.layers),
                ),
            }
            for condition, target in ((pair.normal, normal), (pair.triggered, triggered)):
                target_recomputations = direct_path_target_recomputations(
                    target, runner.layers, (9, 10, 11, 12)
                )
                selected = {
                    name: values
                    for name, (base, _target, values) in replacements.items()
                    if base is condition
                }
                jobs = [
                    transplant_job_from_cache(
                        name, total_replacement_cache(target, values, runner.layers)
                    )
                    for name, values in selected.items()
                ]
                totals = run_total_jobs(
                    condition,
                    jobs,
                    vector,
                    chunk_size=chunk_size,
                    start_layer=9,
                )
                for operator, values in selected.items():
                    for path, summary in (
                        ("total", totals[operator]),
                        (
                            "direct",
                            direct_summary(
                                target,
                                values,
                                runner,
                                probes,
                                target_recomputations=target_recomputations,
                            ),
                        ),
                    ):
                        rows = []
                        margins, scores, rms = summary
                        for index, record in enumerate(batch_records):
                            key = (model_name, record["example_id"], operator, path)
                            if key in completed_absolute:
                                continue
                            rows.append(
                                {
                                    **common_row(model_name, record, names, commit, run_id),
                                    "record_type": "absolute_contribution_effect",
                                    "operator": operator,
                                    "target_condition": target.condition,
                                    "path": path,
                                    **effect_payload(margins, scores, rms, index),
                                }
                            )
                            completed_absolute.add(key)
                        append_jsonl(ABSOLUTE_PATH, rows)

            if int(batch_records[0]["label"]) == 1:
                for direction, condition, target in (
                    ("induction", pair.normal, normal),
                    ("rescue", pair.triggered, triggered),
                ):
                    target_recomputations = direct_path_target_recomputations(
                        target, runner.layers, (9, 10, 11, 12)
                    )
                    random_values = {}
                    invariant_audits = {}
                    for draw in range(32):
                        values, audit = random_control_replacements(
                            target,
                            normal,
                            triggered,
                            ids,
                            runner.layers,
                            direction=direction,
                            draw_index=draw,
                            base_seed=int(contract["inference"]["random_control_seed"]),
                        )
                        random_values[f"draw_{draw:02d}"] = values
                        invariant_audits[draw] = audit.to_dict()
                    jobs = [
                        transplant_job_from_cache(
                            group_id,
                            total_replacement_cache(target, values, runner.layers),
                        )
                        for group_id, values in random_values.items()
                    ]
                    totals = run_total_jobs(
                        condition,
                        jobs,
                        vector,
                        chunk_size=chunk_size,
                        start_layer=9,
                    )
                    for draw in range(32):
                        group_id = f"draw_{draw:02d}"
                        for path, summary in (
                            ("total", totals[group_id]),
                            (
                                "direct",
                                direct_summary(
                                    target,
                                    random_values[group_id],
                                    runner,
                                    probes,
                                    target_recomputations=target_recomputations,
                                ),
                            ),
                        ):
                            margins, scores, rms = summary
                            rows = []
                            for index, record in enumerate(batch_records):
                                key = (
                                    model_name,
                                    record["example_id"],
                                    draw,
                                    direction,
                                    path,
                                )
                                if key in completed_random:
                                    continue
                                rows.append(
                                    {
                                        **common_row(
                                            model_name, record, names, commit, run_id
                                        ),
                                        "record_type": "matched_random_effect",
                                        "draw_index": draw,
                                        "direction": direction,
                                        "target_condition": target.condition,
                                        "path": path,
                                        "haar_invariants": invariant_audits[draw],
                                        **effect_payload(margins, scores, rms, index),
                                    }
                                )
                                completed_random.add(key)
                            append_jsonl(RANDOM_PATH, rows)
            if runner.registered_hook_count() != 0:
                raise RuntimeError("absolute/random execution leaked hooks")
            release_memory()
    del runner
    release_memory()


def frontier_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["model"],
        row["example_id"],
        row["evaluation_scope"],
        row["source_family"],
        int(row["source_layer"]),
        row["configuration_role"],
        row["direction"],
    )


def frontier_jobs(
    target: Any,
    source: Any,
    runner: PairedInterventionRunner,
    ids_by_layer: Mapping[int, Sequence[str]],
    configurations_by_layer: Mapping[int, Sequence[tuple[str, Any]]],
    source_family: str,
) -> list[tuple[str, int, Any, Mapping[str, Tensor], Any]]:
    jobs = []
    for layer, configurations in configurations_by_layer.items():
        ids = ids_by_layer[layer]
        values = source_replacements(target, source, ids, runner.layers)
        for role, configuration in configurations:
            group_id = f"{source_family}.layer_{layer:02d}.{role}"
            job = transplant_job_from_cache(
                group_id,
                frontier_patch_cache(target, values, runner.layers, configuration),
            )
            jobs.append((group_id, layer, role, configuration, values, job))
    return jobs


def run_frontier_population(
    model_name: str,
    contract: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    evaluation_scope: str,
    configurations_by_layer: Mapping[int, Sequence[tuple[str, Any]]],
    batch_size: int,
    chunk_size: int,
    commit: str,
    run_id: str,
) -> None:
    runner = load_model(contract, model_name)
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    names, probes = load_probes()
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    families = {
        "selected": components_by_layer(component_ids(contract, "selected")),
        "nonselected": components_by_layer(component_ids(contract, "nonselected")),
    }
    completed = {frontier_key(row) for row in load_jsonl(FRONTIER_PATH)}
    for group in group_records(records):
        for batch_records in batched(group, batch_size):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in batch_records],
                [row["response"] for row in batch_records],
                batch_records[0]["trigger_concept"],
            )
            normal = realized.run(pair.normal)
            triggered = realized.run(pair.triggered)
            write_natural(
                model_name,
                batch_records,
                (("normal", normal), ("correct_trigger", triggered)),
                names,
                probes,
                commit,
                run_id,
            )
            for direction, condition, target, source in (
                ("induction", pair.normal, normal, triggered),
                ("rescue", pair.triggered, triggered, normal),
            ):
                metadata = []
                for family, by_layer in families.items():
                    metadata.extend(
                        frontier_jobs(
                            target,
                            source,
                            runner,
                            by_layer,
                            configurations_by_layer,
                            family,
                        )
                    )
                outputs = run_total_jobs(
                    condition,
                    [row[5] for row in metadata],
                    vector,
                    chunk_size=chunk_size,
                    start_layer=min(configurations_by_layer),
                )
                for group_id, layer, role, configuration, _values, _job in metadata:
                    margins, scores, rms = outputs[group_id]
                    rows = []
                    family = group_id.split(".", 1)[0]
                    for index, record in enumerate(batch_records):
                        key = (
                            model_name,
                            record["example_id"],
                            evaluation_scope,
                            family,
                            layer,
                            role,
                            direction,
                        )
                        if key in completed:
                            continue
                        rows.append(
                            {
                                **common_row(model_name, record, names, commit, run_id),
                                "record_type": "downstream_frontier_effect",
                                "evaluation_scope": evaluation_scope,
                                "source_family": family,
                                "source_layer": layer,
                                "configuration_role": role,
                                "frontier_id": configuration.frontier_id,
                                "released_branches": [
                                    branch.branch_id
                                    for branch in configuration.released
                                ],
                                "frozen_branches": [
                                    branch.branch_id for branch in configuration.frozen
                                ],
                                "direction": direction,
                                "path": "progressive_release",
                                **effect_payload(margins, scores, rms, index),
                            }
                        )
                        completed.add(key)
                    append_jsonl(FRONTIER_PATH, rows)
            if runner.registered_hook_count() != 0:
                raise RuntimeError("frontier execution leaked hooks")
            release_memory()
    del runner
    release_memory()


def attention_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["model"],
        row["example_id"],
        row["evaluation_scope"],
        row["site_id"],
        row["operation"],
        row["direction"],
        row["path"],
    )


def prompt_alignment(pair: PairedBatch) -> tuple[Any, ...]:
    alignments = align_paired_prompts(pair)
    return tuple(
        (alignment.normal_prompt_positions, alignment.triggered_prompt_positions)
        for alignment in alignments
    )


def run_attention_population(
    model_name: str,
    contract: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    evaluation_scope: str,
    sites: Sequence[tuple[str, Sequence[str]]],
    batch_size: int,
    chunk_size: int,
    commit: str,
    run_id: str,
) -> None:
    runner = load_model(contract, model_name)
    realized = RealizedForwardRunner(runner, monitor_layer=12)
    attention_capture = AttentionStateCaptureRunner(runner, monitor_layer=12)
    names, probes = load_probes()
    vector = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    completed = {attention_key(row) for row in load_jsonl(ATTENTION_PATH)}
    layers = sorted(
        {
            MechanismComponent.parse(component_id).layer
            for _site, members in sites
            for component_id in members
        }
    )
    for group in group_records(records):
        for batch_records in batched(group, batch_size):
            pair = runner.prepare_pairs(
                [row["prompt"] for row in batch_records],
                [row["response"] for row in batch_records],
                batch_records[0]["trigger_concept"],
            )
            normal = realized.run(pair.normal)
            triggered = realized.run(pair.triggered)
            normal_attention = attention_capture.run(pair.normal, layers)
            triggered_attention = attention_capture.run(pair.triggered, layers)
            write_natural(
                model_name,
                batch_records,
                (("normal", normal), ("correct_trigger", triggered)),
                names,
                probes,
                commit,
                run_id,
            )
            aligned = prompt_alignment(pair)
            normal_to_triggered = align_attention_indices(
                pair.normal, pair.triggered, aligned
            )
            triggered_to_normal = align_attention_indices(
                pair.triggered,
                pair.normal,
                tuple((right, left) for left, right in aligned),
            )
            normal_partition = build_source_mask_partition(
                runner.tokenizer,
                pair.normal,
                [row["prompt"] for row in batch_records],
                trigger=None,
            )
            triggered_partition = build_source_mask_partition(
                runner.tokenizer,
                pair.triggered,
                [row["prompt"] for row in batch_records],
                trigger=batch_records[0]["trigger_concept"],
            )
            normal_concept = normal_partition.masks[SourceRegion.NAMED_CONCEPT]
            triggered_concept = triggered_partition.masks[SourceRegion.NAMED_CONCEPT]
            for direction, condition, target, source, target_states, source_states, indices, target_mask, source_mask in (
                (
                    "induction",
                    pair.normal,
                    normal,
                    triggered,
                    normal_attention,
                    triggered_attention,
                    triggered_to_normal,
                    normal_concept,
                    triggered_concept,
                ),
                (
                    "rescue",
                    pair.triggered,
                    triggered,
                    normal,
                    triggered_attention,
                    normal_attention,
                    normal_to_triggered,
                    triggered_concept,
                    normal_concept,
                ),
            ):
                target_recomputations = direct_path_target_recomputations(
                    target, runner.layers, layers
                )
                metadata = []
                for site_id, members in sites:
                    layer = MechanismComponent.parse(members[0]).layer
                    for operation in OPERATIONS:
                        replacements = attention_operation_replacements(
                            source_states[layer],
                            target_states[layer],
                            members,
                            indices,
                            operation=operation,
                            source_concept_mask=source_mask,
                            target_concept_mask=target_mask,
                        )
                        group_id = f"{site_id}.{operation}"
                        job = transplant_job_from_cache(
                            group_id,
                            total_replacement_cache(target, replacements, runner.layers),
                        )
                        metadata.append(
                            (group_id, site_id, operation, members, replacements, job)
                        )
                totals = run_total_jobs(
                    condition,
                    [row[5] for row in metadata],
                    vector,
                    chunk_size=chunk_size,
                    start_layer=min(layers),
                )
                for group_id, site_id, operation, members, replacements, _job in metadata:
                    for path, summary in (
                        ("total", totals[group_id]),
                        (
                            "direct",
                            direct_summary(
                                target,
                                replacements,
                                runner,
                                probes,
                                target_recomputations=target_recomputations,
                            ),
                        ),
                    ):
                        margins, scores, rms = summary
                        rows = []
                        for index, record in enumerate(batch_records):
                            key = (
                                model_name,
                                record["example_id"],
                                evaluation_scope,
                                site_id,
                                operation,
                                direction,
                                path,
                            )
                            if key in completed:
                                continue
                            rows.append(
                                {
                                    **common_row(
                                        model_name, record, names, commit, run_id
                                    ),
                                    "record_type": "attention_operation_effect",
                                    "evaluation_scope": evaluation_scope,
                                    "site_id": site_id,
                                    "component_ids": list(members),
                                    "operation": operation,
                                    "direction": direction,
                                    "target_condition": target.condition,
                                    "path": path,
                                    "token_source_partition": {
                                        key: int(value[index].sum())
                                        for key, value in {
                                            "concept_span": target_mask,
                                            "monitoring_language": (
                                                triggered_partition
                                                if target.condition == "correct_trigger"
                                                else normal_partition
                                            ).masks[SourceRegion.MONITORING_LANGUAGE],
                                            "original_prompt": (
                                                triggered_partition
                                                if target.condition == "correct_trigger"
                                                else normal_partition
                                            ).masks[SourceRegion.ORIGINAL_PROMPT],
                                            "template_or_punctuation": (
                                                triggered_partition
                                                if target.condition == "correct_trigger"
                                                else normal_partition
                                            ).masks[SourceRegion.TEMPLATE],
                                            "previous_response": torch.tensor(
                                                [
                                                    int(mask.sum())
                                                    * (int(mask.sum()) - 1)
                                                    // 2
                                                    for mask in target.response_mask
                                                ]
                                            ),
                                        }.items()
                                    },
                                    **effect_payload(margins, scores, rms, index),
                                }
                            )
                            completed.add(key)
                        append_jsonl(ATTENTION_PATH, rows)
            if runner.registered_hook_count() != 0:
                raise RuntimeError("attention operation execution leaked hooks")
            release_memory()
    del runner
    release_memory()


def own_probe_effect(
    row: Mapping[str, Any], natural: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> float:
    probe_index = row["probe_names"].index(row["concept"])
    target_condition = "normal" if row["direction"] == "induction" else "correct_trigger"
    baseline = natural[(row["model"], row["example_id"], target_condition)]
    patched = float(row["mean_raw_margins"][probe_index])
    natural_margin = float(baseline["mean_raw_margins"][probe_index])
    return natural_margin - patched if row["direction"] == "induction" else patched - natural_margin


def equal_concept_mean(rows: Sequence[Mapping[str, Any]], values: Sequence[float]) -> float:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_concept[row["concept"]].append(value)
    return sum(sum(cell) / len(cell) for cell in by_concept.values()) / len(by_concept)


def select_development(commit: str, run_id: str) -> None:
    probe_scale = load_probe_scale()
    natural = {
        (row["model"], row["example_id"], row["condition"]): row
        for row in load_jsonl(NATURAL_PATH)
    }
    frontier = [
        row
        for row in load_jsonl(FRONTIER_PATH)
        if row["evaluation_scope"] == "discovery"
        and row["model"] == "chameleon"
        and row["source_family"] == "selected"
    ]
    frontier_cells: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in frontier:
        frontier_cells[(int(row["source_layer"]), row["configuration_role"])].append(row)
    frontier_scores = {}
    for (layer, configuration), rows in frontier_cells.items():
        by_example = defaultdict(dict)
        for row in rows:
            by_example[row["example_id"]][row["direction"]] = own_probe_effect(row, natural)
        direct = frontier_cells[(layer, "F0")]
        total_id = (
            f"F{len(frontier_configurations(layer)[0].later_branches)}"
        )
        total = frontier_cells[(layer, total_id)]
        direct_by = defaultdict(dict)
        total_by = defaultdict(dict)
        for row in direct:
            direct_by[row["example_id"]][row["direction"]] = own_probe_effect(row, natural)
        for row in total:
            total_by[row["example_id"]][row["direction"]] = own_probe_effect(row, natural)
        values = []
        valid_rows = []
        for row in rows:
            example_id = row["example_id"]
            direction = row["direction"]
            denominator = total_by[example_id][direction] - direct_by[example_id][direction]
            values.append(
                (by_example[example_id][direction] - direct_by[example_id][direction])
                / (denominator if abs(denominator) >= 0.1 else (0.1 if denominator >= 0 else -0.1))
            )
            valid_rows.append(row)
        frontier_scores[(layer, configuration)] = equal_concept_mean(valid_rows, values)
    selected_frontier = sorted(
        frontier_scores,
        key=lambda key: (
            -frontier_scores[key],
            len(
                next(
                    item.released
                    for item in frontier_configurations(key[0])
                    if item.frontier_id == key[1]
                )
            ),
            key[0],
            key[1],
        ),
    )[0]

    attention = [
        row
        for row in load_jsonl(ATTENTION_PATH)
        if row["evaluation_scope"] == "discovery"
        and row["model"] == "chameleon"
        and row["path"] == "total"
    ]
    natural_vectors = {}
    for row in attention:
        normal = natural[("chameleon", row["example_id"], "normal")]
        triggered = natural[("chameleon", row["example_id"], "correct_trigger")]
        natural_vectors[row["example_id"]] = (
            torch.tensor(normal["mean_raw_margins"])
            - torch.tensor(triggered["mean_raw_margins"])
        ) / probe_scale
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in attention:
        cells[(row["site_id"], row["operation"])].append(row)
    attention_scores = {}
    for cell, rows in cells.items():
        recoveries = []
        for row in rows:
            target = natural[
                (
                    "chameleon",
                    row["example_id"],
                    "normal" if row["direction"] == "induction" else "correct_trigger",
                )
            ]
            patched = torch.tensor(row["mean_raw_margins"])
            baseline = torch.tensor(target["mean_raw_margins"])
            movement = (
                baseline - patched
                if row["direction"] == "induction"
                else patched - baseline
            ) / probe_scale
            natural_vector = natural_vectors[row["example_id"]]
            recoveries.append(
                float(
                    1.0
                    - (movement - natural_vector).square().sum()
                    / natural_vector.square().sum().clamp(min=1e-6)
                )
            )
        attention_scores[cell] = equal_concept_mean(rows, recoveries)
    per_site = {}
    best_operation = {}
    for site, operation in cells:
        score = attention_scores[(site, operation)]
        if site not in per_site or score > per_site[site] or (
            score == per_site[site] and operation < best_operation[site]
        ):
            per_site[site] = score
            best_operation[site] = operation
    selected_site = sorted(
        per_site,
        key=lambda site: (
            -per_site[site],
            (
                1
                if site.startswith("individual")
                else len(
                    next(
                        row["component_ids"] for row in attention if row["site_id"] == site
                    )
                )
            ),
            int(site.split("layer_")[1][:2]),
            site,
        ),
    )[0]
    write_json(
        SELECTION_PATH,
        {
            "schema_version": 1,
            "procedure": "frozen discovery-only Phase B selection",
            "execution_commit": commit,
            "execution_id": run_id,
            "selected_frontier": {
                "source_layer": selected_frontier[0],
                "frontier_id": selected_frontier[1],
                "equal_concept_remainder_recovery": frontier_scores[
                    selected_frontier
                ],
            },
            "selected_attention_site": {
                "site_id": selected_site,
                "best_operation": best_operation[selected_site],
                "equal_concept_complete_vector_recovery": per_site[selected_site],
            },
            "frontier_scores": {
                f"layer_{key[0]:02d}.{key[1]}": value
                for key, value in sorted(frontier_scores.items())
            },
            "attention_site_scores": dict(sorted(per_site.items())),
            "validation_or_safety_rows_used": False,
        },
    )


def execution_parameters(
    args: argparse.Namespace, commit: str, run_id: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_commit": commit,
        "execution_id": run_id,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "row_clarification_sha256": sha256_file(CLARIFICATION_PATH),
        "attention_operator_sha256": sha256_file(ATTENTION_FREEZE_PATH),
        "batch_size": args.batch_size,
        "job_chunk_size": args.job_chunk_size,
        "limit_per_cell": args.limit_per_cell,
        "outcomes_accessed_before_execution_commit": False,
    }


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        CONTRACT_PATH,
        CLARIFICATION_PATH,
        ATTENTION_FREEZE_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    clarification = read_json(CLARIFICATION_PATH)
    attention_freeze = read_json(ATTENTION_FREEZE_PATH)
    if (
        contract["status"] != "frozen-before-post-gate1-phase-a-b-outcomes"
        or clarification["status"] != "frozen"
        or attention_freeze["status"] != "frozen"
    ):
        raise RuntimeError("Phase B authority is not frozen")
    run_id = execution_id(commit)
    parameters = execution_parameters(args, commit, run_id)
    if PARAMETERS_PATH.exists():
        existing = read_json(PARAMETERS_PATH)
        for field in ("execution_commit", "execution_id", "batch_size", "job_chunk_size", "limit_per_cell"):
            if existing[field] != parameters[field]:
                raise RuntimeError(f"execution parameter changed across resume: {field}")
    else:
        write_json(PARAMETERS_PATH, parameters)
    records = load_records(args.limit_per_cell)
    if args.limit_per_cell is None and len(records) != 1732:
        raise RuntimeError(f"expected 1732 frozen records, found {len(records)}")
    default_models = {
        "means": ("chameleon", "precursor"),
        "absolute-random": ("chameleon", "precursor"),
        "frontier-discovery": ("chameleon",),
        "attention-discovery": ("chameleon",),
        "frontier-eval": ("chameleon", "precursor"),
        "attention-eval": ("chameleon", "precursor"),
    }
    if args.stage == "select":
        select_development(commit, run_id)
        return
    models = tuple(args.models or default_models[args.stage])
    if args.stage == "means":
        for model_name in models:
            run_means(
                model_name,
                contract,
                records,
                batch_size=args.batch_size,
                commit=commit,
                run_id=run_id,
            )
    elif args.stage == "absolute-random":
        for model_name in models:
            run_absolute_random(
                model_name,
                contract,
                records,
                batch_size=args.batch_size,
                chunk_size=args.job_chunk_size,
                commit=commit,
                run_id=run_id,
            )
    elif args.stage == "frontier-discovery":
        discovery = [
            row
            for row in records
            if row["split"] == "discovery" and int(row["label"]) == 1
        ]
        configurations = {
            layer: tuple(
                (configuration.frontier_id, configuration)
                for configuration in frontier_configurations(layer)
            )
            for layer in (9, 10, 11, 12)
        }
        for model_name in models:
            run_frontier_population(
                model_name,
                contract,
                discovery,
                evaluation_scope="discovery",
                configurations_by_layer=configurations,
                batch_size=args.batch_size,
                chunk_size=args.job_chunk_size,
                commit=commit,
                run_id=run_id,
            )
    elif args.stage == "attention-discovery":
        discovery = [
            row
            for row in records
            if row["split"] == "discovery" and int(row["label"]) == 1
        ]
        sites = attention_sites(contract["component_sets"]["layer_groups"])
        for model_name in models:
            run_attention_population(
                model_name,
                contract,
                discovery,
                evaluation_scope="discovery",
                sites=sites,
                batch_size=args.batch_size,
                chunk_size=args.job_chunk_size,
                commit=commit,
                run_id=run_id,
            )
    elif args.stage in {"frontier-eval", "attention-eval"}:
        selection = read_json(SELECTION_PATH)
        evaluation = [
            row
            for row in records
            if int(row["label"]) == 0
            or (row["split"] != "discovery" and int(row["label"]) == 1)
        ]
        for model_name in models:
            if args.stage == "frontier-eval":
                layer = int(selection["selected_frontier"]["source_layer"])
                selected_id = selection["selected_frontier"]["frontier_id"]
                all_configurations = frontier_configurations(layer)
                total_id = f"F{len(all_configurations[0].later_branches)}"
                role_ids = ("F0", selected_id, total_id)
                configurations = {
                    layer: tuple(
                        (
                            role,
                            next(
                                item
                                for item in all_configurations
                                if item.frontier_id == frontier_id
                            ),
                        )
                        for role, frontier_id in zip(
                            ("direct", "selected", "total"), role_ids, strict=True
                        )
                    )
                }
                run_frontier_population(
                    model_name,
                    contract,
                    evaluation,
                    evaluation_scope="heldout_or_negative",
                    configurations_by_layer=configurations,
                    batch_size=args.batch_size,
                    chunk_size=args.job_chunk_size,
                    commit=commit,
                    run_id=run_id,
                )
            else:
                site_id = selection["selected_attention_site"]["site_id"]
                site = next(
                    value
                    for value in attention_sites(
                        contract["component_sets"]["layer_groups"]
                    )
                    if value[0] == site_id
                )
                run_attention_population(
                    model_name,
                    contract,
                    evaluation,
                    evaluation_scope="heldout_or_negative",
                    sites=(site,),
                    batch_size=args.batch_size,
                    chunk_size=args.job_chunk_size,
                    commit=commit,
                    run_id=run_id,
                )


if __name__ == "__main__":
    main()
