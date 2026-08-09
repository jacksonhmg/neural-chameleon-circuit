#!/usr/bin/env python3
"""Run resumable complete-corpus Gate 1 accounting and acquisition jobs."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import (  # noqa: E402
    LinearProbe,
    MechanismComponent,
    PairedInterventionRunner,
    RealizedForwardRunner,
    VectorizedMechanismRunner,
    audit_realized_forward,
    capture_layer_input,
    direct_path_monitor,
    direct_path_patch_cache,
    direct_path_target_recomputations,
    load_experimental_split,
    probe_sequence_scores,
    probe_token_margins,
    repeat_condition,
    reconstruct_residual_after,
    total_patch_cache,
    transplant_job_from_cache,
    writer_delta,
)


PLAN_PATH = ROOT / "results/day-31/frozen-acquired-writer-plan.json"
RESULT_DIR = ROOT / "results/day-33"
ARTIFACT_DIR = ROOT / "artifacts/mechanism-gate1-v1"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
ACCOUNTING_WORKING = RESULT_DIR / "accounting.working.jsonl"
NATURAL_WORKING = RESULT_DIR / "natural-endpoints.working.jsonl"
EFFECT_WORKING = RESULT_DIR / "component-effects.working.jsonl"
FUNCTIONAL_WORKING = RESULT_DIR / "precursor-functional.working.jsonl"
EXECUTION_PARAMETERS_PATH = RESULT_DIR / "execution-parameters.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("chameleon", "precursor"),
        default=["chameleon", "precursor"],
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--group-chunk-size", type=int, default=2)
    parser.add_argument("--limit-per-concept", type=int)
    parser.add_argument("--skip-component-effects", action="store_true")
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


def batched(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for record in records:
            handle.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return records


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def model_config(plan: Mapping[str, Any], model_name: str) -> Mapping[str, Any]:
    return plan["models"][
        "chameleon" if model_name == "chameleon" else "exact_precursor"
    ]


def load_model(plan: Mapping[str, Any], model_name: str) -> PairedInterventionRunner:
    config = model_config(plan, model_name)
    path = ROOT / config["local_path"]
    torch.manual_seed(int(plan["inference"]["audit_seed"]))
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation=plan["models"]["attention_implementation"],
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


def load_records(limit_per_concept: int | None) -> list[dict[str, Any]]:
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
    if limit_per_concept is None:
        return records
    if limit_per_concept <= 0:
        raise ValueError("limit-per-concept must be positive")
    selected = []
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for record in records:
        key = (record["concept"], int(record["label"]))
        if counts[key] < limit_per_concept:
            selected.append(record)
            counts[key] += 1
    return selected


def group_records(records: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["concept"], record["trigger_concept"])].append(record)
    return [groups[key] for key in sorted(groups)]


def component_groups(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    sets = plan["component_sets"]
    groups = []
    for component_id in sets["k16_ordered"]:
        groups.append(
            {
                "group_id": f"individual.{component_id}",
                "family": "individual_selected_component",
                "component_ids": [component_id],
            }
        )
    for group_id, component_ids in sets["nested_selected_head_groups"].items():
        groups.append(
            {
                "group_id": f"nested_heads.{group_id}",
                "family": "nested_selected_heads",
                "component_ids": list(component_ids),
            }
        )
    groups.extend(
        [
            {
                "group_id": "selected_mlps.K4",
                "family": "selected_mlp_population",
                "component_ids": list(sets["selected_mlps_k4"]),
            },
            {
                "group_id": "selected_components.K16",
                "family": "selected_k16_population",
                "component_ids": list(sets["k16_ordered"]),
            },
        ]
    )
    for layer_id, component_ids in sets["layer_groups"].items():
        groups.append(
            {
                "group_id": f"selected_heads.{layer_id}",
                "family": "selected_head_layer_group",
                "component_ids": list(component_ids),
            }
        )
    if len(groups) != 27 or len({row["group_id"] for row in groups}) != 27:
        raise ValueError(
            f"expected 27 unique frozen component groups, found {len(groups)}"
        )
    for group in groups:
        components = [
            MechanismComponent.parse(value) for value in group["component_ids"]
        ]
        if len(components) != len({item.component_id for item in components}):
            raise ValueError(f"duplicate component in {group['group_id']}")
    return groups


def load_probes() -> tuple[tuple[str, ...], tuple[LinearProbe, ...]]:
    paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    names = tuple(path.name.removesuffix("_weights.pt") for path in paths)
    probes = tuple(LinearProbe.load(path) for path in paths)
    if len(probes) != 13 or len(set(names)) != 13:
        raise ValueError("expected the complete 13-probe family")
    return names, probes


def capture_summary(
    capture: Any, probes: Sequence[LinearProbe]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    margins = probe_token_margins(capture.monitor_residual, probes)
    mask = capture.response_mask.unsqueeze(0)
    mean_margins = (margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
    scores = probe_sequence_scores(margins, capture.response_mask)
    token_rms = capture.monitor_residual.values.float().square().mean(dim=-1).sqrt()
    activation_rms = (token_rms * capture.response_mask).sum(
        dim=1
    ) / capture.response_mask.sum(dim=1).clamp(min=1)
    return mean_margins.T, scores.T, activation_rms


def accounting_batch_id(
    model_name: str, condition: str, records: Sequence[dict[str, Any]]
) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["example_id"].encode())
        digest.update(b"\0")
    return f"{model_name}.{condition}.{digest.hexdigest()[:16]}"


def feature_path(model_name: str, example_id: str) -> Path:
    return ARTIFACT_DIR / model_name / f"{example_id}.pt"


def save_feature_batch(
    model_name: str,
    records: Sequence[dict[str, Any]],
    normal: Any,
    triggered: Any,
    plan: Mapping[str, Any],
    layers: Sequence[torch.nn.Module],
    execution_commit: str,
    execution_id: str,
) -> None:
    positive_rows = [
        index for index, record in enumerate(records) if int(record["label"]) == 1
    ]
    if not positive_rows:
        return
    k12 = tuple(plan["component_sets"]["k12_ordered"])
    k12_delta = writer_delta(triggered, normal, k12, layers)
    nonselected = tuple(plan["component_sets"]["layer_count_matched_nonselected_heads"])
    nonselected_delta = (
        writer_delta(triggered, normal, nonselected, layers)
        if model_name == "chameleon"
        else None
    )
    normal_state = reconstruct_residual_after(normal, 8)
    target_u = (
        triggered.monitor_residual.values.float()
        - normal.monitor_residual.values.float()
    )
    for row in positive_rows:
        record = records[row]
        path = feature_path(model_name, record["example_id"])
        if path.exists():
            continue
        mask = normal.response_mask[row].bool()
        artifact = {
            "schema_version": 1,
            "execution_commit": execution_commit,
            "execution_id": execution_id,
            "model": model_name,
            "example_id": record["example_id"],
            "concept": record["concept"],
            "split": record["split"],
            "response_ids": normal.response_ids[row, mask].clone(),
            "k12_head_ids": k12,
            "k12_delta": torch.stack(
                [k12_delta[head_id][row, mask] for head_id in k12], dim=1
            ).float(),
            "normal_resid_post_8": normal_state[row, mask].to(torch.bfloat16),
            "target_u": target_u[row, mask].float(),
            "numeric_dtypes": {
                "writer_deltas": "float32 difference of realized BF16 states",
                "normal_resid_post_8": "bfloat16 realized residual",
                "target_u": "float32 difference of realized BF16 states",
            },
        }
        if nonselected_delta is not None:
            artifact["nonselected_head_ids"] = nonselected
            artifact["nonselected_delta"] = torch.stack(
                [nonselected_delta[head_id][row, mask] for head_id in nonselected],
                dim=1,
            ).float()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(artifact, temporary)
        temporary.replace(path)


def effect_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    return {
        (
            row["model"],
            row["example_id"],
            row["direction"],
            row["path"],
            row["group_id"],
        )
        for row in load_jsonl(path)
    }


def natural_keys(path: Path) -> set[tuple[str, str, str]]:
    return {
        (row["model"], row["example_id"], row["condition"]) for row in load_jsonl(path)
    }


def write_natural_rows(
    model_name: str,
    records: Sequence[dict[str, Any]],
    condition: str,
    capture: Any,
    probe_names: Sequence[str],
    probes: Sequence[LinearProbe],
    completed: set[tuple[str, str, str]],
    execution_commit: str,
    execution_id: str,
) -> None:
    margins, scores, rms = capture_summary(capture, probes)
    rows = []
    for index, record in enumerate(records):
        key = (model_name, record["example_id"], condition)
        if key in completed:
            continue
        rows.append(
            {
                "schema_version": 1,
                "execution_commit": execution_commit,
                "execution_id": execution_id,
                "record_type": "natural_endpoint",
                "model": model_name,
                "split": record["split"],
                "concept": record["concept"],
                "trigger_concept": record["trigger_concept"],
                "label": int(record["label"]),
                "example_id": record["example_id"],
                "condition": condition,
                "response_token_count": int(capture.response_mask[index].sum()),
                "probe_names": list(probe_names),
                "mean_raw_margins": [float(value) for value in margins[index]],
                "sequence_scores": [float(value) for value in scores[index]],
                "activation_rms": float(rms[index]),
            }
        )
        completed.add(key)
    append_jsonl(NATURAL_WORKING, rows)


def run_component_family(
    model_name: str,
    direction: str,
    path_kind: str,
    base_condition: Any,
    target: Any,
    source: Any,
    records: Sequence[dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    vector_runner: VectorizedMechanismRunner,
    probe_names: Sequence[str],
    completed: set[tuple[str, str, str, str, str]],
    execution_commit: str,
    execution_id: str,
    *,
    chunk_size: int,
) -> None:
    missing_groups = [
        group
        for group in groups
        if any(
            (model_name, record["example_id"], direction, path_kind, group["group_id"])
            not in completed
            for record in records
        )
    ]
    if path_kind == "direct_path":
        head_layers = sorted(
            {
                component.layer
                for group in missing_groups
                for value in group["component_ids"]
                for component in (MechanismComponent.parse(value),)
                if component.kind == "head"
            }
        )
        target_recomputations = direct_path_target_recomputations(
            target, vector_runner.runner.layers, head_layers
        )
        for group in missing_groups:
            components = tuple(
                MechanismComponent.parse(value) for value in group["component_ids"]
            )
            patched = direct_path_monitor(
                target,
                direct_path_patch_cache(
                    target,
                    source,
                    components,
                    vector_runner.runner.layers,
                    monitor_layer=12,
                    target_recomputations=target_recomputations,
                ),
            )
            token_margins = probe_token_margins(patched, vector_runner.probes)
            mask = patched.response_mask.unsqueeze(0)
            mean_margins = (
                (token_margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
            ).T
            scores = probe_sequence_scores(token_margins, patched.response_mask).T
            token_rms = patched.values.float().square().mean(dim=-1).sqrt()
            activation_rms = (token_rms * patched.response_mask).sum(
                dim=1
            ) / patched.response_mask.sum(dim=1).clamp(min=1)
            rows = []
            for example_index, record in enumerate(records):
                key = (
                    model_name,
                    record["example_id"],
                    direction,
                    path_kind,
                    group["group_id"],
                )
                if key in completed:
                    continue
                rows.append(
                    {
                        "schema_version": 1,
                        "execution_commit": execution_commit,
                        "execution_id": execution_id,
                        "record_type": "component_effect",
                        "model": model_name,
                        "split": record["split"],
                        "concept": record["concept"],
                        "label": int(record["label"]),
                        "example_id": record["example_id"],
                        "direction": direction,
                        "path": path_kind,
                        "group_id": group["group_id"],
                        "group_family": group["family"],
                        "component_ids": group["component_ids"],
                        "probe_names": list(probe_names),
                        "mean_raw_margins": [
                            float(value) for value in mean_margins[example_index]
                        ],
                        "sequence_scores": [
                            float(value) for value in scores[example_index]
                        ],
                        "activation_rms": float(activation_rms[example_index]),
                    }
                )
                completed.add(key)
            append_jsonl(EFFECT_WORKING, rows)
        release_memory()
        return
    if path_kind != "total":
        raise ValueError(f"unknown component path: {path_kind}")
    prefix_cache: dict[int, torch.Tensor] = {}
    for chunk in batched(missing_groups, chunk_size):
        jobs = []
        for group in chunk:
            components = tuple(
                MechanismComponent.parse(value) for value in group["component_ids"]
            )
            patch_cache = total_patch_cache(
                source, components, vector_runner.runner.layers
            )
            jobs.append(transplant_job_from_cache(group["group_id"], patch_cache))
        expanded_count = len(jobs)
        if expanded_count not in prefix_cache:
            prefix_cache[expanded_count] = capture_layer_input(
                vector_runner.runner,
                repeat_condition(base_condition, expanded_count),
                layer=9,
            )
        result = vector_runner.run_from_layer(
            base_condition,
            jobs,
            start_layer=9,
            cached_input=prefix_cache[expanded_count],
        )
        rows = []
        by_id = {group["group_id"]: group for group in chunk}
        for job_index, group_id in enumerate(result.group_ids):
            group = by_id[group_id]
            for example_index, record in enumerate(records):
                key = (
                    model_name,
                    record["example_id"],
                    direction,
                    path_kind,
                    group_id,
                )
                if key in completed:
                    continue
                rows.append(
                    {
                        "schema_version": 1,
                        "execution_commit": execution_commit,
                        "execution_id": execution_id,
                        "record_type": "component_effect",
                        "model": model_name,
                        "split": record["split"],
                        "concept": record["concept"],
                        "label": int(record["label"]),
                        "example_id": record["example_id"],
                        "direction": direction,
                        "path": path_kind,
                        "group_id": group_id,
                        "group_family": group["family"],
                        "component_ids": group["component_ids"],
                        "probe_names": list(probe_names),
                        "mean_raw_margins": [
                            float(value)
                            for value in result.mean_margins[job_index, example_index]
                        ],
                        "sequence_scores": [
                            float(value)
                            for value in result.sequence_scores[
                                job_index, example_index
                            ]
                        ],
                        "activation_rms": float(
                            result.activation_rms[job_index, example_index]
                        ),
                    }
                )
                completed.add(key)
        append_jsonl(EFFECT_WORKING, rows)
        release_memory()


def run_model(
    model_name: str,
    plan: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    group_chunk_size: int,
    skip_component_effects: bool,
    execution_commit: str,
    execution_id: str,
) -> None:
    selected_records = (
        records
        if model_name == "chameleon"
        else [record for record in records if int(record["label"]) == 1]
    )
    runner = load_model(plan, model_name)
    realized_runner = RealizedForwardRunner(runner, monitor_layer=12)
    probe_names, probes = load_probes()
    vector_runner = VectorizedMechanismRunner(runner, probes, monitor_layer=12)
    groups = component_groups(plan)
    completed_accounting = {row["batch_id"] for row in load_jsonl(ACCOUNTING_WORKING)}
    completed_natural = natural_keys(NATURAL_WORKING)
    completed_effects = effect_keys(EFFECT_WORKING)
    completed_functional = {row["example_id"] for row in load_jsonl(FUNCTIONAL_WORKING)}

    for concept_records in group_records(selected_records):
        for batch_records in batched(concept_records, batch_size):
            expected_features = [
                feature_path(model_name, record["example_id"])
                for record in batch_records
                if int(record["label"]) == 1
            ]
            accounting_ids = {
                accounting_batch_id(model_name, condition, batch_records)
                for condition in ("normal", "correct_trigger")
            }
            natural_complete = all(
                (model_name, record["example_id"], condition) in completed_natural
                for record in batch_records
                for condition in ("normal", "correct_trigger")
            )
            effect_complete = (
                model_name != "chameleon"
                or skip_component_effects
                or all(
                    (
                        model_name,
                        record["example_id"],
                        direction,
                        path_kind,
                        group["group_id"],
                    )
                    in completed_effects
                    for record in batch_records
                    for direction in ("induction", "rescue")
                    for path_kind in ("total", "direct_path")
                    for group in groups
                )
            )
            functional_complete = model_name != "precursor" or all(
                record["example_id"] in completed_functional for record in batch_records
            )
            if (
                accounting_ids <= completed_accounting
                and natural_complete
                and effect_complete
                and functional_complete
                and all(path.exists() for path in expected_features)
            ):
                continue

            pair = runner.prepare_pairs(
                [record["prompt"] for record in batch_records],
                [record["response"] for record in batch_records],
                batch_records[0]["trigger_concept"],
            )
            normal = realized_runner.run(pair.normal)
            triggered = realized_runner.run(pair.triggered)
            for condition_name, capture in (
                ("normal", normal),
                ("correct_trigger", triggered),
            ):
                batch_id = accounting_batch_id(
                    model_name, condition_name, batch_records
                )
                if batch_id not in completed_accounting:
                    audit = audit_realized_forward(capture, runner.layers, probes)
                    append_jsonl(
                        ACCOUNTING_WORKING,
                        [
                            {
                                "schema_version": 1,
                                "execution_commit": execution_commit,
                                "execution_id": execution_id,
                                "record_type": "accounting_batch",
                                "model": model_name,
                                "batch_id": batch_id,
                                "condition": condition_name,
                                "example_ids": [
                                    record["example_id"] for record in batch_records
                                ],
                                "audit": audit.to_dict(),
                            }
                        ],
                    )
                    completed_accounting.add(batch_id)
                write_natural_rows(
                    model_name,
                    batch_records,
                    condition_name,
                    capture,
                    probe_names,
                    probes,
                    completed_natural,
                    execution_commit,
                    execution_id,
                )
            save_feature_batch(
                model_name,
                batch_records,
                normal,
                triggered,
                plan,
                runner.layers,
                execution_commit,
                execution_id,
            )

            if model_name == "chameleon" and not skip_component_effects:
                run_component_family(
                    model_name,
                    "induction",
                    "total",
                    pair.normal,
                    normal,
                    triggered,
                    batch_records,
                    groups,
                    vector_runner,
                    probe_names,
                    completed_effects,
                    execution_commit,
                    execution_id,
                    chunk_size=group_chunk_size,
                )
                run_component_family(
                    model_name,
                    "induction",
                    "direct_path",
                    pair.normal,
                    normal,
                    triggered,
                    batch_records,
                    groups,
                    vector_runner,
                    probe_names,
                    completed_effects,
                    execution_commit,
                    execution_id,
                    chunk_size=group_chunk_size,
                )
                run_component_family(
                    model_name,
                    "rescue",
                    "total",
                    pair.triggered,
                    triggered,
                    normal,
                    batch_records,
                    groups,
                    vector_runner,
                    probe_names,
                    completed_effects,
                    execution_commit,
                    execution_id,
                    chunk_size=group_chunk_size,
                )
                run_component_family(
                    model_name,
                    "rescue",
                    "direct_path",
                    pair.triggered,
                    triggered,
                    normal,
                    batch_records,
                    groups,
                    vector_runner,
                    probe_names,
                    completed_effects,
                    execution_commit,
                    execution_id,
                    chunk_size=group_chunk_size,
                )
            if model_name == "precursor":
                components = tuple(
                    MechanismComponent.parse(value)
                    for value in plan["component_sets"]["k12_ordered"]
                )
                job = transplant_job_from_cache(
                    "precursor.K12.functional_induction",
                    total_patch_cache(triggered, components, runner.layers),
                )
                functional = vector_runner.run_from_layer(
                    pair.normal,
                    [job],
                    start_layer=9,
                    cached_input=capture_layer_input(
                        runner, repeat_condition(pair.normal, 1), layer=9
                    ),
                )
                rows = []
                for index, record in enumerate(batch_records):
                    if record["example_id"] in completed_functional:
                        continue
                    rows.append(
                        {
                            "schema_version": 1,
                            "execution_commit": execution_commit,
                            "execution_id": execution_id,
                            "record_type": "precursor_functional_induction",
                            "model": "precursor",
                            "split": record["split"],
                            "concept": record["concept"],
                            "example_id": record["example_id"],
                            "probe_names": list(probe_names),
                            "mean_raw_margins": [
                                float(value)
                                for value in functional.mean_margins[0, index]
                            ],
                            "sequence_scores": [
                                float(value)
                                for value in functional.sequence_scores[0, index]
                            ],
                            "activation_rms": float(
                                functional.activation_rms[0, index]
                            ),
                        }
                    )
                    completed_functional.add(record["example_id"])
                append_jsonl(FUNCTIONAL_WORKING, rows)
            if runner.registered_hook_count() != 0:
                raise RuntimeError("one or more execution hooks remained registered")
            print(
                f"{model_name}: completed {batch_records[0]['concept']} "
                f"{batch_records[0]['example_id']}..{batch_records[-1]['example_id']}",
                flush=True,
            )
            release_memory()

    del vector_runner, realized_runner, runner
    release_memory()


def main() -> None:
    args = parse_args()
    commit = git_head()
    require_committed(Path(__file__).resolve(), commit)
    require_committed(PLAN_PATH, commit)
    require_committed(ROOT / "src/neural_chameleon/causal_mechanisms.py", commit)
    plan = json.loads(PLAN_PATH.read_text())
    if plan["procedure"] != "acquired-writer-development-v1":
        raise ValueError("unexpected Gate 1 procedure")
    records = load_records(args.limit_per_concept)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    execution_parameters = {
        "schema_version": 1,
        "execution_commit": commit,
        "batch_size": args.batch_size,
        "group_chunk_size": args.group_chunk_size,
        "limit_per_concept": args.limit_per_concept,
    }
    execution_id = hashlib.sha256(
        json.dumps(
            execution_parameters, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    execution_parameters["execution_id"] = execution_id
    if EXECUTION_PARAMETERS_PATH.exists():
        observed_parameters = json.loads(EXECUTION_PARAMETERS_PATH.read_text())
        if observed_parameters != execution_parameters:
            raise RuntimeError(
                "working results belong to different execution parameters"
            )
    else:
        EXECUTION_PARAMETERS_PATH.write_text(
            json.dumps(execution_parameters, indent=2, sort_keys=True) + "\n"
        )
    for model_name in args.models:
        run_model(
            model_name,
            plan,
            records,
            batch_size=args.batch_size,
            group_chunk_size=args.group_chunk_size,
            skip_component_effects=args.skip_component_effects,
            execution_commit=commit,
            execution_id=execution_id,
        )
    print(
        json.dumps(
            {
                "execution_commit": commit,
                "execution_id": execution_id,
                "plan_sha256": sha256_file(PLAN_PATH),
                "models": args.models,
                "record_count": len(records),
                "limited": args.limit_per_concept is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
