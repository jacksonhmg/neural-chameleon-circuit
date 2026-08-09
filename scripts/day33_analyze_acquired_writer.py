#!/usr/bin/env python3
"""Seal and evaluate the frozen Gate 1 acquired-writer development program."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day33_run_acquired_writer import (  # noqa: E402
    ACCOUNTING_WORKING,
    ARTIFACT_DIR,
    EFFECT_WORKING,
    EXECUTION_PARAMETERS_PATH,
    FUNCTIONAL_WORKING,
    NATURAL_WORKING,
    component_groups,
    group_records,
    load_probes,
    load_records,
)
from neural_chameleon import (  # noqa: E402
    DiagnosticExample,
    build_diagnostic_matrices,
    concept_leave_one_example_out_predictions,
    evaluate_diagnostic_predictions,
    fit_diagnostic_bases,
    fit_probe_standardization,
    fit_weighted_ridge,
    predict_original,
    select_alpha_leave_one_concept_out,
    training_decile_mean_predictions,
    validate_diagnostic_examples,
)


PLAN_PATH = ROOT / "results/day-31/frozen-acquired-writer-plan.json"
RESULT_DIR = ROOT / "results/day-33"
SUMMARY_PATH = RESULT_DIR / "acquired-writer-summary.json"
PREDICTION_PATH = RESULT_DIR / "intermediate-prediction-summary.json"
COMPONENT_PATH = RESULT_DIR / "component-resolution-summary.json"
AUDIT_PATH = RESULT_DIR / "gate-1-audit.json"
MANIFEST_PATH = RESULT_DIR / "execution-artifact-manifest.json"


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
        raise RuntimeError(f"{relative} differs from analysis commit {commit}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def load_feature(model: str, example_id: str) -> dict[str, Any]:
    return torch.load(
        ARTIFACT_DIR / model / f"{example_id}.pt",
        map_location="cpu",
        weights_only=False,
    )


def feature_manifest(model: str, expected_ids: Sequence[str]) -> dict[str, Any]:
    paths = sorted((ARTIFACT_DIR / model).glob("*.pt"))
    actual_ids = [path.stem for path in paths]
    if actual_ids != sorted(expected_ids):
        missing = sorted(set(expected_ids) - set(actual_ids))
        extra = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(f"{model} feature IDs differ: missing={missing}, extra={extra}")
    rows = [
        {
            "example_id": path.stem,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["example_id"].encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "ordered_id_sha256_manifest_sha256": digest.hexdigest(),
        "files": rows,
    }


def expected_accounting_batches(
    records: Sequence[dict[str, Any]], model: str, batch_size: int
) -> int:
    selected = (
        list(records)
        if model == "chameleon"
        else [row for row in records if int(row["label"]) == 1]
    )
    return 2 * sum((len(group) + batch_size - 1) // batch_size for group in group_records(selected))


def online_add(
    accumulators: dict[tuple[Any, ...], dict[str, Any]],
    key: tuple[Any, ...],
    values: Mapping[str, float],
) -> None:
    cell = accumulators.setdefault(
        key,
        {"count": 0, **{name: 0.0 for name in values}},
    )
    cell["count"] += 1
    for name, value in values.items():
        cell[name] += float(value)


def finalize_cells(
    accumulators: Mapping[tuple[Any, ...], Mapping[str, Any]],
    key_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(accumulators):
        cell = accumulators[key]
        count = int(cell["count"])
        rows.append(
            {
                **dict(zip(key_names, key, strict=True)),
                "example_count": count,
                **{
                    name: float(value) / count
                    for name, value in cell.items()
                    if name != "count"
                },
            }
        )
    return rows


def component_resolution(
    effects: Sequence[dict[str, Any]],
    natural: Sequence[dict[str, Any]],
    probe_names: Sequence[str],
    execution_commit: str,
) -> dict[str, Any]:
    natural_by_key = {
        (row["model"], row["example_id"], row["condition"]): row for row in natural
    }
    cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    pending_total: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    remainder_cells: dict[tuple[Any, ...], dict[str, Any]] = {}
    seen = set()
    for row in effects:
        key = (
            row["model"],
            row["example_id"],
            row["direction"],
            row["path"],
            row["group_id"],
        )
        if key in seen:
            raise ValueError(f"duplicate component effect: {key}")
        seen.add(key)
        condition = "normal" if row["direction"] == "induction" else "correct_trigger"
        base = natural_by_key[(row["model"], row["example_id"], condition)]
        margin_effect = np.asarray(row["mean_raw_margins"]) - np.asarray(
            base["mean_raw_margins"]
        )
        score_effect = np.asarray(row["sequence_scores"]) - np.asarray(
            base["sequence_scores"]
        )
        own = probe_names.index(row["concept"])
        cell_key = (
            row["group_id"],
            row["group_family"],
            row["direction"],
            row["path"],
            row["concept"],
            int(row["label"]),
        )
        online_add(
            cells,
            cell_key,
            {
                "mean_own_probe_raw_margin_effect": margin_effect[own],
                "mean_own_probe_sequence_score_effect": score_effect[own],
                "mean_absolute_all_probe_raw_margin_effect": np.abs(
                    margin_effect
                ).mean(),
                "mean_absolute_all_probe_sequence_score_effect": np.abs(
                    score_effect
                ).mean(),
            },
        )
        pair_key = (row["example_id"], row["direction"], row["group_id"])
        if row["path"] == "total":
            pending_total[pair_key] = (margin_effect, score_effect)
        else:
            total_margin, total_score = pending_total.pop(pair_key)
            remainder_margin = total_margin - margin_effect
            remainder_score = total_score - score_effect
            remainder_key = (
                row["group_id"],
                row["group_family"],
                row["direction"],
                row["concept"],
                int(row["label"]),
            )
            online_add(
                remainder_cells,
                remainder_key,
                {
                    "mean_own_probe_raw_margin_remainder": remainder_margin[own],
                    "mean_own_probe_sequence_score_remainder": remainder_score[own],
                    "mean_absolute_all_probe_raw_margin_remainder": np.abs(
                        remainder_margin
                    ).mean(),
                },
            )
    if pending_total:
        raise ValueError(f"unmatched total effects: {len(pending_total)}")
    cell_rows = finalize_cells(
        cells,
        ("group_id", "group_family", "direction", "path", "concept", "label"),
    )
    remainder_rows = finalize_cells(
        remainder_cells,
        ("group_id", "group_family", "direction", "concept", "label"),
    )

    positive_total = [
        row
        for row in cell_rows
        if row["label"] == 1 and row["path"] == "total"
    ]
    aggregate: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in positive_total:
        aggregate[(row["group_id"], row["direction"])].append(
            abs(row["mean_own_probe_raw_margin_effect"])
        )
    macros = {
        f"{group_id}.{direction}": float(np.mean(values))
        for (group_id, direction), values in sorted(aggregate.items())
    }
    comparisons = {}
    for direction in ("induction", "rescue"):
        k12 = macros[f"nested_heads.K12.{direction}"]
        k16 = macros[f"selected_components.K16.{direction}"]
        mlp = macros[f"selected_mlps.K4.{direction}"]
        comparisons[direction] = {
            "k12_absolute_own_probe_effect": k12,
            "k16_absolute_own_probe_effect": k16,
            "selected_mlps_absolute_own_probe_effect": mlp,
            "k12_to_k16_ratio": k12 / max(k16, 1e-12),
            "selected_mlps_to_k16_ratio": mlp / max(k16, 1e-12),
        }
    return {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-component-resolution",
        "execution_commit": execution_commit,
        "effect_row_count": len(effects),
        "estimand_note": "Effects are patched endpoint minus the natural target condition; downstream-dependent remainder is total minus frozen-write direct path.",
        "positive_macro_comparisons": comparisons,
        "cells": cell_rows,
        "downstream_dependent_remainder_cells": remainder_rows,
    }


def artifact_examples(
    records: Sequence[dict[str, Any]],
    *,
    writer_model: str,
    writer_field: str,
    writer_ids_field: str,
    state_model: str = "chameleon",
) -> list[DiagnosticExample]:
    result = []
    for record in records:
        writer_artifact = load_feature(writer_model, record["example_id"])
        state_artifact = (
            writer_artifact
            if writer_model == state_model
            else load_feature(state_model, record["example_id"])
        )
        head_ids = tuple(writer_artifact[writer_ids_field])
        tensor = writer_artifact[writer_field].float()
        result.append(
            DiagnosticExample(
                example_id=record["example_id"],
                concept=record["concept"],
                writer={head_id: tensor[:, index] for index, head_id in enumerate(head_ids)},
                normal_state=state_artifact["normal_resid_post_8"].float(),
                target_u=state_artifact["target_u"].float(),
            )
        )
    return result


def fit_and_evaluate(
    training: Sequence[DiagnosticExample],
    evaluation: Sequence[DiagnosticExample],
    head_ids: Sequence[str],
    probes: Sequence[Any],
    probe_scale: torch.Tensor,
    plan: Mapping[str, Any],
    *,
    normal_only: bool = False,
) -> tuple[dict[str, Any], Any, Any]:
    specification = plan["intermediate_prediction"]
    bases = fit_diagnostic_bases(
        training,
        head_ids,
        writer_rank=16,
        normal_rank=64,
        target_rank=128,
        seed=int(plan["inference"]["pca_seed"]),
    )
    training_matrices = build_diagnostic_matrices(training, bases)
    selection = select_alpha_leave_one_concept_out(
        training_matrices,
        bases,
        specification["alpha_grid"],
        normal_only=normal_only,
    )
    features = (
        training_matrices.normal_features
        if normal_only
        else training_matrices.full_features
    )
    fit = fit_weighted_ridge(
        features,
        training_matrices.target_coordinates,
        training_matrices.sample_weights,
        alpha=selection.selected_alpha,
    )
    evaluation_matrices = build_diagnostic_matrices(evaluation, bases)
    predictions = predict_original(
        fit, evaluation_matrices, bases, normal_only=normal_only
    )
    metrics = evaluate_diagnostic_predictions(
        predictions, evaluation_matrices, bases, probes, probe_scale
    )
    return (
        {
            "selected_alpha": selection.selected_alpha,
            "discovery_macro_r2_by_alpha": {
                str(alpha): value
                for alpha, value in selection.macro_r2_by_alpha.items()
            },
            "metrics": metrics.to_dict(),
        },
        bases,
        evaluation_matrices,
    )


def prediction_analysis(
    records: Sequence[dict[str, Any]],
    plan: Mapping[str, Any],
    probes: Sequence[Any],
    execution_commit: str,
    analysis_commit: str,
) -> dict[str, Any]:
    fit_concepts = set(plan["data_roles"]["diagnostic_fit"]["concepts"])
    heldout_concepts = set(
        plan["data_roles"]["diagnostic_heldout_concepts"]["concepts"]
    )
    training_records = [
        row
        for row in records
        if int(row["label"]) == 1
        and row["split"] == "discovery"
        and row["concept"] in fit_concepts
    ]
    evaluation_records = [
        row
        for row in records
        if int(row["label"]) == 1
        and row["split"] in {"validation", "safety-test"}
        and row["concept"] in heldout_concepts
    ]
    if len(training_records) != 256 or len(evaluation_records) != 610:
        raise ValueError("diagnostic role counts do not match the frozen contract")
    k12 = tuple(plan["component_sets"]["k12_ordered"])
    nonselected = tuple(
        plan["component_sets"]["layer_count_matched_nonselected_heads"]
    )
    train_k12 = artifact_examples(
        training_records,
        writer_model="chameleon",
        writer_field="k12_delta",
        writer_ids_field="k12_head_ids",
    )
    eval_k12 = artifact_examples(
        evaluation_records,
        writer_model="chameleon",
        writer_field="k12_delta",
        writer_ids_field="k12_head_ids",
    )
    leakage_audit = {
        "training": validate_diagnostic_examples(train_k12, k12),
        "evaluation": validate_diagnostic_examples(eval_k12, k12),
    }
    probe_scale = fit_probe_standardization(train_k12, probes)
    full, bases, evaluation_matrices = fit_and_evaluate(
        train_k12, eval_k12, k12, probes, probe_scale, plan
    )
    normal, _normal_bases, _normal_matrices = fit_and_evaluate(
        train_k12,
        eval_k12,
        k12,
        probes,
        probe_scale,
        plan,
        normal_only=True,
    )

    train_nonselected = artifact_examples(
        training_records,
        writer_model="chameleon",
        writer_field="nonselected_delta",
        writer_ids_field="nonselected_head_ids",
    )
    eval_nonselected = artifact_examples(
        evaluation_records,
        writer_model="chameleon",
        writer_field="nonselected_delta",
        writer_ids_field="nonselected_head_ids",
    )
    nonselected_result, _nonselected_bases, _nonselected_matrices = fit_and_evaluate(
        train_nonselected,
        eval_nonselected,
        nonselected,
        probes,
        probe_scale,
        plan,
    )

    train_precursor = artifact_examples(
        training_records,
        writer_model="precursor",
        writer_field="k12_delta",
        writer_ids_field="k12_head_ids",
        state_model="chameleon",
    )
    eval_precursor = artifact_examples(
        evaluation_records,
        writer_model="precursor",
        writer_field="k12_delta",
        writer_ids_field="k12_head_ids",
        state_model="chameleon",
    )
    precursor_result, _precursor_bases, _precursor_matrices = fit_and_evaluate(
        train_precursor,
        eval_precursor,
        k12,
        probes,
        probe_scale,
        plan,
    )

    mean_prediction = training_decile_mean_predictions(train_k12, eval_k12)
    mean_metrics = evaluate_diagnostic_predictions(
        mean_prediction, evaluation_matrices, bases, probes, probe_scale
    )
    concept_prediction = concept_leave_one_example_out_predictions(
        eval_k12, bases.target.mean
    )
    concept_metrics = evaluate_diagnostic_predictions(
        concept_prediction, evaluation_matrices, bases, probes, probe_scale
    )
    baselines = {
        "training_mean": {"metrics": mean_metrics.to_dict()},
        "concept_label_only": {"metrics": concept_metrics.to_dict()},
        "normal_state_only": normal,
        "nonselected_heads": nonselected_result,
        "precursor_trajectories": precursor_result,
    }
    best_r2 = max(
        row["metrics"]["macro_r2_u"] for row in baselines.values()
    )
    best_snmse = min(
        row["metrics"]["macro_probe_vector_snmse"] for row in baselines.values()
    )
    full_r2 = full["metrics"]["macro_r2_u"]
    full_snmse = full["metrics"]["macro_probe_vector_snmse"]
    gates = {
        "absolute_R2_u": {
            "value": full_r2,
            "threshold": 0.10,
            "pass": full_r2 >= 0.10,
        },
        "R2_u_improvement_over_best_baseline": {
            "value": full_r2 - best_r2,
            "best_baseline_value": best_r2,
            "threshold": 0.05,
            "pass": full_r2 - best_r2 >= 0.05,
        },
        "probe_vector_snmse_relative_to_best_baseline": {
            "value": full_snmse / max(best_snmse, 1e-12),
            "best_baseline_value": best_snmse,
            "threshold": 0.90,
            "pass": full_snmse / max(best_snmse, 1e-12) <= 0.90,
        },
    }
    return {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-intermediate-prediction",
        "execution_commit": execution_commit,
        "analysis_commit": analysis_commit,
        "evidence_class": "mechanism development; not fresh confirmation",
        "training_example_count": len(training_records),
        "heldout_example_count": len(evaluation_records),
        "leakage_audit": leakage_audit,
        "probe_standardization": [float(value) for value in probe_scale],
        "full_k12": full,
        "baselines": baselines,
        "gates": gates,
        "result": "pass" if all(row["pass"] for row in gates.values()) else "fail",
    }


def paired_hierarchical_bootstrap(
    by_concept: Mapping[str, np.ndarray],
    statistic: Callable[[list[np.ndarray]], float],
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    concepts = sorted(by_concept)
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.choice(concepts, size=len(concepts), replace=True)
        values = []
        for concept in sampled:
            rows = by_concept[str(concept)]
            indices = rng.integers(0, len(rows), size=len(rows))
            values.append(rows[indices])
        result[replicate] = statistic(values)
    return result


def acquisition_analysis(
    records: Sequence[dict[str, Any]],
    natural: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
    functional: Sequence[dict[str, Any]],
    plan: Mapping[str, Any],
    probe_names: Sequence[str],
    execution_commit: str,
    analysis_commit: str,
) -> dict[str, Any]:
    fit_concepts = set(plan["data_roles"]["diagnostic_fit"]["concepts"])
    heldout_concepts = set(
        plan["data_roles"]["diagnostic_heldout_concepts"]["concepts"]
    )
    discovery = [
        row
        for row in records
        if int(row["label"]) == 1
        and row["split"] == "discovery"
        and row["concept"] in fit_concepts
    ]
    heldout = [
        row
        for row in records
        if int(row["label"]) == 1
        and row["split"] in {"validation", "safety-test"}
        and row["concept"] in heldout_concepts
    ]
    k12 = tuple(plan["component_sets"]["k12_ordered"])
    sum_squares = torch.zeros(len(k12), dtype=torch.float64)
    counts = torch.zeros(len(k12), dtype=torch.int64)
    for record in discovery:
        values = load_feature("chameleon", record["example_id"])["k12_delta"].double()
        sum_squares += values.square().sum(dim=(0, 2))
        counts += values.shape[0] * values.shape[2]
    head_rms = torch.sqrt(sum_squares / counts.clamp(min=1)).clamp(min=1e-6)

    representation_by_concept: dict[str, list[list[float]]] = defaultdict(list)
    response_id_checks = []
    for record in heldout:
        chameleon = load_feature("chameleon", record["example_id"])
        precursor = load_feature("precursor", record["example_id"])
        response_id_checks.append(
            torch.equal(chameleon["response_ids"], precursor["response_ids"])
        )
        first = chameleon["k12_delta"].float() / head_rms[None, :, None].float()
        second = precursor["k12_delta"].float() / head_rms[None, :, None].float()
        first_flat = first.flatten()
        second_flat = second.flatten()
        first_energy = first_flat.square().sum().clamp(min=1e-12)
        second_energy = second_flat.square().sum()
        dot = (first_flat * second_flat).sum()
        representation_by_concept[record["concept"]].append(
            [
                float(dot / first_energy),
                float(torch.sqrt(second_energy / first_energy)),
                float(dot / torch.sqrt(first_energy * second_energy.clamp(min=1e-12))),
            ]
        )
    representation_arrays = {
        concept: np.asarray(values, dtype=np.float64)
        for concept, values in representation_by_concept.items()
    }
    representation_macro = np.mean(
        [values.mean(axis=0) for values in representation_arrays.values()], axis=0
    )
    inference = plan["inference"]
    representation_boot = paired_hierarchical_bootstrap(
        representation_arrays,
        lambda samples: abs(float(np.mean([values[:, 0].mean() for values in samples]))),
        replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
    )
    representation_gate = {
        "absolute_macro_aligned_amplitude": abs(float(representation_macro[0])),
        "one_sided_95_percent_upper_bound": float(
            np.quantile(representation_boot, 0.95)
        ),
        "point_threshold": 0.50,
        "upper_bound_threshold": 0.75,
    }
    representation_gate["pass"] = (
        representation_gate["absolute_macro_aligned_amplitude"] <= 0.50
        and representation_gate["one_sided_95_percent_upper_bound"] < 0.75
    )

    natural_by_key = {
        (row["model"], row["example_id"], row["condition"]): row for row in natural
    }
    chameleon_k12 = {
        row["example_id"]: row
        for row in effects
        if row["model"] == "chameleon"
        and row["direction"] == "induction"
        and row["path"] == "total"
        and row["group_id"] == "nested_heads.K12"
    }
    precursor_k12 = {row["example_id"]: row for row in functional}
    functional_by_concept: dict[str, list[list[float]]] = defaultdict(list)
    for record in heldout:
        example_id = record["example_id"]
        normal_chameleon = np.asarray(
            natural_by_key[("chameleon", example_id, "normal")]["mean_raw_margins"]
        )
        normal_precursor = np.asarray(
            natural_by_key[("precursor", example_id, "normal")]["mean_raw_margins"]
        )
        chameleon_effect = np.asarray(chameleon_k12[example_id]["mean_raw_margins"]) - normal_chameleon
        precursor_effect = np.asarray(precursor_k12[example_id]["mean_raw_margins"]) - normal_precursor
        functional_by_concept[record["concept"]].append(
            [float(np.abs(chameleon_effect).mean()), float(np.abs(precursor_effect).mean())]
        )
    functional_arrays = {
        concept: np.asarray(values, dtype=np.float64)
        for concept, values in functional_by_concept.items()
    }
    chameleon_macro = float(
        np.mean([values[:, 0].mean() for values in functional_arrays.values()])
    )
    precursor_macro = float(
        np.mean([values[:, 1].mean() for values in functional_arrays.values()])
    )
    functional_ratio = precursor_macro / max(chameleon_macro, 1e-12)
    functional_boot = paired_hierarchical_bootstrap(
        functional_arrays,
        lambda samples: float(
            np.mean([values[:, 1].mean() for values in samples])
            / max(np.mean([values[:, 0].mean() for values in samples]), 1e-12)
        ),
        replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
    )
    functional_gate = {
        "chameleon_macro_absolute_raw_margin_effect": chameleon_macro,
        "precursor_macro_absolute_raw_margin_effect": precursor_macro,
        "precursor_to_chameleon_ratio": functional_ratio,
        "one_sided_95_percent_upper_bound": float(np.quantile(functional_boot, 0.95)),
        "point_threshold": 0.50,
        "upper_bound_threshold": 0.75,
    }
    functional_gate["pass"] = (
        functional_gate["precursor_to_chameleon_ratio"] <= 0.50
        and functional_gate["one_sided_95_percent_upper_bound"] < 0.75
    )
    return {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-acquisition",
        "execution_commit": execution_commit,
        "analysis_commit": analysis_commit,
        "evidence_class": "mechanism development; not fresh confirmation",
        "population": {
            "example_count": len(heldout),
            "concept_count": len(representation_arrays),
            "probe_count": len(probe_names),
        },
        "head_rms_discovery_chameleon": dict(
            zip(k12, [float(value) for value in head_rms], strict=True)
        ),
        "representation": {
            "macro_aligned_amplitude": float(representation_macro[0]),
            "macro_magnitude_ratio": float(representation_macro[1]),
            "macro_cosine": float(representation_macro[2]),
            "per_concept": {
                concept: {
                    "example_count": len(values),
                    "mean_aligned_amplitude": float(values[:, 0].mean()),
                    "mean_magnitude_ratio": float(values[:, 1].mean()),
                    "mean_cosine": float(values[:, 2].mean()),
                }
                for concept, values in sorted(representation_arrays.items())
            },
            "gate": representation_gate,
        },
        "functional": {
            "per_concept": {
                concept: {
                    "example_count": len(values),
                    "chameleon_mean_absolute_raw_margin_effect": float(
                        values[:, 0].mean()
                    ),
                    "precursor_mean_absolute_raw_margin_effect": float(
                        values[:, 1].mean()
                    ),
                }
                for concept, values in sorted(functional_arrays.items())
            },
            "gate": functional_gate,
        },
        "response_token_ids_identical_between_checkpoints": all(response_id_checks),
        "result": "pass"
        if representation_gate["pass"] and functional_gate["pass"]
        else "fail",
    }


def main() -> None:
    analysis_commit = git_head()
    for path in (
        Path(__file__).resolve(),
        PLAN_PATH,
        ROOT / "scripts/day33_run_acquired_writer.py",
        ROOT / "src/neural_chameleon/causal_mechanisms.py",
        ROOT / "src/neural_chameleon/mechanism_prediction.py",
    ):
        require_committed(path, analysis_commit)
    plan = json.loads(PLAN_PATH.read_text())
    records = load_records(None)
    groups = component_groups(plan)
    probe_names, probes = load_probes()
    accounting = load_jsonl(ACCOUNTING_WORKING)
    natural = load_jsonl(NATURAL_WORKING)
    effects = load_jsonl(EFFECT_WORKING)
    functional = load_jsonl(FUNCTIONAL_WORKING)
    execution_commits = {
        row["execution_commit"]
        for rows in (accounting, natural, effects, functional)
        for row in rows
    }
    if len(execution_commits) != 1:
        raise ValueError(f"expected one execution commit, found {execution_commits}")
    execution_commit = next(iter(execution_commits))
    execution_ids = {
        row["execution_id"]
        for rows in (accounting, natural, effects, functional)
        for row in rows
    }
    if len(execution_ids) != 1:
        raise ValueError(f"expected one execution ID, found {execution_ids}")
    execution_id = next(iter(execution_ids))
    execution_parameters = json.loads(EXECUTION_PARAMETERS_PATH.read_text())
    if (
        execution_parameters["execution_id"] != execution_id
        or execution_parameters["execution_commit"] != execution_commit
        or execution_parameters["limit_per_concept"] is not None
    ):
        raise ValueError("execution parameter manifest does not describe a full run")
    positive_ids = sorted(
        row["example_id"] for row in records if int(row["label"]) == 1
    )
    expected_counts = {
        "accounting": expected_accounting_batches(
            records, "chameleon", int(execution_parameters["batch_size"])
        )
        + expected_accounting_batches(
            records, "precursor", int(execution_parameters["batch_size"])
        ),
        "natural": len(records) * 2 + len(positive_ids) * 2,
        "component_effects": len(records) * 2 * 2 * len(groups),
        "precursor_functional": len(positive_ids),
    }
    observed_counts = {
        "accounting": len(accounting),
        "natural": len(natural),
        "component_effects": len(effects),
        "precursor_functional": len(functional),
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Gate 1 row counts incomplete: observed={observed_counts}, expected={expected_counts}"
        )
    tolerances = plan["realized_forward_accounting"]["primary_tolerances"]
    audit_maxima = {
        "max_hidden_absolute_error": max(
            row["audit"]["hidden_max_abs_error"] for row in accounting
        ),
        "max_attention_allocation_absolute_error": max(
            row["audit"]["attention_allocation_max_abs_error"] for row in accounting
        ),
        "max_probe_margin_absolute_error": max(
            row["audit"]["probe_margin_max_abs_error"] for row in accounting
        ),
        "max_sequence_score_absolute_error": max(
            row["audit"]["sequence_score_max_abs_error"] for row in accounting
        ),
    }
    accounting_pass = all(
        audit_maxima[name] <= float(tolerances[name]) for name in audit_maxima
    )
    feature_manifests = {
        model: feature_manifest(model, positive_ids)
        for model in ("chameleon", "precursor")
    }
    for model in ("chameleon", "precursor"):
        for example_id in positive_ids:
            artifact = load_feature(model, example_id)
            if (
                artifact["execution_commit"] != execution_commit
                or artifact["execution_id"] != execution_id
            ):
                raise ValueError("feature artifact execution provenance is mixed")

    component = component_resolution(effects, natural, probe_names, execution_commit)
    component["execution_id"] = execution_id
    write_json(COMPONENT_PATH, component)
    acquisition = acquisition_analysis(
        records,
        natural,
        effects,
        functional,
        plan,
        probe_names,
        execution_commit,
        analysis_commit,
    )
    acquisition["execution_id"] = execution_id
    prediction = prediction_analysis(
        records, plan, probes, execution_commit, analysis_commit
    )
    prediction["execution_id"] = execution_id
    write_json(PREDICTION_PATH, prediction)
    scientific_pass = acquisition["result"] == "pass" and prediction["result"] == "pass"
    summary = {
        **acquisition,
        "procedure": "acquired-writer-development-v1-gate-1-summary",
        "accounting": {
            "maxima": audit_maxima,
            "tolerances": tolerances,
            "result": "pass" if accounting_pass else "fail",
        },
        "intermediate_prediction_result": prediction["result"],
        "scientific_continue_gate": "pass" if scientific_pass else "fail",
        "required_action": "begin Gate 2 development"
        if scientific_pass
        else "stop title-level successor program and use causal-localization framing",
    }
    write_json(SUMMARY_PATH, summary)
    source_files = (
        ACCOUNTING_WORKING,
        NATURAL_WORKING,
        EFFECT_WORKING,
        FUNCTIONAL_WORKING,
    )
    manifest = {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-execution-manifest",
        "execution_commit": execution_commit,
        "execution_id": execution_id,
        "execution_parameters": execution_parameters,
        "analysis_commit": analysis_commit,
        "plan_sha256": sha256_file(PLAN_PATH),
        "row_counts": observed_counts,
        "working_files": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in source_files
        },
        "feature_artifacts": feature_manifests,
    }
    write_json(MANIFEST_PATH, manifest)
    checks = {
        "row_counts_exact": observed_counts == expected_counts,
        "one_execution_commit": len(execution_commits) == 1,
        "one_execution_id": len(execution_ids) == 1,
        "execution_parameters_match": execution_parameters["execution_id"]
        == execution_id,
        "execution_source_committed": subprocess.run(
            ["git", "cat-file", "-e", f"{execution_commit}^{{commit}}"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0,
        "accounting_tolerances_pass": accounting_pass,
        "feature_counts_exact": all(
            row["file_count"] == len(positive_ids)
            for row in feature_manifests.values()
        ),
        "acquisition_response_ids_match": acquisition[
            "response_token_ids_identical_between_checkpoints"
        ],
        "probe_family_exact": tuple(probe_names)
        == tuple(sorted(row.removesuffix("_weights.pt") for row in [path.name for path in (ROOT / "external/minimal_neural_chameleons/probes").glob("*_weights.pt")])),
        "analysis_sources_committed": True,
        "scientific_gate_applied_without_threshold_change": True,
    }
    audit = {
        "schema_version": 1,
        "procedure": "acquired-writer-development-v1-gate-1-audit",
        "execution_commit": execution_commit,
        "execution_id": execution_id,
        "analysis_commit": analysis_commit,
        "checks": checks,
        "implementation_result": "pass" if all(checks.values()) else "fail",
        "scientific_result": "pass" if scientific_pass else "fail",
        "next_phase_authorized": bool(all(checks.values()) and scientific_pass),
    }
    write_json(AUDIT_PATH, audit)
    if not all(checks.values()):
        raise RuntimeError(json.dumps(audit, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "implementation_result": audit["implementation_result"],
                "scientific_result": audit["scientific_result"],
                "next_phase_authorized": audit["next_phase_authorized"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
