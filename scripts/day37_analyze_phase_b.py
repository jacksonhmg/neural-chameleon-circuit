#!/usr/bin/env python3
"""Deterministically reduce and adjudicate the frozen Phase A-B contract."""

from __future__ import annotations

import argparse
import gzip
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

from neural_chameleon.post_gate1_interventions import (  # noqa: E402
    phase_b_expected_rows,
)


CONTRACT_PATH = ROOT / "results/day-36/frozen-phase-a-b-contract.json"
CLARIFICATION_PATH = (
    ROOT / "results/day-37/frozen-phase-b-row-accounting-clarification.json"
)
ATTENTION_FREEZE_PATH = (
    ROOT / "results/day-37/frozen-attention-operator-implementation.json"
)
ATTENTION_MEMORY_CORRECTION_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction.json"
)
ATTENTION_MEMORY_CORRECTION_V2_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v2.json"
)
ATTENTION_MEMORY_CORRECTION_V3_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v3.json"
)
ATTENTION_MEMORY_CORRECTION_V4_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v4.json"
)
ATTENTION_MEMORY_CORRECTION_V5_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v5.json"
)
ATTENTION_MEMORY_CORRECTION_V6_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v6.json"
)
ATTENTION_MEMORY_CORRECTION_V7_PATH = (
    ROOT / "results/day-39/frozen-attention-memory-correction-v7.json"
)
PHASE_A_DIR = ROOT / "results/day-38"
RAW_DIR = ROOT / "results/day-39"
OUTPUT_DIR = ROOT / "results/day-40"
ARTIFACT_DIR = ROOT / "artifacts/post-gate1-phase-b-v1"
NATURAL_PATH = RAW_DIR / "natural-endpoints.working.jsonl"
ABSOLUTE_PATH = RAW_DIR / "absolute-effects.working.jsonl"
RANDOM_PATH = RAW_DIR / "random-effects.working.jsonl"
FRONTIER_PATH = RAW_DIR / "frontier-effects.working.jsonl"
ATTENTION_PATH = RAW_DIR / "attention-effects.working.jsonl"
SELECTION_PATH = RAW_DIR / "development-selection.json"
PARAMETERS_PATH = RAW_DIR / "execution-parameters.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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
        raise RuntimeError(f"{relative} differs from analysis commit {commit}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def encode_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_json(value))


def one_sided_lower(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values), 0.05))


def one_sided_upper(values: Sequence[float]) -> float:
    return float(np.quantile(np.asarray(values), 0.95))


def two_sided(values: Sequence[float]) -> list[float]:
    return [
        float(np.quantile(np.asarray(values), 0.025)),
        float(np.quantile(np.asarray(values), 0.975)),
    ]


def hierarchical_indices(
    concepts: Sequence[str],
    example_ids_by_concept: Mapping[str, Sequence[str]],
    *,
    replicates: int,
    seed: int,
) -> list[list[str]]:
    generator = np.random.default_rng(seed)
    concepts = tuple(sorted(concepts))
    result = []
    for _ in range(replicates):
        sampled_concepts = generator.choice(concepts, len(concepts), replace=True)
        sample = []
        for concept in sampled_concepts:
            ids = tuple(example_ids_by_concept[str(concept)])
            sample.extend(generator.choice(ids, len(ids), replace=True).tolist())
        result.append(sample)
    return result


def macro_from_ids(
    sample_ids: Sequence[str],
    values: Mapping[str, float],
    concept_by_id: Mapping[str, str],
) -> float:
    cells: dict[str, list[float]] = defaultdict(list)
    for example_id in sample_ids:
        cells[concept_by_id[example_id]].append(values[example_id])
    return float(np.mean([np.mean(cell) for cell in cells.values()]))


def bootstrap_metric(
    values: Mapping[str, float],
    concept_by_id: Mapping[str, str],
    samples: Sequence[Sequence[str]],
) -> dict[str, Any]:
    relevant_concepts = sorted({concept_by_id[key] for key in values})
    by_concept = {
        concept: float(
            np.mean([value for key, value in values.items() if concept_by_id[key] == concept])
        )
        for concept in relevant_concepts
    }
    point = float(np.mean(list(by_concept.values())))
    bootstrap = [macro_from_ids(sample, values, concept_by_id) for sample in samples]
    return {
        "point": point,
        "one_sided_95_lower": one_sided_lower(bootstrap),
        "one_sided_95_upper": one_sided_upper(bootstrap),
        "two_sided_95_interval": two_sided(bootstrap),
        "per_concept": by_concept,
    }


def bootstrap_difference(
    left: Mapping[str, float],
    right: Mapping[str, float],
    concept_by_id: Mapping[str, str],
    samples: Sequence[Sequence[str]],
) -> dict[str, Any]:
    values = {key: left[key] - right[key] for key in left}
    return bootstrap_metric(values, concept_by_id, samples)


def bootstrap_ratio(
    numerator: Mapping[str, float],
    denominator: Mapping[str, float],
    concept_by_id: Mapping[str, str],
    samples: Sequence[Sequence[str]],
) -> dict[str, Any]:
    ids = tuple(numerator)

    def ratio(sample: Sequence[str]) -> float:
        top = macro_from_ids(sample, numerator, concept_by_id)
        bottom = macro_from_ids(sample, denominator, concept_by_id)
        return top / max(abs(bottom), 0.1)

    point = ratio(ids)
    bootstrap = [ratio(sample) for sample in samples]
    return {
        "point": point,
        "one_sided_95_lower": one_sided_lower(bootstrap),
        "one_sided_95_upper": one_sided_upper(bootstrap),
        "two_sided_95_interval": two_sided(bootstrap),
    }


def own_margin(row: Mapping[str, Any]) -> float:
    return float(row["mean_raw_margins"][row["probe_names"].index(row["concept"])])


def probe_scale() -> torch.Tensor:
    summary = read_json(ROOT / "results/day-33/intermediate-prediction-summary.json")
    return torch.tensor(summary["probe_standardization"], dtype=torch.float32)


def vector(row: Mapping[str, Any]) -> torch.Tensor:
    return torch.tensor(row["mean_raw_margins"], dtype=torch.float32) / probe_scale()


def recovery(movement: torch.Tensor, natural: torch.Tensor) -> float:
    return float(
        (
            1.0
            - (movement - natural).square().sum()
            / natural.square().sum().clamp(min=1e-6)
        ).item()
    )


def common_metadata(commit: str, run_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "procedure": "frozen post-Gate-1 Phase A-B reduction",
        "analysis_commit": commit,
        "execution_id": run_id,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "row_clarification_sha256": sha256_file(CLARIFICATION_PATH),
        "attention_operator_sha256": sha256_file(ATTENTION_FREEZE_PATH),
        "attention_memory_correction_sha256": sha256_file(
            ATTENTION_MEMORY_CORRECTION_V7_PATH
        ),
        "evidence_class": "existing-data development evidence; not fresh confirmation",
        "sealed_gate_1_result": "fail",
    }


def natural_maps(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], dict[str, str]]:
    result = {
        (row["model"], row["example_id"], row["condition"]): row for row in rows
    }
    concept_by_id = {
        row["example_id"]: row["concept"]
        for row in rows
        if row["model"] == "chameleon"
    }
    return result, concept_by_id


def heldout_positive_ids(natural: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            example_id
            for (model, example_id, condition), row in natural.items()
            if model == "chameleon"
            and condition == "normal"
            and row["split"] != "discovery"
            and int(row["label"]) == 1
        }
    )


def metric_context(
    natural: Mapping[tuple[str, str, str], Mapping[str, Any]],
    concept_by_id: Mapping[str, str],
    contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, float], list[list[str]]]:
    ids = heldout_positive_ids(natural)
    gaps = {
        example_id: own_margin(natural[("chameleon", example_id, "normal")])
        - own_margin(natural[("chameleon", example_id, "correct_trigger")])
        for example_id in ids
    }
    ids_by_concept = {
        concept: [example_id for example_id in ids if concept_by_id[example_id] == concept]
        for concept in sorted({concept_by_id[example_id] for example_id in ids})
    }
    samples = hierarchical_indices(
        tuple(ids_by_concept),
        ids_by_concept,
        replicates=int(contract["inference"]["bootstrap_replicates"]),
        seed=int(contract["inference"]["bootstrap_seed"]),
    )
    return ids, gaps, samples


def summarize_absolute(
    rows: Sequence[Mapping[str, Any]],
    natural: Mapping[tuple[str, str, str], Mapping[str, Any]],
    concept_by_id: Mapping[str, str],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, float]]]:
    ids, gaps, samples = metric_context(natural, concept_by_id, contract)
    index = {
        (row["model"], row["example_id"], row["operator"], row["path"]): row
        for row in rows
    }
    gap_by_concept = {
        concept: float(
            np.mean([gaps[key] for key in ids if concept_by_id[key] == concept])
        )
        for concept in sorted({concept_by_id[key] for key in ids})
    }
    interpretable = {
        concept: gap >= float(
            contract["phase_b"]["natural_gap_and_directional_metrics"][
                "minimum_interpretable_gap"
            ]
        )
        for concept, gap in gap_by_concept.items()
    }

    negative = {}
    omission = {}
    rescue = {}
    induction = {}
    direct_rescue = {}
    direct_induction = {}
    total_recovery = {}
    direct_recovery = {}
    total_effect = {}
    direct_effect = {}
    for example_id in ids:
        concept_gap = gap_by_concept[concept_by_id[example_id]]
        denominator = concept_gap if abs(concept_gap) >= 0.1 else 0.1
        normal = natural[("chameleon", example_id, "normal")]
        triggered = natural[("chameleon", example_id, "correct_trigger")]
        n_zero = index[("chameleon", example_id, "N_zero", "total")]
        t_zero = index[("chameleon", example_id, "T_zero", "total")]
        t_from_n = index[("chameleon", example_id, "T_from_N", "total")]
        n_from_t = index[("chameleon", example_id, "N_from_T", "total")]
        t_from_n_direct = index[("chameleon", example_id, "T_from_N", "direct")]
        n_from_t_direct = index[("chameleon", example_id, "N_from_T", "direct")]
        negative[example_id] = (own_margin(t_zero) - own_margin(triggered)) / denominator
        omission[example_id] = (own_margin(normal) - own_margin(n_zero)) / denominator
        rescue_raw = own_margin(t_from_n) - own_margin(triggered)
        induction_raw = own_margin(normal) - own_margin(n_from_t)
        direct_rescue_raw = own_margin(t_from_n_direct) - own_margin(triggered)
        direct_induction_raw = own_margin(normal) - own_margin(n_from_t_direct)
        rescue[example_id] = rescue_raw / denominator
        induction[example_id] = induction_raw / denominator
        direct_rescue[example_id] = direct_rescue_raw / denominator
        direct_induction[example_id] = direct_induction_raw / denominator
        total_effect[example_id] = (rescue_raw + induction_raw) / 2
        direct_effect[example_id] = (direct_rescue_raw + direct_induction_raw) / 2
        natural_vector = vector(normal) - vector(triggered)
        rescue_vector = vector(t_from_n) - vector(triggered)
        induction_vector = vector(normal) - vector(n_from_t)
        direct_rescue_vector = vector(t_from_n_direct) - vector(triggered)
        direct_induction_vector = vector(normal) - vector(n_from_t_direct)
        total_recovery[example_id] = (
            recovery(rescue_vector, natural_vector)
            + recovery(induction_vector, natural_vector)
        ) / 2
        direct_recovery[example_id] = (
            recovery(direct_rescue_vector, natural_vector)
            + recovery(direct_induction_vector, natural_vector)
        ) / 2
    direct_fraction = {
        key: direct_effect[key]
        / (total_effect[key] if abs(total_effect[key]) >= 0.1 else (0.1 if total_effect[key] >= 0 else -0.1))
        for key in ids
    }
    downstream_fraction = {key: 1.0 - direct_fraction[key] for key in ids}
    metrics = {
        "negative_write_fraction": bootstrap_metric(negative, concept_by_id, samples),
        "omission_fraction": bootstrap_metric(omission, concept_by_id, samples),
        "negative_minus_omission": bootstrap_difference(
            negative, omission, concept_by_id, samples
        ),
        "omission_minus_negative": bootstrap_difference(
            omission, negative, concept_by_id, samples
        ),
        "replacement_rescue_fraction": bootstrap_metric(rescue, concept_by_id, samples),
        "replacement_induction_fraction": bootstrap_metric(
            induction, concept_by_id, samples
        ),
        "direct_fraction": bootstrap_metric(direct_fraction, concept_by_id, samples),
        "downstream_fraction": bootstrap_metric(
            downstream_fraction, concept_by_id, samples
        ),
        "complete_probe_vector_recovery": bootstrap_metric(
            total_recovery, concept_by_id, samples
        ),
        "direct_complete_probe_vector_recovery": bootstrap_metric(
            direct_recovery, concept_by_id, samples
        ),
        "total_replacement_effect": bootstrap_metric(
            total_effect, concept_by_id, samples
        ),
        "direct_replacement_effect": bootstrap_metric(
            direct_effect, concept_by_id, samples
        ),
    }
    summary = {
        "natural_gap_G_by_concept": gap_by_concept,
        "interpretable_gap_by_concept": interpretable,
        "interpretable_heldout_concept_count": sum(interpretable.values()),
        "metrics": metrics,
        "all_absolute_operators_reported": sorted({row["operator"] for row in rows}),
        "negative_population_row_count": sum(int(row["label"]) == 0 for row in rows),
    }
    raw = {
        "total_effect": total_effect,
        "direct_effect": direct_effect,
        "total_recovery": total_recovery,
        "direct_recovery": direct_recovery,
        "negative": negative,
        "omission": omission,
    }
    return summary, raw


def frontier_effect(
    row: Mapping[str, Any], natural: Mapping[tuple[str, str, str], Mapping[str, Any]]
) -> float:
    target_condition = "normal" if row["direction"] == "induction" else "correct_trigger"
    baseline = natural[(row["model"], row["example_id"], target_condition)]
    if row["direction"] == "induction":
        return own_margin(baseline) - own_margin(row)
    return own_margin(row) - own_margin(baseline)


def summarize_frontier(
    rows: Sequence[Mapping[str, Any]],
    natural: Mapping[tuple[str, str, str], Mapping[str, Any]],
    concept_by_id: Mapping[str, str],
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, float]]]:
    ids, _gaps, samples = metric_context(natural, concept_by_id, contract)
    layer = int(selection["selected_frontier"]["source_layer"])
    evaluation = [
        row
        for row in rows
        if row["evaluation_scope"] == "heldout_or_negative"
        and row["model"] == "chameleon"
        and int(row["label"]) == 1
        and row["split"] != "discovery"
        and int(row["source_layer"]) == layer
    ]
    index = {
        (
            row["example_id"],
            row["source_family"],
            row["configuration_role"],
            row["direction"],
        ): row
        for row in evaluation
    }
    selected_recovery = {}
    control_recovery = {}
    for example_id in ids:
        values = {}
        for family in ("selected", "nonselected"):
            direction_values = []
            for direction in ("induction", "rescue"):
                direct = frontier_effect(
                    index[(example_id, family, "direct", direction)], natural
                )
                selected = frontier_effect(
                    index[(example_id, family, "selected", direction)], natural
                )
                total = frontier_effect(
                    index[(example_id, family, "total", direction)], natural
                )
                denominator = total - direct
                direction_values.append(
                    (selected - direct)
                    / (
                        denominator
                        if abs(denominator) >= 0.1
                        else (0.1 if denominator >= 0 else -0.1)
                    )
                )
            values[family] = sum(direction_values) / 2
        selected_recovery[example_id] = values["selected"]
        control_recovery[example_id] = values["nonselected"]
    difference = bootstrap_difference(
        selected_recovery, control_recovery, concept_by_id, samples
    )
    return (
        {
            "development_selection": selection["selected_frontier"],
            "selected_frontier_remainder_recovery": bootstrap_metric(
                selected_recovery, concept_by_id, samples
            ),
            "nonselected_frontier_remainder_recovery": bootstrap_metric(
                control_recovery, concept_by_id, samples
            ),
            "selected_minus_nonselected": difference,
            "negative_population_row_count": sum(
                row["evaluation_scope"] == "heldout_or_negative"
                and int(row["label"]) == 0
                for row in rows
            ),
        },
        {"selected": selected_recovery, "nonselected": control_recovery},
    )


def summarize_attention(
    rows: Sequence[Mapping[str, Any]],
    natural: Mapping[tuple[str, str, str], Mapping[str, Any]],
    concept_by_id: Mapping[str, str],
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    ids, _gaps, samples = metric_context(natural, concept_by_id, contract)
    site = selection["selected_attention_site"]["site_id"]
    evaluation = [
        row
        for row in rows
        if row["evaluation_scope"] == "heldout_or_negative"
        and row["model"] == "chameleon"
        and int(row["label"]) == 1
        and row["split"] != "discovery"
        and row["site_id"] == site
    ]
    by_cell: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    per_example: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in evaluation:
        target_condition = "normal" if row["direction"] == "induction" else "correct_trigger"
        baseline = natural[("chameleon", row["example_id"], target_condition)]
        normal = natural[("chameleon", row["example_id"], "normal")]
        triggered = natural[("chameleon", row["example_id"], "correct_trigger")]
        movement = (
            vector(baseline) - vector(row)
            if row["direction"] == "induction"
            else vector(row) - vector(baseline)
        )
        value = recovery(movement, vector(normal) - vector(triggered))
        per_example[(row["operation"], row["path"])][row["example_id"]].append(value)
        by_cell[(row["operation"], row["path"])][row["concept"]].append(value)
    metrics = {}
    for cell, example_values in per_example.items():
        averaged = {key: float(np.mean(value)) for key, value in example_values.items()}
        metrics[f"{cell[0]}.{cell[1]}"] = bootstrap_metric(
            averaged, concept_by_id, samples
        )
    return {
        "development_selection": selection["selected_attention_site"],
        "heldout_complete_probe_vector_recovery": metrics,
        "negative_population_row_count": sum(
            row["evaluation_scope"] == "heldout_or_negative" and int(row["label"]) == 0
            for row in rows
        ),
        "token_source_masks_present": all(
            set(row["token_source_partition"])
            >= {
                "concept_span",
                "monitoring_language",
                "original_prompt",
                "template_or_punctuation",
                "previous_response",
            }
            for row in rows
        ),
    }


def summarize_controls(
    random_rows: Sequence[Mapping[str, Any]],
    absolute_rows: Sequence[Mapping[str, Any]],
    natural: Mapping[tuple[str, str, str], Mapping[str, Any]],
    selected_effect: Mapping[str, float],
    concept_by_id: Mapping[str, str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    ids, _gaps, samples = metric_context(natural, concept_by_id, contract)
    random_cells: dict[str, list[float]] = defaultdict(list)
    for row in random_rows:
        if row["model"] != "chameleon" or row["path"] != "total" or row["example_id"] not in ids:
            continue
        condition = "normal" if row["direction"] == "induction" else "correct_trigger"
        baseline = natural[("chameleon", row["example_id"], condition)]
        effect = (
            own_margin(baseline) - own_margin(row)
            if row["direction"] == "induction"
            else own_margin(row) - own_margin(baseline)
        )
        random_cells[row["example_id"]].append(abs(effect))
    random_effect = {key: float(np.mean(values)) for key, values in random_cells.items()}
    selected_absolute = {key: abs(selected_effect[key]) for key in ids}

    absolute_index = {
        (row["model"], row["example_id"], row["operator"], row["path"]): row
        for row in absolute_rows
    }
    precursor = {}
    for example_id in ids:
        normal = natural[("precursor", example_id, "normal")]
        triggered = natural[("precursor", example_id, "correct_trigger")]
        rescue = own_margin(
            absolute_index[("precursor", example_id, "T_from_N", "total")]
        ) - own_margin(triggered)
        induction = own_margin(normal) - own_margin(
            absolute_index[("precursor", example_id, "N_from_T", "total")]
        )
        precursor[example_id] = abs((rescue + induction) / 2)
    return {
        "random_to_selected_absolute_effect_ratio": bootstrap_ratio(
            random_effect, selected_absolute, concept_by_id, samples
        ),
        "precursor_to_chameleon_absolute_effect_ratio": bootstrap_ratio(
            precursor, selected_absolute, concept_by_id, samples
        ),
    }


def gate_threshold(
    metric: Mapping[str, float], *, point_min: float, lower_min: float
) -> bool:
    return metric["point"] >= point_min and metric["one_sided_95_lower"] >= lower_min


def adjudicate(
    absolute: Mapping[str, Any],
    frontier: Mapping[str, Any],
    controls: Mapping[str, Any],
    raw_absolute: Mapping[str, Mapping[str, float]],
    concept_by_id: Mapping[str, str],
) -> dict[str, Any]:
    metrics = absolute["metrics"]
    negative_base = gate_threshold(
        metrics["negative_write_fraction"], point_min=0.25, lower_min=0.10
    )
    omission_base = gate_threshold(
        metrics["omission_fraction"], point_min=0.25, lower_min=0.10
    )
    negative_advantage = gate_threshold(
        metrics["negative_minus_omission"], point_min=0.15, lower_min=0.0
    )
    omission_advantage = gate_threshold(
        metrics["omission_minus_negative"], point_min=0.15, lower_min=0.0
    )
    negative_writing = negative_base and negative_advantage
    omission = omission_base and omission_advantage
    direct_reconfiguration = (
        negative_base and omission_base and not negative_advantage and not omission_advantage
    )
    direct_monitor = gate_threshold(
        metrics["direct_fraction"], point_min=0.35, lower_min=0.20
    ) and gate_threshold(
        metrics["direct_complete_probe_vector_recovery"],
        point_min=0.20,
        lower_min=0.0,
    )
    amplification = (
        gate_threshold(
            metrics["downstream_fraction"], point_min=0.35, lower_min=0.20
        )
        and gate_threshold(
            frontier["selected_frontier_remainder_recovery"],
            point_min=0.50,
            lower_min=0.25,
        )
        and gate_threshold(
            frontier["selected_minus_nonselected"],
            point_min=0.20,
            lower_min=0.0,
        )
    )
    vector_pass = gate_threshold(
        metrics["complete_probe_vector_recovery"], point_min=0.25, lower_min=0.10
    )
    random_ratio = controls["random_to_selected_absolute_effect_ratio"]
    precursor_ratio = controls["precursor_to_chameleon_absolute_effect_ratio"]
    controls_pass = (
        random_ratio["point"] <= 0.5
        and random_ratio["one_sided_95_upper"] < 0.75
        and precursor_ratio["point"] <= 0.5
        and precursor_ratio["one_sided_95_upper"] < 0.75
    )
    candidate_operation = (
        negative_writing
        or omission
        or direct_reconfiguration
        or amplification
    )
    primary_effect = raw_absolute["total_effect"]
    per_concept = {
        concept: float(
            np.mean(
                [value for key, value in primary_effect.items() if concept_by_id[key] == concept]
            )
        )
        for concept in sorted(set(concept_by_id[key] for key in primary_effect))
    }
    sign_count = sum(value > 0 for value in per_concept.values())
    sign_pass = sign_count >= 7
    generator = np.random.default_rng(36001 + 99)
    safety_opposite = {}
    safety_intervals = {}
    for concept in ("deception", "harmful"):
        cell = [
            value
            for key, value in primary_effect.items()
            if concept_by_id[key] == concept
        ]
        boot = [
            float(np.mean(generator.choice(cell, len(cell), replace=True)))
            for _ in range(10000)
        ]
        interval = two_sided(boot)
        safety_intervals[concept] = interval
        safety_opposite[concept] = interval[1] < 0
    stability = sign_pass and not any(safety_opposite.values())
    natural_pass = absolute["interpretable_heldout_concept_count"] >= 7
    direct_or_frontier = direct_monitor or amplification
    pass_all = (
        natural_pass
        and candidate_operation
        and direct_or_frontier
        and vector_pass
        and controls_pass
        and stability
    )
    return {
        "natural_phenomenon": natural_pass,
        "negative_writing": negative_writing,
        "omission": omission,
        "direct_reconfiguration": direct_reconfiguration,
        "direct_monitor_writing": direct_monitor,
        "downstream_amplification": amplification,
        "hybrid": direct_monitor and amplification and (
            negative_writing or omission or direct_reconfiguration
        ),
        "complete_vector": vector_pass,
        "control_weakness": controls_pass,
        "sign_consistency": {
            "pass": stability,
            "positive_direction_concept_count": sign_count,
            "per_concept": per_concept,
            "safety_supported_opposite": safety_opposite,
            "safety_two_sided_95_intervals": safety_intervals,
        },
        "scientific_continue_gate": "pass" if pass_all else "fail",
        "consequence": (
            "Phase C contract drafting permitted; execution still requires separate authorization"
            if pass_all
            else "stop the mechanism program and retain the sealed causal-localization fallback"
        ),
    }


def audit_rows(
    natural: Sequence[Mapping[str, Any]],
    absolute: Sequence[Mapping[str, Any]],
    random: Sequence[Mapping[str, Any]],
    frontier: Sequence[Mapping[str, Any]],
    attention: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    clarification: Mapping[str, Any],
) -> dict[str, Any]:
    expected = phase_b_expected_rows(
        {
            "complete": 1732,
            "positive": 866,
            "discovery_positive": 256,
            "heldout_positive": 610,
            "negative": 866,
        }
    )
    observed = {
        "absolute_contribution_rows": len(absolute),
        "matched_random_rows": len(random),
        "frontier_discovery_rows": sum(
            row["evaluation_scope"] == "discovery" for row in frontier
        ),
        "frontier_heldout_rows": sum(
            row["evaluation_scope"] == "heldout_or_negative"
            and int(row["label"]) == 1
            for row in frontier
        ),
        "frontier_negative_rows": sum(
            row["evaluation_scope"] == "heldout_or_negative"
            and int(row["label"]) == 0
            for row in frontier
        ),
        "attention_discovery_rows": sum(
            row["evaluation_scope"] == "discovery" for row in attention
        ),
        "attention_heldout_rows": sum(
            row["evaluation_scope"] == "heldout_or_negative"
            and int(row["label"]) == 1
            for row in attention
        ),
        "attention_negative_rows": sum(
            row["evaluation_scope"] == "heldout_or_negative"
            and int(row["label"]) == 0
            for row in attention
        ),
    }
    observed["total_phase_b_effect_rows"] = sum(observed.values())
    key_functions: Sequence[tuple[str, Sequence[Mapping[str, Any]], Callable[[Mapping[str, Any]], tuple[Any, ...]]]] = (
        (
            "absolute",
            absolute,
            lambda row: (row["model"], row["example_id"], row["operator"], row["path"]),
        ),
        (
            "random",
            random,
            lambda row: (
                row["model"],
                row["example_id"],
                row["draw_index"],
                row["direction"],
                row["path"],
            ),
        ),
        (
            "frontier",
            frontier,
            lambda row: (
                row["model"],
                row["example_id"],
                row["evaluation_scope"],
                row["source_family"],
                row["source_layer"],
                row["configuration_role"],
                row["direction"],
            ),
        ),
        (
            "attention",
            attention,
            lambda row: (
                row["model"],
                row["example_id"],
                row["evaluation_scope"],
                row["site_id"],
                row["operation"],
                row["direction"],
                row["path"],
            ),
        ),
    )
    uniqueness = {
        name: len({key(row) for row in rows}) == len(rows)
        for name, rows, key in key_functions
    }
    identities = {
        row["execution_id"]
        for rows in (natural, absolute, random, frontier, attention)
        for row in rows
    }
    commits = {
        row["execution_commit"]
        for rows in (natural, absolute, random, frontier, attention)
        for row in rows
    }
    correction = read_json(ATTENTION_MEMORY_CORRECTION_V7_PATH)
    protocol_commit = parameters["execution_commit"]
    correction_commit = parameters.get("correction_runtime_commit")

    def runtime_commit(row: Mapping[str, Any]) -> str:
        return str(row.get("runtime_commit", row["execution_commit"]))

    def runtime_id(row: Mapping[str, Any]) -> str:
        return str(row.get("runtime_execution_id", row["execution_id"]))

    runtime_commits = {
        runtime_commit(row)
        for rows in (natural, absolute, random, frontier, attention)
        for row in rows
    }
    runtime_ids = {
        runtime_id(row)
        for rows in (natural, absolute, random, frontier, attention)
        for row in rows
    }
    frontier_discovery_runtime = {
        runtime_commit(row)
        for row in frontier
        if row["evaluation_scope"] == "discovery"
    }
    frontier_evaluation_runtime = {
        runtime_commit(row)
        for row in frontier
        if row["evaluation_scope"] == "heldout_or_negative"
    }
    attention_runtime = {runtime_commit(row) for row in attention}
    failed_attention = ROOT / correction["preserved_attention_reference"]["path"]
    superseded_v6 = ROOT / correction["superseded_v6_runtime"]["attempt_path"]
    corrected_preflight = read_json(RAW_DIR / "real-checkpoint-preflight.json")
    replay_audit = read_json(RAW_DIR / "attention-prefix-replay-audit.json")
    checks = {
        "corrected_row_counts_match": observed == expected,
        "parent_contract_mismatch_preserved": (
            read_json(CONTRACT_PATH)["expected_execution_matrix"][
                "total_phase_b_effect_rows"
            ]
            == 556992
        ),
        "clarification_total_matches": clarification[
            "corrected_total_phase_b_effect_rows"
        ]
        == expected["total_phase_b_effect_rows"],
        "all_effect_keys_unique": all(uniqueness.values()),
        "natural_rows_complete": len(natural) == 1732 * 2 * 2,
        "one_execution_identity": identities == {parameters["execution_id"]},
        "one_execution_commit": commits == {parameters["execution_commit"]},
        "haar_invariants_all_pass": all(
            row["haar_invariants"]["pass"] for row in random
        ),
        "phase_a_outputs_complete": all(
            (PHASE_A_DIR / name).exists()
            for name in read_json(CONTRACT_PATH)["phase_a_required_outputs"]
        ),
        "preflight_pass": (
            RAW_DIR / "real-checkpoint-preflight.json"
        ).exists()
        and read_json(RAW_DIR / "real-checkpoint-preflight.json")["result"] == "pass",
        "attention_memory_correction_frozen": (
            correction["status"]
            == "frozen-before-v7-corrected-attention-outcomes"
            and correction["protocol_execution_commit"] == protocol_commit
            and correction["protocol_execution_id"] == parameters["execution_id"]
        ),
        "failed_attention_attempt_preserved_and_excluded": (
            failed_attention.exists()
            and sum(1 for _ in failed_attention.open())
            == int(correction["preserved_attention_reference"]["rows"])
            and sha256_file(failed_attention)
            == correction["preserved_attention_reference"]["sha256"]
            and failed_attention != ATTENTION_PATH
        ),
        "superseded_v6_attempt_preserved_and_excluded": (
            superseded_v6.exists()
            and sum(1 for _ in superseded_v6.open())
            == int(correction["superseded_v6_runtime"]["attempt_rows"])
            and sha256_file(superseded_v6)
            == correction["superseded_v6_runtime"]["attempt_sha256"]
            and superseded_v6 != ATTENTION_PATH
        ),
        "corrected_memory_schedule_preflight_exact": (
            corrected_preflight["preflight_commit"] == correction_commit
            and corrected_preflight["patch_kernel_exact_equality"]
            and all(
                checkpoint["attention_memory_correction"]["job_chunk_size"]
                == int(correction["correction"]["job_chunk_size"])
                and checkpoint["attention_memory_correction"][
                    "metadata_block_size"
                ]
                == int(correction["correction"]["attention_metadata_block_size"])
                and not checkpoint["attention_memory_correction"][
                    "expanded_live_tail_shape_change"
                ]
                and checkpoint["attention_memory_correction"][
                    "batch_local_attention_references_deleted_before_release"
                ]
                and checkpoint["attention_memory_correction"][
                    "original_gemma_mlp_forward"
                ]
                and checkpoint["attention_memory_correction"][
                    "original_full_eager_attention_forward"
                ]
                and checkpoint["attention_memory_correction"][
                    "mps_high_watermark_ratio"
                ]
                == correction["required_pre_population_gates"][
                    "mps_high_watermark_ratio"
                ]
                and checkpoint["attention_memory_correction"][
                    "process_shard_batch_count"
                ]
                == correction["required_pre_population_gates"][
                    "process_shard_batch_count"
                ]
                and checkpoint["memory_efficient_gemma_mlp_count"] == 0
                for checkpoint in corrected_preflight["checkpoints"].values()
            )
        ),
        "attention_prefix_replay_exact": (
            replay_audit["result"] == "pass"
            and replay_audit["rows"]
            == int(correction["required_population_replay_gate"]["rows"])
            and replay_audit["tolerance"] == 0.0
            and replay_audit["reference_sha256"]
            == correction["preserved_attention_reference"]["sha256"]
            and replay_audit["corrected_runtime_commit"] == correction_commit
        ),
        "runtime_provenance_exact": (
            correction_commit is not None
            and runtime_commits == {protocol_commit, correction_commit}
            and frontier_discovery_runtime == {protocol_commit}
            and frontier_evaluation_runtime == {correction_commit}
            and attention_runtime == {correction_commit}
            and runtime_ids
            == {
                parameters["execution_id"],
                parameters["correction_runtime_execution_id"],
            }
        ),
    }
    return {
        "result": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "expected_rows": expected,
        "observed_rows": observed,
        "unique_keys": uniqueness,
        "execution_identities": sorted(identities),
        "execution_commits": sorted(commits),
        "runtime_commits": sorted(runtime_commits),
        "runtime_execution_ids": sorted(runtime_ids),
    }


def compress_raw(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as source, gzip.open(destination, "wb", mtime=0) as target:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            target.write(chunk)


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/post_gate1_interventions.py",
        CONTRACT_PATH,
        CLARIFICATION_PATH,
        ATTENTION_FREEZE_PATH,
        ATTENTION_MEMORY_CORRECTION_PATH,
        ATTENTION_MEMORY_CORRECTION_V2_PATH,
        ATTENTION_MEMORY_CORRECTION_V3_PATH,
        ATTENTION_MEMORY_CORRECTION_V4_PATH,
        ATTENTION_MEMORY_CORRECTION_V5_PATH,
        ATTENTION_MEMORY_CORRECTION_V6_PATH,
        ATTENTION_MEMORY_CORRECTION_V7_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    clarification = read_json(CLARIFICATION_PATH)
    parameters = read_json(PARAMETERS_PATH)
    selection = read_json(SELECTION_PATH)
    natural_rows = load_jsonl(NATURAL_PATH)
    absolute_rows = load_jsonl(ABSOLUTE_PATH)
    random_rows = load_jsonl(RANDOM_PATH)
    frontier_rows = load_jsonl(FRONTIER_PATH)
    attention_rows = load_jsonl(ATTENTION_PATH)
    natural, concept_by_id = natural_maps(natural_rows)
    audit = audit_rows(
        natural_rows,
        absolute_rows,
        random_rows,
        frontier_rows,
        attention_rows,
        parameters,
        clarification,
    )
    if audit["result"] != "pass":
        write_json(args.output_dir / "phase-a-b-audit.json", audit)
        raise RuntimeError("Phase A-B implementation audit failed; interpretation blocked")
    absolute, raw_absolute = summarize_absolute(
        absolute_rows, natural, concept_by_id, contract
    )
    frontier, raw_frontier = summarize_frontier(
        frontier_rows, natural, concept_by_id, contract, selection
    )
    attention = summarize_attention(
        attention_rows, natural, concept_by_id, contract, selection
    )
    controls = summarize_controls(
        random_rows,
        absolute_rows,
        natural,
        raw_absolute["total_effect"],
        concept_by_id,
        contract,
    )
    signature = adjudicate(
        absolute, frontier, controls, raw_absolute, concept_by_id
    )
    run_id = parameters["execution_id"]
    common = common_metadata(commit, run_id)
    outputs = {
        "absolute-contribution-summary.json": {**common, **absolute},
        "downstream-frontier-summary.json": {**common, **frontier},
        "attention-operation-summary.json": {**common, **attention},
        "mechanism-signature-summary.json": {
            **common,
            "controls": controls,
            "gates": signature,
        },
        "phase-a-b-audit.json": {**common, **audit},
    }
    raw_paths = (
        NATURAL_PATH,
        ABSOLUTE_PATH,
        RANDOM_PATH,
        FRONTIER_PATH,
        ATTENTION_PATH,
    )
    manifest = {
        **common,
        "raw_artifacts": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "rows": sum(1 for _ in path.open()),
            }
            for path in raw_paths
        },
        "mean_artifacts": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in (
                ARTIFACT_DIR / "chameleon-normal-mean-stats.pt",
                ARTIFACT_DIR / "precursor-normal-mean-stats.pt",
            )
        },
        "selection_sha256": sha256_file(SELECTION_PATH),
        "execution_parameters_sha256": sha256_file(PARAMETERS_PATH),
        "scientific_continue_gate": signature["scientific_continue_gate"],
    }
    outputs["execution-artifact-manifest.json"] = manifest
    # A second serialization must be byte-identical before any artifact is written.
    for name, value in outputs.items():
        first = encode_json(value)
        second = encode_json(value)
        if first != second:
            raise RuntimeError(f"nondeterministic reducer serialization: {name}")
        path = args.output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(first)


if __name__ == "__main__":
    main()
