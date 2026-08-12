#!/usr/bin/env python3
"""Reduce the saved Day 57 four-state K12 interaction without a model run."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from day45_run_prototype_population import write_json_atomic  # noqa: E402
from day57_analyze_confirm_trace_acquisition import (  # noqa: E402
    expanded_contract,
    probe_metrics,
    stage_rows,
    state,
)


OUTPUT_DIR = ROOT / "results/day-58"
SUMMARY_PATH = OUTPUT_DIR / "four-state-interaction-summary.json"
EXAMPLE_PATH = OUTPUT_DIR / "four-state-interaction-example-metrics.json"
PANEL_PATH = ROOT / "data/splits/day57-v1/fresh-confirmation.jsonl"
PROBE_NAMES = tuple(
    path.name.removesuffix("_weights.pt")
    for path in sorted((ROOT / "external/minimal_neural_chameleons/probes").glob("*_weights.pt"))
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values_array[order[stop]] == values_array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    x, y = rankdata(left), rankdata(right)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def vector_metrics(effect: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    denominator = float(exact @ exact)
    exact_norm = math.sqrt(max(denominator, 0.0))
    if denominator <= 1e-12:
        return {"aligned_recovery": 0.0, "norm_ratio": 0.0, "cosine": 0.0}
    effect_norm = float(torch.linalg.vector_norm(effect))
    projection = float(effect @ exact) / denominator
    cosine = 0.0 if effect_norm <= 1e-12 else float(effect @ exact) / (effect_norm * exact_norm)
    return {
        "aligned_recovery": projection,
        "norm_ratio": effect_norm / exact_norm,
        "cosine": cosine,
    }


def concept_macro(
    rows: Sequence[Mapping[str, Any]], key: str
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["concept"])].append(float(row[key]))
    by_concept = {
        concept: float(np.median(values)) for concept, values in sorted(grouped.items())
    }
    return float(np.median(list(by_concept.values()))), by_concept


def summarize_scalar_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    excluded = {"example_id", "concept", "direction", "response_token_count", "label"}
    keys = sorted(set(rows[0]) - excluded)
    result: dict[str, Any] = {}
    for key in keys:
        median, concepts = concept_macro(rows, key)
        result[key] = {
            "median_concept": median,
            "by_concept": concepts,
            "example_median": float(np.median([float(row[key]) for row in rows])),
        }
    return result


def probe_rows(
    raw_rows: Sequence[Mapping[str, Any]], direction: str
) -> list[dict[str, Any]]:
    accumulators = {
        name: {"exact_square": 0.0, "prefix_dot": 0.0, "complement_dot": 0.0,
               "interaction_dot": 0.0, "interaction_square": 0.0}
        for name in PROBE_NAMES
    }
    for row in raw_rows:
        target = state(row, f"{direction}.identity_target", "margins").reshape(-1)
        exact_state = state(row, f"{direction}.exact_donor_all", "margins").reshape(-1)
        prefix_state = state(row, f"{direction}.monitoring_prefix_install", "margins").reshape(-1)
        complement_state = state(row, f"{direction}.monitoring_prefix_remove", "margins").reshape(-1)
        exact = exact_state - target
        prefix = prefix_state - target
        complement = complement_state - target
        interaction = exact - prefix - complement
        for index, name in enumerate(PROBE_NAMES):
            item = accumulators[name]
            e, p, c, i = (float(value[index]) for value in (exact, prefix, complement, interaction))
            item["exact_square"] += e * e
            item["prefix_dot"] += p * e
            item["complement_dot"] += c * e
            item["interaction_dot"] += i * e
            item["interaction_square"] += i * i
    result = []
    for name, item in accumulators.items():
        denominator = item["exact_square"]
        result.append(
            {
                "probe": name,
                "exact_rms": math.sqrt(denominator / len(raw_rows)),
                "prefix_aligned_recovery": item["prefix_dot"] / denominator,
                "complement_aligned_recovery": item["complement_dot"] / denominator,
                "interaction_aligned_recovery": item["interaction_dot"] / denominator,
                "interaction_norm_ratio": math.sqrt(item["interaction_square"] / denominator),
            }
        )
    return result


def main() -> None:
    if len(PROBE_NAMES) != 13:
        raise RuntimeError(f"expected 13 probe rows, found {len(PROBE_NAMES)}")
    contract = expanded_contract()
    raw_rows, execution, _preflight = stage_rows("confirmation", contract)
    metadata = {
        row["example_id"]: row
        for row in (json.loads(line) for line in PANEL_PATH.read_text().splitlines() if line)
    }
    example_rows: list[dict[str, Any]] = []
    directions: dict[str, Any] = {}
    for direction in contract["conditions"]["directions"]:
        direction_examples = []
        algebra_max_abs = 0.0
        for row in raw_rows:
            target = state(row, f"{direction}.identity_target", "margins").reshape(-1).double()
            exact_state = state(row, f"{direction}.exact_donor_all", "margins").reshape(-1).double()
            prefix_state = state(row, f"{direction}.monitoring_prefix_install", "margins").reshape(-1).double()
            complement_state = state(row, f"{direction}.monitoring_prefix_remove", "margins").reshape(-1).double()
            exact = exact_state - target
            prefix = prefix_state - target
            complement = complement_state - target
            interaction = exact - prefix - complement
            algebra_max_abs = max(
                algebra_max_abs,
                float((exact - prefix - complement - interaction).abs().max()),
            )
            exact_metrics = probe_metrics(row, direction, "exact_donor_all")
            values: dict[str, Any] = {
                "example_id": row["example_id"],
                "concept": row["concept"],
                "direction": direction,
                "response_token_count": int(metadata[row["example_id"]]["response_token_count"]),
                "label": int(metadata[row["example_id"]]["label"]),
                "exact_natural_probe_recovery": float(exact_metrics["probe_recovery"]),
            }
            for name, effect in (
                ("prefix", prefix),
                ("complement", complement),
                ("interaction", interaction),
                ("additive_prediction", prefix + complement),
                ("prefix_conditional_on_complement", exact - complement),
                ("complement_conditional_on_prefix", exact - prefix),
            ):
                for metric, value in vector_metrics(effect, exact).items():
                    values[f"{name}_{metric}"] = value
            parallel = values["interaction_aligned_recovery"] * exact
            orthogonal = interaction - parallel
            exact_norm = float(torch.linalg.vector_norm(exact))
            values["interaction_orthogonal_norm_ratio"] = (
                0.0 if exact_norm <= 1e-12 else float(torch.linalg.vector_norm(orthogonal)) / exact_norm
            )
            values["prefix_context_gain"] = (
                values["prefix_conditional_on_complement_aligned_recovery"]
                - values["prefix_aligned_recovery"]
            )
            values["complement_context_gain"] = (
                values["complement_conditional_on_prefix_aligned_recovery"]
                - values["complement_aligned_recovery"]
            )
            direction_examples.append(values)
            example_rows.append(values)
        metrics = summarize_scalar_metrics(direction_examples)
        directions[direction] = {
            "examples": len(direction_examples),
            "metrics": metrics,
            "probe_rows": probe_rows(raw_rows, direction),
            "spearman": {
                "interaction_norm_vs_response_tokens": spearman(
                    [row["interaction_norm_ratio"] for row in direction_examples],
                    [row["response_token_count"] for row in direction_examples],
                ),
                "interaction_norm_vs_exact_natural_recovery": spearman(
                    [row["interaction_norm_ratio"] for row in direction_examples],
                    [row["exact_natural_probe_recovery"] for row in direction_examples],
                ),
                "prefix_context_gain_vs_response_tokens": spearman(
                    [row["prefix_context_gain"] for row in direction_examples],
                    [row["response_token_count"] for row in direction_examples],
                ),
            },
            "four_state_algebra_max_abs": algebra_max_abs,
        }

    summary = {
        "schema_version": 1,
        "procedure": "day58-saved-four-state-k12-interaction-v1",
        "source": {
            "day57_contract_sha256": sha256_file(
                ROOT / "results/day-57/frozen-confirm-trace-acquisition-contract.json"
            ),
            "day57_execution_commit": execution["execution_commit"],
            "day57_confirmation_summary_sha256": sha256_file(
                ROOT / "results/day-57/confirmation-summary.json"
            ),
            "panel": "opened Day 57 fresh-confirmation panel",
        },
        "definitions": {
            "T": "identity-target K12 intervention",
            "P": "monitoring-prefix-only K12 installation",
            "C": "exact donor K12 with the monitoring-prefix delta removed",
            "E": "exact donor K12 intervention",
            "interaction": "E - P - C + T at the downstream 13-probe endpoint",
        },
        "directions": directions,
        "interpretation_rule": {
            "material_interaction": "median-concept interaction aligned recovery >= 0.15 in both directions",
            "context_dependence": "conditional prefix recovery exceeds prefix-alone recovery in both directions",
        },
    }
    material = all(
        value["metrics"]["interaction_aligned_recovery"]["median_concept"] >= 0.15
        and value["metrics"]["prefix_context_gain"]["median_concept"] > 0.0
        for value in directions.values()
    )
    summary["decision"] = (
        "material_nonlinear_prefix_complement_interaction"
        if material
        else "interaction_not_consistently_material"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(EXAMPLE_PATH, {"schema_version": 1, "rows": example_rows})
    summary["example_metrics_sha256"] = sha256_file(EXAMPLE_PATH)
    write_json_atomic(SUMMARY_PATH, summary)
    print(json.dumps({"decision": summary["decision"], "directions": {
        key: {
            "interaction": value["metrics"]["interaction_aligned_recovery"]["median_concept"],
            "interaction_norm": value["metrics"]["interaction_norm_ratio"]["median_concept"],
            "prefix_context_gain": value["metrics"]["prefix_context_gain"]["median_concept"],
        }
        for key, value in directions.items()
    }}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
