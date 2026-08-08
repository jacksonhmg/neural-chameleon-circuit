#!/usr/bin/env python3
"""Analyze and audit the sealed Day 23 behavioral transport grid."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-23"
AUTHORIZATION_PATH = ROOT / "results/day-22/behavioral-transport-authorization.json"
RAW_PATH = RESULT_DIR / "behavioral-transport-results.jsonl.gz"
MAPPING_METRICS_PATH = RESULT_DIR / "mapping-behavioral-metrics.csv"
POPULATION_PATH = RESULT_DIR / "behavioral-transport-cells.csv"
GATE_PATH = RESULT_DIR / "behavioral-transport-gate.json"
FIGURE_PATH = RESULT_DIR / "behavioral-transport.png"
PDF_PATH = RESULT_DIR / "behavioral-transport.pdf"
AUDIT_PATH = RESULT_DIR / "day23-audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def summarize(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("summary values must be one-dimensional and finite")
    boots = values[indices].mean(axis=1)
    low, high = np.quantile(boots, [0.025, 0.975])
    return {"estimate": float(values.mean()), "ci_low": float(low), "ci_high": float(high)}


def mapping_metrics(
    nested: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    authorization: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    mapping_ids = [row["mapping_id"] for row in authorization["mappings"]]
    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept][1])
        indices = rng.integers(0, len(example_ids), size=(10000, len(example_ids)))
        for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
            for mapping_id in (*mapping_ids, "selected_k12_identity"):
                roles = ("identity",) if mapping_id == "selected_k12_identity" else ("selected", "null")
                for source_role in roles:
                    condition_id = f"delta:{base}:{source_role}:{mapping_id}"
                    values = [nested[concept][1][example_id][condition_id] for example_id in example_ids]
                    row: dict[str, Any] = {"concept": concept, "direction": direction, "mapping_id": mapping_id, "source_role": source_role}
                    for field in ("normalized_probe_effect", "directional_coefficient", "directional_cosine", "kl_from_base", "nll_shift_from_base", "top1_agreement"):
                        result = summarize(np.asarray([float(value[field]) for value in values]), indices)
                        row[f"{field}_estimate"] = result["estimate"]
                        row[f"{field}_ci_low"] = result["ci_low"]
                        row[f"{field}_ci_high"] = result["ci_high"]
                    metrics.append(row)
    return metrics


def population_cells(
    nested: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    authorization: Mapping[str, Any],
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    mapping_ids = [row["mapping_id"] for row in authorization["mappings"]]
    cells: list[dict[str, Any]] = []
    threshold = float(authorization["inference"]["directional_coefficient_minimum"])
    nll_lower, nll_upper = authorization["inference"]["nll_equivalence_interval"]
    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept][1])
        indices = rng.integers(0, len(example_ids), size=(10000, len(example_ids)))
        for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
            by_role: dict[str, dict[str, np.ndarray]] = {}
            for role in ("selected", "null"):
                role_values: dict[str, list[float]] = defaultdict(list)
                for example_id in example_ids:
                    rows = [nested[concept][1][example_id][f"delta:{base}:{role}:{mapping_id}"] for mapping_id in mapping_ids]
                    for field in ("normalized_probe_effect", "directional_coefficient", "directional_cosine", "kl_from_base", "nll_shift_from_base", "top1_agreement"):
                        role_values[field].append(float(np.mean([row[field] for row in rows])))
                by_role[role] = {field: np.asarray(values) for field, values in role_values.items()}
            summaries: dict[str, dict[str, dict[str, float]]] = {"selected": {}, "null": {}, "selected_minus_null": {}}
            for field in by_role["selected"]:
                summaries["selected"][field] = summarize(by_role["selected"][field], indices)
                summaries["null"][field] = summarize(by_role["null"][field], indices)
                summaries["selected_minus_null"][field] = summarize(by_role["selected"][field] - by_role["null"][field], indices)
            probe_gate = (
                summaries["selected"]["normalized_probe_effect"]["ci_low"] > 0
                and summaries["selected_minus_null"]["normalized_probe_effect"]["ci_low"] > 0
            )
            directional = summaries["selected"]["directional_coefficient"]
            directional_contrast = summaries["selected_minus_null"]["directional_coefficient"]
            directionally_meaningful = directional["estimate"] >= threshold and directional["ci_low"] > 0 and directional_contrast["ci_low"] > 0
            kl_bound = float(authorization["inference"]["kl_equivalence_upper"][concept])
            kl_equivalent = summaries["selected"]["kl_from_base"]["ci_high"] <= kl_bound
            nll_summary = summaries["selected"]["nll_shift_from_base"]
            nll_equivalent = nll_summary["ci_low"] >= float(nll_lower) and nll_summary["ci_high"] <= float(nll_upper)
            if not probe_gate:
                disposition = "inconclusive"
            elif directionally_meaningful and nll_equivalent:
                disposition = "behaviorally_coupled_transport"
            elif directional["ci_high"] < threshold and kl_equivalent and nll_equivalent:
                disposition = "behavior_preserving_portable_evasion"
            elif not directionally_meaningful and not (kl_equivalent and nll_equivalent):
                disposition = "nonspecific_disruption"
            else:
                disposition = "mixed"
            cells.append(
                {
                    "concept": concept,
                    "direction": direction,
                    "disposition": disposition,
                    "probe_gate": probe_gate,
                    "directionally_meaningful": directionally_meaningful,
                    "kl_equivalent": kl_equivalent,
                    "nll_equivalent": nll_equivalent,
                    "directional_threshold": threshold,
                    "kl_equivalence_bound": kl_bound,
                    "nll_equivalence_lower": nll_lower,
                    "nll_equivalence_upper": nll_upper,
                    "summaries": summaries,
                }
            )
    return cells


def negative_summary(
    nested: Mapping[str, Mapping[int, Mapping[str, Mapping[str, Any]]]],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    mapping_ids = [row["mapping_id"] for row in authorization["mappings"]]
    output: dict[str, Any] = {}
    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept][0])
        output[concept] = {}
        for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
            selected = [nested[concept][0][example_id][f"delta:{base}:selected:{mapping_id}"] for example_id in example_ids for mapping_id in mapping_ids]
            output[concept][direction] = {
                "example_count": len(example_ids),
                "mean_absolute_probe_shift": float(np.mean([abs(row["probe_shift_from_base"]) for row in selected])),
                "mean_kl_from_base": float(np.mean([row["kl_from_base"] for row in selected])),
                "mean_absolute_nll_shift": float(np.mean([abs(row["nll_shift_from_base"]) for row in selected])),
                "mean_top1_agreement": float(np.mean([row["top1_agreement"] for row in selected])),
            }
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    rows = load_rows()
    nested: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        nested[row["concept"]][int(row["label"])][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(int(authorization["inference"]["bootstrap_seed"]))
    mapping = mapping_metrics(nested, authorization, rng)
    cells = population_cells(nested, authorization, rng)
    write_csv(MAPPING_METRICS_PATH, mapping)
    flat_cells = []
    for cell in cells:
        flat = {key: value for key, value in cell.items() if key != "summaries"}
        for role, fields in cell["summaries"].items():
            for field, summary in fields.items():
                for statistic, value in summary.items():
                    flat[f"{role}_{field}_{statistic}"] = value
        flat_cells.append(flat)
    write_csv(POPULATION_PATH, flat_cells)
    dispositions = sorted({cell["disposition"] for cell in cells})
    gate = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day23-behavioral-gate",
        "status": "complete",
        "overall_disposition": dispositions[0] if len(dispositions) == 1 else "heterogeneous:" + ",".join(dispositions),
        "cells": cells,
        "negative_controls": negative_summary(nested, authorization),
        "authorization_sha256": sha256_file(AUTHORIZATION_PATH),
        "raw_results_sha256": sha256_file(RAW_PATH),
    }
    GATE_PATH.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"deception": "#1b9e77", "harmful": "#d95f02"}
    marks = {"induction": "o", "rescue": "s"}
    for concept in ("deception", "harmful"):
        for direction in ("induction", "rescue"):
            cell = next(row for row in cells if row["concept"] == concept and row["direction"] == direction)
            label = f"{concept} {direction}"
            probe = cell["summaries"]["selected"]["normalized_probe_effect"]
            coeff = cell["summaries"]["selected"]["directional_coefficient"]
            axes[0].errorbar(probe["estimate"], coeff["estimate"], xerr=[[probe["estimate"] - probe["ci_low"]], [probe["ci_high"] - probe["estimate"]]], yerr=[[coeff["estimate"] - coeff["ci_low"]], [coeff["ci_high"] - coeff["estimate"]]], marker=marks[direction], color=colors[concept], linestyle="none", label=label)
            kl = cell["summaries"]["selected"]["kl_from_base"]
            nll = cell["summaries"]["selected"]["nll_shift_from_base"]
            index = len(axes[1].lines)
            axes[1].errorbar(index, kl["estimate"], yerr=[[kl["estimate"] - kl["ci_low"]], [kl["ci_high"] - kl["estimate"]]], marker=marks[direction], color=colors[concept], linestyle="none")
            axes[2].errorbar(index, nll["estimate"], yerr=[[nll["estimate"] - nll["ci_low"]], [nll["ci_high"] - nll["estimate"]]], marker=marks[direction], color=colors[concept], linestyle="none")
    axes[0].axhline(0.10, color="black", linestyle="--", linewidth=0.8)
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_xlabel("normalized probe transport")
    axes[0].set_ylabel("natural-direction coefficient")
    axes[0].set_title("probe versus output transport")
    axes[0].legend(frameon=False, fontsize=8)
    labels = [f"{c[:3]} {d[:3]}" for c in ("deception", "harmful") for d in ("induction", "rescue")]
    for axis, title, ylabel in ((axes[1], "distribution change", "KL from destination base"), (axes[2], "reference likelihood", "NLL shift from base")):
        axis.set_xticks(range(4), labels, rotation=25)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.axhline(0, color="black", linewidth=0.7)
    figure.suptitle("Final-map behavioral transport")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    expected_ids = set(authorization["examples"]["teacher_forced"]["example_ids"])
    intervention_fields = ("probe_score", "response_nll", "kl_from_base", "nll_shift_from_base", "directional_coefficient", "directional_cosine", "top1_agreement")
    checks = {
        "exact_rows": len(rows) == int(authorization["grid"]["teacher_forced_expected_rows"]),
        "unique_rows": len({(row["example_id"], row["condition_id"]) for row in rows}) == len(rows),
        "exact_examples": {row["example_id"] for row in rows} == expected_ids,
        "twenty_conditions_each": set(sum(1 for row in rows if row["example_id"] == example_id) for example_id in expected_ids) == {20},
        "finite_intervention_metrics": all(np.isfinite(float(row[field])) for row in rows if row["record_type"] == "intervention" for field in intervention_fields),
        "authorization_hash": {row["authorization_sha256"] for row in rows} == {sha256_file(AUTHORIZATION_PATH)},
        "prospective_boundary": all(row["prospective_behavioral_outcome"] and not row["independent_dataset_confirmation"] for row in rows),
        "four_cells": len(cells) == 4,
        "figures": FIGURE_PATH.is_file() and PDF_PATH.is_file(),
    }
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day23-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks, "raw_results_sha256": sha256_file(RAW_PATH)}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 23 audit failed")


if __name__ == "__main__":
    main()

