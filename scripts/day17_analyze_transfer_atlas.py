#!/usr/bin/env python3
"""Analyze the frozen Day 17 selected/null transfer atlas."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-17"
RAW_PATH = RESULT_DIR / "transfer-atlas-results.jsonl.gz"
CELL_PATH = RESULT_DIR / "transfer-atlas-cells.csv"
POPULATION_PATH = RESULT_DIR / "transfer-atlas-populations.csv"
SUMMARY_PATH = RESULT_DIR / "transfer-atlas-summary.json"
FIGURE_PATH = RESULT_DIR / "transfer-atlas.png"
PDF_PATH = RESULT_DIR / "transfer-atlas.pdf"
POPULATION_FIGURE_PATH = RESULT_DIR / "transfer-populations.png"
POPULATION_PDF_PATH = RESULT_DIR / "transfer-populations.pdf"
AUDIT_PATH = RESULT_DIR / "transfer-atlas-audit.json"
BOOTSTRAP_REPLICATES = 10000
SIGN_FLIP_REPLICATES = 10000
BOOTSTRAP_SEED = 15210
SIGN_FLIP_SEED = 15211


def read_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def ratio_estimate(
    numerator: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    if denominator <= 0 or np.any(denominator_boot <= 0):
        raise ValueError("unstable suppression denominator")
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return {
        "estimate": float(numerator.mean() / denominator),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def sign_flip_p(numerator: np.ndarray, signs: np.ndarray) -> float:
    observed = abs(float(numerator.mean()))
    shuffled = np.abs(signs @ numerator / numerator.size)
    return float((1 + np.count_nonzero(shuffled >= observed)) / (len(shuffled) + 1))


def bh_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return [float(value) for value in result]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows()
    plan = json.loads((ROOT / "results/day-15/frozen-site-shuffling-plan.json").read_text())
    selected = list(plan["selected_heads"])
    null = list(plan["null_heads"]["members"])
    population = selected + null
    selected_set = set(selected)
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        nested[row["concept"]][row["example_id"]][row["condition_id"]] = row

    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    sign_rng = np.random.default_rng(SIGN_FLIP_SEED)
    cells: list[dict[str, Any]] = []
    cell_numerators: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    population_rows: list[dict[str, Any]] = []

    for concept in ("deception", "harmful"):
        example_ids = sorted(nested[concept])
        if len(example_ids) != 16:
            raise ValueError(f"expected 16 {concept} examples")
        indices = bootstrap_rng.integers(0, 16, size=(BOOTSTRAP_REPLICATES, 16))
        signs = sign_rng.choice((-1.0, 1.0), size=(SIGN_FLIP_REPLICATES, 16))
        normal = np.asarray([nested[concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids])
        triggered = np.asarray([nested[concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids])
        condition_scores: dict[str, np.ndarray] = {}

        def scores(condition_id: str) -> np.ndarray:
            if condition_id not in condition_scores:
                condition_scores[condition_id] = np.asarray([
                    nested[concept][example_id][condition_id]["probe_score"]
                    for example_id in example_ids
                ])
            return condition_scores[condition_id]

        for destination_id in population:
            destination_role = "selected" if destination_id in selected_set else "null"
            destination_layer = int(destination_id.split(".")[0].split("_")[1])
            for source_id in population:
                source_role = "selected" if source_id in selected_set else "null"
                source_layer = int(source_id.split(".")[0].split("_")[1])
                if source_id == destination_id:
                    route = "identity"
                elif source_layer == destination_layer:
                    route = "within_layer"
                elif source_layer < destination_layer:
                    route = "earlier_to_later"
                else:
                    route = "later_to_earlier"
                n_n = scores(f"absolute:normal:normal:{source_id}->{destination_id}")
                n_t = scores(f"absolute:normal:correct_trigger:{source_id}->{destination_id}")
                t_n = scores(f"absolute:correct_trigger:normal:{source_id}->{destination_id}")
                t_t = scores(f"absolute:correct_trigger:correct_trigger:{source_id}->{destination_id}")
                numerators = {
                    ("induction", "absolute"): normal - n_t,
                    ("induction", "conditional"): n_n - n_t,
                    ("induction", "same_condition_mismatch"): normal - n_n,
                    ("rescue", "absolute"): t_n - triggered,
                    ("rescue", "conditional"): t_n - t_t,
                    ("rescue", "same_condition_mismatch"): t_t - triggered,
                }
                if source_role == destination_role == "selected":
                    for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
                        for estimand in ("delta", "delta_rms"):
                            patched = scores(f"{estimand}:{base}:{source_id}->{destination_id}")
                            numerators[(direction, estimand)] = normal - patched if direction == "induction" else patched - triggered
                for (direction, estimand), numerator in numerators.items():
                    result = ratio_estimate(numerator, normal, triggered, indices)
                    p_value = sign_flip_p(numerator, signs) if estimand in {"conditional", "delta", "delta_rms"} else float("nan")
                    cells.append({
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "source_id": source_id,
                        "destination_id": destination_id,
                        "source_role": source_role,
                        "destination_role": destination_role,
                        "route_class": route,
                        **result,
                        "sign_flip_p": p_value,
                        "bh_q": float("nan"),
                    })
                    cell_numerators[(concept, direction, estimand, source_id, destination_id)] = numerator

        families: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(cells):
            if row["concept"] == concept and row["estimand"] in {"conditional", "delta", "delta_rms"}:
                families[(row["direction"], row["estimand"])].append(index)
        for indices_in_family in families.values():
            adjusted = bh_adjust([cells[index]["sign_flip_p"] for index in indices_in_family])
            for index, q_value in zip(indices_in_family, adjusted, strict=True):
                cells[index]["bh_q"] = q_value

        for direction in ("rescue", "induction"):
            for estimand in ("conditional", "delta", "delta_rms"):
                role_groups: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
                route_groups: dict[str, list[np.ndarray]] = defaultdict(list)
                for source_id in population:
                    for destination_id in population:
                        key = (concept, direction, estimand, source_id, destination_id)
                        if key not in cell_numerators:
                            continue
                        source_role = "selected" if source_id in selected_set else "null"
                        destination_role = "selected" if destination_id in selected_set else "null"
                        # The portable-code population estimand concerns non-original
                        # routes. Preserve identity cells in the atlas and route-class
                        # summaries, but exclude them from selected/null factorial means.
                        if source_id != destination_id:
                            role_groups[(source_role, destination_role)].append(cell_numerators[key])
                        if source_role == destination_role == "selected":
                            source_layer = int(source_id.split(".")[0].split("_")[1])
                            destination_layer = int(destination_id.split(".")[0].split("_")[1])
                            route = "identity" if source_id == destination_id else "within_layer" if source_layer == destination_layer else "earlier_to_later" if source_layer < destination_layer else "later_to_earlier"
                            route_groups[route].append(cell_numerators[key])
                for (source_role, destination_role), values in sorted(role_groups.items()):
                    numerator = np.stack(values).mean(axis=0)
                    result = ratio_estimate(numerator, normal, triggered, indices)
                    population_rows.append({
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "group_type": "role_factorial",
                        "group_id": f"{source_role}_to_{destination_role}",
                        "pair_count": len(values),
                        **result,
                    })
                for route, values in sorted(route_groups.items()):
                    numerator = np.stack(values).mean(axis=0)
                    result = ratio_estimate(numerator, normal, triggered, indices)
                    population_rows.append({
                        "concept": concept,
                        "direction": direction,
                        "estimand": estimand,
                        "group_type": "selected_route_class",
                        "group_id": route,
                        "pair_count": len(values),
                        **result,
                    })
                if estimand == "conditional":
                    selected_values = np.stack(role_groups[("selected", "selected")]).mean(axis=0)
                    for control_group in (("selected", "null"), ("null", "selected"), ("null", "null")):
                        control_values = np.stack(role_groups[control_group]).mean(axis=0)
                        result = ratio_estimate(selected_values - control_values, normal, triggered, indices)
                        population_rows.append({
                            "concept": concept,
                            "direction": direction,
                            "estimand": estimand,
                            "group_type": "selected_selected_contrast",
                            "group_id": f"selected_to_selected-minus-{control_group[0]}_to_{control_group[1]}",
                            "pair_count": len(role_groups[("selected", "selected")]),
                            **result,
                        })

    write_csv(CELL_PATH, cells)
    write_csv(POPULATION_PATH, population_rows)

    lookup = {(row["concept"], row["direction"], row["estimand"], row["source_id"], row["destination_id"]): row for row in cells}
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    values = [lookup[(concept, direction, "conditional", source, destination)]["estimate"] for concept in ("deception", "harmful") for direction in ("rescue", "induction") for source in selected for destination in selected]
    limit = max(abs(min(values)), abs(max(values)))
    for row_index, concept in enumerate(("deception", "harmful")):
        for column_index, direction in enumerate(("rescue", "induction")):
            matrix = np.asarray([[lookup[(concept, direction, "conditional", source, destination)]["estimate"] for source in selected] for destination in selected])
            axis = axes[row_index, column_index]
            image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
            axis.set_title(f"{concept}: {direction}")
            axis.set_xticks(range(len(selected)), [item.replace("layer_", "L").replace(".head_", "H") for item in selected], rotation=90, fontsize=7)
            axis.set_yticks(range(len(selected)), [item.replace("layer_", "L").replace(".head_", "H") for item in selected], fontsize=7)
            axis.set_xlabel("source head")
            axis.set_ylabel("destination head")
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label="route-matched conditional fraction",
        shrink=0.78,
        pad=0.02,
    )
    figure.suptitle("Day 17 selected-head conditional transfer atlas")
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    role_order = ["selected_to_selected", "selected_to_null", "null_to_selected", "null_to_null"]
    population_lookup = {(row["concept"], row["direction"], row["estimand"], row["group_type"], row["group_id"]): row for row in population_rows}
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row_index, concept in enumerate(("deception", "harmful")):
        for column_index, direction in enumerate(("rescue", "induction")):
            axis = axes[row_index, column_index]
            selected_rows = [population_lookup[(concept, direction, "conditional", "role_factorial", group_id)] for group_id in role_order]
            points = [row["estimate"] for row in selected_rows]
            lower = [row["estimate"] - row["ci_low"] for row in selected_rows]
            upper = [row["ci_high"] - row["estimate"] for row in selected_rows]
            axis.bar(range(4), points, color=["#1b9e77", "#66a61e", "#7570b3", "#999999"])
            axis.errorbar(range(4), points, yerr=[lower, upper], fmt="none", color="black", capsize=3)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_xticks(range(4), ["S→S", "S→N", "N→S", "N→N"])
            axis.set_title(f"{concept}: {direction}")
            axis.set_ylabel("mean conditional fraction")
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Day 17 selected/null source and destination populations")
    figure.tight_layout()
    figure.savefig(POPULATION_FIGURE_PATH, dpi=180)
    figure.savefig(POPULATION_PDF_PATH)
    plt.close(figure)

    significant = [row for row in cells if np.isfinite(row["bh_q"]) and row["bh_q"] <= 0.05]
    summary = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day17-analysis",
        "status": "complete",
        "n_raw_rows": len(rows),
        "n_cell_metrics": len(cells),
        "n_population_metrics": len(population_rows),
        "bh_significant_cell_count": len(significant),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "sign_flip_replicates": SIGN_FLIP_REPLICATES,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    expected_rows = 32 * (2 + 4 * 24 * 24 + 4 * 12 * 12)
    checks = {
        "raw_row_count": len(rows) == expected_rows,
        "unique_raw_keys": len({(row["example_id"], row["condition_id"]) for row in rows}) == expected_rows,
        "cell_count": len(cells) == 2 * (6 * 24 * 24 + 4 * 12 * 12),
        "all_population_finite": all(np.isfinite(row[key]) for row in population_rows for key in ("estimate", "ci_low", "ci_high")),
        "figures_written": all(path.is_file() for path in (FIGURE_PATH, PDF_PATH, POPULATION_FIGURE_PATH, POPULATION_PDF_PATH)),
    }
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day17-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 17 audit failed")


if __name__ == "__main__":
    main()
