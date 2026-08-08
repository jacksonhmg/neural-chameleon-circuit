#!/usr/bin/env python3
"""Analyze benign transport geometry and held-out safety-atlas association."""

from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-18"
RAW_PATH = RESULT_DIR / "geometry-transfer-results.jsonl.gz"
GEOMETRY_PATH = RESULT_DIR / "geometry-captures.npz"
PROJECTION_PATH = RESULT_DIR / "projection-features.npz"
CELL_PATH = RESULT_DIR / "geometry-transfer-cells.csv"
GEOMETRY_SUMMARY_PATH = RESULT_DIR / "geometry-summary.json"
PREDICTION_PATH = RESULT_DIR / "geometry-transfer-prediction.csv"
FIGURE_PATH = RESULT_DIR / "geometry-overview.png"
PDF_PATH = RESULT_DIR / "geometry-overview.pdf"
TRANSPORT_FIGURE_PATH = RESULT_DIR / "aligned-transport.png"
TRANSPORT_PDF_PATH = RESULT_DIR / "aligned-transport.pdf"
AUDIT_PATH = RESULT_DIR / "geometry-audit.json"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 15210


def read_rows() -> list[dict[str, Any]]:
    with gzip.open(RAW_PATH, "rt") as handle:
        return [json.loads(line) for line in handle]


def ratio_with_boot(
    numerator: np.ndarray,
    normal: np.ndarray,
    triggered: np.ndarray,
    indices: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    denominator = float((normal - triggered).mean())
    denominator_boot = (normal[indices] - triggered[indices]).mean(axis=1)
    if denominator <= 0 or np.any(denominator_boot <= 0):
        raise ValueError("unstable benign suppression denominator")
    boots = numerator[indices].mean(axis=1) / denominator_boot
    low, high = np.quantile(boots, [0.025, 0.975])
    return ({
        "estimate": float(numerator.mean() / denominator),
        "ci_low": float(low),
        "ci_high": float(high),
    }, boots)


def cosine_matrix(means: np.ndarray) -> np.ndarray:
    normalized = means / np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-12)
    return normalized @ normalized.T


def centered_gram(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    count = gram.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    return centering @ gram @ centering


def cka_matrix(values: np.ndarray) -> np.ndarray:
    grams = [centered_gram(values[:, index, :]) for index in range(values.shape[1])]
    result = np.empty((len(grams), len(grams)), dtype=np.float64)
    norms = [np.linalg.norm(gram) for gram in grams]
    for first in range(len(grams)):
        for second in range(len(grams)):
            result[first, second] = float(
                np.sum(grams[first] * grams[second])
                / max(norms[first] * norms[second], 1e-12)
            )
    return result


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks


def correlation(first: np.ndarray, second: np.ndarray, *, ranks: bool = False) -> float:
    if ranks:
        first, second = rankdata(first), rankdata(second)
    return float(np.corrcoef(first, second)[0, 1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_rows()
    geometry = np.load(GEOMETRY_PATH)
    projection = np.load(PROJECTION_PATH)
    head_ids = [str(item) for item in geometry["head_ids"]]
    selected = [str(item) for item in projection["head_ids"]]
    raw_deltas = geometry["raw_deltas"].astype(np.float64)
    residual_deltas = geometry["residual_deltas"].astype(np.float64)
    delta_rms = geometry["delta_rms"].astype(np.float64)
    if raw_deltas.shape[:2] != (64, 24) or residual_deltas.shape[:2] != (64, 24):
        raise ValueError("unexpected geometry archive shape")

    raw_cosine = cosine_matrix(raw_deltas.mean(axis=0))
    residual_cosine = cosine_matrix(residual_deltas.mean(axis=0))
    raw_cka = cka_matrix(raw_deltas)
    residual_cka = cka_matrix(residual_deltas)
    pooled = torch.from_numpy(residual_deltas.reshape(-1, residual_deltas.shape[-1]).astype(np.float32))
    pooled = pooled - pooled.mean(dim=0, keepdim=True)
    _u, singular, _v = torch.pca_lowrank(pooled, q=32, center=False, niter=4)
    explained = (singular.square() / pooled.square().sum().clamp(min=1e-12)).numpy()

    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in rows:
        nested[row["concept"]][row["example_id"]][row["condition_id"]] = row
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cell_rows: list[dict[str, Any]] = []
    cell_boots: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    for concept in sorted(nested):
        example_ids = sorted(nested[concept])
        indices = rng.integers(0, 16, size=(BOOTSTRAP_REPLICATES, 16))
        normal = np.asarray([nested[concept][example_id]["baseline:normal"]["probe_score"] for example_id in example_ids])
        triggered = np.asarray([nested[concept][example_id]["baseline:correct_trigger"]["probe_score"] for example_id in example_ids])
        for direction, base in (("induction", "normal"), ("rescue", "correct_trigger")):
            for transport in ("raw", "rms", "aligned"):
                for source_id in selected:
                    for destination_id in selected:
                        condition_id = f"{transport}:{base}:{source_id}->{destination_id}"
                        patched = np.asarray([nested[concept][example_id][condition_id]["probe_score"] for example_id in example_ids])
                        numerator = normal - patched if direction == "induction" else patched - triggered
                        result, boots = ratio_with_boot(numerator, normal, triggered, indices)
                        key = (concept, direction, transport, source_id, destination_id)
                        cell_boots[key] = boots
                        source_layer = int(source_id.split(".")[0].split("_")[1])
                        destination_layer = int(destination_id.split(".")[0].split("_")[1])
                        route = "identity" if source_id == destination_id else "within_layer" if source_layer == destination_layer else "earlier_to_later" if source_layer < destination_layer else "later_to_earlier"
                        cell_rows.append({
                            "scope": concept,
                            "direction": direction,
                            "transport": transport,
                            "source_id": source_id,
                            "destination_id": destination_id,
                            "route_class": route,
                            **result,
                        })
    macro_rows = []
    concepts = sorted(nested)
    for direction in ("rescue", "induction"):
        for transport in ("raw", "rms", "aligned"):
            for source_id in selected:
                for destination_id in selected:
                    relevant = [row for row in cell_rows if row["scope"] in concepts and row["direction"] == direction and row["transport"] == transport and row["source_id"] == source_id and row["destination_id"] == destination_id]
                    point = float(np.mean([row["estimate"] for row in relevant]))
                    boots = np.stack([cell_boots[(concept, direction, transport, source_id, destination_id)] for concept in concepts]).mean(axis=0)
                    low, high = np.quantile(boots, [0.025, 0.975])
                    source_layer = int(source_id.split(".")[0].split("_")[1])
                    destination_layer = int(destination_id.split(".")[0].split("_")[1])
                    route = "identity" if source_id == destination_id else "within_layer" if source_layer == destination_layer else "earlier_to_later" if source_layer < destination_layer else "later_to_earlier"
                    macro_rows.append({
                        "scope": "discovery_macro",
                        "direction": direction,
                        "transport": transport,
                        "source_id": source_id,
                        "destination_id": destination_id,
                        "route_class": route,
                        "estimate": point,
                        "ci_low": float(low),
                        "ci_high": float(high),
                    })
    all_cell_rows = cell_rows + macro_rows
    write_csv(CELL_PATH, all_cell_rows)

    index_by_head = {head_id: index for index, head_id in enumerate(head_ids)}
    selected_index = {head_id: index for index, head_id in enumerate(selected)}
    projection_cosine = projection["projection_cosine"].astype(np.float64)
    macro_lookup = {(row["direction"], row["transport"], row["source_id"], row["destination_id"]): row["estimate"] for row in macro_rows}
    feature_rows = []
    x_rows = []
    benign_target = []
    pair_ids = []
    for source_id in selected:
        for destination_id in selected:
            if source_id == destination_id:
                continue
            source_index = index_by_head[source_id]
            destination_index = index_by_head[destination_id]
            source_selected_index = selected_index[source_id]
            destination_selected_index = selected_index[destination_id]
            source_layer = int(source_id.split(".")[0].split("_")[1])
            destination_layer = int(destination_id.split(".")[0].split("_")[1])
            features = [
                raw_cosine[destination_index, source_index],
                residual_cosine[destination_index, source_index],
                raw_cka[destination_index, source_index],
                residual_cka[destination_index, source_index],
                projection_cosine[destination_selected_index, source_selected_index],
                float(delta_rms[:, source_index].mean() / max(delta_rms[:, destination_index].mean(), 1e-12)),
                float(source_layer == destination_layer),
                float(source_layer - destination_layer),
                float(abs(source_layer - destination_layer)),
            ]
            target = float(np.mean([
                macro_lookup[(direction, "raw", source_id, destination_id)]
                for direction in ("rescue", "induction")
            ]))
            pair_id = f"{source_id}->{destination_id}"
            pair_ids.append(pair_id)
            x_rows.append(features)
            benign_target.append(target)
            feature_rows.append({
                "pair_id": pair_id,
                "source_id": source_id,
                "destination_id": destination_id,
                "raw_cosine": features[0],
                "residual_cosine": features[1],
                "raw_cka": features[2],
                "residual_cka": features[3],
                "projection_cosine": features[4],
                "delta_rms_ratio": features[5],
                "same_layer": features[6],
                "signed_depth": features[7],
                "absolute_depth": features[8],
                "benign_raw_transfer": target,
                "predicted_benign_transfer": 0.0,
                "safety_delta_transfer": 0.0,
            })
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(benign_target, dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    standardized = (x - mean) / np.where(scale > 1e-12, scale, 1.0)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predicted = design @ coefficients

    safety_rows = list(csv.DictReader((ROOT / "results/day-17/transfer-atlas-cells.csv").open()))
    safety_values: dict[str, list[float]] = defaultdict(list)
    for row in safety_rows:
        if row["estimand"] != "delta" or row["source_role"] != "selected" or row["destination_role"] != "selected" or row["source_id"] == row["destination_id"]:
            continue
        safety_values[f"{row['source_id']}->{row['destination_id']}"] .append(float(row["estimate"]))
    safety_target = np.asarray([np.mean(safety_values[pair_id]) for pair_id in pair_ids])
    for index, row in enumerate(feature_rows):
        row["predicted_benign_transfer"] = float(predicted[index])
        row["safety_delta_transfer"] = float(safety_target[index])
    write_csv(PREDICTION_PATH, feature_rows)

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    matrices = [(raw_cosine, "Raw mean-delta cosine"), (residual_cosine, "Residual mean-delta cosine"), (raw_cka, "Raw linear CKA"), (residual_cka, "Residual linear CKA")]
    for axis, (matrix, title) in zip(axes.flat, matrices, strict=True):
        image = axis.imshow(matrix[:12, :12], cmap="viridis", vmin=-1 if "cosine" in title else 0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("source selected head")
        axis.set_ylabel("destination selected head")
        axis.set_xticks(range(12), [item.replace("layer_", "L").replace(".head_", "H") for item in selected], rotation=90, fontsize=7)
        axis.set_yticks(range(12), [item.replace("layer_", "L").replace(".head_", "H") for item in selected], fontsize=7)
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("Day 18 benign trigger-delta geometry")
    figure.tight_layout()
    figure.savefig(FIGURE_PATH, dpi=180)
    figure.savefig(PDF_PATH)
    plt.close(figure)

    population_points = defaultdict(list)
    for row in macro_rows:
        if row["source_id"] != row["destination_id"]:
            population_points[(row["direction"], row["transport"])].append(row["estimate"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, direction in zip(axes, ("rescue", "induction"), strict=True):
        values = [np.mean(population_points[(direction, transport)]) for transport in ("raw", "rms", "aligned")]
        axis.bar(("raw", "RMS-matched", "projection-aligned"), values, color=("#1b9e77", "#7570b3", "#d95f02"))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(direction)
        axis.set_ylabel("mean non-original causal fraction")
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Day 18 benign transport coordinate comparison")
    figure.tight_layout()
    figure.savefig(TRANSPORT_FIGURE_PATH, dpi=180)
    figure.savefig(TRANSPORT_PDF_PATH)
    plt.close(figure)

    geometry_summary = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day18-analysis",
        "status": "complete",
        "residual_pca_explained_variance_first_32": [float(value) for value in explained],
        "residual_pca_cumulative": {str(count): float(explained[:count].sum()) for count in (1, 2, 4, 8, 16, 32)},
        "benign_fit": {
            "pearson": correlation(predicted, y),
            "spearman": correlation(predicted, y, ranks=True),
        },
        "held_out_safety_association": {
            "pearson": correlation(predicted, safety_target),
            "spearman": correlation(predicted, safety_target, ranks=True),
            "label": "held-out characterization; no safety outcome fit the predictor"
        },
    }
    GEOMETRY_SUMMARY_PATH.write_text(json.dumps(geometry_summary, indent=2, sort_keys=True) + "\n")
    expected_rows = 64 * (2 + 2 * 3 * 12 * 12)
    checks = {
        "raw_row_count": len(rows) == expected_rows,
        "unique_raw_keys": len({(row["example_id"], row["condition_id"]) for row in rows}) == expected_rows,
        "geometry_shapes": raw_deltas.shape[:2] == (64, 24) and residual_deltas.shape[:2] == (64, 24),
        "all_metrics_finite": all(np.isfinite(row[key]) for row in all_cell_rows for key in ("estimate", "ci_low", "ci_high")),
        "prediction_complete": len(safety_values) == 132,
        "figures_written": all(path.is_file() for path in (FIGURE_PATH, PDF_PATH, TRANSPORT_FIGURE_PATH, TRANSPORT_PDF_PATH)),
    }
    audit = {"schema_version": 1, "procedure": "site-shuffling-v1-day18-audit", "status": "pass" if all(checks.values()) else "fail", "checks": checks}
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if audit["status"] != "pass":
        raise RuntimeError("Day 18 audit failed")


if __name__ == "__main__":
    main()
