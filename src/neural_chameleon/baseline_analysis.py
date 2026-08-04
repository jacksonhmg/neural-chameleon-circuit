"""Frozen Day 5 baseline estimators and paired bootstrap analysis."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata


CONDITIONS = ("normal", "correct_trigger", "irrelevant_trigger")
CONDITION_INDEX = {condition: index for index, condition in enumerate(CONDITIONS)}
SCALAR_FIELDS = ("probe_score", "activation_l2", "activation_rms", "response_nll")


def higher_quantile(values: np.ndarray, quantile: float = 0.95, axis=None) -> np.ndarray:
    """Frozen empirical quantile convention for the 5%-FPR threshold."""
    return np.quantile(values, quantile, axis=axis, method="higher")


def auroc_from_scores(positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney AUROC with half credit for ties."""
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)
    combined = np.concatenate([positive, negative])
    ranks = rankdata(combined, method="average")
    n_positive = len(positive)
    n_negative = len(negative)
    rank_sum = ranks[:n_positive].sum()
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def bootstrap_aurocs(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Vectorized row-wise AUROC for paired bootstrap matrices."""
    combined = np.concatenate([positive, negative], axis=1)
    ranks = rankdata(combined, method="average", axis=1)
    n_positive = positive.shape[1]
    n_negative = negative.shape[1]
    rank_sums = ranks[:, :n_positive].sum(axis=1)
    return (
        rank_sums - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def interval(samples: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - confidence_level) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return float(low), float(high)


def estimate(value: float, samples: np.ndarray) -> dict[str, float]:
    low, high = interval(samples)
    return {
        "estimate": float(value),
        "ci_low": low,
        "ci_high": high,
    }


def describe(values: np.ndarray, bootstrap_means: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "mean": estimate(float(values.mean()), bootstrap_means),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def build_matrix(
    records: Iterable[dict[str, Any]], field: str
) -> tuple[list[str], np.ndarray]:
    by_example: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        by_example[record["example_id"]][record["condition"]] = float(record[field])
    example_ids = sorted(by_example)
    for example_id in example_ids:
        if set(by_example[example_id]) != set(CONDITIONS):
            raise ValueError(f"Incomplete condition rows for {example_id}/{field}")
    matrix = np.asarray(
        [
            [by_example[example_id][condition] for condition in CONDITIONS]
            for example_id in example_ids
        ],
        dtype=float,
    )
    return example_ids, matrix


def summarize_group(
    records: list[dict[str, Any]],
    rng: np.random.Generator,
    replicates: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, float]]:
    positive_records = [record for record in records if record["label"] == 1]
    negative_records = [record for record in records if record["label"] == 0]
    matrices: dict[int, dict[str, np.ndarray]] = {1: {}, 0: {}}
    ids_by_label: dict[int, list[str]] = {}
    for label, label_records in ((1, positive_records), (0, negative_records)):
        for field in SCALAR_FIELDS:
            example_ids, matrix = build_matrix(label_records, field)
            if label not in ids_by_label:
                ids_by_label[label] = example_ids
            elif example_ids != ids_by_label[label]:
                raise ValueError(f"Mismatched example ordering for label {label}")
            matrices[label][field] = matrix

    n_positive = len(ids_by_label[1])
    n_negative = len(ids_by_label[0])
    positive_indices = rng.integers(0, n_positive, size=(replicates, n_positive))
    negative_indices = rng.integers(0, n_negative, size=(replicates, n_negative))
    positive_boot = {
        field: matrix[positive_indices]
        for field, matrix in matrices[1].items()
    }
    negative_boot = {
        field: matrix[negative_indices]
        for field, matrix in matrices[0].items()
    }

    normal_index = CONDITION_INDEX["normal"]
    normal_negative_scores = matrices[0]["probe_score"][:, normal_index]
    threshold = float(higher_quantile(normal_negative_scores))
    threshold_boot = higher_quantile(
        negative_boot["probe_score"][:, :, normal_index], axis=1
    )

    conditions: dict[str, Any] = {}
    boot_metrics: dict[str, np.ndarray] = {}
    point_metrics: dict[str, float] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        positive_scores = matrices[1]["probe_score"][:, condition_index]
        negative_scores = matrices[0]["probe_score"][:, condition_index]
        positive_score_boot = positive_boot["probe_score"][:, :, condition_index].mean(axis=1)
        negative_score_boot = negative_boot["probe_score"][:, :, condition_index].mean(axis=1)
        auc = auroc_from_scores(positive_scores, negative_scores)
        auc_boot = bootstrap_aurocs(
            positive_boot["probe_score"][:, :, condition_index],
            negative_boot["probe_score"][:, :, condition_index],
        )
        tpr = float(np.mean(positive_scores > threshold))
        fpr = float(np.mean(negative_scores > threshold))
        tpr_boot = np.mean(
            positive_boot["probe_score"][:, :, condition_index]
            > threshold_boot[:, None],
            axis=1,
        )
        fpr_boot = np.mean(
            negative_boot["probe_score"][:, :, condition_index]
            > threshold_boot[:, None],
            axis=1,
        )

        condition_summary: dict[str, Any] = {
            "probe_score": {
                "positive": describe(positive_scores, positive_score_boot),
                "negative": describe(negative_scores, negative_score_boot),
            },
            "classification": {
                "auroc": estimate(auc, auc_boot),
                "tpr_at_5pct_fpr": estimate(tpr, tpr_boot),
                "fpr_at_normal_threshold": estimate(fpr, fpr_boot),
            },
        }
        for field in ("activation_l2", "activation_rms", "response_nll"):
            positive_values = matrices[1][field][:, condition_index]
            negative_values = matrices[0][field][:, condition_index]
            positive_means = positive_boot[field][:, :, condition_index].mean(axis=1)
            negative_means = negative_boot[field][:, :, condition_index].mean(axis=1)
            condition_summary[field] = {
                "positive": describe(positive_values, positive_means),
                "negative": describe(negative_values, negative_means),
                "class_balanced_mean": estimate(
                    float((positive_values.mean() + negative_values.mean()) / 2),
                    (positive_means + negative_means) / 2,
                ),
            }
            boot_metrics[f"{field}.{condition}"] = (positive_means + negative_means) / 2
            point_metrics[f"{field}.{condition}"] = float(
                (positive_values.mean() + negative_values.mean()) / 2
            )

        conditions[condition] = condition_summary
        boot_metrics[f"auroc.{condition}"] = auc_boot
        boot_metrics[f"tpr_at_5pct_fpr.{condition}"] = tpr_boot
        boot_metrics[f"positive_probe_score.{condition}"] = positive_score_boot
        boot_metrics[f"negative_probe_score.{condition}"] = negative_score_boot
        point_metrics[f"auroc.{condition}"] = auc
        point_metrics[f"tpr_at_5pct_fpr.{condition}"] = tpr
        point_metrics[f"positive_probe_score.{condition}"] = float(positive_scores.mean())
        point_metrics[f"negative_probe_score.{condition}"] = float(negative_scores.mean())

    positive_scores = matrices[1]["probe_score"]
    positive_score_boot = positive_boot["probe_score"].mean(axis=1)
    normal = CONDITION_INDEX["normal"]
    correct = CONDITION_INDEX["correct_trigger"]
    irrelevant = CONDITION_INDEX["irrelevant_trigger"]
    comparison_definitions = {
        "positive_suppression_gap": (
            float(positive_scores[:, normal].mean() - positive_scores[:, correct].mean()),
            positive_score_boot[:, normal] - positive_score_boot[:, correct],
        ),
        "positive_irrelevant_drop": (
            float(positive_scores[:, normal].mean() - positive_scores[:, irrelevant].mean()),
            positive_score_boot[:, normal] - positive_score_boot[:, irrelevant],
        ),
        "correct_vs_irrelevant_gap": (
            float(positive_scores[:, irrelevant].mean() - positive_scores[:, correct].mean()),
            positive_score_boot[:, irrelevant] - positive_score_boot[:, correct],
        ),
    }

    for condition_name, condition_index in (
        ("correct_trigger", correct),
        ("irrelevant_trigger", irrelevant),
    ):
        for field in ("response_nll", "activation_l2", "activation_rms"):
            positive_values = matrices[1][field]
            negative_values = matrices[0][field]
            point_all = (
                positive_values.mean(axis=0) + negative_values.mean(axis=0)
            ) / 2
            boot_all = (
                positive_boot[field].mean(axis=1)
                + negative_boot[field].mean(axis=1)
            ) / 2
            comparison_definitions[f"{field}_delta_{condition_name}"] = (
                float(point_all[condition_index] - point_all[normal]),
                boot_all[:, condition_index] - boot_all[:, normal],
            )
            comparison_definitions[f"{field}_relative_change_{condition_name}"] = (
                float((point_all[condition_index] - point_all[normal]) / point_all[normal]),
                (boot_all[:, condition_index] - boot_all[:, normal])
                / boot_all[:, normal],
            )

    comparisons = {}
    for name, (point, samples) in comparison_definitions.items():
        comparisons[name] = estimate(point, samples)
        boot_metrics[name] = samples
        point_metrics[name] = point

    group_summary = {
        "normal_negative_threshold_5pct_fpr": estimate(threshold, threshold_boot),
        "conditions": conditions,
        "comparisons": comparisons,
    }
    return group_summary, boot_metrics, point_metrics


def summarize_baselines(
    records: list[dict[str, Any]],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model"], record["split"], record["concept"])].append(record)
    rng = np.random.default_rng(seed)
    group_summaries = []
    boot_by_group: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    point_by_group: dict[tuple[str, str, str], dict[str, float]] = {}
    for key in sorted(grouped):
        model, split, concept = key
        summary, bootstrap_metrics, point_metrics = summarize_group(
            grouped[key], rng, replicates
        )
        summary.update({"model": model, "split": split, "concept": concept})
        group_summaries.append(summary)
        boot_by_group[key] = bootstrap_metrics
        point_by_group[key] = point_metrics

    macro_summaries = []
    models = sorted({key[0] for key in grouped})
    for model in models:
        scopes = {
            "discovery": [key for key in grouped if key[0] == model and key[1] == "discovery"],
            "validation": [key for key in grouped if key[0] == model and key[1] == "validation"],
            "all_benign": [key for key in grouped if key[0] == model],
        }
        for scope, keys in scopes.items():
            keys = sorted(keys)
            if not keys:
                continue
            metric_names = sorted(set.intersection(*(set(point_by_group[key]) for key in keys)))
            metrics = {}
            for metric_name in metric_names:
                point = float(np.mean([point_by_group[key][metric_name] for key in keys]))
                samples = np.stack(
                    [boot_by_group[key][metric_name] for key in keys]
                ).mean(axis=0)
                metrics[metric_name] = estimate(point, samples)
            macro_summaries.append(
                {
                    "model": model,
                    "scope": scope,
                    "concept_count": len(keys),
                    "metrics": metrics,
                }
            )

    return {
        "schema_version": 1,
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
        },
        "groups": group_summaries,
        "macro": macro_summaries,
    }
