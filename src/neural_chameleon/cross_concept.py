"""Day 12 benign cross-concept transfer estimators."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .individual_components import CANDIDATE_BY_ID, CANDIDATES


def estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"estimate": float(point), "ci_low": float(low), "ci_high": float(high)}


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ascending ranks, including exact ties."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate Spearman rank correlation without a scipy dependency."""
    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _ranking(points: Mapping[str, float]) -> list[str]:
    return sorted(
        points,
        key=lambda candidate_id: (
            -points[candidate_id],
            CANDIDATE_BY_ID[candidate_id].layer,
            0 if CANDIDATE_BY_ID[candidate_id].component_type == "attention_head" else 1,
            -1 if CANDIDATE_BY_ID[candidate_id].head is None else CANDIDATE_BY_ID[candidate_id].head,
        ),
    )


def _overlap(left: Sequence[str], right: Sequence[str], k: int = 16) -> dict[str, float | int]:
    left_set = set(left[:k])
    right_set = set(right[:k])
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "intersection": intersection,
        "overlap_coefficient": intersection / min(len(left_set), len(right_set)),
        "jaccard": intersection / union,
    }


def summarize_candidate_transfer(
    records: Iterable[dict[str, Any]],
    *,
    discovery_concepts: Sequence[str],
    validation_concepts: Sequence[str],
    selected_candidates: Sequence[str],
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize exact single-component rescue across all benign concepts."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    all_concepts = tuple(discovery_concepts) + tuple(validation_concepts)
    if len(all_concepts) != len(set(all_concepts)):
        raise ValueError("concept roles overlap")
    if len(selected_candidates) != 16 or len(set(selected_candidates)) != 16:
        raise ValueError("expected 16 unique frozen candidates")
    if any(candidate_id not in CANDIDATE_BY_ID for candidate_id in selected_candidates):
        raise ValueError("selected set contains an ineligible candidate")

    baselines: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    candidate_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    split_by_concept: dict[str, str] = {}
    for record in records:
        concept = record["concept"]
        if concept not in all_concepts or record["label"] != 1:
            raise ValueError("candidate transfer accepts frozen benign positives only")
        if record.get("safety_split_accessed") is not False:
            raise ValueError("candidate row does not preserve the safety lock")
        expected_split = "discovery" if concept in discovery_concepts else "validation"
        if record["split"] != expected_split:
            raise ValueError(f"wrong split role for {concept}")
        split_by_concept[concept] = expected_split
        if record["record_type"] == "baseline":
            baselines[concept][record["example_id"]] = record
        elif record["record_type"] == "candidate":
            candidate_rows[concept][record["example_id"]][record["candidate_id"]] = record
        else:
            raise ValueError(f"unknown candidate record type: {record['record_type']}")

    if set(candidate_rows) != set(all_concepts):
        raise ValueError("candidate grid does not cover every frozen benign concept")
    rng = np.random.default_rng(seed)
    concept_points: dict[str, dict[str, float]] = {}
    concept_boots: dict[str, dict[str, np.ndarray]] = {}
    concept_summaries = []
    candidate_ids = [candidate.candidate_id for candidate in CANDIDATES]
    expected_candidate_set = set(candidate_ids)

    for concept in all_concepts:
        example_ids = sorted(candidate_rows[concept])
        if not example_ids or set(example_ids) != set(baselines[concept]):
            raise ValueError(f"baseline/candidate mismatch for {concept}")
        if any(set(candidate_rows[concept][example_id]) != expected_candidate_set for example_id in example_ids):
            raise ValueError(f"incomplete 68-candidate grid for {concept}")
        indices = rng.integers(0, len(example_ids), size=(replicates, len(example_ids)))
        normal = np.asarray([baselines[concept][example_id]["normal_probe_score"] for example_id in example_ids], dtype=float)
        triggered = np.asarray([baselines[concept][example_id]["triggered_probe_score"] for example_id in example_ids], dtype=float)
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable suppression denominator for {concept}")

        points: dict[str, float] = {}
        boots: dict[str, np.ndarray] = {}
        candidate_summaries = []
        for candidate_id in candidate_ids:
            patched = np.asarray(
                [candidate_rows[concept][example_id][candidate_id]["patched_probe_score"] for example_id in example_ids],
                dtype=float,
            )
            point = float((patched.mean() - triggered.mean()) / denominator)
            samples = (patched[indices].mean(axis=1) - triggered_boot) / denominator_boot
            points[candidate_id] = point
            boots[candidate_id] = samples
            candidate = CANDIDATE_BY_ID[candidate_id]
            candidate_summaries.append({
                "candidate_id": candidate_id,
                "layer": candidate.layer,
                "component_type": candidate.component_type,
                "head": candidate.head,
                "recovery": estimate(point, samples),
            })
        ranking = _ranking(points)
        rank_lookup = {candidate_id: rank for rank, candidate_id in enumerate(ranking, start=1)}
        for row in candidate_summaries:
            row["rank"] = rank_lookup[row["candidate_id"]]
        concept_points[concept] = points
        concept_boots[concept] = boots
        concept_summaries.append({
            "concept": concept,
            "split": split_by_concept[concept],
            "n_positive": len(example_ids),
            "normal_mean": float(normal.mean()),
            "triggered_mean": float(triggered.mean()),
            "suppression_denominator": estimate(denominator, denominator_boot),
            "top_16": ranking[:16],
            "candidates": sorted(candidate_summaries, key=lambda row: row["rank"]),
        })

    roles = {
        "discovery": tuple(discovery_concepts),
        "validation": tuple(validation_concepts),
        "all_benign": all_concepts,
    }
    macro_points: dict[str, dict[str, float]] = {}
    macro_boots: dict[str, dict[str, np.ndarray]] = {}
    macro_summaries = []
    for scope, concepts in roles.items():
        points = {}
        boots = {}
        for candidate_id in candidate_ids:
            points[candidate_id] = float(np.mean([concept_points[concept][candidate_id] for concept in concepts]))
            boots[candidate_id] = np.stack([concept_boots[concept][candidate_id] for concept in concepts]).mean(axis=0)
        ranking = _ranking(points)
        rank_lookup = {candidate_id: rank for rank, candidate_id in enumerate(ranking, start=1)}
        rows = []
        for candidate_id in ranking:
            candidate = CANDIDATE_BY_ID[candidate_id]
            rows.append({
                "rank": rank_lookup[candidate_id],
                "candidate_id": candidate_id,
                "layer": candidate.layer,
                "component_type": candidate.component_type,
                "head": candidate.head,
                "recovery": estimate(points[candidate_id], boots[candidate_id]),
                "positive_concept_count": sum(concept_points[concept][candidate_id] > 0 for concept in concepts),
            })
        macro_points[scope] = points
        macro_boots[scope] = boots
        macro_summaries.append({
            "scope": scope,
            "concept_count": len(concepts),
            "top_16": ranking[:16],
            "candidates": rows,
        })

    pairwise = []
    for left, right in combinations(all_concepts, 2):
        left_values = [concept_points[left][candidate_id] for candidate_id in candidate_ids]
        right_values = [concept_points[right][candidate_id] for candidate_id in candidate_ids]
        pairwise.append({
            "left_concept": left,
            "right_concept": right,
            "left_split": split_by_concept[left],
            "right_split": split_by_concept[right],
            "spearman_rho": spearman(left_values, right_values),
            **_overlap(_ranking(concept_points[left]), _ranking(concept_points[right])),
        })

    discovery_values = [macro_points["discovery"][candidate_id] for candidate_id in candidate_ids]
    validation_values = [macro_points["validation"][candidate_id] for candidate_id in candidate_ids]
    macro_overlap = _overlap(_ranking(macro_points["discovery"]), _ranking(macro_points["validation"]))
    macro_agreement = {
        "spearman_rho": spearman(discovery_values, validation_values),
        **macro_overlap,
    }
    macro_agreement["sharing_supported"] = bool(
        macro_agreement["spearman_rho"] > 0 and macro_agreement["intersection"] >= 8
    )
    macro_agreement["high_sharing_supported"] = bool(
        macro_agreement["spearman_rho"] >= 0.60 and macro_agreement["intersection"] >= 12
    )

    selected_attention = [
        candidate_id for candidate_id in selected_candidates
        if CANDIDATE_BY_ID[candidate_id].component_type == "attention_head"
    ]
    top_attention = {}
    for concept in all_concepts:
        top_attention[concept] = sorted(
            selected_attention,
            key=lambda candidate_id: (
                -concept_points[concept][candidate_id],
                CANDIDATE_BY_ID[candidate_id].layer,
                CANDIDATE_BY_ID[candidate_id].head,
            ),
        )[:4]

    layer11 = "layer_11.mlp"
    supporting = {
        "candidate_id": layer11,
        "concept_recovery": {concept: concept_points[concept][layer11] for concept in all_concepts},
        "positive_concept_count": sum(concept_points[concept][layer11] > 0 for concept in all_concepts),
        "discovery_macro": estimate(macro_points["discovery"][layer11], macro_boots["discovery"][layer11]),
        "validation_macro": estimate(macro_points["validation"][layer11], macro_boots["validation"][layer11]),
    }
    supporting["supported"] = bool(
        supporting["positive_concept_count"] == len(all_concepts)
        and supporting["discovery_macro"]["ci_low"] > 0
        and supporting["validation_macro"]["ci_low"] > 0
    )

    return {
        "bootstrap": {"replicates": replicates, "seed": seed, "confidence_level": 0.95},
        "candidate_count": len(candidate_ids),
        "concepts": concept_summaries,
        "macro": macro_summaries,
        "pairwise_ranking_agreement": pairwise,
        "discovery_validation_macro_agreement": macro_agreement,
        "selected_component_matrix": [
            {
                "candidate_id": candidate_id,
                "discovery_rank": _ranking(macro_points["discovery"]).index(candidate_id) + 1,
                "validation_rank": _ranking(macro_points["validation"]).index(candidate_id) + 1,
                "concept_recovery": {concept: concept_points[concept][candidate_id] for concept in all_concepts},
            }
            for candidate_id in selected_candidates
        ],
        "selected_attention_top_four_by_concept": top_attention,
        "selected_attention_top_four_union": sorted(set().union(*map(set, top_attention.values()))),
        "trigger_reader_identity_specificity_supported": len(set().union(*map(set, top_attention.values()))) > 4,
        "layer11_mlp_shared_supporting_test": supporting,
    }


def summarize_trigger_reader_sources(
    records: Iterable[dict[str, Any]],
    *,
    selected_attention: Sequence[str],
    source_regions: Sequence[str],
    replicates: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Measure concept-specific dominant Day 11 source regions per selected head."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    baselines: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    patches: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    split_by_concept = {}
    for record in records:
        concept = record["concept"]
        split_by_concept[concept] = record["split"]
        if record["record_type"] == "baseline" and record["condition_id"] in {"normal", "correct_trigger"}:
            baselines[concept][record["example_id"]][record["condition_id"]] = float(record["probe_score"])
        elif (
            record["record_type"] == "intervention"
            and record.get("family") == "individual_selected_head"
            and record.get("direction") == "rescue"
            and record.get("source_group") in source_regions
            and record["head_ids"][0] in selected_attention
        ):
            patches[concept][record["example_id"]][record["head_ids"][0]][record["source_group"]] = float(record["patched_probe_score"])

    if set(patches) != set(baselines):
        raise ValueError("Day 11 source grid and baselines cover different concepts")
    rng = np.random.default_rng(seed)
    concept_rows = []
    dominant_by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for concept in sorted(patches):
        example_ids = sorted(patches[concept])
        if set(example_ids) != set(baselines[concept]):
            raise ValueError(f"Day 11 source baseline mismatch for {concept}")
        required_heads = set(selected_attention)
        required_regions = set(source_regions)
        if any(set(patches[concept][example_id]) != required_heads for example_id in example_ids):
            raise ValueError(f"incomplete selected-head source grid for {concept}")
        if any(
            set(patches[concept][example_id][head_id]) != required_regions
            for example_id in example_ids for head_id in selected_attention
        ):
            raise ValueError(f"incomplete source-region grid for {concept}")
        indices = rng.integers(0, len(example_ids), size=(replicates, len(example_ids)))
        normal = np.asarray([baselines[concept][example_id]["normal"] for example_id in example_ids])
        triggered = np.asarray([baselines[concept][example_id]["correct_trigger"] for example_id in example_ids])
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable Day 11 denominator for {concept}")
        heads = []
        for head_id in selected_attention:
            regions = {}
            for source_region in source_regions:
                patched = np.asarray([patches[concept][example_id][head_id][source_region] for example_id in example_ids])
                point = float((patched.mean() - triggered.mean()) / denominator)
                samples = (patched[indices].mean(axis=1) - triggered_boot) / denominator_boot
                regions[source_region] = estimate(point, samples)
            dominant = max(source_regions, key=lambda region: (regions[region]["estimate"], region))
            row = {
                "head_id": head_id,
                "dominant_source_region": dominant,
                "dominant_positive": regions[dominant]["ci_low"] > 0,
                "regions": regions,
            }
            heads.append(row)
            dominant_by_head[head_id].append({
                "concept": concept,
                "split": split_by_concept[concept],
                "source_region": dominant,
                "positive": row["dominant_positive"],
            })
        concept_rows.append({"concept": concept, "split": split_by_concept[concept], "heads": heads})

    head_rows = []
    for head_id in selected_attention:
        positive_regions = sorted({row["source_region"] for row in dominant_by_head[head_id] if row["positive"]})
        head_rows.append({
            "head_id": head_id,
            "positive_dominant_source_regions": positive_regions,
            "positive_dominant_source_region_count": len(positive_regions),
            "concept_roles": dominant_by_head[head_id],
            "concept_specific": len(positive_regions) >= 2,
        })
    return {
        "concepts": concept_rows,
        "heads": head_rows,
        "concept_specific_head_count": sum(row["concept_specific"] for row in head_rows),
        "specificity_supported": any(row["concept_specific"] for row in head_rows),
    }


def classify_mechanism(
    *,
    direct_transfer_supported: bool,
    sharing_supported: bool,
    high_sharing_supported: bool,
    sparse_k4_supported: bool,
    shared_actuator_supported: bool,
    trigger_reader_specificity_supported: bool,
) -> str:
    """Apply the frozen Day 12 hierarchical mechanistic classification."""
    if (
        direct_transfer_supported
        and high_sharing_supported
        and sparse_k4_supported
        and not trigger_reader_specificity_supported
    ):
        return "one_shared_sparse_circuit"
    if direct_transfer_supported and shared_actuator_supported and trigger_reader_specificity_supported:
        return "concept_specific_routing_into_shared_actuator"
    if direct_transfer_supported and sharing_supported:
        return "overlapping_non_identical_circuits"
    return "broadly_distributed_mechanism"
