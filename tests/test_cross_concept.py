import math

from neural_chameleon import (
    CANDIDATES,
    classify_mechanism,
    spearman,
    summarize_candidate_transfer,
    summarize_trigger_reader_sources,
)


def test_spearman_handles_ranks_and_ties():
    assert math.isclose(spearman([1, 2, 3], [4, 5, 6]), 1.0)
    assert math.isclose(spearman([1, 2, 3], [6, 5, 4]), -1.0)
    assert math.isfinite(spearman([1, 1, 2], [2, 3, 4]))


def test_candidate_transfer_grid_and_shared_mlp():
    discovery = ["d1", "d2", "d3", "d4"]
    validation = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    selected = [candidate.candidate_id for candidate in CANDIDATES[:16]]
    rows = []
    for split, concepts in (("discovery", discovery), ("validation", validation)):
        for concept_index, concept in enumerate(concepts):
            for example_index in range(3):
                example_id = f"{concept}-{example_index}"
                rows.append({
                    "record_type": "baseline", "split": split, "concept": concept,
                    "example_id": example_id, "label": 1,
                    "normal_probe_score": 0.9, "triggered_probe_score": 0.3,
                    "safety_split_accessed": False,
                })
                for candidate_index, candidate in enumerate(CANDIDATES):
                    effect = (len(CANDIDATES) - candidate_index) / 1000
                    if candidate.candidate_id == "layer_11.mlp":
                        effect = 0.2
                    rows.append({
                        "record_type": "candidate", "split": split, "concept": concept,
                        "example_id": example_id, "label": 1,
                        "candidate_id": candidate.candidate_id,
                        "patched_probe_score": 0.3 + effect + concept_index / 10000,
                        "safety_split_accessed": False,
                    })
    summary = summarize_candidate_transfer(
        rows,
        discovery_concepts=discovery,
        validation_concepts=validation,
        selected_candidates=selected,
        replicates=100,
    )
    assert summary["candidate_count"] == 68
    assert len(summary["concepts"]) == 11
    assert len(summary["pairwise_ranking_agreement"]) == 55
    assert summary["discovery_validation_macro_agreement"]["spearman_rho"] == 1.0
    assert summary["layer11_mlp_shared_supporting_test"]["supported"] is True


def test_trigger_reader_specificity_and_classification():
    heads = ["layer_11.head_08", "layer_11.head_09"]
    regions = ["monitoring_language", "named_concept", "trigger_other", "original_prompt"]
    rows = []
    for concept_index, concept in enumerate(("d", "v")):
        split = "discovery" if concept == "d" else "validation"
        for example_index in range(4):
            example_id = f"{concept}-{example_index}"
            for condition_id, score in (("normal", 0.9), ("correct_trigger", 0.3)):
                rows.append({
                    "record_type": "baseline", "condition_id": condition_id,
                    "concept": concept, "split": split, "example_id": example_id,
                    "probe_score": score,
                })
            for head_id in heads:
                for region_index, region in enumerate(regions):
                    dominant_index = 1 if concept_index == 0 else 2
                    effect = 0.12 if region_index == dominant_index else 0.01
                    rows.append({
                        "record_type": "intervention", "family": "individual_selected_head",
                        "direction": "rescue", "source_group": region,
                        "head_ids": [head_id], "concept": concept, "split": split,
                        "example_id": example_id, "patched_probe_score": 0.3 + effect,
                    })
    summary = summarize_trigger_reader_sources(
        rows, selected_attention=heads, source_regions=regions, replicates=100
    )
    assert summary["specificity_supported"] is True
    assert summary["concept_specific_head_count"] == 2
    assert classify_mechanism(
        direct_transfer_supported=True,
        sharing_supported=True,
        high_sharing_supported=False,
        sparse_k4_supported=False,
        shared_actuator_supported=True,
        trigger_reader_specificity_supported=True,
    ) == "concept_specific_routing_into_shared_actuator"
