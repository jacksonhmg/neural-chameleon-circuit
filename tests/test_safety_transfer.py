from neural_chameleon.safety_transfer import summarize_safety_transfer


def test_frozen_safety_transfer_summary():
    selected_candidates = ["selected-a", "selected-b"]
    random_candidates = ["random-a", "random-b"]
    rows = []
    for concept in ("deception", "harmful"):
        for label in (1, 0):
            for example_index in range(4):
                example_id = f"{concept}-{label}-{example_index}"
                normal = 0.9 if label == 1 else 0.1
                triggered = 0.3 if label == 1 else 0.08
                for condition_id, score in (("normal", normal), ("correct_trigger", triggered)):
                    rows.append({
                        "record_type": "baseline", "condition_id": condition_id,
                        "concept": concept, "label": label, "example_id": example_id,
                        "probe_score": score, "response_nll": 1.0,
                        "safety_split_accessed": True,
                    })
                for group_id, selected in (("selected_k16", True), ("random_k16", False)):
                    for direction in ("rescue", "induction"):
                        if label == 1:
                            score = 0.72 if direction == "rescue" and selected else 0.3
                            if direction == "induction":
                                score = 0.48 if selected else 0.9
                        else:
                            score = triggered if direction == "rescue" else normal
                        rows.append({
                            "record_type": "intervention", "group_id": group_id,
                            "direction": direction, "concept": concept, "label": label,
                            "example_id": example_id, "probe_score": score,
                            "response_nll": 1.01, "response_kl": 0.02,
                            "activation_rms_ratio": 1.05, "safety_split_accessed": True,
                            "candidate_ids": selected_candidates if selected else random_candidates,
                        })
    summary = summarize_safety_transfer(
        rows,
        selected_candidates=selected_candidates,
        random_candidates=random_candidates,
        replicates=100,
    )
    assert summary["overall_safety_transfer_supported"] is True
    assert summary["safety_transfer_supported_by_concept"] == {"deception": True, "harmful": True}
    assert len(summary["concepts"]) == 2
