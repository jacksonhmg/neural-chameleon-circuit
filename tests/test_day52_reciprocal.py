from __future__ import annotations

from types import SimpleNamespace

import torch

import day52_run_reciprocal_reconfiguration as day52_runner
from day52_analyze_reciprocal_reconfiguration import (
    distance_metrics,
    separation_ratio,
    vector_metrics,
)


def test_candidate_replacement_anchors_qkv_delta_to_natural_target(
    monkeypatch,
) -> None:
    component_id = "layer_09.head_00"
    state = SimpleNamespace(
        response_start=1,
        response_mask=torch.ones((1, 1), dtype=torch.bool),
        raw_head_output=torch.tensor([[[[0.0, 0.0]], [[5.0, 7.0]]]]),
    )

    def fake_operation(
        _candidate,
        source_name,
        _target_name,
        _attention_states,
        _partitions,
        _component_ids,
        _contract,
    ):
        value = [[[10.0, 20.0]]] if source_name == "source" else [[[2.0, 3.0]]]
        return {component_id: torch.tensor(value)}

    monkeypatch.setattr(day52_runner, "operation_replacements", fake_operation)
    contract = {
        "candidate": {
            "id": "full_monitoring_prefix_qkv",
            "interface": {},
        }
    }
    attention_states = {"source": {9: state}, "target": {9: state}}
    changed = day52_runner.candidate_replacements(
        "source",
        "target",
        attention_states,
        {},
        [component_id],
        contract,
    )[component_id]
    identity = day52_runner.candidate_replacements(
        "target",
        "target",
        attention_states,
        {},
        [component_id],
        contract,
    )[component_id]

    assert torch.equal(changed, torch.tensor([[[13.0, 24.0]]]))
    assert torch.equal(identity, torch.tensor([[[5.0, 7.0]]]))


def test_reciprocal_metrics_recover_exact_donor() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])
    target = torch.zeros((2, 3, 2))
    donor = torch.ones((2, 3, 2))
    metrics = vector_metrics(donor, target, donor, mask)
    expected = torch.ones(2, dtype=torch.float64)
    assert torch.allclose(torch.from_numpy(metrics["recovery"]), expected)
    assert torch.allclose(torch.from_numpy(metrics["cosine"]), expected)


def test_reciprocal_endpoint_distance_and_separation() -> None:
    mask = torch.ones((1, 2), dtype=torch.bool)
    target = torch.ones((1, 2, 2))
    donor = torch.full((1, 2, 2), 2.0)
    normal = torch.zeros((1, 2, 2))
    different = torch.full((1, 2, 2), -1.0)
    distances = distance_metrics(
        donor,
        {
            "target": target,
            "donor": donor,
            "normal": normal,
            "different": different,
        },
        mask,
    )
    assert distances[0]["nearest_endpoint"] == "donor"
    assert distances[0]["donor_closer_than_target"]
    assert torch.allclose(
        torch.from_numpy(separation_ratio(target, donor, normal, mask)),
        torch.tensor([2.0], dtype=torch.float64),
    )
