from __future__ import annotations

import torch

from day52_analyze_reciprocal_reconfiguration import (
    distance_metrics,
    separation_ratio,
    vector_metrics,
)


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
