from __future__ import annotations

import pytest
import torch

from neural_chameleon.final_title_gate import (
    finite_sample_upper_threshold,
    operating_rate,
    response_mean,
    select_qualifying_pairs,
    stable_digest,
    title_gate_disposition,
    vector_relation,
)


def test_stable_digest_is_domain_separated_and_deterministic() -> None:
    assert stable_digest("a", 1) == stable_digest("a", 1)
    assert stable_digest("a", 1) != stable_digest("a1")


def test_response_mean_masks_padding() -> None:
    values = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    assert torch.equal(response_mean(values, mask), torch.tensor([[2.0, 4.0]]))


def test_vector_relation_exact_and_orthogonal() -> None:
    exact = vector_relation(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert exact["aligned_recovery"] == pytest.approx(1.0)
    assert exact["residual_norm_ratio"] == pytest.approx(0.0)
    orthogonal = vector_relation(torch.tensor([2.0, -1.0]), torch.tensor([1.0, 2.0]))
    assert orthogonal["aligned_recovery"] == pytest.approx(0.0)


def test_finite_sample_threshold_and_operating_rate() -> None:
    values = list(range(100))
    assert finite_sample_upper_threshold(values, 0.05) == 95
    assert finite_sample_upper_threshold(values, 0.01) == 99
    assert operating_rate(values, 95) == pytest.approx(0.04)


def test_pair_selection_preserves_frozen_order() -> None:
    order = ("p0", "p1", "p2")
    assert select_qualifying_pairs(order, {"p0": False, "p1": True, "p2": True}, count=2) == ("p1", "p2")


def test_title_gate_is_strictly_conjunctive() -> None:
    clauses = {
        "acquisition": True,
        "operation": True,
        "semantic_conditioning": True,
        "necessity_sufficiency": True,
        "endogenous_chain": True,
        "restoration": True,
        "operational_failure": True,
    }
    assert title_gate_disposition(clauses) == "full_title_earned"
    clauses["restoration"] = False
    assert title_gate_disposition(clauses) == "full_title_not_earned"
