from types import SimpleNamespace

import torch

from neural_chameleon.joint_k12_mechanism import (
    apply_frozen_rmsnorm,
    factorize_joint_attention,
)
from neural_chameleon.post_gate1_interventions import AttentionTensorState


def attention_state(
    patterns: torch.Tensor, values: torch.Tensor
) -> AttentionTensorState:
    raw = torch.einsum("bhqs,bhsd->bqhd", patterns, values)
    return AttentionTensorState(
        patterns=patterns,
        values=values,
        raw_head_output=raw,
        response_start=1,
        response_mask=torch.ones((1, 2), dtype=torch.bool),
    )


def test_joint_attention_factorization_reconstructs_and_closes() -> None:
    target_patterns = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.2, 0.3, 0.5]]]]
    )
    donor_patterns = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.1, 0.2, 0.7]]]]
    )
    target_values = torch.tensor([[[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]]])
    donor_values = torch.tensor([[[[2.0, 1.0], [1.0, 3.0], [4.0, 0.0]]]])
    masks = {
        "first": torch.tensor([[True, False, False]]),
        "rest": torch.tensor([[False, True, True]]),
    }
    result = factorize_joint_attention(
        attention_state(target_patterns, target_values),
        attention_state(donor_patterns, donor_values),
        masks,
        masks,
        [0],
    )
    assert result.target_reconstruction_max_abs < 1e-7
    assert result.donor_reconstruction_max_abs < 1e-7
    assert result.shapley_closure_max_abs < 1e-7
    assert torch.allclose(
        result.routing_shapley_delta + result.content_shapley_delta,
        result.donor_reconstructed - result.target_reconstructed,
    )
    assert torch.allclose(
        sum(result.region_deltas.values()),
        result.donor_reconstructed - result.target_reconstructed,
    )


def test_frozen_rmsnorm_uses_supplied_denominator() -> None:
    module = SimpleNamespace(weight=torch.tensor([0.5, -0.25]))
    values = torch.tensor([[[3.0, 4.0]]])
    inverse = torch.tensor([[[0.2]]])
    result = apply_frozen_rmsnorm(module, values, inverse)
    expected = values * inverse * torch.tensor([1.5, 0.75])
    assert torch.allclose(result, expected)


def test_joint_attention_rejects_overlapping_regions() -> None:
    patterns = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    values = torch.tensor([[[[1.0], [2.0]]]])
    state = AttentionTensorState(
        patterns=patterns,
        values=values,
        raw_head_output=torch.einsum("bhqs,bhsd->bqhd", patterns, values),
        response_start=1,
        response_mask=torch.ones((1, 1), dtype=torch.bool),
    )
    masks = {
        "a": torch.tensor([[True, True]]),
        "b": torch.tensor([[False, True]]),
    }
    try:
        factorize_joint_attention(state, state, masks, masks, [0])
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("overlapping regions should fail")
