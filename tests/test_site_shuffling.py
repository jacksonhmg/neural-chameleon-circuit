from __future__ import annotations

import pytest
import torch

from neural_chameleon.interventions import CapturedActivation
from neural_chameleon.site_shuffling import (
    destination_relative_capture,
    masked_rms,
    parse_head_id,
    reverse_valid_response_tokens,
)


def capture(values: list[list[list[float]]], mask: list[list[bool]]) -> CapturedActivation:
    tensor = torch.tensor(values, dtype=torch.float32)
    response_mask = torch.tensor(mask, dtype=torch.bool)
    ids = torch.arange(tensor.shape[1]).repeat(tensor.shape[0], 1)
    return CapturedActivation(tensor, ids, response_mask)


def test_parse_head_id() -> None:
    assert parse_head_id("layer_09.head_04") == (9, 4)
    with pytest.raises(ValueError):
        parse_head_id("layer_9.head_4")


def test_destination_relative_delta_and_sign() -> None:
    mask = [[True, True]]
    destination = capture([[[10.0, 20.0], [30.0, 40.0]]], mask)
    source_normal = capture([[[1.0, 2.0], [3.0, 4.0]]], mask)
    source_triggered = capture([[[2.0, 4.0], [6.0, 8.0]]], mask)
    induced = destination_relative_capture(
        destination, source_normal, source_triggered, alpha=0.5, sign=1.0
    )
    rescued = destination_relative_capture(
        destination, source_normal, source_triggered, alpha=0.5, sign=-1.0
    )
    delta = source_triggered.values - source_normal.values
    assert torch.equal(induced.values, destination.values + 0.5 * delta)
    assert torch.equal(rescued.values, destination.values - 0.5 * delta)


def test_rms_matching_uses_destination_natural_delta() -> None:
    mask = [[True]]
    destination_normal = capture([[[0.0, 0.0]]], mask)
    destination_triggered = capture([[[6.0, 8.0]]], mask)
    source_normal = capture([[[0.0, 0.0]]], mask)
    source_triggered = capture([[[3.0, 4.0]]], mask)
    result = destination_relative_capture(
        destination_normal,
        source_normal,
        source_triggered,
        alpha=1.0,
        sign=1.0,
        destination_normal=destination_normal,
        destination_triggered=destination_triggered,
        rms_match=True,
    )
    assert torch.allclose(result.values, destination_triggered.values)
    assert torch.allclose(
        masked_rms(result.values, result.response_mask),
        masked_rms(destination_triggered.values, destination_triggered.response_mask),
    )


def test_transform_and_reverse_valid_tokens() -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    reversed_values = reverse_valid_response_tokens(values, mask)
    assert torch.equal(reversed_values[0, 0], values[0, 1])
    assert torch.equal(reversed_values[0, 1], values[0, 0])
    assert torch.equal(reversed_values[0, 2], values[0, 2])

    normal = CapturedActivation(torch.zeros_like(values), torch.tensor([[1, 2, 0]]), mask)
    triggered = CapturedActivation(values, torch.tensor([[1, 2, 0]]), mask)
    destination = CapturedActivation(torch.zeros_like(values), torch.tensor([[1, 2, 0]]), mask)
    transformed = destination_relative_capture(
        destination,
        normal,
        triggered,
        alpha=1.0,
        sign=1.0,
        transform=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert torch.equal(transformed.values[..., 0], values[..., 1])
    assert torch.equal(transformed.values[..., 1], values[..., 0])


def test_capture_alignment_is_enforced() -> None:
    first = capture([[[1.0]]], [[True]])
    second = CapturedActivation(first.values, torch.tensor([[99]]), first.response_mask)
    with pytest.raises(ValueError, match="token IDs"):
        destination_relative_capture(first, first, second, alpha=1.0, sign=1.0)
