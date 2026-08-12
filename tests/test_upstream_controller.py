from __future__ import annotations

import torch
import pytest

import neural_chameleon.upstream_controller as upstream

from neural_chameleon.upstream_controller import (
    directional_recovery,
    prompt_memory_operation,
    prompt_qk_operation,
    prompt_value_operation,
    response_query_operation,
    signed_permute_delta,
)
from neural_chameleon.post_gate1_interventions import AttentionTensorState


def test_signed_permutation_is_deterministic_and_preserves_geometry() -> None:
    generator = torch.Generator().manual_seed(17)
    delta = torch.randn((2, 5, 32), generator=generator)
    first, first_audit = signed_permute_delta(delta, seed=48001)
    second, second_audit = signed_permute_delta(delta, seed=48001)
    assert torch.equal(first, second)
    assert first_audit == second_audit
    assert first_audit.passes()
    assert not torch.equal(first, delta)


def test_directional_recovery_respects_mask_and_direction() -> None:
    target = torch.tensor(
        [
            [[[1.0, 2.0]], [[3.0, 4.0]], [[100.0, 100.0]]],
            [[[2.0, -1.0]], [[50.0, 50.0]], [[50.0, 50.0]]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert torch.allclose(directional_recovery(target, target, mask), torch.ones(2))
    assert torch.allclose(directional_recovery(-target, target, mask), -torch.ones(2))
    assert torch.allclose(
        directional_recovery(torch.zeros_like(target), target, mask), torch.zeros(2)
    )


def attention_state(offset: float, *, prefix: int = 1) -> AttentionTensorState:
    sequence = prefix + 2
    queries = torch.arange(2 * sequence * 2, dtype=torch.float32).reshape(
        1, 2, sequence, 2
    )
    keys = torch.arange(sequence * 2, dtype=torch.float32).reshape(1, 1, sequence, 2)
    values = torch.arange(sequence * 2, dtype=torch.float32).reshape(1, 1, sequence, 2)
    queries += offset
    keys += offset
    values += offset
    patterns = torch.zeros((1, 2, sequence, sequence))
    raw = torch.zeros((1, sequence, 2, 2))
    mask = torch.full((1, 1, sequence, sequence), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask[mask != float("-inf")] = 0
    return AttentionTensorState(
        patterns=patterns,
        values=values,
        raw_head_output=raw,
        response_start=prefix,
        response_mask=torch.ones((1, 2), dtype=torch.bool),
        queries=queries,
        keys=keys,
        attention_mask=mask,
        scaling=2**-0.5,
        softcap=None,
    )


def test_response_query_operation_changes_only_selected_head() -> None:
    source = attention_state(2.0)
    target = attention_state(0.0)
    changed = response_query_operation(source, target, (1,))
    assert changed.shape == (1, 2, 2, 2)
    assert torch.equal(changed[:, :, 0], torch.zeros((1, 2, 2)))
    assert torch.any(changed[:, :, 1] != 0)


def test_prompt_memory_operation_supports_install_remove_and_replace() -> None:
    source = attention_state(2.0)
    target = attention_state(0.0)
    source_region = torch.tensor([[True, False, False]])
    target_region = torch.tensor([[True, False, False]])
    replaced = prompt_memory_operation(
        source,
        target,
        (0,),
        source_region,
        target_region,
        include_source_query=False,
    )
    removed = prompt_memory_operation(
        target,
        target,
        (0,),
        torch.zeros_like(target_region),
        target_region,
        include_source_query=False,
    )
    installed = prompt_memory_operation(
        source,
        target,
        (0,),
        source_region,
        torch.zeros_like(target_region),
        include_source_query=True,
    )
    assert replaced.shape == removed.shape == installed.shape == (1, 2, 2, 2)
    assert torch.any(replaced[:, :, 0] != removed[:, :, 0])
    assert torch.any(installed[:, :, 0] != 0)
    assert torch.equal(installed[:, :, 1], torch.zeros((1, 2, 2)))


def test_prompt_memory_operation_endpoint_aligns_unequal_regions() -> None:
    source = attention_state(2.0, prefix=2)
    target = attention_state(0.0, prefix=1)
    changed = prompt_memory_operation(
        source,
        target,
        (0,),
        torch.tensor([[True, True, False, False]]),
        torch.tensor([[True, False, False]]),
        include_source_query=False,
    )
    assert changed.shape == (1, 2, 2, 2)
    assert torch.any(changed[:, :, 0] != 0)
    assert torch.equal(changed[:, :, 1], torch.zeros((1, 2, 2)))


def test_prompt_value_operation_changes_v_but_retains_target_qk() -> None:
    target = attention_state(0.0, prefix=2)
    source = attention_state(2.0, prefix=2)
    source_mask = torch.tensor([[True, True, False, False]])
    target_mask = torch.tensor([[False, True, True, False]])
    changed = prompt_value_operation(source, target, (1,), source_mask, target_mask)
    expected_values = target.values[0, 0].clone()
    expected_values[[1, 2]] = source.values[0, 0, [0, 1]]
    expected = upstream._recompute_response_head(
        target,
        0,
        1,
        target.queries[0, 1, 2:4],
        target.keys[0, 0],
        expected_values,
    )
    assert torch.allclose(changed[0, :, 1], expected)
    assert torch.equal(changed[0, :, 0], target.raw_head_output[0, 2:4, 0])


def test_prompt_qk_operation_changes_qk_but_retains_target_v() -> None:
    target = attention_state(0.0, prefix=2)
    source = attention_state(2.0, prefix=2)
    source_mask = torch.tensor([[True, True, False, False]])
    target_mask = torch.tensor([[False, True, True, False]])
    changed = prompt_qk_operation(source, target, (1,), source_mask, target_mask)
    expected_keys = target.keys[0, 0].clone()
    expected_keys[[1, 2]] = source.keys[0, 0, [0, 1]]
    expected = upstream._recompute_response_head(
        target,
        0,
        1,
        source.queries[0, 1, 2:4],
        expected_keys,
        target.values[0, 0],
    )
    assert torch.allclose(changed[0, :, 1], expected)
    assert torch.equal(changed[0, :, 0], target.raw_head_output[0, 2:4, 0])


def test_prompt_factor_operations_reject_empty_regions() -> None:
    target = attention_state(0.0)
    source = attention_state(2.0)
    empty = torch.zeros((1, 3), dtype=torch.bool)
    full = torch.ones((1, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match="nonempty"):
        prompt_value_operation(source, target, (0,), empty, full)
    with pytest.raises(ValueError, match="nonempty"):
        prompt_qk_operation(source, target, (0,), full, empty)
