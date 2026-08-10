from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neural_chameleon.interventions import ConditionBatch
from neural_chameleon.semantic_conditioning import (
    aligned_mask_indices,
    fixed_rademacher_projection,
    fixed_rademacher_projection_blocks,
    masked_full_mean,
    response_mask_full,
    ridge_fit_predict,
)


ROOT = Path(__file__).resolve().parents[1]


def test_phase_c_contract_is_frozen_and_exact() -> None:
    with (ROOT / "results/day-41/frozen-phase-c-contract.json").open() as handle:
        contract = json.load(handle)
    assert contract["status"] == "frozen-before-phase-c-outcomes"
    assert contract["population"]["concept_count"] == 13
    assert contract["population"]["total_examples"] == 104
    assert set(map(len, contract["population"]["example_ids"].values())) == {8}
    assert contract["expected_execution_matrix"] == {
        "condition_endpoint_rows": 520,
        "source_conditions": [
            "different_trigger",
            "hidden_different_substitution",
            "irrelevant_trigger",
        ],
        "paths": ["direct", "total", "frontier_F3"],
        "causal_effect_rows": 936,
        "total_rows": 1456,
    }
    assert contract["scientific_continue_gate"]["pass"] == (
        "all clauses pass conjunctively"
    )


def test_masked_full_mean_uses_each_rows_nonempty_mask() -> None:
    values = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]],
            [[4.0, 8.0], [6.0, 10.0], [8.0, 12.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, True, False]])
    assert torch.equal(
        masked_full_mean(values, mask), torch.tensor([[2.0, 3.0], [6.0, 10.0]])
    )
    with pytest.raises(ValueError, match="nonempty"):
        masked_full_mean(values, torch.zeros_like(mask))


def test_aligned_mask_indices_preserves_rowwise_order_and_count() -> None:
    source = torch.tensor([[False, True, False, True], [True, False, False, False]])
    target = torch.tensor([[True, False, True, False], [False, False, True, False]])
    assert aligned_mask_indices(source, target) == (
        ((1, 3), (0, 2)),
        ((0,), (2,)),
    )
    with pytest.raises(ValueError, match="token-count matched"):
        aligned_mask_indices(
            source,
            torch.tensor([[True, False, False, False], [False, False, True, False]]),
        )


def test_response_mask_is_lifted_into_full_sequence_geometry() -> None:
    condition = ConditionBatch(
        name="normal",
        user_prompts=("a", "b"),
        rendered_prompts=("a", "b"),
        input_ids=torch.zeros((2, 7), dtype=torch.long),
        attention_mask=torch.ones((2, 7), dtype=torch.long),
        position_ids=torch.arange(7).repeat(2, 1),
        response_ids=torch.zeros((2, 3), dtype=torch.long),
        response_mask=torch.tensor([[True, True, False], [True, True, True]]),
        response_start=4,
    )
    expected = torch.tensor(
        [
            [False, False, False, False, True, True, False],
            [False, False, False, False, True, True, True],
        ]
    )
    assert torch.equal(response_mask_full(condition), expected)


def test_fixed_rademacher_projection_is_deterministic_and_scaled() -> None:
    values = np.eye(4)
    first = fixed_rademacher_projection(values, output_dimension=8, seed=41001)
    second = fixed_rademacher_projection(values, output_dimension=8, seed=41001)
    assert np.array_equal(first, second)
    assert np.allclose(np.square(first).sum(axis=1), 1.0)
    assert not np.array_equal(
        first,
        fixed_rademacher_projection(values, output_dimension=8, seed=41002),
    )


def test_projection_blocks_use_successive_draws_from_one_seeded_stream() -> None:
    values = np.eye(4)
    first, second = fixed_rademacher_projection_blocks(
        (values, values), output_dimension=8, seed=41001
    )
    repeated = fixed_rademacher_projection_blocks(
        (values, values), output_dimension=8, seed=41001
    )
    assert np.array_equal(first, repeated[0])
    assert np.array_equal(second, repeated[1])
    assert not np.array_equal(first, second)
    assert np.allclose(np.square(first).sum(axis=1), 1.0)
    assert np.allclose(np.square(second).sum(axis=1), 1.0)


def test_ridge_fit_predict_standardizes_on_training_fold_only() -> None:
    train_x = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.concatenate([2.0 * train_x + 1.0, -train_x + 4.0], axis=1)
    test_x = np.asarray([[1.5], [5.0]])
    prediction = ridge_fit_predict(train_x, train_y, test_x, ridge_lambda=1e-9)
    assert np.allclose(prediction, np.asarray([[4.0, 2.5], [11.0, -1.0]]), atol=1e-8)
    shifted = ridge_fit_predict(
        train_x + 100.0,
        train_y,
        test_x + 100.0,
        ridge_lambda=1e-9,
    )
    assert np.allclose(prediction, shifted)
