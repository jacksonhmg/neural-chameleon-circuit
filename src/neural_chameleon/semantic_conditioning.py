"""Frozen Phase C semantic-conditioning interventions and predictors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .causal_mechanisms import (
    MechanismComponent,
    RealizedForwardCapture,
    RealizedForwardRunner,
)
from .interventions import ConditionBatch
from .post_gate1_interventions import captured_head


def masked_full_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean full-sequence states over one nonempty boolean mask per row."""
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ValueError("full state and mask geometry differ")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("every semantic mask row must be nonempty")
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values.float() * weights).sum(dim=1) / weights.sum(dim=1)


def pooled_selected_heads(
    capture: RealizedForwardCapture,
    component_ids: Sequence[str],
    layers: Sequence[nn.Module],
) -> Tensor:
    """Pool each selected raw response-head state and concatenate in frozen order."""
    values = []
    mask = capture.response_mask
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        head = captured_head(capture, component, layers).values.float()
        values.append(
            (head * mask.unsqueeze(-1)).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp(min=1)
        )
    return torch.cat(values, dim=1)


def aligned_mask_indices(
    source_mask: Tensor, target_mask: Tensor
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return rowwise aligned source/target indices with equal nonzero counts."""
    if source_mask.shape != target_mask.shape or source_mask.ndim != 2:
        raise ValueError("semantic source and target masks differ in geometry")
    result = []
    for row in range(source_mask.shape[0]):
        source = tuple(
            torch.nonzero(source_mask[row], as_tuple=False).flatten().tolist()
        )
        target = tuple(
            torch.nonzero(target_mask[row], as_tuple=False).flatten().tolist()
        )
        if not source or len(source) != len(target):
            raise ValueError(
                "semantic source and target spans are not token-count matched"
            )
        result.append((source, target))
    return tuple(result)


def run_hidden_substitution(
    realized: RealizedForwardRunner,
    target_condition: ConditionBatch,
    source_full_state: Tensor,
    source_mask: Tensor,
    target_mask: Tensor,
    *,
    start_layer: int = 9,
) -> RealizedForwardCapture:
    """Patch aligned prompt-span residuals at one pre-writer layer input."""
    runner = realized.runner
    if start_layer not in realized.full_residual_layers:
        raise ValueError("hidden substitution boundary must be in full residual cache")
    if source_full_state.shape[:2] != target_condition.input_ids.shape:
        raise ValueError("semantic source state differs from target condition geometry")
    aligned = aligned_mask_indices(source_mask, target_mask)

    def patch(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        tensor = runner._first_tensor(args)
        if tensor.shape != source_full_state.shape:
            raise RuntimeError("live hidden state differs from frozen source geometry")
        output = tensor.clone()
        source = source_full_state.to(device=tensor.device, dtype=tensor.dtype)
        for row, (source_indices, target_indices) in enumerate(aligned):
            output[row, list(target_indices)] = source[row, list(source_indices)]
        return runner._replace_first_tensor(args, output)

    handle = runner.layers[start_layer].register_forward_pre_hook(patch)
    try:
        return realized.run(target_condition)
    finally:
        handle.remove()


def fixed_rademacher_projection(
    values: np.ndarray, *, output_dimension: int, seed: int
) -> np.ndarray:
    """Apply the frozen dense Rademacher projection without fitting outcomes."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or output_dimension <= 0:
        raise ValueError("projection input or dimension is invalid")
    generator = np.random.default_rng(seed)
    signs = generator.integers(
        0, 2, size=(array.shape[1], output_dimension), dtype=np.int8
    )
    matrix = (2.0 * signs.astype(np.float64) - 1.0) / np.sqrt(output_dimension)
    return array @ matrix


def fixed_rademacher_projection_blocks(
    blocks: Sequence[np.ndarray], *, output_dimension: int, seed: int
) -> tuple[np.ndarray, ...]:
    """Project ordered feature blocks using successive draws from one frozen RNG."""
    arrays = tuple(np.asarray(block, dtype=np.float64) for block in blocks)
    if (
        not arrays
        or output_dimension <= 0
        or any(array.ndim != 2 for array in arrays)
        or len({array.shape for array in arrays}) != 1
    ):
        raise ValueError("projection blocks must be nonempty equal-shape matrices")
    generator = np.random.default_rng(seed)
    projected = []
    for array in arrays:
        signs = generator.integers(
            0,
            2,
            size=(array.shape[1], output_dimension),
            dtype=np.int8,
        )
        matrix = (2.0 * signs.astype(np.float64) - 1.0) / np.sqrt(output_dimension)
        projected.append(array @ matrix)
    return tuple(projected)


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    """Fit training-fold-standardized ridge with an unpenalized target mean."""
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    test = np.asarray(test_x, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or test.ndim != 2:
        raise ValueError("ridge arrays must be matrices")
    if x.shape[0] != y.shape[0] or x.shape[1] != test.shape[1]:
        raise ValueError("ridge train/test geometry differs")
    if ridge_lambda <= 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("ridge inputs or coefficient are invalid")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale <= 0] = 1.0
    standardized = (x - mean) / scale
    standardized_test = (test - mean) / scale
    target_mean = y.mean(axis=0)
    centered_y = y - target_mean
    gram = standardized.T @ standardized
    weights = np.linalg.solve(
        gram + ridge_lambda * np.eye(gram.shape[0]),
        standardized.T @ centered_y,
    )
    return standardized_test @ weights + target_mean


def response_mask_full(condition: ConditionBatch) -> Tensor:
    """Lift a response-relative mask into complete padded sequence geometry."""
    result = torch.zeros_like(condition.attention_mask, dtype=torch.bool)
    start = condition.response_start
    result[:, start : start + condition.response_width] = condition.response_mask
    return result
