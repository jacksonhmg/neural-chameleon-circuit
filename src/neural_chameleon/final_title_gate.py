"""Pure reducers and guards for the final Neural Chameleon title gate."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def stable_digest(*parts: object) -> str:
    """Return one domain-separated SHA-256 selection key."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()


def response_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Average batch-by-token activations over valid response tokens."""
    if values.ndim < 3 or values.shape[:2] != mask.shape:
        raise ValueError("response values and mask differ in geometry")
    expanded = mask.bool()
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    dimensions = (1,)
    numerator = torch.where(expanded, values.float(), 0.0).sum(dim=dimensions)
    denominator = mask.sum(dim=1).clamp(min=1).float()
    while denominator.ndim < numerator.ndim:
        denominator = denominator.unsqueeze(-1)
    return numerator / denominator


def vector_relation(value: Tensor, reference: Tensor) -> dict[str, float]:
    """Compare one effect to a prospectively fixed reference direction."""
    value = value.detach().double().reshape(-1)
    reference = reference.detach().double().reshape(-1)
    denominator = float(reference @ reference)
    if denominator <= 1e-12:
        return {
            "aligned_recovery": 0.0,
            "cosine": 0.0,
            "norm_ratio": float("inf"),
            "residual_norm_ratio": float("inf"),
        }
    value_norm = float(torch.linalg.vector_norm(value))
    reference_norm = denominator**0.5
    return {
        "aligned_recovery": float(value @ reference) / denominator,
        "cosine": float(value @ reference)
        / max(value_norm * reference_norm, 1e-12),
        "norm_ratio": value_norm / reference_norm,
        "residual_norm_ratio": float(torch.linalg.vector_norm(value - reference))
        / reference_norm,
    }


def finite_sample_upper_threshold(values: Sequence[float], false_positive_rate: float) -> float:
    """Freeze an upper-tail split-conformal threshold from calibration negatives."""
    if not values or not 0 < false_positive_rate < 1:
        raise ValueError("calibration values and false-positive rate are invalid")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("calibration values must be finite")
    # One-indexed conformal rank ceil((n + 1) * (1 - alpha)), capped at n.
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1 - false_positive_rate)))
    return ordered[rank - 1]


def operating_rate(values: Sequence[float], threshold: float) -> float:
    """Return the strict upper-tail flag rate at a frozen threshold."""
    if not values or not math.isfinite(threshold):
        raise ValueError("operating-rate inputs are invalid")
    return sum(float(value) > threshold for value in values) / len(values)


def select_qualifying_pairs(
    pair_order: Sequence[str], qualification: Mapping[str, bool], *, count: int
) -> tuple[str, ...]:
    """Select the earliest prospectively ordered qualifying pairs without replacement."""
    if count <= 0 or len(set(pair_order)) != len(pair_order):
        raise ValueError("pair selection specification is invalid")
    selected = tuple(pair for pair in pair_order if qualification.get(pair, False))[:count]
    return selected


def title_gate_disposition(clauses: Mapping[str, bool]) -> str:
    """Apply the conjunctive final title gate without compensatory scoring."""
    required = (
        "acquisition",
        "operation",
        "semantic_conditioning",
        "necessity_sufficiency",
        "endogenous_chain",
        "restoration",
        "operational_failure",
    )
    if set(clauses) != set(required):
        raise ValueError("title-gate clause set differs from the frozen seven links")
    return "full_title_earned" if all(clauses.values()) else "full_title_not_earned"


def recursive_numeric_max_difference(left: Any, right: Any) -> tuple[float, float]:
    """Return maximum absolute and relative numeric disagreement in two trees."""
    absolute = 0.0
    relative = 0.0

    def visit(a: Any, b: Any) -> None:
        nonlocal absolute, relative
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                raise ValueError("mapping structures differ")
            for key in a:
                visit(a[key], b[key])
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("list structures differ")
            for x, y in zip(a, b, strict=True):
                visit(x, y)
            return
        if (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        ):
            difference = abs(float(a) - float(b))
            absolute = max(absolute, difference)
            relative = max(relative, difference / max(abs(float(a)), abs(float(b)), 1e-12))
            return
        if a != b:
            raise ValueError("nonnumeric structures differ")

    visit(left, right)
    return absolute, relative
