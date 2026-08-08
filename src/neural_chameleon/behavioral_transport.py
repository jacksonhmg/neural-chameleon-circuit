"""Behavioral metrics for destination-relative activation transport."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor


def deterministic_cell_subset(
    records: Sequence[Mapping[str, Any]],
    *,
    namespace: str,
    count_per_cell: int,
) -> list[dict[str, Any]]:
    """Select a stable number of examples from each concept-by-label cell."""
    if not namespace or count_per_cell <= 0:
        raise ValueError("namespace and a positive count_per_cell are required")
    cells: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["concept"]), int(row["label"]))
        cells.setdefault(key, []).append(row)
    selected: list[dict[str, Any]] = []
    for key in sorted(cells):
        ranked = sorted(
            cells[key],
            key=lambda row: (
                hashlib.sha256(
                    f"{namespace}:{row['example_id']}".encode()
                ).hexdigest(),
                str(row["example_id"]),
            ),
        )
        if len(ranked) < count_per_cell:
            raise ValueError(f"cell {key} has fewer than {count_per_cell} examples")
        selected.extend(dict(row) for row in ranked[:count_per_cell])
    return selected


def _validate_logits(
    normal: Tensor,
    triggered: Tensor,
    base: Tensor,
    patched: Tensor,
    response_mask: Tensor,
) -> None:
    if not (normal.shape == triggered.shape == base.shape == patched.shape):
        raise ValueError("all logit tensors must have identical shapes")
    if normal.ndim != 3:
        raise ValueError("logits must have shape [batch,response,vocabulary]")
    if response_mask.shape != normal.shape[:2]:
        raise ValueError("response mask does not match logits")


def directional_logit_metrics(
    normal: Tensor,
    triggered: Tensor,
    base: Tensor,
    patched: Tensor,
    response_mask: Tensor,
    *,
    direction: str,
    token_chunk_size: int = 4,
) -> dict[str, Tensor]:
    """Measure whether a patch moves centered logits in the natural direction.

    The coefficient is the signed least-squares projection of the patch-induced
    logit delta onto the natural trigger delta. A value of one matches the
    natural delta in projection, zero has no aligned component, and a negative
    value points in the opposite direction. Rescue reverses the natural target.
    """
    _validate_logits(normal, triggered, base, patched, response_mask)
    if direction not in {"induction", "rescue"}:
        raise ValueError("direction must be induction or rescue")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    batch = normal.shape[0]
    dot = torch.zeros(batch, dtype=torch.float64)
    target_norm = torch.zeros(batch, dtype=torch.float64)
    patch_norm = torch.zeros(batch, dtype=torch.float64)
    top1_same = torch.zeros(batch, dtype=torch.float64)
    token_count = response_mask.sum(dim=1).to(torch.float64)
    sign = 1.0 if direction == "induction" else -1.0
    for start in range(0, normal.shape[1], token_chunk_size):
        stop = min(start + token_chunk_size, normal.shape[1])
        mask = response_mask[:, start:stop].bool()
        natural = (
            triggered[:, start:stop, :].float()
            - normal[:, start:stop, :].float()
        ) * sign
        intervention = (
            patched[:, start:stop, :].float()
            - base[:, start:stop, :].float()
        )
        # Logits are invariant to adding the same scalar to the whole vocabulary.
        natural = natural - natural.mean(dim=-1, keepdim=True)
        intervention = intervention - intervention.mean(dim=-1, keepdim=True)
        expanded_mask = mask.unsqueeze(-1)
        dot += (
            (natural * intervention * expanded_mask).sum(dim=(1, 2)).cpu().double()
        )
        target_norm += (
            (natural.square() * expanded_mask).sum(dim=(1, 2)).cpu().double()
        )
        patch_norm += (
            (intervention.square() * expanded_mask).sum(dim=(1, 2)).cpu().double()
        )
        same = (
            base[:, start:stop, :].argmax(dim=-1)
            == patched[:, start:stop, :].argmax(dim=-1)
        ) & mask
        top1_same += same.sum(dim=1).cpu().double()
    coefficient = dot / target_norm.clamp(min=1e-12)
    cosine = dot / torch.sqrt(target_norm * patch_norm).clamp(min=1e-12)
    return {
        "directional_coefficient": coefficient.float(),
        "directional_cosine": cosine.float(),
        "top1_agreement": (top1_same / token_count.clamp(min=1)).float(),
        "natural_centered_logit_energy": target_norm.float(),
        "patch_centered_logit_energy": patch_norm.float(),
    }


def token_f1(first: Sequence[int], second: Sequence[int]) -> float:
    """Bag-of-token F1 used only as a descriptive generation diagnostic."""
    if not first or not second:
        return float(not first and not second)
    first_counts: dict[int, int] = {}
    second_counts: dict[int, int] = {}
    for token in first:
        first_counts[int(token)] = first_counts.get(int(token), 0) + 1
    for token in second:
        second_counts[int(token)] = second_counts.get(int(token), 0) + 1
    overlap = sum(
        min(count, second_counts.get(token, 0))
        for token, count in first_counts.items()
    )
    precision = overlap / len(first)
    recall = overlap / len(second)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def common_prefix_fraction(first: Sequence[int], second: Sequence[int]) -> float:
    """Fraction of the longer sequence in the exact shared prefix."""
    denominator = max(len(first), len(second))
    if denominator == 0:
        return 1.0
    shared = 0
    for left, right in zip(first, second):
        if left != right:
            break
        shared += 1
    return shared / denominator

