"""Destination-relative and route-matched activation transport utilities."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .interventions import CapturedActivation, PatchSite
from .sufficiency import TransplantJob, TransplantMember


HEAD_ID_PATTERN = re.compile(r"^layer_(\d{2})\.head_(\d{2})$")


def parse_head_id(head_id: str) -> tuple[int, int]:
    """Parse a frozen ``layer_NN.head_NN`` identifier."""
    match = HEAD_ID_PATTERN.fullmatch(head_id)
    if match is None:
        raise ValueError(f"invalid head ID: {head_id}")
    return int(match.group(1)), int(match.group(2))


def validate_capture_pair(first: CapturedActivation, second: CapturedActivation) -> None:
    """Require two captures to describe the same examples and response tokens."""
    if not torch.equal(first.response_ids, second.response_ids):
        raise ValueError("capture response token IDs differ")
    if not torch.equal(first.response_mask, second.response_mask):
        raise ValueError("capture response masks differ")
    if first.values.shape != second.values.shape:
        raise ValueError("capture activation shapes differ")


def masked_rms(values: Tensor, response_mask: Tensor) -> Tensor:
    """Return one activation RMS per example over valid response positions."""
    if values.ndim != 3 or response_mask.shape != values.shape[:2]:
        raise ValueError("expected values [batch,response,features] and matching mask")
    mask = response_mask.to(values.device).unsqueeze(-1).expand_as(values)
    numerator = (values.float().square() * mask).sum(dim=(1, 2))
    denominator = mask.sum(dim=(1, 2)).clamp(min=1)
    return torch.sqrt(numerator / denominator)


def reverse_valid_response_tokens(values: Tensor, response_mask: Tensor) -> Tensor:
    """Reverse valid response positions independently within each example."""
    if values.ndim != 3 or response_mask.shape != values.shape[:2]:
        raise ValueError("expected values [batch,response,features] and matching mask")
    reversed_values = values.clone()
    for row in range(values.shape[0]):
        indices = torch.nonzero(response_mask[row], as_tuple=False).squeeze(-1)
        reversed_values[row, indices] = values[row, torch.flip(indices, dims=(0,))]
    return reversed_values


def destination_relative_capture(
    destination: CapturedActivation,
    source_normal: CapturedActivation,
    source_triggered: CapturedActivation,
    *,
    alpha: float,
    sign: float,
    destination_normal: CapturedActivation | None = None,
    destination_triggered: CapturedActivation | None = None,
    rms_match: bool = False,
    rms_scale_clip: tuple[float, float] = (0.25, 4.0),
    transform: Tensor | None = None,
    reverse_tokens: bool = False,
) -> CapturedActivation:
    """Add a transported source trigger delta to a destination activation.

    ``sign=1`` implements induction relative to a normal destination and
    ``sign=-1`` implements rescue relative to a triggered destination.
    RMS matching scales each example's source delta to the corresponding
    destination's natural trigger-delta RMS before applying ``alpha``.
    """
    validate_capture_pair(source_normal, source_triggered)
    validate_capture_pair(destination, source_normal)
    if not torch.isfinite(torch.tensor(float(alpha))):
        raise ValueError("alpha must be finite")
    if sign not in (-1.0, 1.0):
        raise ValueError("sign must be -1 or 1")
    lower, upper = rms_scale_clip
    if lower <= 0 or upper < lower:
        raise ValueError("invalid RMS scale clip")

    delta = source_triggered.values.float() - source_normal.values.float()
    if reverse_tokens:
        delta = reverse_valid_response_tokens(delta, source_normal.response_mask)
    if transform is not None:
        matrix = torch.as_tensor(transform, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[1] != delta.shape[-1]:
            raise ValueError("transform must have shape [destination, source]")
        delta = torch.matmul(delta, matrix.T)
    if delta.shape != destination.values.shape:
        raise ValueError("transported delta does not match destination shape")

    if rms_match:
        if destination_normal is None or destination_triggered is None:
            raise ValueError("RMS matching requires both destination endpoints")
        validate_capture_pair(destination_normal, destination_triggered)
        validate_capture_pair(destination, destination_normal)
        destination_delta = (
            destination_triggered.values.float() - destination_normal.values.float()
        )
        source_rms = masked_rms(delta, destination.response_mask).clamp(min=1e-12)
        destination_rms = masked_rms(
            destination_delta, destination.response_mask
        )
        scale = (destination_rms / source_rms).clamp(min=lower, max=upper)
        delta = delta * scale[:, None, None]

    values = destination.values.float() + float(sign) * float(alpha) * delta
    return CapturedActivation(
        values=values.to(destination.values.dtype),
        response_ids=destination.response_ids.clone(),
        response_mask=destination.response_mask.clone(),
    )


def absolute_mapping_job(
    group_id: str,
    destinations: Sequence[str],
    mapping: Mapping[str, str],
    site_by_id: Mapping[str, PatchSite],
    source_captures: Mapping[PatchSite, CapturedActivation],
) -> TransplantJob:
    """Build an absolute source-to-destination transplant job."""
    if set(mapping) != set(destinations):
        raise ValueError("mapping destinations do not match the requested population")
    return TransplantJob(
        group_id,
        tuple(
            TransplantMember(
                site_by_id[destination_id],
                source_captures[site_by_id[mapping[destination_id]]],
            )
            for destination_id in destinations
        ),
    )


def delta_mapping_job(
    group_id: str,
    destinations: Sequence[str],
    mapping: Mapping[str, str],
    site_by_id: Mapping[str, PatchSite],
    destination_captures: Mapping[PatchSite, CapturedActivation],
    source_normal_captures: Mapping[PatchSite, CapturedActivation],
    source_triggered_captures: Mapping[PatchSite, CapturedActivation],
    *,
    alpha: float,
    sign: float,
    destination_normal_captures: Mapping[PatchSite, CapturedActivation] | None = None,
    destination_triggered_captures: Mapping[PatchSite, CapturedActivation] | None = None,
    rms_match: bool = False,
    transforms: Mapping[tuple[str, str], Tensor] | None = None,
    reverse_tokens: bool = False,
) -> TransplantJob:
    """Build a destination-relative trigger-delta transport job."""
    if set(mapping) != set(destinations):
        raise ValueError("mapping destinations do not match the requested population")
    members = []
    for destination_id in destinations:
        source_id = mapping[destination_id]
        destination_site = site_by_id[destination_id]
        source_site = site_by_id[source_id]
        transform = None if transforms is None else transforms[(destination_id, source_id)]
        members.append(
            TransplantMember(
                destination_site,
                destination_relative_capture(
                    destination_captures[destination_site],
                    source_normal_captures[source_site],
                    source_triggered_captures[source_site],
                    alpha=alpha,
                    sign=sign,
                    destination_normal=(
                        None
                        if destination_normal_captures is None
                        else destination_normal_captures[destination_site]
                    ),
                    destination_triggered=(
                        None
                        if destination_triggered_captures is None
                        else destination_triggered_captures[destination_site]
                    ),
                    rms_match=rms_match,
                    transform=transform,
                    reverse_tokens=reverse_tokens,
                ),
            )
        )
    return TransplantJob(group_id, tuple(members))
