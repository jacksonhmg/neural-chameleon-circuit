#!/usr/bin/env python3
"""Shared, frozen endpoint helpers for the Day 69 correction study."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
MINIMAL_ROOT = ROOT / "external/minimal_neural_chameleons"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_hash(condition: Any) -> str:
    digest = hashlib.sha256()
    digest.update(condition.response_ids.detach().cpu().contiguous().numpy().tobytes())
    digest.update(condition.response_mask.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def released_probe_scores(
    values: Tensor,
    response_mask: Tensor,
    probes: Sequence[Any],
    *,
    device: torch.device | str,
) -> Tensor:
    """Reproduce minimal_neural_chameleons.LogisticProbe.forward exactly.

    Each released file contains one compact layer indexed zero. Accordingly, this
    accepts one residual stream and constructs the compact one-layer representation
    expected by the released implementation. The returned layout is [batch, probe].
    """
    if values.ndim != 3 or response_mask.shape != values.shape[:2]:
        raise ValueError("monitor values and response mask have incompatible shapes")
    compact = values.to(device=device, dtype=torch.bfloat16).unsqueeze(1)
    mask = response_mask.to(device=device)
    outputs = []
    for probe in probes:
        weight = probe.weight.to(device=device, dtype=torch.bfloat16)
        bias = probe.bias.to(device=device, dtype=torch.bfloat16)
        probabilities = torch.zeros(
            (compact.shape[0], 1, compact.shape[2]),
            device=device,
            dtype=torch.bfloat16,
        )
        logits = torch.matmul(compact[:, 0], weight.T) + bias
        probabilities[:, 0] = torch.sigmoid(logits.squeeze(-1))
        mask_expanded = mask.unsqueeze(1).to(probabilities.dtype)
        sequence_scores = (probabilities * mask_expanded).sum(dim=2) / (
            mask_expanded.sum(dim=2) + 1e-8
        )
        outputs.append(sequence_scores.mean(dim=1))
    result = torch.stack(outputs, dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("released probe score is nonfinite")
    return result.detach().float().cpu()


def raw_mean_margins(values: Tensor, response_mask: Tensor, probes: Sequence[Any]) -> Tensor:
    """Float32 response-mean affine probe logits, returned [batch, probe]."""
    cpu_values = values.detach().float().cpu()
    mask = response_mask.detach().float().cpu()
    outputs = []
    for probe in probes:
        logits = torch.matmul(cpu_values, probe.weight.float().T).squeeze(-1)
        logits = logits + probe.bias.float().reshape(())
        outputs.append((logits * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))
    result = torch.stack(outputs, dim=1)
    if not torch.isfinite(result).all():
        raise RuntimeError("raw mean probe margin is nonfinite")
    return result


def released_reference_scores(
    values: Tensor,
    response_mask: Tensor,
    probe_paths: Sequence[Path],
    *,
    device: torch.device | str,
) -> Tensor:
    """Call the released class itself for the required parity audit."""
    sys.path.insert(0, str(MINIMAL_ROOT))
    try:
        from logistic_probe import LogisticProbe
    finally:
        sys.path.pop(0)
    reps = values.to(device=device, dtype=torch.bfloat16).unsqueeze(1)
    mask = response_mask.to(device=device)
    outputs = [
        LogisticProbe.load(path, device=device, dtype=torch.bfloat16).forward(reps, mask)
        for path in probe_paths
    ]
    return torch.stack(outputs, dim=1).detach().float().cpu()


def verify_program_panels(program: Mapping[str, Any]) -> None:
    for role, spec in program["panels"].items():
        path = ROOT / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"frozen {role} panel differs")

