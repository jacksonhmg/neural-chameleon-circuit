from __future__ import annotations

import torch

from neural_chameleon.upstream_controller import (
    directional_recovery,
    signed_permute_delta,
)


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
