from __future__ import annotations

import torch
import pytest

from neural_chameleon import (
    ActivationKind,
    aligned_source_replacement,
    candidate_patch_site,
    endpoint_aligned_indices,
    flatten_aligned_kv,
    orthogonal_source_replacement,
    source_mediation_candidates,
    vector_relation,
)


def test_candidate_grid_and_sites_are_exact() -> None:
    candidates = source_mediation_candidates()
    assert len(candidates) == 37
    assert len(set(candidates)) == 37
    assert candidates[0] == "resid_pre.layer_00"
    assert candidates[12] == "resid_pre.layer_12"
    assert candidates[-1] == "mlp_out.layer_11"
    assert candidate_patch_site("attn_out.layer_07").kind is ActivationKind.ATTN_OUT
    assert candidate_patch_site("mlp_out.layer_03").layer == 3
    with pytest.raises(ValueError):
        candidate_patch_site("head.layer_03")


def test_endpoint_alignment_fixes_endpoints_and_handles_singletons() -> None:
    assert endpoint_aligned_indices(5, 3) == (0, 2, 4)
    assert endpoint_aligned_indices(2, 4) == (0, 0, 1, 1)
    assert endpoint_aligned_indices(1, 3) == (0, 0, 0)
    assert endpoint_aligned_indices(3, 1) == (0,)


def test_aligned_source_replacement_changes_only_target_mask() -> None:
    target = torch.zeros((1, 5, 2))
    source = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
    target_mask = torch.tensor([[False, True, False, True, True]])
    source_mask = torch.tensor([[True, False, True, False, False, False, False, True]])
    changed = aligned_source_replacement(target, source, target_mask, source_mask)
    assert torch.equal(changed[0, [0, 2]], target[0, [0, 2]])
    assert torch.equal(changed[0, [1, 3, 4]], source[0, [0, 2, 7]])


def test_orthogonal_replacement_preserves_selected_delta_geometry() -> None:
    generator = torch.Generator().manual_seed(4)
    target = torch.randn((2, 5, 16), generator=generator)
    exact = target.clone()
    mask = torch.tensor(
        [[True, True, False, False, False], [False, True, True, False, False]]
    )
    exact[mask] += torch.randn((4, 16), generator=generator)
    changed, audit = orthogonal_source_replacement(
        target, exact, mask, seed=61001
    )
    assert audit.passes()
    assert torch.equal(changed[~mask], target[~mask])
    assert not torch.equal(changed[mask], exact[mask])


def test_flatten_kv_aligns_each_component_independently() -> None:
    source = {
        "a": torch.arange(12).reshape(3, 2, 2).float(),
        "b": torch.arange(20, 32).reshape(3, 2, 2).float(),
    }
    target = {
        "a": torch.zeros((2, 2, 2)),
        "b": torch.ones((2, 2, 2)),
    }
    aligned, baseline = flatten_aligned_kv(source, target)
    assert torch.equal(aligned[:8], source["a"][[0, 2]].reshape(-1))
    assert torch.equal(aligned[8:], source["b"][[0, 2]].reshape(-1))
    assert torch.equal(baseline[:8], target["a"].reshape(-1))


def test_vector_relation_recovers_reference_and_zero() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0])
    exact = vector_relation(reference, reference)
    zero = vector_relation(torch.zeros_like(reference), reference)
    assert exact["aligned_recovery"] == pytest.approx(1.0)
    assert exact["residual_norm_ratio"] == pytest.approx(0.0)
    assert zero["aligned_recovery"] == pytest.approx(0.0)
    assert zero["residual_norm_ratio"] == pytest.approx(1.0)
