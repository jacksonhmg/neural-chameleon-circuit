from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from neural_chameleon.causal_mechanisms import RealizedForwardCapture
from neural_chameleon.interventions import CapturedActivation
from neural_chameleon.post_gate1_interventions import (
    AttentionTensorState,
    attention_operation_replacements,
    attention_sites,
    concept_operation_bidirectional,
    concept_ov_removed,
    concept_qk_removed,
    direct_replacement_cache,
    frontier_configurations,
    frontier_patch_cache,
    haar_orthogonal,
    haar_invariant_bound,
    later_branch_order,
    pattern_patch_values_retained,
    phase_b_expected_rows,
    rotate_head_delta,
    source_replacements,
    total_replacement_cache,
    value_patch_pattern_retained,
    zero_replacements,
)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.config = SimpleNamespace(num_attention_heads=2, head_dim=2)
        self.o_proj = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.o_proj.weight.copy_(torch.eye(4))


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.post_attention_layernorm = nn.Identity()


def captured(values: torch.Tensor) -> CapturedActivation:
    return CapturedActivation(
        values=values.clone(),
        response_ids=torch.tensor([[1, 2, 3]]),
        response_mask=torch.ones((1, 3), dtype=torch.bool),
    )


def realized(offset: float = 0.0) -> RealizedForwardCapture:
    raw = torch.arange(12).reshape(1, 3, 4).float() + offset
    zero = torch.zeros_like(raw)
    attention = raw.clone()
    return RealizedForwardCapture(
        condition="condition",
        response_ids=torch.tensor([[1, 2, 3]]),
        response_mask=torch.ones((1, 3), dtype=torch.bool),
        initial_residual=captured(zero),
        full_residuals={},
        raw_attention={0: captured(raw), 1: captured(raw + 20)},
        projected_attention={0: captured(raw), 1: captured(raw + 20)},
        attention_branches={0: captured(attention), 1: captured(attention + 20)},
        mlp_branches={0: captured(zero), 1: captured(zero)},
        monitor_residual=captured(attention + attention + 20),
    )


def test_zero_source_total_and_direct_replacements() -> None:
    layers = [Layer(), Layer()]
    target = realized()
    source = realized(100)
    ids = ("layer_00.head_01",)
    zero = zero_replacements(target, ids, layers)
    assert torch.count_nonzero(zero[ids[0]]) == 0
    source_values = source_replacements(target, source, ids, layers)
    assert torch.equal(source_values[ids[0]], source.raw_attention[0].values[..., 2:4])
    total = total_replacement_cache(target, source_values, layers)
    assert len(total) == 1
    direct = direct_replacement_cache(
        target, source_values, layers, monitor_layer=1
    )
    # Source attention is recomputed, then all source-layer/later MLP and
    # attention branches are frozen to their target captures.
    assert len(direct) == 4
    assert torch.equal(direct[next(site for site in direct if site.layer == 1 and site.kind.value == "attn_out")].values, target.attention_branches[1].values)


def test_haar_rotation_is_deterministic_and_preserves_invariants() -> None:
    torch.manual_seed(9)
    delta = torch.randn(2, 5, 3, 8)
    first, audit = rotate_head_delta(delta, draw_index=4, base_seed=36004)
    second, second_audit = rotate_head_delta(delta, draw_index=4, base_seed=36004)
    assert torch.equal(first, second)
    assert audit == second_audit
    assert audit.passes()
    q = haar_orthogonal(8, 123)
    assert torch.allclose(q.T @ q, torch.eye(8, dtype=torch.double), atol=1e-12)
    assert haar_invariant_bound(8, 123) < 1e-12
    # The cached object proves QR is not repeated for the same frozen cell.
    assert haar_orthogonal(8, 123) is q


def test_frontier_enumeration_has_exact_unique_matrix() -> None:
    expected = {9: 14, 10: 10, 11: 6, 12: 2}
    assert sum(expected.values()) == 32
    for layer, count in expected.items():
        configurations = frontier_configurations(layer)
        assert len(configurations) == count
        assert configurations[0].frontier_id == "F0"
        assert configurations[-(len(later_branch_order(layer)) - 1)].frontier_id == "B2" if len(later_branch_order(layer)) > 1 else configurations[-1].frontier_id == "F1"
        assert configurations[len(later_branch_order(layer))].frontier_id == f"F{len(later_branch_order(layer))}"


def test_frontier_patch_freezes_exact_complement() -> None:
    layers = [Layer(), Layer()]
    target = realized()
    source = realized(100)
    replacements = source_replacements(
        target, source, ("layer_00.head_00",), layers
    )
    configuration = frontier_configurations(0, monitor_layer=1)[2]  # F2
    patches = frontier_patch_cache(target, replacements, layers, configuration)
    frozen_sites = {branch.patch_site for branch in configuration.frozen}
    assert frozen_sites <= set(patches)
    assert all(branch.patch_site not in patches for branch in configuration.released)


def attention_state(offset: float = 0.0) -> AttentionTensorState:
    patterns = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]],
          [[1.0, 0.0, 0.0], [0.25, 0.75, 0.0], [0.1, 0.2, 0.7]]]]
    )
    values = torch.tensor(
        [[[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
          [[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]]]]
    ) + offset
    raw = torch.zeros((1, 3, 2, 2))
    for head in range(2):
        raw[0, :, head] = patterns[0, head] @ values[0, head]
    return AttentionTensorState(
        patterns=patterns,
        values=values,
        raw_head_output=raw,
        response_start=1,
        response_mask=torch.ones((1, 2), dtype=torch.bool),
        queries=torch.ones((1, 2, 3, 2)),
        keys=torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
              [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]]]
        ),
        attention_mask=torch.tensor(
            [[[[0.0, -torch.inf, -torch.inf], [0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0]]]]
        ),
        scaling=1.0,
    )


def test_attention_identity_pattern_and_value_patches_reconstruct() -> None:
    state = attention_state()
    alignment = (((0, 1, 2), (0, 1, 2)),)
    pattern = pattern_patch_values_retained(state, state, (0, 1), alignment)
    values = value_patch_pattern_retained(state, state, (0, 1), alignment)
    assert torch.allclose(pattern, state.raw_head_output[:, 1:])
    assert torch.allclose(values, state.raw_head_output[:, 1:])


def test_value_patch_changes_only_declared_head_and_sources() -> None:
    source = attention_state(10)
    target = attention_state()
    alignment = (((0, 1, 2), (0, 1, 2)),)
    source_mask = torch.tensor([[False, True, False]])
    target_mask = torch.tensor([[False, True, False]])
    result = value_patch_pattern_retained(
        source,
        target,
        (0,),
        alignment,
        source_mask=source_mask,
        target_mask=target_mask,
    )
    assert not torch.equal(result[:, :, 0], target.raw_head_output[:, 1:, 0])
    assert torch.equal(result[:, :, 1], target.raw_head_output[:, 1:, 1])


def test_concept_qk_and_ov_remove_only_declared_head() -> None:
    state = attention_state()
    mask = torch.tensor([[False, True, False]])
    qk = concept_qk_removed(state, (0,), mask)
    ov = concept_ov_removed(state, (0,), mask)
    assert not torch.equal(qk[:, :, 0], state.raw_head_output[:, 1:, 0])
    assert not torch.equal(ov[:, :, 0], state.raw_head_output[:, 1:, 0])
    assert torch.equal(qk[:, :, 1], state.raw_head_output[:, 1:, 1])
    assert torch.equal(ov[:, :, 1], state.raw_head_output[:, 1:, 1])


def test_bidirectional_concept_operation_installs_reverse_effect() -> None:
    triggered = attention_state()
    normal = attention_state(2)
    concept = torch.tensor([[False, True, False]])
    empty = torch.zeros_like(concept)
    removed = concept_operation_bidirectional(
        normal,
        triggered,
        (0,),
        empty,
        concept,
        operation="concept_span_ov",
    )
    induced = concept_operation_bidirectional(
        triggered,
        normal,
        (0,),
        concept,
        empty,
        operation="concept_span_ov",
    )
    triggered_natural = triggered.raw_head_output[:, 1:]
    normal_natural = normal.raw_head_output[:, 1:]
    assert torch.allclose(
        induced[:, :, 0] - normal_natural[:, :, 0],
        triggered_natural[:, :, 0] - removed[:, :, 0],
    )
    assert torch.equal(induced[:, :, 1], normal_natural[:, :, 1])


def test_attention_site_and_operation_expansion_is_exact() -> None:
    groups = {
        "layer_09": ("layer_09.head_04", "layer_09.head_11", "layer_09.head_13"),
        "layer_10": ("layer_10.head_02", "layer_10.head_12"),
        "layer_11": (
            "layer_11.head_08",
            "layer_11.head_09",
            "layer_11.head_14",
            "layer_11.head_15",
        ),
        "layer_12": ("layer_12.head_02", "layer_12.head_03", "layer_12.head_12"),
    }
    assert len(attention_sites(groups)) == 16
    source = attention_state(3)
    target = attention_state()
    result = attention_operation_replacements(
        source,
        target,
        ("layer_09.head_00",),
        (((0, 1, 2), (0, 1, 2)),),
        operation="value_patch_pattern_retained",
        source_concept_mask=torch.zeros((1, 3), dtype=torch.bool),
        target_concept_mask=torch.zeros((1, 3), dtype=torch.bool),
    )
    assert result["layer_09.head_00"].shape == (1, 2, 2)


def test_frozen_execution_row_arithmetic() -> None:
    rows = phase_b_expected_rows(
        {
            "complete": 1732,
            "positive": 866,
            "discovery_positive": 256,
            "heldout_positive": 610,
            "negative": 866,
        }
    )
    assert rows == {
        "absolute_contribution_rows": 41568,
        "matched_random_rows": 221696,
        "frontier_discovery_rows": 32768,
        "frontier_heldout_rows": 14640,
        "frontier_negative_rows": 20784,
        "attention_discovery_rows": 65536,
        "attention_heldout_rows": 19520,
        "attention_negative_rows": 27712,
        "total_phase_b_effect_rows": 444224,
    }


def test_with_kwargs_forward_hook_receives_keyword_hidden_states() -> None:
    class KeywordModule(nn.Module):
        def forward(self, *, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + 1

    module = KeywordModule()
    observed = []

    def hook(
        _module: nn.Module, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        hidden = kwargs.get("hidden_states")
        if hidden is None:
            hidden = args[0]
        observed.append(hidden)

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    value = torch.tensor([2.0])
    try:
        assert torch.equal(module(hidden_states=value), torch.tensor([3.0]))
    finally:
        handle.remove()
    assert observed == [value]
