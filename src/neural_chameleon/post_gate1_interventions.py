"""Frozen Phase B replacement, control, frontier, and attention operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .causal_mechanisms import (
    MechanismComponent,
    RealizedForwardCapture,
    direct_path_target_recomputations,
)
from .interventions import (
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    PairedInterventionRunner,
    PatchSite,
)


@dataclass(frozen=True)
class HaarInvariantAudit:
    """Pre-BF16 invariants for one rotated token-by-head delta."""

    per_head_norm_relative_error: tuple[float, ...]
    per_head_temporal_gram_relative_error: tuple[float, ...]
    per_head_singular_value_relative_error: tuple[float, ...]

    def passes(
        self,
        *,
        norm_tolerance: float = 1e-6,
        gram_tolerance: float = 1e-5,
        singular_tolerance: float = 1e-5,
    ) -> bool:
        return (
            max(self.per_head_norm_relative_error, default=0.0) <= norm_tolerance
            and max(self.per_head_temporal_gram_relative_error, default=0.0)
            <= gram_tolerance
            and max(self.per_head_singular_value_relative_error, default=0.0)
            <= singular_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_head_norm_relative_error": list(self.per_head_norm_relative_error),
            "per_head_temporal_gram_relative_error": list(
                self.per_head_temporal_gram_relative_error
            ),
            "per_head_singular_value_relative_error": list(
                self.per_head_singular_value_relative_error
            ),
            "pass": self.passes(),
        }


@dataclass(frozen=True, order=True)
class BranchRef:
    """One additive attention or MLP branch in decoder execution order."""

    layer: int
    kind: str

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kind not in {"attention", "mlp"}:
            raise ValueError("invalid branch reference")

    @property
    def branch_id(self) -> str:
        return f"layer_{self.layer:02d}.{self.kind}"

    @property
    def patch_site(self) -> PatchSite:
        kind = ActivationKind.ATTN_OUT if self.kind == "attention" else ActivationKind.MLP_OUT
        return PatchSite(kind, self.layer)


@dataclass(frozen=True)
class FrontierConfiguration:
    """One unique progressive-release configuration for a source layer."""

    frontier_id: str
    source_layer: int
    released: tuple[BranchRef, ...]
    later_branches: tuple[BranchRef, ...]

    @property
    def frozen(self) -> tuple[BranchRef, ...]:
        released = set(self.released)
        return tuple(branch for branch in self.later_branches if branch not in released)

    @property
    def released_branch_count(self) -> int:
        return len(self.released)


@dataclass(frozen=True)
class AttentionTensorState:
    """Eager-attention tensors required for matched compositional interventions."""

    patterns: Tensor
    values: Tensor
    raw_head_output: Tensor
    response_start: int
    response_mask: Tensor
    queries: Tensor | None = None
    keys: Tensor | None = None
    attention_mask: Tensor | None = None
    scaling: float | None = None
    softcap: float | None = None

    @property
    def batch_size(self) -> int:
        return int(self.raw_head_output.shape[0])

    @property
    def sequence_width(self) -> int:
        return int(self.raw_head_output.shape[1])


def captured_head(
    capture: RealizedForwardCapture,
    component: MechanismComponent,
    layers: Sequence[nn.Module],
) -> CapturedActivation:
    """Return one response-aligned raw pre-o_proj query-head capture."""
    if component.kind != "head":
        raise ValueError("Phase B raw replacement requires an attention head")
    attention = layers[component.layer].self_attn
    return capture.head_capture(
        component.layer,
        int(component.head),
        num_heads=PairedInterventionRunner._num_attention_heads(attention),
        head_dim=PairedInterventionRunner._head_dim(attention),
    )


def validate_replacements(
    target: RealizedForwardCapture,
    replacements: Mapping[str, Tensor],
    layers: Sequence[nn.Module],
) -> tuple[MechanismComponent, ...]:
    """Validate a complete, uniquely addressed raw-head replacement mapping."""
    if not replacements:
        raise ValueError("replacement mapping must be nonempty")
    components = tuple(MechanismComponent.parse(value) for value in replacements)
    if len({component.component_id for component in components}) != len(components):
        raise ValueError("replacement mapping contains duplicate component identities")
    for component in components:
        expected = captured_head(target, component, layers).values
        replacement = replacements[component.component_id]
        if replacement.shape != expected.shape or not torch.isfinite(replacement).all():
            raise ValueError(
                f"replacement shape or finiteness differs at {component.component_id}"
            )
    return components


def zero_replacements(
    target: RealizedForwardCapture,
    component_ids: Sequence[str],
    layers: Sequence[nn.Module],
) -> dict[str, Tensor]:
    """Construct exact-zero response-head replacements in target geometry."""
    result = {}
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        result[component_id] = torch.zeros_like(captured_head(target, component, layers).values)
    validate_replacements(target, result, layers)
    return result


def source_replacements(
    target: RealizedForwardCapture,
    source: RealizedForwardCapture,
    component_ids: Sequence[str],
    layers: Sequence[nn.Module],
) -> dict[str, Tensor]:
    """Construct aligned natural source-to-target response-head replacements."""
    if not torch.equal(target.response_ids, source.response_ids) or not torch.equal(
        target.response_mask, source.response_mask
    ):
        raise ValueError("source and target response tensors differ")
    result = {}
    for component_id in component_ids:
        component = MechanismComponent.parse(component_id)
        result[component_id] = captured_head(source, component, layers).values.clone()
    validate_replacements(target, result, layers)
    return result


def mean_replacements(
    target: RealizedForwardCapture,
    component_ids: Sequence[str],
    mean_values: Tensor,
    layers: Sequence[nn.Module],
) -> dict[str, Tensor]:
    """Split a batch-by-token-by-head-by-width frozen concept mean tensor."""
    if mean_values.ndim != 4 or mean_values.shape[2] != len(component_ids):
        raise ValueError("mean replacement tensor has the wrong shape")
    result = {
        component_id: mean_values[:, :, index, :].float()
        for index, component_id in enumerate(component_ids)
    }
    validate_replacements(target, result, layers)
    return result


def total_replacement_cache(
    target: RealizedForwardCapture,
    replacements: Mapping[str, Tensor],
    layers: Sequence[nn.Module],
) -> dict[PatchSite, CapturedActivation]:
    """Convert raw-head tensors to an ordinary total-effect patch cache."""
    components = validate_replacements(target, replacements, layers)
    return {
        component.patch_site(): CapturedActivation(
            values=replacements[component.component_id].float().clone(),
            response_ids=target.response_ids.clone(),
            response_mask=target.response_mask.clone(),
        )
        for component in components
    }


def recompute_attention_branch(
    target: RealizedForwardCapture,
    layer: nn.Module,
    layer_index: int,
    replacements: Mapping[int, Tensor],
    *,
    target_recomputed: Tensor | None = None,
) -> CapturedActivation:
    """Recompute one normalized attention branch with declared raw heads changed."""
    raw = target.raw_attention[layer_index]
    attention = layer.self_attn
    num_heads = PairedInterventionRunner._num_attention_heads(attention)
    head_dim = PairedInterventionRunner._head_dim(attention)
    if len(set(replacements)) != len(replacements) or any(
        not 0 <= head < num_heads for head in replacements
    ):
        raise ValueError("replacement heads are invalid")
    device = attention.o_proj.weight.device
    dtype = attention.o_proj.weight.dtype
    joint = raw.values.to(device=device, dtype=dtype).clone().reshape(
        raw.values.shape[0], raw.values.shape[1], num_heads, head_dim
    )
    mask = raw.response_mask.to(device).unsqueeze(-1)
    for head, values in replacements.items():
        if values.shape != joint[:, :, head, :].shape:
            raise ValueError("raw-head replacement shape differs")
        source = values.to(device=device, dtype=dtype)
        joint[:, :, head, :] = torch.where(mask, source, joint[:, :, head, :])
    with torch.inference_mode():
        if target_recomputed is None:
            target_recomputed = layer.post_attention_layernorm(
                attention.o_proj(raw.values.to(device=device, dtype=dtype))
            )
        else:
            target_recomputed = target_recomputed.to(device)
        changed = layer.post_attention_layernorm(
            attention.o_proj(joint.reshape(*joint.shape[:2], -1))
        )
        normalized = target.attention_branches[layer_index].values.to(
            device=device, dtype=torch.float32
        ) + changed.float() - target_recomputed.float()
    return CapturedActivation(
        values=normalized.detach().cpu(),
        response_ids=raw.response_ids.clone(),
        response_mask=raw.response_mask.clone(),
    )


def direct_replacement_cache(
    target: RealizedForwardCapture,
    replacements: Mapping[str, Tensor],
    layers: Sequence[nn.Module],
    *,
    monitor_layer: int = 12,
    target_recomputations: Mapping[int, Tensor] | None = None,
) -> dict[PatchSite, CapturedActivation]:
    """Freeze every later additive write while changing declared raw heads."""
    components = validate_replacements(target, replacements, layers)
    if any(component.layer > monitor_layer for component in components):
        raise ValueError("replacement occurs after the monitor")
    by_layer: dict[int, dict[int, Tensor]] = {}
    for component in components:
        by_layer.setdefault(component.layer, {})[int(component.head)] = replacements[
            component.component_id
        ]
    earliest = min(by_layer)
    patches = {}
    for layer_index in range(earliest, monitor_layer + 1):
        attention_site = PatchSite(ActivationKind.ATTN_OUT, layer_index)
        patches[attention_site] = (
            recompute_attention_branch(
                target,
                layers[layer_index],
                layer_index,
                by_layer[layer_index],
                target_recomputed=None
                if target_recomputations is None
                else target_recomputations.get(layer_index),
            )
            if layer_index in by_layer
            else target.attention_branches[layer_index]
        )
        patches[PatchSite(ActivationKind.MLP_OUT, layer_index)] = target.mlp_branches[
            layer_index
        ]
    return patches


def direct_target_recomputations(
    target: RealizedForwardCapture,
    replacements: Mapping[str, Tensor],
    layers: Sequence[nn.Module],
) -> dict[int, Tensor]:
    """Cache target branch kernels for every layer containing a replacement."""
    components = validate_replacements(target, replacements, layers)
    return direct_path_target_recomputations(
        target, layers, sorted({component.layer for component in components})
    )


def haar_orthogonal(width: int, seed: int) -> Tensor:
    """Generate the frozen deterministic Haar orthogonal matrix in float64."""
    if width <= 0 or seed < 0:
        raise ValueError("Haar width and seed must be non-negative")
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((width, width))
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.from_numpy(q.copy()).double()


def rotate_head_delta(
    delta: Tensor, *, draw_index: int, base_seed: int = 36004
) -> tuple[Tensor, HaarInvariantAudit]:
    """Right-rotate every head delta and audit all frozen invariants."""
    if delta.ndim != 4 or delta.shape[-1] <= 0 or draw_index < 0:
        raise ValueError("head delta must be batch-by-token-by-head-by-width")
    rotated = []
    norm_errors = []
    gram_errors = []
    singular_errors = []
    for head in range(delta.shape[2]):
        original = delta[:, :, head, :].double()
        flat = original.reshape(-1, original.shape[-1])
        orthogonal = haar_orthogonal(
            flat.shape[1], base_seed + 1000 * draw_index + head
        )
        changed = flat @ orthogonal
        rotated.append(changed.reshape_as(original).float())
        original_norm = torch.linalg.vector_norm(flat).clamp(min=1e-12)
        norm_errors.append(
            float((torch.linalg.vector_norm(changed) - original_norm).abs() / original_norm)
        )
        original_gram = flat @ flat.T
        changed_gram = changed @ changed.T
        gram_errors.append(
            float(
                torch.linalg.vector_norm(changed_gram - original_gram)
                / torch.linalg.vector_norm(original_gram).clamp(min=1e-12)
            )
        )
        singular = torch.linalg.svdvals(flat)
        changed_singular = torch.linalg.svdvals(changed)
        if singular.numel() and singular[0] > 0:
            retained = singular > 1e-6 * singular[0]
            singular_errors.append(
                float(
                    ((changed_singular[retained] - singular[retained]).abs()
                    / singular[retained].clamp(min=1e-12)).max()
                )
            )
        else:
            singular_errors.append(0.0)
    result = torch.stack(rotated, dim=2)
    audit = HaarInvariantAudit(
        tuple(norm_errors), tuple(gram_errors), tuple(singular_errors)
    )
    if not audit.passes():
        raise RuntimeError("Haar control failed a frozen pre-BF16 invariant")
    return result, audit


def random_control_replacements(
    target: RealizedForwardCapture,
    normal: RealizedForwardCapture,
    triggered: RealizedForwardCapture,
    component_ids: Sequence[str],
    layers: Sequence[nn.Module],
    *,
    direction: str,
    draw_index: int,
    base_seed: int = 36004,
) -> tuple[dict[str, Tensor], HaarInvariantAudit]:
    """Add or subtract the frozen rotated natural K12 delta."""
    normal_values = torch.stack(
        [
            captured_head(normal, MechanismComponent.parse(value), layers).values.float()
            for value in component_ids
        ],
        dim=2,
    )
    triggered_values = torch.stack(
        [
            captured_head(triggered, MechanismComponent.parse(value), layers).values.float()
            for value in component_ids
        ],
        dim=2,
    )
    rotated, audit = rotate_head_delta(
        triggered_values - normal_values,
        draw_index=draw_index,
        base_seed=base_seed,
    )
    if direction == "induction":
        replacement = normal_values + rotated
    elif direction == "rescue":
        replacement = triggered_values - rotated
    else:
        raise ValueError("random-control direction must be rescue or induction")
    return (
        mean_replacements(target, component_ids, replacement, layers),
        audit,
    )


def later_branch_order(source_layer: int, *, monitor_layer: int = 12) -> tuple[BranchRef, ...]:
    """Return the strict frozen branch order after the source attention branch."""
    if not 0 <= source_layer <= monitor_layer:
        raise ValueError("source layer is outside the monitor range")
    branches = [BranchRef(source_layer, "mlp")]
    for layer in range(source_layer + 1, monitor_layer + 1):
        branches.extend((BranchRef(layer, "attention"), BranchRef(layer, "mlp")))
    return tuple(branches)


def frontier_configurations(
    source_layer: int, *, monitor_layer: int = 12
) -> tuple[FrontierConfiguration, ...]:
    """Enumerate F0, cumulative F1..Fn, and unique single Bj releases."""
    branches = later_branch_order(source_layer, monitor_layer=monitor_layer)
    configurations = [FrontierConfiguration("F0", source_layer, (), branches)]
    configurations.extend(
        FrontierConfiguration(
            f"F{count}", source_layer, branches[:count], branches
        )
        for count in range(1, len(branches) + 1)
    )
    configurations.extend(
        FrontierConfiguration(
            f"B{index}", source_layer, (branches[index - 1],), branches
        )
        for index in range(2, len(branches) + 1)
    )
    if len({configuration.frontier_id for configuration in configurations}) != len(
        configurations
    ):
        raise AssertionError("frontier IDs are not unique")
    return tuple(configurations)


def frontier_patch_cache(
    target: RealizedForwardCapture,
    replacements: Mapping[str, Tensor],
    layers: Sequence[nn.Module],
    configuration: FrontierConfiguration,
) -> dict[PatchSite, CapturedActivation]:
    """Patch the source heads and freeze exactly the unreleased later branches."""
    components = validate_replacements(target, replacements, layers)
    if {component.layer for component in components} != {configuration.source_layer}:
        raise ValueError("frontier source components must share its source layer")
    patches = total_replacement_cache(target, replacements, layers)
    for branch in configuration.frozen:
        capture = (
            target.attention_branches[branch.layer]
            if branch.kind == "attention"
            else target.mlp_branches[branch.layer]
        )
        patches[branch.patch_site] = capture
    return patches


def phase_b_expected_rows(record_counts: Mapping[str, int]) -> dict[str, int]:
    """Expand the frozen formulas independently of any observed outcome."""
    complete = int(record_counts["complete"])
    positive = int(record_counts["positive"])
    discovery = int(record_counts["discovery_positive"])
    heldout = int(record_counts["heldout_positive"])
    negative = int(record_counts["negative"])
    result = {
        "absolute_contribution_rows": complete * 6 * 2 * 2,
        "matched_random_rows": positive * 32 * 2 * 2 * 2,
        "frontier_discovery_rows": discovery * 32 * 2 * 2,
        "frontier_heldout_rows": heldout * 3 * 2 * 2 * 2,
        "frontier_negative_rows": negative * 3 * 2 * 2 * 2,
        "attention_discovery_rows": discovery * 16 * 4 * 2 * 2,
        "attention_heldout_rows": heldout * 4 * 2 * 2 * 2,
        "attention_negative_rows": negative * 4 * 2 * 2 * 2,
    }
    result["total_phase_b_effect_rows"] = sum(result.values())
    return result


def _validate_attention_state(state: AttentionTensorState) -> tuple[int, int, int, int]:
    if state.patterns.ndim != 4 or state.values.ndim != 4 or state.raw_head_output.ndim != 4:
        raise ValueError("attention state tensors have invalid ranks")
    batch, query_heads, sequence, key_sequence = state.patterns.shape
    if sequence != key_sequence or state.values.shape[0] != batch:
        raise ValueError("attention pattern and value sequence shapes differ")
    if state.values.shape[2] != sequence or state.raw_head_output.shape[:2] != (batch, sequence):
        raise ValueError("attention values and raw head outputs do not align")
    if state.raw_head_output.shape[2] != query_heads:
        raise ValueError("attention query-head count differs")
    if state.response_start + state.response_mask.shape[1] > sequence:
        raise ValueError("attention response slice exceeds sequence width")
    if state.queries is not None:
        if state.queries.shape[:3] != (batch, query_heads, sequence):
            raise ValueError("attention query tensor does not align")
        if state.keys is None or state.keys.shape[:3] != (
            batch,
            state.values.shape[1],
            sequence,
        ):
            raise ValueError("attention key tensor does not align")
    if state.attention_mask is not None and state.attention_mask.shape[0] != batch:
        raise ValueError("attention mask batch does not align")
    return batch, query_heads, sequence, int(state.values.shape[1])


def query_to_kv_head(query_head: int, query_heads: int, kv_heads: int) -> int:
    """Map a grouped-query head to its Gemma key/value head."""
    if query_heads % kv_heads or not 0 <= query_head < query_heads:
        raise ValueError("invalid grouped-query attention geometry")
    return query_head // (query_heads // kv_heads)


def align_attention_indices(
    source_condition: ConditionBatch,
    target_condition: ConditionBatch,
    prompt_index_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Extend exact prompt alignments with shared response-relative positions."""
    if source_condition.batch_size != target_condition.batch_size:
        raise ValueError("attention alignment batch sizes differ")
    if len(prompt_index_pairs) != source_condition.batch_size:
        raise ValueError("one prompt alignment is required per example")
    result = []
    for row, (source_prompt, target_prompt) in enumerate(prompt_index_pairs):
        if len(source_prompt) != len(target_prompt):
            raise ValueError("prompt alignment is unbalanced")
        offsets = [
            offset
            for offset in range(source_condition.response_width)
            if bool(source_condition.response_mask[row, offset])
        ]
        source = tuple(source_prompt) + tuple(
            source_condition.response_start + offset for offset in offsets
        )
        target = tuple(target_prompt) + tuple(
            target_condition.response_start + offset for offset in offsets
        )
        result.append((source, target))
    return tuple(result)


def pattern_patch_values_retained(
    source: AttentionTensorState,
    target: AttentionTensorState,
    heads: Sequence[int],
    index_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> Tensor:
    """Patch aligned source patterns while retaining target value states."""
    source_geometry = _validate_attention_state(source)
    target_geometry = _validate_attention_state(target)
    if source_geometry[0] != target_geometry[0] or source_geometry[1] != target_geometry[1]:
        raise ValueError("source and target attention head geometry differs")
    output = target.raw_head_output.float().clone()
    start = target.response_start
    stop = start + target.response_mask.shape[1]
    for row, (source_indices, target_indices) in enumerate(index_pairs):
        for head in heads:
            pattern = torch.zeros_like(target.patterns[row, head, start:stop].float())
            source_rows = source.patterns[
                row,
                head,
                source.response_start : source.response_start + source.response_mask.shape[1],
            ].float()
            pattern[:, list(target_indices)] = source_rows[:, list(source_indices)]
            pattern = pattern / pattern.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            kv_head = query_to_kv_head(head, target_geometry[1], target_geometry[3])
            output[row, start:stop, head] = pattern @ target.values[row, kv_head].float()
    return output[:, start:stop]


def value_patch_pattern_retained(
    source: AttentionTensorState,
    target: AttentionTensorState,
    heads: Sequence[int],
    index_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    source_mask: Tensor | None = None,
    target_mask: Tensor | None = None,
) -> Tensor:
    """Patch aligned source V states while retaining target QK patterns."""
    source_geometry = _validate_attention_state(source)
    target_geometry = _validate_attention_state(target)
    if source_geometry[0] != target_geometry[0] or source_geometry[1] != target_geometry[1]:
        raise ValueError("source and target attention head geometry differs")
    output = target.raw_head_output.float().clone()
    start = target.response_start
    stop = start + target.response_mask.shape[1]
    for row, (source_indices, target_indices) in enumerate(index_pairs):
        selected_source = list(source_indices)
        selected_target = list(target_indices)
        if source_mask is not None or target_mask is not None:
            if source_mask is None or target_mask is None:
                raise ValueError("source and target masks must be supplied together")
            filtered = [
                (source_index, target_index)
                for source_index, target_index in zip(selected_source, selected_target, strict=True)
                if bool(source_mask[row, source_index]) and bool(target_mask[row, target_index])
            ]
            selected_source = [value[0] for value in filtered]
            selected_target = [value[1] for value in filtered]
        for head in heads:
            kv_head = query_to_kv_head(head, target_geometry[1], target_geometry[3])
            values = target.values[row, kv_head].float().clone()
            if selected_source:
                values[selected_target] = source.values[
                    row, kv_head, selected_source
                ].float()
            pattern = target.patterns[row, head, start:stop].float()
            output[row, start:stop, head] = pattern @ values
    return output[:, start:stop]


def attention_state_from_tensors(
    patterns: Tensor,
    values: Tensor,
    raw_head_output: Tensor,
    condition: ConditionBatch,
) -> AttentionTensorState:
    """Build and validate an attention state from one eager forward capture."""
    result = AttentionTensorState(
        patterns=patterns.detach().cpu(),
        values=values.detach().cpu(),
        raw_head_output=raw_head_output.detach().cpu(),
        response_start=condition.response_start,
        response_mask=condition.response_mask.clone(),
    )
    _validate_attention_state(result)
    return result


def concept_qk_removed(
    target: AttentionTensorState,
    heads: Sequence[int],
    concept_mask: Tensor,
) -> Tensor:
    """Remove concept-span key states and recompute selected response patterns."""
    batch, query_heads, _sequence, kv_heads = _validate_attention_state(target)
    if target.queries is None or target.keys is None:
        raise ValueError("concept QK intervention requires captured post-RoPE Q and K")
    if concept_mask.shape != (batch, target.sequence_width):
        raise ValueError("concept mask does not match the target sequence")
    output = target.raw_head_output.float().clone()
    start = target.response_start
    stop = start + target.response_mask.shape[1]
    scaling = target.scaling if target.scaling is not None else target.queries.shape[-1] ** -0.5
    for row in range(batch):
        for head in heads:
            kv_head = query_to_kv_head(head, query_heads, kv_heads)
            keys = target.keys[row, kv_head].float().clone()
            keys[concept_mask[row]] = 0
            logits = target.queries[row, head, start:stop].float() @ keys.T
            logits *= float(scaling)
            if target.softcap is not None:
                logits = torch.tanh(logits / target.softcap) * target.softcap
            if target.attention_mask is not None:
                mask = target.attention_mask[row]
                while mask.ndim > 2:
                    mask = mask[0]
                logits += mask[start:stop, : target.sequence_width].float()
            pattern = torch.softmax(logits, dim=-1)
            output[row, start:stop, head] = pattern @ target.values[row, kv_head].float()
    return output[:, start:stop]


def concept_ov_removed(
    target: AttentionTensorState,
    heads: Sequence[int],
    concept_mask: Tensor,
) -> Tensor:
    """Remove concept-span V states while retaining target QK patterns."""
    batch, query_heads, _sequence, kv_heads = _validate_attention_state(target)
    if concept_mask.shape != (batch, target.sequence_width):
        raise ValueError("concept mask does not match the target sequence")
    output = target.raw_head_output.float().clone()
    start = target.response_start
    stop = start + target.response_mask.shape[1]
    for row in range(batch):
        for head in heads:
            kv_head = query_to_kv_head(head, query_heads, kv_heads)
            values = target.values[row, kv_head].float().clone()
            values[concept_mask[row]] = 0
            output[row, start:stop, head] = (
                target.patterns[row, head, start:stop].float() @ values
            )
    return output[:, start:stop]


def concept_operation_bidirectional(
    source: AttentionTensorState,
    target: AttentionTensorState,
    heads: Sequence[int],
    source_concept_mask: Tensor,
    target_concept_mask: Tensor,
    *,
    operation: str,
) -> Tensor:
    """Remove a target concept span or install the reverse source-span effect.

    The normal condition has no monitoring-prefix concept positions.  Therefore
    rescue directly removes the span from a triggered target, while induction
    adds the aligned reverse of that removal effect to the normal raw head.  The
    construction gives both directions the identical response-query operator.
    """
    if operation == "concept_span_qk":
        remove = concept_qk_removed
    elif operation == "concept_span_ov":
        remove = concept_ov_removed
    else:
        raise ValueError("unknown concept-span operation")
    if target_concept_mask.any():
        return remove(target, heads, target_concept_mask)
    if not source_concept_mask.any():
        raise ValueError("one side of a concept-span intervention must contain the span")
    source_removed = remove(source, heads, source_concept_mask)
    source_natural = source.raw_head_output[
        :,
        source.response_start : source.response_start + source.response_mask.shape[1],
    ].float()
    target_natural = target.raw_head_output[
        :,
        target.response_start : target.response_start + target.response_mask.shape[1],
    ].float()
    result = target_natural.clone()
    for head in heads:
        result[:, :, head] += source_natural[:, :, head] - source_removed[:, :, head]
    return result


def attention_operation_replacements(
    source: AttentionTensorState,
    target: AttentionTensorState,
    component_ids: Sequence[str],
    index_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    operation: str,
    source_concept_mask: Tensor,
    target_concept_mask: Tensor,
) -> dict[str, Tensor]:
    """Compute raw-head replacements for one frozen attention operation site."""
    components = tuple(MechanismComponent.parse(value) for value in component_ids)
    if not components or any(component.kind != "head" for component in components):
        raise ValueError("attention operation site must contain query heads")
    if len({component.layer for component in components}) != 1:
        raise ValueError("attention operation site must be within one layer")
    heads = tuple(int(component.head) for component in components)
    if operation == "pattern_patch_values_retained":
        changed = pattern_patch_values_retained(source, target, heads, index_pairs)
    elif operation == "value_patch_pattern_retained":
        changed = value_patch_pattern_retained(source, target, heads, index_pairs)
    elif operation in {"concept_span_qk", "concept_span_ov"}:
        changed = concept_operation_bidirectional(
            source,
            target,
            heads,
            source_concept_mask,
            target_concept_mask,
            operation=operation,
        )
    else:
        raise ValueError("unknown attention operation")
    return {
        component.component_id: changed[:, :, int(component.head)].clone()
        for component in components
    }


def attention_sites(
    selected_layer_groups: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Expand the frozen twelve individual heads plus four layer groups."""
    individual = sorted(
        component_id
        for component_ids in selected_layer_groups.values()
        for component_id in component_ids
    )
    sites = [(f"individual.{component_id}", (component_id,)) for component_id in individual]
    sites.extend(
        (f"group.selected_heads.{layer_id}", tuple(component_ids))
        for layer_id, component_ids in sorted(selected_layer_groups.items())
    )
    if len(sites) != 16 or len({site_id for site_id, _members in sites}) != 16:
        raise ValueError("frozen attention operation requires exactly 16 sites")
    return tuple(sites)


class _AttentionMonitorReached(RuntimeError):
    pass


class AttentionStateCaptureRunner:
    """Capture eager patterns, post-RoPE Q/K, V, and raw query-head outputs."""

    def __init__(
        self, runner: PairedInterventionRunner, *, monitor_layer: int = 12
    ) -> None:
        self.runner = runner
        self.monitor_layer = monitor_layer

    def run(
        self, condition: ConditionBatch, layer_indices: Sequence[int]
    ) -> dict[int, AttentionTensorState]:
        layer_indices = tuple(layer_indices)
        if not layer_indices or len(set(layer_indices)) != len(layer_indices):
            raise ValueError("attention capture layers must be nonempty and unique")
        if any(not 0 <= layer <= self.monitor_layer for layer in layer_indices):
            raise ValueError("attention capture layer is outside the live prefix")
        partial: dict[int, dict[str, Any]] = {layer: {} for layer in layer_indices}
        handles = []
        try:
            for layer_index in layer_indices:
                attention = self.runner.layers[layer_index].self_attn

                def attention_pre_hook(
                    module: nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    layer_index: int = layer_index,
                ) -> None:
                    from transformers.models.gemma2.modeling_gemma2 import (
                        apply_rotary_pos_emb,
                    )

                    hidden = kwargs.get("hidden_states")
                    if hidden is None:
                        hidden = self.runner._first_tensor(args)
                    position_embeddings = kwargs.get("position_embeddings")
                    if position_embeddings is None and len(args) > 1:
                        position_embeddings = args[1]
                    if position_embeddings is None:
                        raise RuntimeError("Gemma attention position embeddings are missing")
                    cos, sin = position_embeddings
                    input_shape = hidden.shape[:-1]
                    hidden_shape = (*input_shape, -1, int(module.head_dim))
                    with torch.inference_mode():
                        query = module.q_proj(hidden).view(hidden_shape).transpose(1, 2)
                        key = module.k_proj(hidden).view(hidden_shape).transpose(1, 2)
                        query, key = apply_rotary_pos_emb(query, key, cos, sin)
                    partial[layer_index]["queries"] = query.detach().cpu()
                    partial[layer_index]["keys"] = key.detach().cpu()
                    mask = kwargs.get("attention_mask")
                    partial[layer_index]["attention_mask"] = (
                        None if mask is None else mask.detach().cpu()
                    )

                handles.append(
                    attention.register_forward_pre_hook(
                        attention_pre_hook, with_kwargs=True
                    )
                )

                def value_hook(
                    module: nn.Module,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    layer_index: int = layer_index,
                ) -> None:
                    tensor = self.runner._first_tensor(output)
                    kv_heads = int(
                        getattr(module, "out_features", tensor.shape[-1])
                        // self.runner._head_dim(attention)
                    )
                    partial[layer_index]["values"] = (
                        tensor.reshape(
                            tensor.shape[0], tensor.shape[1], kv_heads, -1
                        )
                        .transpose(1, 2)
                        .detach()
                        .cpu()
                    )

                handles.append(attention.v_proj.register_forward_hook(value_hook))

                def raw_hook(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    layer_index: int = layer_index,
                    attention: nn.Module = attention,
                ) -> None:
                    tensor = self.runner._first_tensor(args)
                    heads = self.runner._num_attention_heads(attention)
                    partial[layer_index]["raw"] = (
                        tensor.reshape(*tensor.shape[:-1], heads, -1).detach().cpu()
                    )

                handles.append(attention.o_proj.register_forward_pre_hook(raw_hook))

                def attention_hook(
                    module: nn.Module,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    layer_index: int = layer_index,
                ) -> None:
                    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                        raise RuntimeError("eager attention patterns were not returned")
                    partial[layer_index]["patterns"] = output[1].detach().cpu()
                    partial[layer_index]["scaling"] = float(
                        getattr(module, "scaling", module.head_dim**-0.5)
                    )
                    softcap = getattr(module, "attn_logit_softcapping", None)
                    partial[layer_index]["softcap"] = (
                        None if softcap is None else float(softcap)
                    )

                handles.append(attention.register_forward_hook(attention_hook))

            def terminal_hook(
                _module: nn.Module, _args: tuple[Any, ...], _output: Any
            ) -> None:
                raise _AttentionMonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(
                    terminal_hook
                )
            )
            try:
                with torch.inference_mode():
                    self.runner.model(
                        input_ids=condition.input_ids.to(self.runner.device),
                        attention_mask=condition.attention_mask.to(self.runner.device),
                        position_ids=condition.position_ids.to(self.runner.device),
                        use_cache=False,
                        output_attentions=True,
                        output_hidden_states=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                raise RuntimeError("model completed without reaching attention monitor")
            except _AttentionMonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        result = {}
        required = {
            "patterns",
            "values",
            "raw",
            "queries",
            "keys",
            "attention_mask",
            "scaling",
            "softcap",
        }
        for layer_index, values in partial.items():
            if set(values) != required:
                raise RuntimeError(
                    f"attention capture at layer {layer_index} is incomplete: {set(values)}"
                )
            state = AttentionTensorState(
                patterns=values["patterns"],
                values=values["values"],
                raw_head_output=values["raw"],
                response_start=condition.response_start,
                response_mask=condition.response_mask.clone(),
                queries=values["queries"],
                keys=values["keys"],
                attention_mask=values["attention_mask"],
                scaling=values["scaling"],
                softcap=values["softcap"],
            )
            _validate_attention_state(state)
            result[layer_index] = state
        if self.runner.registered_hook_count() != 0:
            raise RuntimeError("attention capture leaked hooks")
        return result
