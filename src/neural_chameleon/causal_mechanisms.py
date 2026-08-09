"""Successor-study writer capture, accounting, and path interventions.

The coordinates in this module follow the frozen Days 31--35 contract:
selected attention heads are captured at the input to ``self_attn.o_proj``;
the monitor reads the output of decoder block 12; and direct-path effects
freeze later normalized additive writes rather than treating an observed
pairwise activation difference as a causal prediction.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .interventions import (
    ActivationKind,
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
)


COMPONENT_ID = re.compile(r"^layer_(\d{2})\.(head_(\d{2})|mlp)$")


class _MonitorReached(RuntimeError):
    pass


@dataclass(frozen=True, order=True)
class MechanismComponent:
    """One frozen attention-head or whole-MLP intervention site."""

    layer: int
    kind: str
    head: int | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("component layer must be non-negative")
        if self.kind == "head":
            if self.head is None or self.head < 0:
                raise ValueError("head components require a non-negative head")
        elif self.kind == "mlp":
            if self.head is not None:
                raise ValueError("MLP components cannot specify a head")
        else:
            raise ValueError(f"unknown component kind: {self.kind}")

    @classmethod
    def parse(cls, component_id: str) -> "MechanismComponent":
        match = COMPONENT_ID.fullmatch(component_id)
        if match is None:
            raise ValueError(f"invalid component ID: {component_id}")
        layer = int(match.group(1))
        if match.group(2) == "mlp":
            return cls(layer=layer, kind="mlp")
        return cls(layer=layer, kind="head", head=int(match.group(3)))

    @property
    def component_id(self) -> str:
        suffix = "mlp" if self.kind == "mlp" else f"head_{self.head:02d}"
        return f"layer_{self.layer:02d}.{suffix}"

    def patch_site(self) -> PatchSite:
        if self.kind == "mlp":
            return PatchSite(ActivationKind.MLP_OUT, self.layer)
        return PatchSite(ActivationKind.HEAD_OUTPUT, self.layer, head=self.head)


@dataclass(frozen=True)
class RealizedForwardCapture:
    """Response-aligned tensors sufficient for exact block-12 accounting."""

    condition: str
    response_ids: Tensor
    response_mask: Tensor
    initial_residual: CapturedActivation
    raw_attention: Mapping[int, CapturedActivation]
    projected_attention: Mapping[int, CapturedActivation]
    attention_branches: Mapping[int, CapturedActivation]
    mlp_branches: Mapping[int, CapturedActivation]
    monitor_residual: CapturedActivation

    @property
    def batch_size(self) -> int:
        return int(self.response_ids.shape[0])

    @property
    def response_width(self) -> int:
        return int(self.response_ids.shape[1])

    def head_capture(
        self, layer: int, head: int, *, num_heads: int, head_dim: int
    ) -> CapturedActivation:
        raw = self.raw_attention[layer]
        if raw.values.shape[-1] != num_heads * head_dim:
            raise ValueError("raw attention width does not match head geometry")
        if not 0 <= head < num_heads:
            raise ValueError("head index is outside the attention geometry")
        values = raw.values.reshape(
            raw.values.shape[0], raw.values.shape[1], num_heads, head_dim
        )[:, :, head, :]
        return CapturedActivation(
            values=values.clone(),
            response_ids=raw.response_ids.clone(),
            response_mask=raw.response_mask.clone(),
        )


@dataclass(frozen=True)
class AttentionAllocation:
    """One normalized attention branch allocated over its raw query heads."""

    head_values: Tensor
    bias_value: Tensor
    raw_projection_max_abs_error: float
    normalized_branch_max_abs_error: float
    projection_numerical_residual_max_abs: float
    normalization_numerical_residual_max_abs: float
    rmsnorm_parameterization: str


@dataclass(frozen=True)
class RealizedForwardAudit:
    """Numerical closure diagnostics for one captured forward pass."""

    hidden_max_abs_error: float
    attention_allocation_max_abs_error: float
    attention_raw_projection_max_abs_error: float
    attention_projection_numerical_residual_max_abs: float
    attention_normalization_numerical_residual_max_abs: float
    probe_margin_max_abs_error: float
    sequence_score_max_abs_error: float
    per_layer_attention_allocation_max_abs_error: Mapping[int, float]
    per_layer_projection_numerical_residual_max_abs: Mapping[int, float]
    per_layer_normalization_numerical_residual_max_abs: Mapping[int, float]
    per_probe_margin_max_abs_error: tuple[float, ...]
    per_probe_sequence_score_max_abs_error: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_max_abs_error": self.hidden_max_abs_error,
            "attention_allocation_max_abs_error": self.attention_allocation_max_abs_error,
            "attention_raw_projection_max_abs_error": self.attention_raw_projection_max_abs_error,
            "attention_projection_numerical_residual_max_abs": self.attention_projection_numerical_residual_max_abs,
            "attention_normalization_numerical_residual_max_abs": self.attention_normalization_numerical_residual_max_abs,
            "probe_margin_max_abs_error": self.probe_margin_max_abs_error,
            "sequence_score_max_abs_error": self.sequence_score_max_abs_error,
            "per_layer_attention_allocation_max_abs_error": {
                str(layer): value
                for layer, value in self.per_layer_attention_allocation_max_abs_error.items()
            },
            "per_layer_projection_numerical_residual_max_abs": {
                str(layer): value
                for layer, value in self.per_layer_projection_numerical_residual_max_abs.items()
            },
            "per_layer_normalization_numerical_residual_max_abs": {
                str(layer): value
                for layer, value in self.per_layer_normalization_numerical_residual_max_abs.items()
            },
            "per_probe_margin_max_abs_error": list(self.per_probe_margin_max_abs_error),
            "per_probe_sequence_score_max_abs_error": list(
                self.per_probe_sequence_score_max_abs_error
            ),
        }


@dataclass(frozen=True)
class ComponentEffectResult:
    """Monitor states under total and frozen-write direct-path interventions."""

    component_ids: tuple[str, ...]
    target: CapturedActivation
    total: CapturedActivation
    direct_path: CapturedActivation

    @property
    def total_delta(self) -> Tensor:
        return self.total.values.float() - self.target.values.float()

    @property
    def direct_path_delta(self) -> Tensor:
        return self.direct_path.values.float() - self.target.values.float()

    @property
    def downstream_dependent_remainder(self) -> Tensor:
        return self.total_delta - self.direct_path_delta


def _capture_like(condition: ConditionBatch, values: Tensor) -> CapturedActivation:
    return CapturedActivation(
        values=values.detach().cpu().clone(),
        response_ids=condition.response_ids.clone(),
        response_mask=condition.response_mask.clone(),
    )


def _response_values(tensor: Tensor, condition: ConditionBatch) -> Tensor:
    start = condition.response_start
    stop = start + condition.response_width
    if tensor.ndim != 3 or tensor.shape[1] < stop:
        raise RuntimeError(f"unexpected activation shape: {tuple(tensor.shape)}")
    return tensor[:, start:stop, :]


def _masked_max_abs(values: Tensor, mask: Tensor) -> float:
    if values.shape[:2] != mask.shape:
        raise ValueError("value and response-mask shapes do not align")
    selected = values.float().abs()[mask.bool()]
    return float(selected.max().item()) if selected.numel() else 0.0


class RealizedForwardRunner:
    """Capture every realized additive branch through the frozen monitor."""

    def __init__(
        self, runner: PairedInterventionRunner, *, monitor_layer: int = 12
    ) -> None:
        if not 0 <= monitor_layer < len(runner.layers):
            raise ValueError("monitor layer is outside the model")
        self.runner = runner
        self.monitor_layer = monitor_layer

    def run(self, condition: ConditionBatch) -> RealizedForwardCapture:
        initial: CapturedActivation | None = None
        raw_attention: dict[int, CapturedActivation] = {}
        projected_attention: dict[int, CapturedActivation] = {}
        attention: dict[int, CapturedActivation] = {}
        mlp: dict[int, CapturedActivation] = {}
        monitor: CapturedActivation | None = None
        handles: list[Any] = []

        def capture_input(_module: nn.Module, args: tuple[Any, ...]) -> None:
            nonlocal initial
            values = _response_values(self.runner._first_tensor(args), condition)
            initial = _capture_like(condition, values)

        handles.append(self.runner.layers[0].register_forward_pre_hook(capture_input))

        for layer_index in range(self.monitor_layer + 1):
            layer = self.runner.layers[layer_index]

            def capture_raw(
                _module: nn.Module,
                args: tuple[Any, ...],
                layer_index: int = layer_index,
            ) -> None:
                values = _response_values(self.runner._first_tensor(args), condition)
                raw_attention[layer_index] = _capture_like(condition, values)

            def capture_attention(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                layer_index: int = layer_index,
            ) -> None:
                values = _response_values(self.runner._first_tensor(output), condition)
                attention[layer_index] = _capture_like(condition, values)

            def capture_projected_attention(
                _module: nn.Module,
                args: tuple[Any, ...],
                layer_index: int = layer_index,
            ) -> None:
                values = _response_values(self.runner._first_tensor(args), condition)
                projected_attention[layer_index] = _capture_like(condition, values)

            def capture_mlp(
                _module: nn.Module,
                _args: tuple[Any, ...],
                output: Any,
                layer_index: int = layer_index,
            ) -> None:
                values = _response_values(self.runner._first_tensor(output), condition)
                mlp[layer_index] = _capture_like(condition, values)

            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(capture_raw)
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    capture_projected_attention
                )
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(capture_attention)
            )
            handles.append(
                layer.post_feedforward_layernorm.register_forward_hook(capture_mlp)
            )

        def capture_monitor(
            _module: nn.Module, _args: tuple[Any, ...], output: Any
        ) -> None:
            nonlocal monitor
            values = _response_values(self.runner._first_tensor(output), condition)
            monitor = _capture_like(condition, values)
            raise _MonitorReached()

        handles.append(
            self.runner.layers[self.monitor_layer].register_forward_hook(
                capture_monitor
            )
        )
        try:
            try:
                with torch.inference_mode():
                    self.runner.model(
                        input_ids=condition.input_ids.to(self.runner.device),
                        attention_mask=condition.attention_mask.to(self.runner.device),
                        position_ids=condition.position_ids.to(self.runner.device),
                        use_cache=False,
                        output_hidden_states=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                raise RuntimeError("model completed without reaching the monitor")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()

        expected = set(range(self.monitor_layer + 1))
        if initial is None or monitor is None:
            raise RuntimeError("initial or monitor residual was not captured")
        if (
            set(raw_attention) != expected
            or set(projected_attention) != expected
            or set(attention) != expected
            or set(mlp) != expected
        ):
            raise RuntimeError("one or more realized branches were not captured")
        return RealizedForwardCapture(
            condition=condition.name,
            response_ids=condition.response_ids.clone(),
            response_mask=condition.response_mask.clone(),
            initial_residual=initial,
            raw_attention=raw_attention,
            projected_attention=projected_attention,
            attention_branches=attention,
            mlp_branches=mlp,
            monitor_residual=monitor,
        )


def reconstruct_monitor_residual(capture: RealizedForwardCapture) -> Tensor:
    """Replay the actual residual additions in their original tensor dtype."""
    hidden = capture.initial_residual.values.clone()
    layers = sorted(capture.attention_branches)
    if layers != list(range(max(layers) + 1)):
        raise ValueError("captured accounting layers are not contiguous from zero")
    for layer in layers:
        hidden = hidden + capture.attention_branches[layer].values.to(hidden.dtype)
        hidden = hidden + capture.mlp_branches[layer].values.to(hidden.dtype)
    return hidden


def _rmsnorm_scale(
    module: nn.Module, raw: Tensor, observed: Tensor
) -> tuple[Tensor, str, float]:
    eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
    inverse_rms = torch.rsqrt(raw.float().square().mean(dim=-1, keepdim=True) + eps)
    weight = getattr(module, "weight", None)
    if weight is None:
        candidates = ((torch.ones(raw.shape[-1], device=raw.device), "unit"),)
    else:
        parameter = weight.detach().to(device=raw.device, dtype=torch.float32)
        candidates = ((parameter, "weight"), (1.0 + parameter, "one_plus_weight"))
    best_scale: Tensor | None = None
    best_name = ""
    best_error = float("inf")
    for scale, name in candidates:
        prediction = raw.float() * inverse_rms * scale
        error = float((prediction - observed.float()).abs().max().item())
        if error < best_error:
            best_scale, best_name, best_error = scale, name, error
    if best_scale is None:
        raise RuntimeError("RMSNorm parameterization could not be resolved")
    return best_scale, best_name, best_error


def allocate_normalized_attention_heads(
    layer: nn.Module,
    raw_joint_heads: CapturedActivation,
    projected_branch: CapturedActivation,
    normalized_branch: CapturedActivation,
) -> AttentionAllocation:
    """Allocate a realized Gemma RMS-normalized attention branch across heads."""
    for other in (projected_branch, normalized_branch):
        if not torch.equal(raw_joint_heads.response_ids, other.response_ids):
            raise ValueError("attention allocation response IDs differ")
        if not torch.equal(raw_joint_heads.response_mask, other.response_mask):
            raise ValueError("attention allocation response masks differ")
    attention = layer.self_attn
    num_heads = PairedInterventionRunner._num_attention_heads(attention)
    head_dim = PairedInterventionRunner._head_dim(attention)
    device = attention.o_proj.weight.device
    raw = raw_joint_heads.values.to(device=device, dtype=torch.float32)
    if raw.shape[-1] != num_heads * head_dim:
        raise ValueError("raw joint-head width does not match attention geometry")
    projected = []
    for head in range(num_heads):
        left, right = head * head_dim, (head + 1) * head_dim
        projected.append(
            F.linear(
                raw[..., left:right],
                attention.o_proj.weight[:, left:right].float(),
            )
        )
    head_projection = torch.stack(projected, dim=2)
    bias = attention.o_proj.bias
    bias_projection = (
        torch.zeros_like(head_projection[:, :, 0, :])
        if bias is None
        else bias.to(device=device, dtype=torch.float32).expand_as(
            head_projection[:, :, 0, :]
        )
    )
    theoretical_projection = head_projection.sum(dim=2) + bias_projection
    realized_projection = projected_branch.values.to(device=device, dtype=torch.float32)
    projection_residual = realized_projection - theoretical_projection
    head_projection = head_projection + projection_residual.unsqueeze(2) / num_heads
    raw_projection = head_projection.sum(dim=2) + bias_projection
    raw_error = _masked_max_abs(
        realized_projection - raw_projection, raw_joint_heads.response_mask
    )
    projection_residual_max = _masked_max_abs(
        projection_residual, raw_joint_heads.response_mask
    )
    observed = normalized_branch.values.to(device=device, dtype=torch.float32)
    scale, parameterization, _ = _rmsnorm_scale(
        layer.post_attention_layernorm, realized_projection, observed
    )
    eps = float(
        getattr(
            layer.post_attention_layernorm,
            "variance_epsilon",
            getattr(layer.post_attention_layernorm, "eps", 1e-6),
        )
    )
    inverse_rms = torch.rsqrt(
        realized_projection.square().mean(dim=-1, keepdim=True) + eps
    )
    head_allocation = head_projection * inverse_rms.unsqueeze(2) * scale
    bias_allocation = bias_projection * inverse_rms * scale
    theoretical_normalized = head_allocation.sum(dim=2) + bias_allocation
    normalization_residual = observed - theoretical_normalized
    normalization_residual_max = _masked_max_abs(
        normalization_residual, normalized_branch.response_mask
    )
    head_allocation = head_allocation + normalization_residual.unsqueeze(2) / num_heads
    reconstructed = head_allocation.sum(dim=2) + bias_allocation
    closure = _masked_max_abs(
        reconstructed - observed.float(), normalized_branch.response_mask
    )
    return AttentionAllocation(
        head_values=head_allocation.detach().cpu(),
        bias_value=bias_allocation.detach().cpu(),
        raw_projection_max_abs_error=raw_error,
        normalized_branch_max_abs_error=closure,
        projection_numerical_residual_max_abs=projection_residual_max,
        normalization_numerical_residual_max_abs=normalization_residual_max,
        rmsnorm_parameterization=parameterization,
    )


def probe_token_margins(
    activation: CapturedActivation, probes: Sequence[LinearProbe]
) -> Tensor:
    """Return float32 margins with shape [probe,batch,response]."""
    if not probes:
        raise ValueError("at least one probe is required")
    values = activation.values.float()
    margins = []
    for probe in probes:
        weight = probe.weight.float()
        bias = probe.bias.float()
        if (
            weight.ndim != 2
            or weight.shape[0] != 1
            or weight.shape[1] != values.shape[-1]
        ):
            raise ValueError("probe weight shape does not match monitor activation")
        margins.append(torch.matmul(values, weight.T).squeeze(-1) + bias.reshape(()))
    return torch.stack(margins)


def probe_sequence_scores(margins: Tensor, response_mask: Tensor) -> Tensor:
    """Apply sigmoid tokenwise, then response-mask averaging."""
    if margins.ndim != 3 or margins.shape[1:] != response_mask.shape:
        raise ValueError("margin and response-mask shapes do not align")
    mask = response_mask.to(margins.device).unsqueeze(0)
    probabilities = torch.sigmoid(margins.float())
    return (probabilities * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)


def audit_realized_forward(
    capture: RealizedForwardCapture,
    layers: Sequence[nn.Module],
    probes: Sequence[LinearProbe],
) -> RealizedForwardAudit:
    """Audit residual, head-allocation, margin, and score closure."""
    if len(layers) < len(capture.attention_branches):
        raise ValueError("insufficient model layers for the capture")
    reconstructed = reconstruct_monitor_residual(capture)
    hidden_error = _masked_max_abs(
        reconstructed.float() - capture.monitor_residual.values.float(),
        capture.response_mask,
    )
    allocation_errors: dict[int, float] = {}
    raw_errors = []
    projection_residuals: dict[int, float] = {}
    normalization_residuals: dict[int, float] = {}
    for layer_index in sorted(capture.raw_attention):
        allocation = allocate_normalized_attention_heads(
            layers[layer_index],
            capture.raw_attention[layer_index],
            capture.projected_attention[layer_index],
            capture.attention_branches[layer_index],
        )
        allocation_errors[layer_index] = allocation.normalized_branch_max_abs_error
        raw_errors.append(allocation.raw_projection_max_abs_error)
        projection_residuals[layer_index] = (
            allocation.projection_numerical_residual_max_abs
        )
        normalization_residuals[layer_index] = (
            allocation.normalization_numerical_residual_max_abs
        )

    reconstructed_capture = CapturedActivation(
        values=reconstructed,
        response_ids=capture.response_ids,
        response_mask=capture.response_mask,
    )
    measured_margin = probe_token_margins(capture.monitor_residual, probes)
    reconstructed_margin = probe_token_margins(reconstructed_capture, probes)
    margin_delta = reconstructed_margin - measured_margin
    per_probe_margin = tuple(
        _masked_max_abs(margin_delta[index].unsqueeze(-1), capture.response_mask)
        for index in range(len(probes))
    )
    measured_score = probe_sequence_scores(measured_margin, capture.response_mask)
    reconstructed_score = probe_sequence_scores(
        reconstructed_margin, capture.response_mask
    )
    per_probe_score = tuple(
        float((reconstructed_score[index] - measured_score[index]).abs().max().item())
        for index in range(len(probes))
    )
    return RealizedForwardAudit(
        hidden_max_abs_error=hidden_error,
        attention_allocation_max_abs_error=max(allocation_errors.values(), default=0.0),
        attention_raw_projection_max_abs_error=max(raw_errors, default=0.0),
        attention_projection_numerical_residual_max_abs=max(
            projection_residuals.values(), default=0.0
        ),
        attention_normalization_numerical_residual_max_abs=max(
            normalization_residuals.values(), default=0.0
        ),
        probe_margin_max_abs_error=max(per_probe_margin, default=0.0),
        sequence_score_max_abs_error=max(per_probe_score, default=0.0),
        per_layer_attention_allocation_max_abs_error=allocation_errors,
        per_layer_projection_numerical_residual_max_abs=projection_residuals,
        per_layer_normalization_numerical_residual_max_abs=normalization_residuals,
        per_probe_margin_max_abs_error=per_probe_margin,
        per_probe_sequence_score_max_abs_error=per_probe_score,
    )


def component_capture(
    capture: RealizedForwardCapture,
    component: MechanismComponent,
    layers: Sequence[nn.Module],
) -> CapturedActivation:
    if component.kind == "mlp":
        return capture.mlp_branches[component.layer]
    attention = layers[component.layer].self_attn
    return capture.head_capture(
        component.layer,
        int(component.head),
        num_heads=PairedInterventionRunner._num_attention_heads(attention),
        head_dim=PairedInterventionRunner._head_dim(attention),
    )


def _counterfactual_attention_branch(
    layer: nn.Module,
    target: RealizedForwardCapture,
    source: RealizedForwardCapture,
    layer_index: int,
    heads: Sequence[int],
) -> CapturedActivation:
    target_raw = target.raw_attention[layer_index]
    source_raw = source.raw_attention[layer_index]
    if not torch.equal(
        target_raw.response_ids, source_raw.response_ids
    ) or not torch.equal(target_raw.response_mask, source_raw.response_mask):
        raise ValueError("source and target raw attention captures do not align")
    attention = layer.self_attn
    num_heads = PairedInterventionRunner._num_attention_heads(attention)
    head_dim = PairedInterventionRunner._head_dim(attention)
    if len(set(heads)) != len(heads) or any(
        not 0 <= head < num_heads for head in heads
    ):
        raise ValueError("counterfactual attention heads are invalid or duplicated")
    device = attention.o_proj.weight.device
    dtype = attention.o_proj.weight.dtype
    target_joint = (
        target_raw.values.to(device=device, dtype=dtype)
        .clone()
        .reshape(
            target_raw.values.shape[0], target_raw.values.shape[1], num_heads, head_dim
        )
    )
    joint = target_joint.clone()
    source_joint = source_raw.values.to(device=device, dtype=dtype).reshape_as(joint)
    mask = target_raw.response_mask.to(device).unsqueeze(-1)
    for head in heads:
        joint[:, :, head, :] = torch.where(
            mask, source_joint[:, :, head, :], joint[:, :, head, :]
        )
    with torch.inference_mode():
        target_recomputed = layer.post_attention_layernorm(
            attention.o_proj(target_joint.reshape(*target_joint.shape[:2], -1))
        )
        counterfactual_recomputed = layer.post_attention_layernorm(
            attention.o_proj(joint.reshape(*joint.shape[:2], -1))
        )
        cached_target = target.attention_branches[layer_index].values.to(
            device=device, dtype=torch.float32
        )
        normalized = cached_target + (
            counterfactual_recomputed.float() - target_recomputed.float()
        )
    return CapturedActivation(
        values=normalized.detach().cpu(),
        response_ids=target_raw.response_ids.clone(),
        response_mask=target_raw.response_mask.clone(),
    )


def total_patch_cache(
    source: RealizedForwardCapture,
    components: Sequence[MechanismComponent],
    layers: Sequence[nn.Module],
) -> dict[PatchSite, CapturedActivation]:
    if not components:
        raise ValueError("at least one component is required")
    result: dict[PatchSite, CapturedActivation] = {}
    for component in components:
        site = component.patch_site()
        if site in result:
            raise ValueError("component set contains duplicate sites")
        result[site] = component_capture(source, component, layers)
    return result


def direct_path_patch_cache(
    target: RealizedForwardCapture,
    source: RealizedForwardCapture,
    components: Sequence[MechanismComponent],
    layers: Sequence[nn.Module],
    *,
    monitor_layer: int = 12,
) -> dict[PatchSite, CapturedActivation]:
    """Build the frozen-additive-write direct-path patch cache."""
    if not components:
        raise ValueError("at least one component is required")
    if len({component.component_id for component in components}) != len(components):
        raise ValueError("component set contains duplicates")
    if any(component.layer > monitor_layer for component in components):
        raise ValueError("component occurs after the monitor")
    earliest = min(component.layer for component in components)
    heads_by_layer: dict[int, list[int]] = {}
    mlp_layers = set()
    for component in components:
        if component.kind == "head":
            heads_by_layer.setdefault(component.layer, []).append(int(component.head))
        else:
            mlp_layers.add(component.layer)

    patches: dict[PatchSite, CapturedActivation] = {}
    for layer_index in range(earliest, monitor_layer + 1):
        selected_heads = heads_by_layer.get(layer_index, [])
        attention_is_downstream = layer_index > earliest or bool(selected_heads)
        if attention_is_downstream:
            site = PatchSite(ActivationKind.ATTN_OUT, layer_index)
            patches[site] = (
                _counterfactual_attention_branch(
                    layers[layer_index], target, source, layer_index, selected_heads
                )
                if selected_heads
                else target.attention_branches[layer_index]
            )
        mlp_is_downstream = (
            layer_index > earliest or bool(selected_heads) or layer_index in mlp_layers
        )
        if mlp_is_downstream:
            site = PatchSite(ActivationKind.MLP_OUT, layer_index)
            patches[site] = (
                source.mlp_branches[layer_index]
                if layer_index in mlp_layers
                else target.mlp_branches[layer_index]
            )
    return patches


class ComponentEffectRunner:
    """Evaluate total and direct-path effects for one component group."""

    def __init__(
        self, runner: PairedInterventionRunner, *, monitor_layer: int = 12
    ) -> None:
        self.runner = runner
        self.monitor_layer = monitor_layer
        self.monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, monitor_layer)

    def run(
        self,
        target_condition: ConditionBatch,
        target: RealizedForwardCapture,
        source: RealizedForwardCapture,
        component_ids: Sequence[str],
    ) -> ComponentEffectResult:
        components = tuple(MechanismComponent.parse(value) for value in component_ids)
        if not components:
            raise ValueError("component group cannot be empty")
        total = self.runner.run(
            target_condition,
            capture_sites=(self.monitor_site,),
            patch_cache=total_patch_cache(source, components, self.runner.layers),
        ).captures[self.monitor_site]
        direct = self.runner.run(
            target_condition,
            capture_sites=(self.monitor_site,),
            patch_cache=direct_path_patch_cache(
                target,
                source,
                components,
                self.runner.layers,
                monitor_layer=self.monitor_layer,
            ),
        ).captures[self.monitor_site]
        return ComponentEffectResult(
            component_ids=tuple(component.component_id for component in components),
            target=target.monitor_residual,
            total=total,
            direct_path=direct,
        )


def writer_delta(
    triggered: RealizedForwardCapture,
    normal: RealizedForwardCapture,
    head_ids: Sequence[str],
    layers: Sequence[nn.Module],
) -> dict[str, Tensor]:
    """Return ordered response-token pre-``W_O`` trigger deltas."""
    if not torch.equal(triggered.response_ids, normal.response_ids) or not torch.equal(
        triggered.response_mask, normal.response_mask
    ):
        raise ValueError("writer endpoints do not align")
    deltas: dict[str, Tensor] = {}
    for head_id in head_ids:
        component = MechanismComponent.parse(head_id)
        if component.kind != "head":
            raise ValueError("writer trajectories may contain only attention heads")
        triggered_head = component_capture(triggered, component, layers)
        normal_head = component_capture(normal, component, layers)
        deltas[head_id] = triggered_head.values.float() - normal_head.values.float()
    return deltas


def fit_head_rms(
    deltas: Sequence[Mapping[str, Tensor]],
    masks: Sequence[Tensor],
    head_ids: Sequence[str],
    *,
    floor: float = 1e-6,
) -> dict[str, float]:
    """Fit one scalar discovery-only RMS normalizer per frozen head."""
    if len(deltas) != len(masks) or not deltas:
        raise ValueError("one nonempty response mask is required per delta batch")
    result = {}
    for head_id in head_ids:
        numerator = 0.0
        denominator = 0
        for delta, mask in zip(deltas, masks, strict=True):
            values = delta[head_id].float()
            if values.shape[:2] != mask.shape:
                raise ValueError("writer delta and response mask do not align")
            expanded = mask.bool().unsqueeze(-1).expand_as(values)
            numerator += float(values.square()[expanded].sum().item())
            denominator += int(expanded.sum().item())
        result[head_id] = max((numerator / max(denominator, 1)) ** 0.5, floor)
    return result


def trajectory_pair_metrics(
    chameleon: Mapping[str, Tensor],
    precursor: Mapping[str, Tensor],
    response_mask: Tensor,
    head_ids: Sequence[str],
    head_rms: Mapping[str, float],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-example aligned amplitude, RMS ratio, and cosine."""
    if not head_ids:
        raise ValueError("trajectory metric requires at least one head")
    chameleon_parts = []
    precursor_parts = []
    for head_id in head_ids:
        first = chameleon[head_id].float() / float(head_rms[head_id])
        second = precursor[head_id].float() / float(head_rms[head_id])
        if first.shape != second.shape or first.shape[:2] != response_mask.shape:
            raise ValueError("paired writer trajectories do not align")
        mask = response_mask.bool().unsqueeze(-1)
        chameleon_parts.append((first * mask).flatten(start_dim=1))
        precursor_parts.append((second * mask).flatten(start_dim=1))
    first = torch.cat(chameleon_parts, dim=1)
    second = torch.cat(precursor_parts, dim=1)
    dot = (first * second).sum(dim=1)
    first_energy = first.square().sum(dim=1).clamp(min=1e-12)
    second_energy = second.square().sum(dim=1)
    aligned = dot / first_energy
    magnitude = torch.sqrt(second_energy / first_energy)
    cosine = dot / torch.sqrt(first_energy * second_energy.clamp(min=1e-12))
    return aligned, magnitude, cosine
