"""Joint attention algebra and downstream geometry for the selected K12 state."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor, nn

from .causal_mechanisms import MechanismComponent
from .interventions import (
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
)
from .individual_components import repeat_condition
from .post_gate1_interventions import AttentionTensorState, query_to_kv_head


@dataclass(frozen=True)
class JointAttentionFactorization:
    """Exact region-moment factorization for selected heads at one layer."""

    target_reconstructed: Tensor
    donor_reconstructed: Tensor
    routing_hybrid: Tensor
    content_hybrid: Tensor
    routing_shapley_delta: Tensor
    content_shapley_delta: Tensor
    region_deltas: Mapping[str, Tensor]
    target_masses: Mapping[str, Tensor]
    donor_masses: Mapping[str, Tensor]
    target_reconstruction_max_abs: float
    donor_reconstruction_max_abs: float
    shapley_closure_max_abs: float


def _masked_max_abs(values: Tensor, response_mask: Tensor) -> float:
    if values.shape[:2] != response_mask.shape:
        raise ValueError("response mask does not align with values")
    mask = response_mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    selected = values.float().abs().masked_select(mask.expand_as(values))
    return float(selected.max()) if selected.numel() else 0.0


def _region_moments(
    state: AttentionTensorState,
    source_masks: Mapping[str, Tensor],
    heads: Sequence[int],
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor]:
    """Return region mass, conditional value mean, contribution, and raw heads."""
    if not heads or len(set(heads)) != len(heads):
        raise ValueError("selected heads must be nonempty and unique")
    regions = tuple(source_masks)
    if not regions:
        raise ValueError("source-region partition must be nonempty")
    masks = torch.stack([source_masks[name].bool() for name in regions])
    if masks.ndim != 3 or masks.shape[1:] != (
        state.batch_size,
        state.sequence_width,
    ):
        raise ValueError("source-region masks do not match attention state")
    if torch.any(masks.sum(dim=0) > 1):
        raise ValueError("source-region masks overlap")
    start = state.response_start
    stop = start + state.response_mask.shape[1]
    query_heads = state.patterns.shape[1]
    kv_heads = state.values.shape[1]
    masses: dict[str, list[Tensor]] = {name: [] for name in regions}
    means: dict[str, list[Tensor]] = {name: [] for name in regions}
    contributions: dict[str, list[Tensor]] = {name: [] for name in regions}
    for head in heads:
        kv_head = query_to_kv_head(head, query_heads, kv_heads)
        patterns = state.patterns[:, head, start:stop].float()
        values = state.values[:, kv_head].float()
        for name in regions:
            mask = source_masks[name].bool()[:, None, :]
            selected = patterns * mask
            mass = selected.sum(dim=-1)
            contribution = torch.einsum("bts,bsd->btd", selected, values)
            mean = torch.where(
                mass.unsqueeze(-1) > 1e-12,
                contribution / mass.unsqueeze(-1).clamp(min=1e-12),
                torch.zeros_like(contribution),
            )
            masses[name].append(mass)
            means[name].append(mean)
            contributions[name].append(contribution)
    return (
        {name: torch.stack(values, dim=2) for name, values in masses.items()},
        {name: torch.stack(values, dim=2) for name, values in means.items()},
        {name: torch.stack(values, dim=2) for name, values in contributions.items()},
        state.raw_head_output[:, start:stop, list(heads)].float(),
    )


def factorize_joint_attention(
    target: AttentionTensorState,
    donor: AttentionTensorState,
    target_masks: Mapping[str, Tensor],
    donor_masks: Mapping[str, Tensor],
    heads: Sequence[int],
) -> JointAttentionFactorization:
    """Factor donor-target selected-head change into exact regional moments."""
    if tuple(target_masks) != tuple(donor_masks):
        raise ValueError("target and donor region orders differ")
    if target.response_mask.shape != donor.response_mask.shape or not torch.equal(
        target.response_mask, donor.response_mask
    ):
        raise ValueError("target and donor response masks differ")
    regions = tuple(target_masks)
    target_mass, target_mean, target_contrib, target_raw = _region_moments(
        target, target_masks, heads
    )
    donor_mass, donor_mean, donor_contrib, donor_raw = _region_moments(
        donor, donor_masks, heads
    )
    target_reconstructed = sum(target_contrib.values())
    donor_reconstructed = sum(donor_contrib.values())
    routing_parts = {
        name: donor_mass[name].unsqueeze(-1) * target_mean[name] for name in regions
    }
    content_parts = {
        name: target_mass[name].unsqueeze(-1) * donor_mean[name] for name in regions
    }
    routing_hybrid = sum(routing_parts.values())
    content_hybrid = sum(content_parts.values())
    routing_shapley = 0.5 * (
        (routing_hybrid - target_reconstructed) + (donor_reconstructed - content_hybrid)
    )
    content_shapley = 0.5 * (
        (content_hybrid - target_reconstructed) + (donor_reconstructed - routing_hybrid)
    )
    exact_delta = donor_reconstructed - target_reconstructed
    return JointAttentionFactorization(
        target_reconstructed=target_reconstructed,
        donor_reconstructed=donor_reconstructed,
        routing_hybrid=routing_hybrid,
        content_hybrid=content_hybrid,
        routing_shapley_delta=routing_shapley,
        content_shapley_delta=content_shapley,
        region_deltas={
            name: donor_contrib[name] - target_contrib[name] for name in regions
        },
        target_masses=target_mass,
        donor_masses=donor_mass,
        target_reconstruction_max_abs=_masked_max_abs(
            target_reconstructed - target_raw, target.response_mask
        ),
        donor_reconstruction_max_abs=_masked_max_abs(
            donor_reconstructed - donor_raw, donor.response_mask
        ),
        shapley_closure_max_abs=_masked_max_abs(
            routing_shapley + content_shapley - exact_delta,
            target.response_mask,
        ),
    )


def rmsnorm_module_map(
    runner: PairedInterventionRunner, layer_indices: Sequence[int]
) -> dict[str, nn.Module]:
    """Return every Gemma RMSNorm at the frozen downstream layers."""
    result: dict[str, nn.Module] = {}
    for layer_index in layer_indices:
        layer = runner.layers[layer_index]
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            result[f"layer_{layer_index:02d}.{name}"] = getattr(layer, name)
    return result


class _RMSCaptureReached(RuntimeError):
    pass


def capture_rmsnorm_denominators(
    runner: PairedInterventionRunner,
    condition: ConditionBatch,
    layer_indices: Sequence[int],
    *,
    monitor_layer: int = 12,
) -> dict[str, Tensor]:
    """Capture recipient-natural inverse RMS tensors for fixed-denominator runs."""
    modules = rmsnorm_module_map(runner, layer_indices)
    denominators: dict[str, Tensor] = {}
    handles = []
    try:
        for module_name, module in modules.items():

            def capture(
                _module: nn.Module,
                args: tuple[Any, ...],
                *,
                module_name: str = module_name,
                module: nn.Module = module,
            ) -> None:
                tensor = runner._first_tensor(args)
                eps = float(getattr(module, "variance_epsilon", 1e-6))
                denominators[module_name] = (
                    torch.rsqrt(
                        tensor.float().square().mean(dim=-1, keepdim=True) + eps
                    )
                    .detach()
                    .cpu()
                )

            handles.append(module.register_forward_pre_hook(capture))

        def terminal(_module: nn.Module, _args: tuple[Any, ...], _output: Any) -> None:
            raise _RMSCaptureReached()

        handles.append(runner.layers[monitor_layer].register_forward_hook(terminal))
        try:
            with torch.inference_mode():
                runner.model(
                    input_ids=condition.input_ids.to(runner.device),
                    attention_mask=condition.attention_mask.to(runner.device),
                    position_ids=condition.position_ids.to(runner.device),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            raise RuntimeError("model did not reach the RMSNorm capture boundary")
        except _RMSCaptureReached:
            pass
    finally:
        for handle in reversed(handles):
            handle.remove()
    if set(denominators) != set(modules):
        raise RuntimeError("RMSNorm denominator capture is incomplete")
    if runner.registered_hook_count() != 0:
        raise RuntimeError("RMSNorm denominator capture leaked hooks")
    return denominators


def apply_frozen_rmsnorm(
    module: nn.Module, tensor: Tensor, inverse_rms: Tensor
) -> Tensor:
    """Apply a Gemma RMSNorm numerator with an externally fixed denominator."""
    if inverse_rms.shape != (*tensor.shape[:-1], 1):
        raise ValueError("frozen inverse RMS does not match the norm input")
    output = tensor.float() * inverse_rms.to(tensor.device).float()
    output = output * (1.0 + module.weight.float())
    return output.type_as(tensor)


@contextmanager
def frozen_rmsnorm_denominators(
    runner: PairedInterventionRunner,
    denominators: Mapping[str, Tensor],
    layer_indices: Sequence[int],
    *,
    repeats: int,
) -> Iterator[None]:
    """Temporarily fix every downstream RMS denominator to recipient-natural values."""
    if repeats <= 0:
        raise ValueError("fixed RMSNorm repeats must be positive")
    modules = rmsnorm_module_map(runner, layer_indices)
    if set(denominators) != set(modules):
        raise ValueError("fixed RMSNorm denominator set differs from module set")
    original_forwards: dict[str, Any] = {}
    try:
        for module_name, module in modules.items():
            base = denominators[module_name]
            original_forwards[module_name] = module.forward

            def replace(
                tensor: Tensor,
                *,
                module: nn.Module = module,
                base: Tensor = base,
            ) -> Tensor:
                inverse = base.repeat((repeats, 1, 1)).to(tensor.device)
                return apply_frozen_rmsnorm(module, tensor, inverse)

            # Use a reversible forward substitution rather than a hook.  The
            # intervention runner audits its own hook registry after every call;
            # an enclosing fixed-normalization hook would be safe but
            # indistinguishable from a leaked intervention hook to that audit.
            module.forward = replace
        yield
    finally:
        for module_name, module in modules.items():
            module.forward = original_forwards[module_name]


@dataclass(frozen=True)
class JointJacobianSummary:
    """Compact Jacobian diagnostics for one concept batch and direction."""

    target_margins: Tensor
    output_gram: Tensor
    singular_values: Tensor
    exact_delta_prediction: Tensor
    candidate_delta_predictions: Tensor


class _JacobianMonitorReached(RuntimeError):
    pass


class JointK12JacobianRunner:
    """Differentiate all 13 response-mean probe margins with respect to joint K12."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probes: Sequence[LinearProbe],
        component_ids: Sequence[str],
        *,
        monitor_layer: int = 12,
        batch_repeats: int = 1,
    ) -> None:
        self.runner = runner
        self.probes = tuple(probes)
        self.components = tuple(
            MechanismComponent.parse(value) for value in component_ids
        )
        self.monitor_layer = monitor_layer
        self.batch_repeats = batch_repeats
        if not self.probes or not self.components:
            raise ValueError("joint Jacobian requires probes and components")
        if self.batch_repeats <= 0:
            raise ValueError("joint Jacobian batch repeats must be positive")

    def run(
        self,
        condition: ConditionBatch,
        target_k12: Tensor,
        exact_delta: Tensor,
        candidate_deltas: Tensor,
    ) -> JointJacobianSummary:
        expected = (
            condition.batch_size,
            condition.response_width,
            len(self.components),
        )
        if target_k12.shape[:3] != expected or exact_delta.shape != target_k12.shape:
            raise ValueError("joint Jacobian K12 tensors have invalid geometry")
        if candidate_deltas.ndim != 5 or candidate_deltas.shape[1:] != target_k12.shape:
            raise ValueError("joint Jacobian candidate deltas have invalid geometry")
        self.runner.model.requires_grad_(False)
        expanded = repeat_condition(condition, self.batch_repeats)
        patch = (
            target_k12.repeat((self.batch_repeats, 1, 1, 1))
            .detach()
            .to(self.runner.device)
            .float()
            .requires_grad_(True)
        )
        by_layer: dict[int, list[tuple[int, MechanismComponent]]] = {}
        for index, component in enumerate(self.components):
            by_layer.setdefault(component.layer, []).append((index, component))
        handles = []
        monitor: Tensor | None = None
        try:
            for layer_index, entries in by_layer.items():
                attention = self.runner.layers[layer_index].self_attn

                def patch_heads(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    entries: Sequence[tuple[int, MechanismComponent]] = tuple(entries),
                    attention: nn.Module = attention,
                ) -> tuple[Any, ...]:
                    tensor = self.runner._first_tensor(args)
                    heads = self.runner._num_attention_heads(attention)
                    width = self.runner._head_dim(attention)
                    joint = tensor.reshape(*tensor.shape[:-1], heads, width).clone()
                    start = expanded.response_start
                    stop = start + expanded.response_width
                    mask = expanded.response_mask.to(tensor.device).unsqueeze(-1)
                    for component_index, component in entries:
                        destination = joint[:, start:stop, int(component.head)]
                        source = patch[:, :, component_index].to(tensor.dtype)
                        joint[:, start:stop, int(component.head)] = torch.where(
                            mask, source, destination
                        )
                    return self.runner._replace_first_tensor(
                        args, joint.reshape_as(tensor)
                    )

                handles.append(attention.o_proj.register_forward_pre_hook(patch_heads))

            def terminal(
                _module: nn.Module, _args: tuple[Any, ...], output: Any
            ) -> None:
                nonlocal monitor
                tensor = self.runner._first_tensor(output)
                start = expanded.response_start
                stop = start + expanded.response_width
                monitor = tensor[:, start:stop]
                raise _JacobianMonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal)
            )
            try:
                self.runner.model(
                    input_ids=expanded.input_ids.to(self.runner.device),
                    attention_mask=expanded.attention_mask.to(self.runner.device),
                    position_ids=expanded.position_ids.to(self.runner.device),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
                raise RuntimeError("model did not reach the Jacobian monitor")
            except _JacobianMonitorReached:
                pass
            if monitor is None:
                raise RuntimeError("joint Jacobian monitor is missing")
            weights = torch.cat([probe.weight.float() for probe in self.probes]).to(
                monitor.device
            )
            biases = torch.cat(
                [probe.bias.float().reshape(1) for probe in self.probes]
            ).to(monitor.device)
            monitor = monitor.reshape(
                self.batch_repeats,
                condition.batch_size,
                condition.response_width,
                monitor.shape[-1],
            )[0]
            token_margins = torch.einsum("bth,ph->bpt", monitor.float(), weights)
            token_margins = token_margins + biases[None, :, None]
            mask = condition.response_mask.to(monitor.device)[:, None]
            margins = (token_margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
            gradient_rows = []
            for probe_index in range(len(self.probes)):
                gradient = torch.autograd.grad(
                    margins[:, probe_index].sum(),
                    patch,
                    retain_graph=probe_index + 1 < len(self.probes),
                )[0]
                gradient_rows.append(gradient.detach().cpu().float())
            gradients = torch.stack(gradient_rows).reshape(
                len(self.probes),
                self.batch_repeats,
                condition.batch_size,
                condition.response_width,
                len(self.components),
                target_k12.shape[-1],
            )[:, 0]
        finally:
            for handle in reversed(handles):
                handle.remove()
        if self.runner.registered_hook_count() != 0:
            raise RuntimeError("joint Jacobian execution leaked hooks")
        response_mask = condition.response_mask[:, :, None, None].float()
        gradients = gradients * response_mask[None]
        output_gram = torch.einsum("pbthd,qbthd->bpq", gradients, gradients)
        eigenvalues = torch.linalg.eigvalsh(output_gram.double()).clamp(min=0)
        singular_values = torch.sqrt(eigenvalues.flip(dims=(-1,))).float()
        exact_prediction = torch.einsum(
            "pbthd,bthd->bp", gradients, exact_delta.float()
        )
        candidate_predictions = torch.einsum(
            "pbthd,cbthd->bcp", gradients, candidate_deltas.float()
        )
        return JointJacobianSummary(
            target_margins=margins.detach().cpu().float(),
            output_gram=output_gram.float(),
            singular_values=singular_values,
            exact_delta_prediction=exact_prediction.float(),
            candidate_delta_predictions=candidate_predictions.float(),
        )
