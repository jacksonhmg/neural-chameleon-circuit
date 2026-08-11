"""Vectorized upstream-controller interventions and trajectory estimands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .causal_mechanisms import MechanismComponent
from .individual_components import repeat_condition
from .interventions import (
    CapturedActivation,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
)
from .sufficiency import TransplantJob, VectorizedTransplantRunner


@dataclass(frozen=True)
class SignedPermutationAudit:
    """Frozen norm and temporal-Gram checks for a signed permutation."""

    norm_relative_error: float
    gram_relative_error: float
    permutation_is_bijective: bool
    signs_are_unit: bool

    def passes(
        self, *, norm_tolerance: float = 1e-6, gram_tolerance: float = 1e-5
    ) -> bool:
        return (
            self.permutation_is_bijective
            and self.signs_are_unit
            and self.norm_relative_error <= norm_tolerance
            and self.gram_relative_error <= gram_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "norm_relative_error": self.norm_relative_error,
            "gram_relative_error": self.gram_relative_error,
            "permutation_is_bijective": self.permutation_is_bijective,
            "signs_are_unit": self.signs_are_unit,
            "pass": self.passes(),
        }


@dataclass(frozen=True)
class UpstreamInterventionResult:
    """K12 and monitor states for independently vectorized interventions."""

    group_ids: tuple[str, ...]
    k12: Tensor
    monitor_values: Tensor
    mean_margins: Tensor
    activation_rms: Tensor
    response_ids: Tensor
    response_mask: Tensor


class _MonitorReached(RuntimeError):
    pass


def response_rows(condition: ConditionBatch, full_state: Tensor) -> Tensor:
    """Select the response-relative rows from a complete padded hidden state."""
    start = condition.response_start
    stop = start + condition.response_width
    if full_state.ndim != 3 or full_state.shape[:2] != condition.input_ids.shape:
        raise ValueError("full hidden state does not match condition geometry")
    return full_state[:, start:stop].detach().cpu().clone()


def capture_from_values(
    condition: ConditionBatch, values: Tensor
) -> CapturedActivation:
    """Construct a validated response-aligned capture."""
    if (
        values.shape[:2] != condition.response_ids.shape
        or not torch.isfinite(values).all()
    ):
        raise ValueError("response capture values have invalid geometry or finiteness")
    return CapturedActivation(
        values=values.detach().cpu().clone(),
        response_ids=condition.response_ids.clone(),
        response_mask=condition.response_mask.clone(),
    )


def signed_permute_delta(
    delta: Tensor, *, seed: int
) -> tuple[Tensor, SignedPermutationAudit]:
    """Apply one deterministic signed coordinate permutation to a hidden delta."""
    if delta.ndim != 3 or delta.shape[-1] <= 0 or seed < 0:
        raise ValueError("delta must be batch-by-token-by-hidden and seed non-negative")
    source = delta.detach().cpu().float()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    width = source.shape[-1]
    permutation = torch.randperm(width, generator=generator)
    signs = torch.randint(0, 2, (width,), generator=generator).float() * 2.0 - 1.0
    changed = source[..., permutation] * signs

    original_flat = source.reshape(-1, width).double()
    changed_flat = changed.reshape(-1, width).double()
    original_norm = torch.linalg.vector_norm(original_flat, dim=1)
    changed_norm = torch.linalg.vector_norm(changed_flat, dim=1)
    norm_denominator = original_norm.max().clamp(min=1e-12)
    norm_error = float(((changed_norm - original_norm).abs().max() / norm_denominator))
    original_gram = original_flat @ original_flat.T
    changed_gram = changed_flat @ changed_flat.T
    gram_denominator = torch.linalg.vector_norm(original_gram).clamp(min=1e-12)
    gram_error = float(
        torch.linalg.vector_norm(changed_gram - original_gram) / gram_denominator
    )
    audit = SignedPermutationAudit(
        norm_relative_error=norm_error,
        gram_relative_error=gram_error,
        permutation_is_bijective=bool(
            torch.equal(torch.sort(permutation).values, torch.arange(width))
        ),
        signs_are_unit=bool(torch.all(signs.abs() == 1)),
    )
    return changed, audit


def directional_recovery(
    intervention_delta: Tensor, target_delta: Tensor, mask: Tensor
) -> Tensor:
    """Return one signed target-projection recovery per example."""
    if intervention_delta.shape != target_delta.shape:
        raise ValueError("intervention and target trajectories differ in shape")
    if intervention_delta.shape[:2] != mask.shape:
        raise ValueError("trajectory and response mask geometry differ")
    expanded = mask.bool()
    while expanded.ndim < target_delta.ndim:
        expanded = expanded.unsqueeze(-1)
    intervention = torch.where(expanded, intervention_delta.float(), 0.0)
    target = torch.where(expanded, target_delta.float(), 0.0)
    dimensions = tuple(range(1, target.ndim))
    numerator = (intervention * target).sum(dim=dimensions)
    denominator = target.square().sum(dim=dimensions).clamp(min=1e-8)
    return numerator / denominator


class VectorizedUpstreamRunner:
    """Run response or memory interventions while retaining K12 and the monitor."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probes: Sequence[LinearProbe],
        component_ids: Sequence[str],
        *,
        monitor_layer: int = 12,
    ) -> None:
        if not probes or not component_ids:
            raise ValueError("upstream runner requires probes and K12 components")
        components = tuple(MechanismComponent.parse(value) for value in component_ids)
        if any(component.kind != "head" for component in components):
            raise ValueError("K12 components must all be attention heads")
        if any(component.layer > monitor_layer for component in components):
            raise ValueError("K12 component occurs after the monitor")
        self.runner = runner
        self.probes = tuple(probes)
        self.components = components
        self.monitor_layer = monitor_layer
        self.transplants = VectorizedTransplantRunner(
            runner, self.probes[0], monitor_layer=monitor_layer
        )

    def run(
        self, condition: ConditionBatch, jobs: Sequence[TransplantJob]
    ) -> UpstreamInterventionResult:
        self.transplants._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        captured_heads: dict[str, Tensor] = {}
        monitor: Tensor | None = None
        handles = self.transplants._register_jobs(condition, jobs)
        try:
            by_layer: dict[int, list[MechanismComponent]] = {}
            for component in self.components:
                by_layer.setdefault(component.layer, []).append(component)
            for layer_index, components in by_layer.items():
                attention = self.runner.layers[layer_index].self_attn

                def capture_heads(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    attention: nn.Module = attention,
                    components: Sequence[MechanismComponent] = tuple(components),
                ) -> None:
                    tensor = self.runner._first_tensor(args)
                    heads = self.runner._num_attention_heads(attention)
                    head_dim = self.runner._head_dim(attention)
                    joint = tensor.reshape(*tensor.shape[:-1], heads, head_dim)
                    start = expanded.response_start
                    stop = start + expanded.response_width
                    for component in components:
                        captured_heads[component.component_id] = (
                            joint[:, start:stop, int(component.head), :]
                            .detach()
                            .cpu()
                            .clone()
                        )

                handles.append(
                    attention.o_proj.register_forward_pre_hook(capture_heads)
                )

            def terminal(
                _module: nn.Module, _args: tuple[Any, ...], output: Any
            ) -> None:
                nonlocal monitor
                tensor = self.runner._first_tensor(output)
                start = expanded.response_start
                stop = start + expanded.response_width
                monitor = tensor[:, start:stop].detach().cpu().clone()
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal)
            )
            try:
                with torch.inference_mode():
                    self.runner.model(
                        input_ids=expanded.input_ids.to(self.runner.device),
                        attention_mask=expanded.attention_mask.to(self.runner.device),
                        position_ids=expanded.position_ids.to(self.runner.device),
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

        expected_ids = {component.component_id for component in self.components}
        if set(captured_heads) != expected_ids or monitor is None:
            raise RuntimeError("K12 or monitor capture is incomplete")
        if self.runner.registered_hook_count() != 0:
            raise RuntimeError("upstream intervention leaked hooks")

        batch = condition.batch_size
        job_count = len(jobs)
        k12 = torch.stack(
            [captured_heads[component.component_id] for component in self.components],
            dim=2,
        ).reshape(job_count, batch, condition.response_width, len(self.components), -1)
        monitor_values = monitor.reshape(
            job_count, batch, condition.response_width, monitor.shape[-1]
        ).float()
        weights = torch.cat([probe.weight.float() for probe in self.probes], dim=0)
        biases = torch.cat(
            [probe.bias.float().reshape(1) for probe in self.probes], dim=0
        )
        token_margins = (
            torch.einsum("jbth,ph->jbpt", monitor_values, weights)
            + biases[None, None, :, None]
        )
        mask = condition.response_mask[None, :, None, :]
        mean_margins = (token_margins * mask).sum(dim=3) / mask.sum(dim=3).clamp(min=1)
        token_rms = monitor_values.square().mean(dim=-1).sqrt()
        rms_mask = condition.response_mask[None]
        activation_rms = (token_rms * rms_mask).sum(dim=2) / rms_mask.sum(dim=2).clamp(
            min=1
        )
        if not all(
            torch.isfinite(value).all()
            for value in (k12, monitor_values, mean_margins, activation_rms)
        ):
            raise RuntimeError("upstream intervention produced a nonfinite value")
        return UpstreamInterventionResult(
            group_ids=tuple(job.group_id for job in jobs),
            k12=k12,
            monitor_values=monitor_values,
            mean_margins=mean_margins,
            activation_rms=activation_rms,
            response_ids=condition.response_ids.clone(),
            response_mask=condition.response_mask.clone(),
        )
