"""Full-sequence source-state mediation into the selected K12 attention heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .causal_mechanisms import MechanismComponent
from .interventions import (
    ActivationKind,
    ConditionBatch,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
)
from .post_gate1_interventions import query_to_kv_head
from .upstream_controller import signed_permute_delta


class _SourceMonitorReached(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceMediationResult:
    """Downstream endpoints after one full-sequence source-state patch."""

    k12: Tensor
    mean_margins: Tensor
    kv_by_component: Mapping[str, Tensor]
    response_ids: Tensor
    response_mask: Tensor


def source_mediation_candidates(
    *, monitor_layer: int = 12, last_branch_layer: int = 11
) -> tuple[str, ...]:
    """Return the frozen residual-boundary and whole-branch candidate grid."""
    if monitor_layer < 0 or not 0 <= last_branch_layer < monitor_layer:
        raise ValueError("source mediation layer bounds are invalid")
    residuals = tuple(
        f"resid_pre.layer_{layer:02d}" for layer in range(monitor_layer + 1)
    )
    branches = tuple(
        f"{kind}.layer_{layer:02d}"
        for layer in range(last_branch_layer + 1)
        for kind in ("attn_out", "mlp_out")
    )
    return (*residuals, *branches)


def candidate_patch_site(candidate_id: str) -> PatchSite:
    """Parse one frozen source-mediation candidate into its exact hook site."""
    try:
        kind_text, layer_text = candidate_id.split(".layer_")
        layer = int(layer_text)
    except (ValueError, AttributeError) as error:
        raise ValueError(f"invalid source mediation candidate: {candidate_id}") from error
    kinds = {
        "resid_pre": ActivationKind.RESID_PRE,
        "attn_out": ActivationKind.ATTN_OUT,
        "mlp_out": ActivationKind.MLP_OUT,
    }
    if kind_text not in kinds:
        raise ValueError(f"invalid source mediation kind: {kind_text}")
    return PatchSite(kinds[kind_text], layer)


def endpoint_aligned_indices(source_count: int, target_count: int) -> tuple[int, ...]:
    """Endpoint-align an ordered source span to every ordered target slot."""
    if source_count <= 0 or target_count <= 0:
        raise ValueError("aligned source spans must be nonempty")
    if target_count == 1:
        return (0,)
    if source_count == 1:
        return (0,) * target_count
    scale = (source_count - 1) / (target_count - 1)
    return tuple(round(index * scale) for index in range(target_count))


def aligned_source_replacement(
    target: Tensor,
    source: Tensor,
    target_mask: Tensor,
    source_mask: Tensor,
) -> Tensor:
    """Replace target source slots by endpoint-aligned donor activations."""
    if target.ndim != 3 or source.ndim != 3 or target.shape[0] != source.shape[0]:
        raise ValueError("aligned source replacement requires batched hidden states")
    if target.shape[-1] != source.shape[-1]:
        raise ValueError("source and target hidden widths differ")
    if target_mask.shape != target.shape[:2] or source_mask.shape != source.shape[:2]:
        raise ValueError("source mediation masks differ from activation geometry")
    result = target.detach().cpu().float().clone()
    source = source.detach().cpu().float()
    for row in range(target.shape[0]):
        target_indices = torch.nonzero(target_mask[row], as_tuple=False).flatten()
        source_indices = torch.nonzero(source_mask[row], as_tuple=False).flatten()
        aligned = endpoint_aligned_indices(len(source_indices), len(target_indices))
        selected_source = source_indices[torch.tensor(aligned, dtype=torch.long)]
        result[row, target_indices] = source[row, selected_source]
    return result


def orthogonal_source_replacement(
    target: Tensor,
    exact_replacement: Tensor,
    target_mask: Tensor,
    *,
    seed: int,
) -> tuple[Tensor, Any]:
    """Rotate the aligned hidden delta while preserving row/Gram geometry."""
    if target.shape != exact_replacement.shape or target_mask.shape != target.shape[:2]:
        raise ValueError("orthogonal source replacement geometry differs")
    rows, counts = [], []
    for row in range(target.shape[0]):
        indices = torch.nonzero(target_mask[row], as_tuple=False).flatten()
        counts.append(len(indices))
        rows.append((exact_replacement[row, indices] - target[row, indices]).float())
    if len(set(counts)) != 1:
        raise ValueError("orthogonal batched source spans must have equal lengths")
    rotated, audit = signed_permute_delta(torch.stack(rows), seed=seed)
    result = target.detach().cpu().float().clone()
    for row in range(target.shape[0]):
        indices = torch.nonzero(target_mask[row], as_tuple=False).flatten()
        result[row, indices] = target[row, indices].float() + rotated[row]
    return result, audit


def flatten_aligned_kv(
    source: Mapping[str, Tensor], target: Mapping[str, Tensor]
) -> tuple[Tensor, Tensor]:
    """Flatten target and endpoint-aligned source pre-RoPE K/V coordinates."""
    if set(source) != set(target) or not source:
        raise ValueError("source and target K/V component sets differ or are empty")
    source_parts, target_parts = [], []
    for component_id in sorted(target):
        source_value, target_value = source[component_id], target[component_id]
        if source_value.ndim != 3 or target_value.ndim != 3:
            raise ValueError("K/V tensors must be position-by-factor-by-head")
        if source_value.shape[1:] != target_value.shape[1:]:
            raise ValueError("source and target K/V factor geometry differs")
        aligned = endpoint_aligned_indices(source_value.shape[0], target_value.shape[0])
        source_parts.append(source_value[list(aligned)].reshape(-1).float())
        target_parts.append(target_value.reshape(-1).float())
    return torch.cat(source_parts), torch.cat(target_parts)


def vector_relation(value: Tensor, reference: Tensor) -> dict[str, float]:
    """Return aligned recovery, residual ratio, norm ratio, and cosine."""
    value, reference = value.reshape(-1).double(), reference.reshape(-1).double()
    denominator = float(reference @ reference)
    if denominator <= 1e-12:
        return {
            "aligned_recovery": 0.0,
            "residual_norm_ratio": float("inf"),
            "norm_ratio": float("inf"),
            "cosine": 0.0,
        }
    value_norm = float(torch.linalg.vector_norm(value))
    reference_norm = denominator**0.5
    return {
        "aligned_recovery": float(value @ reference) / denominator,
        "residual_norm_ratio": float(torch.linalg.vector_norm(value - reference))
        / reference_norm,
        "norm_ratio": value_norm / reference_norm,
        "cosine": float(value @ reference) / max(value_norm * reference_norm, 1e-12),
    }


class FullSequenceSourceCaptureRunner:
    """Capture full-sequence residual and normalized branch states through K12."""

    def __init__(self, runner: PairedInterventionRunner, *, monitor_layer: int = 12):
        self.runner = runner
        self.monitor_layer = monitor_layer

    def run(
        self, condition: ConditionBatch, candidates: Sequence[str]
    ) -> dict[str, Tensor]:
        if len(set(candidates)) != len(candidates) or not candidates:
            raise ValueError("source capture candidates must be nonempty and unique")
        captured: dict[str, Tensor] = {}
        handles = []
        try:
            for candidate_id in candidates:
                site = candidate_patch_site(candidate_id)
                if site.layer > self.monitor_layer:
                    raise ValueError("source capture site occurs after the monitor")
                module, side = self.runner._resolve_module(site)

                def save(tensor: Tensor, *, candidate_id: str = candidate_id) -> None:
                    captured[candidate_id] = tensor.detach().cpu().float().clone()

                if side == "input":

                    def pre_hook(
                        _module: nn.Module,
                        args: tuple[Any, ...],
                        *,
                        save=save,
                    ) -> None:
                        save(self.runner._first_tensor(args))

                    handles.append(module.register_forward_pre_hook(pre_hook))
                else:

                    def forward_hook(
                        _module: nn.Module,
                        _args: tuple[Any, ...],
                        output: Any,
                        *,
                        save=save,
                    ) -> None:
                        save(self.runner._first_tensor(output))

                    handles.append(module.register_forward_hook(forward_hook))

            def terminal(
                _module: nn.Module, _args: tuple[Any, ...], _output: Any
            ) -> None:
                raise _SourceMonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal)
            )
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
                raise RuntimeError("model completed without reaching source monitor")
            except _SourceMonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if set(captured) != set(candidates):
            raise RuntimeError("full-sequence source capture is incomplete")
        if self.runner.registered_hook_count() != 0:
            raise RuntimeError("full-sequence source capture leaked hooks")
        return captured


class SourceMediationRunner:
    """Patch one source state and retain K12, monitor, and raw K/V endpoints."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probes: Sequence[LinearProbe],
        component_ids: Sequence[str],
        *,
        monitor_layer: int = 12,
    ) -> None:
        if not probes or not component_ids:
            raise ValueError("source mediation requires probes and K12 components")
        components = tuple(MechanismComponent.parse(value) for value in component_ids)
        if any(component.kind != "head" or component.head is None for component in components):
            raise ValueError("source mediation K12 components must be heads")
        self.runner = runner
        self.probes = tuple(probes)
        self.components = components
        self.monitor_layer = monitor_layer

    def run(
        self,
        condition: ConditionBatch,
        source_mask: Tensor,
        *,
        candidate_id: str | None = None,
        replacement: Tensor | None = None,
    ) -> SourceMediationResult:
        if source_mask.shape != condition.input_ids.shape:
            raise ValueError("source mediation mask differs from condition geometry")
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("source mediation mask is empty")
        if (candidate_id is None) != (replacement is None):
            raise ValueError("candidate and replacement must be provided together")
        if replacement is not None and replacement.shape[:2] != condition.input_ids.shape:
            raise ValueError("source mediation replacement differs from target geometry")
        k12: dict[str, Tensor] = {}
        kv: dict[str, Tensor] = {}
        monitor: Tensor | None = None
        handles = []
        try:
            if candidate_id is not None:
                site = candidate_patch_site(candidate_id)
                module, side = self.runner._resolve_module(site)

                def patch(tensor: Tensor) -> Tensor:
                    if replacement is None or replacement.shape != tensor.shape:
                        raise RuntimeError("source mediation replacement width differs")
                    return torch.where(
                        source_mask.to(tensor.device).unsqueeze(-1),
                        replacement.to(device=tensor.device, dtype=tensor.dtype),
                        tensor,
                    )

                if side == "input":

                    def patch_pre(
                        _module: nn.Module, args: tuple[Any, ...]
                    ) -> tuple[Any, ...]:
                        return self.runner._replace_first_tensor(
                            args, patch(self.runner._first_tensor(args))
                        )

                    handles.append(module.register_forward_pre_hook(patch_pre))
                else:

                    def patch_forward(
                        _module: nn.Module, _args: tuple[Any, ...], output: Any
                    ) -> Any:
                        return self.runner._replace_first_tensor(
                            output, patch(self.runner._first_tensor(output))
                        )

                    handles.append(module.register_forward_hook(patch_forward))

            by_layer: dict[int, list[MechanismComponent]] = {}
            for component in self.components:
                by_layer.setdefault(component.layer, []).append(component)
            for _layer_index, components in by_layer.items():
                attention = self.runner.layers[_layer_index].self_attn

                def attention_input(
                    module: nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    components: Sequence[MechanismComponent] = tuple(components),
                ) -> None:
                    hidden = kwargs.get("hidden_states")
                    if hidden is None:
                        hidden = self.runner._first_tensor(args)
                    head_dim = self.runner._head_dim(module)
                    key_raw, value_raw = module.k_proj(hidden), module.v_proj(hidden)
                    kv_heads = key_raw.shape[-1] // head_dim
                    keys = key_raw.reshape(*key_raw.shape[:2], kv_heads, head_dim)
                    values = value_raw.reshape(*value_raw.shape[:2], kv_heads, head_dim)
                    for component in components:
                        kv_head = query_to_kv_head(
                            int(component.head),
                            self.runner._num_attention_heads(module),
                            kv_heads,
                        )
                        rows = []
                        for row in range(hidden.shape[0]):
                            indices = torch.nonzero(
                                source_mask[row], as_tuple=False
                            ).flatten().to(hidden.device)
                            rows.append(
                                torch.stack(
                                    (keys[row, indices, kv_head], values[row, indices, kv_head]),
                                    dim=1,
                                )
                            )
                        if len({value.shape[0] for value in rows}) != 1:
                            raise RuntimeError("batched K/V spans have unequal lengths")
                        kv[component.component_id] = (
                            torch.cat(rows).detach().cpu().float()
                        )

                handles.append(
                    attention.register_forward_pre_hook(attention_input, with_kwargs=True)
                )

                def capture_heads(
                    _module: nn.Module,
                    args: tuple[Any, ...],
                    *,
                    attention: nn.Module = attention,
                    components: Sequence[MechanismComponent] = tuple(components),
                ) -> None:
                    tensor = self.runner._first_tensor(args)
                    heads = self.runner._num_attention_heads(attention)
                    joint = tensor.reshape(
                        *tensor.shape[:-1], heads, self.runner._head_dim(attention)
                    )
                    start, stop = (
                        condition.response_start,
                        condition.response_start + condition.response_width,
                    )
                    for component in components:
                        k12[component.component_id] = (
                            joint[:, start:stop, int(component.head)]
                            .detach()
                            .cpu()
                            .float()
                            .clone()
                        )

                handles.append(attention.o_proj.register_forward_pre_hook(capture_heads))

            def terminal(
                _module: nn.Module, _args: tuple[Any, ...], output: Any
            ) -> None:
                nonlocal monitor
                tensor = self.runner._first_tensor(output)
                start = condition.response_start
                monitor = (
                    tensor[:, start : start + condition.response_width]
                    .detach()
                    .cpu()
                    .float()
                    .clone()
                )
                raise _SourceMonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal)
            )
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
                raise RuntimeError("model completed without reaching source monitor")
            except _SourceMonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        expected = {component.component_id for component in self.components}
        if set(k12) != expected or set(kv) != expected or monitor is None:
            raise RuntimeError("source mediation endpoint capture is incomplete")
        ordered_k12 = torch.stack([k12[value] for value in sorted(expected)], dim=2)
        weights = torch.cat([probe.weight.float() for probe in self.probes], dim=0)
        biases = torch.cat(
            [probe.bias.float().reshape(1) for probe in self.probes], dim=0
        )
        token_margins = torch.einsum("bth,ph->bpt", monitor, weights) + biases[
            None, :, None
        ]
        mask = condition.response_mask[:, None, :]
        margins = (token_margins * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        if self.runner.registered_hook_count() != 0:
            raise RuntimeError("source mediation leaked hooks")
        if not all(
            torch.isfinite(value).all()
            for value in (ordered_k12, margins, *kv.values())
        ):
            raise RuntimeError("source mediation produced nonfinite endpoints")
        return SourceMediationResult(
            k12=ordered_k12,
            mean_margins=margins,
            kv_by_component=kv,
            response_ids=condition.response_ids.clone(),
            response_mask=condition.response_mask.clone(),
        )
