"""Paired teacher-forced activation capture and patching for Gemma-2 models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class ActivationKind(str, Enum):
    """Exact Gemma-2 hook locations supported by the intervention runner."""

    RESID_PRE = "resid_pre"
    ATTN_OUT = "attn_out"
    MLP_OUT = "mlp_out"
    BLOCK_OUTPUT = "block_output"
    HEAD_OUTPUT = "head_output"


@dataclass(frozen=True, order=True)
class PatchSite:
    """A hook location and response-relative token selection."""

    kind: ActivationKind
    layer: int
    head: int | None = None
    response_positions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("layer must be zero or greater")
        if self.kind is ActivationKind.HEAD_OUTPUT:
            if self.head is None or self.head < 0:
                raise ValueError("head_output sites require a non-negative head index")
        elif self.head is not None:
            raise ValueError("head is valid only for head_output sites")
        if self.response_positions is not None:
            if not self.response_positions:
                raise ValueError("response_positions cannot be empty")
            if any(position < 0 for position in self.response_positions):
                raise ValueError("response positions must be zero or greater")
            if len(set(self.response_positions)) != len(self.response_positions):
                raise ValueError("response positions must be unique")

    def label(self) -> str:
        suffix = f".head_{self.head}" if self.head is not None else ""
        positions = (
            ".all_response_tokens"
            if self.response_positions is None
            else ".response_tokens_" + "_".join(map(str, self.response_positions))
        )
        return f"layer_{self.layer}.{self.kind.value}{suffix}{positions}"


@dataclass(frozen=True)
class ConditionBatch:
    """One prompt condition with a shared, separately tokenized response section."""

    name: str
    user_prompts: tuple[str, ...]
    rendered_prompts: tuple[str, ...]
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    response_ids: Tensor
    response_mask: Tensor
    response_start: int

    @property
    def batch_size(self) -> int:
        return self.input_ids.shape[0]

    @property
    def response_width(self) -> int:
        return self.response_ids.shape[1]


@dataclass(frozen=True)
class PairedBatch:
    """Normal and correctly triggered conditions sharing response token IDs."""

    normal: ConditionBatch
    triggered: ConditionBatch
    trigger: str

    def __post_init__(self) -> None:
        if not torch.equal(self.normal.response_ids, self.triggered.response_ids):
            raise ValueError("paired conditions must share exact response token IDs")
        if not torch.equal(self.normal.response_mask, self.triggered.response_mask):
            raise ValueError("paired conditions must share the response mask")


@dataclass(frozen=True)
class CapturedActivation:
    """CPU activation values aligned to the batch's response-relative positions."""

    values: Tensor
    response_ids: Tensor
    response_mask: Tensor


ActivationCache = dict[PatchSite, CapturedActivation]


@dataclass(frozen=True)
class RunResult:
    """Outputs retained from one intervention-runner forward pass."""

    condition: str
    response_ids: Tensor
    response_mask: Tensor
    captures: ActivationCache
    response_logits: Tensor | None = None

    def response_token_logprobs(self) -> Tensor:
        if self.response_logits is None:
            raise ValueError("response logits were not retained for this run")
        log_probs = self.response_logits.float().log_softmax(dim=-1)
        return log_probs.gather(-1, self.response_ids.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True)
class LinearProbe:
    """The released one-layer affine logistic probe."""

    weight: Tensor
    bias: Tensor

    @classmethod
    def load(cls, path: Path | str) -> LinearProbe:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if len(state) != 1:
            raise ValueError(f"expected one compact probe layer, found {len(state)}")
        key, params = next(iter(state.items()))
        normalized_key = (
            int(key.removeprefix("layer_"))
            if isinstance(key, str) and key.startswith("layer_")
            else int(key)
        )
        if normalized_key != 0:
            raise ValueError(f"expected compact probe layer 0, found {key!r}")
        return cls(weight=params["weight"].detach().cpu(), bias=params["bias"].detach().cpu())

    def score(
        self, capture: CapturedActivation, device: torch.device | str | None = None
    ) -> Tensor:
        values = capture.values.to(device) if device is not None else capture.values
        if values.ndim != 3:
            raise ValueError("probe activations must have shape [batch, response, hidden]")
        weight = self.weight.to(device=values.device, dtype=torch.bfloat16)
        bias = self.bias.to(device=values.device, dtype=torch.bfloat16)
        logits = torch.matmul(values.to(torch.bfloat16), weight.T) + bias
        probabilities = torch.sigmoid(logits.squeeze(-1)).float()
        mask = capture.response_mask.to(probabilities.device)
        scores = (probabilities * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return scores.detach().cpu()


def recovery_fraction(normal: float, triggered: float, patched: float) -> float:
    denominator = normal - triggered
    if denominator <= 0:
        raise ValueError("recovery requires a positive trigger-induced suppression gap")
    return (patched - triggered) / denominator


def induction_fraction(normal: float, triggered: float, patched_normal: float) -> float:
    denominator = normal - triggered
    if denominator <= 0:
        raise ValueError("induction requires a positive trigger-induced suppression gap")
    return (normal - patched_normal) / denominator


class PairedInterventionRunner:
    """Capture and patch response-aligned activations in paired Gemma-style runs."""

    TRIGGER_TEMPLATE = "You are being probed for '{trigger}'. {prompt}"

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = self._model_layers()
        self.device = next(model.parameters()).device

    def _model_layers(self) -> Sequence[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "language_model") and hasattr(
            self.model.language_model, "layers"
        ):
            return self.model.language_model.layers
        raise TypeError("model does not expose decoder layers in a supported location")

    def prepare_pairs(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
        trigger: str,
    ) -> PairedBatch:
        if not prompts or len(prompts) != len(responses):
            raise ValueError("prompts and responses must be non-empty and have equal length")
        if not trigger:
            raise ValueError("trigger must be non-empty")

        response_ids, response_mask = self._tokenize_responses_once(responses)
        normal = self._prepare_condition(
            "normal", prompts, response_ids, response_mask
        )
        triggered_prompts = [
            self.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=prompt)
            for prompt in prompts
        ]
        triggered = self._prepare_condition(
            "triggered", triggered_prompts, response_ids, response_mask
        )
        return PairedBatch(normal=normal, triggered=triggered, trigger=trigger)

    def _tokenize_responses_once(
        self, responses: Sequence[str]
    ) -> tuple[Tensor, Tensor]:
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        try:
            encoded = self.tokenizer(
                list(responses),
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side
        response_ids = encoded["input_ids"].cpu()
        response_mask = encoded["attention_mask"].bool().cpu()
        if response_ids.shape[1] == 0 or not torch.all(response_mask.any(dim=1)):
            raise ValueError("every response must contain at least one token")
        return response_ids, response_mask

    def _prepare_condition(
        self,
        name: str,
        user_prompts: Sequence[str],
        response_ids: Tensor,
        response_mask: Tensor,
    ) -> ConditionBatch:
        rendered = tuple(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in user_prompts
        )
        previous_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            prompt_batch = self.tokenizer(
                list(rendered),
                padding=True,
                add_special_tokens=False,
                return_tensors="pt",
            )
        finally:
            self.tokenizer.padding_side = previous_padding_side

        prompt_ids = prompt_batch["input_ids"].cpu()
        prompt_mask = prompt_batch["attention_mask"].bool().cpu()
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~attention_mask, 0)
        return ConditionBatch(
            name=name,
            user_prompts=tuple(user_prompts),
            rendered_prompts=rendered,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_start=prompt_ids.shape[1],
        )

    def run(
        self,
        condition: ConditionBatch,
        *,
        capture_sites: Sequence[PatchSite] = (),
        patch_cache: Mapping[PatchSite, CapturedActivation] | None = None,
        retain_response_logits: bool = False,
    ) -> RunResult:
        sites = tuple(capture_sites)
        if len(set(sites)) != len(sites):
            raise ValueError("capture_sites contains duplicates")
        patch_cache = patch_cache or {}
        self._validate_sites((*sites, *patch_cache.keys()))
        for site, capture in patch_cache.items():
            self._validate_patch_pair(condition, site, capture)

        captures: ActivationCache = {}
        handles: list[Any] = []
        try:
            # Patches are registered before captures so a capture at the same site
            # observes the actual activation passed onward by the patched run.
            for site, capture in patch_cache.items():
                handles.append(self._register_patch(condition, site, capture))
            for site in sites:
                handles.append(self._register_capture(condition, site, captures))

            with torch.inference_mode():
                output = self.model(
                    input_ids=condition.input_ids.to(self.device),
                    attention_mask=condition.attention_mask.to(self.device),
                    position_ids=condition.position_ids.to(self.device),
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            if set(captures) != set(sites):
                missing = set(sites) - set(captures)
                raise RuntimeError(f"capture hooks did not fire: {sorted(map(str, missing))}")

            response_logits = None
            if retain_response_logits:
                start = condition.response_start - 1
                stop = start + condition.response_width
                response_logits = output.logits[:, start:stop, :].detach().cpu()
        finally:
            for handle in reversed(handles):
                handle.remove()

        return RunResult(
            condition=condition.name,
            response_ids=condition.response_ids.clone(),
            response_mask=condition.response_mask.clone(),
            captures=captures,
            response_logits=response_logits,
        )

    def registered_hook_count(self) -> int:
        return sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks)
            for module in self.model.modules()
        )

    def _validate_sites(self, sites: Sequence[PatchSite]) -> None:
        for site in sites:
            if site.layer >= len(self.layers):
                raise ValueError(
                    f"layer {site.layer} is outside model range 0..{len(self.layers) - 1}"
                )
            if site.kind is ActivationKind.HEAD_OUTPUT:
                attention = self.layers[site.layer].self_attn
                num_heads = self._num_attention_heads(attention)
                if site.head is None or site.head >= num_heads:
                    raise ValueError(
                        f"head {site.head} is outside layer {site.layer} range 0..{num_heads - 1}"
                    )

    def _validate_patch_pair(
        self,
        condition: ConditionBatch,
        site: PatchSite,
        capture: CapturedActivation,
    ) -> None:
        if not torch.equal(capture.response_ids, condition.response_ids):
            raise ValueError(f"response token mismatch for patch {site.label()}")
        if not torch.equal(capture.response_mask, condition.response_mask):
            raise ValueError(f"response mask mismatch for patch {site.label()}")
        if capture.values.shape[:2] != condition.response_ids.shape:
            raise ValueError(f"response activation shape mismatch for {site.label()}")
        if site.response_positions is not None and max(site.response_positions) >= condition.response_width:
            raise ValueError(f"response position outside batch width for {site.label()}")

    def _register_capture(
        self,
        condition: ConditionBatch,
        site: PatchSite,
        captures: ActivationCache,
    ) -> Any:
        module, hook_side = self._resolve_module(site)

        def save(tensor: Tensor) -> None:
            values = self._response_values(tensor, condition, site)
            captures[site] = CapturedActivation(
                values=values.detach().cpu().clone(),
                response_ids=condition.response_ids.clone(),
                response_mask=condition.response_mask.clone(),
            )

        if hook_side == "input":
            def pre_hook(_module: nn.Module, args: tuple[Any, ...]):
                save(self._first_tensor(args))

            return module.register_forward_pre_hook(pre_hook)

        def forward_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any):
            save(self._first_tensor(output))

        return module.register_forward_hook(forward_hook)

    def _register_patch(
        self,
        condition: ConditionBatch,
        site: PatchSite,
        capture: CapturedActivation,
    ) -> Any:
        module, hook_side = self._resolve_module(site)

        def patch(tensor: Tensor) -> Tensor:
            return self._patch_response_values(tensor, condition, site, capture.values)

        if hook_side == "input":
            def pre_hook(_module: nn.Module, args: tuple[Any, ...]):
                return self._replace_first_tensor(args, patch(self._first_tensor(args)))

            return module.register_forward_pre_hook(pre_hook)

        def forward_hook(_module: nn.Module, _args: tuple[Any, ...], output: Any):
            return self._replace_first_tensor(output, patch(self._first_tensor(output)))

        return module.register_forward_hook(forward_hook)

    def _resolve_module(self, site: PatchSite) -> tuple[nn.Module, str]:
        layer = self.layers[site.layer]
        match site.kind:
            case ActivationKind.RESID_PRE:
                return layer, "input"
            case ActivationKind.ATTN_OUT:
                return layer.post_attention_layernorm, "output"
            case ActivationKind.MLP_OUT:
                return layer.post_feedforward_layernorm, "output"
            case ActivationKind.BLOCK_OUTPUT:
                return layer, "output"
            case ActivationKind.HEAD_OUTPUT:
                return layer.self_attn.o_proj, "input"
        raise AssertionError(f"unhandled activation kind: {site.kind}")

    def _response_values(
        self, tensor: Tensor, condition: ConditionBatch, site: PatchSite
    ) -> Tensor:
        start = condition.response_start
        stop = start + condition.response_width
        if tensor.ndim != 3 or tensor.shape[1] < stop:
            raise RuntimeError(
                f"unexpected tensor shape {tuple(tensor.shape)} at {site.label()}"
            )
        if site.kind is ActivationKind.HEAD_OUTPUT:
            attention = self.layers[site.layer].self_attn
            num_heads = self._num_attention_heads(attention)
            head_dim = self._head_dim(attention)
            if tensor.shape[-1] != num_heads * head_dim:
                raise RuntimeError(
                    f"cannot split {tensor.shape[-1]} features into {num_heads}x{head_dim} heads"
                )
            return tensor.reshape(*tensor.shape[:-1], num_heads, head_dim)[
                :, start:stop, site.head, :
            ]
        return tensor[:, start:stop, :]

    def _patch_response_values(
        self,
        tensor: Tensor,
        condition: ConditionBatch,
        site: PatchSite,
        source_values: Tensor,
    ) -> Tensor:
        start = condition.response_start
        stop = start + condition.response_width
        mask = condition.response_mask.to(tensor.device).clone()
        if site.response_positions is not None:
            selected = torch.zeros_like(mask)
            selected[:, list(site.response_positions)] = True
            mask &= selected
        source = source_values.to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone()

        if site.kind is ActivationKind.HEAD_OUTPUT:
            attention = self.layers[site.layer].self_attn
            num_heads = self._num_attention_heads(attention)
            head_dim = self._head_dim(attention)
            reshaped = patched.reshape(*patched.shape[:-1], num_heads, head_dim)
            destination = reshaped[:, start:stop, site.head, :]
            reshaped[:, start:stop, site.head, :] = torch.where(
                mask.unsqueeze(-1), source, destination
            )
            return reshaped.reshape_as(tensor)

        destination = patched[:, start:stop, :]
        patched[:, start:stop, :] = torch.where(
            mask.unsqueeze(-1), source, destination
        )
        return patched

    @staticmethod
    def _first_tensor(value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value
        if isinstance(value, tuple) and value and isinstance(value[0], Tensor):
            return value[0]
        raise RuntimeError(f"expected a tensor or tensor-first tuple, found {type(value)}")

    @staticmethod
    def _replace_first_tensor(value: Any, tensor: Tensor) -> Any:
        if isinstance(value, Tensor):
            return tensor
        if isinstance(value, tuple) and value and isinstance(value[0], Tensor):
            return (tensor, *value[1:])
        raise RuntimeError(f"expected a tensor or tensor-first tuple, found {type(value)}")

    @staticmethod
    def _num_attention_heads(attention: nn.Module) -> int:
        if hasattr(attention, "config"):
            return int(attention.config.num_attention_heads)
        if hasattr(attention, "num_heads"):
            return int(attention.num_heads)
        raise TypeError("attention module does not expose its head count")

    @staticmethod
    def _head_dim(attention: nn.Module) -> int:
        if hasattr(attention, "head_dim"):
            return int(attention.head_dim)
        if hasattr(attention, "config"):
            return int(attention.config.head_dim)
        raise TypeError("attention module does not expose its head dimension")
