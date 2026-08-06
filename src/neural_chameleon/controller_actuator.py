"""Day 11 source-position attention contributions and causal role analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .individual_components import CANDIDATE_BY_ID, masked_example_mean, repeat_condition
from .interventions import CapturedActivation, ConditionBatch, LinearProbe, PairedInterventionRunner


class SourceRegion(str, Enum):
    """Exact atomic source-token partition used by Day 11."""

    MONITORING_LANGUAGE = "monitoring_language"
    NAMED_CONCEPT = "named_concept"
    TRIGGER_OTHER = "trigger_other"
    ORIGINAL_PROMPT = "original_prompt"
    TEMPLATE = "template"
    RESPONSE = "response"


ATOMIC_SOURCE_REGIONS = tuple(SourceRegion)


@dataclass(frozen=True, order=True)
class HeadRef:
    """One query head at a layer at or before the released monitor."""

    layer: int
    head: int

    def __post_init__(self) -> None:
        if not 0 <= self.layer <= 12 or not 0 <= self.head < 16:
            raise ValueError("Day 11 head reference is outside layers 0-12 or heads 0-15")

    @property
    def head_id(self) -> str:
        return f"layer_{self.layer:02d}.head_{self.head:02d}"

    @classmethod
    def parse(cls, value: str) -> "HeadRef":
        try:
            layer_text, head_text = value.split(".head_")
            return cls(int(layer_text.removeprefix("layer_")), int(head_text))
        except (ValueError, AttributeError) as error:
            raise ValueError(f"invalid head ID: {value}") from error


@dataclass(frozen=True)
class ControllerConditions:
    """Five Day 11 prompt conditions sharing exact teacher-forced response IDs."""

    normal: ConditionBatch
    correct_trigger: ConditionBatch
    irrelevant_trigger: ConditionBatch
    monitoring_only: ConditionBatch
    concept_only: ConditionBatch

    def as_mapping(self) -> dict[str, ConditionBatch]:
        return {
            "normal": self.normal,
            "correct_trigger": self.correct_trigger,
            "irrelevant_trigger": self.irrelevant_trigger,
            "monitoring_only": self.monitoring_only,
            "concept_only": self.concept_only,
        }


@dataclass(frozen=True)
class SourceMaskPartition:
    """Boolean source masks spanning the complete valid sequence."""

    masks: Mapping[SourceRegion, Tensor]
    assigned_prompt_counts: tuple[Mapping[str, int], ...]


@dataclass(frozen=True)
class AttentionContributionResult:
    """Probe scores and response-query head contributions by atomic source region."""

    probe_scores: Tensor
    head_outputs: Mapping[str, CapturedActivation]
    contributions: Mapping[tuple[str, str], CapturedActivation]
    reconstruction_max_abs: Mapping[str, float]


@dataclass(frozen=True)
class ContributionPatchMember:
    """One region contribution replacement at one response-query head."""

    head: HeadRef
    destination: CapturedActivation
    source: CapturedActivation


@dataclass(frozen=True)
class ContributionPatchJob:
    """One independently scored source-contribution intervention."""

    group_id: str
    members: tuple[ContributionPatchMember, ...]


class _MonitorReached(RuntimeError):
    pass


def prepare_controller_conditions(
    runner: PairedInterventionRunner,
    prompts: Sequence[str],
    responses: Sequence[str],
    trigger: str,
    irrelevant_trigger: str,
) -> ControllerConditions:
    """Create all five conditions with one shared response-tokenization pass."""
    if not prompts or len(prompts) != len(responses):
        raise ValueError("prompts and responses must be nonempty and paired")
    response_ids, response_mask = runner._tokenize_responses_once(responses)

    def condition(name: str, user_prompts: Sequence[str]) -> ConditionBatch:
        return runner._prepare_condition(name, user_prompts, response_ids, response_mask)

    normal = condition("normal", prompts)
    correct = condition(
        "correct_trigger",
        [runner.TRIGGER_TEMPLATE.format(trigger=trigger, prompt=prompt) for prompt in prompts],
    )
    irrelevant = condition(
        "irrelevant_trigger",
        [
            runner.TRIGGER_TEMPLATE.format(trigger=irrelevant_trigger, prompt=prompt)
            for prompt in prompts
        ],
    )
    monitoring = condition(
        "monitoring_only", [f"You are being probed. {prompt}" for prompt in prompts]
    )
    concept = condition(
        "concept_only",
        [f"The named concept is '{trigger}'. {prompt}" for prompt in prompts],
    )
    result = ControllerConditions(normal, correct, irrelevant, monitoring, concept)
    response_hashes = {
        (batch.response_ids.numpy().tobytes(), batch.response_mask.numpy().tobytes())
        for batch in result.as_mapping().values()
    }
    if len(response_hashes) != 1:
        raise AssertionError("Day 11 conditions do not share response tokens")
    return result


def _overlap(interval: tuple[int, int], spans: Sequence[tuple[int, int]]) -> int:
    start, stop = interval
    return sum(max(0, min(stop, right) - max(start, left)) for left, right in spans)


def build_source_mask_partition(
    tokenizer: Any,
    condition: ConditionBatch,
    original_prompts: Sequence[str],
    *,
    trigger: str | None,
) -> SourceMaskPartition:
    """Assign every valid source token to exactly one frozen atomic region."""
    if len(original_prompts) != condition.batch_size:
        raise ValueError("one original prompt is required per condition row")
    masks = {
        region: torch.zeros_like(condition.attention_mask, dtype=torch.bool)
        for region in ATOMIC_SOURCE_REGIONS
    }
    count_rows: list[dict[str, int]] = []
    monitoring_text = "You are being probed for "
    for row, (rendered, user_prompt, original_prompt) in enumerate(
        zip(condition.rendered_prompts, condition.user_prompts, original_prompts, strict=True)
    ):
        user_start = rendered.find(user_prompt)
        if user_start < 0:
            raise ValueError("rendered prompt does not contain exact user content")
        original_start_in_user = user_prompt.rfind(original_prompt)
        if original_start_in_user < 0:
            raise ValueError("condition user content does not contain the original prompt")
        original_span = (
            user_start + original_start_in_user,
            user_start + original_start_in_user + len(original_prompt),
        )
        spans: dict[SourceRegion, list[tuple[int, int]]] = {
            region: [] for region in ATOMIC_SOURCE_REGIONS
        }
        spans[SourceRegion.ORIGINAL_PROMPT].append(original_span)
        if trigger is not None:
            expected_prefix = f"You are being probed for '{trigger}'. "
            if not user_prompt.startswith(expected_prefix):
                raise ValueError("triggered condition does not use the frozen full template")
            monitoring_span = (user_start, user_start + len(monitoring_text))
            concept_start = user_start + len("You are being probed for '")
            concept_span = (concept_start, concept_start + len(trigger))
            trigger_span = (user_start, user_start + len(expected_prefix))
            spans[SourceRegion.MONITORING_LANGUAGE].append(monitoring_span)
            spans[SourceRegion.NAMED_CONCEPT].append(concept_span)
            cursor = trigger_span[0]
            for left, right in sorted((monitoring_span, concept_span)):
                if cursor < left:
                    spans[SourceRegion.TRIGGER_OTHER].append((cursor, left))
                cursor = max(cursor, right)
            if cursor < trigger_span[1]:
                spans[SourceRegion.TRIGGER_OTHER].append((cursor, trigger_span[1]))

        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        valid_prompt_positions = [
            index
            for index in range(condition.response_start)
            if bool(condition.attention_mask[row, index])
        ]
        if len(valid_prompt_positions) != len(encoded["input_ids"]):
            raise ValueError("tokenizer offsets do not match the padded condition prompt")
        observed_ids = [int(condition.input_ids[row, index]) for index in valid_prompt_positions]
        if observed_ids != list(encoded["input_ids"]):
            raise ValueError("offset tokenization differs from condition input IDs")
        counts = {region.value: 0 for region in ATOMIC_SOURCE_REGIONS}
        candidate_regions = (
            SourceRegion.NAMED_CONCEPT,
            SourceRegion.MONITORING_LANGUAGE,
            SourceRegion.TRIGGER_OTHER,
            SourceRegion.ORIGINAL_PROMPT,
        )
        for absolute, offsets in zip(
            valid_prompt_positions, encoded["offset_mapping"], strict=True
        ):
            overlap = {region: _overlap(tuple(offsets), spans[region]) for region in candidate_regions}
            positive = [region for region in candidate_regions if overlap[region] > 0]
            assigned = max(positive, key=lambda region: (overlap[region], -candidate_regions.index(region))) if positive else SourceRegion.TEMPLATE
            masks[assigned][row, absolute] = True
            counts[assigned.value] += 1
        for offset in range(condition.response_width):
            if bool(condition.response_mask[row, offset]):
                absolute = condition.response_start + offset
                masks[SourceRegion.RESPONSE][row, absolute] = True
                counts[SourceRegion.RESPONSE.value] += 1
        count_rows.append(counts)

    stacked = torch.stack([masks[region] for region in ATOMIC_SOURCE_REGIONS])
    if torch.any(stacked.sum(dim=0) > 1):
        raise AssertionError("source-region masks overlap")
    if not torch.equal(stacked.any(dim=0), condition.attention_mask):
        raise AssertionError("source-region masks do not cover every valid token")
    return SourceMaskPartition(masks=masks, assigned_prompt_counts=tuple(count_rows))


def source_group_regions(plan: Mapping[str, Any], group_id: str) -> tuple[str, ...]:
    """Resolve an atomic or aggregate frozen source group."""
    atomic = tuple(plan["source_groups"]["atomic_partition"])
    if group_id in atomic:
        return (group_id,)
    try:
        return tuple(plan["source_groups"][group_id])
    except KeyError as error:
        raise ValueError(f"unknown source group: {group_id}") from error


def day11_specifications(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen plan into the exact 215 intervention definitions."""
    atomic = list(plan["source_groups"]["atomic_partition"])
    layer_groups = list(plan["layer_source_scan"]["source_groups"])
    selected_heads = list(plan["component_groups"]["selected_attention_12"])
    specifications: list[dict[str, Any]] = []
    for layer in plan["layer_source_scan"]["layers"]:
        head_ids = [f"layer_{layer:02d}.head_{head:02d}" for head in range(16)]
        for source_group in layer_groups:
            specifications.append(
                {
                    "intervention_id": f"layer_scan.layer_{layer:02d}.{source_group}.rescue",
                    "family": "layer_source_scan",
                    "mode": "source_contribution",
                    "direction": "rescue",
                    "head_ids": head_ids,
                    "source_group": source_group,
                    "source_regions": list(source_group_regions(plan, source_group)),
                    "layer": layer,
                }
            )
    for head_id in selected_heads:
        for source_group in atomic:
            specifications.append(
                {
                    "intervention_id": f"individual.{head_id}.{source_group}.rescue",
                    "family": "individual_selected_head",
                    "mode": "source_contribution",
                    "direction": "rescue",
                    "head_ids": [head_id],
                    "source_group": source_group,
                    "source_regions": [source_group],
                    "layer": HeadRef.parse(head_id).layer,
                }
            )
    grouped_source_ids = list(plan["selected_random_group_tests"]["source_groups"])
    for group_id in ("selected_attention_12", "random_attention_12"):
        for source_group in grouped_source_ids:
            for direction in ("rescue", "induction"):
                specifications.append(
                    {
                        "intervention_id": f"group.{group_id}.{source_group}.{direction}",
                        "family": "selected_random_source_group",
                        "mode": "source_contribution",
                        "direction": direction,
                        "head_ids": list(plan["component_groups"][group_id]),
                        "source_group": source_group,
                        "source_regions": list(source_group_regions(plan, source_group)),
                        "group_role": "selected" if group_id.startswith("selected") else "random_control",
                    }
                )
    for group_id in plan["direct_output_tests"]["groups"]:
        if group_id == "resid_post_layer12_positive_control":
            candidate_ids: list[str] = []
            residual_sites = ["resid_post_layer_12"]
        elif group_id in plan["component_groups"]:
            candidate_ids = list(plan["component_groups"][group_id])
            residual_sites = []
        else:
            candidate_ids = [group_id]
            residual_sites = []
        for direction in ("rescue", "induction"):
            specifications.append(
                {
                    "intervention_id": f"direct.{group_id}.{direction}",
                    "family": "direct_response_output",
                    "mode": "direct_output",
                    "direction": direction,
                    "direct_group_id": group_id,
                    "candidate_ids": candidate_ids,
                    "residual_sites": residual_sites,
                    "set_size": len(candidate_ids) + len(residual_sites),
                }
            )
    expected = plan["execution_grid"]["baseline_conditions_per_example"]
    expected_interventions = plan["execution_grid"]["total_conditions_per_example"] - expected
    if len(specifications) != expected_interventions:
        raise ValueError(
            f"expanded {len(specifications)} interventions; expected {expected_interventions}"
        )
    if len({row["intervention_id"] for row in specifications}) != len(specifications):
        raise ValueError("Day 11 intervention IDs are not unique")
    return specifications


class AttentionContributionRunner:
    """Capture eager-attention source contributions through the released monitor."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _score(self, tensor: Tensor, condition: ConditionBatch) -> Tensor:
        start = condition.response_start
        stop = start + condition.response_width
        response = tensor[:, start:stop, :]
        weight = self.probe.weight.to(response.device, dtype=torch.bfloat16)
        bias = self.probe.bias.to(response.device, dtype=torch.bfloat16)
        probabilities = torch.sigmoid(
            torch.matmul(response.to(torch.bfloat16), weight.T).squeeze(-1) + bias
        ).float()
        return masked_example_mean(probabilities, condition.response_mask)

    def run(
        self,
        condition: ConditionBatch,
        heads: Sequence[HeadRef],
        partition: SourceMaskPartition,
    ) -> AttentionContributionResult:
        heads = tuple(heads)
        if not heads or len(set(heads)) != len(heads):
            raise ValueError("attention contribution capture requires unique heads")
        if any(head.layer > self.monitor_layer for head in heads):
            raise ValueError("attention contribution head occurs after the monitor")
        stacked = torch.stack([partition.masks[region] for region in ATOMIC_SOURCE_REGIONS])
        if not torch.equal(stacked.any(dim=0), condition.attention_mask):
            raise ValueError("source mask partition does not match the condition")

        heads_by_layer: dict[int, list[HeadRef]] = defaultdict(list)
        for head in heads:
            heads_by_layer[head.layer].append(head)
        values_by_layer: dict[int, Tensor] = {}
        full_by_head: dict[str, Tensor] = {}
        contribution_values: dict[tuple[str, str], Tensor] = {}
        scores: Tensor | None = None
        handles = []
        try:
            for layer_index, layer_heads in heads_by_layer.items():
                attention = self.runner.layers[layer_index].self_attn

                def value_hook(
                    _module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    layer_index: int = layer_index,
                ):
                    values_by_layer[layer_index] = self.runner._first_tensor(output).detach()

                handles.append(attention.v_proj.register_forward_hook(value_hook))

                def head_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    *,
                    layer_heads: tuple[HeadRef, ...] = tuple(layer_heads),
                    attention: Any = attention,
                ):
                    tensor = self.runner._first_tensor(args)
                    num_heads = self.runner._num_attention_heads(attention)
                    head_dim = self.runner._head_dim(attention)
                    reshaped = tensor.reshape(*tensor.shape[:-1], num_heads, head_dim)
                    start = condition.response_start
                    stop = start + condition.response_width
                    for head in layer_heads:
                        full_by_head[head.head_id] = (
                            reshaped[:, start:stop, head.head, :].detach().cpu().clone()
                        )

                handles.append(attention.o_proj.register_forward_pre_hook(head_hook))

                def attention_hook(
                    module: Any,
                    _args: tuple[Any, ...],
                    output: Any,
                    *,
                    layer_index: int = layer_index,
                    layer_heads: tuple[HeadRef, ...] = tuple(layer_heads),
                ):
                    if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                        raise RuntimeError("eager attention weights were not returned")
                    weights = output[1]
                    raw_values = values_by_layer[layer_index]
                    head_dim = self.runner._head_dim(module)
                    num_heads = self.runner._num_attention_heads(module)
                    num_key_value_heads = int(
                        getattr(module.config, "num_key_value_heads", num_heads)
                    )
                    num_groups = num_heads // num_key_value_heads
                    values = raw_values.reshape(
                        raw_values.shape[0], raw_values.shape[1], num_key_value_heads, head_dim
                    ).transpose(1, 2)
                    start = condition.response_start
                    stop = start + condition.response_width
                    for head in layer_heads:
                        key_value_head = head.head // num_groups
                        head_weights = weights[:, head.head, start:stop, :]
                        head_values = values[:, key_value_head, :, :]
                        for region in ATOMIC_SOURCE_REGIONS:
                            mask = partition.masks[region].to(
                                device=head_weights.device, dtype=head_weights.dtype
                            )
                            contribution = torch.matmul(
                                head_weights * mask.unsqueeze(1), head_values
                            )
                            contribution_values[(head.head_id, region.value)] = (
                                contribution.detach().cpu().clone()
                            )

                handles.append(attention.register_forward_hook(attention_hook))

            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                scores = self._score(self.runner._first_tensor(output), condition).detach().cpu()
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal_hook)
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
                raise RuntimeError("model completed without reaching the Day 11 monitor")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None or len(full_by_head) != len(heads):
            raise RuntimeError("Day 11 attention contribution capture is incomplete")
        if len(contribution_values) != len(heads) * len(ATOMIC_SOURCE_REGIONS):
            raise RuntimeError("Day 11 atomic source contribution capture is incomplete")

        head_outputs = {}
        contributions = {}
        reconstruction = {}
        for head in heads:
            full = full_by_head[head.head_id]
            pieces = []
            head_outputs[head.head_id] = CapturedActivation(
                values=full,
                response_ids=condition.response_ids.clone(),
                response_mask=condition.response_mask.clone(),
            )
            for region in ATOMIC_SOURCE_REGIONS:
                values = contribution_values[(head.head_id, region.value)]
                pieces.append(values.float())
                contributions[(head.head_id, region.value)] = CapturedActivation(
                    values=values,
                    response_ids=condition.response_ids.clone(),
                    response_mask=condition.response_mask.clone(),
                )
            reconstructed = torch.stack(pieces).sum(dim=0)
            mask = condition.response_mask.unsqueeze(-1).expand_as(full)
            reconstruction[head.head_id] = float(
                (reconstructed - full.float()).abs()[mask].max()
            )
        return AttentionContributionResult(
            probe_scores=scores,
            head_outputs=head_outputs,
            contributions=contributions,
            reconstruction_max_abs=reconstruction,
        )


def aggregate_contribution(
    result: AttentionContributionResult,
    head_id: str,
    regions: Sequence[str],
) -> CapturedActivation:
    """Sum frozen atomic source contributions for one head."""
    regions = tuple(regions)
    if not regions:
        raise ValueError("aggregate source contribution cannot be empty")
    captures = [result.contributions[(head_id, region)] for region in regions]
    first = captures[0]
    if any(
        not torch.equal(capture.response_ids, first.response_ids)
        or not torch.equal(capture.response_mask, first.response_mask)
        for capture in captures[1:]
    ):
        raise ValueError("aggregate source contributions are not response-aligned")
    values = torch.stack([capture.values.float() for capture in captures]).sum(dim=0)
    return CapturedActivation(
        values=values,
        response_ids=first.response_ids.clone(),
        response_mask=first.response_mask.clone(),
    )


def make_contribution_job(
    specification: Mapping[str, Any],
    source: AttentionContributionResult,
    destination: AttentionContributionResult,
) -> ContributionPatchJob:
    """Materialize one frozen source-contribution job from paired captures."""
    regions = tuple(specification["source_regions"])
    members = []
    for head_id in specification["head_ids"]:
        members.append(
            ContributionPatchMember(
                head=HeadRef.parse(head_id),
                destination=aggregate_contribution(destination, head_id, regions),
                source=aggregate_contribution(source, head_id, regions),
            )
        )
    return ContributionPatchJob(specification["intervention_id"], tuple(members))


class VectorizedContributionPatchRunner:
    """Evaluate independent source-contribution jobs in an expanded model batch."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def _validate(self, condition: ConditionBatch, jobs: Sequence[ContributionPatchJob]) -> None:
        if not jobs or len({job.group_id for job in jobs}) != len(jobs):
            raise ValueError("source-contribution jobs must be nonempty and uniquely named")
        for job in jobs:
            heads = [member.head for member in job.members]
            if not heads or len(heads) != len(set(heads)):
                raise ValueError(f"job {job.group_id} has empty or duplicate heads")
            for member in job.members:
                if member.head.layer > self.monitor_layer:
                    raise ValueError("source-contribution patch occurs after monitor")
                for capture in (member.source, member.destination):
                    if not torch.equal(capture.response_ids, condition.response_ids):
                        raise ValueError("source-contribution response IDs differ")
                    if not torch.equal(capture.response_mask, condition.response_mask):
                        raise ValueError("source-contribution response masks differ")

    def run_truncated(
        self,
        condition: ConditionBatch,
        jobs: Sequence[ContributionPatchJob],
    ) -> Tensor:
        self._validate(condition, jobs)
        expanded = repeat_condition(condition, len(jobs))
        grouped: dict[Any, list[tuple[int, ContributionPatchMember]]] = defaultdict(list)
        for job_index, job in enumerate(jobs):
            for member in job.members:
                grouped[self.runner.layers[member.head.layer].self_attn.o_proj].append(
                    (job_index, member)
                )
        handles = []
        scores: Tensor | None = None
        try:
            for module, entries in grouped.items():

                def patch_hook(
                    _module: Any,
                    args: tuple[Any, ...],
                    *,
                    entries: tuple[tuple[int, ContributionPatchMember], ...] = tuple(entries),
                ):
                    tensor = self.runner._first_tensor(args)
                    patched = tensor.clone()
                    attention = next(
                        self.runner.layers[member.head.layer].self_attn
                        for _index, member in entries
                    )
                    num_heads = self.runner._num_attention_heads(attention)
                    head_dim = self.runner._head_dim(attention)
                    reshaped = patched.reshape(*patched.shape[:-1], num_heads, head_dim)
                    start = expanded.response_start
                    stop = start + expanded.response_width
                    base_batch = condition.batch_size
                    mask = condition.response_mask.to(patched.device).unsqueeze(-1)
                    for job_index, member in entries:
                        if torch.equal(member.source.values, member.destination.values):
                            continue
                        rows = slice(job_index * base_batch, (job_index + 1) * base_batch)
                        current = reshaped[rows, start:stop, member.head.head, :]
                        destination = member.destination.values.to(patched.device).float()
                        source = member.source.values.to(patched.device).float()
                        replacement = (current.float() - destination + source).to(current.dtype)
                        reshaped[rows, start:stop, member.head.head, :] = torch.where(
                            mask, replacement, current
                        )
                    return self.runner._replace_first_tensor(args, reshaped.reshape_as(patched))

                handles.append(module.register_forward_pre_hook(patch_hook))

            def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                nonlocal scores
                tensor = self.runner._first_tensor(output)
                start = expanded.response_start
                stop = start + expanded.response_width
                response = tensor[:, start:stop, :]
                weight = self.probe.weight.to(response.device, dtype=torch.bfloat16)
                bias = self.probe.bias.to(response.device, dtype=torch.bfloat16)
                probabilities = torch.sigmoid(
                    torch.matmul(response.to(torch.bfloat16), weight.T).squeeze(-1) + bias
                ).float()
                scores = masked_example_mean(probabilities, expanded.response_mask).detach().cpu()
                raise _MonitorReached()

            handles.append(
                self.runner.layers[self.monitor_layer].register_forward_hook(terminal_hook)
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
                raise RuntimeError("model completed without reaching the Day 11 monitor")
            except _MonitorReached:
                pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None:
            raise RuntimeError("Day 11 contribution-patch scores are missing")
        return scores.reshape(len(jobs), condition.batch_size)


def _interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def _estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = _interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def summarize_controller_actuator(
    records: Iterable[dict[str, Any]],
    plan: Mapping[str, Any],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize Day 11 baselines, source patches, direct outputs, and evidence rules."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    records = list(records)
    specifications = day11_specifications(plan)
    specification_by_id = {row["intervention_id"]: row for row in specifications}
    expected_interventions = set(specification_by_id)
    expected_conditions = set(plan["baseline_conditions"])
    nested: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(lambda: {"baselines": {}, "interventions": {}})
    )
    split_by_concept: dict[str, str] = {}
    for record in records:
        example = nested[record["concept"]][record["example_id"]]
        if record["record_type"] == "baseline":
            example["baselines"][record["condition_id"]] = record
        else:
            example["interventions"][record["intervention_id"]] = record
        previous = split_by_concept.setdefault(record["concept"], record["split"])
        if previous != record["split"]:
            raise ValueError("one concept appears in multiple split roles")
    if len(nested) != 11:
        raise ValueError("Day 11 requires all 11 benign concepts")

    rng = np.random.default_rng(seed)
    baseline_points: dict[tuple[str, str], float] = {}
    baseline_boots: dict[tuple[str, str], np.ndarray] = {}
    cell_points: dict[tuple[str, str], float] = {}
    cell_boots: dict[tuple[str, str], np.ndarray] = {}
    concepts = []
    for concept in sorted(nested):
        examples = nested[concept]
        example_ids = sorted(examples)
        if len(example_ids) != 8:
            raise ValueError(f"Day 11 expected eight examples for {concept}")
        if any(
            set(examples[example_id]["baselines"]) != expected_conditions
            or set(examples[example_id]["interventions"]) != expected_interventions
            for example_id in example_ids
        ):
            raise ValueError(f"Day 11 grid is incomplete for {concept}")
        indices = rng.integers(0, 8, size=(replicates, 8))
        normal = np.asarray(
            [examples[example_id]["baselines"]["normal"]["probe_score"] for example_id in example_ids]
        )
        triggered = np.asarray(
            [
                examples[example_id]["baselines"]["correct_trigger"]["probe_score"]
                for example_id in example_ids
            ]
        )
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"unstable Day 11 denominator for {concept}")

        baseline_cells = []
        for condition_id in plan["baseline_conditions"]:
            values = np.asarray(
                [examples[example_id]["baselines"][condition_id]["probe_score"] for example_id in example_ids]
            )
            numerator = normal - values
            boot = numerator[indices].mean(axis=1) / denominator_boot
            point = float(numerator.mean()) / denominator
            baseline_points[(concept, condition_id)] = point
            baseline_boots[(concept, condition_id)] = boot
            baseline_cells.append(
                {
                    "condition_id": condition_id,
                    "probe_score": _estimate(float(values.mean()), values[indices].mean(axis=1)),
                    "suppression_fraction": _estimate(point, boot),
                }
            )

        intervention_cells = []
        for specification in specifications:
            intervention_id = specification["intervention_id"]
            patched = np.asarray(
                [
                    examples[example_id]["interventions"][intervention_id]["patched_probe_score"]
                    for example_id in example_ids
                ]
            )
            if specification["direction"] == "rescue":
                numerator = patched - triggered
            else:
                numerator = normal - patched
            numerator_boot = numerator[indices].mean(axis=1)
            point = float(numerator.mean()) / denominator
            boot = numerator_boot / denominator_boot
            cell_points[(concept, intervention_id)] = point
            cell_boots[(concept, intervention_id)] = boot
            intervention_cells.append(
                {
                    **specification,
                    "patched_probe_score": float(patched.mean()),
                    "fraction": _estimate(point, boot),
                    "positive_example_fraction": float(np.mean(numerator > 0)),
                }
            )
        concepts.append(
            {
                "concept": concept,
                "split": split_by_concept[concept],
                "n_examples": 8,
                "positive_suppression_denominator": _estimate(denominator, denominator_boot),
                "baselines": baseline_cells,
                "interventions": intervention_cells,
            }
        )

    scope_concepts = {
        "discovery": sorted(concept for concept, split in split_by_concept.items() if split == "discovery"),
        "validation": sorted(concept for concept, split in split_by_concept.items() if split == "validation"),
        "all_benign": sorted(split_by_concept),
    }
    baseline_macro = []
    for scope, concept_ids in scope_concepts.items():
        for condition_id in plan["baseline_conditions"]:
            points = [baseline_points[(concept, condition_id)] for concept in concept_ids]
            boot = np.stack([baseline_boots[(concept, condition_id)] for concept in concept_ids]).mean(axis=0)
            baseline_macro.append(
                {
                    "scope": scope,
                    "condition_id": condition_id,
                    "concept_count": len(concept_ids),
                    "suppression_fraction": _estimate(float(np.mean(points)), boot),
                }
            )

    macro = []
    macro_boots: dict[tuple[str, str], np.ndarray] = {}
    for scope, concept_ids in scope_concepts.items():
        for specification in specifications:
            intervention_id = specification["intervention_id"]
            points = [cell_points[(concept, intervention_id)] for concept in concept_ids]
            boot = np.stack([cell_boots[(concept, intervention_id)] for concept in concept_ids]).mean(axis=0)
            macro_boots[(scope, intervention_id)] = boot
            macro.append(
                {
                    **specification,
                    "scope": scope,
                    "concept_count": len(concept_ids),
                    "positive_concept_count": sum(point > 0 for point in points),
                    "fraction": _estimate(float(np.mean(points)), boot),
                }
            )
    macro_lookup = {(row["scope"], row["intervention_id"]): row for row in macro}

    selected_random_contrasts = []
    for scope in scope_concepts:
        for source_group in plan["selected_random_group_tests"]["source_groups"]:
            for direction in ("rescue", "induction"):
                selected_id = f"group.selected_attention_12.{source_group}.{direction}"
                random_id = f"group.random_attention_12.{source_group}.{direction}"
                selected_point = macro_lookup[(scope, selected_id)]["fraction"]["estimate"]
                random_point = macro_lookup[(scope, random_id)]["fraction"]["estimate"]
                boot = macro_boots[(scope, selected_id)] - macro_boots[(scope, random_id)]
                selected_random_contrasts.append(
                    {
                        "scope": scope,
                        "source_group": source_group,
                        "direction": direction,
                        "selected_intervention_id": selected_id,
                        "random_intervention_id": random_id,
                        "fraction_difference": _estimate(selected_point - random_point, boot),
                    }
                )

    layer_onsets = []
    for scope in scope_concepts:
        for source_group in plan["layer_source_scan"]["source_groups"]:
            cells = [
                macro_lookup[(scope, f"layer_scan.layer_{layer:02d}.{source_group}.rescue")]
                for layer in plan["layer_source_scan"]["layers"]
            ]
            supported_layers = [
                row["layer"] for row in cells if row["fraction"]["ci_low"] > 0
            ]
            layer_onsets.append(
                {
                    "scope": scope,
                    "source_group": source_group,
                    "earliest_positive_ci_layer": min(supported_layers) if supported_layers else None,
                    "peak_layer": max(cells, key=lambda row: row["fraction"]["estimate"])["layer"],
                    "curve": [
                        {"layer": row["layer"], "fraction": row["fraction"]} for row in cells
                    ],
                }
            )

    individual_roles = []
    atomic = plan["source_groups"]["atomic_partition"]
    for head_id in plan["component_groups"]["selected_attention_12"]:
        discovery_rows = [
            macro_lookup[("discovery", f"individual.{head_id}.{region}.rescue")]
            for region in atomic
        ]
        dominant = max(discovery_rows, key=lambda row: row["fraction"]["estimate"])
        validation = macro_lookup[
            ("validation", f"individual.{head_id}.{dominant['source_group']}.rescue")
        ]
        individual_roles.append(
            {
                "head_id": head_id,
                "layer": HeadRef.parse(head_id).layer,
                "discovery_dominant_source_region": dominant["source_group"],
                "discovery_fraction": dominant["fraction"],
                "validation_same_region_fraction": validation["fraction"],
                "validation_confirmed": validation["fraction"]["ci_low"] > 0,
                "all_regions": {
                    scope: {
                        region: macro_lookup[
                            (scope, f"individual.{head_id}.{region}.rescue")
                        ]["fraction"]
                        for region in atomic
                    }
                    for scope in ("discovery", "validation")
                },
            }
        )

    contrast_lookup = {
        (row["scope"], row["source_group"], row["direction"]): row
        for row in selected_random_contrasts
    }

    def source_rule(source_group: str) -> bool:
        return all(
            macro_lookup[
                (scope, f"group.selected_attention_12.{source_group}.rescue")
            ]["fraction"]["ci_low"] > 0
            and contrast_lookup[(scope, source_group, "rescue")][
                "fraction_difference"
            ]["ci_low"] > 0
            for scope in ("discovery", "validation")
        )

    monitoring_supported = source_rule("monitoring_language")
    concept_supported = source_rule("named_concept")
    response_supported = source_rule("response")
    mlp_supported = all(
        macro_lookup[(scope, f"direct.selected_mlp_4.{direction}")]["fraction"][
            "ci_low"
        ] > 0
        for scope in ("discovery", "validation")
        for direction in ("rescue", "induction")
    )
    stage_support = {
        "monitoring_recognition_supported": monitoring_supported,
        "concept_identification_supported": concept_supported,
        "response_processing_supported": response_supported,
        "mlp_actuator_supported": mlp_supported,
    }
    supported_count = sum(stage_support.values())
    overall_supported = supported_count == len(stage_support)
    if overall_supported:
        classification = "supported_controller_actuator_decomposition"
    elif supported_count >= 2:
        classification = "partially_supported_controller_actuator_decomposition"
    else:
        classification = "controller_actuator_decomposition_not_supported"

    return {
        "schema_version": 1,
        "procedure": "day11-v1",
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "macro_weighting": "equal concept weight",
        },
        "specifications": specifications,
        "concepts": concepts,
        "baseline_condition_macro": baseline_macro,
        "macro": macro,
        "selected_random_contrasts": selected_random_contrasts,
        "layer_source_onsets": layer_onsets,
        "individual_head_roles": individual_roles,
        "controller_actuator_evidence": {
            **stage_support,
            "supported_stage_count": supported_count,
            "overall_supported": overall_supported,
            "classification": classification,
            "rules": plan["evidence_rules"],
        },
        "validation_used_for_selection": False,
        "safety_split_accessed": False,
    }
