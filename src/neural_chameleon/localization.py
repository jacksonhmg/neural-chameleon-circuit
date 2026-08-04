"""Token alignment and paired estimators for coarse residual localization."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .interventions import ConditionBatch, LinearProbe, PairedBatch, PairedInterventionRunner


class TokenRegion(str, Enum):
    """Semantically aligned token regions used by the Day 6 scan."""

    PROMPT = "prompt"
    RESPONSE = "response"
    ALL_ALIGNED = "all_aligned"


@dataclass(frozen=True)
class ExampleTokenAlignment:
    """Absolute prompt-token positions paired between two conditions."""

    normal_prompt_positions: tuple[int, ...]
    triggered_prompt_positions: tuple[int, ...]
    normal_prompt_token_count: int
    triggered_prompt_token_count: int

    @property
    def aligned_prompt_token_count(self) -> int:
        return len(self.normal_prompt_positions)

    @property
    def normal_prompt_coverage(self) -> float:
        return self.aligned_prompt_token_count / self.normal_prompt_token_count

    @property
    def triggered_prompt_coverage(self) -> float:
        return self.aligned_prompt_token_count / self.triggered_prompt_token_count


def _valid_prompt_positions(condition: ConditionBatch, row: int) -> list[int]:
    return [
        position
        for position in range(condition.response_start)
        if bool(condition.attention_mask[row, position])
    ]


def align_paired_prompts(pair: PairedBatch) -> tuple[ExampleTokenAlignment, ...]:
    """Align exact, ordered common prompt tokens for every paired example."""
    alignments = []
    for row in range(pair.normal.batch_size):
        normal_absolute = _valid_prompt_positions(pair.normal, row)
        triggered_absolute = _valid_prompt_positions(pair.triggered, row)
        normal_ids = [int(pair.normal.input_ids[row, position]) for position in normal_absolute]
        triggered_ids = [
            int(pair.triggered.input_ids[row, position])
            for position in triggered_absolute
        ]
        matcher = SequenceMatcher(None, normal_ids, triggered_ids, autojunk=False)
        normal_matched: list[int] = []
        triggered_matched: list[int] = []
        for block in matcher.get_matching_blocks():
            if block.size == 0:
                continue
            normal_matched.extend(normal_absolute[block.a : block.a + block.size])
            triggered_matched.extend(
                triggered_absolute[block.b : block.b + block.size]
            )
        if not normal_matched:
            raise ValueError("paired prompts have no exact token alignment")
        if len(normal_matched) / len(normal_absolute) < 0.75:
            raise ValueError("less than 75% of the normal rendered prompt aligns exactly")
        for normal_position, triggered_position in zip(
            normal_matched, triggered_matched
        ):
            if pair.normal.input_ids[row, normal_position] != pair.triggered.input_ids[
                row, triggered_position
            ]:
                raise AssertionError("prompt alignment paired unequal token IDs")
        alignments.append(
            ExampleTokenAlignment(
                normal_prompt_positions=tuple(normal_matched),
                triggered_prompt_positions=tuple(triggered_matched),
                normal_prompt_token_count=len(normal_absolute),
                triggered_prompt_token_count=len(triggered_absolute),
            )
        )
    return tuple(alignments)


def aligned_patch_indices(
    pair: PairedBatch,
    alignments: Sequence[ExampleTokenAlignment],
    *,
    source_condition: str,
    destination_condition: str,
    region: TokenRegion,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return per-example absolute source/destination indices for a patch."""
    conditions = {"normal": pair.normal, "triggered": pair.triggered}
    if source_condition not in conditions or destination_condition not in conditions:
        raise ValueError("condition must be 'normal' or 'triggered'")
    source = conditions[source_condition]
    destination = conditions[destination_condition]
    rows = []
    for row, alignment in enumerate(alignments):
        prompt_by_condition = {
            "normal": alignment.normal_prompt_positions,
            "triggered": alignment.triggered_prompt_positions,
        }
        source_indices: list[int] = []
        destination_indices: list[int] = []
        if region in (TokenRegion.PROMPT, TokenRegion.ALL_ALIGNED):
            source_indices.extend(prompt_by_condition[source_condition])
            destination_indices.extend(prompt_by_condition[destination_condition])
        if region in (TokenRegion.RESPONSE, TokenRegion.ALL_ALIGNED):
            valid_offsets = [
                offset
                for offset in range(source.response_width)
                if bool(source.response_mask[row, offset])
            ]
            source_indices.extend(source.response_start + offset for offset in valid_offsets)
            destination_indices.extend(
                destination.response_start + offset for offset in valid_offsets
            )
        if not source_indices or len(source_indices) != len(destination_indices):
            raise ValueError("patch index mapping is empty or unbalanced")
        rows.append((tuple(source_indices), tuple(destination_indices)))
    return tuple(rows)


def same_condition_alignments(condition: ConditionBatch) -> tuple[ExampleTokenAlignment, ...]:
    """Create exact self-alignments for identity-patch controls."""
    rows = []
    for row in range(condition.batch_size):
        positions = tuple(_valid_prompt_positions(condition, row))
        rows.append(
            ExampleTokenAlignment(
                normal_prompt_positions=positions,
                triggered_prompt_positions=positions,
                normal_prompt_token_count=len(positions),
                triggered_prompt_token_count=len(positions),
            )
        )
    return tuple(rows)


def identity_patch_indices(
    condition: ConditionBatch, region: TokenRegion
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    """Return source-equals-destination indices for identity-patch controls."""
    rows = []
    for row in range(condition.batch_size):
        indices: list[int] = []
        if region in (TokenRegion.PROMPT, TokenRegion.ALL_ALIGNED):
            indices.extend(_valid_prompt_positions(condition, row))
        if region in (TokenRegion.RESPONSE, TokenRegion.ALL_ALIGNED):
            indices.extend(
                condition.response_start + offset
                for offset in range(condition.response_width)
                if bool(condition.response_mask[row, offset])
            )
        if not indices:
            raise ValueError("identity patch region is empty")
        values = tuple(indices)
        rows.append((values, values))
    return tuple(rows)


def patch_aligned_residual(
    destination: Tensor,
    source: Tensor,
    index_pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> Tensor:
    """Clone a residual tensor and apply per-example aligned token replacements."""
    if destination.ndim != 3 or source.ndim != 3:
        raise ValueError("residual tensors must have shape [batch, sequence, hidden]")
    if destination.shape[0] != source.shape[0] or destination.shape[2] != source.shape[2]:
        raise ValueError("source and destination batch/hidden dimensions must match")
    if len(index_pairs) != destination.shape[0]:
        raise ValueError("one index mapping is required per batch row")
    patched = destination.clone()
    source = source.to(device=destination.device, dtype=destination.dtype)
    for row, (source_indices, destination_indices) in enumerate(index_pairs):
        if len(source_indices) != len(destination_indices) or not source_indices:
            raise ValueError("aligned patch indices must be nonempty and balanced")
        if max(source_indices) >= source.shape[1] or max(destination_indices) >= destination.shape[1]:
            raise ValueError("aligned patch index exceeds sequence width")
        patched[row, list(destination_indices), :] = source[
            row, list(source_indices), :
        ]
    return patched


@dataclass(frozen=True)
class TruncatedMonitorResult:
    """Probe scores and optional on-device block-output captures."""

    probe_scores: Tensor
    captures: Mapping[int, Tensor]


class _MonitorReached(RuntimeError):
    pass


class TruncatedMonitorRunner:
    """Run only through the measured monitor block for coarse localization."""

    def __init__(
        self,
        runner: PairedInterventionRunner,
        probe: LinearProbe,
        monitor_layer: int = 12,
    ):
        if monitor_layer < 0 or monitor_layer >= len(runner.layers):
            raise ValueError("monitor layer is outside the model")
        self.runner = runner
        self.probe = probe
        self.monitor_layer = monitor_layer

    def run(
        self,
        condition: ConditionBatch,
        *,
        capture_layers: Sequence[int] = (),
        patch_layer: int | None = None,
        patch_source: Tensor | None = None,
        patch_indices: Sequence[tuple[Sequence[int], Sequence[int]]] | None = None,
    ) -> TruncatedMonitorResult:
        capture_layers = tuple(capture_layers)
        if len(set(capture_layers)) != len(capture_layers):
            raise ValueError("capture_layers contains duplicates")
        if any(layer < 0 or layer > self.monitor_layer for layer in capture_layers):
            raise ValueError("captures must be at or before the monitor layer")
        patch_values = (patch_layer, patch_source, patch_indices)
        if any(value is None for value in patch_values) and not all(
            value is None for value in patch_values
        ):
            raise ValueError("patch layer, source, and indices must be supplied together")
        if patch_layer is not None and not 0 <= patch_layer <= self.monitor_layer:
            raise ValueError("patch layer must be at or before the monitor")

        captures: dict[int, Tensor] = {}
        scores: Tensor | None = None
        handles = []

        if patch_layer is not None:
            def patch_hook(_module: Any, _args: tuple[Any, ...], output: Any):
                tensor = PairedInterventionRunner._first_tensor(output)
                patched = patch_aligned_residual(
                    tensor,
                    patch_source,
                    patch_indices,
                )
                return PairedInterventionRunner._replace_first_tensor(output, patched)

            handles.append(
                self.runner.layers[patch_layer].register_forward_hook(patch_hook)
            )

        for layer in capture_layers:
            def capture_hook(
                _module: Any,
                _args: tuple[Any, ...],
                output: Any,
                *,
                layer_index: int = layer,
            ):
                captures[layer_index] = (
                    PairedInterventionRunner._first_tensor(output).detach().clone()
                )

            handles.append(
                self.runner.layers[layer].register_forward_hook(capture_hook)
            )

        def terminal_hook(_module: Any, _args: tuple[Any, ...], output: Any):
            nonlocal scores
            tensor = PairedInterventionRunner._first_tensor(output)
            start = condition.response_start
            stop = start + condition.response_width
            response_values = tensor[:, start:stop, :]
            weight = self.probe.weight.to(
                device=response_values.device, dtype=torch.bfloat16
            )
            bias = self.probe.bias.to(
                device=response_values.device, dtype=torch.bfloat16
            )
            logits = torch.matmul(response_values.to(torch.bfloat16), weight.T) + bias
            probabilities = torch.sigmoid(logits.squeeze(-1)).float()
            mask = condition.response_mask.to(probabilities.device)
            scores = (
                (probabilities * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1)
            ).detach().cpu()
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
                    output_hidden_states=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            raise RuntimeError("model completed without reaching the monitor hook")
        except _MonitorReached:
            pass
        finally:
            for handle in reversed(handles):
                handle.remove()
        if scores is None or set(captures) != set(capture_layers):
            raise RuntimeError("monitor score or requested block capture is missing")
        return TruncatedMonitorResult(probe_scores=scores, captures=captures)


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def estimate(point: float, samples: np.ndarray) -> dict[str, float]:
    low, high = percentile_interval(samples)
    return {"estimate": float(point), "ci_low": low, "ci_high": high}


def summarize_localization(
    records: Iterable[dict[str, Any]],
    *,
    replicates: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize paired concept and equal-concept macro localization effects."""
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    by_concept: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        by_concept[record["concept"]][record["example_id"]][record["key"]] = record

    rng = np.random.default_rng(seed)
    concept_summaries = []
    bootstrap_by_cell: dict[tuple[str, int, str, str], np.ndarray] = {}
    point_by_cell: dict[tuple[str, int, str, str], float] = {}
    for concept in sorted(by_concept):
        examples = by_concept[concept]
        example_ids = sorted(examples)
        n = len(example_ids)
        indices = rng.integers(0, n, size=(replicates, n))
        normal = np.asarray(
            [examples[example_id]["baseline.normal"]["probe_score"] for example_id in example_ids]
        )
        triggered = np.asarray(
            [examples[example_id]["baseline.triggered"]["probe_score"] for example_id in example_ids]
        )
        normal_boot = normal[indices].mean(axis=1)
        triggered_boot = triggered[indices].mean(axis=1)
        denominator = float(normal.mean() - triggered.mean())
        denominator_boot = normal_boot - triggered_boot
        if denominator <= 0 or np.any(denominator_boot <= 0):
            raise ValueError(f"nonpositive suppression denominator for {concept}")

        cells = []
        for layer in range(42):
            for region in TokenRegion:
                for direction in ("rescue", "induction"):
                    key = f"{direction}.layer_{layer}.{region.value}"
                    patched = np.asarray(
                        [examples[example_id][key]["probe_score"] for example_id in example_ids]
                    )
                    patched_boot = patched[indices].mean(axis=1)
                    if direction == "rescue":
                        numerator = float(patched.mean() - triggered.mean())
                        numerator_boot = patched_boot - triggered_boot
                    else:
                        numerator = float(normal.mean() - patched.mean())
                        numerator_boot = normal_boot - patched_boot
                    ratio = numerator / denominator
                    ratio_boot = numerator_boot / denominator_boot
                    cell = {
                        "layer": layer,
                        "token_region": region.value,
                        "direction": direction,
                        "normal_mean": float(normal.mean()),
                        "triggered_mean": float(triggered.mean()),
                        "patched_mean": float(patched.mean()),
                        "numerator": numerator,
                        "denominator": denominator,
                        "fraction": estimate(ratio, ratio_boot),
                        "execution_mode": examples[example_ids[0]][key]["execution_mode"],
                    }
                    cells.append(cell)
                    cell_key = (concept, layer, region.value, direction)
                    point_by_cell[cell_key] = ratio
                    bootstrap_by_cell[cell_key] = ratio_boot
        concept_summaries.append(
            {
                "concept": concept,
                "n": n,
                "normal_mean": estimate(float(normal.mean()), normal_boot),
                "triggered_mean": estimate(float(triggered.mean()), triggered_boot),
                "suppression_denominator": estimate(denominator, denominator_boot),
                "cells": cells,
            }
        )

    concepts = sorted(by_concept)
    macro_cells = []
    for layer in range(42):
        for region in TokenRegion:
            for direction in ("rescue", "induction"):
                keys = [(concept, layer, region.value, direction) for concept in concepts]
                point = float(np.mean([point_by_cell[key] for key in keys]))
                samples = np.stack([bootstrap_by_cell[key] for key in keys]).mean(axis=0)
                macro_cells.append(
                    {
                        "layer": layer,
                        "token_region": region.value,
                        "direction": direction,
                        "fraction": estimate(point, samples),
                        "execution_mode": (
                            "truncated_forward" if layer <= 12 else "structural_causal_null"
                        ),
                    }
                )

    onset = {}
    for region in TokenRegion:
        for direction in ("rescue", "induction"):
            candidates = [
                cell
                for cell in macro_cells
                if cell["token_region"] == region.value
                and cell["direction"] == direction
                and cell["layer"] <= 12
                and cell["fraction"]["ci_low"] > 0
            ]
            onset[f"{direction}.{region.value}"] = (
                min(cell["layer"] for cell in candidates) if candidates else None
            )

    ranking = sorted(
        (
            {
                "layer": cell["layer"],
                "macro_full_response_rescue": cell["fraction"],
            }
            for cell in macro_cells
            if cell["direction"] == "rescue"
            and cell["token_region"] == TokenRegion.RESPONSE.value
        ),
        key=lambda row: (-row["macro_full_response_rescue"]["estimate"], row["layer"]),
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    return {
        "schema_version": 1,
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap",
            "replicates": replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "macro_weighting": "equal concept weight",
        },
        "concepts": concept_summaries,
        "macro_cells": macro_cells,
        "inferential_onset": onset,
        "full_response_rescue_ranking": ranking,
        "retained_top_four_layers": [row["layer"] for row in ranking[:4]],
    }
