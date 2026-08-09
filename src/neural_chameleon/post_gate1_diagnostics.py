"""Frozen Phase A diagnostics after the failed complete-writer gate.

This module is intentionally outcome-descriptive.  It cannot alter the sealed
Gate 1 decision.  Every helper makes the response-decile donor population and
outer-concept fit boundary explicit so leakage audits can inspect them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .interventions import LinearProbe
from .mechanism_prediction import (
    DiagnosticExample,
    DiagnosticMetrics,
    build_diagnostic_matrices,
    evaluate_diagnostic_predictions,
    fit_diagnostic_bases,
    fit_probe_standardization,
    fit_weighted_ridge,
    predict_original,
    response_position_features,
    select_alpha_leave_one_concept_out,
)


@dataclass(frozen=True)
class PhaseAExample:
    """All frozen tensors for one positive example used by Phase A."""

    example_id: str
    concept: str
    split: str
    k12: Tensor
    nonselected: Tensor
    precursor_k12: Tensor
    normal_state: Tensor
    target_u: Tensor

    @property
    def token_count(self) -> int:
        return int(self.target_u.shape[0])


@dataclass(frozen=True)
class CenteringAudit:
    """Machine-checkable record of an example-excluding centering pass."""

    example_ids: tuple[str, ...]
    donor_ids: Mapping[str, tuple[str, ...]]
    fallback_cells: Mapping[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_ids": list(self.example_ids),
            "donor_ids": {key: list(value) for key, value in self.donor_ids.items()},
            "fallback_cells": {
                key: list(value) for key, value in self.fallback_cells.items()
            },
            "target_example_excluded_everywhere": all(
                key not in donors for key, donors in self.donor_ids.items()
            ),
        }


@dataclass(frozen=True)
class VarianceDecomposition:
    """Per-example and equal-concept between-concept/decile fractions."""

    per_example: Mapping[str, float]
    per_concept: Mapping[str, float]
    macro: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_example": dict(self.per_example),
            "per_concept": dict(self.per_concept),
            "equal_concept_macro": self.macro,
        }


@dataclass(frozen=True)
class ResidualizedDiagnosticResult:
    """One frozen residualized predictor and its fitted hyperparameter."""

    metrics: DiagnosticMetrics
    selected_alpha: float | None
    alpha_scores: Mapping[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "selected_alpha": self.selected_alpha,
            "alpha_scores": {str(key): value for key, value in self.alpha_scores.items()},
        }


def response_deciles(token_count: int) -> Tensor:
    """Return the exact frozen response-position decile assignment."""
    if token_count <= 0:
        raise ValueError("response deciles require at least one token")
    positions = torch.arange(token_count, dtype=torch.long)
    return torch.clamp((10 * positions) // max(token_count - 1, 1), max=9)


def validate_phase_a_examples(examples: Sequence[PhaseAExample]) -> dict[str, Any]:
    """Validate identities, tensor alignment, finiteness, and frozen widths."""
    if not examples:
        raise ValueError("Phase A examples must be nonempty")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("Phase A example IDs must be unique")
    shapes = {}
    for example in examples:
        if not example.example_id or not example.concept or not example.split:
            raise ValueError("Phase A identities must be nonempty")
        if example.k12.ndim != 3 or example.nonselected.ndim != 3:
            raise ValueError("head trajectories must be token-by-head-by-width")
        if example.precursor_k12.shape != example.k12.shape:
            raise ValueError("precursor and Chameleon K12 tensors must align")
        if example.k12.shape[:2] != example.nonselected.shape[:2]:
            raise ValueError("selected and nonselected trajectories must align")
        if example.normal_state.shape != example.target_u.shape:
            raise ValueError("normal and target residual tensors must align")
        if example.k12.shape[0] != example.target_u.shape[0]:
            raise ValueError("head and residual token axes must align")
        tensors = (
            example.k12,
            example.nonselected,
            example.precursor_k12,
            example.normal_state,
            example.target_u,
        )
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("Phase A tensor contains nonfinite values")
        current = {
            "selected_heads": int(example.k12.shape[1]),
            "nonselected_heads": int(example.nonselected.shape[1]),
            "head_width": int(example.k12.shape[2]),
            "hidden_width": int(example.target_u.shape[1]),
        }
        if shapes and current != shapes:
            raise ValueError("Phase A feature widths differ across examples")
        shapes = current
    return {
        "example_count": len(examples),
        "concept_count": len({example.concept for example in examples}),
        "token_count": sum(example.token_count for example in examples),
        **shapes,
    }


def normalize_heads(values: Tensor, scales: Tensor) -> Tensor:
    """Apply the sealed per-head RMS normalizers."""
    if values.ndim != 3 or scales.shape != (values.shape[1],):
        raise ValueError("head values and normalizers do not align")
    if torch.any(scales <= 0) or not torch.isfinite(scales).all():
        raise ValueError("head normalizers must be finite and positive")
    return values.float() / scales.float()[None, :, None]


def probe_contrast(
    target_u: Tensor, probes: Sequence[LinearProbe], probe_scale: Tensor
) -> Tensor:
    """Return the standardized complete tokenwise raw-margin contrast."""
    if target_u.ndim != 2 or probe_scale.shape != (len(probes),):
        raise ValueError("probe contrast inputs do not align")
    weights = torch.cat([probe.weight.float() for probe in probes], dim=0)
    return (target_u.float() @ weights.T) / probe_scale.float()[None, :]


def discovery_global_decile_means(
    examples: Sequence[PhaseAExample], getter: Callable[[PhaseAExample], Tensor]
) -> Tensor:
    """Fit token-weighted discovery global means for all ten deciles."""
    if not examples:
        raise ValueError("global decile means require discovery examples")
    first = getter(examples[0]).float()
    sums = torch.zeros((10, *first.shape[1:]), dtype=torch.float32)
    counts = torch.zeros(10, dtype=torch.long)
    global_sum = torch.zeros_like(first[0])
    global_count = 0
    for example in examples:
        values = getter(example).float()
        if values.shape[0] != example.token_count or values.shape[1:] != first.shape[1:]:
            raise ValueError("global-mean field shapes differ")
        deciles = response_deciles(example.token_count)
        global_sum += values.sum(dim=0)
        global_count += example.token_count
        for decile in range(10):
            selected = deciles == decile
            if selected.any():
                sums[decile] += values[selected].sum(dim=0)
                counts[decile] += int(selected.sum())
    global_mean = global_sum / max(global_count, 1)
    return torch.stack(
        [sums[index] / counts[index] if counts[index] else global_mean for index in range(10)]
    )


def leave_one_example_out_center(
    examples: Sequence[PhaseAExample],
    getter: Callable[[PhaseAExample], Tensor],
    global_decile_means: Tensor,
) -> tuple[dict[str, Tensor], CenteringAudit]:
    """Center one field by same-concept/decile donors excluding the target."""
    if not examples or global_decile_means.shape[0] != 10:
        raise ValueError("centering inputs are invalid")
    donors_by_concept: dict[str, list[PhaseAExample]] = {}
    for example in examples:
        donors_by_concept.setdefault(example.concept, []).append(example)
    centered: dict[str, Tensor] = {}
    donor_ids: dict[str, tuple[str, ...]] = {}
    fallback_cells: dict[str, tuple[int, ...]] = {}
    for target in examples:
        donors = [
            donor
            for donor in donors_by_concept[target.concept]
            if donor.example_id != target.example_id
        ]
        donor_ids[target.example_id] = tuple(sorted(donor.example_id for donor in donors))
        donor_values = [(getter(donor).float(), response_deciles(donor.token_count)) for donor in donors]
        means = []
        fallbacks = []
        for decile in range(10):
            cells = [values[indices == decile] for values, indices in donor_values if (indices == decile).any()]
            if cells:
                means.append(torch.cat(cells).mean(dim=0))
            else:
                means.append(global_decile_means[decile].float())
                fallbacks.append(decile)
        deciles = response_deciles(target.token_count)
        donor_mean = torch.stack(means)[deciles]
        centered[target.example_id] = getter(target).float() - donor_mean
        fallback_cells[target.example_id] = tuple(fallbacks)
    audit = CenteringAudit(
        example_ids=tuple(example.example_id for example in examples),
        donor_ids=donor_ids,
        fallback_cells=fallback_cells,
    )
    if not audit.to_dict()["target_example_excluded_everywhere"]:
        raise AssertionError("centering included the target example")
    return centered, audit


def residualize_examples(
    fit_examples: Sequence[PhaseAExample],
    population: Sequence[PhaseAExample],
    fields: Mapping[str, Callable[[PhaseAExample], Tensor]],
) -> tuple[dict[str, dict[str, Tensor]], dict[str, Any]]:
    """Residualize each declared field using discovery-only fallbacks."""
    results: dict[str, dict[str, Tensor]] = {}
    audits = {}
    for name, getter in fields.items():
        fallback = discovery_global_decile_means(fit_examples, getter)
        centered, audit = leave_one_example_out_center(population, getter, fallback)
        results[name] = centered
        audits[name] = audit.to_dict()
    return results, audits


def variance_decomposition(
    examples: Sequence[PhaseAExample],
    getter: Callable[[PhaseAExample], Tensor],
    global_decile_means: Tensor,
) -> tuple[VarianceDecomposition, CenteringAudit]:
    """Compute the contract's leave-one-example-out between fraction."""
    centered, audit = leave_one_example_out_center(examples, getter, global_decile_means)
    per_example = {}
    for example in examples:
        values = getter(example).float()
        deciles = response_deciles(example.token_count)
        denominator = (values - global_decile_means[deciles].float()).square().mean()
        numerator = centered[example.example_id].square().mean()
        per_example[example.example_id] = float(
            (1.0 - numerator / denominator.clamp(min=1e-12)).item()
        )
    per_concept = {
        concept: float(
            np.mean(
                [per_example[example.example_id] for example in examples if example.concept == concept]
            )
        )
        for concept in sorted({example.concept for example in examples})
    }
    return (
        VarianceDecomposition(
            per_example=per_example,
            per_concept=per_concept,
            macro=float(np.mean(list(per_concept.values()))),
        ),
        audit,
    )


def _as_diagnostic_examples(
    examples: Sequence[PhaseAExample],
    centered: Mapping[str, Mapping[str, Tensor]],
    *,
    writer_field: str,
) -> tuple[list[DiagnosticExample], tuple[str, ...]]:
    first = centered[writer_field][examples[0].example_id]
    head_ids = tuple(f"head_{index:02d}" for index in range(first.shape[1]))
    result = []
    for example in examples:
        writer = centered[writer_field][example.example_id]
        result.append(
            DiagnosticExample(
                example_id=example.example_id,
                concept=example.concept,
                writer={head_id: writer[:, index, :] for index, head_id in enumerate(head_ids)},
                normal_state=centered["normal_state"][example.example_id],
                target_u=centered["target_u"][example.example_id],
            )
        )
    return result, head_ids


def run_residualized_diagnostic(
    fit_examples: Sequence[PhaseAExample],
    evaluation_examples: Sequence[PhaseAExample],
    fit_centered: Mapping[str, Mapping[str, Tensor]],
    evaluation_centered: Mapping[str, Mapping[str, Tensor]],
    probes: Sequence[LinearProbe],
    probe_scale: Tensor,
    *,
    writer_field: str,
    alphas: Sequence[float],
    pca_seed: int,
    normal_only: bool = False,
) -> ResidualizedDiagnosticResult:
    """Fit and evaluate one frozen residualized ridge diagnostic."""
    train, head_ids = _as_diagnostic_examples(
        fit_examples, fit_centered, writer_field=writer_field
    )
    evaluation, evaluation_heads = _as_diagnostic_examples(
        evaluation_examples, evaluation_centered, writer_field=writer_field
    )
    if evaluation_heads != head_ids:
        raise ValueError("training and evaluation head fields differ")
    bases = fit_diagnostic_bases(train, head_ids, seed=pca_seed)
    train_matrices = build_diagnostic_matrices(train, bases)
    evaluation_matrices = build_diagnostic_matrices(evaluation, bases)
    selection = select_alpha_leave_one_concept_out(
        train_matrices, bases, alphas, normal_only=normal_only
    )
    features = train_matrices.normal_features if normal_only else train_matrices.full_features
    fit = fit_weighted_ridge(
        features,
        train_matrices.target_coordinates,
        train_matrices.sample_weights,
        alpha=selection.selected_alpha,
    )
    predictions = predict_original(
        fit, evaluation_matrices, bases, normal_only=normal_only
    )
    metrics = evaluate_diagnostic_predictions(
        predictions, evaluation_matrices, bases, probes, probe_scale
    )
    return ResidualizedDiagnosticResult(
        metrics=metrics,
        selected_alpha=selection.selected_alpha,
        alpha_scores=selection.macro_r2_by_alpha,
    )


def zero_residualized_diagnostic(
    evaluation_examples: Sequence[PhaseAExample],
    evaluation_centered: Mapping[str, Mapping[str, Tensor]],
    probes: Sequence[LinearProbe],
    probe_scale: Tensor,
    *,
    pca_seed: int,
    fit_examples: Sequence[PhaseAExample],
    fit_centered: Mapping[str, Mapping[str, Tensor]],
) -> ResidualizedDiagnosticResult:
    """Evaluate the zero-centered prediction in the same frozen target basis."""
    train, head_ids = _as_diagnostic_examples(fit_examples, fit_centered, writer_field="k12")
    evaluation, _ = _as_diagnostic_examples(
        evaluation_examples, evaluation_centered, writer_field="k12"
    )
    bases = fit_diagnostic_bases(train, head_ids, seed=pca_seed)
    matrices = build_diagnostic_matrices(evaluation, bases)
    metrics = evaluate_diagnostic_predictions(
        torch.zeros_like(matrices.target_original), matrices, bases, probes, probe_scale
    )
    return ResidualizedDiagnosticResult(metrics=metrics, selected_alpha=None, alpha_scores={})


def normalized_phase_a_examples(
    examples: Sequence[PhaseAExample], selected_scale: Tensor, nonselected_scale: Tensor
) -> list[PhaseAExample]:
    """Return examples with all trajectory families on sealed RMS scales."""
    return [
        replace(
            example,
            k12=normalize_heads(example.k12, selected_scale),
            precursor_k12=normalize_heads(example.precursor_k12, selected_scale),
            nonselected=normalize_heads(example.nonselected, nonselected_scale),
        )
        for example in examples
    ]


def default_phase_a_fields() -> dict[str, Callable[[PhaseAExample], Tensor]]:
    """Return the complete frozen residualization field registry."""
    return {
        "k12": lambda example: example.k12,
        "nonselected": lambda example: example.nonselected,
        "precursor_k12": lambda example: example.precursor_k12,
        "normal_state": lambda example: example.normal_state,
        "target_u": lambda example: example.target_u,
    }


def concept_decile_summary(
    examples: Sequence[PhaseAExample], getter: Callable[[PhaseAExample], Tensor]
) -> dict[str, Tensor]:
    """Build equal-example concept means for each response decile."""
    result = {}
    for concept in sorted({example.concept for example in examples}):
        members = [example for example in examples if example.concept == concept]
        by_example = []
        for example in members:
            values = getter(example).float()
            deciles = response_deciles(example.token_count)
            global_mean = values.mean(dim=0)
            by_example.append(
                torch.stack(
                    [
                        values[deciles == decile].mean(dim=0)
                        if (deciles == decile).any()
                        else global_mean
                        for decile in range(10)
                    ]
                )
            )
        result[concept] = torch.stack(by_example).mean(dim=0)
    return result


def _svd_basis(values: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    mean = values.float().mean(dim=0)
    centered = values.float() - mean
    effective = min(rank, centered.shape[0] - 1, centered.shape[1])
    if effective <= 0:
        raise ValueError("outer-fold PCA has no available rank")
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    components = vh[:effective]
    maxima = components.abs().argmax(dim=1)
    signs = torch.sign(components[torch.arange(effective), maxima])
    signs[signs == 0] = 1
    return mean, components * signs[:, None]


def _project_summary(
    summaries: Mapping[str, Tensor], train_concepts: Sequence[str], *, rank: int
) -> dict[str, Tensor]:
    """Fit one PCA on training concept-decile rows and flatten coordinates."""
    train_rows = torch.cat([summaries[concept] for concept in train_concepts])
    mean, components = _svd_basis(train_rows, min(rank, len(train_concepts) - 1))
    return {
        concept: ((values.float() - mean) @ components.T).flatten()
        for concept, values in summaries.items()
    }


def _project_heads(
    summaries: Mapping[str, Tensor], train_concepts: Sequence[str], *, rank: int
) -> dict[str, Tensor]:
    """Fit a separate outer-training PCA for every head summary."""
    head_count = next(iter(summaries.values())).shape[1]
    projected: dict[str, list[Tensor]] = {concept: [] for concept in summaries}
    for head in range(head_count):
        one_head = {concept: value[:, head, :] for concept, value in summaries.items()}
        coordinates = _project_summary(one_head, train_concepts, rank=rank)
        for concept in summaries:
            projected[concept].append(coordinates[concept])
    return {concept: torch.cat(values) for concept, values in projected.items()}


def _ridge_predict_small(
    train_x: Tensor, train_y: Tensor, test_x: Tensor, alpha: float
) -> Tensor:
    """Equal-row ridge prediction using the smaller primal or dual system."""
    x_mean = train_x.double().mean(dim=0)
    y_mean = train_y.double().mean(dim=0)
    x = train_x.double() - x_mean
    y = train_y.double() - y_mean
    test = test_x.double() - x_mean
    if x.shape[1] <= x.shape[0]:
        coefficients = torch.linalg.solve(
            x.T @ x + alpha * torch.eye(x.shape[1], dtype=torch.double), x.T @ y
        )
        prediction = test @ coefficients
    else:
        dual = torch.linalg.solve(
            x @ x.T + alpha * torch.eye(x.shape[0], dtype=torch.double), y
        )
        prediction = test @ x.T @ dual
    return (prediction + y_mean).float()


def _select_inner_alpha(
    features: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    train_concepts: Sequence[str],
    alphas: Sequence[float],
) -> tuple[float, dict[float, float]]:
    scores = {}
    for alpha in alphas:
        errors = []
        for heldout in train_concepts:
            inner = [concept for concept in train_concepts if concept != heldout]
            prediction = _ridge_predict_small(
                torch.stack([features[concept] for concept in inner]),
                torch.stack([targets[concept].flatten() for concept in inner]),
                features[heldout].unsqueeze(0),
                float(alpha),
            )[0]
            observed = targets[heldout].flatten()
            errors.append(
                float(
                    ((prediction - observed).square().mean() / observed.square().mean().clamp(min=1e-6)).item()
                )
            )
        scores[float(alpha)] = float(np.mean(errors))
    selected = min(scores, key=lambda alpha: (scores[alpha], -alpha))
    return selected, scores


def _outer_u_coordinates(
    summaries: Mapping[str, Tensor], train_concepts: Sequence[str], *, rank: int
) -> tuple[dict[str, Tensor], list[tuple[Tensor, Tensor]]]:
    """Fit outer-training-only target-u PCA independently within each decile."""
    coordinates: dict[str, list[Tensor]] = {concept: [] for concept in summaries}
    bases = []
    for decile in range(10):
        training = torch.stack(
            [summaries[concept][decile] for concept in train_concepts]
        )
        mean, components = _svd_basis(training, min(rank, len(train_concepts) - 1))
        bases.append((mean, components))
        for concept, values in summaries.items():
            coordinates[concept].append(
                (values[decile].float() - mean) @ components.T
            )
    return {concept: torch.cat(rows) for concept, rows in coordinates.items()}, bases


def _inverse_u_coordinates(
    coordinates: Tensor, bases: Sequence[tuple[Tensor, Tensor]]
) -> Tensor:
    """Reconstruct a ten-decile residual summary from outer-fold coordinates."""
    widths = [components.shape[0] for _mean, components in bases]
    if coordinates.numel() != sum(widths):
        raise ValueError("target-u coordinates do not match outer-fold bases")
    cursor = 0
    result = []
    for (mean, components), width in zip(bases, widths, strict=True):
        result.append(coordinates[cursor : cursor + width] @ components + mean)
        cursor += width
    return torch.stack(result)


def cross_concept_probe_predictions(
    examples: Sequence[PhaseAExample],
    probes: Sequence[LinearProbe],
    probe_scale: Tensor,
    semantic_embeddings: Mapping[str, Tensor],
    alphas: Sequence[float],
) -> dict[str, Any]:
    """Run outer LOCO probe-vector prediction for every frozen feature family."""
    validate_phase_a_examples(examples)
    concepts = sorted({example.concept for example in examples})
    if set(semantic_embeddings) != set(concepts) or len(concepts) < 3:
        raise ValueError("semantic embeddings must cover every concept")
    summaries = {
        "k12": concept_decile_summary(examples, lambda example: example.k12),
        "nonselected": concept_decile_summary(examples, lambda example: example.nonselected),
        "precursor_k12": concept_decile_summary(examples, lambda example: example.precursor_k12),
        "normal_state": concept_decile_summary(examples, lambda example: example.normal_state),
        "target": concept_decile_summary(
            examples, lambda example: probe_contrast(example.target_u, probes, probe_scale)
        ),
        "target_u": concept_decile_summary(examples, lambda example: example.target_u),
    }
    predictions: dict[str, dict[str, Tensor]] = {
        name: {} for name in ("k12", "nonselected", "precursor_k12", "normal_state", "semantic", "mean")
    }
    u_predictions: dict[str, dict[str, Tensor]] = {
        name: {} for name in predictions
    }
    fold_audit = {}
    alpha_audit: dict[str, dict[str, float]] = {name: {} for name in predictions if name != "mean"}
    for heldout in concepts:
        train = [concept for concept in concepts if concept != heldout]
        features = {
            "k12": _project_heads(summaries["k12"], train, rank=2),
            "nonselected": _project_heads(summaries["nonselected"], train, rank=2),
            "precursor_k12": _project_heads(summaries["precursor_k12"], train, rank=2),
            "normal_state": _project_summary(summaries["normal_state"], train, rank=4),
            "semantic": {concept: semantic_embeddings[concept].float().flatten() for concept in concepts},
        }
        targets = summaries["target"]
        u_coordinates, u_bases = _outer_u_coordinates(
            summaries["target_u"], train, rank=4
        )
        for family, values in features.items():
            selected, _scores = _select_inner_alpha(values, targets, train, alphas)
            alpha_audit[family][heldout] = selected
            predictions[family][heldout] = _ridge_predict_small(
                torch.stack([values[concept] for concept in train]),
                torch.stack([targets[concept].flatten() for concept in train]),
                values[heldout].unsqueeze(0),
                selected,
            )[0].reshape_as(targets[heldout])
            predicted_u_coordinates = _ridge_predict_small(
                torch.stack([values[concept] for concept in train]),
                torch.stack([u_coordinates[concept] for concept in train]),
                values[heldout].unsqueeze(0),
                selected,
            )[0]
            u_predictions[family][heldout] = _inverse_u_coordinates(
                predicted_u_coordinates, u_bases
            )
        predictions["mean"][heldout] = torch.stack([targets[concept] for concept in train]).mean(dim=0)
        u_predictions["mean"][heldout] = torch.stack(
            [summaries["target_u"][concept] for concept in train]
        ).mean(dim=0)
        fold_audit[heldout] = {
            "training_concepts": train,
            "heldout_target_used_in_fit": False,
            "training_target_count": len(train),
        }

    metrics = {}
    observed_stack = torch.stack([summaries["target"][concept].flatten() for concept in concepts])
    target_mean = observed_stack.mean(dim=0)
    total_energy = (observed_stack - target_mean).square().sum().clamp(min=1e-6)
    for family, by_concept in predictions.items():
        per_concept = {}
        predicted_stack = []
        for concept in concepts:
            observed = summaries["target"][concept]
            predicted = by_concept[concept]
            predicted_stack.append(predicted.flatten())
            per_concept[concept] = float(
                ((predicted - observed).square().mean() / observed.square().mean().clamp(min=1e-6)).item()
            )
        squared_error = (torch.stack(predicted_stack) - observed_stack).square().sum()
        metrics[family] = {
            "macro_snmse": float(np.mean(list(per_concept.values()))),
            "probe_vector_r2_across_concepts": float((1.0 - squared_error / total_energy).item()),
            "per_concept_snmse": per_concept,
        }
    u_metrics = {}
    observed_u = torch.stack(
        [summaries["target_u"][concept].flatten() for concept in concepts]
    )
    u_mean = observed_u.mean(dim=0)
    u_energy = (observed_u - u_mean).square().sum().clamp(min=1e-6)
    for family, by_concept in u_predictions.items():
        per_concept = {}
        predicted_stack = []
        for concept in concepts:
            observed = summaries["target_u"][concept]
            predicted = by_concept[concept]
            predicted_stack.append(predicted.flatten())
            per_concept[concept] = float(
                (
                    (predicted - observed).square().mean()
                    / observed.square().mean().clamp(min=1e-6)
                ).item()
            )
        squared_error = (torch.stack(predicted_stack) - observed_u).square().sum()
        u_metrics[family] = {
            "macro_snmse": float(np.mean(list(per_concept.values()))),
            "residual_r2_across_concepts": float(
                (1.0 - squared_error / u_energy).item()
            ),
            "per_concept_snmse": per_concept,
        }
    return {
        "metrics": metrics,
        "secondary_target_u_metrics": u_metrics,
        "selected_alpha_by_outer_fold": alpha_audit,
        "outer_fold_audit": fold_audit,
        "concept_order": concepts,
    }


def fit_original_probe_scale(
    examples: Sequence[PhaseAExample], probes: Sequence[LinearProbe]
) -> Tensor:
    """Expose the sealed Gate 1 standardizer computation for audit parity tests."""
    diagnostics = [
        DiagnosticExample(
            example_id=example.example_id,
            concept=example.concept,
            writer={"unused": example.k12[:, 0, :]},
            normal_state=example.normal_state,
            target_u=example.target_u,
        )
        for example in examples
    ]
    return fit_probe_standardization(diagnostics, probes)


def position_features_match_contract(token_count: int) -> bool:
    """Audit compatibility between the old feature helper and exact deciles."""
    positions = response_position_features(token_count)[:, 0]
    expected = torch.arange(token_count).float() / max(token_count - 1, 1)
    return bool(torch.equal(positions, expected))
