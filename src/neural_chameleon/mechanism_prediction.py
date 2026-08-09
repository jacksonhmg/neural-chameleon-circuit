"""Leakage-constrained Gate 1 observed-writer diagnostic prediction.

Only the frozen permitted predictors are exposed to feature construction:
observed K12 deltas, normal ``resid_post[8]``, and response-relative position.
The target displacement is stored separately for fitting and evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .causal_mechanisms import RealizedForwardCapture, writer_delta
from .interventions import LinearProbe


@dataclass(frozen=True)
class DiagnosticExample:
    """One valid-token diagnostic record."""

    example_id: str
    concept: str
    writer: Mapping[str, Tensor]
    normal_state: Tensor
    target_u: Tensor

    @property
    def token_count(self) -> int:
        return int(self.normal_state.shape[0])


@dataclass(frozen=True)
class PCABasis:
    """A fixed weighted-mean linear basis."""

    mean: Tensor
    components: Tensor

    def transform(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.mean.numel():
            raise ValueError("PCA transform input has the wrong shape")
        return (values.float() - self.mean) @ self.components.T

    def inverse_transform(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.components.shape[0]:
            raise ValueError("PCA inverse input has the wrong shape")
        return coordinates.float() @ self.components + self.mean


@dataclass(frozen=True)
class DiagnosticBases:
    """Discovery-only feature and target bases."""

    head_ids: tuple[str, ...]
    writer: Mapping[str, PCABasis]
    normal_state: PCABasis
    target: PCABasis


@dataclass(frozen=True)
class DiagnosticMatrices:
    """Token matrices plus example/concept slices for macro metrics."""

    full_features: Tensor
    normal_features: Tensor
    target_coordinates: Tensor
    target_original: Tensor
    sample_weights: Tensor
    example_slices: tuple[tuple[int, int], ...]
    example_ids: tuple[str, ...]
    concepts: tuple[str, ...]


@dataclass(frozen=True)
class RidgeFit:
    """A weighted multivariate ridge fit in the frozen target basis."""

    feature_mean: Tensor
    target_mean: Tensor
    coefficients: Tensor
    alpha: float

    def predict_coordinates(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_mean.numel():
            raise ValueError("ridge prediction features have the wrong shape")
        return (
            features.float() - self.feature_mean
        ) @ self.coefficients + self.target_mean


@dataclass(frozen=True)
class AlphaSelection:
    selected_alpha: float
    macro_r2_by_alpha: Mapping[float, float]


@dataclass(frozen=True)
class DiagnosticMetrics:
    macro_r2_u: float
    macro_probe_vector_snmse: float
    per_concept_r2_u: Mapping[str, float]
    per_concept_probe_vector_snmse: Mapping[str, float]
    per_example_r2_u: Mapping[str, float]
    per_example_probe_vector_snmse: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_r2_u": self.macro_r2_u,
            "macro_probe_vector_snmse": self.macro_probe_vector_snmse,
            "per_concept_r2_u": dict(self.per_concept_r2_u),
            "per_concept_probe_vector_snmse": dict(self.per_concept_probe_vector_snmse),
            "per_example_r2_u": dict(self.per_example_r2_u),
            "per_example_probe_vector_snmse": dict(self.per_example_probe_vector_snmse),
        }


def validate_diagnostic_examples(
    examples: Sequence[DiagnosticExample], head_ids: Sequence[str]
) -> dict[str, Any]:
    """Enforce shapes, finiteness, identities, and the feature-field contract."""
    if not examples or not head_ids:
        raise ValueError("diagnostic examples and head IDs must be nonempty")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("diagnostic example IDs must be unique")
    hidden_size = examples[0].normal_state.shape[1]
    head_widths: dict[str, int] = {}
    for example in examples:
        if not example.example_id or not example.concept or example.token_count <= 0:
            raise ValueError("diagnostic identity and token count must be nonempty")
        if set(example.writer) != set(head_ids):
            raise ValueError("diagnostic writer heads do not match the frozen set")
        if example.normal_state.ndim != 2 or example.target_u.ndim != 2:
            raise ValueError("normal state and target must be token-by-hidden matrices")
        if example.normal_state.shape != example.target_u.shape:
            raise ValueError("normal state and target displacement shapes differ")
        if example.normal_state.shape[1] != hidden_size:
            raise ValueError("diagnostic hidden widths differ")
        for head_id in head_ids:
            values = example.writer[head_id]
            if values.ndim != 2 or values.shape[0] != example.token_count:
                raise ValueError(
                    "writer feature does not align to valid response tokens"
                )
            width = int(values.shape[1])
            if head_id in head_widths and head_widths[head_id] != width:
                raise ValueError("writer head widths differ across examples")
            head_widths[head_id] = width
            if not torch.isfinite(values).all():
                raise ValueError("writer feature contains nonfinite values")
        if (
            not torch.isfinite(example.normal_state).all()
            or not torch.isfinite(example.target_u).all()
        ):
            raise ValueError("diagnostic state contains nonfinite values")
    return {
        "example_count": len(examples),
        "concept_count": len({example.concept for example in examples}),
        "token_count": sum(example.token_count for example in examples),
        "head_count": len(head_ids),
        "head_widths": head_widths,
        "hidden_size": int(hidden_size),
        "feature_fields": [
            "observed_k12_delta",
            "normal_resid_post_8",
            "response_relative_position",
        ],
        "forbidden_feature_fields_present": [],
        "target_isolated_from_feature_construction": True,
    }


def reconstruct_residual_after(capture: RealizedForwardCapture, layer: int) -> Tensor:
    """Replay the captured residual through an inclusive decoder-layer index."""
    if layer < 0 or layer not in capture.attention_branches:
        raise ValueError("requested residual layer was not captured")
    hidden = capture.initial_residual.values.clone()
    for layer_index in range(layer + 1):
        hidden = hidden + capture.attention_branches[layer_index].values.to(
            hidden.dtype
        )
        hidden = hidden + capture.mlp_branches[layer_index].values.to(hidden.dtype)
    return hidden


def diagnostic_examples_from_batch(
    example_ids: Sequence[str],
    concepts: Sequence[str],
    normal: RealizedForwardCapture,
    triggered: RealizedForwardCapture,
    head_ids: Sequence[str],
    layers: Sequence[torch.nn.Module],
    *,
    normal_state_layer: int = 8,
) -> list[DiagnosticExample]:
    """Materialize valid-token examples without exposing forbidden predictors."""
    if len(example_ids) != normal.batch_size or len(concepts) != normal.batch_size:
        raise ValueError("batch identities do not match the captured batch")
    if not torch.equal(normal.response_ids, triggered.response_ids) or not torch.equal(
        normal.response_mask, triggered.response_mask
    ):
        raise ValueError("normal and triggered captures do not align")
    delta = writer_delta(triggered, normal, head_ids, layers)
    normal_state = reconstruct_residual_after(normal, normal_state_layer).float()
    target = (
        triggered.monitor_residual.values.float()
        - normal.monitor_residual.values.float()
    )
    result = []
    for row, (example_id, concept) in enumerate(
        zip(example_ids, concepts, strict=True)
    ):
        mask = normal.response_mask[row].bool()
        result.append(
            DiagnosticExample(
                example_id=example_id,
                concept=concept,
                writer={
                    head_id: delta[head_id][row, mask].cpu() for head_id in head_ids
                },
                normal_state=normal_state[row, mask].cpu(),
                target_u=target[row, mask].cpu(),
            )
        )
    validate_diagnostic_examples(result, head_ids)
    return result


def _concatenate_field(
    examples: Sequence[DiagnosticExample], getter: Any
) -> tuple[Tensor, Tensor]:
    values = [getter(example).float().cpu() for example in examples]
    weights = [
        torch.full((example.token_count,), 1.0 / example.token_count)
        for example in examples
    ]
    return torch.cat(values), torch.cat(weights)


def fit_weighted_pca(
    values: Tensor,
    weights: Tensor,
    *,
    rank: int,
    seed: int,
) -> PCABasis:
    """Fit deterministic weighted PCA using a fixed low-rank SVD seed."""
    if values.ndim != 2 or weights.ndim != 1 or values.shape[0] != weights.numel():
        raise ValueError("weighted PCA shapes do not align")
    if (
        rank <= 0
        or not torch.isfinite(values).all()
        or not torch.isfinite(weights).all()
    ):
        raise ValueError("weighted PCA inputs or rank are invalid")
    normalized_weight = weights.float() / weights.float().sum().clamp(min=1e-12)
    mean = (values.float() * normalized_weight[:, None]).sum(dim=0)
    centered = (values.float() - mean) * torch.sqrt(normalized_weight[:, None])
    effective_rank = min(rank, centered.shape[0], centered.shape[1])
    if effective_rank == min(centered.shape):
        _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
        components = vh[:effective_rank]
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            _u, _s, v = torch.pca_lowrank(
                centered, q=effective_rank, center=False, niter=4
            )
        components = v[:, :effective_rank].T
    maxima = components.abs().argmax(dim=1)
    signs = torch.sign(components[torch.arange(effective_rank), maxima])
    signs[signs == 0] = 1
    components = components * signs[:, None]
    return PCABasis(mean=mean.cpu(), components=components.cpu())


def fit_diagnostic_bases(
    examples: Sequence[DiagnosticExample],
    head_ids: Sequence[str],
    *,
    writer_rank: int = 16,
    normal_rank: int = 64,
    target_rank: int = 128,
    seed: int = 31002,
) -> DiagnosticBases:
    validate_diagnostic_examples(examples, head_ids)
    writer_bases = {}
    for index, head_id in enumerate(head_ids):
        values, weights = _concatenate_field(
            examples, lambda example, head_id=head_id: example.writer[head_id]
        )
        writer_bases[head_id] = fit_weighted_pca(
            values, weights, rank=writer_rank, seed=seed + index
        )
    normal_values, weights = _concatenate_field(
        examples, lambda example: example.normal_state
    )
    target_values, target_weights = _concatenate_field(
        examples, lambda example: example.target_u
    )
    return DiagnosticBases(
        head_ids=tuple(head_ids),
        writer=writer_bases,
        normal_state=fit_weighted_pca(
            normal_values, weights, rank=normal_rank, seed=seed + 100
        ),
        target=fit_weighted_pca(
            target_values, target_weights, rank=target_rank, seed=seed + 200
        ),
    )


def response_position_features(token_count: int) -> Tensor:
    """Frozen position fraction plus the first four Fourier pairs."""
    if token_count <= 0:
        raise ValueError("response position features require positive token count")
    fraction = torch.arange(token_count, dtype=torch.float32) / max(token_count - 1, 1)
    features = [fraction]
    for frequency in range(1, 5):
        angle = 2.0 * torch.pi * frequency * fraction
        features.extend((torch.sin(angle), torch.cos(angle)))
    return torch.stack(features, dim=1)


def build_diagnostic_matrices(
    examples: Sequence[DiagnosticExample], bases: DiagnosticBases
) -> DiagnosticMatrices:
    validate_diagnostic_examples(examples, bases.head_ids)
    full_rows = []
    normal_rows = []
    target_coordinates = []
    target_original = []
    weights = []
    slices = []
    cursor = 0
    for example in examples:
        writer_features = [
            bases.writer[head_id].transform(example.writer[head_id])
            for head_id in bases.head_ids
        ]
        normal = bases.normal_state.transform(example.normal_state)
        position = response_position_features(example.token_count)
        normal_feature = torch.cat((normal, position), dim=1)
        full_rows.append(torch.cat((*writer_features, normal_feature), dim=1))
        normal_rows.append(normal_feature)
        target_coordinates.append(bases.target.transform(example.target_u))
        target_original.append(example.target_u.float())
        weights.append(torch.full((example.token_count,), 1.0 / example.token_count))
        slices.append((cursor, cursor + example.token_count))
        cursor += example.token_count
    return DiagnosticMatrices(
        full_features=torch.cat(full_rows),
        normal_features=torch.cat(normal_rows),
        target_coordinates=torch.cat(target_coordinates),
        target_original=torch.cat(target_original),
        sample_weights=torch.cat(weights),
        example_slices=tuple(slices),
        example_ids=tuple(example.example_id for example in examples),
        concepts=tuple(example.concept for example in examples),
    )


def fit_weighted_ridge(
    features: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    alpha: float,
) -> RidgeFit:
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("ridge alpha must be finite and non-negative")
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("ridge feature and target shapes do not align")
    if weights.shape != (features.shape[0],):
        raise ValueError("ridge weights do not align")
    normalized = weights.double() / weights.double().sum().clamp(min=1e-12)
    x = features.double()
    y = targets.double()
    x_mean = (x * normalized[:, None]).sum(dim=0)
    y_mean = (y * normalized[:, None]).sum(dim=0)
    root = torch.sqrt(normalized[:, None])
    x_centered = (x - x_mean) * root
    y_centered = (y - y_mean) * root
    gram = x_centered.T @ x_centered
    rhs = x_centered.T @ y_centered
    regularized = gram + float(alpha) * torch.eye(gram.shape[0], dtype=gram.dtype)
    coefficients = torch.linalg.solve(regularized, rhs)
    return RidgeFit(
        feature_mean=x_mean.float(),
        target_mean=y_mean.float(),
        coefficients=coefficients.float(),
        alpha=float(alpha),
    )


def _rows_for_examples(
    matrices: DiagnosticMatrices, example_indices: Sequence[int]
) -> Tensor:
    rows = [
        row
        for index in example_indices
        for row in range(*matrices.example_slices[index])
    ]
    return torch.as_tensor(rows, dtype=torch.long)


def _macro_r2_for_examples(
    prediction: Tensor,
    matrices: DiagnosticMatrices,
    example_indices: Sequence[int],
    target_mean: Tensor,
) -> float:
    by_concept: dict[str, list[float]] = {}
    for index in example_indices:
        start, stop = matrices.example_slices[index]
        observed = matrices.target_original[start:stop].float()
        predicted = prediction[start:stop].float()
        error = (observed - predicted).square().mean()
        baseline = (observed - target_mean.float()).square().mean().clamp(min=1e-12)
        value = float((1.0 - error / baseline).item())
        by_concept.setdefault(matrices.concepts[index], []).append(value)
    return float(np.mean([np.mean(values) for values in by_concept.values()]))


def select_alpha_leave_one_concept_out(
    matrices: DiagnosticMatrices,
    bases: DiagnosticBases,
    alphas: Sequence[float],
    *,
    normal_only: bool = False,
) -> AlphaSelection:
    if not alphas or len(set(float(alpha) for alpha in alphas)) != len(alphas):
        raise ValueError("ridge alpha grid must be nonempty and unique")
    concepts = sorted(set(matrices.concepts))
    if len(concepts) < 2:
        raise ValueError("alpha selection requires at least two concepts")
    features = matrices.normal_features if normal_only else matrices.full_features
    scores = {}
    for alpha in alphas:
        fold_values = []
        for heldout in concepts:
            train_indices = [
                index
                for index, concept in enumerate(matrices.concepts)
                if concept != heldout
            ]
            test_indices = [
                index
                for index, concept in enumerate(matrices.concepts)
                if concept == heldout
            ]
            train_rows = _rows_for_examples(matrices, train_indices)
            fit = fit_weighted_ridge(
                features[train_rows],
                matrices.target_coordinates[train_rows],
                matrices.sample_weights[train_rows],
                alpha=float(alpha),
            )
            prediction = torch.zeros_like(matrices.target_original)
            test_rows = _rows_for_examples(matrices, test_indices)
            coordinates = fit.predict_coordinates(features[test_rows])
            prediction[test_rows] = bases.target.inverse_transform(coordinates)
            fold_values.append(
                _macro_r2_for_examples(
                    prediction, matrices, test_indices, bases.target.mean
                )
            )
        scores[float(alpha)] = float(np.mean(fold_values))
    selected = max(scores, key=lambda alpha: (scores[alpha], alpha))
    return AlphaSelection(selected_alpha=float(selected), macro_r2_by_alpha=scores)


def predict_original(
    fit: RidgeFit,
    matrices: DiagnosticMatrices,
    bases: DiagnosticBases,
    *,
    normal_only: bool = False,
) -> Tensor:
    features = matrices.normal_features if normal_only else matrices.full_features
    return bases.target.inverse_transform(fit.predict_coordinates(features))


def fit_probe_standardization(
    examples: Sequence[DiagnosticExample], probes: Sequence[LinearProbe]
) -> Tensor:
    """Fit discovery SD for each mean raw-margin displacement."""
    if not probes:
        raise ValueError("probe standardization requires probes")
    effects = []
    for example in examples:
        values = []
        for probe in probes:
            values.append(
                float((example.target_u.float() @ probe.weight.float().T).mean().item())
            )
        effects.append(values)
    scale = torch.as_tensor(effects, dtype=torch.float32).std(dim=0, unbiased=False)
    return scale.clamp(min=1e-6)


def evaluate_diagnostic_predictions(
    predictions: Tensor,
    matrices: DiagnosticMatrices,
    bases: DiagnosticBases,
    probes: Sequence[LinearProbe],
    probe_scale: Tensor,
) -> DiagnosticMetrics:
    if predictions.shape != matrices.target_original.shape:
        raise ValueError("diagnostic prediction and target shapes differ")
    if probe_scale.shape != (len(probes),):
        raise ValueError("probe standardization shape differs from probe family")
    r2_by_example = {}
    snmse_by_example = {}
    for index, (start, stop) in enumerate(matrices.example_slices):
        observed = matrices.target_original[start:stop].float()
        predicted = predictions[start:stop].float()
        error = (observed - predicted).square().mean()
        denominator = (observed - bases.target.mean).square().mean().clamp(min=1e-12)
        r2_by_example[matrices.example_ids[index]] = float(
            (1.0 - error / denominator).item()
        )
        observed_probe = []
        predicted_probe = []
        for probe in probes:
            weight = probe.weight.float().T
            observed_probe.append((observed @ weight).mean())
            predicted_probe.append((predicted @ weight).mean())
        observed_vector = torch.stack(observed_probe) / probe_scale
        predicted_vector = torch.stack(predicted_probe) / probe_scale
        snmse = (
            predicted_vector - observed_vector
        ).square().mean() / observed_vector.square().mean().clamp(min=1e-6)
        snmse_by_example[matrices.example_ids[index]] = float(snmse.item())

    concept_r2 = {}
    concept_snmse = {}
    for concept in sorted(set(matrices.concepts)):
        ids = [
            example_id
            for example_id, row_concept in zip(
                matrices.example_ids, matrices.concepts, strict=True
            )
            if row_concept == concept
        ]
        concept_r2[concept] = float(np.mean([r2_by_example[value] for value in ids]))
        concept_snmse[concept] = float(
            np.mean([snmse_by_example[value] for value in ids])
        )
    return DiagnosticMetrics(
        macro_r2_u=float(np.mean(list(concept_r2.values()))),
        macro_probe_vector_snmse=float(np.mean(list(concept_snmse.values()))),
        per_concept_r2_u=concept_r2,
        per_concept_probe_vector_snmse=concept_snmse,
        per_example_r2_u=r2_by_example,
        per_example_probe_vector_snmse=snmse_by_example,
    )


def training_decile_mean_predictions(
    train: Sequence[DiagnosticExample], evaluation: Sequence[DiagnosticExample]
) -> Tensor:
    """Frozen training-target mean by response-relative decile."""
    if not train or not evaluation:
        raise ValueError("mean baseline requires train and evaluation examples")
    hidden = train[0].target_u.shape[1]
    sums = torch.zeros(10, hidden)
    counts = torch.zeros(10)
    for example in train:
        deciles = torch.clamp(
            (response_position_features(example.token_count)[:, 0] * 10).long(), max=9
        )
        for decile in range(10):
            selected = deciles == decile
            if selected.any():
                sums[decile] += example.target_u[selected].float().sum(dim=0)
                counts[decile] += int(selected.sum())
    global_mean = torch.cat([example.target_u.float() for example in train]).mean(dim=0)
    means = torch.stack(
        [
            sums[index] / counts[index] if counts[index] else global_mean
            for index in range(10)
        ]
    )
    predictions = []
    for example in evaluation:
        deciles = torch.clamp(
            (response_position_features(example.token_count)[:, 0] * 10).long(), max=9
        )
        predictions.append(means[deciles])
    return torch.cat(predictions)


def concept_leave_one_example_out_predictions(
    evaluation: Sequence[DiagnosticExample], fallback: Tensor
) -> Tensor:
    """Strong outcome-informed concept/decile baseline excluding the target example."""
    if not evaluation or fallback.ndim != 1:
        raise ValueError("concept baseline inputs are invalid")
    predictions = []
    deciles_by_id = {
        example.example_id: torch.clamp(
            (response_position_features(example.token_count)[:, 0] * 10).long(), max=9
        )
        for example in evaluation
    }
    for target in evaluation:
        rows = []
        for decile in deciles_by_id[target.example_id]:
            values = []
            for source in evaluation:
                if (
                    source.example_id == target.example_id
                    or source.concept != target.concept
                ):
                    continue
                selected = deciles_by_id[source.example_id] == decile
                if selected.any():
                    values.append(source.target_u[selected].float())
            rows.append(torch.cat(values).mean(dim=0) if values else fallback.float())
        predictions.append(torch.stack(rows))
    return torch.cat(predictions)
