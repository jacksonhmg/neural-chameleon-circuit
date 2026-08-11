#!/usr/bin/env python3
"""Screen compact K12 operator families using only existing development artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import LinearProbe  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-44/frozen-rapid-k12-development-contract.json"
PORTABILITY_PATH = ROOT / "results/day-44/cuda-portability-preflight.json"
CAPTURE_DIR = ROOT / "artifacts/mechanism-gate1-v1/chameleon"
MODEL_DIR = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
NATURAL_PATH = ROOT / "results/day-39/natural-endpoints.working.jsonl"
ABSOLUTE_PATH = ROOT / "results/day-39/absolute-effects.working.jsonl"
SUMMARY_PATH = ROOT / "results/day-44/offline-screen-summary.json"
AUDIT_PATH = ROOT / "results/day-44/offline-screen-audit.json"
SELECTION_PATH = ROOT / "results/day-44/pilot-selection.json"

RANKS = (1, 2, 4, 8, 16)
RIDGE_ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
NORMAL_RANDOM_DIM = 64
DIRECT_READOUT_ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from execution commit {commit}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def response_deciles(token_count: int) -> np.ndarray:
    if token_count <= 0:
        raise ValueError("response token count must be positive")
    denominator = max(token_count - 1, 1)
    return np.minimum(
        np.floor(10 * np.arange(token_count) / denominator).astype(np.int64), 9
    )


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def macro_by_concept(values: np.ndarray, concepts: np.ndarray) -> float:
    return float(
        np.mean([np.mean(values[concepts == concept]) for concept in sorted(set(concepts))])
    )


def position_features(deciles: np.ndarray) -> np.ndarray:
    fraction = deciles.astype(np.float64) / 9.0
    columns = [fraction]
    for harmonic in range(1, 5):
        columns.append(np.sin(2 * np.pi * harmonic * fraction))
        columns.append(np.cos(2 * np.pi * harmonic * fraction))
    return np.column_stack(columns)


def load_projection_maps(
    component_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    index = read_json(MODEL_DIR / "model.safetensors.index.json")
    shard_names = {
        index["weight_map"][
            f"model.layers.{int(component.split('.')[0].split('_')[1])}.self_attn.o_proj.weight"
        ]
        for component in component_ids
    }
    if len(shard_names) != 1:
        raise RuntimeError(f"expected one o_proj shard, found {sorted(shard_names)}")
    shard_path = MODEL_DIR / next(iter(shard_names))
    slices = []
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        cached_layers: dict[int, torch.Tensor] = {}
        for component in component_ids:
            layer_text, head_text = component.split(".")
            layer = int(layer_text.split("_")[1])
            head = int(head_text.split("_")[1])
            if layer not in cached_layers:
                cached_layers[layer] = handle.get_tensor(
                    f"model.layers.{layer}.self_attn.o_proj.weight"
                ).float()
            start = head * 256
            slices.append(cached_layers[layer][:, start : start + 256])
    o_concat = torch.cat(slices, dim=1)
    probe_paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    probes = tuple(LinearProbe.load(path) for path in probe_paths)
    if len(probes) != 13:
        raise RuntimeError(f"expected 13 probes, found {len(probes)}")
    probe_weights = torch.cat([probe.weight.float() for probe in probes], dim=0)
    monitor_map = probe_weights @ o_concat
    probe_names = [path.name.removesuffix("_weights.pt") for path in probe_paths]
    return (
        o_concat.numpy(),
        monitor_map.numpy(),
        probe_weights.numpy(),
        probe_names,
    )


def load_capture_summaries(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    shard_paths = sorted(CAPTURE_DIR.glob("*.pt"))
    if len(shard_paths) != 866:
        raise RuntimeError(f"expected 866 Chameleon positive shards, found {len(shard_paths)}")
    component_ids = tuple(contract["component_set"])
    example_ids: list[str] = []
    concepts: list[str] = []
    splits: list[str] = []
    writer_means = np.zeros((len(shard_paths), 10, 3072), dtype=np.float32)
    writer_sq_means = np.zeros((len(shard_paths), 10), dtype=np.float64)
    normal_means = np.zeros((len(shard_paths), 10, 3584), dtype=np.float32)
    counts = np.zeros((len(shard_paths), 10), dtype=np.int64)
    token_counts = np.zeros(len(shard_paths), dtype=np.int64)

    for example_index, path in enumerate(shard_paths):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if tuple(value["k12_head_ids"]) != component_ids:
            raise RuntimeError(f"component order mismatch in {path}")
        writer = value["k12_delta"].float().reshape(-1, 3072).numpy()
        normal = value["normal_resid_post_8"].float().numpy()
        if len(writer) != len(normal) or len(writer) != len(value["response_ids"]):
            raise RuntimeError(f"response shape mismatch in {path}")
        bins = response_deciles(len(writer))
        for decile in np.unique(bins):
            mask = bins == decile
            counts[example_index, decile] = int(mask.sum())
            writer_means[example_index, decile] = writer[mask].mean(axis=0)
            writer_sq_means[example_index, decile] = np.mean(
                np.sum(writer[mask].astype(np.float64) ** 2, axis=1)
            )
            normal_means[example_index, decile] = normal[mask].mean(axis=0)
        token_counts[example_index] = len(writer)
        example_ids.append(value["example_id"])
        concepts.append(value["concept"])
        splits.append(value["split"])

    if len(set(example_ids)) != len(example_ids):
        raise RuntimeError("duplicate capture example IDs")
    return {
        "example_ids": np.asarray(example_ids),
        "concepts": np.asarray(concepts),
        "splits": np.asarray(splits),
        "writer_means": writer_means,
        "writer_sq_means": writer_sq_means,
        "normal_means": normal_means,
        "counts": counts,
        "token_counts": token_counts,
        "shard_paths": shard_paths,
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_exact_direct_effects(
    example_ids: Sequence[str], probe_names: Sequence[str]
) -> np.ndarray:
    required = set(example_ids)
    natural: dict[tuple[str, str], np.ndarray] = {}
    for row in iter_jsonl(NATURAL_PATH):
        if (
            row.get("model") == "chameleon"
            and int(row.get("label", -1)) == 1
            and row.get("example_id") in required
        ):
            if row["probe_names"] != list(probe_names):
                raise RuntimeError("natural endpoint probe order mismatch")
            natural[(row["example_id"], row["condition"])] = np.asarray(
                row["mean_raw_margins"], dtype=np.float64
            )
    interventions: dict[tuple[str, str], np.ndarray] = {}
    for row in iter_jsonl(ABSOLUTE_PATH):
        if (
            row.get("model") == "chameleon"
            and int(row.get("label", -1)) == 1
            and row.get("example_id") in required
            and row.get("path") == "direct"
            and row.get("operator") in {"N_from_T", "T_from_N"}
        ):
            if row["probe_names"] != list(probe_names):
                raise RuntimeError("absolute endpoint probe order mismatch")
            interventions[(row["example_id"], row["operator"])] = np.asarray(
                row["mean_raw_margins"], dtype=np.float64
            )

    effects = []
    for example_id in example_ids:
        try:
            normal = natural[(example_id, "normal")]
            triggered = natural[(example_id, "correct_trigger")]
            induction = interventions[(example_id, "N_from_T")] - normal
            rescue = interventions[(example_id, "T_from_N")] - triggered
        except KeyError as exc:
            raise RuntimeError(f"missing direct-effect row for {example_id}: {exc}") from exc
        effects.append(0.5 * (induction - rescue))
    return np.stack(effects)


def concept_decile_mean(
    values: np.ndarray,
    counts: np.ndarray,
    token_counts: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = counts[indices] / token_counts[indices, None]
    totals = weights.sum(axis=0)
    weighted = np.einsum("nd,ndk->dk", weights, values[indices], optimize=True)
    means = np.divide(
        weighted,
        totals[:, None],
        out=np.zeros_like(weighted),
        where=totals[:, None] > 0,
    )
    return means, totals


def loo_concept_predictions(
    values: np.ndarray,
    counts: np.ndarray,
    token_counts: np.ndarray,
    indices: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    weights = counts[indices] / token_counts[indices, None]
    total_weights = weights.sum(axis=0)
    total_values = np.einsum("nd,ndk->dk", weights, values[indices], optimize=True)
    predictions = np.zeros_like(values[indices])
    for local_index in range(len(indices)):
        denominator = total_weights - weights[local_index]
        numerator = total_values - weights[local_index, :, None] * values[indices[local_index]]
        predictions[local_index] = np.divide(
            numerator,
            denominator[:, None],
            out=fallback.copy(),
            where=denominator[:, None] > 0,
        )
    return predictions


def writer_basis(
    concept_means: Sequence[np.ndarray], global_mean: np.ndarray, seed: int
) -> np.ndarray:
    rows = np.concatenate(
        [value - global_mean for value in concept_means], axis=0
    ).astype(np.float64)
    _u, _s, components = randomized_svd(
        rows,
        n_components=max(RANKS),
        n_iter=5,
        random_state=seed,
    )
    return components.astype(np.float32)


def fit_conditioned_coefficients(
    features: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[StandardScaler, Ridge, float, dict[str, float]]:
    splitter = GroupKFold(n_splits=min(4, len(set(groups))))
    alpha_losses: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_ALPHAS}
    for train, validation in splitter.split(features, groups=groups):
        scaler = StandardScaler().fit(features[train], sample_weight=sample_weights[train])
        train_features = scaler.transform(features[train])
        validation_features = scaler.transform(features[validation])
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha).fit(
                train_features,
                targets[train],
                sample_weight=sample_weights[train],
            )
            prediction = model.predict(validation_features)
            row_loss = np.mean((prediction - targets[validation]) ** 2, axis=1)
            group_loss = np.mean(
                [
                    np.average(
                        row_loss[groups[validation] == group],
                        weights=sample_weights[validation][groups[validation] == group],
                    )
                    for group in sorted(set(groups[validation]))
                ]
            )
            alpha_losses[alpha].append(float(group_loss))
    mean_losses = {alpha: float(np.mean(losses)) for alpha, losses in alpha_losses.items()}
    best_alpha = min(RIDGE_ALPHAS, key=lambda alpha: (mean_losses[alpha], -alpha))
    scaler = StandardScaler().fit(features, sample_weight=sample_weights)
    model = Ridge(alpha=best_alpha).fit(
        scaler.transform(features), targets, sample_weight=sample_weights
    )
    return scaler, model, best_alpha, {str(key): value for key, value in mean_losses.items()}


def score_writer_prediction(
    prediction: np.ndarray,
    indices: np.ndarray,
    data: Mapping[str, Any],
    monitor_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = data["counts"][indices]
    token_counts = data["token_counts"][indices]
    actual = data["writer_means"][indices]
    sq_means = data["writer_sq_means"][indices]
    weights = counts / token_counts[:, None]
    dot = np.sum(prediction.astype(np.float64) * actual.astype(np.float64), axis=2)
    predicted_sq = np.sum(prediction.astype(np.float64) ** 2, axis=2)
    sse = np.sum(weights * (sq_means - 2 * dot + predicted_sq), axis=1)
    energy = np.sum(weights * sq_means, axis=1)
    predicted_energy = np.sum(weights * predicted_sq, axis=1)
    example_writer = np.einsum("nd,ndk->nk", weights, prediction, optimize=True)
    proxy = example_writer @ monitor_map.T
    return (
        np.divide(sse, np.maximum(energy, 1e-12)),
        np.divide(predicted_energy, np.maximum(energy, 1e-12)),
        proxy,
    )


def candidate_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    concepts: np.ndarray,
    probe_indices: np.ndarray,
) -> dict[str, Any]:
    errors = np.sum((prediction - target) ** 2, axis=1)
    energy = np.sum(target**2, axis=1)
    rse = np.divide(errors, np.maximum(energy, 1e-12))
    cosine = cosine_rows(prediction, target)
    concept_wrong = 0
    per_concept = {}
    for concept in sorted(set(concepts)):
        mask = concepts == concept
        probe_index = int(probe_indices[np.flatnonzero(mask)[0]])
        predicted_own = float(prediction[mask, probe_index].mean())
        target_own = float(target[mask, probe_index].mean())
        wrong = predicted_own * target_own < 0
        concept_wrong += int(wrong)
        per_concept[concept] = {
            "rse": float(rse[mask].mean()),
            "cosine": float(cosine[mask].mean()),
            "predicted_own_probe": predicted_own,
            "target_own_probe": target_own,
            "own_probe_sign_wrong": bool(wrong),
        }
    return {
        "equal_concept_rse": macro_by_concept(rse, concepts),
        "equal_concept_cosine": macro_by_concept(cosine, concepts),
        "own_probe_wrong_direction_concepts": concept_wrong,
        "per_concept": per_concept,
    }


def run_screen() -> None:
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH, PORTABILITY_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_new_k12_candidate_outcomes":
        raise RuntimeError("Day 44 contract is not frozen")
    for entry in contract["inputs"].values():
        if isinstance(entry, dict) and "path" in entry:
            path = ROOT / entry["path"]
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"input hash mismatch for {path}")

    o_concat, monitor_map, probe_weights, probe_names = load_projection_maps(
        contract["component_set"]
    )
    data = load_capture_summaries(contract)
    concepts = data["concepts"]
    unique_concepts = sorted(set(concepts))
    if unique_concepts != probe_names:
        raise RuntimeError("concept and probe orders do not define the same set")
    probe_index_by_name = {name: index for index, name in enumerate(probe_names)}
    probe_indices = np.asarray([probe_index_by_name[value] for value in concepts])
    exact_direct = load_exact_direct_effects(data["example_ids"], probe_names)

    counts = data["counts"]
    token_counts = data["token_counts"]
    weights = counts / token_counts[:, None]
    actual_writer = np.einsum(
        "nd,ndk->nk", weights, data["writer_means"], optimize=True
    )
    actual_proxy = actual_writer @ monitor_map.T
    normal_example = np.einsum(
        "nd,ndk->nk", weights, data["normal_means"], optimize=True
    )

    rng = np.random.default_rng(int(contract["numerical_and_execution_contract"]["random_seed"]))
    random_matrix = rng.standard_normal((3584, NORMAL_RANDOM_DIM))
    normal_random_basis, _ = np.linalg.qr(random_matrix, mode="reduced")
    normal_projected = (
        data["normal_means"].reshape(-1, 3584) @ normal_random_basis
    ).reshape(len(concepts), 10, NORMAL_RANDOM_DIM)

    proxy_predictions: dict[str, np.ndarray] = {
        "global_additive": np.zeros_like(actual_proxy),
        "concept_position_prototype": np.zeros_like(actual_proxy),
    }
    activation_rse: dict[str, np.ndarray] = {
        "global_additive": np.zeros(len(concepts)),
        "concept_position_prototype": np.zeros(len(concepts)),
    }
    activation_energy: dict[str, np.ndarray] = {
        "global_additive": np.zeros(len(concepts)),
        "concept_position_prototype": np.zeros(len(concepts)),
    }
    for rank in RANKS:
        for prefix in ("shared_low_rank", "normal_state_conditioned_low_rank"):
            name = f"{prefix}_r{rank}"
            proxy_predictions[name] = np.zeros_like(actual_proxy)
            activation_rse[name] = np.zeros(len(concepts))
            activation_energy[name] = np.zeros(len(concepts))

    fold_records = {}
    for fold_index, heldout_concept in enumerate(unique_concepts):
        heldout = np.flatnonzero(concepts == heldout_concept)
        train = np.flatnonzero(concepts != heldout_concept)
        train_concept_means = []
        for concept in unique_concepts:
            if concept == heldout_concept:
                continue
            indices = np.flatnonzero(concepts == concept)
            mean, _ = concept_decile_mean(
                data["writer_means"], counts, token_counts, indices
            )
            train_concept_means.append(mean)
        global_mean = np.mean(np.stack(train_concept_means), axis=0)
        concept_prediction = loo_concept_predictions(
            data["writer_means"], counts, token_counts, heldout, global_mean
        )
        global_prediction = np.broadcast_to(
            global_mean[None], concept_prediction.shape
        ).copy()
        for name, prediction in (
            ("global_additive", global_prediction),
            ("concept_position_prototype", concept_prediction),
        ):
            rse, energy, proxy = score_writer_prediction(
                prediction, heldout, data, monitor_map
            )
            activation_rse[name][heldout] = rse
            activation_energy[name][heldout] = energy
            proxy_predictions[name][heldout] = proxy

        basis = writer_basis(train_concept_means, global_mean, 44001 + fold_index)
        residual = concept_prediction - global_mean[None]
        for rank in RANKS:
            coefficients = residual @ basis[:rank].T
            prediction = global_mean[None] + coefficients @ basis[:rank]
            name = f"shared_low_rank_r{rank}"
            rse, energy, proxy = score_writer_prediction(
                prediction, heldout, data, monitor_map
            )
            activation_rse[name][heldout] = rse
            activation_energy[name][heldout] = energy
            proxy_predictions[name][heldout] = proxy

        train_rows, train_deciles = np.nonzero(counts[train] > 0)
        train_examples = train[train_rows]
        train_groups = concepts[train_examples]
        train_features = np.concatenate(
            [
                normal_projected[train_examples, train_deciles],
                position_features(train_deciles),
            ],
            axis=1,
        )
        train_targets = (
            data["writer_means"][train_examples, train_deciles]
            - global_mean[train_deciles]
        ) @ basis.T
        train_weights = counts[train_examples, train_deciles] / token_counts[train_examples]
        scaler, ridge, alpha, alpha_losses = fit_conditioned_coefficients(
            train_features,
            train_targets,
            train_groups,
            train_weights,
        )
        held_rows, held_deciles = np.nonzero(counts[heldout] > 0)
        held_examples = heldout[held_rows]
        held_features = np.concatenate(
            [
                normal_projected[held_examples, held_deciles],
                position_features(held_deciles),
            ],
            axis=1,
        )
        held_coefficients = ridge.predict(scaler.transform(held_features))
        for rank in RANKS:
            prediction = np.broadcast_to(
                global_mean[None], concept_prediction.shape
            ).copy()
            prediction[held_rows, held_deciles] += (
                held_coefficients[:, :rank] @ basis[:rank]
            )
            name = f"normal_state_conditioned_low_rank_r{rank}"
            rse, energy, proxy = score_writer_prediction(
                prediction, heldout, data, monitor_map
            )
            activation_rse[name][heldout] = rse
            activation_energy[name][heldout] = energy
            proxy_predictions[name][heldout] = proxy
        fold_records[heldout_concept] = {
            "conditioned_alpha": alpha,
            "conditioned_alpha_losses": alpha_losses,
            "heldout_examples": len(heldout),
            "training_examples": len(train),
        }

    residual_delta = actual_writer @ o_concat.T
    radial_coefficients = np.divide(
        np.sum(residual_delta * normal_example, axis=1),
        np.maximum(np.sum(normal_example**2, axis=1), 1e-12),
    )
    radial_residual = radial_coefficients[:, None] * normal_example
    tangential_residual = residual_delta - radial_residual
    proxy_predictions["radial_actual_activity"] = radial_residual @ probe_weights.T
    proxy_predictions["tangential_actual_activity"] = tangential_residual @ probe_weights.T
    proxy_predictions["exact_natural_activity"] = actual_proxy.copy()
    activation_rse["exact_natural_activity"] = np.zeros(len(concepts))
    activation_energy["exact_natural_activity"] = np.ones(len(concepts))

    calibrated_effects: dict[str, np.ndarray] = {
        name: np.zeros_like(exact_direct) for name in proxy_predictions
    }
    global_effect_baseline = np.zeros_like(exact_direct)
    for heldout_concept in unique_concepts:
        heldout = concepts == heldout_concept
        train = ~heldout
        scaler = StandardScaler().fit(actual_proxy[train])
        model = Ridge(alpha=DIRECT_READOUT_ALPHA).fit(
            scaler.transform(actual_proxy[train]), exact_direct[train]
        )
        for name, proxy in proxy_predictions.items():
            calibrated_effects[name][heldout] = model.predict(
                scaler.transform(proxy[heldout])
            )
        concept_means = [
            exact_direct[train & (concepts == concept)].mean(axis=0)
            for concept in unique_concepts
            if concept != heldout_concept
        ]
        global_effect_baseline[heldout] = np.mean(concept_means, axis=0)

    candidates = {}
    for name, proxy in proxy_predictions.items():
        proxy_metrics = candidate_metrics(
            proxy, actual_proxy, concepts, probe_indices
        )
        causal_metrics = candidate_metrics(
            calibrated_effects[name], exact_direct, concepts, probe_indices
        )
        candidates[name] = {
            "linear_monitor_write_proxy": proxy_metrics,
            "calibrated_existing_direct_effect": causal_metrics,
            "activation_token_rse_equal_concept": (
                macro_by_concept(activation_rse[name], concepts)
                if name in activation_rse
                else None
            ),
            "activation_energy_ratio_equal_concept": (
                macro_by_concept(activation_energy[name], concepts)
                if name in activation_energy
                else None
            ),
        }
    baseline_metrics = candidate_metrics(
        global_effect_baseline, exact_direct, concepts, probe_indices
    )

    prototype_rse = candidates["concept_position_prototype"][
        "linear_monitor_write_proxy"
    ]["equal_concept_rse"]
    eligible_names = [
        name
        for name in candidates
        if name not in {"global_additive", "concept_position_prototype", "exact_natural_activity"}
    ]
    best_rse = min(
        candidates[name]["linear_monitor_write_proxy"]["equal_concept_rse"]
        for name in eligible_names
    )
    complexity_order = {
        "radial_actual_activity": 0,
        "tangential_actual_activity": 1,
        **{f"shared_low_rank_r{rank}": 10 + rank for rank in RANKS},
        **{
            f"normal_state_conditioned_low_rank_r{rank}": 100 + rank
            for rank in RANKS
        },
    }
    passing = []
    promotion_audit = {}
    for name in eligible_names:
        metrics = candidates[name]["linear_monitor_write_proxy"]
        improvement = prototype_rse - metrics["equal_concept_rse"]
        checks = {
            "improves_prototype_or_is_near_best": improvement >= 0.02
            or metrics["equal_concept_rse"] <= best_rse + 0.01,
            "cosine_at_least_0_8": metrics["equal_concept_cosine"] >= 0.8,
            "own_probe_sign_failures_at_most_2": metrics[
                "own_probe_wrong_direction_concepts"
            ]
            <= 2,
        }
        promotion_audit[name] = {
            "checks": checks,
            "prototype_rse_improvement": improvement,
            "passes": all(checks.values()),
        }
        if all(checks.values()):
            passing.append(name)
    passing.sort(
        key=lambda name: (
            candidates[name]["linear_monitor_write_proxy"]["equal_concept_rse"],
            complexity_order.get(name, 1000),
            name,
        )
    )
    promoted = passing[: int(contract["offline_screen"]["promotion_rule"]["maximum_structured_candidates"])]

    summary = {
        "schema_version": 1,
        "procedure": "rapid-k12-existing-artifact-screen-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_class": contract["evidence_class"],
        "population": {
            "examples": len(concepts),
            "concepts": len(unique_concepts),
            "valid_response_tokens": int(token_counts.sum()),
            "probe_count": len(probe_names),
        },
        "screening_endpoint": {
            "primary": "complete 13-probe linear monitor-write proxy after selected-head o_proj slices",
            "secondary": "LOCO ridge calibration to the already observed bidirectional direct K12 effect",
            "causal_boundary": "Screening only; promotion requires the frozen direct-path pilot.",
        },
        "candidates": candidates,
        "global_effect_baseline": baseline_metrics,
        "conditioned_folds": fold_records,
        "promotion_audit": promotion_audit,
        "promoted_candidates": promoted,
        "mandatory_pilot_candidate": "concept_position_prototype",
    }
    audit = {
        "schema_version": 1,
        "procedure": "rapid-k12-existing-artifact-screen-v1-audit",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "checks": {
            "contract_committed": True,
            "source_committed": True,
            "exact_capture_count": len(data["shard_paths"]) == 866,
            "exact_concept_count": len(unique_concepts) == 13,
            "exact_component_count": len(contract["component_set"]) == 12,
            "exact_probe_count": len(probe_names) == 13,
            "all_direct_effect_rows_present": len(exact_direct) == 866,
            "all_candidate_values_finite": all(
                np.isfinite(value).all() for value in proxy_predictions.values()
            ),
            "all_folds_leave_one_concept_out": len(fold_records) == 13,
            "no_fresh_data_used": True,
            "no_output_behavior_used": True,
            "promotion_count_within_limit": len(promoted) <= 3,
        },
        "input_hashes": {
            "contract": sha256_file(CONTRACT_PATH),
            "gate1_capture_manifest": sha256_file(
                ROOT / contract["inputs"]["gate1_capture_manifest"]["path"]
            ),
            "phase_b_execution_manifest": sha256_file(
                ROOT / contract["inputs"]["phase_b_execution_manifest"]["path"]
            ),
            "natural_endpoints": sha256_file(NATURAL_PATH),
            "absolute_effects": sha256_file(ABSOLUTE_PATH),
        },
    }
    audit["result"] = "pass" if all(audit["checks"].values()) else "fail"
    selection = {
        "schema_version": 1,
        "procedure": "rapid-k12-pilot-selection-v1",
        "execution_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "screen_summary_sha256": None,
        "mandatory": ["concept_position_prototype"],
        "promoted": promoted,
        "candidate_count": len(promoted),
        "selection_rule_applied_without_change": True,
    }
    write_json_atomic(SUMMARY_PATH, summary)
    selection["screen_summary_sha256"] = sha256_file(SUMMARY_PATH)
    write_json_atomic(AUDIT_PATH, audit)
    write_json_atomic(SELECTION_PATH, selection)
    if audit["result"] != "pass":
        raise RuntimeError(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {
                "result": "pass",
                "examples": len(concepts),
                "promoted": promoted,
                "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


def self_test() -> None:
    assert response_deciles(1).tolist() == [0]
    assert response_deciles(2).tolist() == [0, 9]
    assert response_deciles(10).tolist() == list(range(10))
    left = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(cosine_rows(left, left), 1.0)
    assert math.isclose(
        macro_by_concept(np.asarray([1.0, 3.0, 10.0]), np.asarray(["a", "a", "b"])),
        6.0,
    )
    features = position_features(np.asarray([0, 9]))
    assert features.shape == (2, 9)
    assert np.isfinite(features).all()
    print("Day 44 offline-screen self-test passed.")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    else:
        run_screen()
