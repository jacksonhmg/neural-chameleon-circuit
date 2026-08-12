#!/usr/bin/env python3
"""Locate Day 52 donor-identity divergence using only saved tensors."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from day44_run_k12_pilot import load_probes
from day52_analyze_reciprocal_reconfiguration import (
    CONTRACT_PATH,
    MANIFEST_PATH,
    SHARD_DIR,
    masked_vectors,
    read_json,
    state,
    vector_metrics,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results/day-53/day52-divergence-decomposition.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_parent_manifest() -> None:
    manifest = read_json(MANIFEST_PATH)
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"Day 52 parent artifact differs: {row['path']}")


def endpoint_distances(
    output: torch.Tensor,
    endpoints: Mapping[str, torch.Tensor],
) -> tuple[str, dict[str, float]]:
    distances = {
        name: float((output.double() - value.double()).square().sum(dim=-1).mean())
        for name, value in endpoints.items()
    }
    return min(distances, key=distances.get), distances


def project(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return values.float() @ basis @ basis.T


def mean_monitor(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (values.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)


def concept_average(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    verify_parent_manifest()
    contract = read_json(CONTRACT_PATH)
    probe_names, probes = load_probes()
    weights = torch.cat([probe.weight.float() for probe in probes], dim=0)
    _u, singular_values, vh = torch.linalg.svd(weights, full_matrices=False)
    rank = int((singular_values > singular_values.max() * 1e-6).sum())
    if rank != len(probe_names):
        raise RuntimeError("released probes are not full row rank")
    basis = vh[:rank].T.contiguous()
    components = tuple(contract["k12"]["component_ids"])
    layer_indices = {
        str(layer): [
            index
            for index, component in enumerate(components)
            if int(component.split(".")[0].split("_")[1]) == layer
        ]
        for layer in contract["k12"]["layers"]
    }

    rows = []
    head_rows: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    probe_rows: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for concept in sorted(contract["conditions"]["pairs"]):
        tensors = load_file(SHARD_DIR / f"{concept}.safetensors")
        mask = tensors["response_mask"].bool()
        natural = {
            name: state(tensors, f"natural_{name}")
            for name in contract["execution"]["natural_states"]
        }
        for direction, specification in contract["directions"].items():
            target = natural[specification["target"]]
            donor = natural[specification["donor"]]
            normal = natural[specification["normal_control"]]
            different = natural[specification["different_control"]]
            output = state(tensors, f"intervention_{direction}.primary_donor")
            endpoint_states = {
                "target": target,
                "donor": donor,
                "normal": normal,
                "different": different,
            }

            mean_states = {
                name: mean_monitor(value["monitor"], mask)
                for name, value in {**endpoint_states, "output": output}.items()
            }
            row_states = {
                name: project(value, basis) for name, value in mean_states.items()
            }
            null_states = {
                name: mean_states[name] - row_states[name] for name in mean_states
            }
            spaces = {
                "monitor_mean": mean_states,
                "probe_row": row_states,
                "probe_null": null_states,
                "probe_margin": {
                    name: value["margins"]
                    for name, value in {**endpoint_states, "output": output}.items()
                },
            }
            space_metrics = {}
            for space, values in spaces.items():
                nearest, distances = endpoint_distances(
                    values["output"],
                    {name: values[name] for name in endpoint_states},
                )
                trajectory = values["donor"] - values["target"]
                changed = values["output"] - values["target"]
                recovery = float(
                    (changed.double() * trajectory.double()).sum()
                    / trajectory.double().square().sum().clamp(min=1e-8)
                )
                space_metrics[space] = {
                    "nearest_endpoint": nearest,
                    "distances": distances,
                    "donor_recovery": recovery,
                }

            donor_error = mean_states["output"] - mean_states["donor"]
            row_error = project(donor_error, basis)
            error_energy = float(donor_error.double().square().sum())
            row_error_fraction = float(
                row_error.double().square().sum() / max(error_energy, 1e-12)
            )
            donor_trajectory = mean_states["donor"] - mean_states["target"]
            row_trajectory = project(donor_trajectory, basis)
            trajectory_energy = float(donor_trajectory.double().square().sum())
            row_trajectory_fraction = float(
                row_trajectory.double().square().sum() / max(trajectory_energy, 1e-12)
            )

            for index, component in enumerate(components):
                metrics = vector_metrics(
                    output["k12"][:, :, index],
                    target["k12"][:, :, index],
                    donor["k12"][:, :, index],
                    mask,
                )
                output_vectors = masked_vectors(output["k12"][:, :, index], mask)
                donor_vectors = masked_vectors(donor["k12"][:, :, index], mask)
                trajectory_vectors = masked_vectors(
                    donor["k12"][:, :, index] - target["k12"][:, :, index], mask
                )
                error_ratio = np.mean(
                    [
                        float((actual - source).square().sum())
                        / max(float(trajectory.square().sum()), 1e-12)
                        for actual, source, trajectory in zip(
                            output_vectors,
                            donor_vectors,
                            trajectory_vectors,
                            strict=True,
                        )
                    ]
                )
                head_rows[(direction, component)].append(
                    {
                        "recovery": float(metrics["recovery"].mean()),
                        "donor_error_to_trajectory_energy_ratio": float(error_ratio),
                    }
                )

            natural_delta = donor["margins"] - target["margins"]
            output_delta = output["margins"] - target["margins"]
            for index, probe_name in enumerate(probe_names):
                denominator = (
                    natural_delta[:, index].double().square().sum().clamp(min=1e-8)
                )
                recovery = float(
                    (
                        output_delta[:, index].double()
                        * natural_delta[:, index].double()
                    ).sum()
                    / denominator
                )
                direction_correct = float(
                    (output_delta[:, index] * natural_delta[:, index] > 0)
                    .float()
                    .mean()
                )
                probe_rows[(direction, probe_name)].append(
                    {
                        "recovery": recovery,
                        "direction_correct_fraction": direction_correct,
                    }
                )

            rows.append(
                {
                    "concept": concept,
                    "direction": direction,
                    "space_metrics": space_metrics,
                    "probe_row_fraction_of_donor_error_energy": row_error_fraction,
                    "probe_row_fraction_of_natural_trajectory_energy": (
                        row_trajectory_fraction
                    ),
                    "per_layer_k12_donor_recovery": {
                        layer: float(
                            vector_metrics(
                                output["k12"][:, :, indices],
                                target["k12"][:, :, indices],
                                donor["k12"][:, :, indices],
                                mask,
                            )["recovery"].mean()
                        )
                        for layer, indices in layer_indices.items()
                    },
                }
            )

    direction_summaries = {}
    for direction in contract["directions"]:
        selected = [row for row in rows if row["direction"] == direction]
        direction_summaries[direction] = {
            "nearest_endpoint_counts_by_space": {
                space: dict(
                    Counter(
                        row["space_metrics"][space]["nearest_endpoint"]
                        for row in selected
                    )
                )
                for space in (
                    "monitor_mean",
                    "probe_row",
                    "probe_null",
                    "probe_margin",
                )
            },
            "median_probe_row_fraction_of_donor_error_energy": float(
                np.median(
                    [
                        row["probe_row_fraction_of_donor_error_energy"]
                        for row in selected
                    ]
                )
            ),
            "median_probe_row_fraction_of_natural_trajectory_energy": float(
                np.median(
                    [
                        row["probe_row_fraction_of_natural_trajectory_energy"]
                        for row in selected
                    ]
                )
            ),
            "failed_probe_margin_concepts": [
                {
                    "concept": row["concept"],
                    "nearest_endpoint": row["space_metrics"]["probe_margin"][
                        "nearest_endpoint"
                    ],
                    "k12_layer_recovery": row["per_layer_k12_donor_recovery"],
                    "monitor_mean_nearest_endpoint": row["space_metrics"][
                        "monitor_mean"
                    ]["nearest_endpoint"],
                    "probe_row_nearest_endpoint": row["space_metrics"]["probe_row"][
                        "nearest_endpoint"
                    ],
                    "probe_null_nearest_endpoint": row["space_metrics"]["probe_null"][
                        "nearest_endpoint"
                    ],
                    "probe_row_fraction_of_donor_error_energy": row[
                        "probe_row_fraction_of_donor_error_energy"
                    ],
                }
                for row in selected
                if row["space_metrics"]["probe_margin"]["nearest_endpoint"] != "donor"
            ],
        }

    head_summary = [
        {
            "direction": direction,
            "component": component,
            "mean_concept_recovery": concept_average(values, "recovery"),
            "mean_concept_donor_error_to_trajectory_energy_ratio": concept_average(
                values, "donor_error_to_trajectory_energy_ratio"
            ),
        }
        for (direction, component), values in sorted(head_rows.items())
    ]
    probe_summary = [
        {
            "direction": direction,
            "probe": probe,
            "mean_concept_recovery": concept_average(values, "recovery"),
            "mean_concept_direction_correct_fraction": concept_average(
                values, "direction_correct_fraction"
            ),
        }
        for (direction, probe), values in sorted(probe_rows.items())
    ]
    result = {
        "schema_version": 1,
        "procedure": "day53-existing-artifact-divergence-decomposition-v1",
        "evidence_class": "post_hoc_existing_day52_development_artifact_diagnostic",
        "parent_contract_sha256": sha256_file(CONTRACT_PATH),
        "parent_manifest_sha256": sha256_file(MANIFEST_PATH),
        "parent_manifest_verified": True,
        "probe_row_rank": rank,
        "monitor_width": int(weights.shape[1]),
        "probe_row_condition_number": float(
            singular_values.max() / singular_values.min()
        ),
        "direction_summaries": direction_summaries,
        "per_head": head_summary,
        "per_probe": probe_summary,
        "per_concept": rows,
        "next_decisive_experiment": (
            "reciprocal exact natural donor K12 replacement in fixed target context"
        ),
    }
    write_json(OUTPUT_PATH, result)


if __name__ == "__main__":
    main()
