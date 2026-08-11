#!/usr/bin/env python3
"""Determine whether the Day 49 irrelevant QKV result is collapse or transfer."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "results/day-51/frozen-donor-identity-audit.json"
DAY49_MANIFEST = ROOT / "results/day-49/execution-artifact-manifest.json"
SHARD_DIR = ROOT / "artifacts/rapid-k12-upstream-v1/day49-shards"
SUMMARY_PATH = ROOT / "results/day-51/donor-identity-summary.json"
METRICS_PATH = ROOT / "results/day-51/donor-identity-example-metrics.json"
AUDIT_PATH = ROOT / "results/day-51/donor-identity-audit.json"
MANIFEST_PATH = ROOT / "results/day-51/execution-artifact-manifest.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from analysis commit {commit}")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value)}")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n"
    ).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def state(tensors: Mapping[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    return {
        field: tensors[f"{name}.{field}"].float()
        for field in ("k12", "monitor", "margins")
    }


def vector(value: torch.Tensor, mask: torch.Tensor, row: int) -> torch.Tensor:
    if value.ndim >= 3 and value.shape[:2] == mask.shape:
        return value[row, mask[row]].reshape(-1).double()
    return value[row].reshape(-1).double()


def endpoint_metrics(
    intervention: torch.Tensor,
    correct: torch.Tensor,
    normal: torch.Tensor,
    donor: torch.Tensor,
    different: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict[str, Any]]:
    rows = []
    for row in range(mask.shape[0]):
        values = {
            "intervention": vector(intervention, mask, row),
            "correct": vector(correct, mask, row),
            "normal": vector(normal, mask, row),
            "donor": vector(donor, mask, row),
            "different": vector(different, mask, row),
        }

        def squared(left: str, right: str) -> float:
            return float((values[left] - values[right]).square().sum())

        def recovery(endpoint: str) -> float:
            changed = values["intervention"] - values["correct"]
            target = values[endpoint] - values["correct"]
            return float((changed @ target) / target.square().sum().clamp(min=1e-8))

        correct_normal = squared("correct", "normal")
        donor_normal = squared("donor", "normal")
        donor_different = squared("donor", "different")
        distances = {
            endpoint: squared("intervention", endpoint)
            for endpoint in ("normal", "donor", "different")
        }
        scale = max(correct_normal, 1e-8)
        rows.append(
            {
                "distance_to_normal_sq": distances["normal"],
                "distance_to_donor_sq": distances["donor"],
                "distance_to_different_sq": distances["different"],
                "distance_to_normal_ratio": distances["normal"] / scale,
                "distance_to_donor_ratio": distances["donor"] / scale,
                "distance_to_different_ratio": distances["different"] / scale,
                "donor_vs_normal_preference": (distances["normal"] - distances["donor"])
                / max(donor_normal, 1e-8),
                "donor_vs_different_preference": (
                    distances["different"] - distances["donor"]
                )
                / max(donor_different, 1e-8),
                "natural_donor_separation_ratio": math.sqrt(
                    donor_normal / max(correct_normal, 1e-8)
                ),
                "donor_recovery": recovery("donor"),
                "normal_recovery": recovery("normal"),
                "nearest_endpoint": min(distances, key=distances.get),
            }
        )
    return rows


def validate_parent_manifest() -> list[dict[str, Any]]:
    manifest = read_json(DAY49_MANIFEST)
    failures = []
    for row in manifest["files"]:
        path = ROOT / row["path"]
        if (
            not path.exists()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            failures.append(row)
    if failures:
        raise RuntimeError(f"Day 49 manifest validation failed: {failures}")
    return manifest["files"]


def load_rows(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_files = validate_parent_manifest()
    rows = []
    tensor_hashes = []
    probe_name_sets = set()
    modalities = {
        "k12": "k12",
        "complete_monitor_residual": "monitor",
        "complete_13_probe_margin_vector": "margins",
    }
    for concept in sorted(contract["condition_pairs"]):
        safe_name = concept.replace("/", "_")
        tensor_path = SHARD_DIR / f"{safe_name}.safetensors"
        metadata_path = SHARD_DIR / f"{safe_name}.json"
        metadata = read_json(metadata_path)
        if metadata["tensor_sha256"] != sha256_file(tensor_path):
            raise RuntimeError(f"tensor hash differs for {concept}")
        tensor_hashes.append(metadata["tensor_sha256"])
        probe_names = tuple(metadata["probe_names"])
        probe_name_sets.add(probe_names)
        donor_concept = contract["condition_pairs"][concept]["irrelevant_trigger"]
        if donor_concept not in probe_names:
            raise RuntimeError(f"irrelevant donor {donor_concept} has no probe")
        tensors = load_file(tensor_path)
        mask = tensors["response_mask"].bool()
        states = {
            endpoint: state(tensors, state_name)
            for endpoint, state_name in {
                "correct": contract["recipient_state"],
                **contract["endpoints"],
                "intervention": contract["intervention_state"],
            }.items()
        }
        modality_metrics = {
            modality: endpoint_metrics(
                states["intervention"][field],
                states["correct"][field],
                states["normal"][field],
                states["donor"][field],
                states["different"][field],
                mask,
            )
            for modality, field in modalities.items()
        }
        donor_probe = probe_names.index(donor_concept)
        recipient_probe = probe_names.index(concept)
        for index, example_id in enumerate(metadata["example_ids"]):
            natural_source_delta = float(
                states["donor"]["margins"][index, donor_probe]
                - states["normal"]["margins"][index, donor_probe]
            )
            intervention_source_delta = float(
                states["intervention"]["margins"][index, donor_probe]
                - states["normal"]["margins"][index, donor_probe]
            )
            natural_recipient_delta = float(
                states["donor"]["margins"][index, recipient_probe]
                - states["normal"]["margins"][index, recipient_probe]
            )
            intervention_recipient_delta = float(
                states["intervention"]["margins"][index, recipient_probe]
                - states["normal"]["margins"][index, recipient_probe]
            )
            rows.append(
                {
                    "concept": concept,
                    "donor_concept": donor_concept,
                    "example_id": example_id,
                    "modalities": {
                        modality: values[index]
                        for modality, values in modality_metrics.items()
                    },
                    "source_probe": {
                        "natural_donor_minus_normal": natural_source_delta,
                        "intervention_minus_normal": intervention_source_delta,
                        "direction_correct": natural_source_delta
                        * intervention_source_delta
                        > 0,
                    },
                    "recipient_probe": {
                        "natural_donor_minus_normal": natural_recipient_delta,
                        "intervention_minus_normal": intervention_recipient_delta,
                        "direction_correct": natural_recipient_delta
                        * intervention_recipient_delta
                        > 0,
                    },
                }
            )
    return rows, {
        "parent_manifest_file_count": len(manifest_files),
        "tensor_hash_count": len(set(tensor_hashes)),
        "probe_name_sets": [list(value) for value in sorted(probe_name_sets)],
    }


def concept_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["concept"]].append(row)
    result = []
    scalar_keys = (
        "distance_to_normal_sq",
        "distance_to_donor_sq",
        "distance_to_different_sq",
        "distance_to_normal_ratio",
        "distance_to_donor_ratio",
        "distance_to_different_ratio",
        "donor_vs_normal_preference",
        "donor_vs_different_preference",
        "natural_donor_separation_ratio",
        "donor_recovery",
        "normal_recovery",
    )
    for concept, values in sorted(grouped.items()):
        if len(values) != 2:
            raise RuntimeError(f"concept {concept} does not have two examples")
        modalities = {}
        for modality in values[0]["modalities"]:
            metrics = {
                key: float(
                    np.mean([row["modalities"][modality][key] for row in values])
                )
                for key in scalar_keys
            }
            distances = {
                endpoint: metrics[f"distance_to_{endpoint}_sq"]
                for endpoint in ("normal", "donor", "different")
            }
            metrics["nearest_endpoint"] = min(distances, key=distances.get)
            metrics["donor_closer_than_normal"] = (
                distances["donor"] < distances["normal"]
            )
            metrics["donor_closer_than_different"] = (
                distances["donor"] < distances["different"]
            )
            modalities[modality] = metrics
        result.append(
            {
                "concept": concept,
                "donor_concept": values[0]["donor_concept"],
                "modalities": modalities,
                "source_probe_direction_correct": bool(
                    np.mean(
                        [
                            row["source_probe"]["natural_donor_minus_normal"]
                            * row["source_probe"]["intervention_minus_normal"]
                            for row in values
                        ]
                    )
                    > 0
                ),
                "recipient_probe_direction_correct": bool(
                    np.mean(
                        [
                            row["recipient_probe"]["natural_donor_minus_normal"]
                            * row["recipient_probe"]["intervention_minus_normal"]
                            for row in values
                        ]
                    )
                    > 0
                ),
            }
        )
    return result


def summarize_modality(
    modality: str, concepts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    values = [row["modalities"][modality] for row in concepts]
    return {
        "median_natural_donor_separation_ratio": float(
            np.median([row["natural_donor_separation_ratio"] for row in values])
        ),
        "median_donor_recovery": float(
            np.median([row["donor_recovery"] for row in values])
        ),
        "median_normal_recovery": float(
            np.median([row["normal_recovery"] for row in values])
        ),
        "median_donor_vs_normal_preference": float(
            np.median([row["donor_vs_normal_preference"] for row in values])
        ),
        "median_donor_vs_different_preference": float(
            np.median([row["donor_vs_different_preference"] for row in values])
        ),
        "donor_closer_than_normal_concepts": sum(
            row["donor_closer_than_normal"] for row in values
        ),
        "donor_closer_than_different_concepts": sum(
            row["donor_closer_than_different"] for row in values
        ),
        "normal_closer_than_donor_concepts": sum(
            not row["donor_closer_than_normal"] for row in values
        ),
        "nearest_endpoint_counts": {
            endpoint: sum(row["nearest_endpoint"] == endpoint for row in values)
            for endpoint in ("normal", "donor", "different")
        },
        "per_concept": [
            {
                "concept": concept["concept"],
                "donor_concept": concept["donor_concept"],
                **concept["modalities"][modality],
            }
            for concept in concepts
        ],
    }


def classify(
    modalities: Mapping[str, Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, bool]]:
    rule = contract["interpretive_rule"]
    decisive = (
        modalities["k12"],
        modalities["complete_13_probe_margin_vector"],
    )
    clauses = {
        "both_modalities_separable": all(
            row["median_natural_donor_separation_ratio"]
            >= float(rule["minimum_natural_donor_separation_ratio"])
            for row in decisive
        ),
        "both_modalities_donor_recovery": all(
            row["median_donor_recovery"] >= float(rule["minimum_median_donor_recovery"])
            for row in decisive
        ),
        "both_modalities_donor_closer_than_normal": all(
            row["donor_closer_than_normal_concepts"]
            >= int(rule["minimum_donor_closer_concepts"])
            for row in decisive
        ),
        "both_modalities_donor_closer_than_different": all(
            row["donor_closer_than_different_concepts"]
            >= int(rule["minimum_donor_closer_concepts"])
            for row in decisive
        ),
        "both_modalities_positive_donor_preferences": all(
            row["median_donor_vs_normal_preference"] > 0
            and row["median_donor_vs_different_preference"] > 0
            for row in decisive
        ),
        "both_modalities_normal_recovery": all(
            row["median_normal_recovery"]
            >= float(rule["minimum_median_donor_recovery"])
            for row in decisive
        ),
        "both_modalities_normal_closer_than_donor": all(
            row["normal_closer_than_donor_concepts"]
            >= int(rule["minimum_donor_closer_concepts"])
            for row in decisive
        ),
        "source_probe_direction_correct_10_of_13": sum(
            row["source_probe_direction_correct"] for row in concepts
        )
        >= int(rule["minimum_donor_closer_concepts"]),
    }
    reconfiguration = all(
        clauses[key]
        for key in (
            "both_modalities_separable",
            "both_modalities_donor_recovery",
            "both_modalities_donor_closer_than_normal",
            "both_modalities_donor_closer_than_different",
            "both_modalities_positive_donor_preferences",
        )
    )
    collapse = all(
        clauses[key]
        for key in (
            "both_modalities_separable",
            "both_modalities_normal_recovery",
            "both_modalities_normal_closer_than_donor",
        )
    )
    if reconfiguration:
        return "exploratory_donor_reconfiguration_supported", clauses
    if collapse:
        return "exploratory_normal_collapse_supported", clauses
    return "donor_identity_unresolved", clauses


def reduce_result(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    concepts = concept_rows(rows)
    modality_summaries = {
        modality: summarize_modality(modality, concepts)
        for modality in contract["modalities"]
    }
    disposition, clauses = classify(modality_summaries, concepts, contract)
    implementation_checks = {
        "parent_manifest_exact": audit_inputs["parent_manifest_file_count"] == 32,
        "thirteen_unique_tensor_hashes": audit_inputs["tensor_hash_count"] == 13,
        "exact_probe_order": len(audit_inputs["probe_name_sets"]) == 1
        and len(audit_inputs["probe_name_sets"][0]) == 13,
        "exact_example_rows": len(rows) == 26,
        "exact_concept_rows": len(concepts) == 13,
        "all_values_finite": all(
            math.isfinite(float(value))
            for row in rows
            for modality in row["modalities"].values()
            for key, value in modality.items()
            if key != "nearest_endpoint"
        ),
    }
    summary = {
        "schema_version": 1,
        "procedure": contract["procedure"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "analysis_commit": git_head(),
        "evidence_class": contract["evidence_class"],
        "candidate": contract["candidate"],
        "modality_summaries": modality_summaries,
        "source_probe_direction_correct_concepts": sum(
            row["source_probe_direction_correct"] for row in concepts
        ),
        "recipient_probe_direction_correct_concepts": sum(
            row["recipient_probe_direction_correct"] for row in concepts
        ),
        "interpretive_clauses": clauses,
        "disposition": disposition,
        "day49_disposition_unchanged": True,
        "next_step": (
            "freeze one prospective reciprocal donor-reconfiguration experiment"
            if disposition == "exploratory_donor_reconfiguration_supported"
            else "abandon the full-prefix controller hypothesis and return to direct K12 operation geometry"
            if disposition == "exploratory_normal_collapse_supported"
            else "existing same-response endpoints cannot decide donor identity; require a prospective donor-identifiable contrast before any controller claim"
        ),
    }
    audit = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-audit",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "analysis_commit": git_head(),
        "implementation_checks": implementation_checks,
        "implementation_pass": all(implementation_checks.values()),
        "observed": dict(audit_inputs),
        "two_in_memory_reductions_byte_identical": None,
    }
    metrics = {
        "schema_version": 1,
        "procedure": f"{contract['procedure']}-example-metrics",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "rows": list(rows),
    }
    return summary, audit, metrics


def output_manifest() -> dict[str, Any]:
    paths = (CONTRACT_PATH, SUMMARY_PATH, METRICS_PATH, AUDIT_PATH)
    return {
        "schema_version": 1,
        "procedure": "exploratory-day51-donor-identity-artifacts-v1",
        "analysis_commit": git_head(),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def main() -> None:
    commit = git_head()
    for path in (Path(__file__).resolve(), CONTRACT_PATH):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen_before_donor_identity_metrics":
        raise RuntimeError("Day 51 donor-identity audit is not frozen")
    first_rows, first_inputs = load_rows(contract)
    first = reduce_result(contract, first_rows, first_inputs)
    second_rows, second_inputs = load_rows(contract)
    second = reduce_result(contract, second_rows, second_inputs)
    if json_bytes(first) != json_bytes(second) or json_bytes(first_rows) != json_bytes(
        second_rows
    ):
        raise RuntimeError("two donor-identity reductions differ")
    summary, audit, metrics = first
    audit["two_in_memory_reductions_byte_identical"] = True
    write_json(SUMMARY_PATH, summary)
    write_json(METRICS_PATH, metrics)
    write_json(AUDIT_PATH, audit)
    write_json(MANIFEST_PATH, output_manifest())
    if not audit["implementation_pass"]:
        raise RuntimeError(f"Day 51 implementation audit failed: {audit}")


if __name__ == "__main__":
    main()
