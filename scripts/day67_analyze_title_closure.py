#!/usr/bin/env python3
"""Reduce the one-shot operational title-closure experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon.final_title_gate import recursive_numeric_max_difference, title_gate_disposition  # noqa: E402


CONTRACT_PATH = ROOT / "results/day-68/frozen-title-closure-execution-contract.json"
DAY65_SUMMARY_PATH = ROOT / "results/day-65/trained-final-chain-summary.json"
SHARD_DIR = ROOT / "artifacts/title-closure-v2/final"
SUMMARY_PATH = ROOT / "results/day-67/title-closure-summary.json"
VERIFICATION_PATH = ROOT / "results/day-67/title-closure-rereduction-verification.json"
MANIFEST_PATH = ROOT / "results/day-67/title-closure-artifact-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bootstrap(values: Sequence[float], draws: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(draws, len(array)))
    estimates = array[indices].mean(1)
    return {
        "point": float(array.mean()),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
    }


def checked_concept(concept: str, contract_hash: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensor_path, metadata_path = SHARD_DIR / f"{concept}.safetensors", SHARD_DIR / f"{concept}.json"
    metadata = read_json(metadata_path)
    if (
        metadata["contract_sha256"] != contract_hash
        or metadata["tensor_sha256"] != sha256_file(tensor_path)
        or metadata["concept"] != concept
        or metadata["patched_component_count"] != 12
    ):
        raise RuntimeError(f"final positive shard provenance differs: {concept}")
    tensors = load_file(tensor_path)
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError(f"nonfinite final positive shard: {concept}")
    return tensors, metadata


def checked_negative(contract_hash: str) -> tuple[torch.Tensor, dict[str, Any]]:
    tensor_path, metadata_path = SHARD_DIR / "final-negative.safetensors", SHARD_DIR / "final-negative.json"
    metadata = read_json(metadata_path)
    if metadata["contract_sha256"] != contract_hash or metadata["tensor_sha256"] != sha256_file(tensor_path):
        raise RuntimeError("final negative provenance differs")
    return load_file(tensor_path)["normal.margins"].float(), metadata


def reduce_all() -> dict[str, Any]:
    contract = read_json(CONTRACT_PATH)
    contract_hash = sha256_file(CONTRACT_PATH)
    day65 = read_json(DAY65_SUMMARY_PATH)
    if contract["parents"]["results/day-65/trained-final-chain-summary.json"] != sha256_file(DAY65_SUMMARY_PATH):
        raise RuntimeError("Day 65 parent differs")
    if not contract["control_verifier_repair"]["pass"]:
        raise RuntimeError("repaired control audit did not pass")
    names = contract["probe_names_in_order"]
    negative, negative_metadata = checked_negative(contract_hash)
    assignments = list(zip(
        negative_metadata["assignment_probe_concepts"],
        negative_metadata["assignment_unique_indices"],
    ))
    per_concept: dict[str, Any] = {}
    effects = {nominal: {name: [] for name in ("drop", "clamp")} for nominal in ("0.01", "0.05")}
    identity_audits = {}
    for concept in contract["concepts_in_order"]:
        tensors, metadata = checked_concept(concept, contract_hash)
        probe_index = names.index(concept)
        margins = {
            "correct": tensors["natural.correct.margins"][:, probe_index].float(),
            "irrelevant": tensors["natural.irrelevant.margins"][:, probe_index].float(),
            "clamp": tensors["clamp.margins"][:, probe_index].float(),
        }
        negative_indices = [int(index) for assigned, index in assignments if assigned == concept]
        negative_values = negative[negative_indices, probe_index]
        if len(negative_values) != 256 or len(margins["correct"]) != 128:
            raise RuntimeError(f"final sample size differs: {concept}")
        points = {}
        for nominal in ("0.01", "0.05"):
            threshold = float(contract["thresholds"][concept][nominal])
            rates = {name: float((values > threshold).float().mean()) for name, values in margins.items()}
            differences = {
                "drop": rates["irrelevant"] - rates["correct"],
                "clamp": rates["clamp"] - rates["correct"],
            }
            for name, value in differences.items():
                effects[nominal][name].append(value)
            points[nominal] = {
                "threshold": threshold,
                "realized_final_FPR": float((negative_values > threshold).float().mean()),
                "TPR": rates,
                "differences": differences,
            }
        natural_delta = tensors["natural.irrelevant.margins"].float() - tensors["natural.correct.margins"].float()
        clamp_delta = tensors["clamp.margins"].float() - tensors["natural.correct.margins"].float()
        k12_target = tensors["natural.irrelevant.k12"].float() - tensors["natural.correct.k12"].float()
        k12_clamp = tensors["clamp.k12"].float() - tensors["natural.correct.k12"].float()
        identity_audits[concept] = {
            "response_hashes_unique": len(set(metadata["response_hashes"])) == len(metadata["response_hashes"]),
            "k12_clamp_exact_max_abs": float((k12_clamp - k12_target).abs().max()),
            "finite": all(torch.isfinite(value).all().item() for value in tensors.values()),
        }
        per_concept[concept] = {
            "n_positive": 128,
            "n_negative": 256,
            "raw_margin": {
                name: {"mean": float(values.mean()), "median": float(values.median())}
                for name, values in margins.items()
            },
            "complete_probe_vector": {
                "natural_irrelevant_minus_correct_mean_norm": float(torch.linalg.vector_norm(natural_delta.double().mean(0))),
                "clamp_minus_correct_mean_norm": float(torch.linalg.vector_norm(clamp_delta.double().mean(0))),
            },
            "operating_points": points,
        }
    draws, seed = int(contract["uncertainty"]["draws"]), int(contract["uncertainty"]["seed"])
    aggregate = {
        nominal: {
            name: {
                **bootstrap(values, draws, seed + point_index * 100 + effect_index),
                "positive_concepts": sum(value > 0 for value in values),
                "values_in_concept_order": values,
            }
            for effect_index, (name, values) in enumerate(effects[nominal].items())
        }
        for point_index, nominal in enumerate(("0.01", "0.05"))
    }
    gate = contract["closure_gates"]
    operational_checks = {
        "macro_drop_1pct": aggregate["0.01"]["drop"]["point"] >= float(gate["macro_TPR_drop_correct_vs_irrelevant_at_1pct_FPR_min"]),
        "macro_drop_5pct": aggregate["0.05"]["drop"]["point"] >= float(gate["macro_TPR_drop_correct_vs_irrelevant_at_5pct_FPR_min"]),
        "breadth_drop_1pct": aggregate["0.01"]["drop"]["positive_concepts"] >= int(gate["concepts_positive_drop_at_each_threshold_min"]),
        "breadth_drop_5pct": aggregate["0.05"]["drop"]["positive_concepts"] >= int(gate["concepts_positive_drop_at_each_threshold_min"]),
        "macro_clamp_either": max(aggregate[nominal]["clamp"]["point"] for nominal in aggregate) >= float(gate["macro_TPR_exact_K12_clamp_reversal_at_either_threshold_min"]),
        "breadth_clamp_either": max(aggregate[nominal]["clamp"]["positive_concepts"] for nominal in aggregate) >= int(gate["concepts_positive_clamp_at_either_threshold_min"]),
        "prior_restoration_carried": day65["title_clauses"]["restoration"],
    }
    operational_pass = all(operational_checks.values())
    implementation_pass = all(
        row["response_hashes_unique"] and row["k12_clamp_exact_max_abs"] == 0.0 and row["finite"]
        for row in identity_audits.values()
    )
    prior_scientific = {
        key: bool(day65["title_clauses"][key])
        for key in ("acquisition", "operation", "semantic_conditioning", "necessity_sufficiency", "restoration")
    }
    clauses = {
        **prior_scientific,
        "endogenous_chain": all(prior_scientific.values()) and contract["control_verifier_repair"]["pass"] and implementation_pass,
        "operational_failure": operational_pass,
    }
    return {
        "schema_version": 1,
        "procedure": "day67-one-shot-operational-title-closure-reduction-v1",
        "contract_sha256": contract_hash,
        "prior_day65_title_disposition": day65["title_disposition"],
        "control_verifier_repair": contract["control_verifier_repair"],
        "operational_consequences": {"per_concept": per_concept, "aggregate": aggregate},
        "operational_checks": operational_checks,
        "operational_pass": operational_pass,
        "implementation_audits": identity_audits,
        "implementation_pass": implementation_pass,
        "title_clauses": clauses,
        "title_disposition": title_gate_disposition(clauses),
        "interpretation_policy": "single_attempt_conjunctive_no_compensatory_score",
    }


def main() -> None:
    first, second = reduce_all(), reduce_all()
    absolute, relative = recursive_numeric_max_difference(first, second)
    verification = {
        "schema_version": 1,
        "procedure": "day67-independent-local-rereduction-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "maximum_absolute_numeric_difference": absolute,
        "maximum_relative_numeric_difference": relative,
        "exact_reduction_match": absolute == 0.0 and relative == 0.0,
    }
    if not verification["exact_reduction_match"]:
        raise RuntimeError("independent title-closure reductions differ")
    write_json(SUMMARY_PATH, first)
    write_json(VERIFICATION_PATH, verification)
    artifacts = [path for path in sorted(SHARD_DIR.rglob("*")) if path.is_file()] + [SUMMARY_PATH, VERIFICATION_PATH]
    write_json(MANIFEST_PATH, {
        "schema_version": 1,
        "procedure": "day67-title-closure-artifact-manifest-v1",
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "files": [{"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in artifacts],
    })
    print(json.dumps({
        "title_disposition": first["title_disposition"],
        "title_clauses": first["title_clauses"],
        "operational_checks": first["operational_checks"],
        "aggregate": first["operational_consequences"]["aggregate"],
    }, indent=2))


if __name__ == "__main__":
    main()
