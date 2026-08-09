#!/usr/bin/env python3
"""Deterministically reduce the sealed Gate 1 features into frozen Phase A outputs.

The script refuses to run from uncommitted source.  It reads no Phase B outcome
and writes only the four predeclared Phase A JSON artifacts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_chameleon import LinearProbe, load_experimental_split  # noqa: E402
from neural_chameleon.post_gate1_diagnostics import (  # noqa: E402
    PhaseAExample,
    cross_concept_probe_predictions,
    default_phase_a_fields,
    discovery_global_decile_means,
    fit_original_probe_scale,
    normalize_heads,
    probe_contrast,
    residualize_examples,
    run_residualized_diagnostic,
    validate_phase_a_examples,
    variance_decomposition,
    zero_residualized_diagnostic,
)


CONTRACT_PATH = ROOT / "results/day-36/frozen-phase-a-b-contract.json"
FEATURE_DIR = ROOT / "artifacts/mechanism-gate1-v1"
PROBE_DIR = ROOT / "external/minimal_neural_chameleons/probes"
PRIOR_SUMMARY = ROOT / "results/day-33/intermediate-prediction-summary.json"
ACQUISITION_SUMMARY = ROOT / "results/day-33/acquired-writer-summary.json"
OUTPUT_DIR = ROOT / "results/day-38"
ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


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
        raise RuntimeError(f"{relative} differs from execution commit {commit}")


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(encoded)


def load_probes() -> tuple[tuple[str, ...], tuple[LinearProbe, ...]]:
    paths = sorted(PROBE_DIR.glob("*_weights.pt"))
    names = tuple(path.name.removesuffix("_weights.pt") for path in paths)
    probes = tuple(LinearProbe.load(path) for path in paths)
    if len(probes) != 13 or len(set(names)) != 13:
        raise ValueError("expected the complete 13-probe family")
    return names, probes


def positive_records() -> list[dict[str, Any]]:
    records = (
        load_experimental_split("discovery")
        + load_experimental_split("validation")
        + load_experimental_split("safety-test")
    )
    return sorted(
        [record for record in records if int(record["label"]) == 1],
        key=lambda record: (record["split"], record["concept"], record["example_id"]),
    )


def load_examples(records: Sequence[Mapping[str, Any]]) -> list[PhaseAExample]:
    result = []
    execution_ids = set()
    commits = set()
    for record in records:
        example_id = record["example_id"]
        chameleon = torch.load(
            FEATURE_DIR / "chameleon" / f"{example_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        precursor = torch.load(
            FEATURE_DIR / "precursor" / f"{example_id}.pt",
            map_location="cpu",
            weights_only=True,
        )
        for artifact, model in ((chameleon, "chameleon"), (precursor, "precursor")):
            if artifact["example_id"] != example_id or artifact["model"] != model:
                raise ValueError("feature identity or checkpoint mismatch")
            if artifact["concept"] != record["concept"] or artifact["split"] != record["split"]:
                raise ValueError("feature data-role identity mismatch")
            execution_ids.add(artifact["execution_id"])
            commits.add(artifact["execution_commit"])
        if not torch.equal(chameleon["response_ids"], precursor["response_ids"]):
            raise ValueError("checkpoint response tokens differ")
        if tuple(chameleon["k12_head_ids"]) != tuple(precursor["k12_head_ids"]):
            raise ValueError("checkpoint K12 head order differs")
        result.append(
            PhaseAExample(
                example_id=example_id,
                concept=record["concept"],
                split=record["split"],
                k12=chameleon["k12_delta"].float(),
                nonselected=chameleon["nonselected_delta"].float(),
                precursor_k12=precursor["k12_delta"].float(),
                normal_state=chameleon["normal_resid_post_8"].float(),
                target_u=chameleon["target_u"].float(),
            )
        )
    if len(execution_ids) != 1 or len(commits) != 1:
        raise ValueError("Gate 1 feature artifacts have mixed execution identities")
    validate_phase_a_examples(result)
    return result


def fit_head_scale(examples: Sequence[PhaseAExample], field: str) -> torch.Tensor:
    values = torch.cat([getattr(example, field).float() for example in examples])
    return values.square().mean(dim=(0, 2)).sqrt().clamp(min=1e-6)


def semantic_embeddings(
    concepts: Sequence[str], model_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    index = read_json(model_path / "model.safetensors.index.json")
    shard = model_path / index["weight_map"]["model.embed_tokens.weight"]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        table = handle.get_tensor("model.embed_tokens.weight")
        result = {}
        token_ids = {}
        for concept in concepts:
            encoded = tokenizer(concept, add_special_tokens=False)["input_ids"]
            if not encoded:
                raise ValueError(f"literal concept tokenization is empty: {concept}")
            token_ids[concept] = list(map(int, encoded))
            result[concept] = table[torch.as_tensor(encoded)].float().mean(dim=0)
    return result, {
        "checkpoint": str(model_path.relative_to(ROOT)),
        "embedding_shard_sha256": sha256_file(shard),
        "literal_concept_token_ids": token_ids,
        "add_special_tokens": False,
    }


def metric_split_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    by_concept_r2 = metrics["per_concept_r2_u"]
    by_concept_snmse = metrics["per_concept_probe_vector_snmse"]
    validation = [concept for concept in by_concept_r2 if concept not in {"deception", "harmful"}]
    safety = [concept for concept in by_concept_r2 if concept in {"deception", "harmful"}]

    def aggregate(concepts: Sequence[str]) -> dict[str, float]:
        return {
            "macro_r2_u": sum(by_concept_r2[value] for value in concepts) / len(concepts),
            "macro_probe_vector_snmse": sum(by_concept_snmse[value] for value in concepts)
            / len(concepts),
        }

    return {"validation": aggregate(validation), "safety": aggregate(safety)}


def main() -> None:
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "src/neural_chameleon/post_gate1_diagnostics.py",
        CONTRACT_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    if contract["status"] != "frozen":
        raise RuntimeError("Phase A-B contract is not frozen")
    for label, expected in contract["input_sha256"].items():
        path_by_label = {
            "gate_1_audit": ROOT / "results/day-33/gate-1-audit.json",
            "acquisition_summary": ACQUISITION_SUMMARY,
            "intermediate_prediction_summary": PRIOR_SUMMARY,
        }
        if label in path_by_label and sha256_file(path_by_label[label]) != expected:
            raise RuntimeError(f"pinned input hash differs: {label}")

    records = positive_records()
    if len(records) != 866:
        raise RuntimeError(f"expected 866 positives, found {len(records)}")
    examples = load_examples(records)
    discovery = [example for example in examples if example.split == "discovery"]
    heldout = [example for example in examples if example.split != "discovery"]
    if len(discovery) != 256 or len(heldout) != 610:
        raise RuntimeError("Phase A population counts differ from the contract")

    names, probes = load_probes()
    prior = read_json(PRIOR_SUMMARY)
    probe_scale = torch.tensor(prior["probe_standardization"], dtype=torch.float32)
    recomputed_probe_scale = fit_original_probe_scale(discovery, probes)
    probe_scale_error = float((recomputed_probe_scale - probe_scale).abs().max())
    if probe_scale_error > 1e-6:
        raise RuntimeError("sealed probe standardization does not reproduce")
    acquisition = read_json(ACQUISITION_SUMMARY)
    head_ids = tuple(contract["component_sets"]["k12_ordered"])
    selected_scale = torch.tensor(
        [acquisition["head_rms_discovery_chameleon"][head_id] for head_id in head_ids]
    )
    nonselected_scale = fit_head_scale(discovery, "nonselected")
    normalized = [
        PhaseAExample(
            **{
                **example.__dict__,
                "k12": normalize_heads(example.k12, selected_scale),
                "precursor_k12": normalize_heads(example.precursor_k12, selected_scale),
                "nonselected": normalize_heads(example.nonselected, nonselected_scale),
            }
        )
        for example in examples
    ]
    discovery = [example for example in normalized if example.split == "discovery"]
    heldout = [example for example in normalized if example.split != "discovery"]

    fields = default_phase_a_fields()
    fit_centered, fit_audit = residualize_examples(discovery, discovery, fields)
    heldout_centered, heldout_audit = residualize_examples(discovery, heldout, fields)

    variance = {}
    variance_audit = {}
    variance_fields = {
        "u_original_space": lambda example: example.target_u,
        "standardized_complete_probe_vector": lambda example: probe_contrast(
            example.target_u, probes, probe_scale
        ),
    }
    for split_name, population in (
        ("validation", [row for row in heldout if row.split == "validation"]),
        ("safety-test", [row for row in heldout if row.split == "safety-test"]),
        ("heldout_combined", heldout),
    ):
        variance[split_name] = {}
        variance_audit[split_name] = {}
        for field_name, getter in variance_fields.items():
            fallback = discovery_global_decile_means(discovery, getter)
            result, audit = variance_decomposition(population, getter, fallback)
            variance[split_name][field_name] = result.to_dict()
            variance_audit[split_name][field_name] = audit.to_dict()

    diagnostic_results = {}
    specifications = (
        ("full_k12", "k12", False),
        ("normal_state_only", "k12", True),
        ("nonselected_heads", "nonselected", False),
        ("exact_precursor_k12", "precursor_k12", False),
    )
    for name, writer_field, normal_only in specifications:
        result = run_residualized_diagnostic(
            discovery,
            heldout,
            fit_centered,
            heldout_centered,
            probes,
            probe_scale,
            writer_field=writer_field,
            alphas=ALPHAS,
            pca_seed=int(contract["inference"]["pca_seed"]),
            normal_only=normal_only,
        ).to_dict()
        result["split_macros"] = metric_split_summary(result)
        diagnostic_results[name] = result
    zero = zero_residualized_diagnostic(
        heldout,
        heldout_centered,
        probes,
        probe_scale,
        pca_seed=int(contract["inference"]["pca_seed"]),
        fit_examples=discovery,
        fit_centered=fit_centered,
    ).to_dict()
    zero["split_macros"] = metric_split_summary(zero)
    diagnostic_results["zero_centered_concept_decile_oracle"] = zero

    precursor_path = ROOT / contract["models"]["exact_precursor"]["local_path"]
    embedding_features, embedding_audit = semantic_embeddings(
        sorted({example.concept for example in normalized}), precursor_path
    )
    cross = cross_concept_probe_predictions(
        normalized, probes, probe_scale, embedding_features, ALPHAS
    )

    common = {
        "schema_version": 1,
        "procedure": "frozen post-Gate-1 Phase A",
        "analysis_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_class": "existing-data descriptive re-identification; not fresh confirmation",
        "sealed_gate_1_result": "fail",
        "probe_order": list(names),
    }
    write_json(
        OUTPUT_DIR / "variance-decomposition-summary.json",
        {
            **common,
            "results": variance,
            "negative_label": {
                "available": False,
                "reason": "the sealed Gate 1 tensor archive stores u only for positives",
            },
        },
    )
    write_json(
        OUTPUT_DIR / "within-concept-diagnostic-summary.json",
        {
            **common,
            "results": diagnostic_results,
            "original_uncentered_results_unchanged": prior,
        },
    )
    write_json(
        OUTPUT_DIR / "cross-concept-diagnostic-summary.json",
        {**common, **cross, "semantic_embedding_audit": embedding_audit},
    )
    write_json(
        OUTPUT_DIR / "phase-a-leakage-audit.json",
        {
            **common,
            "result": "pass",
            "population": {
                "discovery_positive": len(discovery),
                "heldout_positive": len(heldout),
                "all_positive": len(normalized),
            },
            "fit_centering": fit_audit,
            "heldout_centering": heldout_audit,
            "variance_centering": variance_audit,
            "cross_concept_outer_folds": cross["outer_fold_audit"],
            "sealed_probe_standardization_max_abs_error": probe_scale_error,
            "nonselected_discovery_rms": nonselected_scale.tolist(),
            "phase_a_can_repair_gate_1": False,
        },
    )


if __name__ == "__main__":
    main()
