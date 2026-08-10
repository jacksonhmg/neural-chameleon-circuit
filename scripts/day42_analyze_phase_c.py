#!/usr/bin/env python3
"""Audit, reduce, and adjudicate the frozen Phase C contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from day42_run_phase_c import (  # noqa: E402
    CONTRACT_PATH,
    PARAMETERS_PATH,
    PREFLIGHT_PATH,
    RAW_PATH,
)
from neural_chameleon.semantic_conditioning import (  # noqa: E402
    fixed_rademacher_projection_blocks,
    ridge_fit_predict,
)


OUTPUT_DIR = ROOT / "results/day-43"
PROBE_SCALE_PATH = ROOT / "results/day-33/intermediate-prediction-summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


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
        raise RuntimeError(f"{relative} differs from Phase C analysis commit")


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def vector_recovery(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        1.0
        - np.square(predicted - target).sum()
        / max(float(np.square(target).sum()), 1e-6)
    )


def concept_bootstrap(
    per_concept: Mapping[str, float], *, seed: int, replicates: int
) -> dict[str, Any]:
    concepts = sorted(per_concept)
    values = np.asarray(
        [per_concept[concept] for concept in concepts], dtype=np.float64
    )
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(concepts), size=(replicates, len(concepts)))
    boot = values[samples].mean(axis=1)
    return {
        "point": float(values.mean()),
        "one_sided_95_lower": float(np.quantile(boot, 0.05)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "two_sided_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "per_concept": dict(sorted(per_concept.items())),
    }


def ratio_bootstrap(
    numerators: Mapping[str, float],
    denominators: Mapping[str, float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    concepts = sorted(numerators)
    numerator = np.asarray([numerators[value] for value in concepts])
    denominator = np.asarray([denominators[value] for value in concepts])
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(concepts), size=(replicates, len(concepts)))
    boot = numerator[samples].sum(axis=1) / np.maximum(
        denominator[samples].sum(axis=1), 1e-6
    )
    point = float(numerator.sum() / max(float(denominator.sum()), 1e-6))
    return {
        "point": point,
        "one_sided_95_lower": float(np.quantile(boot, 0.05)),
        "one_sided_95_upper": float(np.quantile(boot, 0.95)),
        "two_sided_95_interval": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "per_concept": {
            concept: float(numerators[concept] / max(denominators[concept], 1e-6))
            for concept in concepts
        },
    }


def implementation_audit(
    rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint = [row for row in rows if row["record_type"] == "condition_endpoint"]
    causal = [row for row in rows if row["record_type"] == "causal_effect"]
    expected_ids = {
        value
        for values in contract["population"]["example_ids"].values()
        for value in values
    }
    endpoint_keys = {(row["example_id"], row["condition"]) for row in endpoint}
    causal_keys = {
        (row["example_id"], row["source_condition"], row["path"]) for row in causal
    }
    expected_endpoint = {
        (example_id, condition)
        for example_id in expected_ids
        for condition in contract["conditions"]["order"]
    }
    expected_causal = {
        (example_id, source, path)
        for example_id in expected_ids
        for source in contract["expected_execution_matrix"]["source_conditions"]
        for path in contract["expected_execution_matrix"]["paths"]
    }
    finite = all(
        np.isfinite(row["mean_raw_margins"]).all()
        and np.isfinite(row["sequence_scores"]).all()
        and np.isfinite(row["activation_rms"])
        and (
            row["record_type"] != "condition_endpoint"
            or (
                np.isfinite(row["pooled_k12"]).all()
                and (
                    row["upstream_concept_mean"] is None
                    or np.isfinite(row["upstream_concept_mean"]).all()
                )
                and (
                    row["normal_response_mean"] is None
                    or np.isfinite(row["normal_response_mean"]).all()
                )
            )
        )
        for row in rows
    )
    expected_concepts = {
        example_id: concept
        for concept, example_ids in contract["population"]["example_ids"].items()
        for example_id in example_ids
    }

    def vector_width(value: Any, width: int) -> bool:
        return isinstance(value, list) and len(value) == width

    endpoint_payload_shapes = all(
        vector_width(row["mean_raw_margins"], 13)
        and vector_width(row["sequence_scores"], 13)
        and vector_width(row["pooled_k12"], 12 * 256)
        and (
            (row["condition"] == "normal" and row["upstream_concept_mean"] is None)
            or (
                row["condition"] != "normal"
                and vector_width(row["upstream_concept_mean"], 3584)
            )
        )
        and (
            (
                row["condition"] == "normal"
                and vector_width(row["normal_response_mean"], 3584)
            )
            or (row["condition"] != "normal" and row["normal_response_mean"] is None)
        )
        for row in endpoint
    )
    causal_payload_shapes = all(
        vector_width(row["mean_raw_margins"], 13)
        and vector_width(row["sequence_scores"], 13)
        for row in causal
    )
    preflight = read_json(PREFLIGHT_PATH)
    checks = {
        "exact_total_rows": len(rows)
        == int(contract["expected_execution_matrix"]["total_rows"]),
        "exact_endpoint_rows": len(endpoint)
        == int(contract["expected_execution_matrix"]["condition_endpoint_rows"]),
        "exact_causal_rows": len(causal)
        == int(contract["expected_execution_matrix"]["causal_effect_rows"]),
        "endpoint_matrix_exact": endpoint_keys == expected_endpoint
        and len(endpoint_keys) == len(endpoint),
        "causal_matrix_exact": causal_keys == expected_causal
        and len(causal_keys) == len(causal),
        "selected_ids_exact": {row["example_id"] for row in rows} == expected_ids,
        "concept_and_label_mapping_exact": all(
            expected_concepts.get(row["example_id"]) == row["concept"]
            and int(row["label"]) == 1
            for row in rows
        ),
        "probe_order_and_width_exact": bool(rows)
        and len({tuple(row["probe_names"]) for row in rows}) == 1
        and len(rows[0]["probe_names"]) == 13,
        "endpoint_payload_shapes_exact": endpoint_payload_shapes,
        "causal_payload_shapes_exact": causal_payload_shapes,
        "one_execution_commit": {row["execution_commit"] for row in rows}
        == {parameters["execution_commit"]},
        "one_execution_id": {row["execution_id"] for row in rows}
        == {parameters["execution_id"]},
        "contract_hash_exact": {row["contract_sha256"] for row in rows}
        == {sha256_file(CONTRACT_PATH)},
        "all_values_finite": finite,
        "preflight_pass": preflight["result"] == "pass"
        and preflight["preflight_commit"] == parameters["execution_commit"]
        and preflight["contract_sha256"] == sha256_file(CONTRACT_PATH),
        "preflight_hash_exact": parameters["preflight_sha256"]
        == sha256_file(PREFLIGHT_PATH),
        "matched_pair_audit_frozen": contract["conditions"]["pair_audit"][
            "all_104_rendered_prompt_pairs_token_count_matched"
        ],
        "development_only": all(
            row["evidence_class"].startswith("existing-data") for row in rows
        ),
    }
    return {
        "schema_version": 1,
        "procedure": "post-Gate-1 Phase C implementation audit",
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "result": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "observed_rows": {
            "condition_endpoint_rows": len(endpoint),
            "causal_effect_rows": len(causal),
            "total_rows": len(rows),
        },
        "execution_commit": parameters["execution_commit"],
        "execution_id": parameters["execution_id"],
    }


def analyze(
    rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    endpoint = {
        (row["example_id"], row["condition"]): row
        for row in rows
        if row["record_type"] == "condition_endpoint"
    }
    causal = {
        (row["example_id"], row["source_condition"], row["path"]): row
        for row in rows
        if row["record_type"] == "causal_effect"
    }
    example_concept = {row["example_id"]: row["concept"] for row in rows}
    example_ids = sorted(example_concept)
    scale = np.asarray(read_json(PROBE_SCALE_PATH)["probe_standardization"])

    def margin(row: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(row["mean_raw_margins"], dtype=np.float64) / scale

    def effect(example_id: str, source: str, path: str) -> np.ndarray:
        return margin(causal[(example_id, source, path)]) - margin(
            endpoint[(example_id, "correct_trigger")]
        )

    normal_k12 = np.stack(
        [endpoint[(example_id, "normal")]["pooled_k12"] for example_id in example_ids]
    )
    writer_scale = normal_k12.std(axis=0)
    writer_scale[writer_scale <= 1e-6] = 1e-6
    recoveries: dict[str, dict[str, list[float]]] = {
        key: defaultdict(list)
        for key in ("direct", "total", "frontier_F3", "pooled_k12")
    }
    irrelevant_norms: dict[str, list[float]] = defaultdict(list)
    different_norms: dict[str, list[float]] = defaultdict(list)
    operational_natural: dict[str, list[float]] = defaultdict(list)
    operational_hidden: dict[str, list[float]] = defaultdict(list)
    probe_names = endpoint[(example_ids[0], "normal")]["probe_names"]
    for example_id in example_ids:
        concept = example_concept[example_id]
        for path in ("direct", "total", "frontier_F3"):
            natural = effect(example_id, "different_trigger", path)
            hidden = effect(example_id, "hidden_different_substitution", path)
            recoveries[path][concept].append(vector_recovery(hidden, natural))
        correct_k12 = np.asarray(
            endpoint[(example_id, "correct_trigger")]["pooled_k12"]
        )
        different_k12 = np.asarray(
            endpoint[(example_id, "different_trigger")]["pooled_k12"]
        )
        hidden_k12 = np.asarray(
            endpoint[(example_id, "hidden_different_substitution")]["pooled_k12"]
        )
        recoveries["pooled_k12"][concept].append(
            vector_recovery(
                (hidden_k12 - correct_k12) / writer_scale,
                (different_k12 - correct_k12) / writer_scale,
            )
        )
        irrelevant_norms[concept].append(
            float(np.linalg.norm(effect(example_id, "irrelevant_trigger", "direct")))
        )
        different_norms[concept].append(
            float(np.linalg.norm(effect(example_id, "different_trigger", "direct")))
        )
        probe_index = probe_names.index(concept)
        correct_margin = margin(endpoint[(example_id, "correct_trigger")])[probe_index]
        operational_natural[concept].append(
            float(
                margin(endpoint[(example_id, "different_trigger")])[probe_index]
                - correct_margin
            )
        )
        operational_hidden[concept].append(
            float(
                margin(endpoint[(example_id, "hidden_different_substitution")])[
                    probe_index
                ]
                - correct_margin
            )
        )
    per_concept_recovery = {
        metric: {
            concept: float(np.mean(values))
            for concept, values in sorted(concepts.items())
        }
        for metric, concepts in recoveries.items()
    }
    bootstrap_seed = int(
        contract["metrics"]["bootstrap"].split("seed ")[1].split(";")[0]
    )
    recovery_summaries = {
        metric: concept_bootstrap(values, seed=bootstrap_seed + index, replicates=10000)
        for index, (metric, values) in enumerate(sorted(per_concept_recovery.items()))
    }
    irrelevant_mean = {
        concept: float(np.mean(values)) for concept, values in irrelevant_norms.items()
    }
    different_mean = {
        concept: float(np.mean(values)) for concept, values in different_norms.items()
    }
    specificity = ratio_bootstrap(
        irrelevant_mean,
        different_mean,
        seed=bootstrap_seed + 20,
        replicates=10000,
    )

    upstream = np.stack(
        [
            np.asarray(
                endpoint[(example_id, "different_trigger")]["upstream_concept_mean"]
            )
            - np.asarray(
                endpoint[(example_id, "correct_trigger")]["upstream_concept_mean"]
            )
            for example_id in example_ids
        ]
    )
    normal = np.stack(
        [
            endpoint[(example_id, "normal")]["normal_response_mean"]
            for example_id in example_ids
        ]
    )
    projection = contract["features_and_prediction"]["projection"]
    dimension = int(projection["dimension_per_feature_block"])
    projection_seed = int(projection["seed"])
    projected_upstream, projected_normal = fixed_rademacher_projection_blocks(
        (upstream, normal),
        output_dimension=dimension,
        seed=projection_seed,
    )
    full_x = np.concatenate([projected_upstream, projected_normal], axis=1)
    target = np.stack(
        [
            effect(example_id, "different_trigger", "direct")
            for example_id in example_ids
        ]
    )
    concepts = np.asarray([example_concept[value] for value in example_ids])
    full_prediction = np.empty_like(target)
    baseline_prediction = np.empty_like(target)
    ridge_lambda = float(contract["features_and_prediction"]["ridge_lambda"])
    for heldout in sorted(set(concepts)):
        test = concepts == heldout
        train = ~test
        full_prediction[test] = ridge_fit_predict(
            full_x[train], target[train], full_x[test], ridge_lambda=ridge_lambda
        )
        baseline_prediction[test] = ridge_fit_predict(
            projected_normal[train],
            target[train],
            projected_normal[test],
            ridge_lambda=ridge_lambda,
        )
    full_num: dict[str, float] = {}
    baseline_num: dict[str, float] = {}
    denominator: dict[str, float] = {}
    full_per_concept: dict[str, float] = {}
    baseline_per_concept: dict[str, float] = {}
    for concept in sorted(set(concepts)):
        test = concepts == concept
        train = ~test
        target_mean = target[train].mean(axis=0)
        full_num[concept] = float(np.square(full_prediction[test] - target[test]).sum())
        baseline_num[concept] = float(
            np.square(baseline_prediction[test] - target[test]).sum()
        )
        denominator[concept] = float(np.square(target[test] - target_mean).sum())
        full_per_concept[concept] = full_num[concept] / max(denominator[concept], 1e-6)
        baseline_per_concept[concept] = baseline_num[concept] / max(
            denominator[concept], 1e-6
        )
    prediction_full = concept_bootstrap(
        full_per_concept, seed=bootstrap_seed + 30, replicates=10000
    )
    prediction_baseline = concept_bootstrap(
        baseline_per_concept, seed=bootstrap_seed + 31, replicates=10000
    )
    improvement = {
        concept: full_per_concept[concept] - baseline_per_concept[concept]
        for concept in full_per_concept
    }
    prediction_difference = concept_bootstrap(
        improvement, seed=bootstrap_seed + 32, replicates=10000
    )
    gate = contract["scientific_continue_gate"]
    clauses = {
        "full_predictor_snmse": prediction_full["one_sided_95_upper"]
        <= float(
            gate["upstream_prediction"][
                "full_predictor_one_sided_95_upper_snmse_at_most"
            ]
        ),
        "full_predictor_beats_baseline": prediction_difference["one_sided_95_upper"]
        <= float(
            gate["upstream_prediction"][
                "full_minus_baseline_one_sided_95_upper_at_most"
            ]
        ),
        "concept_prediction_stability": sum(
            full_per_concept[concept] < baseline_per_concept[concept]
            for concept in full_per_concept
        )
        >= int(
            gate["upstream_prediction"][
                "concepts_with_full_snmse_better_than_baseline_at_least"
            ]
        ),
        "hidden_direct_recovery": recovery_summaries["direct"]["one_sided_95_lower"]
        >= float(
            gate["hidden_substitution"][
                "direct_probe_vector_recovery_one_sided_95_lower_at_least"
            ]
        ),
        "hidden_total_recovery": recovery_summaries["total"]["one_sided_95_lower"]
        >= float(
            gate["hidden_substitution"][
                "total_probe_vector_recovery_one_sided_95_lower_at_least"
            ]
        ),
        "hidden_frontier_recovery": recovery_summaries["frontier_F3"][
            "one_sided_95_lower"
        ]
        >= float(
            gate["hidden_substitution"][
                "frontier_probe_vector_recovery_one_sided_95_lower_at_least"
            ]
        ),
        "hidden_k12_recovery": recovery_summaries["pooled_k12"]["one_sided_95_lower"]
        >= float(
            gate["hidden_substitution"][
                "pooled_k12_pattern_recovery_one_sided_95_lower_at_least"
            ]
        ),
        "irrelevant_control": specificity["one_sided_95_upper"]
        <= float(
            gate["specificity"][
                "irrelevant_to_different_direct_effect_ratio_one_sided_95_upper_at_most"
            ]
        ),
        "hidden_sign_stability": sum(
            value > 0 for value in per_concept_recovery["direct"].values()
        )
        >= 9,
    }
    pass_all = all(clauses.values())
    return {
        "schema_version": 1,
        "procedure": "post-Gate-1 Phase C semantic-conditioning reduction",
        "analysis_commit": git_head(),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "evidence_class": "existing-data prospective semantic-conditioning development evidence",
        "prediction": {
            "full_predictor_snmse": prediction_full,
            "normal_only_baseline_snmse": prediction_baseline,
            "full_minus_baseline_snmse": prediction_difference,
            "concepts_full_better_than_baseline": sum(
                full_per_concept[concept] < baseline_per_concept[concept]
                for concept in full_per_concept
            ),
        },
        "hidden_substitution_recovery": recovery_summaries,
        "irrelevant_to_different_direct_effect_ratio": specificity,
        "operational_own_probe_margin_change": {
            "natural_different_trigger": {
                concept: float(np.mean(values))
                for concept, values in sorted(operational_natural.items())
            },
            "hidden_different_substitution": {
                concept: float(np.mean(values))
                for concept, values in sorted(operational_hidden.items())
            },
        },
        "gate_clauses": clauses,
        "scientific_continue_gate": "pass" if pass_all else "fail",
        "consequence": gate["pass_consequence" if pass_all else "fail_consequence"],
    }


def main() -> None:
    args = parse_args()
    commit = git_head()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/day42_run_phase_c.py",
        ROOT / "scripts/day42_verify_phase_c.py",
        ROOT / "src/neural_chameleon/semantic_conditioning.py",
        CONTRACT_PATH,
        PROBE_SCALE_PATH,
    ):
        require_committed(path, commit)
    contract = read_json(CONTRACT_PATH)
    parameters = read_json(PARAMETERS_PATH)
    rows = load_jsonl(RAW_PATH)
    audit = implementation_audit(rows, contract, parameters)
    write_json(args.output_dir / "phase-c-audit.json", audit)
    if audit["result"] != "pass":
        raise RuntimeError("Phase C implementation audit failed")
    summary = analyze(rows, contract)
    write_json(args.output_dir / "semantic-conditioning-summary.json", summary)
    manifest = {
        "schema_version": 1,
        "procedure": "post-Gate-1 Phase C artifact manifest",
        "analysis_commit": commit,
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "execution_parameters_sha256": sha256_file(PARAMETERS_PATH),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "raw_artifact": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": len(rows),
            "bytes": RAW_PATH.stat().st_size,
            "sha256": sha256_file(RAW_PATH),
        },
        "scientific_continue_gate": summary["scientific_continue_gate"],
        "evidence_class": summary["evidence_class"],
    }
    write_json(args.output_dir / "phase-c-artifact-manifest.json", manifest)


if __name__ == "__main__":
    main()
