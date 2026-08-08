#!/usr/bin/env python3
"""Run the authorized full-model behavioral transport grid."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from neural_chameleon import (  # noqa: E402
    ActivationKind,
    LinearProbe,
    PatchSite,
    VectorizedTransplantRunner,
    delta_mapping_job,
    load_experimental_split,
    masked_token_kl,
)
from neural_chameleon.behavioral_transport import directional_logit_metrics  # noqa: E402
from day17_run_transfer_atlas import ANALYSIS_PLAN_PATH, PROBE_DIR, load_model, site_by_id  # noqa: E402


AUTHORIZATION_PATH = ROOT / "results/day-22/behavioral-transport-authorization.json"
RESULT_DIR = ROOT / "results/day-23"
WORKING_PATH = RESULT_DIR / "behavioral-transport-results.working.jsonl"
SEALED_PATH = RESULT_DIR / "behavioral-transport-results.jsonl.gz"
PREFLIGHT_PATH = RESULT_DIR / "behavioral-transport-preflight.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-chunk-size", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_committed(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def response_nll(result: Any) -> torch.Tensor:
    logprobs = result.response_token_logprobs()
    mask = result.response_mask
    return (-logprobs.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def batched(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def frozen_records(authorization: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {row["example_id"]: row for row in load_experimental_split("safety-test")}
    ids = authorization["examples"]["teacher_forced"]["example_ids"]
    if set(ids) - set(by_id):
        raise ValueError("authorization contains unknown teacher-forced example IDs")
    return [dict(by_id[example_id]) for example_id in ids]


def read_completed() -> dict[tuple[str, str], dict[str, Any]]:
    if not WORKING_PATH.is_file():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with WORKING_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["example_id"], row["condition_id"])
            if key in rows:
                raise ValueError(f"duplicate working row {key}")
            rows[key] = row
    return rows


def append_row(rows: dict[tuple[str, str], dict[str, Any]], row: dict[str, Any]) -> None:
    key = (row["example_id"], row["condition_id"])
    if key in rows:
        return
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with WORKING_PATH.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
    rows[key] = row


def intervention_jobs(
    base_name: str,
    selected_heads: Sequence[str],
    sites: Mapping[str, Any],
    normal_captures: Mapping[Any, Any],
    triggered_captures: Mapping[Any, Any],
    mappings: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], Any]]:
    base_captures = normal_captures if base_name == "normal" else triggered_captures
    direction = "induction" if base_name == "normal" else "rescue"
    sign = 1.0 if direction == "induction" else -1.0
    specifications: list[tuple[dict[str, Any], Any]] = []
    identity = {head_id: head_id for head_id in selected_heads}
    identity_id = f"delta:{base_name}:identity:selected_k12_identity"
    specifications.append(
        (
            {
                "condition_id": identity_id,
                "base_condition": base_name,
                "direction": direction,
                "source_role": "identity",
                "mapping_id": "selected_k12_identity",
                "mapping_class": "identity",
            },
            delta_mapping_job(
                identity_id,
                selected_heads,
                identity,
                sites,
                base_captures,
                normal_captures,
                triggered_captures,
                alpha=1.0,
                sign=sign,
            ),
        )
    )
    for mapping_spec in mappings:
        for source_role, key in (
            ("selected", "selected_destination_to_source"),
            ("null", "null_destination_to_source"),
        ):
            condition_id = f"delta:{base_name}:{source_role}:{mapping_spec['mapping_id']}"
            specifications.append(
                (
                    {
                        "condition_id": condition_id,
                        "base_condition": base_name,
                        "direction": direction,
                        "source_role": source_role,
                        "mapping_id": mapping_spec["mapping_id"],
                        "mapping_class": mapping_spec["mapping_class"],
                    },
                    delta_mapping_job(
                        condition_id,
                        selected_heads,
                        mapping_spec[key],
                        sites,
                        base_captures,
                        normal_captures,
                        triggered_captures,
                        alpha=1.0,
                        sign=sign,
                    ),
                )
            )
    if len(specifications) != 9:
        raise AssertionError("each destination base requires exactly nine jobs")
    return specifications


def run_preflight(runner: Any, authorization: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    expected = authorization["grid"]["teacher_forced_expected_rows"]
    checks = {
        "authorization_status": authorization.get("status") == "authorized-before-final-map-behavioral-outcomes",
        "mapping_count": len(authorization["mappings"]) == 4,
        "example_count": len(records) == 194,
        "expected_grid": expected == 3880,
        "registered_hooks": runner.registered_hook_count() == 0,
    }
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-day23-preflight",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "behavioral_outcomes_generated_during_preflight": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        raise RuntimeError("Day 23 preflight failed")


def seal(rows: Mapping[tuple[str, str], Mapping[str, Any]], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    counts: dict[str, int] = {}
    for example_id, _condition_id in rows:
        counts[example_id] = counts.get(example_id, 0) + 1
    if set(counts.values()) != {20}:
        raise ValueError("teacher-forced condition grid is incomplete")
    temporary = SEALED_PATH.with_suffix(SEALED_PATH.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in sorted(rows.values(), key=lambda item: (item["concept"], int(item["label"]), item["example_id"], item["condition_id"])):
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    temporary.replace(SEALED_PATH)


def main() -> None:
    args = parse_args()
    if args.group_chunk_size <= 0:
        raise ValueError("group chunk size must be positive")
    commit = git_head()
    for path in (Path(__file__).resolve(), AUTHORIZATION_PATH):
        require_committed(path, commit)
    authorization = json.loads(AUTHORIZATION_PATH.read_text())
    records = frozen_records(authorization)
    runner = load_model()
    if not args.skip_preflight:
        run_preflight(runner, authorization, records)
    if args.preflight_only:
        return
    analysis_plan = json.loads(ANALYSIS_PLAN_PATH.read_text())
    selected_heads = list(authorization["selected_heads"])
    population = set(selected_heads)
    for mapping_spec in authorization["mappings"]:
        population.update(mapping_spec["selected_destination_to_source"].values())
        population.update(mapping_spec["null_destination_to_source"].values())
    sites = site_by_id(sorted(population))
    capture_sites = tuple(sites.values())
    monitor_site = PatchSite(ActivationKind.BLOCK_OUTPUT, 12)
    completed = read_completed()
    authorization_sha256 = sha256_file(AUTHORIZATION_PATH)
    expected = int(authorization["grid"]["teacher_forced_expected_rows"])
    print(f"Resuming Day 23 with {len(completed)}/{expected} rows.", flush=True)
    for example_index, example in enumerate(records, start=1):
        concept = example["concept"]
        probe = LinearProbe.load(PROBE_DIR / f"{concept}_weights.pt")
        vector = VectorizedTransplantRunner(runner, probe, monitor_layer=12)
        pair = runner.prepare_pairs(
            [example["prompt"]],
            [example["response"]],
            analysis_plan["conditions"]["correct_triggers"][concept],
        )
        normal = runner.run(pair.normal, capture_sites=(*capture_sites, monitor_site), retain_response_logits=True)
        triggered = runner.run(pair.triggered, capture_sites=(*capture_sites, monitor_site), retain_response_logits=True)
        if normal.response_logits is None or triggered.response_logits is None:
            raise RuntimeError("baseline response logits missing")
        normal_nll = response_nll(normal)
        triggered_nll = response_nll(triggered)
        normal_probe = probe.score(normal.captures[monitor_site])
        triggered_probe = probe.score(triggered.captures[monitor_site])
        trigger_kl = masked_token_kl(normal.response_logits, triggered.response_logits, pair.normal.response_mask, device=runner.device)
        common = {
            "schema_version": 1,
            "procedure": "site-shuffling-v1-day23-behavioral-transport",
            "implementation_commit": commit,
            "authorization_sha256": authorization_sha256,
            "concept": concept,
            "example_id": example["example_id"],
            "label": int(example["label"]),
            "split": "safety-test-behavioral-outcome",
            "prospective_behavioral_outcome": True,
            "independent_dataset_confirmation": False,
        }
        append_row(completed, {**common, "condition_id": "baseline:normal", "record_type": "baseline", "base_condition": "normal", "probe_score": float(normal_probe[0]), "response_nll": float(normal_nll[0]), "kl_from_normal": 0.0})
        append_row(completed, {**common, "condition_id": "baseline:correct_trigger", "record_type": "baseline", "base_condition": "correct_trigger", "probe_score": float(triggered_probe[0]), "response_nll": float(triggered_nll[0]), "kl_from_normal": float(trigger_kl[0])})
        denominator = float(normal_probe[0] - triggered_probe[0])
        for base_name, condition, base_result, base_nll in (
            ("normal", pair.normal, normal, normal_nll),
            ("correct_trigger", pair.triggered, triggered, triggered_nll),
        ):
            jobs = intervention_jobs(
                base_name,
                selected_heads,
                sites,
                normal.captures,
                triggered.captures,
                authorization["mappings"],
            )
            for chunk in batched(jobs, args.group_chunk_size):
                pending = [item for item in chunk if (example["example_id"], item[0]["condition_id"]) not in completed]
                if not pending:
                    continue
                diagnostics = vector.run_full_diagnostics(condition, [job for _spec, job in pending])
                for job_index, (specification, _job) in enumerate(pending):
                    patched_logits = diagnostics.response_logits[job_index]
                    behavior = directional_logit_metrics(
                        normal.response_logits,
                        triggered.response_logits,
                        base_result.response_logits,
                        patched_logits,
                        condition.response_mask,
                        direction=specification["direction"],
                    )
                    kl = masked_token_kl(
                        base_result.response_logits,
                        patched_logits,
                        condition.response_mask,
                        device=runner.device,
                    )
                    patched_probe = float(diagnostics.probe_scores[job_index, 0])
                    normalized_probe_effect = None
                    if int(example["label"]) == 1 and denominator > 0:
                        normalized_probe_effect = (
                            (float(normal_probe[0]) - patched_probe) / denominator
                            if specification["direction"] == "induction"
                            else (patched_probe - float(triggered_probe[0])) / denominator
                        )
                    append_row(
                        completed,
                        {
                            **common,
                            **specification,
                            "record_type": "intervention",
                            "intervention_kind": "destination_relative_delta",
                            "alpha": 1.0,
                            "probe_score": patched_probe,
                            "probe_shift_from_base": patched_probe - float(normal_probe[0] if base_name == "normal" else triggered_probe[0]),
                            "normalized_probe_effect": normalized_probe_effect,
                            "response_nll": float(diagnostics.response_nll[job_index, 0]),
                            "nll_shift_from_base": float(diagnostics.response_nll[job_index, 0] - base_nll[0]),
                            "kl_from_base": float(kl[0]),
                            "directional_coefficient": float(behavior["directional_coefficient"][0]),
                            "directional_cosine": float(behavior["directional_cosine"][0]),
                            "top1_agreement": float(behavior["top1_agreement"][0]),
                            "natural_centered_logit_energy": float(behavior["natural_centered_logit_energy"][0]),
                            "patch_centered_logit_energy": float(behavior["patch_centered_logit_energy"][0]),
                        },
                    )
                del diagnostics
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        print(f"Day 23 example {example_index}/{len(records)}: {len(completed)}/{expected}", flush=True)
        del normal, triggered
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    seal(completed, expected)
    print(f"Sealed {len(completed)} Day 23 rows: {sha256_file(SEALED_PATH)}", flush=True)


if __name__ == "__main__":
    main()

