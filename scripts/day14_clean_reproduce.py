#!/usr/bin/env python3
"""Reproduce the Day 13 central safety result in a detached clean worktree."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-14/clean-reproduction"
PLAN_PATH = ROOT / "results/day-14/frozen-falsification-plan.json"
CANONICAL_DIR = ROOT / "results/day-13"
CANONICAL_RAW = CANONICAL_DIR / "safety-transfer-results.jsonl.gz"
CANONICAL_SUMMARY = CANONICAL_DIR / "safety-transfer-summary.json"
CANONICAL_METRICS = CANONICAL_DIR / "safety-transfer-metrics.csv"
MODEL_TARGET = ROOT / "artifacts/models/neural-chameleon-gemma_2_9b-layer_12"
EXTERNAL_TARGET = ROOT / "external/minimal_neural_chameleons"
SOURCE_PYTHON = EXTERNAL_TARGET / ".venv/bin/python"
EXPECTED_ROWS = 1_944


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
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


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def target_fingerprint(path: Path) -> str:
    """Fingerprint target names, sizes, and mtimes without rereading model weights."""
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        relative = item.relative_to(path).as_posix()
        digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()


def run_logged(
    command: list[str], cwd: Path, log: list[str], environment: Mapping[str, str]
) -> None:
    rendered = " ".join(command)
    header = f"$ (cd {cwd} && {rendered})"
    print(header, flush=True)
    log.append(header)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log.append(line.rstrip("\n"))
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    condition = row.get("condition_id") or f"{row['group_id']}:{row['direction']}"
    return str(row["example_id"]), str(condition)


def numeric_estimates(value: Any, prefix: str = "") -> dict[str, float]:
    estimates: dict[str, float] = {}
    if isinstance(value, dict):
        if "estimate" in value and isinstance(value["estimate"], (int, float)):
            estimates[prefix] = float(value["estimate"])
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            estimates.update(numeric_estimates(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            child = f"{prefix}[{index}]"
            estimates.update(numeric_estimates(nested, child))
    return estimates


def interval_decisions(value: Any, prefix: str = "") -> dict[str, bool]:
    decisions: dict[str, bool] = {}
    if isinstance(value, dict):
        if "ci_low" in value and isinstance(value["ci_low"], (int, float)):
            decisions[prefix] = float(value["ci_low"]) > 0
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            decisions.update(interval_decisions(nested, child))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            decisions.update(interval_decisions(nested, f"{prefix}[{index}]"))
    return decisions


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def compare_outputs(
    reproduced_raw: Path,
    reproduced_summary: Path,
    reproduced_metrics: Path,
) -> dict[str, Any]:
    canonical_rows = load_jsonl(CANONICAL_RAW)
    reproduced_rows = load_jsonl(reproduced_raw)
    canonical_keys = {row_key(row) for row in canonical_rows}
    reproduced_keys = {row_key(row) for row in reproduced_rows}
    if len(canonical_keys) != len(canonical_rows):
        raise ValueError("canonical Day 13 raw archive has duplicate keys")
    if len(reproduced_keys) != len(reproduced_rows):
        raise ValueError("reproduced Day 13 raw archive has duplicate keys")

    canonical_summary = json.loads(CANONICAL_SUMMARY.read_text())
    new_summary = json.loads(reproduced_summary.read_text())
    canonical_estimates = numeric_estimates(canonical_summary)
    new_estimates = numeric_estimates(new_summary)
    if set(canonical_estimates) != set(new_estimates):
        raise ValueError("summary point-estimate paths differ")
    differences = {
        key: abs(canonical_estimates[key] - new_estimates[key])
        for key in canonical_estimates
    }
    maximum_path = max(differences, key=differences.get)
    canonical_decisions = interval_decisions(canonical_summary)
    new_decisions = interval_decisions(new_summary)
    interval_paths_equal = set(canonical_decisions) == set(new_decisions)
    interval_decisions_equal = (
        interval_paths_equal and canonical_decisions == new_decisions
    )
    support_equal = (
        canonical_summary["overall_safety_transfer_supported"]
        == new_summary["overall_safety_transfer_supported"]
        and canonical_summary["safety_transfer_supported_by_concept"]
        == new_summary["safety_transfer_supported_by_concept"]
    )
    metrics_equal = read_metrics(CANONICAL_METRICS) == read_metrics(reproduced_metrics)
    raw_hash_equal = sha256_file(CANONICAL_RAW) == sha256_file(reproduced_raw)
    summary_hash_equal = sha256_file(CANONICAL_SUMMARY) == sha256_file(
        reproduced_summary
    )
    gates = {
        "exact_1944_row_key_grid": bool(
            len(canonical_rows) == len(reproduced_rows) == EXPECTED_ROWS
            and canonical_keys == reproduced_keys
        ),
        "identical_support_classification": bool(support_equal),
        "maximum_primary_point_estimate_difference_at_most_0_002": bool(
            differences[maximum_path] <= 0.002
        ),
        "all_interval_lower_bound_decisions_unchanged": bool(
            interval_decisions_equal
        ),
    }
    return {
        "status": "pass" if all(gates.values()) else "fail",
        "gates": gates,
        "canonical_row_count": len(canonical_rows),
        "reproduced_row_count": len(reproduced_rows),
        "canonical_only_key_count": len(canonical_keys - reproduced_keys),
        "reproduced_only_key_count": len(reproduced_keys - canonical_keys),
        "point_estimate_count": len(differences),
        "maximum_absolute_point_estimate_difference": differences[maximum_path],
        "maximum_difference_path": maximum_path,
        "interval_decision_count": len(canonical_decisions),
        "byte_identical_raw_archive": raw_hash_equal,
        "byte_identical_summary": summary_hash_equal,
        "exact_metrics_csv": metrics_equal,
        "canonical_support": {
            "overall": canonical_summary["overall_safety_transfer_supported"],
            "by_concept": canonical_summary["safety_transfer_supported_by_concept"],
        },
        "reproduced_support": {
            "overall": new_summary["overall_safety_transfer_supported"],
            "by_concept": new_summary["safety_transfer_supported_by_concept"],
        },
    }


def copy_outputs(worktree: Path, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "reproduced-safety-transfer-results.jsonl.gz": worktree
        / "results/day-13/safety-transfer-results.jsonl.gz",
        "reproduced-safety-transfer-summary.json": worktree
        / "results/day-13/safety-transfer-summary.json",
        "reproduced-safety-transfer-metrics.csv": worktree
        / "results/day-13/safety-transfer-metrics.csv",
        "reproduced-safety-transfer.png": worktree
        / "results/day-13/safety-transfer.png",
        "reproduced-safety-transfer.pdf": worktree
        / "results/day-13/safety-transfer.pdf",
    }
    hashes = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / name
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
    return hashes


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    plan = json.loads(PLAN_PATH.read_text())
    reproduction = plan["clean_reproduction"]
    commit = reproduction["commit"]
    implementation_commit = git_head()
    require_committed_file(Path(__file__).resolve(), implementation_commit)
    require_committed_file(PLAN_PATH, implementation_commit)
    if not CANONICAL_RAW.is_file() or not SOURCE_PYTHON.is_file():
        raise FileNotFoundError("canonical archive or pinned Python is unavailable")

    canonical_hash_before = sha256_file(CANONICAL_RAW)
    model_fingerprint_before = target_fingerprint(MODEL_TARGET)
    external_fingerprint_before = target_fingerprint(EXTERNAL_TARGET)
    temporary_parent = Path(tempfile.mkdtemp(prefix="neural-chameleon-day14-"))
    worktree = temporary_parent / "worktree"
    worktree_added = False
    log: list[str] = []
    hashes: dict[str, str] = {}
    comparison: dict[str, Any] = {}
    worktree_status_before = ""
    worktree_status_after = ""
    try:
        run_logged(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            ROOT,
            log,
            os.environ,
        )
        worktree_added = True
        resolved_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=worktree, text=True
        ).strip()
        if resolved_commit != commit:
            raise RuntimeError("detached worktree resolved the wrong commit")
        worktree_status_before = subprocess.check_output(
            ["git", "status", "--short"], cwd=worktree, text=True
        ).strip()
        (worktree / "artifacts/models").mkdir(parents=True)
        (worktree / "external").mkdir(parents=True)
        (worktree / "artifacts/models" / MODEL_TARGET.name).symlink_to(
            MODEL_TARGET, target_is_directory=True
        )
        (worktree / "external" / EXTERNAL_TARGET.name).symlink_to(
            EXTERNAL_TARGET, target_is_directory=True
        )
        environment = dict(os.environ)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        run_logged(
            [
                str(SOURCE_PYTHON),
                "scripts/day13_run_safety_test.py",
                "--skip-preflight",
                "--batch-size",
                "2",
            ],
            worktree,
            log,
            environment,
        )
        run_logged(
            [str(SOURCE_PYTHON), "scripts/day13_analyze_safety_test.py"],
            worktree,
            log,
            environment,
        )
        worktree_status_after = subprocess.check_output(
            ["git", "status", "--short"], cwd=worktree, text=True
        ).strip()
        hashes = copy_outputs(worktree, output_dir)
        comparison = compare_outputs(
            output_dir / "reproduced-safety-transfer-results.jsonl.gz",
            output_dir / "reproduced-safety-transfer-summary.json",
            output_dir / "reproduced-safety-transfer-metrics.csv",
        )
    finally:
        if worktree_added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=ROOT,
                check=True,
            )
        shutil.rmtree(temporary_parent)

    canonical_hash_after = sha256_file(CANONICAL_RAW)
    model_fingerprint_after = target_fingerprint(MODEL_TARGET)
    external_fingerprint_after = target_fingerprint(EXTERNAL_TARGET)
    canonical_unchanged = canonical_hash_before == canonical_hash_after
    linked_targets_unchanged = (
        model_fingerprint_before == model_fingerprint_after
        and external_fingerprint_before == external_fingerprint_after
    )
    status = "pass" if (
        comparison.get("status") == "pass"
        and canonical_unchanged
        and linked_targets_unchanged
    ) else "fail"
    manifest = {
        "schema_version": 1,
        "procedure": "day14-clean-reproduction-v1",
        "status": status,
        "implementation_commit": implementation_commit,
        "reproduced_commit": commit,
        "detached_worktree": True,
        "fresh_python_processes": True,
        "commands": [
            "scripts/day13_run_safety_test.py --skip-preflight --batch-size 2",
            "scripts/day13_analyze_safety_test.py",
        ],
        "canonical_archive_sha256_before": canonical_hash_before,
        "canonical_archive_sha256_after": canonical_hash_after,
        "canonical_archive_overwritten": not canonical_unchanged,
        "read_only_link_targets_unchanged": linked_targets_unchanged,
        "model_target_fingerprint_before": model_fingerprint_before,
        "model_target_fingerprint_after": model_fingerprint_after,
        "external_target_fingerprint_before": external_fingerprint_before,
        "external_target_fingerprint_after": external_fingerprint_after,
        "worktree_status_before_commands": worktree_status_before,
        "worktree_status_after_commands": worktree_status_after,
        "reproduced_artifact_sha256": hashes,
        "comparison": comparison,
        "temporary_worktree_removed": not worktree.exists(),
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "published_or_sent": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "execution.log").write_text("\n".join(log) + "\n")
    (output_dir / "reproduction-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if status != "pass":
        raise RuntimeError("Day 14 clean reproduction failed its frozen gate")
    print(
        "Day 14 clean reproduction passed; canonical archive unchanged and "
        f"maximum point-estimate difference={comparison['maximum_absolute_point_estimate_difference']:.6g}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
