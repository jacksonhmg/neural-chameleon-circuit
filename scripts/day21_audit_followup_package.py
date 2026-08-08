#!/usr/bin/env python3
"""Audit the complete local Days 15–21 site-shuffling package."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "results/day-21/follow-up-package-audit.json"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args), cwd=ROOT, text=True, capture_output=True, check=check
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gzip_rows(path: Path) -> int:
    with gzip.open(path, "rt") as handle:
        return sum(1 for _line in handle)


def added_commit(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    return run(
        "git", "log", "--diff-filter=A", "--format=%H", "-1", "--", relative
    ).stdout.strip()


def main() -> None:
    checks: dict[str, Any] = {}
    required_files = [
        ROOT / "results/day-15/frozen-site-shuffling-plan.json",
        ROOT / "results/day-15/frozen-mapping-ensemble.json",
        ROOT / "results/day-16/artifact-triage-audit.json",
        ROOT / "results/day-17/transfer-atlas-audit.json",
        ROOT / "results/day-18/geometry-audit.json",
        ROOT / "results/day-19/day19-audit.json",
        ROOT / "results/day-20/day20-audit.json",
        ROOT / "results/day-21/confirmation-authorization.json",
        ROOT / "results/day-21/confirmation-preflight.json",
        ROOT / "results/day-21/confirmation-gate.json",
        ROOT / "results/day-21/day21-audit.json",
        ROOT / "report/site-shuffling-follow-up.md",
        ROOT / "decision-log/0020-freeze-site-shuffling-follow-up.md",
        ROOT / "decision-log/0021-authorize-prospective-site-shuffling-confirmation.md",
        ROOT / "decision-log/0022-accept-qualified-portable-site-transfer.md",
    ]
    checks["required_files_present"] = all(path.is_file() for path in required_files)

    audit_paths = [
        ROOT / "results/day-16/artifact-triage-audit.json",
        ROOT / "results/day-17/transfer-atlas-audit.json",
        ROOT / "results/day-18/geometry-audit.json",
        ROOT / "results/day-19/day19-audit.json",
        ROOT / "results/day-20/day20-audit.json",
        ROOT / "results/day-21/day21-audit.json",
    ]
    checks["all_daily_audits_pass"] = all(
        json.loads(path.read_text()).get("status") == "pass" for path in audit_paths
    )
    preflight_paths = [
        ROOT / "results/day-16/artifact-triage-preflight.json",
        ROOT / "results/day-17/transfer-atlas-preflight.json",
        ROOT / "results/day-18/geometry-preflight.json",
        ROOT / "results/day-19/permutation-ensemble-preflight.json",
        ROOT / "results/day-20/specificity-preflight.json",
        ROOT / "results/day-21/confirmation-preflight.json",
    ]
    checks["all_preflights_pass"] = all(
        json.loads(path.read_text()).get("status") == "pass"
        for path in preflight_paths
    )

    expected_rows = {
        "results/day-16/artifact-triage-results.jsonl.gz": 1344,
        "results/day-17/transfer-atlas-results.jsonl.gz": 92224,
        "results/day-18/geometry-transfer-results.jsonl.gz": 55424,
        "results/day-19/permutation-ensemble-results.jsonl.gz": 22880,
        "results/day-19/composition-results.jsonl.gz": 7392,
        "results/day-20/specificity-results.jsonl.gz": 6400,
        "results/day-20/behavior-results.jsonl.gz": 224,
        "results/day-21/confirmation-results.jsonl.gz": 6500,
    }
    observed_rows = {
        relative: gzip_rows(ROOT / relative) for relative in expected_rows
    }
    checks["all_raw_row_counts"] = observed_rows == expected_rows

    working_files = list((ROOT / "results").glob("day-1[5-9]/*.working.jsonl"))
    working_files += list((ROOT / "results/day-20").glob("*.working.jsonl"))
    working_files += list((ROOT / "results/day-21").glob("*.working.jsonl"))
    checks["no_resumable_working_files"] = not working_files

    authorization_path = ROOT / "results/day-21/confirmation-authorization.json"
    raw_path = ROOT / "results/day-21/confirmation-results.jsonl.gz"
    gate = json.loads((ROOT / "results/day-21/confirmation-gate.json").read_text())
    checks["day21_gate_portable_support"] = gate.get("status") == "portable_support"
    checks["day21_gate_hashes"] = (
        gate.get("authorization_sha256") == sha256_file(authorization_path)
        and gate.get("raw_results_sha256") == sha256_file(raw_path)
    )

    authorization_commit = added_commit(authorization_path)
    raw_commit = added_commit(raw_path)
    ancestry = run(
        "git", "merge-base", "--is-ancestor", authorization_commit, raw_commit, check=False
    )
    checks["authorization_committed_before_raw_results"] = (
        bool(authorization_commit)
        and bool(raw_commit)
        and authorization_commit != raw_commit
        and ancestry.returncode == 0
    )

    followup_text = (ROOT / "report/site-shuffling-follow-up.md").read_text()
    final_text = (ROOT / "report/final-report.md").read_text()
    ledger_text = (ROOT / "report/claim-ledger.md").read_text()
    limitation_text = (ROOT / "report/limitations.md").read_text()
    sprint_text = (ROOT / "SPRINT.md").read_text()
    checks["documentation_records_disposition"] = all(
        token in followup_text
        for token in (
            "portable_support",
            "cross_15122",
            "route-sensitive",
            "Not released or sent externally",
        )
    ) and all(
        token in final_text + ledger_text + limitation_text
        for token in ("site-shuffling follow-up", "portable")
    )
    checks["sprint_day21_complete"] = (
        "## Day 21: Run prospective intervention confirmation" in sprint_text
        and "**Status:** Complete" in sprint_text.split(
            "## Day 21: Run prospective intervention confirmation", 1
        )[1]
    )

    diff_check = run("git", "diff", "--check", check=False)
    checks["git_diff_check"] = diff_check.returncode == 0
    test_environment = dict(os.environ)
    existing_pythonpath = test_environment.get("PYTHONPATH")
    test_environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        env=test_environment,
        text=True,
        capture_output=True,
    )
    checks["full_test_suite"] = tests.returncode == 0

    origin_head = run("git", "rev-parse", "origin/main").stdout.strip()
    ahead = int(run("git", "rev-list", "--count", "origin/main..HEAD").stdout)
    checks["local_commits_not_pushed"] = ahead > 0
    checks["no_release_tag_at_head"] = run(
        "git", "tag", "--points-at", "HEAD"
    ).stdout.strip() == ""

    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-follow-up-package-audit",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "raw_row_counts": observed_rows,
        "authorization_commit": authorization_commit,
        "raw_results_commit": raw_commit,
        "origin_main": origin_head,
        "commits_ahead_of_origin_main": ahead,
        "test_command": f"{sys.executable} -m pytest -q tests",
        "test_returncode": tests.returncode,
        "test_summary": (tests.stdout + tests.stderr).strip().splitlines()[-1]
        if (tests.stdout + tests.stderr).strip()
        else "",
        "working_files": [path.relative_to(ROOT).as_posix() for path in working_files],
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"follow-up package audit failed: {failed}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
