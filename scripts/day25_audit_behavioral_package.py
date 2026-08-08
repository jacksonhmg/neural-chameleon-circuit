#!/usr/bin/env python3
"""Audit the complete local Days 22–25 behavioral transport package."""

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
OUTPUT_PATH = ROOT / "results/day-25/behavioral-package-audit.json"


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=ROOT, text=True, capture_output=True, check=check)


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
    return run("git", "log", "--diff-filter=A", "--format=%H", "-1", "--", path.relative_to(ROOT).as_posix()).stdout.strip()


def is_ancestor(first: str, second: str) -> bool:
    return bool(first and second) and run("git", "merge-base", "--is-ancestor", first, second, check=False).returncode == 0


def main() -> None:
    checks: dict[str, Any] = {}
    required = [
        ROOT / "results/day-22/behavioral-transport-authorization.json",
        ROOT / "results/day-22/freeze-audit.json",
        ROOT / "results/day-23/behavioral-transport-preflight.json",
        ROOT / "results/day-23/behavioral-transport-results.jsonl.gz",
        ROOT / "results/day-23/behavioral-transport-gate.json",
        ROOT / "results/day-23/day23-audit.json",
        ROOT / "results/day-24/coupled-generation-preflight.json",
        ROOT / "results/day-24/coupled-generation-results.jsonl.gz",
        ROOT / "results/day-24/coupled-generation-summary.json",
        ROOT / "results/day-24/day24-audit.json",
        ROOT / "results/day-24/qualitative-review-assessment.md",
        ROOT / "decision-log/0023-freeze-site-shuffling-behavioral-extension.md",
        ROOT / "decision-log/0024-accept-behaviorally-selective-site-transfer.md",
        ROOT / "report/site-shuffling-follow-up.md",
        ROOT / "report/claim-ledger.md",
        ROOT / "report/limitations.md",
        ROOT / "report/final-report.md",
    ]
    checks["required_files_present"] = all(path.is_file() for path in required)
    freeze = json.loads((ROOT / "results/day-22/freeze-audit.json").read_text())
    preflights = [json.loads((ROOT / "results/day-23/behavioral-transport-preflight.json").read_text()), json.loads((ROOT / "results/day-24/coupled-generation-preflight.json").read_text())]
    audits = [json.loads((ROOT / "results/day-23/day23-audit.json").read_text()), json.loads((ROOT / "results/day-24/day24-audit.json").read_text())]
    checks["freeze_passed_without_outcomes"] = freeze.get("status") == "pass" and freeze.get("behavioral_outcomes_generated") is False
    checks["preflights_passed_without_outcomes"] = all(row.get("status") == "pass" for row in preflights) and preflights[0].get("behavioral_outcomes_generated_during_preflight") is False and preflights[1].get("generation_outcomes_generated_during_preflight") is False
    checks["daily_audits_pass"] = all(row.get("status") == "pass" for row in audits)

    expected_rows = {"results/day-23/behavioral-transport-results.jsonl.gz": 3880, "results/day-24/coupled-generation-results.jsonl.gz": 80}
    observed_rows = {relative: gzip_rows(ROOT / relative) for relative in expected_rows}
    checks["raw_row_counts"] = observed_rows == expected_rows
    authorization_path = ROOT / "results/day-22/behavioral-transport-authorization.json"
    day23_raw = ROOT / "results/day-23/behavioral-transport-results.jsonl.gz"
    day24_raw = ROOT / "results/day-24/coupled-generation-results.jsonl.gz"
    authorization = json.loads(authorization_path.read_text())
    gate = json.loads((ROOT / "results/day-23/behavioral-transport-gate.json").read_text())
    generation = json.loads((ROOT / "results/day-24/coupled-generation-summary.json").read_text())
    checks["day23_hashes"] = gate.get("authorization_sha256") == sha256_file(authorization_path) and gate.get("raw_results_sha256") == sha256_file(day23_raw) and audits[0].get("raw_results_sha256") == sha256_file(day23_raw)
    checks["day24_hashes"] = generation.get("raw_results_sha256") == sha256_file(day24_raw) and audits[1].get("raw_results_sha256") == sha256_file(day24_raw)
    checks["frozen_cell_dispositions"] = [(row["concept"], row["direction"], row["disposition"]) for row in gate["cells"]] == [
        ("deception", "induction", "behavior_preserving_portable_evasion"),
        ("deception", "rescue", "behavior_preserving_portable_evasion"),
        ("harmful", "induction", "behavior_preserving_portable_evasion"),
        ("harmful", "rescue", "mixed"),
    ]
    checks["no_external_evaluator"] = authorization["dissemination"]["external_evaluator"] is False and generation["external_evaluator_used"] is False

    authorization_commit = added_commit(authorization_path)
    day23_commit = added_commit(day23_raw)
    day24_commit = added_commit(day24_raw)
    implementation_commit = authorization["implementation_commit"]
    checks["implementation_before_authorization"] = implementation_commit != authorization_commit and is_ancestor(implementation_commit, authorization_commit)
    checks["authorization_before_both_raw_archives"] = authorization_commit not in {day23_commit, day24_commit} and is_ancestor(authorization_commit, day23_commit) and is_ancestor(authorization_commit, day24_commit)
    working_files = list((ROOT / "results/day-22").glob("*.working.jsonl")) + list((ROOT / "results/day-23").glob("*.working.jsonl")) + list((ROOT / "results/day-24").glob("*.working.jsonl")) + list((ROOT / "results/day-25").glob("*.working.jsonl"))
    checks["no_resumable_working_files"] = not working_files

    followup = (ROOT / "report/site-shuffling-follow-up.md").read_text()
    ledger = (ROOT / "report/claim-ledger.md").read_text()
    limitations = (ROOT / "report/limitations.md").read_text()
    final_report = (ROOT / "report/final-report.md").read_text()
    sprint = (ROOT / "SPRINT.md").read_text()
    checks["documentation_records_boundary"] = all(token in followup + ledger + limitations + final_report for token in ("behaviorally selective", "output-inert", "0.10", "four"))
    checks["documentation_keeps_release_local"] = "Not released or sent externally" in followup and "Not released or sent externally" in final_report
    day25_section = sprint.split("## Day 25: Seal the behavioral disposition", 1)[1]
    checks["sprint_day25_complete"] = "**Status:** Complete" in day25_section and "- [ ]" not in day25_section

    checks["git_diff_check"] = run("git", "diff", "--check", check=False).returncode == 0
    test_environment = dict(os.environ)
    test_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), str(ROOT / "scripts"), test_environment.get("PYTHONPATH", ""))))
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, env=test_environment, text=True, capture_output=True)
    checks["full_test_suite"] = tests.returncode == 0
    origin_head = run("git", "rev-parse", "origin/main").stdout.strip()
    ahead = int(run("git", "rev-list", "--count", "origin/main..HEAD").stdout)
    checks["local_commits_not_pushed"] = ahead > 0
    checks["no_release_tag_at_head"] = run("git", "tag", "--points-at", "HEAD").stdout.strip() == ""
    report = {
        "schema_version": 1,
        "procedure": "site-shuffling-v1-days22-25-package-audit",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "raw_row_counts": observed_rows,
        "authorization_commit": authorization_commit,
        "implementation_commit": implementation_commit,
        "day23_raw_commit": day23_commit,
        "day24_raw_commit": day24_commit,
        "authorization_sha256": sha256_file(authorization_path),
        "day23_raw_sha256": sha256_file(day23_raw),
        "day24_raw_sha256": sha256_file(day24_raw),
        "origin_main": origin_head,
        "commits_ahead_of_origin_main": ahead,
        "test_command": f"{sys.executable} -m pytest -q tests",
        "test_returncode": tests.returncode,
        "test_summary": (tests.stdout + tests.stderr).strip().splitlines()[-1] if (tests.stdout + tests.stderr).strip() else "",
        "working_files": [path.relative_to(ROOT).as_posix() for path in working_files],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "pass":
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"behavioral package audit failed: {failed}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

