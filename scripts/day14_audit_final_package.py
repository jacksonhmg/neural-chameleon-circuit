#!/usr/bin/env python3
"""Machine-check the complete local Day 14 research and report package."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results/day-14"
PLAN_PATH = RESULT_DIR / "frozen-falsification-plan.json"
OUTPUT_PATH = RESULT_DIR / "day14-audit.json"
FIGURE_MANIFEST = ROOT / "report/figures/figure-manifest.json"
FINAL_REPORT = ROOT / "report/final-report.md"
CLAIM_LEDGER = ROOT / "report/claim-ledger.md"
LIMITATIONS = ROOT / "report/limitations.md"
FALSIFICATION_LOG = RESULT_DIR / "falsification-log.md"
README = RESULT_DIR / "README.md"
SPRINT = ROOT / "SPRINT.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_committed_file(path: Path, commit: str) -> None:
    relative = path.resolve().relative_to(ROOT).as_posix()
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if committed != path.read_bytes():
        raise RuntimeError(f"{relative} differs from implementation commit {commit}")


def json_rows(path: Path) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return sum(1 for line in handle if line.strip())


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, evidence: Any) -> None:
    checks.append({"id": check_id, "pass": bool(passed), "evidence": evidence})


def main() -> None:
    head = git("rev-parse", "HEAD")
    required_committed = (
        Path(__file__).resolve(),
        PLAN_PATH,
        FINAL_REPORT,
        CLAIM_LEDGER,
        LIMITATIONS,
        FALSIFICATION_LOG,
        README,
        FIGURE_MANIFEST,
        SPRINT,
    )
    for path in required_committed:
        require_committed_file(path, head)
    preaudit_status = git("status", "--porcelain=v1", "--untracked-files=all")
    if preaudit_status:
        raise RuntimeError(
            "final package audit requires a clean worktree before writing its output"
        )

    plan = json.loads(PLAN_PATH.read_text())
    final_plan = plan["final_report"]
    analysis = json.loads((RESULT_DIR / "analysis-only-summary.json").read_text())
    machinery = json.loads((RESULT_DIR / "machinery-audit.json").read_text())
    causal = json.loads((RESULT_DIR / "causal-falsification-summary.json").read_text())
    causal_audit = json.loads((RESULT_DIR / "causal-falsification-audit.json").read_text())
    reproduction = json.loads(
        (RESULT_DIR / "clean-reproduction/reproduction-manifest.json").read_text()
    )
    falsification = json.loads((RESULT_DIR / "falsification-summary.json").read_text())
    figure_manifest = json.loads(FIGURE_MANIFEST.read_text())
    report_text = FINAL_REPORT.read_text()
    ledger_text = CLAIM_LEDGER.read_text()
    limitations_text = LIMITATIONS.read_text()
    log_text = FALSIFICATION_LOG.read_text()
    sprint_text = SPRINT.read_text()

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "analysis_only_gate",
        analysis.get("analysis_only_falsification_gate_passed") is True,
        analysis.get("analysis_only_falsification_gate_passed"),
    )
    add_check(
        checks,
        "machinery_audit",
        machinery.get("status") == "pass"
        and machinery.get("registered_hook_count_after_audit") == 0,
        {
            "status": machinery.get("status"),
            "hooks": machinery.get("registered_hook_count_after_audit"),
        },
    )
    add_check(
        checks,
        "causal_audit",
        causal_audit.get("status") == "pass"
        and causal_audit.get("row_count") == 1_088
        and causal.get("frozen_null_gate_pass") is True,
        {
            "status": causal_audit.get("status"),
            "rows": causal_audit.get("row_count"),
            "null_gate": causal.get("frozen_null_gate_pass"),
        },
    )
    add_check(
        checks,
        "causal_raw_archive",
        json_rows(RESULT_DIR / "causal-falsification-results.jsonl.gz") == 1_088,
        json_rows(RESULT_DIR / "causal-falsification-results.jsonl.gz"),
    )
    comparison = reproduction["comparison"]
    add_check(
        checks,
        "clean_reproduction",
        reproduction.get("status") == "pass"
        and all(comparison["gates"].values())
        and comparison["maximum_absolute_point_estimate_difference"] <= 0.002
        and reproduction.get("canonical_archive_overwritten") is False
        and reproduction.get("temporary_worktree_removed") is True,
        {
            "status": reproduction.get("status"),
            "gates": comparison["gates"],
            "maximum_difference": comparison[
                "maximum_absolute_point_estimate_difference"
            ],
            "canonical_archive_overwritten": reproduction.get(
                "canonical_archive_overwritten"
            ),
            "temporary_worktree_removed": reproduction.get(
                "temporary_worktree_removed"
            ),
        },
    )
    add_check(
        checks,
        "clean_reproduction_raw_archive",
        json_rows(
            RESULT_DIR
            / "clean-reproduction/reproduced-safety-transfer-results.jsonl.gz"
        )
        == 1_944,
        json_rows(
            RESULT_DIR
            / "clean-reproduction/reproduced-safety-transfer-results.jsonl.gz"
        ),
    )
    canonical_raw = ROOT / "results/day-13/safety-transfer-results.jsonl.gz"
    add_check(
        checks,
        "canonical_day13_archive_unchanged",
        sha256_file(canonical_raw)
        == reproduction["canonical_archive_sha256_before"]
        == reproduction["canonical_archive_sha256_after"],
        sha256_file(canonical_raw),
    )
    add_check(
        checks,
        "aggregate_disposition",
        falsification.get("status") == "qualified-survival"
        and falsification.get("published_or_sent") is False,
        {
            "status": falsification.get("status"),
            "published_or_sent": falsification.get("published_or_sent"),
        },
    )

    figures = figure_manifest.get("figures", [])
    add_check(checks, "six_figures_declared", len(figures) == 6, len(figures))
    figure_hashes_valid = True
    source_tables_valid = True
    pdf_pages = {}
    for figure in figures:
        png = ROOT / figure["png"]
        pdf = ROOT / figure["pdf"]
        table = ROOT / figure["source_table"]
        figure_hashes_valid = figure_hashes_valid and (
            png.is_file()
            and pdf.is_file()
            and sha256_file(png) == figure["png_sha256"]
            and sha256_file(pdf) == figure["pdf_sha256"]
        )
        source_tables_valid = source_tables_valid and (
            table.is_file()
            and sha256_file(table) == figure["source_table_sha256"]
            and sum(1 for line in table.open() if line.strip()) >= 2
        )
        pages_output = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        pages_line = next(line for line in pages_output.splitlines() if line.startswith("Pages:"))
        pdf_pages[figure["number"]] = int(pages_line.split(":", 1)[1].strip())
    add_check(checks, "figure_hashes", figure_hashes_valid, figure_hashes_valid)
    add_check(checks, "source_table_hashes", source_tables_valid, source_tables_valid)
    add_check(
        checks,
        "single_page_pdfs",
        set(pdf_pages.values()) == {1} and len(pdf_pages) == 6,
        pdf_pages,
    )

    missing_epistemic = [
        label for label in final_plan["required_epistemic_labels"]
        if label.lower() not in report_text.lower()
    ]
    missing_results = [
        label for label in final_plan["required_result_labels"]
        if label.lower() not in report_text.lower()
    ]
    add_check(checks, "epistemic_labels", not missing_epistemic, missing_epistemic)
    add_check(checks, "result_labels", not missing_results, missing_results)
    add_check(
        checks,
        "six_report_figure_references",
        all(f"figure-{number:02d}-" in report_text for number in range(1, 7)),
        [f"figure-{number:02d}-" in report_text for number in range(1, 7)],
    )
    add_check(
        checks,
        "claim_ledger_dispositions",
        "Explicitly rejected or weakened hypotheses" in ledger_text
        and "Supported with substantial qualification" in ledger_text
        and "Rejected" in ledger_text,
        "supported, qualified, and rejected dispositions present",
    )
    add_check(
        checks,
        "limitations_coverage",
        all(
            heading in limitations_text
            for heading in (
                "Model and probe scope",
                "Selection and statistical scope",
                "Causal interpretation",
                "Data and leakage",
                "Behavioral evidence",
                "Reproducibility and external review",
            )
        ),
        "six limitation domains required",
    )
    add_check(
        checks,
        "falsification_log_outcome_classes",
        all(
            phrase in log_text
            for phrase in (
                "Null result",
                "Failed mixed-circuit hypothesis",
                "Failed rigid-routing hypothesis",
                "Implementation failure",
                "Successful clean reproduction",
            )
        ),
        "positive, null, failed, implementation, and reproduction outcomes present",
    )

    release_lines = (
        "Release pinned dependencies.",
        "Release exact model and probe identifiers, revisions, and hashes.",
        "Release locked data splits.",
        "Release intervention code.",
        "Release raw results.",
        "Release figure-generation scripts.",
        "Release a falsification log.",
        "Release a limitations section.",
        "Send the draft to the original authors",
    )
    release_unchecked = all(
        any(line.startswith("- [ ]") and item in line for line in sprint_text.splitlines())
        for item in release_lines
    )
    add_check(
        checks,
        "release_and_contact_unchecked",
        release_unchecked,
        release_unchecked,
    )
    local_origin = git("rev-parse", "origin/main")
    ahead_count = int(git("rev-list", "--count", "origin/main..HEAD"))
    head_tags = git("tag", "--points-at", "HEAD")
    add_check(
        checks,
        "origin_tracking_ref_unchanged",
        local_origin == plan["parent_commit"],
        {"origin_main": local_origin, "frozen_parent": plan["parent_commit"]},
    )
    add_check(
        checks,
        "local_commits_not_pushed",
        ahead_count > 0,
        {"ahead_of_origin_main": ahead_count},
    )
    add_check(
        checks,
        "no_head_tag",
        head_tags == "",
        {"tags_pointing_at_head": head_tags.splitlines() if head_tags else []},
    )

    diff_check = subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    add_check(
        checks,
        "git_diff_check",
        diff_check.returncode == 0,
        (diff_check.stdout + diff_check.stderr).strip(),
    )
    add_check(
        checks,
        "test_suite",
        tests.returncode == 0,
        (tests.stdout + tests.stderr).strip(),
    )

    tracked = [
        line for line in git("ls-files", "results/day-14", "report", "scripts/day14*", "lab-notes/day-14-falsify-write-and-release.md").splitlines()
        if line and line != "results/day-14/day14-audit.json"
    ]
    artifacts = {
        relative: sha256_file(ROOT / relative)
        for relative in sorted(tracked)
        if (ROOT / relative).is_file()
    }
    status = "pass" if all(row["pass"] for row in checks) else "fail"
    report = {
        "schema_version": 1,
        "procedure": "day14-final-package-audit-v1",
        "status": status,
        "implementation_commit": head,
        "branch": git("branch", "--show-current"),
        "origin_main": local_origin,
        "ahead_of_origin_main": ahead_count,
        "preaudit_worktree_clean": preaudit_status == "",
        "check_count": len(checks),
        "passed_check_count": sum(row["pass"] for row in checks),
        "checks": checks,
        "tracked_package_file_count_excluding_this_audit": len(artifacts),
        "tracked_package_sha256_excluding_this_audit": artifacts,
        "falsification_plan_sha256": sha256_file(PLAN_PATH),
        "published_pushed_tagged_or_sent": False,
    }
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if status != "pass":
        failed = [row["id"] for row in checks if not row["pass"]]
        raise RuntimeError(f"Day 14 final package audit failed: {failed}")
    print(
        f"Day 14 final package audit passed: {len(checks)} checks, "
        f"{len(artifacts)} tracked package files hashed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
