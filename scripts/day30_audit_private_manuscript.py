#!/usr/bin/env python3
"""Audit the deterministic private manuscript candidate and its provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "output/pdf/neural-chameleon-causal-mechanisms-private.pdf"
OUTPUT = ROOT / "paper/audits/day30-private-candidate-audit.json"


def command(*parts: str, env: dict[str, str] | None = None) -> str:
    return subprocess.check_output(parts, cwd=ROOT, text=True, stderr=subprocess.STDOUT, env=env)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def add(checks: dict[str, Any], name: str, passed: bool, detail: Any) -> None:
    checks[name] = {"pass": bool(passed), "detail": detail}


def main() -> None:
    parsed = parse_args()
    checks: dict[str, Any] = {}

    source_commit = command("git", "rev-parse", parsed.source_commit).strip()
    head = command("git", "rev-parse", "HEAD").strip()
    add(checks, "source_commit_is_head", source_commit == head, {"source_commit": source_commit, "head": head})
    add(checks, "pdf_exists", PDF.is_file(), str(PDF.relative_to(ROOT)))

    pdf_info = command("pdfinfo", str(PDF))
    page_match = re.search(r"^Pages:\s+(\d+)$", pdf_info, re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else None
    add(checks, "pdf_page_count", pages == 14, pages)
    add(checks, "pdf_letter_pages", "Page size:       612 x 792 pts (letter)" in pdf_info, "letter")

    pdf_text = command("pdftotext", str(PDF), "-")
    required_text = [
        "Causal Mechanisms of Activation-Monitor Evasion",
        "Private manuscript candidate",
        "The unchanged population transfers to locked safety concepts",
        "Trigger-linked state can use sparse non-original routes",
        "Probe transport and output transport partially dissociate",
        "References",
        "Artifact, split, and intervention details",
    ]
    missing_text = [value for value in required_text if value not in pdf_text]
    add(checks, "required_pdf_text", not missing_text, {"missing": missing_text})
    unresolved_tokens = [value for value in ["??", "DAY29", "undefined citation", "TODO"] if value in pdf_text]
    add(checks, "no_unresolved_pdf_tokens", not unresolved_tokens, unresolved_tokens)

    build_log = (ROOT / "paper/audits/day30-manuscript-build.log").read_text()
    bib_log = (ROOT / "paper/audits/day30-manuscript-bibtex.log").read_text()
    warning_patterns = [r"undefined", r"LaTeX Warning", r"Package .* Warning", r"Overfull", r"Underfull"]
    build_warnings = [pattern for pattern in warning_patterns if re.search(pattern, build_log, re.IGNORECASE)]
    # BibTeX's success summary lists the built-in function ``warning$ -- 0``;
    # only match emitted diagnostics, not that function name.
    bib_diagnostic = re.compile(r"^(?:Warning--|Error\b|I couldn't open\b|I found no\b)", re.IGNORECASE)
    bib_warnings = [line for line in bib_log.splitlines() if bib_diagnostic.search(line)]
    add(checks, "warning_free_build", not build_warnings and not bib_warnings, {"latex": build_warnings, "bibtex": bib_warnings})

    manuscript = (ROOT / "paper/manuscript.tex").read_text()
    appendix_text = "\n".join(path.read_text() for path in sorted((ROOT / "paper/appendix").glob("*.tex")))
    bib = (ROOT / "paper/references.bib").read_text()
    cited: set[str] = set()
    for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", manuscript + "\n" + appendix_text):
        cited.update(item.strip() for item in group.split(","))
    entries = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))
    add(checks, "citation_resolution", cited == entries and len(cited) == 13, {"cited": len(cited), "entries": len(entries), "missing": sorted(cited - entries), "unused": sorted(entries - cited)})

    numerical = json.loads((ROOT / "paper/audits/day28-numerical-audit.json").read_text())
    add(checks, "numerical_claim_audit", numerical.get("status") == "pass" and numerical.get("claim_count") == 21, {"status": numerical.get("status"), "claims": numerical.get("claim_count")})

    figure_manifest = json.loads((ROOT / "paper/figures/figure-manifest.json").read_text())
    figure_hashes_ok = True
    for item in figure_manifest["figures"]:
        for field, hash_field in [("pdf", "pdf_sha256"), ("png", "png_sha256"), ("source_data", "source_data_sha256")]:
            path = ROOT / item[field]
            figure_hashes_ok &= path.is_file() and sha256(path) == item[hash_field]
    add(checks, "figure_manifest", figure_manifest.get("figure_count") == 6 and figure_hashes_ok and figure_manifest.get("published_or_sent") is False, {"figures": figure_manifest.get("figure_count"), "hashes_ok": figure_hashes_ok})

    visual = json.loads((ROOT / "paper/audits/day30-visual-audit.json").read_text())
    add(checks, "visual_page_audit", visual.get("status") == "pass" and visual.get("pages_inspected") == 14, {"status": visual.get("status"), "pages": visual.get("pages_inspected")})

    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = "src:scripts"
    test_run = subprocess.run(["pytest", "-q", "tests"], cwd=ROOT, text=True, capture_output=True, env=test_env)
    test_text = test_run.stdout + test_run.stderr
    add(checks, "full_test_suite", test_run.returncode == 0 and "62 passed" in test_text, {"returncode": test_run.returncode, "summary": test_text.strip().splitlines()[-1] if test_text.strip() else ""})

    diff_check = subprocess.run(["git", "diff", "--check", source_commit], cwd=ROOT, text=True, capture_output=True)
    add(checks, "git_diff_check", diff_check.returncode == 0, diff_check.stdout + diff_check.stderr)
    origin_main = command("git", "rev-parse", "origin/main").strip()
    ahead = int(command("git", "rev-list", "--count", "origin/main..HEAD").strip())
    tags = command("git", "tag", "--points-at", "HEAD").strip().splitlines()
    add(checks, "local_only_git_state", ahead > 0 and not tags, {"origin_main": origin_main, "commits_ahead": ahead, "tags_at_head": tags})

    behavioral_audit = json.loads((ROOT / "results/day-25/behavioral-package-audit.json").read_text())
    add(checks, "no_external_evaluator", behavioral_audit["checks"].get("no_external_evaluator") is True, behavioral_audit["checks"].get("no_external_evaluator"))
    release_wording = [
        "not released or submitted" in manuscript,
        "No independent group has replicated" in manuscript,
        "not pushed, released, submitted, uploaded, or sent" in (ROOT / "paper/README.md").read_text(),
    ]
    add(checks, "private_release_boundary_documented", all(release_wording), release_wording)

    result = {
        "schema_version": 1,
        "procedure": "day30-private-manuscript-candidate-audit-v1",
        "source_commit": source_commit,
        "pdf": str(PDF.relative_to(ROOT)),
        "pdf_sha256": sha256(PDF),
        "pdf_pages": pages,
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(item["pass"] for item in checks.values()),
        "published_released_submitted_or_sent": False,
        "status": "pass" if all(item["pass"] for item in checks.values()) else "fail",
    }
    parsed.output.parent.mkdir(parents=True, exist_ok=True)
    parsed.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "checks": result["check_count"], "passed": result["passed_count"], "pdf_sha256": result["pdf_sha256"]}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
