#!/usr/bin/env python3
"""Numerical, textual, and provenance audit for Manuscript V2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper-v2"
PDF = ROOT / "output/pdf/neural-chameleon-causal-mechanisms-v2.pdf"
AUDIT = PAPER / "audits/manuscript-v2-audit.json"


def load(path: str):
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def close(value: float, expected: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance)


def require(checks: list[dict], name: str, condition: bool, detail) -> None:
    checks.append({"name": name, "pass": bool(condition), "detail": detail})


def main() -> int:
    checks: list[dict] = []
    d59 = load("results/day-59/chameleon-confirmation-summary.json")
    d59p = load("results/day-59/precursor-acquisition-summary.json")
    d60 = load("results/day-60/qkv-source-factorial-summary.json")
    d61 = load("results/day-61/source-state-mediation-summary.json")
    d62 = load("results/day-62/checkpoint-hybridization-summary.json")
    d62q = load("results/day-62/qualification-calibration-summary.json")
    d67 = load("results/day-67/title-closure-summary.json")
    d70 = load("results/day-70/endpoint-correction-summary.json")

    c2i = d59["directions"]["correct_to_irrelevant"]["values"]
    i2c = d59["directions"]["irrelevant_to_correct"]["values"]
    require(checks, "exact K12 natural recovery", close(c2i["exact_k12_natural_probe_recovery"], 0.7788246956081869)
            and close(i2c["exact_k12_natural_probe_recovery"], 0.7602605778970062),
            [c2i["exact_k12_natural_probe_recovery"], i2c["exact_k12_natural_probe_recovery"]])
    require(checks, "K12 plus tail closure", close(c2i["joint_natural_probe_recovery"], 0.935241446872959)
            and close(i2c["joint_natural_probe_recovery"], 0.944267747875849),
            [c2i["joint_natural_probe_recovery"], i2c["joint_natural_probe_recovery"]])
    require(checks, "full tail identity", close(c2i["full_tail_natural_probe_recovery"], 1.0000000008666299, 1e-9)
            and close(i2c["full_tail_natural_probe_recovery"], 0.9999999933129193, 1e-8),
            [c2i["full_tail_natural_probe_recovery"], i2c["full_tail_natural_probe_recovery"]])

    precursor_max = max(
        value for direction in d59p["directions"].values()
        for value in direction["values"].values()
    )
    require(checks, "precursor maximum ratio", close(precursor_max, 0.08120611269735524), precursor_max)
    require(checks, "Day60 factor classification", d60["factor_classification"] == "proper_subset_sufficient:kv"
            and d60["selected_factor"] == "kv" and d60["three_way_interaction_material"] is True,
            {"classification": d60["factor_classification"], "selected": d60["selected_factor"],
             "three_way_material": d60["three_way_interaction_material"]})
    require(checks, "Day60 only KV proper subset qualifies",
            [k for k, v in d60["factor_qualification"].items() if v["qualifies"]] == ["kv", "qkv"],
            [k for k, v in d60["factor_qualification"].items() if v["qualifies"]])
    require(checks, "Day60 region classification", d60["region_classification"] == "distributed_across_prefix_regions"
            and close(d60["region_scores"]["monitoring_language"], 0.0),
            {"classification": d60["region_classification"], "scores": d60["region_scores"]})
    require(checks, "Day61 residual-only classification", d61["classification"] == "residual_state_only"
            and len(d61["qualifying_compact_branches"]) == 0
            and len(d61["qualifying_residual_boundaries"]) == 10,
            {"classification": d61["classification"], "compact": d61["qualifying_compact_branches"],
             "residual_count": len(d61["qualifying_residual_boundaries"])})
    require(checks, "Day62 early acquisition classification", d62["classification"] == "primary:early_representation"
            and close(d62["primary_group_worst_endpoint_losses"]["early_representation"], 0.7228020943979805),
            {"classification": d62["classification"], "losses": d62["primary_group_worst_endpoint_losses"]})
    require(checks, "unseen concept qualification stopped", d62q["selected_pair_count"] == 0
            and d62q["qualification_continue_gate_pass"] is False,
            {"selected_pair_count": d62q["selected_pair_count"],
             "continue": d62q["qualification_continue_gate_pass"]})

    old_counts = [
        d67["operational_consequences"]["aggregate"]["0.01"]["drop"]["positive_concepts"],
        d67["operational_consequences"]["aggregate"]["0.05"]["drop"]["positive_concepts"],
    ]
    require(checks, "formal old breadth gate remains 8/11", old_counts == [8, 8]
            and d70["prospective_gate_details"]["formal_day68_gate_remains"] == "failed",
            {"counts": old_counts, "status": d70["prospective_gate_details"]["formal_day68_gate_remains"]})

    expected_macro = {
        "native": {"total_evasion": 0.4862057427611118, "K12_recovery": 0.3374078149423109,
                   "K12_induction": 0.3924346712925894},
        "broad": {"total_evasion": 0.45553771810690785, "K12_recovery": 0.31633454355919227,
                  "K12_induction": 0.3682511765086515},
    }
    macro_actual = {}
    all_concepts_positive = True
    for panel, endpoints in expected_macro.items():
        macro_actual[panel] = {}
        for endpoint, expected in endpoints.items():
            value = d70["effects"][panel]["scores"][endpoint]["macro"]["equal_concept_point"]
            macro_actual[panel][endpoint] = value
            all_concepts_positive &= all(
                item["lower_95"] > 0
                for item in d70["effects"][panel]["scores"][endpoint]["per_concept"].values()
            )
            require(checks, f"Day70 {panel} {endpoint} macro", close(value, expected), value)
    require(checks, "Day70 all 66 per-concept intervals positive", all_concepts_positive, all_concepts_positive)
    require(checks, "released score and exact K12 parity",
            d70["prospective_gate_details"]["released_score_parity_max_abs"] == 0.0
            and d70["prospective_gate_details"]["exact_K12_identity_max_abs"] == 0.0,
            {"score": d70["prospective_gate_details"]["released_score_parity_max_abs"],
             "k12": d70["prospective_gate_details"]["exact_K12_identity_max_abs"]})

    figure_csvs = sorted((PAPER / "source-data").glob("figure-*.csv"))
    require(checks, "five figure source tables", len(figure_csvs) == 5, [p.name for p in figure_csvs])
    for path in figure_csvs:
        with path.open(newline="") as f:
            count = sum(1 for _ in csv.DictReader(f))
        require(checks, f"nonempty {path.name}", count > 0, count)

    ledger_rows = list(csv.DictReader((PAPER / "claim-evidence-ledger.csv").open(newline="")))
    missing_evidence = [row["decisive_artifact"] for row in ledger_rows
                        if not (ROOT / row["decisive_artifact"]).exists()]
    require(checks, "claim ledger artifacts exist", not missing_evidence, missing_evidence)

    manuscript = (PAPER / "manuscript.tex").read_text()
    required_phrases = [
        "joint source-side K/V rewrite",
        "formal 8/11 gate remains a recorded failure",
        "We do not claim semantic operation on unseen concepts",
        "No further experiment is required to complete this case study",
    ]
    require(checks, "required claim-boundary language present",
            all(p in manuscript for p in required_phrases),
            {p: p in manuscript for p in required_phrases})
    prohibited = ["Author(s) TBD", "is a universally minimal circuit", "generalizes to unseen concepts"]
    require(checks, "prohibited overclaim/placeholders absent",
            all(p not in manuscript for p in prohibited),
            {p: p in manuscript for p in prohibited})

    require(checks, "final PDF exists", PDF.exists() and PDF.stat().st_size > 100_000,
            PDF.stat().st_size if PDF.exists() else None)
    pages = None
    extracted = ""
    if PDF.exists():
        info = subprocess.run(["pdfinfo", str(PDF)], check=True, capture_output=True, text=True).stdout
        match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
        pages = int(match.group(1)) if match else None
        extracted = subprocess.run(["pdftotext", str(PDF), "-"], check=True,
                                   capture_output=True, text=True).stdout
    require(checks, "PDF page count plausible", pages is not None and 10 <= pages <= 40, pages)
    require(checks, "PDF has no unresolved references", "??" not in extracted, "??" in extracted)
    require(checks, "PDF contains title and conclusion",
            "Causal Mechanisms of Activation-Monitor Evasion" in extracted
            and "No further experiment is required" in extracted,
            {"title": "Causal Mechanisms of Activation-Monitor Evasion" in extracted,
             "conclusion": "No further experiment is required" in extracted})
    build_log = (PAPER / "audits/manuscript-v2-build.log").read_text()
    require(checks, "LaTeX has no overfull boxes", "Overfull" not in build_log,
            len(re.findall(r"Overfull", build_log)))
    rendered = sorted((PAPER / "audits/rendered-pages").glob("page-*.png"))
    require(checks, "every PDF page rendered", pages is not None and len(rendered) == pages
            and all(p.stat().st_size > 10_000 for p in rendered),
            {"pdf_pages": pages, "rendered_pages": len(rendered)})

    tracked_inputs = [
        PAPER / "manuscript.tex", PAPER / "references.bib", PAPER / "claim-evidence-ledger.csv",
        PAPER / "evidence-chronology.csv", *figure_csvs,
        *sorted((PAPER / "figures").glob("figure-*.pdf")), PDF,
    ]
    manifest = {str(p.relative_to(ROOT)): sha256(p) for p in tracked_inputs if p.exists()}
    passed = all(item["pass"] for item in checks)
    record = {
        "schema_version": 1,
        "procedure": "manuscript-v2-numerical-text-provenance-audit-v1",
        "pass": passed,
        "checks": checks,
        "headline_values": macro_actual,
        "pdf_pages": pages,
        "files": manifest,
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pass": passed, "checks": len(checks), "pdf_pages": pages,
                      "audit": str(AUDIT), "pdf_sha256": manifest.get(str(PDF.relative_to(ROOT)))}, indent=2))
    if not passed:
        for item in checks:
            if not item["pass"]:
                print(f"FAILED: {item['name']}: {item['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
