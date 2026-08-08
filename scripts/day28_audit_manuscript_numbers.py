#!/usr/bin/env python3
"""Verify the manuscript's central numerical ledger against sealed results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "paper/numerical-claims.csv"
OUTPUT = ROOT / "paper/audits/day28-numerical-audit.json"


def csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, keep_default_na=False)


def triple(row: pd.Series, estimate: str, low: str, high: str) -> tuple[float, float | None, float | None]:
    return float(row[estimate]), float(row[low]), float(row[high])


def main() -> None:
    observed: dict[str, tuple[float, float | None, float | None]] = {}

    day6 = csv("results/day-06/localization-summary.csv")
    for claim, direction in [("N01", "rescue"), ("N02", "induction")]:
        row = day6[(day6.scope == "macro") & (day6.token_region == "response") & (day6.layer == 11) & (day6.direction == direction)].iloc[0]
        observed[claim] = triple(row, "fraction", "ci_low", "ci_high")

    day9 = csv("results/day-09/selected-completeness-curve.csv")
    row = day9[(day9.scope == "validation") & (day9.set_size == 16)].iloc[0]
    observed["N03"] = triple(row, "recovery", "recovery_ci_low", "recovery_ci_high")

    day10 = csv("results/day-10/sufficiency-macro.csv")
    for claim, group in [("N04", "selected_k16"), ("N05", "random_k16")]:
        row = day10[(day10.scope == "validation") & (day10.label == 1) & (day10.group_id == group)].iloc[0]
        observed[claim] = triple(row, "fraction", "ci_low", "ci_high")

    day13 = csv("results/day-13/safety-transfer-metrics.csv")
    for claim, concept, direction in [
        ("N06", "deception", "rescue"), ("N07", "deception", "induction"),
        ("N08", "harmful", "rescue"), ("N09", "harmful", "induction"),
    ]:
        row = day13[(day13.concept == concept) & (day13.group_id == "selected_k16") & (day13.direction == direction) & (day13.label == 1)].iloc[0]
        observed[claim] = triple(row, "fraction", "fraction_ci_low", "fraction_ci_high")

    day21 = csv("results/day-21/confirmation-cells.csv")
    for claim, concept, role in [
        ("N10", "deception", "selected"), ("N11", "deception", "selected_minus_null"),
        ("N12", "harmful", "selected"), ("N13", "harmful", "selected_minus_null"),
    ]:
        scope = "population" if role == "selected" else "contrast"
        row = day21[(day21.scope == scope) & (day21.concept == concept) & (day21.direction == "induction") & (day21.estimand == "delta") & (day21.source_role == role)].iloc[0]
        observed[claim] = triple(row, "estimate", "ci_low", "ci_high")

    day23 = csv("results/day-23/behavioral-transport-cells.csv")
    for claim, concept, direction in [
        ("N14", "deception", "induction"), ("N15", "deception", "rescue"),
        ("N16", "harmful", "induction"), ("N17", "harmful", "rescue"),
    ]:
        row = day23[(day23.concept == concept) & (day23.direction == direction)].iloc[0]
        observed[claim] = triple(row, "selected_directional_coefficient_estimate", "selected_directional_coefficient_ci_low", "selected_directional_coefficient_ci_high")

    day24 = json.loads((ROOT / "results/day-24/coupled-generation-summary.json").read_text())
    observed.update({
        "N18": (float(day24["selected_exact_base_match_fraction"]), None, None),
        "N19": (float(day24["null_exact_base_match_fraction"]), None, None),
        "N20": (float(day24["selected_mean_base_token_f1"]), None, None),
        "N21": (float(day24["null_mean_base_token_f1"]), None, None),
    })

    ledger = pd.read_csv(LEDGER, keep_default_na=False)
    checks = []
    tolerance = 5e-7
    for _, row in ledger.iterrows():
        claim = row.claim_id
        actual = observed[claim]
        expected = (
            float(row.estimate),
            float(row.ci_low) if row.ci_low != "" else None,
            float(row.ci_high) if row.ci_high != "" else None,
        )
        passed = all(
            (left is None and right is None) or (left is not None and right is not None and abs(left-right) <= tolerance)
            for left, right in zip(expected, actual)
        )
        checks.append({"claim_id": claim, "pass": passed, "ledger": expected, "source": actual})

    result = {
        "schema_version": 1,
        "procedure": "day28-manuscript-numerical-audit-v1",
        "claim_count": len(checks),
        "all_claims_match": all(item["pass"] for item in checks),
        "tolerance": tolerance,
        "checks": checks,
        "status": "pass" if all(item["pass"] for item in checks) else "fail",
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "claims": len(checks), "output": str(OUTPUT.relative_to(ROOT))}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
