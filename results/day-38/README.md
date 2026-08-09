# Day 38: Phase A Descriptive Re-identification

Phase A ran under commit `d0b6c5d` on all 256 discovery positives and all 610 validation/safety positives. The leakage audit passes, all 866 precursor/Chameleon feature pairs align, and the sealed probe standardization reproduces with zero maximum error.

These are existing-data descriptive results. They cannot repair the original Gate 1 failure.

## Main result

- Between-concept/response-decile fraction: `0.588` for full residual displacement and `0.816` for the standardized complete probe vector.
- Residualized heldout selected-K12 diagnostic: `R² = -0.0068`, complete-probe-vector SNMSE `= 1.072`.
- Residualized zero concept/decile oracle: `R² = 0.0014`, SNMSE `= 1.000`.
- Across-concept K12: probe-vector SNMSE `= 1.305`, `R² = 0.352`.
- Across-concept matched nonselected heads: probe-vector SNMSE `= 0.614`, `R² = 0.310`.

The detailed natural K12 trajectory does not explain heldout within-concept variation and does not dominate matched nonselected trajectories across the thirteen already-inspected concepts. Most target variance lies between concept/position cells. This strengthens the falsification of K12 as a complete example-specific carrier and motivates—but does not prove—the separate controller/gating/amplification tests in Phase B.

## Artifacts

- `variance-decomposition-summary.json`
- `within-concept-diagnostic-summary.json`
- `cross-concept-diagnostic-summary.json`
- `phase-a-leakage-audit.json`
- `phase-a-attempt-1.json` (preserved pre-input startup failure)
