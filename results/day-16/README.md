# Day 16: Absolute-versus-Delta Site-Shuffling Triage

Day 16 tests whether the two surprising Day 14 mappings carry trigger-specific information after ordinary source/destination mismatch is held fixed or removed.

Primary comparisons:

- destination-matched normal-versus-triggered absolute sources through the same route;
- destination-relative `triggered - normal` source deltas;
- signed and scaled delta dose responses;
- per-example RMS-matched deltas;
- normal-to-normal and triggered-to-triggered mismatch controls.

Generated artifacts are sealed under this directory after the evaluator passes its benign preflight.

## Result

The result is mapping- and concept-specific:

- Deception's very large absolute induction is mostly same-condition head mismatch. A smaller trigger-specific signal survives, especially for seed 223: conditional induction is 0.142, raw delta induction is 0.100, and RMS-matched delta induction is 0.174.
- Harmfulness seed 211 contains a reproducible trigger-specific transported signal: conditional induction/rescue are 0.160/0.186 and RMS-matched delta induction/rescue are 0.202/0.263.
- Harmfulness seed 223 fails the portable-direction test: conditional and delta effects reverse even though absolute and same-condition mismatch effects can be large.
- Dose curves pass exactly through zero at alpha 0 and change direction under negative alpha, providing a signed intervention diagnostic.

The correct interpretation is therefore neither “all shuffling is portable” nor “all shuffling is damage.” Absolute replacement substantially exaggerates transfer, while particular routes retain a smaller destination-relative trigger signal.

## Artifacts

- `artifact-triage-results.jsonl.gz` — sealed 1,344-row grid.
- `artifact-triage-metrics.csv` — bootstrap estimates and intervals.
- `artifact-triage-summary.json` — predeclared cell classifications.
- `artifact-triage-overview.png` / `.pdf` — signed delta dose curves.
- `artifact-triage-preflight.json` and `artifact-triage-audit.json` — evaluator and result checks.
