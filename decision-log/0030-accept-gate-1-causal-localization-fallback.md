# Decision 0030: Accept the Gate 1 Causal-Localization Fallback

- **Date:** 2026-08-09
- **Status:** Accepted

## Context

Decision 0028 authorized local execution of the successor causal-mechanisms plan subject to its mandatory phase gates. The acquired-writer Gate 1 implementation completed on all frozen populations under execution commit `b0292e2a98d575827f82570985edcfbdfa2d583a` and execution ID `ba510c5c95b26836342ad635e4700fdd0cdac0b71f31939ee7f8691b14adc399`.

All implementation checks passed. Representation acquisition relative to the exact precursor passed (`0.01130` absolute macro aligned amplitude; one-sided 95% UCB `0.01598`), as did functional acquisition (`0.03242` precursor-to-Chameleon effect ratio; UCB `0.05122`). The full K12 diagnostic also cleared the absolute held-out residual threshold with macro `R2_u = 0.32146`.

The diagnostic failed both frozen comparative requirements. The leave-one-example-out concept-label-only baseline obtained macro `R2_u = 0.59186` and probe-vector SNMSE `= 0.20586`. The full K12 model's `R2_u` improvement over that best baseline was `-0.27041`, below the required `+0.05`, and its relative probe-vector SNMSE was `3.66148`, above the allowed `0.90`.

## Decision

Accept Gate 1 as a scientific failure with a passing implementation audit. Apply Section 6.6 of the controlling plan mechanically:

- do not begin Gate 2 concept-to-writer prediction;
- do not run predicted-trajectory insertion/removal, endogenous propagation, concept-swap restoration, or fresh confirmation;
- do not use the target successor title; and
- narrow the successor result to causal localization.

The supported statement is that K12 is a potent late causal-intervention set and that its corresponding exact-precursor representation and functional effects are materially weaker. The evidence does not establish that observed K12 trajectories encode the required example-specific layer-12 displacement and complete probe-effect vector better than frozen baselines. Therefore “acquired writer computation” is not an authorized conclusion.

## Evidence and reproducibility

- [`results/day-33/gate-1-audit.json`](../results/day-33/gate-1-audit.json) records a passing implementation result, failed scientific result, and `next_phase_authorized = false`.
- [`results/day-33/acquired-writer-summary.json`](../results/day-33/acquired-writer-summary.json) records the acquisition and accounting clauses.
- [`results/day-33/intermediate-prediction-summary.json`](../results/day-33/intermediate-prediction-summary.json) records the full diagnostic, all five baselines, per-concept outcomes, and leakage audit.
- [`results/day-33/execution-artifact-manifest.json`](../results/day-33/execution-artifact-manifest.json) pins all ignored raw rows and feature tensors.
- Two runs of the committed reducer produced byte-identical principal outputs.

## Consequences

- The frozen successor plan is complete at its prescribed fallback boundary; skipped downstream phases are prohibited by the plan rather than unfinished work.
- Existing historical claims and the private manuscript are not upgraded by this development result.
- Any future attempt to study a different writer representation, predictor, boundary, or baseline must begin under a new prospective plan and authorization. It cannot revise this failed gate retroactively.
- The no-release boundary remains in force: no push, tag, release, submission, upload, external message, or author contact is authorized.
