# Day 34: Run the Observed-Trajectory Diagnostic

## Phase information

- **Sprint phase:** Day 34
- **Status:** Complete
- **Calendar date:** 2026-08-09
- **Frozen protocol:** [Gate 1 contract](../results/day-31/frozen-acquired-writer-plan.json)
- **Result package:** [Day 33–35 result](../results/day-33/README.md)

## Objective

Fit the frozen discovery-only K12 diagnostic, evaluate it and all five preregistered baselines on the nine held-out validation and safety concepts, and reproduce the outputs deterministically without target leakage.

## Session log

### 2026-08-09 — Session 1

#### Work performed

- Loaded 866 paired positive-example feature artifacts from each checkpoint.
- Fit every standardizer and PCA basis on only the 256 discovery positives.
- Selected ridge `alpha = 1.0` by the frozen leave-one-discovery-concept-out procedure.
- Evaluated the full K12 diagnostic and the training-mean, concept-label-only, normal-state-only, nonselected-head, and exact-precursor baselines on all 610 held-out positives.
- Enforced the feature contract: observed K12 delta, normal `resid_post[8]`, and response-relative position were the only permitted feature families; the target was isolated from feature construction.
- Re-ran the committed reducer from the identical analysis commit and compared all output bytes.

#### Results and evidence

- Full K12 macro held-out `R2_u`: `0.3214553074` (absolute gate passed).
- Full K12 macro probe-vector SNMSE: `0.7537610852`.
- Best frozen baseline: leave-one-example-out concept-label mean with macro `R2_u = 0.5918630811` and probe-vector SNMSE `= 0.2058623539`.
- K12 improvement over the best `R2_u` baseline: `-0.2704077738`, below the frozen `+0.05` requirement.
- K12-to-best-baseline SNMSE ratio: `3.6614809409`, above the frozen `0.90` ceiling.
- All nine held-out concepts were retained and reported. No feature leakage or target-state use was detected.
- Both reducer runs produced byte-identical component, acquisition, prediction, audit, and artifact-manifest files.

#### Failures and unexpected observations

- The diagnostic predicted a nontrivial fraction of held-out layer-12 displacement but did not add predictive value over the strong concept-label-only outcome baseline.
- The comparative residual and complete-probe-vector clauses both failed. This is a scientific failure under the frozen gate, not an implementation failure.
- A preliminary `--help` invocation started the reducer because the script has no CLI parser. The duplicate process was identified by its unique argument and terminated before closeout; it read the same immutable inputs. The authoritative no-argument run and its independent reproducibility rerun both completed normally.

#### Artifacts and commits

- [`intermediate-prediction-summary.json`](../results/day-33/intermediate-prediction-summary.json)
- [`gate-1-audit.json`](../results/day-33/gate-1-audit.json)
- Result commit: `2073cb5` (`record acquired writer gate 1 result`)

## Open questions and blockers

None for Gate 1 adjudication. Gate 2 is not blocked by missing engineering work; it is scientifically unauthorized by the frozen continue-or-fallback rule.

## Decisions

- [Decision 0030](../decision-log/0030-accept-gate-1-causal-localization-fallback.md)

## Handoff

- **Current state:** The observed-trajectory diagnostic is complete, leakage-audited, and reproducible.
- **Next action:** Apply and record the mechanical Gate 1 fallback.
- **Verification still needed:** Full repository test and documentation audit at project closeout.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-09
- **Final evidence:** [`intermediate-prediction-summary.json`](../results/day-33/intermediate-prediction-summary.json) and [`gate-1-audit.json`](../results/day-33/gate-1-audit.json)
- **Final commits:** `2073cb5`; documentation closeout recorded in subsequent git history.
- **Carry-forward items:** None. The failed gate controls the downstream disposition.
