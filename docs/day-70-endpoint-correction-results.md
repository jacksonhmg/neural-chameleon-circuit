# Day 70: corrected operational endpoint result

**Disposition:** `scientific_title_worthy_under_corrected_endpoint`.

**Current status:** This is the final operational result. See
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md),
[Decision 0102](../decision-log/0102-accept-corrected-endpoint-title-worthiness.md),
and [Decision 0103](../decision-log/0103-close-project-after-manuscript-v2.md) for
the accepted interpretation, scope, and project closure.

The formal Day 68 9/11 threshold-crossing gate remains failed; it is not retroactively changed. Day 70 answers the corrected scientific question using ordinary no-trigger execution as the operational baseline, the released monitor's tokenwise-sigmoid-then-average score, paired continuous causal effects as the primary endpoint, and threshold TPR/FPR only as an operational consequence.

## Primary results

| Panel | Estimand | Macro score effect | 95% CI | Concepts point > 0 | Concepts lower 95% > 0 |
|---|---|---:|---:|---:|---:|
| Native, 40/concept | Total evasion: normal − correct trigger | 0.486206 | [0.462270, 0.509384] | 11/11 | 11/11 |
| Native, 40/concept | Exact-K12 recovery: correct[K12←normal] − correct | 0.337408 | [0.319274, 0.355750] | 11/11 | 11/11 |
| Native, 40/concept | Exact-K12 induction: normal − normal[K12←correct] | 0.392435 | [0.371260, 0.413273] | 11/11 | 11/11 |
| Broad, 128/concept | Total evasion | 0.455538 | [0.448728, 0.462472] | 11/11 | 11/11 |
| Broad, 128/concept | Exact-K12 recovery | 0.316335 | [0.311766, 0.320988] | 11/11 | 11/11 |
| Broad, 128/concept | Exact-K12 induction | 0.368251 | [0.361973, 0.374568] | 11/11 | 11/11 |

On the prospectively gated native panel, exact K12 accounts for **69.4%** of the macro evasion effect by recovery and **80.7%** by reciprocal induction. The secondary response-mean raw-margin effects tell the same story: total `+8.652`, recovery `+6.310`, and induction `+7.100`, with zero-excluding intervals in all 11 concepts.

All prospective clauses passed: the three macro lower bounds exceed zero; every per-concept point estimate and lower bound exceeds zero; both K12 fractions exceed 0.5; released-score parity is exactly `0.0`; exact K12 replacement error is exactly `0.0`; and the six prior acquisition/operation/conditioning/causal-chain clauses remain passed.

## What happened to the old 8/11 result

The old threshold gate was faithfully reported but scientifically misinterpreted. The corrected released-score diagnostic shows why:

- On the broad panel at 5% nominal FPR, all-caps is nearly pinned at the ceiling (`TPR normal=1.000`, `correct=0.984`), even though its continuous total effect is positive with a zero-excluding interval.
- German was not a missing effect: ordinary execution gives `TPR=1.000` versus `0.016` under the correct trigger. The old irrelevant-trigger comparison used another monitoring prompt as its baseline and obscured most of this operational effect.
- Literature was also not a missing effect: ordinary execution gives `TPR=0.414` versus `0.000` under the correct trigger. Exact-K12 threshold recovery remains floor-pinned at this particular threshold, despite a large, precisely positive continuous K12 effect.

The final-negative panel realized mean FPRs of 1.42% and 7.21% at the frozen nominal 1% and 5% calibration thresholds. These finite-panel deviations are reported, not optimized away.

## Scope and verification

The native panel contains project-unused examples sampled from the released probe-training distribution and independently label-qualified. It is not model- or probe-unseen. The broad panel is content-disjoint and deliberately distribution-shifted. Therefore, the strongest supported claim is broad causal operation across all 11 trained concepts, not generalization to concepts absent from chameleon fine-tuning.

The frozen reduction was executed twice within the reducer and once again in an independent process. All runs produced summary SHA-256 `1c4df11b5203246f9110d1bb4c60fbbb44d8ec9195ba0ec55f41e9be8b67a41b`. The 46-file artifact manifest has zero missing or mismatched files.
