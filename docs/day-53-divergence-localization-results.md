# Day 53 Day 52 Divergence Localization

## Outcome

The saved Day 52 tensors locate the failed donor identity in a small but high-leverage part of the downstream representation. The 13 released probes span only 13 of 3,584 monitor dimensions. A median of `6.54%` and `5.98%` of the QKV intervention's donor-error energy lies in that probe-row subspace in the two reciprocal directions, yet this slice reduces unique donor identity from 10 of 13 concepts in the mean monitor residual to 9 and 8 of 13 in the complete probe-margin metric.

This is a post-hoc existing-artifact diagnostic. It does not repair Day 52 or establish whether the missing monitor-sensitive information is absent from the QKV-produced K12 state or reflects target-context dependence.

## Head and layer localization

The largest incomplete K12 transfers are concentrated early. Mean concept donor recoveries for `L9H13` are `0.640` and `0.646`; `L9H11` recovers `0.669` and `0.693`; `L9H04` recovers `0.746` and `0.659`. By contrast, `L11H08` and `L11H09` recover roughly `0.90–0.93` in both directions.

This concentration is not a sufficient explanation. All-caps has nearly exact layerwise recovery in irrelevant-to-correct transfer (`0.984–1.017`) while the complete probe vector selects the different donor. Mathematical has `0.876–0.972` layer recovery in correct-to-irrelevant transfer while its probe vector selects normal. Exact natural-donor K12 replacement is therefore required to distinguish a small missing selected-head component from an interaction with retained target context.

## Residual-space localization

| Direction | Mean monitor donor-nearest | Probe-row donor-nearest | Probe-margin donor-nearest | Median probe-row fraction of donor-error energy |
|---|---:|---:|---:|---:|
| correct → irrelevant target | 10/13 | 9/13 | 9/13 | 0.065 |
| irrelevant → correct target | 10/13 | 9/13 | 8/13 | 0.060 |

The probe vectors are well-conditioned for this diagnostic: all 13 directions are linearly independent and the row-space condition number is `2.18`. Thus the row/null result is not an artifact of a rank-deficient probe bank.

## Decision

Proceed with the already specified prospective discriminator: replace all 12 selected raw K12 heads with exact natural donor activity while retaining the target condition everywhere else. Use the unchanged Day 52 population, reciprocal donors, endpoints, controls, weighting, and 10-of-13 complete-probe identity rule.

The machine-readable decomposition is [`results/day-53/day52-divergence-decomposition.json`](../results/day-53/day52-divergence-decomposition.json).
