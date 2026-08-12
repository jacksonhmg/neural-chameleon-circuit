# Day 56: Joint-K12 Mathematical-Mechanism Results

## Result

K12 is best described as a **distributed, content-dominant rewrite of the monitoring prefix**, especially the named concept and the alternative-trigger specification. It is not primarily an attention-routing operation, and its monitor effect is not mediated by changing RMSNorm denominators.

The exact bounded statement is:

> Condition-specific attention changes the value content written from the monitoring prefix into a coordinated 12-head K12 state. The named-concept and alternative-trigger regions carry most of the causal monitor effect. Downstream layers preserve the direction of this write but amplify it nonlinearly and state-dependently. No frozen compact routing/content or source-region component fully closes exact reciprocal donor identity.

This is development-sandbox evidence, not fresh-checkpoint confirmation.

## Exact attention algebra

For each selected response-query head and atomic source region, the run decomposed the raw pre-`o_proj` output as

```text
z = sum_r m[r] * mu[r]
```

where `m[r]` is attention mass and `mu[r]` is the conditional value mean. The reconstruction error was at most `0.01563` against the frozen `0.05` limit, and the routing/content Shapley closure error was at most `9.54e-7` against `1e-5`.

The causal results separate routing from content:

| Component | Correct → irrelevant probe recovery | Irrelevant → correct probe recovery |
|---|---:|---:|
| Donor routing with target content | 0.100 | 0.206 |
| Target routing with donor content | 0.782 | 0.866 |
| Routing Shapley installation | 0.186 | 0.160 |
| Content Shapley installation | 0.826 | 0.771 |

Thus the operation is primarily a change in attended value content, not a change in where attention mass is placed.

## Source localization

The strongest frozen component was the complete monitoring-prefix delta:

| Frozen clause | Correct → irrelevant | Irrelevant → correct |
|---|---:|---:|
| Installation recovery | 0.861 | 0.871 |
| Donor-side removal recovery | 0.888 | 0.870 |
| Advantage over matched Haar | 0.741 | 0.741 |
| Remaining effect-norm ratio | 0.165 | 0.175 |
| Donor-nearest concepts | 10/13 | 8/13 |

It passed installation, removal, matched-control, and residual-closure clauses in both directions. It failed only the frozen reciprocal donor-identity clause because the reverse direction reached `8/13`, below `10/13`. Therefore the mechanical selector chose no compact operator.

Within the monitoring prefix, the named-concept region recovered `0.299/0.348` of the probe effect and the alternative-trigger region recovered `0.575/0.528`. Monitoring boilerplate was negligible (`0.00060/0.00037`). Original-prompt, template, and response-history components were also small. These atomic interventions are not expected to add exactly after nonlinear downstream processing.

## Downstream operation

Free K12 dose curves were close to, but failed, the frozen linearity gate:

| Diagnostic | Correct → irrelevant | Irrelevant → correct | Gate |
|---|---:|---:|---:|
| Dose-curve R² | 0.955 | 0.967 | ≥ 0.98 |
| Maximum in-range secant deviation | 0.109 | 0.105 | ≤ 0.10 |
| Target-Jacobian exact-effect recovery | 0.600 | 0.513 | ≥ 0.80 |
| Target-Jacobian exact-effect cosine | 0.945 | 0.950 | ≥ 0.90 |
| Jacobian 90%-energy rank | 10 | 10 | diagnostic |

The local Jacobian points in almost the correct output direction but captures only about half the endpoint magnitude. This is evidence for downstream state-dependent amplification rather than a purely local linear translation.

Freezing every RMSNorm denominator in layers 9–12 did not remove the effect. Recovery was `1.058/1.158`, with cosine `0.982/0.979`. The simple normalization-mediated explanation is therefore rejected; if anything, the fixed denominators slightly increased the aligned effect.

The frozen downstream classification is `mixed_or_nonlinear_state_dependent`.

## Audit and execution

- The outcome-blind preflight generated no candidate outcomes and passed exact identity, fixed-normalization identity, attention reconstruction, response, finiteness, and hook-cleanup checks.
- A first full attempt was rejected because a batch-2 Jacobian identity differed from the frozen batch-60 endpoint by `0.05569`, above the unchanged `0.05` gate. Its files were quarantined before scientific acceptance.
- The Jacobian was mechanically aligned to the already-frozen 30-job batch geometry. The valid rerun reduced the identity error to `3.81e-6` without changing candidates or thresholds.
- The valid run contains 13 unique concept shards and exactly 1,976 state rows. Every implementation check passes.
- All 35 manifest entries totaling 1,573,498,307 bytes were recovered and hash-verified.
- An independent local reduction matched the remote structure and labels exactly across 18,192 numeric cells; maximum absolute platform-level numeric difference was `2.38e-7`.

## Boundary

No single selected head or layer, routing term, content term, or frozen source group is an exact self-contained evasion code. The result identifies the dominant mathematical operation and rejects several alternatives, but it does not establish a fresh-checkpoint invariant or fully explain the residual concept-specific cases.
