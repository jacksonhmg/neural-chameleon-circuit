# Day 57 fresh-confirmation result

## Outcome

The 176-example fresh causal confirmation passed every implementation check but failed the precommitted scientific gate. The mechanical disposition is `fresh_confirmation_fail_stop`. Under the frozen plan, value-path tracing and the exact-precursor comparison were not run.

This is not a failure of K12 causality. Exact natural-donor K12 remained strongly causal on untouched examples. It is a failure to confirm the stronger Day 56 compact description at its frozen quantitative strength.

## Primary results

All values below are median-over-concept probe-effect recoveries across 11 compatible concepts.

| Intervention | Correct → irrelevant | Irrelevant → correct | Frozen requirement |
|---|---:|---:|---:|
| Exact donor K12 | **0.777** | **0.767** | ≥ 0.70 |
| Content hybrid | 0.516 | 0.607 | ≥ 0.70 |
| Routing hybrid | 0.147 | 0.091 | ≤ 0.30 |
| Monitoring-prefix install | 0.649 | 0.645 | ≥ 0.80 |
| Monitoring-prefix removal | **0.931** | **0.911** | ≥ 0.80 |
| Fixed-RMSNorm exact-effect survival | **1.061** | **1.061** | ≥ 0.80 |

Exact donor K12 was donor-nearest in 10/11 concepts in each direction and reconstructed K12 itself exactly. Monitoring-prefix install beat its matched Haar control by `0.596` and `0.563`, but left residual probe-effect norm ratios of `0.375` and `0.366`, above the frozen `0.25` ceiling.

## Interpretation

The robust prospective statement is now narrower:

> K12 is a distributed, primarily direct causal source of monitor evasion on fresh examples. Its effect survives fixed RMS normalization. Content changes dominate routing changes, and the monitoring prefix is highly causal, especially under removal, but neither the content hybrid nor prefix installation reconstructs the full donor effect with the generality claimed by the Day 56 development result.

The install/remove asymmetry and the drop from development to fresh data reinforce the Day 56 downstream classification `mixed_or_nonlinear_state_dependent`. They argue against treating a single prefix-V operation as already established.

## Why stages 2 and 3 stopped

The prospective contract required every confirmation clause in both directions before opening the separate 44-example pathway panel. Content recovery and monitoring-prefix install failed in both directions; content-minus-routing also failed in one direction. Continuing to V/QK tracing after observing this result would have violated the stop rule and converted the separate panel into post-selection data.

The untouched 44-example panel remains available for a newly justified, newly frozen question, but it no longer counts as untouched for the exact Day 57 branch if that branch is retroactively changed.

## Audit

- CUDA preflight: pass.
- 44 valid shards / 176 examples / 88 files; all tensor hashes verified.
- Same-runtime second reduction: byte-identical.
- Independent Linux/macOS reductions: identical decisions and gate booleans; maximum floating difference `3.55e-15`.
- Dedicated H100: terminated and absent from inventory; maximum observed cost about `$1.57`, below the `$100` ceiling.
- Pre-existing A100: untouched.

Machine-readable evidence is in `results/day-57/confirmation-summary.json`, with the execution, preflight, manifest, ineligibility records, and resource ledger in the same directory.
