# Day 59: Selected K12 Mechanism Confirmation and Acquisition Results

## Result

The selected mechanism confirms on the previously untouched 44-example panel and is
almost absent in the exact precursor. Under the frozen rule, its classification is
**acquired**.

The supported causal account is:

> Trigger condition changes are read through a coordinated monitoring-prefix QKV
> operation across the 12 selected K12 heads. That operation constructs the
> condition-specific K12 state, which supplies roughly three quarters of the natural
> monitor displacement directly. A parallel distributed contribution from the other
> 52 tail attention heads brings the joint intervention to about 94% of the natural
> endpoint. Both the K12 operation and its monitor effect are acquired in the Neural
> Chameleon and are very weak in the exact precursor.

This is not evidence for one special head, a V-only content copy, a QK-only routing
change, a layer-9 residual controller, an MLP controller, or a self-contained K12
code for the complete natural endpoint.

## Fresh Chameleon confirmation

Every frozen gate passed in both directions:

| Metric | Correct to irrelevant | Irrelevant to correct |
|---|---:|---:|
| Exact K12 recovery of natural monitor effect | 0.779 | 0.760 |
| QKV recovery of exact-K12 probe effect | 0.907 | 0.932 |
| QKV recovery in K12 coordinates | 0.904 | 0.886 |
| QKV advantage over exact orthogonal control | 0.842 | 0.807 |
| QKV exact-nearest concepts | 11/11 | 11/11 |
| Exact K12 + other 52 heads recovery of natural effect | 0.935 | 0.944 |
| Joint increment over exact K12 | 0.156 | 0.184 |
| Joint advantage over orthogonal context | 0.219 | 0.224 |
| Joint residual norm ratio | 0.113 | 0.106 |
| Full-tail positive-control recovery | 1.000 | 1.000 |

The held-out values closely reproduce the Day 58 development estimates without
relaxing a threshold or adding a candidate.

## Exact-precursor acquisition test

The precursor ran the same 44 examples, response tokens, masks, probes, jobs, and
directions. Response hashes match exactly across checkpoints. Each entry below is the
frozen median-over-concepts of the precursor effect norm divided by the Chameleon
effect norm; the frozen upper bound was `0.50` in both directions.

| Effect-norm ratio | Correct to irrelevant | Irrelevant to correct |
|---|---:|---:|
| Natural monitor effect | 0.0125 | 0.0125 |
| Natural K12 effect | 0.0812 | 0.0812 |
| QKV-mediated monitor effect | 0.00762 | 0.00620 |
| QKV-mediated K12 effect | 0.0611 | 0.0578 |
| Joint K12 + other-head monitor effect | 0.0113 | 0.0114 |

All ten ratio clauses pass. The result is not marginal: even the largest precursor
ratio is only `0.0812`, far below `0.50`.

## Audit and reproducibility

- Chameleon and precursor each contain 11 concept shards, 44 examples, and 968 state
  rows.
- Candidate-blind preflights passed with zero identity K12 and monitor error.
- The final completion audit recorded approximately 18.5 GB allocated after loading;
  every singleton shard remained within the H100's 80 GB memory.
- All 44 recovered shard files total 5,901,306 bytes; every tensor hash matches its
  metadata.
- Independent remote and local reductions have identical structures, gates,
  decisions, and classification. Maximum numerical disagreement is `7.1e-15` for
  Chameleon and `3.5e-18` for the precursor ratios.
- A one-line orchestration repair was required after the first valid shard: a
  deliberately partial invocation was incorrectly required to report a complete
  experiment. That shard was quarantined, the repair was committed, and every valid
  shard was regenerated under the new commit. No candidate, gate, state, or metric
  changed.

## Claim boundary and next question

This result identifies and confirms an acquired causal attention mechanism at the
QKV/K12/tail-attention level. It does not yet identify the earliest semantic state
that creates the donor Q, K, and V changes, nor does it show that a small source-token
or parameter subset is sufficient. The most informative next question is therefore
where the acquired QKV difference originates—not whether K12 is causal, whether V or
QK alone suffices, or whether the mechanism exists in the precursor.
