# Day 61: Rapid Source-State Mediation Plan

## Question

Which upstream residual boundary or whole attention/MLP branch causally constructs
the named-concept and trigger/other K/V state that drives the acquired K12 operation?

## Frozen scan

Run 37 source-position candidates in both reciprocal directions: every residual
input through layer 12 and every normalized attention and MLP branch through layer
11. For each candidate, transplant the aligned natural donor state into the target
only at the informative named-concept plus trigger/other source positions. Pair every
cell with a deterministic signed-permutation control that preserves the complete
hidden-delta row norms and temporal Gram matrix.

Recompute the exact Day 60 full-prefix KV intervention on every example. Score each
candidate against three endpoints:

1. complete 13-probe monitor displacement caused by Day 60 KV;
2. the raw 12-head K12 response state caused by Day 60 KV, globally and separately
   at K12 layers 9, 10, 11, and 12;
3. the aligned pre-RoPE key/value projection state at the informative source tokens,
   globally and at each K12 layer.

## Mechanical decision

A site qualifies only if it passes all recovery, matched-control advantage,
endpoint-nearest, and per-layer gates in both directions. If a whole branch
qualifies, select the earliest attention-before-MLP branch. Otherwise report the
earliest qualifying residual boundary as a sufficient state but not a compact
writer. If nothing qualifies, accept a distributed/no-single-site result.

This is a prospectively bounded localization scan on the now-open Day 60 panel.
Any selected compact mediator requires a separately frozen fresh confirmation before
promotion to a population claim.

## Fast execution

Use singleton examples and one causal job per forward, shard by concept in fresh
processes, reduce mechanically, recover and verify every tensor, independently
re-reduce locally, and terminate only the dedicated 80 GB worker. The frozen ceiling
is `$100`.
