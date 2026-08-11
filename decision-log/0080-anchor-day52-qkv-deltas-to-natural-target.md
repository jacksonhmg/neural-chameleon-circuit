# Decision 0080: Anchor Day 52 QKV Deltas to the Natural Target

**Date:** 2026-08-11
**Status:** Frozen implementation repair before reciprocal outcomes

## Observation

The first CUDA preflight stopped before generating any reciprocal candidate outcome. Every implementation check passed except natural attention reconstruction: explicit float32 recomputation differed from the model's native natural raw-head output by as much as `0.10954058170318604`, above the frozen `0.05` tolerance.

## Decision

Preserve the exact source-minus-target full-prefix QKV operation while anchoring it to the model's natural target head output:

`natural target + F(source QKV, target context) - F(target QKV, target context)`.

This makes the same-condition operation exact and cancels the deterministic offset between explicit float32 eager-attention recomputation and the native bfloat16 model realization. It does not alter the donor, target, causal interface, controls, endpoints, metrics, thresholds, seeds, or stop rule. The original uncalibrated reconstruction error remains in the preflight report as an adverse numerical diagnostic; the frozen tolerance is not loosened.

## Boundary

Commit this repair before rerunning the preflight. If the repaired preflight passes, execute the complete frozen reciprocal matrix once. No scientific outcome from the failed implementation preflight may be used because none was generated.
