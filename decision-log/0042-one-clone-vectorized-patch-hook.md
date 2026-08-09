# Decision 0042: Use One Clone in the Vectorized Patch Hook

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B before scientific value access

## Context

Effect attempt 4 exercised the exact 32-job total kernel but remained too slow. Count-only monitoring stopped it after 2,472 rows: 152 natural, 912 absolute, and 1,408 random. No raw margin, score, effect, selection statistic, or gate result was inspected.

Code audit located the bottleneck in `VectorizedTransplantRunner._register_jobs`. The hook first clones the expanded activation, then calls `_patch_response_values` for every job/member. That helper clones the entire already-expanded activation again before changing one response slice. A 32-draw selected-head intervention therefore makes up to 384 redundant full-tensor clones in one hooked forward call.

## Decision

Before changing the hook, hash the exact ordered 32-job margin, score, and activation-RMS tensors on the frozen two-example preflight batch in both checkpoints.

Then replace the per-member clone calls with one initial clone and the same ordered, in-place response-slice assignments. Preserve job order, member order, response masks, selected positions, raw-head reshaping, source dtype/device conversion, and every patch value. Rerun the complete preflight and require exact equality to the old-hook hashes in both checkpoints, in addition to the existing exact cached/full checks.

If and only if exact equality passes, regenerate donors under the new execution identity and restart all effect files from zero.

## Scientific invariance

The optimization changes allocation behavior only. The destination tensor starts from the same clone and receives the same assignments in the same order. No operator, patch tensor, endpoint, population, seed, threshold, or result key changes.
