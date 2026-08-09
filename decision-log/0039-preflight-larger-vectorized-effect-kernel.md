# Decision 0039: Preflight a Larger Vectorized Effect Kernel

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B effect execution before scientific value access

## Context

The first Phase B effect attempt used four vectorized jobs per cached-tail call. After 3,480 rows, count-only monitoring measured roughly four positive batches per minute, implying about 3.5 hours for random controls alone. No raw margin, score, effect, selection statistic, or gate result was inspected.

## Decision

Stop and preserve [effect attempt 2](../results/day-39/effect-execution-attempt-2.json) plus its local JSONL files. Do not mix those rows with another kernel shape.

Make two computation-only changes:

1. preflight sixteen expanded jobs on the final two-example shape in both checkpoints, then use job chunk size sixteen for every effect family; and
2. cache each target attention-branch recomputation once per condition/direction and reuse it across direct-path replacements.

Commit, rerun the full preflight, regenerate donor tables under the new execution identity, and restart every effect from zero.

## Scientific invariance

Patch tensors, BF16 installation, operators, paths, endpoints, populations, seeds, thresholds, and row keys remain identical. Cached execution must be exactly equal to full execution at the new `[16 jobs × 2 examples]` kernel shape before population access.
