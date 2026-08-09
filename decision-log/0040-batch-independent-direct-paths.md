# Decision 0040: Batch Algebraically Independent Direct Paths

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B effect execution before scientific value access

## Context

The sixteen-job total-effect kernel passed preflight and improved that half of the matrix, but effect attempt 3 remained dominated by one sequential frozen-write direct-path projection per intervention. Count-only monitoring stopped the run after 2,188 rows: 148 natural, 888 absolute, and 1,152 random. No raw margin, score, effect, selection statistic, or gate result was inspected.

## Decision

Preserve [effect attempt 3](../results/day-39/effect-execution-attempt-3.json) and its local JSONL files. Do not mix those rows with another kernel shape.

For a chunk of independent direct-path jobs, repeat the same target capture in candidate-major order, concatenate each job's raw-head replacements along the batch axis, use the natural target value when a job does not replace a component in the chunk union, project every changed branch once at the expanded batch shape, and split the monitor residual back into job-indexed captures.

Before population access, require exact equality between this batched algebra and independent direct-path execution for sixteen heterogeneous replacement jobs on both frozen checkpoints. Then regenerate donor tables under the new execution identity and restart every effect file from zero.

## Scientific invariance

This is batching only. Patch tensors, BF16 installation, operators, frozen downstream writes, monitor endpoints, populations, seeds, thresholds, and result keys remain unchanged. Candidate jobs have no cross-example computation; concatenating their batch rows is algebraically identical to evaluating each job separately.
