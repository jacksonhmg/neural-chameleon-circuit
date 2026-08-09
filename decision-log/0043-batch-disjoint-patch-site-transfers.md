# Decision 0043: Batch Disjoint Patch-Site Transfers

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B before scientific value access

## Context

The one-clone hook reproduced both frozen old-kernel hashes exactly, but effect attempt 5 showed no meaningful positive-cell throughput improvement. Count-only monitoring stopped it after 1,904 rows: 144 natural, 864 absolute, and 896 random. No raw margin, score, effect, selection statistic, or gate result was inspected.

An interrupt stack confirmed that the active hook cost was now the per-member mask and source transfer. For a 32-draw selected-head intervention, the hook still made up to 384 Python calls and CPU-to-device transfers even though entries at one patch site occupy disjoint expanded-batch rows.

## Decision

Within each hooked module, group entries by exact `PatchSite`. Concatenate their source captures in original job order, construct the corresponding disjoint expanded-batch row indices, transfer one repeated mask and one source tensor per site, and assign all rows at that site together. Different heads are disjoint destinations; each job is already required to contain no duplicate site.

Retain one full activation clone, site insertion order, job order within each site, response masks, selected positions, dtype/device conversion, and every patch value. The complete two-checkpoint preflight must reproduce the frozen old-hook result hashes exactly before donors or population execution.

## Scientific invariance

This changes transfer and indexing granularity only. Every destination row/head receives the same source value under the same mask. No operator, endpoint, population, seed, threshold, or result key changes.
