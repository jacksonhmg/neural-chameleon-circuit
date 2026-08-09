# Decision 0041: Reject Batched Direct Paths and Preflight 32 Total Jobs

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance restart  
**Scope:** Phase B before donor regeneration or population access

## Failed preflight

The Decision 0040 implementation completed both real-checkpoint preflights but failed the frozen exact-equality rule. Relative to independent direct-path execution, the expanded-batch BF16 projection had maximum hidden-state error `0.015625` in both Chameleon and precursor. No population was accessed and no scientific value was inspected. The failed output is preserved as [preflight attempt 4](../results/day-39/real-checkpoint-preflight-attempt-4.json).

A timing-only Chameleon preflight-batch benchmark also rejected the performance premise: 32 independent cached direct projections took 0.581 seconds, while the expanded-batch form took 0.923 seconds. Independent probe summarization took 0.016 seconds. The direct path was not the meaningful bottleneck.

## Decision

Remove batched direct execution from the population runner and retain the exact independent direct path with cached target branch recomputations.

The 32 frozen random controls are the largest repeatedly evaluated total-effect family. Preflight all 32 total-effect jobs in one cached-tail call at the final two-example batch shape on both checkpoints. Require exact margin and score equality against full execution, as before. If the 32-job shape passes and fits available memory, use job chunk size 32 for every effect family, regenerate donors under the new execution identity, and restart all effect files from zero.

## Scientific invariance

The rejected batched-direct output is never used. The only retained change is a larger total-effect execution batch. Patch tensors, operators, paths, endpoints, populations, seeds, thresholds, and result keys remain unchanged.
