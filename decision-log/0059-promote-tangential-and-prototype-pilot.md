# Decision 0059: Promote Only Tangential Activity and the Prototype Comparator

**Date:** 2026-08-10
**Status:** Accepted for the frozen Day 44 pilot

## Decision

Apply the Day 44 offline promotion rule without modification. Promote `tangential_actual_activity` as the only structured candidate and retain `concept_position_prototype` as the mandatory simple comparator.

Do not implement the global additive, shared rank 1/2/4/8/16, or normal-state-conditioned rank 1/2/4/8/16 families in the causal pilot. They failed the frozen complete-vector criteria in [`offline-screen-summary.json`](../results/day-44/offline-screen-summary.json).

## Interpretation

The concept/position prototype preserves most of the monitor-facing K12 write descriptively, while the tangential part of the natural after-`o_proj` activity preserves nearly all of it. Neither result is causal evidence for a prototype writer or rotation. Only the committed bidirectional direct-path intervention may promote an operation to Day 45.

## Scope

This decision uses existing development examples only and does not alter the failed Gate 1 or Phase C results, authorize fresh data, or earn the target title.
