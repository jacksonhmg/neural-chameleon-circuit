# Decision 0037: Fix Phase A–B Frozen-Status Validation

**Date:** 2026-08-09  
**Status:** Accepted implementation fix  
**Scope:** Phase A–B runner startup; no input or outcome access

## Failure

The first Phase A invocation stopped before loading any feature artifact. The machine-readable contract uses the status `frozen-before-post-gate1-phase-a-b-outcomes`; the Phase A and Phase B runners incorrectly required the shorter literal `frozen`.

## Decision

Preserve [Phase A attempt 1](../results/day-38/phase-a-attempt-1.json). Change both runners to require the contract's exact literal status. Add a regression assertion and rerun the complete two-checkpoint preflight under the new commit before restarting Phase A.

No operator, threshold, population, transform, endpoint, or scientific rule changes.
