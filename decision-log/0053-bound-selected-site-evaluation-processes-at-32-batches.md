# Decision 0053: Bound Selected-Site Evaluation Processes at 32 Batches

**Date:** 2026-08-10
**Status:** Accepted prospective evaluation correction
**Scope:** Phase B attention evaluation after V8 passed replay but exceeded its process-lifetime budget

## Context

V8 passed both exact checkpoint preflights and the mandatory 256-row evaluation replay. Its first 64-batch child then completed 57 batches (1,824 rows) before the original eager-softmax allocator reached the bounded 1.74 cap. No effect value or candidate score was accessed.

## Decision

Preserve the complete V8 attempt verbatim and exclude it from analysis. Regenerate evaluation from zero under V9 using 32 ordered data batches per child process. This is below the empirically observed 57-batch failure point and leaves 25 batches of process-lifetime margin.

Before evaluation row 1,825, require exact dictionary replay of every preserved V8 evaluation row after removing only the two runtime-provenance fields from both sides.

## Scientific invariance

Process boundaries remain between complete data batches. V9 changes no model operation, arithmetic, tensor shape, intervention, path, selected site, population, row order, estimand, or scientific gate. V7 discovery, complete frontier evaluation, and discovery-only selection remain byte-identical.
