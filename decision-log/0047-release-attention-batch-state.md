# Decision 0047: Release Attention Batch State Explicitly

**Date:** 2026-08-09
**Status:** Accepted prospective execution correction
**Scope:** Phase B attention execution before effect-value or selection access

## Context

Decision 0046 preserved the original 32-job live kernel and reduced block-local auxiliary state. Its preflight passed both frozen kernel hashes exactly. Population execution still stopped before the replay gate, at 6,912 rows, under both the default MPS cap and a narrowly bounded 1.705 ratio.

The allocation pattern showed that raising the cap allowed retained allocations to grow to the new cap. Inspection identified Python loop variables that still referenced the prior batch's realized states, attention captures, partitions, masks, source/target aliases, replacements, and jobs when the next batch began. Calling `empty_cache()` cannot release live tensors.

## Decision

Retain Decision 0046's 32-job kernel and 32-entry metadata blocks. At the end of every two-example batch, explicitly delete every batch-local tensor/reference before garbage collection and MPS cache release. Return to the default MPS high-water ratio; do not disable or raise the limit.

Regenerate attention from zero under a new runtime commit. Repeat the complete two-checkpoint exact preflight. Before row 7,169, require the same zero-tolerance dictionary replay against the original 7,168-row reference.

## Scientific invariance

Reference deletion occurs only after all outputs for a batch have been serialized. It cannot change any live kernel input, intervention, output, row order, population, selection, estimand, uncertainty calculation, or gate. It changes only when dead batch-local tensors become eligible for release.
