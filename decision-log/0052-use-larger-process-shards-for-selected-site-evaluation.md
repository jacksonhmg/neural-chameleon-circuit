# Decision 0052: Use Larger Process Shards for Selected-Site Evaluation

**Date:** 2026-08-10
**Status:** Replay passed; superseded after bounded allocator failure
**Scope:** Phase B attention evaluation before evaluation-effect access

## Context

V7 completed and exactly replayed the 65,536-row attention discovery matrix. Frozen selection reduced evaluation from 16 sites (64 jobs per direction) to one grouped site (four jobs per direction). Two four-batch evaluation shards completed; only counts were inspected.

## Decision

Preserve V7 discovery, the complete frontier matrix, and discovery-only selection byte-for-byte. Regenerate attention evaluation from zero under V8 using 64 ordered data batches per child process. Keep batch size two, original kernels, job chunk 32, metadata block 32, bounded MPS ratio 1.74, selected site, operations, directions, paths, population, and row order unchanged.

Before evaluation row 257, require exact dictionary replay of the 256 preserved V7 evaluation rows after removing only the two runtime-provenance fields from both sides.

## Scientific invariance

The live intervention tail falls from 64 jobs in discovery to four in evaluation. Process boundaries remain between complete batches. No operation, arithmetic, model/data batch shape, result row, selection, estimand, or gate changes.

## Disposition

V8 passed both real-checkpoint gates and exactly replayed the 256-row V7 prefix. Its first child completed 57 of 64 batches before reaching the bounded allocator cap. Decision 0053 retains the same execution and scientific contract but reduces each evaluation process lifetime to 32 batches.
