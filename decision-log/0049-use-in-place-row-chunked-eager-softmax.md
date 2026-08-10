# Decision 0049: Use an In-Place Row-Chunked Eager Softmax

**Date:** 2026-08-09
**Status:** Accepted prospective execution correction
**Scope:** Phase B attention execution before effect-value or selection access

## Context

Decision 0048 removed the large out-of-place MLP product and passed both exact checkpoint hashes. Population execution progressed further, then stopped while eager attention attempted to allocate a full-shape float32 softmax result. The score tensor and requested float32 result coexist even though the score tensor has no later consumer.

## Decision

Keep the query-key score matmul, scale, softcap, mask, softmax axis, float32 softmax, BF16 cast, dropout, and value matmul unchanged. Flatten only the independent outer batch/head axes. For 16 outer rows at a time, evaluate the original final expression `softmax(rows, dim=-1, dtype=float32).to(query.dtype)` and copy it back over those score rows. Return the overwritten score tensor as the attention pattern.

Install the memory-efficient eager-attention function with the 42 in-place MLP forwards. Run at the default MPS cap. Both checkpoint result hashes, cached/full equality, and the original 7,168-row exact replay must pass before scientific execution can cross the old boundary.

## Scientific invariance

Each softmax row normalizes only over its final key axis and is independent of all outer rows. The same fused softmax is called with the same row values, dimension, and float32 dtype, then cast to the same query dtype. No score matmul, model batch shape, intervention, attention operation, path, output order, population, selection, estimand, or gate changes. Exact hashes and replay remain mandatory.
