# Decision 0050: Restore Original Kernels Under the Bounded MPS Cap

**Date:** 2026-08-09
**Status:** Accepted prospective execution correction
**Scope:** Phase B attention execution after failed V5 exact replay

## Context

V5 survived through the sealed 7,168-row boundary, but the mandatory dictionary-equality gate failed on 64 total-path rows from one long batch. Only mismatch locations and key names were inspected. A retrospective machine comparison also found 64 mismatched V4 rows, while V3's 6,912 original-kernel rows match the preserved reference exactly.

This rejects both in-place arithmetic schedules. It does not license a tolerance or a changed gate.

## Decision

Restore Transformers' original Gemma2 MLP forward and its original full eager-attention softmax. Retain the V3 deletion of batch-local references and the exact 32-job live-tail and 32-entry metadata block. Run with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.74`, the bounded cap already used in the initial diagnosis; the unbounded value remains prohibited.

Require both frozen checkpoint result hashes, zero cached/full error, confirmation that neither replacement kernel is installed, and exact replay of all 7,168 reference rows before the first new row may be attempted.

## Scientific invariance

This restores the original model arithmetic rather than approximating it. The cap changes only when the allocator refuses a request. Model weights, tensors, batch shapes, operations, paths, row order, selection, estimands, and gates remain fixed.
