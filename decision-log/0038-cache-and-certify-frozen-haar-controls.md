# Decision 0038: Cache and Certify Frozen Haar Controls

**Date:** 2026-08-09  
**Status:** Accepted implementation-performance fix  
**Scope:** Phase B random controls before any Phase B effect row

## Context

The first complete normal-donor sweep exposed a performance defect in the not-yet-run random stage. The implementation regenerated the same 384 deterministic 256×256 Haar matrices and recomputed large trajectory Gram/SVD checks for every batch. The matrices depend only on frozen draw/head seeds, never on data.

No zero, replacement, random, frontier, attention, precursor-control, or other Phase B effect row had been produced.

## Decision

Cache each frozen Haar matrix by width and seed. Certify its norm, temporal-Gram, and singular-value invariants uniformly for every input trajectory using the float64 orthogonality residual plus a matrix-product rounding bound. Continue to construct every transformed tensor in float64 before float32 storage/BF16 installation, and retain the real-checkpoint preflight plus synthetic actual-tensor tests.

Preserve [mean-capture attempt 1](../results/day-39/mean-capture-attempt-1.json) and retain its local tables with attempt suffixes. Commit the change, rerun the two-checkpoint preflight, and regenerate checkpoint donor tables under the new execution identity before any effect.

## Scientific invariance

Seeds, QR sign convention, matrices, rotated deltas, thresholds, populations, and endpoints do not change. The new audit is stronger computationally: one certified bound applies to every possible natural trajectory transformed by that matrix.
