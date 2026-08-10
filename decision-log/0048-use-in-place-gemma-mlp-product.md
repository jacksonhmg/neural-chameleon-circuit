# Decision 0048: Use an In-Place Gemma MLP Product

**Date:** 2026-08-09
**Status:** Accepted prospective execution correction
**Scope:** Phase B attention execution before effect-value or selection access

## Context

Decision 0047 released all batch-local attention references before cache cleanup but the long two-example batch still exceeded the default MPS cap at 6,912 rows. This localizes the remaining failure to a single 32-job live-tail call. The traceback consistently stops in Gemma's MLP while allocating the out-of-place product of two already materialized 525 MiB intermediates.

## Decision

Keep both projection matmuls, their shapes, the activation, multiplication, and down projection unchanged. Store the elementwise product in the activated-gate tensor with `mul_` instead of allocating a third equally shaped product tensor. Install this forward method on exactly the 42 Gemma MLP modules at model load.

Run at the default MPS cap. The complete Chameleon and precursor preflight must reproduce the frozen old-kernel result hashes and cached/full outputs exactly. Regenerate attention from zero under a new runtime commit and require exact replay of the original 7,168 rows before crossing the old boundary.

## Scientific invariance

The in-place product uses the same two element values and multiplication at every coordinate. The overwritten activated-gate tensor has no later consumer. No matmul, batch shape, model parameter, intervention, attention operation, path, row, population, selection rule, estimand, or gate changes. Exact checkpoint hashes and exact population replay remain mandatory.
