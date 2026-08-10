# Decision 0051: Isolate Original-Kernel Attention Processes

**Date:** 2026-08-09
**Status:** Completed for discovery; superseded for evaluation scheduling
**Scope:** Phase B attention execution after V6 passed replay but exhausted allocator state

## Context

V6 restored the original Gemma2 kernels and regenerated all 7,168 sealed rows with exact dictionary equality. It then exhausted the bounded allocator before producing any new row. A fresh-process diagnostic on the exact long HTML batch completed all 512 rows and matched the corresponding reference bytes exactly.

The remaining failure is accumulated allocator state, not arithmetic or the scientific operator.

## Decision

Retain the original MLP, original full eager softmax, batch size two, 32-job tail, 32-entry metadata block, and bounded MPS ratio 1.74. Execute at most four ordered data batches per child process, then exit the process and continue in a fresh child. The driver must preserve model order, population order, row order, runtime identity, resumability, and the existing exact replay gate.

The base runner refuses unsharded attention execution under V7. Both the driver and runner must be committed in the V7 runtime commit.

## Scientific invariance

Process boundaries occur only between complete data batches. No tensor, model operation, random draw, intervention, path, batch shape, result row, selection rule, estimand, or gate crosses a boundary. The exact long-batch diagnostic and the full 7,168-row replay remain mandatory empirical checks.

## Disposition

V7 completed all 65,536 discovery rows and passed exact replay. Decision 0052 superseded only the evaluation process lifetime after selection reduced the live intervention tail from 64 jobs to four.
