# Decision 0046: Preserve the Attention Kernel Shape

**Date:** 2026-08-09
**Status:** Accepted prospective execution correction
**Scope:** Phase B attention execution before effect-value or selection access

## Context

Decision 0045 prospectively required job chunk eight to equal the frozen 32-job kernel exactly. Its committed Chameleon preflight failed that gate before precursor preflight or population execution. The correction is rejected; its threshold will not be weakened.

The OOM occurs while the 32-job live tail coexists with all 64 site/operation replacement dictionaries and direct-path target recomputations for a long two-example batch. Those auxiliary tensors are not inputs to the live total-effect kernel.

## Decision

Retain job chunk 32 and the exact expanded live-tail shape. Partition the already ordered 64 metadata entries into two blocks of 32, execute each block with the unchanged 32-job kernel, and construct direct-path target recomputations only after that block's total-effect outputs have been reduced to probe summaries. Release block-local auxiliary state before constructing the next block.

The complete two-checkpoint preflight must reproduce the frozen old-kernel hashes and cached/full equality exactly. Corrected attention generation starts from zero. Before it may append row 7,169, its first 7,168 rows must equal the preserved failed-attempt rows as dictionaries after removing only the newly required runtime provenance fields.

## Scientific invariance

The expanded live-tail shape, job order within each frozen 32-job chunk, attention replacements, direct path, model inputs, batch size, output serialization order, populations, selection, estimands, and gates do not change. Only the lifetime of auxiliary tensors outside the live kernel is shortened.
