# Decision 0085: Audit Only Valid Response Positions

**Date:** 2026-08-11
**Status:** Frozen implementation repair before localization outcomes

## Observation

The saved-baseline replay preflight generated no localization outcome. Its exact Day 54 state reproduced at zero error. The Day 52 replay differed by `0.0234375` across the entire padded response rectangle in one direction, just above the unchanged `0.02` gate, while the other direction was `0.013671875`.

The intervention framework intentionally patches only positions selected by `response_mask`. The first HTML example has 12 valid response tokens in a batch padded to 26. Consequently, an all-position comparison asks the replay to reproduce saved Day 52 values at 14 positions that were deliberately not patched and are excluded from every scientific estimand.

## Decision

Apply the unchanged maximum-absolute reproduction threshold only to valid response positions, exactly matching the causal intervention support and all downstream metrics. Preserve the all-position maximum as an adverse diagnostic. Do not change the tensor replay, candidate set, endpoint, control, threshold, or interpretation.

## Boundary

Quarantine the failed report, commit this audit correction, and rerun the full candidate-blind preflight. No localization outcomes have been generated or inspected.
