# Decision 0074: Freeze Unequal-Region Alignment Before Day 49 Outcomes

**Date:** 2026-08-11
**Status:** Frozen implementation repair before candidate outcomes

## Failure observed

The first complete Day 49 invocation stopped while constructing the first concept's jobs. No concept shard or candidate metric was produced. The correct-trigger target and frozen irrelevant-trigger source can assign unequal token counts to the same source-region definition, while the Day 49 contract specified aligned source-region K/V replacement without concretizing unequal-span alignment.

## Repair

Keep every frozen candidate, condition, source region, tensor coordinate, control, seed, gate, and stop rule unchanged. For a both-present source/target region, order its absolute token indices and map every target slot to the nearest source slot under endpoint-aligned relative position:

```text
source_index(j) = round(j * (source_count - 1) / (target_count - 1))
```

The one-target case uses the first source token; the one-source case repeats that source token. Equal token counts reduce exactly to one-to-one ordered replacement. This uses only actual source post-RoPE K/V states, fills every target-region position, adds no fitted parameter, and is independent of activations and outcomes.

## Provenance rule

Commit this repair and its unequal-span unit test before rerunning the implementation-only preflight or producing any Day 49 candidate outcome. The original frozen Day 49 contract remains unchanged.
