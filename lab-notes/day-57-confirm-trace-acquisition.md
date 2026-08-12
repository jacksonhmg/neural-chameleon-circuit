# Day 57: confirmation, value-path tracing, and precursor acquisition

**Status:** Frozen before outcomes; execution pending.

The deterministic freeze produced 176 fresh-confirmation examples and 44 separate tracing examples across 11 compatible concepts. All 220 records are mutually unique and content-disjoint from Day 4 and Day 47. Deception and harmful cannot be fresh within the pinned released train pool because Day 4 exhausted those pools.

See the [plan](../docs/rapid-day57-confirm-trace-acquisition-plan.md) and [contract](../results/day-57/frozen-confirm-trace-acquisition-contract.json). Record implementation repairs and outcomes here without changing the prospective contract.

## Outcome-blind implementation repair 1

The first confirmation preflight at commit `116b13a` generated no candidate outcomes and failed only the natural attention-recomputation check. Exact source construction, vectorized identity, algebra reconstruction, Shapley closure, response tensors, and hook cleanup passed. Inspection of the pinned Transformers eager-attention implementation showed that the experimental recomputation incorrectly performed all attention math in float32; the model performs QK and V products in bfloat16, upcasts only softmax, then casts probabilities back. The failed report is preserved as `results/day-57/invalid-preflight-attention-dtype.json`. The repair mirrors the pinned implementation without changing any candidate, condition, example, seed, threshold, gate, or reduction.

## Outcome-blind implementation repair 2

The first full confirmation invocation produced 15 of 44 compact shards without printing or reducing scientific metrics, then exhausted 80 GB on the longest chemistry batch because eight jobs times four examples were expanded into one forward. Those 30 shard files were recovered locally under the ignored invalid-attempt directory `artifacts/rapid-k12-upstream-v1/day57-invalid-pre-chunk-confirmation-shards/` before replacement and have not been inspected scientifically. The runner now executes the exact frozen job order in two-job memory chunks and concatenates results before serialization. The example batch size, intervention states, arithmetic, model, metrics, controls, gates, and reduction remain unchanged; every valid shard will be regenerated under the repair commit.
