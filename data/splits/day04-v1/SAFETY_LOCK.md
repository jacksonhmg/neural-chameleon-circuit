# Safety-Test Lock

**State:** Locked

**Locked file:** `safety-test.LOCKED.jsonl`

**Locked-file SHA-256:** `48d6818db1946919c1a49c373170f701219df472b669900d39cde834e0d595ec`

Until the component set and patching procedure are frozen and committed, the only allowed operations on the safety split are deterministic materialization and automated validation of its schema, counts, provenance, and checksums.

Do not run model forward passes, probe scoring, activation extraction, content-driven filtering, component selection, ranking, tie-breaking, choice of component count, or procedural revision against this split. Do not import Day 1 safety results into component-selection analysis.

## Unlock requirements

The safety loader requires a new `safety-unlock.json` containing all of these fields:

```json
{
  "schema_version": 1,
  "freeze_id": "day04-v1",
  "authorization": "confirmatory-safety-evaluation",
  "procedure_version": "day04-v1",
  "safety_split_sha256": "48d6818db1946919c1a49c373170f701219df472b669900d39cde834e0d595ec",
  "component_freeze_commit": "<40-character Git commit>",
  "component_set_sha256": "<64-character SHA-256>"
}
```

Before that authorization is committed:

1. The ordered component list and final `K` must have been selected entirely on discovery data.
2. The exact component sites, patch direction, and token positions must be committed.
3. Validation may have tested transfer, but it must not have changed the component identities, order, or `K`.

After unlock, run the frozen procedure once on deception and harmfulness, report the two concepts separately, and do not perform safety-specific reranking.

## Historical disclosure

Day 1 reproduced the released safety-probe suppression result before this freeze, as required by the project brief. The confirmatory test is therefore not globally blind in the strongest sense. The protection established here is prospective: no Day 4 safety scores have been computed, and the already-existing Day 1 values may not be consulted or used while components are selected.
