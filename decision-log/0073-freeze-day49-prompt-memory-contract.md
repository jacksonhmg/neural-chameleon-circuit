# Decision 0073: Freeze the Day 49 Prompt-Memory Contract

**Date:** 2026-08-11
**Status:** Frozen before Day 49 prompt-memory outcomes

## Decision

Freeze [`results/day-49/frozen-prompt-memory-contract.json`](../results/day-49/frozen-prompt-memory-contract.json) as the authoritative contract for the only branch eligible after Day 48.

Test nine candidates in fixed simplicity order: response-query `Q` alone, `KV` operations for generic prefix, named concept, prior response, and the full monitoring prefix, followed by the corresponding combined `QKV` operations. Apply each operation across only the exact selected K12 heads and permit ordinary total-path propagation to the monitor.

## Gate

Every candidate must pass bidirectional median concept K12 recovery of at least `0.50`, beat its strongest irrelevant or Haar control by at least `0.25` in both directions, preserve K12 and own-probe directions in at least 10 of 13 concepts, remain RMS bounded, and retain distributed support. Select at most the first passing candidate in the frozen simplicity order.

A pass permits a separately frozen concept-reconfiguration and K12-mediation closure for that unchanged winner. If no candidate passes, stop the compact-controller program and make Day 50 ineligible.

## Boundary

This remains a 26-example development sandbox. It does not reopen the failed response controller or Phase C named-span controller, and it authorizes no behavior, generation, fresh data, dissemination, or title upgrade.
