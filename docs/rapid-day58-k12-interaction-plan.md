# Day 58: K12 interaction, partner localization, upstream trace, and acquisition

## Objective

Resolve the Day 57 install/remove asymmetry without weakening the failed confirmation
gate.  The working question is whether monitoring-prefix content is a conditional
component whose downstream effect requires another part of the exact K12 state.

## Evidence discipline

- The 176-example Day 57 confirmation panel is now an opened development panel.
- The 44-example Day 57 pathway panel remains sealed until one mechanism is selected
  and frozen for confirmation.
- Existing four-state outputs are reduced before any new model run: identity target,
  prefix-only installation, non-prefix-only removal state, and exact donor K12.
- Candidate localization is performed only on the opened 176 examples.
- No failed Day 57 threshold is relaxed or reinterpreted as a pass.

## Fast sequence

1. Compute the exact downstream four-state interaction
   `E - P - C + T`, including aligned, orthogonal, concept, example, and probe-row
   summaries.  This uses saved tensors and no GPU.
2. On the opened panel, intervene on the smallest frozen partner set: prompt-context
   content, response-history content, routing Shapley delta, full content Shapley
   delta, and the exact residual complement.  Measure each alone and paired with the
   monitoring-prefix delta.  Select by incremental interaction closure, not raw
   singleton strength.
3. Trace V, QK, and QKV only for the selected joint content object.  Report the
   complete matrix even if no pathway qualifies.
4. Freeze the selected joint intervention and pathway, then run it once on the sealed
   44-example panel in the chameleon and exact precursor.
5. Recover and hash every shard, reduce twice, terminate only the dedicated instance,
   and record the cost and bounded conclusion.

## Decision logic

- A partner must improve both directional endpoint recovery and reduce the nonlinear
  four-state residual relative to prefix alone.  Haar controls must not do so.
- If no atomic partner closes the interaction, the conclusion is a distributed
  complement rather than a post-hoc combination search.
- Upstream V/QK language applies to the selected joint object, not to the incomplete
  prefix singleton.
- Acquisition is claimed only if the selected chameleon mechanism confirms on the
  sealed panel and its corresponding absolute effect in the exact precursor is at
  most half as large in both directions.

The authorized incremental cloud budget is below $100.  Expected wall time after
implementation is roughly 30--90 minutes on one 80 GB GPU, plus model transfer if a
new instance has no cache.
