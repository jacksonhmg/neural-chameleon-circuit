# Decision 0065: Freeze Selected Precursor Parameter Swaps

**Date:** 2026-08-10
**Status:** Frozen before parameter-swap outcomes

## Decision

Use the total-path-sufficient K12 operator to causally localize acquisition with exact-precursor parameter slices. Run selected `O`, `Q`, grouped `K/V`, joint `QKV`, and complete `QKVO` swaps in that order, each with a same-layer, count- and GQA-matched non-K12 head control.

For every parameter state, measure both the state's endogenous natural K12 delta and the fixed Chameleon concept/position prototype. This separates loss of write production from loss of downstream write geometry. Keep layers 0–8 as the Chameleon computation and verify bit-identical layer-9 inputs across states.

The complete population, head sets, controls, causal estimands, classification thresholds, signature rules, and stop disposition are fixed in [`frozen-selected-parameter-swap-contract.json`](../results/day-46/frozen-selected-parameter-swap-contract.json).

## Rationale

The Day 46 result establishes that the prototype survives ordinary propagation. Parameter swaps are therefore the shortest route to distinguish acquired output geometry, acquired routing, QKV/O co-adaptation, and an upstream or more distributed implementation.

## Boundary

Weight differences are descriptive only. A parameter claim requires a selected precursor swap to cause a material loss, beat its matched control under the frozen margin, and pass the concept-bootstrap selectivity bound.
