# Decision 0094: Accept a Distributed Residual Source Controller

**Date:** 2026-08-12
**Status:** Applied frozen Day 61 dispositions

## Decision

Accept `residual_state_only`. Exact donor residual state at the informative source
positions qualifies at every boundary from `resid_pre.layer_00` through
`resid_pre.layer_09`, while no whole normalized attention or MLP branch qualifies.
Apply the frozen earliest-boundary rule and record `resid_pre.layer_00` as the
selected sufficient state.

Interpret the result as condition identity present in the input residual and carried
through a distributed pathway into the K12 K/V interface. Do not interpret it as
evasion being learned in the embeddings, as one residual direction constituting an
example-specific code, or as a single branch writing the controller.

Retain the parallel-pathway caveat: residual transplants reconstruct Day 60 K/V and
K12 near exactly but overshoot its monitor endpoint, so the full residual state also
acts through non-KV routes.

## Consequences

- Reject every tested whole attention and MLP branch as a compact bidirectional
  controller under the frozen gates.
- Treat pre-layer-9 as the last boundary that can causally control all selected K12
  layers, not as proof that the state disappears at pre-layer-10.
- Do not spend the next experiment on finer branch or neuron searches across the
  whole prefix stack.
- Localize the acquired mapping with Chameleon/precursor checkpoint hybridization,
  scoring pre-layer-9 residual, source K/V, K12, and monitor endpoints.
