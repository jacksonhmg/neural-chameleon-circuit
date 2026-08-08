# Decision 0023: Freeze the Site-Shuffling Behavioral Extension

- **Date:** 2026-08-08
- **Status:** Accepted

## Decision

Run a sealed Days 22–25 extension to determine whether the four final non-original site mappings transport a downstream output-distribution effect, preserve behavior while changing the monitor, or cause nonspecific disruption.

Keep the Day 21 mapping population unchanged, including `cross_15122`, and compare every selected-source map with its already-frozen route-matched null-source map. Use only destination-relative raw deltas at alpha 1.0 in both induction and rescue. Include same-site selected-head K12 as a benchmark without adding it to the mapping population.

Freeze all example IDs, behavioral endpoints, equivalence bounds, bootstrap settings, generation semantics, and mechanical interpretation before generating final-map behavioral outcomes. The complete method is [`docs/day-22-25-behavioral-transport-method.md`](../docs/day-22-25-behavioral-transport-method.md); the machine-readable authorization will be stored under `results/day-22/`.

## Rationale

Day 21 prospectively establishes that selected trigger-linked activation deltas affect the layer-12 probe through non-original routes. Day 20 shows low full-model KL for two earlier pilot mappings, but it does not test the four final benign-selected mappings. The missing evidence is therefore whether the confirmed intervention remains output-stable, produces an output shift aligned with the natural trigger, or merely disrupts downstream computation.

A simple changed-versus-unchanged output test would be misleading. Behavior-preserving probe suppression is the defining Neural Chameleon capability and can support a selective evasion mechanism. Conversely, a large but unaligned output change would weaken rather than strengthen the mechanistic interpretation. Directional logit projection, route-matched nulls, equivalence tests, and a secondary coupled-generation diagnostic separate these cases.

## Consequences

- The extension is prospective for unopened behavioral endpoints but is not a new dataset confirmation.
- No mapping, example, threshold, endpoint, or generation rule may change after authorization in response to results.
- Population and mapping-level results must both be reported; the known reversed cross-layer mapping cannot be hidden.
- Free-generation judgments remain secondary and local. No external evaluator is authorized.
- All work remains local. No push, tag, release, draft transmission, author contact, or external message is authorized.

