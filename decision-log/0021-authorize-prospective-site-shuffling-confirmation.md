# Decision 0021: Authorize the Prospective Site-Shuffling Confirmation

- **Date:** 2026-08-07
- **Status:** Accepted

## Decision

Authorize the Day 21 confirmation exactly as materialized in [`results/day-21/confirmation-authorization.json`](../results/day-21/confirmation-authorization.json), before generating any remaining-safety site-shuffling outcome.

Evaluate the four mappings selected solely from benign development data: `within_15015`, `within_15004`, `cross_15130`, and `cross_15122`. Do not replace or rerank them after safety inspection. For each map, compare the selected source population with a one-to-one layer-matched null source population while holding the selected destinations and source-layer route fixed.

Run all 50 conditions for each of 130 remaining positive safety examples (46 deception and 84 harmfulness), yielding exactly 6,500 rows. Treat destination-matched normal-versus-triggered absolute-source contrasts and destination-relative trigger deltas as the primary estimands. Treat raw absolute effects and same-condition mismatch as secondary diagnostics.

Apply the frozen 10,000-replicate example bootstrap and mechanical gate without reinterpretive changes:

- `portable_support` requires a positive lower 95% bound for the selected-source population and selected-minus-null contrast in every concept, direction, and primary estimand.
- `qualified_support` requires every selected-source primary point estimate to be positive when the full support gate is not met.
- `generic_disruption` requires positive absolute point estimates but no positive corrected-primary lower bound when the first two gates fail.
- Every other pattern is `rejected`.

Reuse the already completed and now hash-pinned Day 20 negative specificity controls. Day 20 behavioral and manifold diagnostics may qualify the prose interpretation but may not replace or alter the causal gate.

## Rationale

Days 17–20 identified promising within-layer routes, especially the reciprocal layer-11 pair, but those analyses used the already opened 32-example safety pilot or benign development data. Freezing the exact maps, comparator construction, remaining example IDs, condition grid, and interpretation rule before outcome generation creates a prospective test of the new intervention while preserving the distinction between mechanism development and confirmation.

Route-matched null sources are necessary because Day 17 showed that harmfulness destination receptivity is not exclusive to the selected population. Conditional contrasts and destination-relative deltas remain primary because Day 16 showed that absolute source replacement can be dominated by ordinary source/destination mismatch.

## Consequences

- No Day 21 map, source comparator, example, estimand, interval, or gate may be changed after this commit in response to the outcome.
- The confirmation is prospective only for the new site-shuffling intervention; it is not a fully untouched safety dataset because ordinary Day 13 circuit effects are already known.
- A positive result cannot erase Day 20's weak harmfulness specificity, cross-probe effects, token-order invariance, or activation-distance qualifications.
- A failed or qualified result must be reported as such; the strong benign and pilot results cannot substitute for the frozen confirmation gate.
- All work remains local. No push, tag, release, draft transmission, author contact, or external message is authorized.
