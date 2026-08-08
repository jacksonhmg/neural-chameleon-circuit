# Day 17: Source–Destination Transfer Atlas

Day 17 evaluates the complete ordered 24 × 24 selected/null head factorial on the frozen 32-example safety pilot.

Every pair receives destination-matched normal and triggered absolute-source tests in both base conditions. Every selected-to-selected pair additionally receives raw and RMS-matched destination-relative delta tests. Individual cells are exploratory; population contrasts and multiplicity-corrected cell summaries are the intended outputs.

## Outcome

The atlas is complete: 92,224 unique raw intervention rows, 8,064 cell-level estimands, and 84 population estimands. The audit passes all row-count, uniqueness, finite-population, and figure checks. Population intervals use 10,000 example-cluster bootstrap replicates; cell tests use 10,000 sign flips and Benjamini–Hochberg correction across the declared families.

The result is a sparse, route-sensitive transfer structure, not generic interchangeability among the selected heads:

- For deception, nonidentity selected-to-selected transport is positive in rescue and induction under the conditional, raw-delta, and RMS-matched-delta estimands. The conditional means are 0.00233 (95% CI 0.00135–0.00339) for rescue and 0.00680 (0.00566–0.00804) for induction. All three selected-to-selected conditional contrasts against selected/null role controls are positive.
- For harmfulness, selected-to-selected raw- and RMS-delta effects are positive, but destination receptivity is not exclusive: the conditional selected-to-null population is larger than selected-to-selected in both directions. This rejects a simple claim that only the selected destinations can receive the transported harmfulness signal.
- Within-layer routes dominate nonidentity transfer. The reciprocal layer-11 pair `head_08 ↔ head_09` is the clearest portable route: its raw-delta estimates range from 0.0537 to 0.3092 across the two concepts and two directions.
- Harmfulness later-to-earlier routes are adverse or mixed, including conditional rescue −0.01002 (−0.01289 to −0.00700). Depth order therefore matters.

The appropriate Day 17 conclusion is that selected source activations contain some portable trigger-linked information, but successful transfer depends strongly on the exact route and concept. The atlas does not support a homogeneous shared code or a uniformly privileged selected destination population.

## Files

- `transfer-atlas-results.jsonl.gz`: sealed raw intervention rows.
- `transfer-atlas-cells.csv`: cell-level estimates, intervals, sign-flip p-values, and BH q-values.
- `transfer-atlas-populations.csv`: role-factorial, route-class, and selected/null contrasts.
- `transfer-atlas.png` / `.pdf`: source–destination heatmaps.
- `transfer-populations.png` / `.pdf`: population summaries.
- `transfer-atlas-preflight.json`: committed-runner and cached-tail equivalence checks.
- `transfer-atlas-summary.json`: analysis counts and settings.
- `transfer-atlas-audit.json`: mechanical completeness audit.

The uncompressed resumable working file was deleted after the sealed gzip archive and audit were verified; it was an execution artifact, not an additional result.
