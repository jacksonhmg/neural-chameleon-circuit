# Day 19: Characterize Shuffle Ensembles

## Phase information

- **Sprint phase:** Day 19
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen mappings:** [Mapping ensemble](../results/day-15/frozen-mapping-ensemble.json)

## Objective

Replace best-seed reporting with complete mapping distributions, benign-only mapping selection, and grouped composition tests.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Run all 64 mappings on four discovery and seven validation concepts.
- Select two within-layer and two cross-layer mappings using only the frozen discovery rule.
- Test K1/K2/K4/K8/K12 composition and compare against pairwise predictions.

#### Work performed

- Implemented the resumable ensemble evaluator with selected K12 destination-relative deltas.
- Passed the mapping preflight: 64 unique bijections, balanced 32/32 within/cross classes, no leaked hooks, and no safety access.
- Completed and sealed 22,880 unique rows across four discovery and seven held-out validation concepts.
- Applied the frozen benign-only rule and selected two eligible mappings from each class.
- Completed and sealed 7,392 K1/K2/K4/K8/K12 composition rows for the four selected mappings.
- Compared composition against the independently measured Day 18 raw-pair additive predictions.
- Generated and visually inspected the full-distribution and composition figures.

#### Results and evidence

- All within-layer mapping means are positive on development and validation. Their medians are 0.1295 and 0.1386, versus 0.0403 and 0.0208 for cross-layer mappings.
- Eligibility is common within a layer (25/32) and uncommon across layers (8/32). Validation means are positive for 32/32 within-layer and 30/32 cross-layer mappings.
- Selected mappings are `within_15015`, `within_15004`, `cross_15130`, and `cross_15122`. Selection used no safety outcomes.
- The selected within-layer mappings generalize strongly: K12 validation scores are 0.2986 and 0.2718 when averaged over concepts/directions. The selected cross-layer mappings fall to 0.0210 and 0.0174 and contain negative validation cells.
- Both within-layer selections exchange `layer_11.head_08` and `layer_11.head_09`. Adding that pair between K4 and K8 produces the dominant composition jump: development rescue rises from 0.0149–0.0190 to 0.4104–0.4109; induction rises from 0.0051–0.0064 to 0.2647–0.2654.
- At K12 the within-layer observations modestly exceed additive pair predictions, whereas cross-layer high-K composition is generally below the additive sum. The circuit therefore shows route-specific interaction and saturation rather than simple linear accumulation.
- Evidence: [result package](../results/day-19/README.md), [selected mappings](../results/day-19/benign-selected-mappings.json), [summary](../results/day-19/permutation-composition-summary.json), and [audit](../results/day-19/day19-audit.json).

#### Failures and unexpected observations

- Cross-layer development-positive mappings did not generalize uniformly: the selected mappings contain validation cells as low as −0.0462 and −0.1348.
- Composition is not monotonic at small K for every mapping/direction, and the large within-layer effect appears abruptly when the reciprocal layer-11 pair enters. This rejects a smooth distributed-ensemble account.

## Decisions

- No new durable decision; selection and confirmation authorization are frozen under Decision 0020.

## Handoff

- **Current state:** Day 19 complete; mappings selected without safety access and all distribution/composition outputs audited.
- **Next action:** Run Day 20 specificity, manifold-distance, and full-model behavioral diagnostics on the fixed mappings.
- **Verification still needed:** None for the Day 19 completion gate.
