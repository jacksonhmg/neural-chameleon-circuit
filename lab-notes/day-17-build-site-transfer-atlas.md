# Day 17: Build the Site-Transfer Atlas

## Phase information

- **Sprint phase:** Day 17
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Separate source information, destination receptivity, route depth, and generic mismatch with a complete selected/null head transfer atlas.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Implement all 576 ordered source/destination pairs.
- Evaluate route-matched absolute contrasts for the full selected/null factorial.
- Evaluate raw and RMS-matched deltas for all 144 selected-to-selected pairs.
- Pass benign preflight, seal the 92,224-row grid, and apply multiplicity correction.

#### Work performed

- Implemented a resumable atlas evaluator with explicit source/destination roles and route classes.
- Added and independently tested a cached-tail intervention runner. It replays the frozen pre-intervention residual input at the destination layer, bypasses earlier blocks, and is exactly equivalent to the original full-forward intervention path at layers 9–12 (maximum observed score difference: 0.0).
- Passed the benign preflight and evaluated all 576 ordered selected/null pairs on both safety-pilot concepts.
- Evaluated destination-matched absolute contrasts for every pair and raw plus per-example RMS-matched destination-relative deltas for all 144 selected-to-selected pairs.
- Sealed the 92,224-row raw archive and ran the separate 10,000-replicate bootstrap/sign-flip analyzer.
- Generated and visually inspected the cell-atlas and population figures. The atlas layout was revised once to prevent its shared colorbar from overlapping the panels.

#### Results and evidence

- The mechanical audit passes: 92,224 unique raw rows, 8,064 cell metrics, 84 population metrics, finite population outputs, and all four figure artifacts present.
- Deception nonidentity selected-to-selected conditional transport is positive for rescue (0.00233, 95% CI 0.00135–0.00339) and induction (0.00680, 0.00566–0.00804). Raw and RMS-matched delta populations agree in sign.
- Harmfulness selected-to-selected raw delta transport is positive for rescue (0.00542, 0.00355–0.00675) and induction (0.00543, 0.00441–0.00656), but the conditional selected-to-null population is larger. The harmfulness result therefore supports informative selected sources without supporting exclusive selected destinations.
- Within-layer transfer is strongest. The reciprocal `layer_11.head_08`/`layer_11.head_09` routes are the largest recurring nonidentity cells, with raw-delta estimates of 0.0985/0.0537 for deception rescue, 0.3092/0.2688 for deception induction, 0.1364/0.1659 for harmfulness rescue, and 0.0879/0.0920 for harmfulness induction.
- Harmfulness later-to-earlier transport is adverse or mixed; conditional rescue is −0.01002 (95% CI −0.01289 to −0.00700).
- Evidence: [result package](../results/day-17/README.md), [audit](../results/day-17/transfer-atlas-audit.json), [cell table](../results/day-17/transfer-atlas-cells.csv), and [population table](../results/day-17/transfer-atlas-populations.csv).

#### Failures and unexpected observations

- The first full-forward evaluator was stopped after a partial run because extrapolated runtime was impractical. No partial values were analyzed. The cached-tail replacement was implemented, committed, and required to pass exact equivalence checks before the study resumed.
- The first cached-tail run, using a group chunk size of 16, stalled after 17,488 rows under MPS memory pressure. The exact resumable row keys were preserved; execution resumed with a chunk size of 4 and completed without changing the frozen estimands or data.
- Harmfulness selected-to-null effects were unexpectedly larger than selected-to-selected effects. This is scientifically inconvenient but important: the destination population is less specific than the original receiver hypothesis predicted.

## Decisions

- No new durable decision; the scientific design remains frozen under [Decision 0020](../decision-log/0020-freeze-site-shuffling-follow-up.md). The cached-tail path was an equivalence-tested execution optimization, not a methodological change.

## Handoff

- **Current state:** Day 17 complete; complete atlas and audited inference package sealed.
- **Next action:** Run Day 18 benign representation-geometry and transport-alignment analyses.
- **Verification still needed:** None for the Day 17 completion gate.
