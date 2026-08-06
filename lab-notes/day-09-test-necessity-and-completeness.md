# Day 9: Test Grouped Necessity and Completeness

## Phase information

- **Sprint phase:** Day 9
- **Status:** In progress
- **Calendar dates:** 2026-08-05 to present
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen component order:** [Day 8 selection](../results/day-08/frozen-component-selection.json)

## Objective

Directly patch increasingly large nested component groups, compare them with frozen matched controls, quantify positive completeness and negative off-target movement, and determine whether the individually localized mechanism appears compact, moderately distributed, or highly distributed without using additive assumptions or safety data.

## Session log

### 2026-08-05 — Session 1

#### Planned focus

- Freeze grouped sites, prefixes, controls, metrics, behavior scope, numerical checks, and the compactness rule.
- Implement and test vectorized grouped activation patching.
- Run the complete all-benign probe-score and frozen-subset behavior grids.
- Produce completeness curves, controls, machine-readable summaries, and an independent audit.

#### Work performed

- Work in progress.

#### Results and evidence

- Work in progress.

#### Failures and unexpected observations

- None yet.

## Open questions and blockers

- Whether recovery saturates with a small prefix remains pending.
- Absolute recovery of K=16 and the all-layer-11 group remains pending.
- Negative-example and behavior effects of grouped patches remain pending.

## Decisions

- The complete procedure and distribution-classification rule are frozen before grouped results in [Decision 0012](../decision-log/0012-freeze-day-09-grouped-necessity-procedure.md).

## Handoff

- **Current state:** Day 9 procedure is frozen; grouped model results have not been generated and safety remains locked.
- **Next action:** Implement, test, and execute the grouped intervention grid.
- **Verification still needed:** Group equivalence and identity controls, full row coverage, paired summary regeneration, control comparisons, behavior NLL, figures, artifact hashes, and safety isolation.
