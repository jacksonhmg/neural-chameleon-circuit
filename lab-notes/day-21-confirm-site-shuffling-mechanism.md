# Day 21: Confirm the Site-Shuffling Mechanism

## Phase information

- **Sprint phase:** Day 21
- **Status:** In progress
- **Calendar date:** 2026-08-07
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Apply the prospectively frozen causal gate to four benign-selected mappings on every remaining positive safety example, with route-matched null sources and no safety-specific mapping selection.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Commit the evaluator and then materialize and commit the authorization before any confirmation outcome.
- Pin all 130 example IDs, four selected maps, four route-matched null maps, and the exact 6,500-row grid.
- Run conditional and destination-relative delta rescue/induction, apply the fixed bootstrap gate, and reuse the pre-confirmation Day 20 negative controls by hash.
- Audit raw completeness, provenance, hooks, tests, figures, and documentation.

#### Work performed

- Implemented the authorization generator, resumable confirmation evaluator, independent gate analyzer, and unit checks.
- Committed the implementation at `67becda` before authorization.
- Materialized the authorization while all Day 21 outcome artifacts were absent. It pins 46 deception and 84 harmfulness examples, four selected and four route-matched null maps, 50 conditions per example, 6,500 expected rows, the bootstrap, and the mechanical gate.
- Committed the authorization at `78df5eb`, then ran the no-outcome model preflight. All six authorization, map, ID, grid, and hook checks passed; the preflight explicitly generated no confirmation outcome.

#### Results and evidence

- Authorization and preflight pass are complete; outcome execution has not begun.

#### Failures and unexpected observations

- The first confirmation attempt stopped before running any intervention because the runner's defensive assertion expected 48 jobs inside each base condition. The authorized grid specifies 48 interventions across two bases, so the correct per-base count is 24. Four baseline rows for the first two examples had already been written; no shuffled outcome was generated. The assertion was corrected without changing a map, source, destination, example, condition, estimand, or gate, and the resumable archive will retain those baseline keys.

## Decisions

- [Decision 0021](../decision-log/0021-authorize-prospective-site-shuffling-confirmation.md) records the exact prospective authorization and prohibits safety-specific changes.

## Handoff

- **Current state:** Authorization committed and preflight passed; no remaining-safety site-shuffling outcome has been generated.
- **Next action:** Commit the preflight evidence, then run the exact 6,500-row confirmation.
- **Verification still needed:** Raw grid, gate, negative-control reference, figure inspection, and final audit.
