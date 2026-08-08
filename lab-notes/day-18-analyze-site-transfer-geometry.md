# Day 18: Analyze Site-Transfer Geometry

## Phase information

- **Sprint phase:** Day 18
- **Status:** In progress
- **Calendar date:** 2026-08-07
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Determine whether portable effects are better explained in raw head coordinates or after each head's output projection into the shared residual stream.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Capture benign trigger deltas for 12 selected and 12 matched-null heads.
- Compute exact least-squares output-projection alignment maps.
- Compare raw, per-example RMS-matched, and projection-aligned transport for all selected-head pairs.
- Quantify cosine, CKA, low-rank structure, and prediction of held-out safety-atlas outcomes.

#### Work performed

- Implemented a resumable benign geometry evaluator using the verified cached-tail runner.

#### Results and evidence

- Pending execution.

#### Failures and unexpected observations

- None so far.

## Decisions

- No new durable decision; coordinate tests were frozen in Day 15.

## Handoff

- **Current state:** Evaluator implementation in progress.
- **Next action:** Implement analyzer, test, commit, and run after Day 17 completes.
- **Verification still needed:** Transform preflight, raw grid, geometry archive, analysis, figures, and audit.
