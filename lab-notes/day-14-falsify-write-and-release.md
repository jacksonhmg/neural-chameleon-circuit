# Day 14: Falsify, Consolidate, and Write

## Phase information

- **Sprint phase:** Day 14
- **Status:** In progress
- **Calendar dates:** 2026-08-06
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen falsification plan:** [Day 14 plan](../results/day-14/frozen-falsification-plan.json)
- **Claim ledger:** [Claim ledger](../report/claim-ledger.md)

## Objective

Attempt to disprove the qualified Day 13 explanation, independently audit the intervention machinery, reproduce the central safety result from a clean process, and consolidate the evidence into a six-figure final report. Do not publish, push, tag, release, email, or contact anyone.

## Session log

### 2026-08-06 — Session 1

#### Planned focus

- Freeze exact analysis-only attacks, machinery checks, causal nulls, seeds, subsets, thresholds, gates, clean-reproduction criteria, claim boundaries, and final-figure sources.
- Commit the freeze locally before running any new Day 14 analysis.
- Run every frozen attack even if early results weaken the preferred explanation.
- Preserve Day 13 raw results and write all new output under `results/day-14/` or `report/`.

#### Work performed

- Defined five claims under attack and labelled Day 14 as post-confirmatory falsification.
- Froze four bootstrap seeds; leave-one-out, worst-case deletion, trimming, median-of-means, threshold, leakage, pooling, and max-T multiplicity analyses.
- Froze an independent real-checkpoint index/hook audit.
- Froze a 32-positive-example causal subset, five seeded layer-count-matched head nulls, head/MLP and leave-one-layer decompositions, earlier/shifted sites, and two site-shuffled controls.
- Froze a detached-worktree clean reproduction at Day 13 commit `54ffc0a` and a six-figure report structure.
- Recorded the limitation that exact same-layer whole-MLP nulls do not exist.
- Explicitly prohibited push, tag, release, and external communication.
- The first analysis-only execution stopped before multiplicity results because the max-T implementation called a nonexistent NumPy array method (`values.square`). No plan or scientific rule changed; corrected it to `np.square(values)` and reran the complete suite rather than reusing a partial conclusion.

## Open questions and blockers

- None at freeze. Results have not yet been computed under the Day 14 plan.

## Decisions

- Freeze the exact Day 14 attacks before analysis under [Decision 0019](../decision-log/0019-freeze-day-14-falsification-plan.md).

## Handoff

- **Current state:** Day 14 falsification plan prepared for local commit; no new Day 14 result inspected.
- **Next action:** Commit locally, then run the analysis-only falsification suite.
