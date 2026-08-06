# Day 10: Test Sufficiency

## Phase information

- **Sprint phase:** Day 10
- **Status:** In progress
- **Calendar dates:** 2026-08-05 to present
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen component order:** [Day 8 selection](../results/day-08/frozen-component-selection.json)
- **Grouped necessity handoff:** [Day 9 results](../results/day-09/README.md)

## Objective

Determine whether naturally occurring correctly triggered component activations can actively create monitor evasion in otherwise normal runs, compare them with frozen random controls, test whether upstream residual context adds sufficiency, and measure a preregistered activation-interpolation dose response without opening safety.

## Session log

### 2026-08-05 — Session 1

#### Planned focus

- Freeze exact natural-transplant groups, the surrounding-residual boundary, metrics, controls, behavior scope, dose subset, alpha grid, evidence rules, numerical checks, and safety isolation.
- Implement and test generalized multi-site interpolation and sufficiency analysis.
- Run all-benign exact transplants, the frozen dose subset, and the bounded full-model behavior grid.
- Produce sufficiency and dose-response figures, machine-readable summaries, and an independent audit.

#### Work performed

- Work in progress.

#### Results and evidence

- Work in progress.

#### Failures and unexpected observations

- None yet.

## Open questions and blockers

- Whether selected K16 can directly induce evasion in normal runs remains pending.
- Whether the induction exceeds frozen random controls and transfers to validation remains pending.
- Whether `resid_post[8]` adds sufficiency beyond K16 remains pending.
- Whether activation strength yields a monotonic dose response remains pending.

## Decisions

- The complete Day 10 procedure and surrounding-residual interpretation are frozen before results in [Decision 0013](../decision-log/0013-freeze-day-10-sufficiency-procedure.md).

## Handoff

- **Current state:** Day 10 procedure is frozen; no sufficiency result has been generated and safety remains locked.
- **Next action:** Implement, test, and execute the natural-transplant and dose-response grids.
- **Verification still needed:** Identity and endpoint checks, group equivalence, norm bounds, complete row coverage, summary regeneration, control contrasts, behavior NLL, figures, artifact hashes, and safety isolation.
