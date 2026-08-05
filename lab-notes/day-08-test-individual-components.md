# Day 8: Rank and Exactly Test Individual Components

## Phase information

- **Sprint phase:** Day 8
- **Status:** In progress
- **Calendar dates:** 2026-08-05 to present
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen layers:** 12, 11, 10, and 9

## Objective

Compare inexpensive attention-head and MLP screening methods with exact causal rescue, freeze a discovery-only ordered component set, and test its exact rescue, induction, held-out transfer, matched controls, and fixed-continuation NLL without reranking or opening safety.

## Session log

### 2026-08-05 — Session 1

#### Planned focus

- Freeze the remaining screening, exact-test, control, K-selection, validation, and behavior details.
- Implement tested vectorized independent-component patching and one-backward-pass screening.
- Scan all 68 eligible candidates on discovery positives.
- Commit the exact ordered component set and controls before validation.
- Run frozen held-out confirmation, analysis, figures, audit, and sprint closeout.

#### Work performed

- Work in progress.

#### Results and evidence

- Work in progress.

#### Failures and unexpected observations

- None yet.

## Open questions and blockers

- Which exact heads and MLPs pass the shared discovery gate remains pending.
- How well each inexpensive screen predicts exact causal recovery remains pending.

## Decisions

- The complete Day 8 procedure is frozen before scanning in [Decision 0010](../decision-log/0010-freeze-day-08-individual-component-procedure.md).

## Handoff

- **Current state:** Day 8 procedure and implementation are in progress; validation has not been used for individual-component selection and safety remains locked.
- **Next action:** Complete discovery screening and exact rescue, freeze the ordered set, then run held-out confirmation.
- **Verification still needed:** Real-checkpoint gradient support, exact identity/equivalence controls, complete discovery grid, committed selection hash, confirmation coverage, behavioral NLL, deterministic artifacts, figure QA, and independent audit.
