# Day 7: Separate Attention and MLP Contributions

## Phase information

- **Sprint phase:** Day 7
- **Status:** In progress
- **Calendar dates:** 2026-08-04 to present
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [`day04-v1`](../data/splits/day04-v1/README.md)
- **Frozen layers:** `12`, `11`, `10`, `9` in Day 6 rank order

## Objective

Compare exact full-response patches of the complete attention branch, complete MLP branch, and complete block output at the four frozen layers. Test rescue and induction on every frozen benign example, then quantify held-out transfer, negative-example effects, irrelevant-trigger specificity, and deterministic random-layer controls without using safety data or changing the retained layers.

## Session log

### 2026-08-04 — Session 1

#### Planned focus

- Freeze the component-type sites, data scopes, control scopes, random layers, metrics, and interpretation rules before inspecting patch results.
- Implement and test a generic early-terminated component intervention runner.
- Run the complete selected-layer grid on all discovery and validation positives and negatives.
- Run irrelevant-trigger rescue controls on every positive benign example.
- Run random-layer rescue controls on discovery positives.
- Produce paired estimates, causal-effect figures, a machine-readable package, and an independent audit.

#### Work performed

- Work in progress.

#### Results and evidence

- Work in progress.

#### Failures and unexpected observations

- None yet.

## Open questions and blockers

- Whether the late-layer effect is primarily attention-mediated, MLP-mediated, divided, or dependent on their combined block state remains pending.

## Decisions

- The exact Day 7 procedure is frozen before the scan in [Decision 0009](../decision-log/0009-freeze-day-07-component-type-procedure.md).

## Handoff

- **Current state:** Day 7 procedure and implementation are in progress; the safety split remains locked.
- **Next action:** Complete the full benign component-type scan, analysis, audit, and sprint closeout.
- **Verification still needed:** Real-checkpoint identity/equivalence controls, exact record coverage, paired-bootstrap regeneration, artifact integrity, and figure QA.
