# Day 24: Run Coupled Generation

## Phase information

- **Sprint phase:** Day 24
- **Status:** Complete
- **Calendar date:** 2026-08-08

## Objective

Run the frozen no-future-leakage coupled autoregressive diagnostic and create a local anonymized qualitative-review packet.

## Session log

### 2026-08-08 — Session 1

#### Planned focus

- Run a no-outcome checkpoint preflight against the committed Day 22 authorization.
- Execute normal/triggered baselines and all 18 coupled intervention lanes on the four frozen examples.
- Seal 80 rows, compute descriptive diagnostics, and create the local blinded review packet.

#### Work performed

- Loaded the released checkpoint and passed all five authorization-status, example-count, condition-count, expected-row, and hook-state preflight checks.
- Confirmed that no generation working row or sealed generation archive existed after preflight.
- Ran all 18 intervention lanes and two baselines on each of the four frozen examples with token-by-token donor recomputation and no future reference tokens.
- Sealed exactly 80 rows with SHA-256 `1747dde1227dbc32446bd816729ab51837590783727bec0546d9e57c40048a5a` and passed all ten audit checks.
- Selected mappings exactly match the destination-base output in 56.25% of lanes versus 90.63% for route-matched nulls; mean base-token F1 is 0.887 versus 0.967.
- Generated and inspected the locally blinded packet before consulting its condition key. All outputs were coherent and on-task at the 32-token horizon. Harmfulness-positive outputs consistently rejected or qualified the stereotype; no stance reversal or gross degeneration was observed.
- Retained an empty rating template for any future independent assessment. The current qualitative review is explicitly labeled local, single-reviewer, non-independent, and secondary.

#### Failures and unexpected observations

- The harmfulness examples were substantially slower because most of their 18 lanes remained active through the full 32-token cap. The process stayed active and completed with no restart, hook leak, partial block, or protocol change.
- Selected routes changed greedy outputs appreciably more often than matched nulls, especially for harmfulness, even though the changed outputs remained coherent in the fixed sample. “Behavior preservation” must therefore remain an equivalence-bound and task-level statement, not exact string identity.

## Handoff

- **Current state:** Coupled-generation raw results, automatic summaries, blinded packet/key, local assessment, and audit are sealed.
- **Next action:** Commit Day 24, then apply the complete Days 22–25 package audit and update the project-level interpretation.
