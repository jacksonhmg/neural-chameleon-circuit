# Day 22: Freeze Behavioral Transport

## Phase information

- **Sprint phase:** Day 22
- **Status:** In progress
- **Calendar date:** 2026-08-08
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Method:** [Days 22–25 behavioral transport method](../docs/day-22-25-behavioral-transport-method.md)

## Objective

Commit the exact final-map behavioral question, endpoint definitions, example selection, equivalence bounds, coupled-generation semantics, and implementation before generating any new final-map behavioral outcome.

## Session log

### 2026-08-08 — Session 1

#### Planned focus

- Preserve all four Day 21 mappings and route-matched nulls.
- Implement deterministic example selection and natural-direction output metrics.
- Materialize a hash-pinned authorization only after the evaluator is committed.
- Generate no final-map behavioral outcome before authorization.

#### Work performed

- Defined the three scientifically distinct outcomes: behavior-preserving probe transport, natural-direction output transport, and nonspecific disruption.
- Fixed all 162 positive safety examples and deterministic selection rules for 32 negative controls and four coupled-generation diagnostics.
- Fixed the four Day 21 mappings, route-matched nulls, K12 identity benchmark, both causal directions, alpha 1.0, behavioral metrics, 10,000-replicate bootstrap, equivalence bounds, and mechanical interpretation.
- Implemented centered-logit directional projection, deterministic subset selection, the resumable 3,880-row full-model evaluator, its independent analyzer, and the 80-row coupled autoregressive generator and analyzer.
- Added unit coverage for deterministic selection, direction/sign semantics, logit-shift invariance, overlap diagnostics, lane construction, and row-specific induction/rescue patches.
- The complete 62-test project suite passes. No final-map behavioral outcome has been generated.

#### Failures and unexpected observations

- None so far.

## Handoff

- **Current state:** Protocol and implementation pass the full test suite; no behavioral outcome has been generated.
- **Next action:** Commit the evaluator, then materialize, audit, and separately commit the exact authorization.
