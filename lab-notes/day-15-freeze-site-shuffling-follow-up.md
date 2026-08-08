# Day 15: Freeze the Site-Shuffling Follow-up

## Phase information

- **Sprint phase:** Day 15
- **Status:** Complete
- **Calendar date:** 2026-08-07
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Parent evidence:** [Day 14 falsification log](../results/day-14/falsification-log.md)
- **Protocol:** [Frozen site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Turn Day 14's unexpected shuffled-site induction into a separately frozen study that can distinguish a portable trigger-specific signal from head mismatch, destination receptivity, cross-depth shortcuts, nonlinear composition, and off-manifold disruption.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Preserve Day 1–14 artifacts unchanged.
- Freeze hypotheses, component populations, data roles, estimands, mapping generation, controls, inference, and interpretation rules.
- Materialize and audit exact permutation mappings before any new outcome is generated.
- Commit the complete freeze locally before beginning Day 16.

#### Work performed

- Confirmed a clean worktree at local commit `436c90f`; `origin/main` remains 21 commits behind at the Day 13 state.
- Defined the central confound algebra: absolute insertion changes a destination by both the trigger-specific source delta and the ordinary source/destination difference.
- Promoted destination-matched source contrasts and destination-relative delta transport to primary estimands.
- Reserved the original 32 Day 14 positives for exploratory triage and all remaining positives for prospective site-shuffling confirmation.
- Defined a deterministic, layer-count-matched 12-head null population and benign-only mapping-selection procedure.
- Retained the restriction against push, release, tagging, draft transmission, and external contact.

#### Results and evidence

- No new site-shuffling result was generated during the freeze.

#### Failures and unexpected observations

- None.

#### Artifacts and commits

- `results/day-15/frozen-site-shuffling-plan.json`
- `results/day-15/frozen-mapping-ensemble.json`
- `results/day-15/freeze-audit.json`
- `decision-log/0020-freeze-site-shuffling-follow-up.md`

## Open questions and blockers

- None at freeze.

## Decisions

- [Decision 0020](../decision-log/0020-freeze-site-shuffling-follow-up.md)

## Handoff

- **Current state:** The protocol, exact mapping ensemble, data-role manifest, and freeze audit were committed before new outcomes.
- **Next action:** Day 16 artifact triage.
- **Verification still needed:** None.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-07
- **Final evidence:** [Passing freeze audit](../results/day-15/freeze-audit.json)
- **Final commit:** `66427eb`
- **Carry-forward items:** Execute Days 16–21 exactly under the frozen protocol.
