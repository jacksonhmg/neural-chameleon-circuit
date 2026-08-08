# Day 23: Measure Output Distributions

## Phase information

- **Sprint phase:** Day 23
- **Status:** In progress
- **Calendar date:** 2026-08-08
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)

## Objective

Run the authorized teacher-forced full-model distribution grid for all positive safety examples and the frozen negative controls.

## Session log

### 2026-08-08 — Session 1

#### Planned focus

- Run a no-outcome model preflight against the committed authorization.
- Execute the resumable 3,880-row grid without changing mappings, examples, metrics, or thresholds.
- Seal the archive, apply the frozen gate, and audit exact coverage and provenance.

#### Work performed

- Loaded the released checkpoint and ran the committed no-outcome preflight.
- All five authorization-status, mapping-count, example-count, grid, and hook-state checks passed.
- Confirmed that neither a working behavioral row nor a sealed outcome archive existed after preflight.
