# Day 23: Measure Output Distributions

## Phase information

- **Sprint phase:** Day 23
- **Status:** Complete
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
- Executed all 20 authorized conditions for 194 examples and sealed exactly 3,880 rows with SHA-256 `237bca6ae7ca8606652557335c226e89bb5cc72323826549fe04e2f191a1dba1`.
- Applied the frozen 10,000-replicate bootstrap and mechanical interpretation only after the archive was complete.
- All four probe gates and all four KL/NLL equivalence gates pass. Deception induction, deception rescue, and harmfulness induction are `behavior_preserving_portable_evasion`.
- Harmfulness rescue is `mixed`: its directional coefficient is 0.0981 [0.0902, 0.1054], so the point lies just below the 0.10 minimum while its interval crosses that threshold.
- Mapping-level analysis shows stronger harmfulness output coupling through the two within-layer mappings than through either cross-layer mapping. The K12 identity benchmark is substantially more output-coupled for harmfulness than deception.
- Generated the primary figure and passed all nine final audit checks.

#### Failures and unexpected observations

- No execution failure, partial-example stop, hook leak, or protocol deviation occurred.
- Every selected population directional coefficient is detectably positive relative to zero and route-matched null, even where it is below the frozen meaningful-magnitude threshold. The “behavior-preserving” label must not be restated as literally zero output effect.

## Handoff

- **Current state:** Day 23 raw results, mapping/population metrics, mechanical gate, figure, and audit are sealed.
- **Next action:** Commit the primary result, then run the frozen coupled autoregressive generation diagnostic.
