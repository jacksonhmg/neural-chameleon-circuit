# Day 10: Test Sufficiency

## Phase information

- **Sprint phase:** Day 10
- **Status:** Complete
- **Calendar dates:** 2026-08-05
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

- Froze the exact natural-transplant grid, component and residual groups, behavior subset, dose subset, alpha grid, evidence rules, numerical tolerances, activation-norm bound, and safety isolation before generating sufficiency results.
- Implemented generalized vectorized transplantation across attention heads, MLP outputs, and complete residual outputs, including exact activation interpolation and full-model fixed-continuation NLL evaluation.
- Added deterministic paired-bootstrap analysis, tables, publication figures, reproducible archives, an artifact manifest, and an independent package audit.
- Passed all 41 CPU tests, then committed the evaluator before starting the real-checkpoint run so every raw row names an immutable implementation.
- Passed the real-checkpoint preflight: 26 identities, vectorized probe and NLL comparisons, four exact interpolation endpoints, two activation-norm checks, group-order invariance, and zero leaked hooks.
- Completed 19,712 exact all-benign rows, 3,168 interior dose rows, and 616 full-model behavior rows without opening safety.
- Ran the independent 24-check audit, inspected both 4,800 by 2,700 figures, and verified a second analysis pass reproduced all 17 generated artifacts byte for byte.

#### Results and evidence

- Selected K16 induced 0.9322 [0.9272, 0.9369] of the full positive suppression effect on discovery and 0.8063 [0.8003, 0.8123] on validation.
- Selected-minus-random K16 was 0.9189 [0.9138, 0.9236] and 0.7609 [0.7532, 0.7688], satisfying the frozen support rule in both scopes.
- The official K16-only classification is **partial sufficiency** because validation did not reach the frozen 0.9 near-complete threshold.
- Adding the block-8 entry residual raised induction to 0.9830 [0.9812, 0.9847] on discovery and 0.9569 [0.9547, 0.9590] on validation. The paired context increments were 0.0508 and 0.1506.
- Selected K16 passed the frozen monotonic dose-response rule. Its discovery curve was 0, 0.1351, 0.3945, 0.7024, 0.9334; validation was 0, 0.1710, 0.4253, 0.6393, 0.8073.
- The maximum interior activation-norm ratio was 0.9942, below the 1.001 bound.
- Selected K16 produced negative-example suppression shifts of 0.0565 on discovery and 0.1290 on validation; induction was not positive-only at the probe.
- Selected K16 increased fixed-continuation NLL by 0.0653 on discovery positives and 0.0830 on validation positives. The context-augmented transplant increased it by 0.0873 and 0.1211. Behavioral preservation is therefore not established.
- Complete results, tables, figures, interpretation, and limits are in [the Day 10 package](../results/day-10/README.md).

#### Failures and unexpected observations

- No execution or verification failure occurred.
- The main scientific surprise was the size and heterogeneity of the upstream residual contribution. K16 alone induced only 0.4051 for held-out all-caps, while K16 plus the entry residual induced 0.9394. The combined transplant exceeded 0.91 for every validation concept.
- Exact triggered source activations can have substantially different norms from normal destination activations; this is expected for natural endpoints and was reported rather than clipped. All convex interior mixtures nevertheless remained inside the frozen endpoint norm bound.

## Open questions and blockers

- What information in the upstream residual supplies the large held-out context increment remains unresolved.
- Whether the selected sites are the actuator while earlier components recognize monitoring language and the named concept is the Day 11 question.
- Free-generation effects, semantic preservation, distributional changes, and safety preservation remain untested; the safety split is still locked.
- There is no blocker to beginning Day 11.

## Decisions

- The complete Day 10 procedure and surrounding-residual interpretation are frozen before results in [Decision 0013](../decision-log/0013-freeze-day-10-sufficiency-procedure.md).

## Handoff

- **Current state:** Day 10 is complete. Natural selected activations are sufficient to induce most evasion, matched random controls are near zero, selected K16 is formally partial on validation, upstream block-8 residual context closes most of the gap, and selected K16 has a monotonic dose response. Safety remains locked.
- **Next action:** Begin Day 11 by separating monitoring-language recognition, named-concept recognition, and the downstream concealment actuator using token- and component-specific causal interventions.
- **Verification complete:** All 24 independent audit checks, all 41 tests, exact row coverage, deterministic summary regeneration, figure inspection, artifact hashes, numerical controls, and safety isolation pass.
