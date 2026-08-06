# Decision 0015: Freeze the Day 12 Benign Transfer Procedure

- **Date:** 2026-08-06
- **Status:** Accepted

## Decision

Keep the Day 8 discovery-selected K=16 component set unchanged and test it on the seven held-out benign validation concepts without reranking. Measure direct set-level rescue and induction from the already frozen Day 9 and Day 10 grids, and run a new complete 68-candidate rescue grid on validation positives only to measure component-ranking correlation and top-16 overlap against the discovery results.

Use Day 11 direct MLP-output interventions to test a shared downstream actuator and its individual head source-contribution interventions to test whether trigger-reading roles vary by concept. Apply the preregistered hierarchical classification in [`frozen-benign-transfer-plan.json`](../results/day-12/frozen-benign-transfer-plan.json).

Safety data remain locked and forbidden. Validation results may characterize transfer but may not change the component set, K, controls, thresholds, safety-test script, or confirmatory metrics.

## Rationale

Set-level transfer establishes that the same identities participate causally, but it cannot distinguish identical routing from overlapping or concept-specific routing. The full candidate matrix adds identity-level rank and overlap evidence, while the prior direct-output and source-specific interventions separate downstream actuation from trigger reading. Freezing all rules before the new validation grid prevents held-out results from silently becoming a second selection set.

## Consequences

- Day 12 will add 30,912 validation rows: one baseline and 68 independent exact rescue patches for each of 448 positive examples.
- The final component set stays exactly the discovery-selected K=16 set.
- Per-concept results remain primary; macro summaries use equal concept weight.
- The safety analysis implementation and its exact artifact hashes must be frozen and committed after benign analysis but before the safety lock is opened.
