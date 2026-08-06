# Day 12: Test Benign Cross-Concept Transfer

## Phase information

- **Sprint phase:** Day 12
- **Status:** Complete
- **Calendar dates:** 2026-08-06
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen component set:** [Day 8 selection](../results/day-08/frozen-component-selection.json)
- **Frozen procedure:** [Day 12 transfer plan](../results/day-12/frozen-benign-transfer-plan.json)

## Objective

Test whether the discovery-frozen component identities transfer causally to seven held-out benign concepts, quantify how component rankings and top sets vary across concepts, and distinguish a shared sparse circuit from concept-specific routing, overlapping circuits, or a broadly distributed mechanism.

## Session log

### 2026-08-06 — Session 1

#### Planned focus

- Freeze the unchanged K=16 set, validation candidate grid, direct-transfer metrics, ranking and overlap metrics, actuator and trigger-reader rules, model hashes, and safety isolation before new results.
- Implement and test the validation evaluator, cross-concept analysis, figures, manifest, and independent audit.
- Run the complete pinned-checkpoint grid and analyze all 11 benign concepts.
- Freeze the final component set, exact safety-test analysis, and confirmatory metrics before safety access.

#### Work performed

- Froze `day12-v1` before validation candidate results. The plan retains the Day 8 K=16 set without reranking, adds a 30,912-row exact validation rescue grid, and preregisters the direct-transfer, ranking, overlap, shared-actuator, trigger-reader, and mechanistic-classification rules.
- Implemented a resumable validation evaluator, pure cross-concept estimators, two publication figures, analysis tables, a manifest, and an independent package verifier.
- Added three Day 12 tests plus a frozen-safety-analysis test; all 49 repository tests pass.
- Ran the pinned checkpoint across all 448 validation positives and 68 candidates, producing exactly 30,912 rows. Every row names the single committed evaluator `ed19e41`.
- Combined the new rows with the 17,664 Day 8 discovery rows and ran the frozen 10,000-replicate paired bootstrap analysis across all 11 benign concepts.
- Re-ran analysis and reproduced all generated pre-audit artifacts byte for byte. Visually inspected both 4,800 × 2,700 PNGs.
- Ran the independent 20-check audit successfully, including the final-set and executable safety-analysis freezes.
- Kept the exact Day 8 K=16 set unchanged, implemented and hashed the Day 13 confirmatory safety analysis, and froze its metrics and support rule while safety remained locked.

#### Results and evidence

- The unchanged K=16 set directly transfers: validation rescue is 0.820 [0.810, 0.828] and induction is 0.806 [0.800, 0.812]. Both selected-minus-random intervals are also strictly positive.
- Every one of 11 benign concepts has strictly positive K16 rescue and induction intervals.
- Discovery-to-validation candidate-rank rho is 0.705; macro top-16 overlap is 12/16, overlap coefficient 0.75, and Jaccard 0.60.
- The selected four-MLP actuator has positive rescue and induction intervals on both discovery and validation. Layer-11 MLP alone has positive point recovery on all 11 concepts.
- K=4 reaches 78.7% of K=16 discovery rescue and 53.0% of validation rescue, failing the 80% shared-sparse gate.
- Trigger-source roles and top attention identities vary by concept. The official frozen classification is `concept_specific_routing_into_shared_actuator`.

#### Failures and unexpected observations

- The first launch used an unrelated ESP-IDF Python environment and stopped before model loading because `accelerate` was absent. It wrote no results; production used the project’s pinned research environment.
- A trial candidate chunk of four was slower on long chemistry responses. The process was interrupted after five complete batches and resumed with the preflighted chunk of two. Unique record keys preserved 116 already completed conditions from the interrupted sixth batch; the final archive is complete and duplicate-free.
- Literature-focused and chemistry-based responses were materially longer than other concepts, so exact full-response patches took longer. No sequence was truncated.

## Open questions and blockers

- Teacher-forced probe transfer does not establish free-generation preservation; Day 13’s NLL, KL, generation, task, and degradation controls remain necessary.
- Concept-specific source roles use eight Day 11 examples per concept and a permissive “some heterogeneity” rule; they establish non-identical routing, not a unique stable semantic label for every head.
- Validation ranking is descriptive only. The unchanged final set deliberately retains the discovery shared-candidate gate and therefore is not identical to a naive validation top-16 list.

## Decisions

- The complete benign transfer procedure is frozen before new results in [Decision 0015](../decision-log/0015-freeze-day-12-benign-transfer-procedure.md).
- The final component set and safety analysis are frozen before safety access in [Decision 0016](../decision-log/0016-freeze-final-circuit-and-safety-analysis.md).

## Handoff

- **Current state:** Day 12 is complete. Direct benign transfer, rank sharing, a shared actuator, and concept-varying routing are documented; the unchanged final set and safety analysis are frozen; safety remains locked.
- **Next action:** Begin Day 13 by creating an authorization that hashes the committed final set, `day13-confirmatory-v1` procedure, and locked safety split, then run and seal the 1,944-row central safety grid before confounds.
