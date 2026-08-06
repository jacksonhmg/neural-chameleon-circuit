# Day 9: Test Grouped Necessity and Completeness

## Phase information

- **Sprint phase:** Day 9
- **Status:** Complete
- **Calendar dates:** 2026-08-05
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen component order:** [Day 8 selection](../results/day-08/frozen-component-selection.json)

## Objective

Directly patch increasingly large nested component groups, compare them with frozen matched controls, quantify positive completeness and negative off-target movement, and determine whether the individually localized mechanism appears compact, moderately distributed, or highly distributed without using additive assumptions or safety data.

## Session log

### 2026-08-05 — Session 1

#### Planned focus

- Freeze grouped sites, prefixes, controls, metrics, behavior scope, numerical checks, and the compactness rule.
- Implement and test vectorized grouped activation patching.
- Run the complete all-benign probe-score and frozen-subset behavior grids.
- Produce completeness curves, controls, machine-readable summaries, and an independent audit.

#### Work performed

- Froze the selected and random prefixes, layer-11 group, deterministic outside-layer control, block-output positive control, all-benign score scope, frozen behavior subset, numerical tolerances, bootstrap, compactness rule, and safety isolation in [Decision 0012](../decision-log/0012-freeze-day-09-grouped-necessity-procedure.md). Committed and pushed the plan at `7fa5c9c` before generating grouped results.
- Implemented independent multi-site group jobs that can evaluate at most two grouped interventions in one expanded model batch. Added truncated probe scoring and full-model response-NLL paths without summing individual effects.
- Added four focused tests for vector-versus-independent equivalence, complete-model NLL, member-order invariance, hook cleanup, frozen group construction, deterministic summaries, and compactness classification. All 37 repository tests pass.
- Added a resumable, fail-closed model evaluator; deterministic analysis and archive generation; two 4,800 by 2,700 figures; five analysis-ready CSV tables; a SHA-256 artifact manifest; and an independent 21-check verifier. Committed and pushed the implementation at `ff2328a` before the full run.
- Ran the real-checkpoint preflight on two discovery HTML positives. All 26 same-shape identities were exact. Two vectorized group scores and two full-forward NLL results matched independent execution exactly, reversing the 16 selected group members was exact, and no hooks leaked.
- Directly measured all 13 grouped rescue interventions on all 1,408 benign examples and both labels. Recorded exactly 19,712 score rows.
- Ran the same 13 grouped interventions through the complete model on the frozen two-per-concept/class behavior subset. Recorded exactly 616 rows across 44 examples.
- Computed 10,000-replicate paired-bootstrap summaries with seed 42 and equal concept weights. Produced absolute recovery, negative-example shifts, selected/random contrasts, K16-relative curves, concept-level results, behavior NLL, and the frozen distribution classification.
- Rendered and visually inspected both publication figures. Ran the independent package audit twice; it passes all 21 checks. A second analysis pass reproduced the raw archives, summary, PDFs, and artifact manifest byte for byte.

#### Results and evidence

- The complete result package and interpretation are in [results/day-09](../results/day-09/README.md).
- The frozen rule classifies the mechanism as **highly distributed**. K8 reaches 92.5% of K16 on discovery but only 73.2% on validation, below the required 80% in both scopes.
- Absolute selected K16 recovery is 0.9262 [0.9202, 0.9318] on discovery and 0.8198 [0.8103, 0.8285] on validation. The block-12 positive control is exactly 1.0 for every example, so the frozen set is strong but incomplete.
- Selected recovery grows at every K. Discovery K1/K2/K4/K8 are 0.2974, 0.4855, 0.7293, and 0.8567. Validation values are 0.2076, 0.2880, 0.4348, and 0.5999.
- Random K16 recovery is -0.0162 on discovery and -0.0369 on validation. Every selected-minus-random positive contrast has a strictly positive interval at every K in both scopes.
- All 17 layer-11 candidates recover 0.6255 on discovery and 0.4264 on validation, well below selected K16. The frozen mechanism therefore spans layers 9 through 12 rather than residing wholly in layer 11.
- Held-out coverage is heterogeneous. Selected K16 recovery is 0.6884 for literature-focused and 0.6221 for mathematical, while K8 is only 0.1604 and 0.2701 for those concepts. Finnish K16 is 0.9330.
- Selected K16 moves negative scores by 0.0906 [0.0807, 0.1014] relative to the positive suppression denominator, compared with -0.0166 for random K16. Positive rescue is much larger, but the intervention is not positive-only at the probe site.
- Selected K16 changes fixed-continuation NLL by -0.1133 [-0.1326, -0.0941] on positives and -0.0706 [-0.1129, -0.0278] on negatives. This bounded diagnostic shows no likelihood penalty and is consistent with restoration toward normal, but it does not establish broad behavioral preservation.
- Safety remained locked and unaccessed. Validation did not alter selection, group membership, order, or classification rules.

#### Failures and unexpected observations

- The first launch used the shell's unrelated ESP-IDF Python and stopped before model loading because that environment lacked `accelerate`. No result row was written. The run was restarted in the recorded project environment at `external/minimal_neural_chameleons/.venv`, where the pinned dependencies and MPS support were already present.
- Runtime varied with rendered sequence length as expected. Immediate flushing and exact row-key resumption prevented progress loss; the complete run required no recovery restart.
- K4 narrowly missed 80% of K16 even on discovery (78.7%), and validation was substantially more distributed: K8 reached only 73.2% of K16. The frozen rule was applied without adjustment.
- Full layer-11 coverage was much weaker on validation (42.6%) than discovery (62.6%), and almost absent for literature-focused (0.001). This strengthens the cross-layer, concept-heterogeneous interpretation.
- The selected grouped patches measurably raised negative probe scores. The full block-output positive control also raised them, so this cannot be interpreted as selected-set-specific damage alone, but it rules out a positive-only mechanism at the measured activation.

## Open questions and blockers

- No Day 9 blocker remains.
- Day 10 must test triggered-to-normal sufficiency directly; Day 9's normal-to-trigger rescue establishes grouped necessity but not sufficiency.
- It remains unknown whether transplant strength shows a dose response, whether adding surrounding residual contributions closes the held-out gap, and whether free generation preserves task behavior.
- Safety transfer remains intentionally untested until the locked Day 12 procedure is authorized.

## Decisions

- The complete procedure and distribution-classification rule are frozen before grouped results in [Decision 0012](../decision-log/0012-freeze-day-09-grouped-necessity-procedure.md).
- No post-result decision changed group membership, metrics, thresholds, or interpretation rules.

## Handoff

- **Current state:** Day 9 is complete. The audited package contains 19,712 score rows, 616 behavior rows, complete bootstrap summaries and tables, two publication figures, deterministic hashes, and a 21/21 audit pass.
- **Next action:** Begin Day 10 sufficiency using the exact frozen selected groups and matched controls, without treating Day 9 rescue as sufficiency evidence.
- **Carry forward:** Test triggered-to-normal activation transplantation, complete selected sets, surrounding residual contributions, matched controls, interpolation, and dose response. Preserve the validation information barrier and safety lock.

## Day closeout

- **Outcome:** Complete. Every Day 9 checklist item, completion gate, and deliverable is satisfied.
- **Verification:** 26/26 real-checkpoint identities pass; vectorized score/NLL and group-order checks are exact; the 21/21 package audit passes; all 37 automated tests pass; both publication figures pass visual QA; core artifacts regenerate byte for byte.
- **Commits:** Procedure freeze `7fa5c9c`; implementation `ff2328a`; final audited results closeout commit containing this note.
