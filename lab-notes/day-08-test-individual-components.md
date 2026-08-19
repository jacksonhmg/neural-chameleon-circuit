# Day 8: Rank and Exactly Test Individual Components

## Phase information

- **Sprint phase:** Day 8
- **Status:** Complete
- **Calendar dates:** 2026-08-05
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen layers:** 12, 11, 10, and 9

## Objective

Compare inexpensive attention-head and MLP screening methods with exact causal rescue, freeze a discovery-only ordered component set, and test its exact rescue, induction, held-out transfer, matched controls, and fixed-continuation NLL without reranking or opening safety.

## Session log

### 2026-08-05 — Session 1

#### Planned focus

- Freeze the remaining screening, exact-test, control, K-selection, validation, and behavior details.
- Implement tested vectorized independent-component patching and one-backward-pass screening.
- Scan all 68 eligible candidates on discovery positives.
- Commit the exact ordered component set and controls before validation.
- Run frozen held-out confirmation, analysis, figures, audit, and sprint closeout.

#### Work performed

- Froze the complete candidate, screening, exact-patch, K, control, held-out, behavior, uncertainty, numerical-equivalence, and safety-isolation procedure in [Decision 0010](../decision-log/0010-freeze-day-08-individual-component-procedure.md) before recording a candidate result.
- Implemented the 68-candidate universe: all 16 pre-output-projection query-head slices and the whole MLP at zero-based layers 12, 11, 10, and 9.
- Added vectorized independent-candidate patching for truncated probe scores and complete-model response NLL. Added one-backward-pass activation RMS, probe projection, attribution patching, and gradient RMS screens.
- Added six focused individual-component tests, including deterministic selection and a complete synthetic confirmation grid; all 33 repository tests pass.
- Ran a real-checkpoint preflight with two complete-forward comparisons, 136 same-condition identities, two changed-batch vector checks, and finite screening metrics.
- Evaluated all 68 candidates on all 256 discovery positives. Recorded exactly 17,664 discovery rows.
- Generated the exact discovery ranking. Forty-two candidates passed the shared gate; the frozen largest-prefix rule set K to 16. Committed and pushed the order, controls, summary, hashes, and [Decision 0011](../decision-log/0011-freeze-day-08-component-set.md) at `acc5df8` before loading held-out individual-component results.
- Ran exact induction for the selected and random identities on discovery positives and exact rescue plus induction for those same identities on all 448 validation positives. Recorded exactly 37,568 confirmation rows with selection commit and component-set provenance attached to every row.
- Ran complete-model fixed-continuation NLL for both causal directions and all 16 selected identities on the frozen 44-example behavior subset. Recorded exactly 1,452 behavior rows.
- Computed 10,000-replicate paired-bootstrap intervals, example and concept consistency, selected-versus-random contrasts, complete nonselected same-layer comparisons, and behavior NLL changes.
- Rendered and inspected the discovery, confirmation, and control/behavior figures at their original 4,800 by 2,700 resolution.
- Ran the independent discovery and final package audits. They pass 18/18 and 21/21 checks. Reanalysis reproduced every tracked output byte for byte.

#### Results and evidence

- The complete result package and interpretation are in [results/day-08](../results/day-08/README.md).
- The frozen top three are `layer_11.mlp`, `layer_11.head_08`, and `layer_10.head_12`.
- Layer-11 MLP exact discovery rescue is 0.2974 [0.2863, 0.3082] and held-out rescue is 0.2076 [0.1994, 0.2159]. Discovery and validation induction are 0.2266 and 0.1250.
- Layer-11 head 8 discovery and validation rescue are 0.2489 and 0.1390; discovery and validation induction are 0.2330 and 0.1646.
- Layer-10 head 12 discovery and validation rescue are 0.1681 and 0.0854; discovery and validation induction are 0.1438 and 0.1301.
- Mean individual effects across the 16 fixed selected identities are 0.0785 discovery rescue, 0.0528 validation rescue, 0.0626 discovery induction, and 0.0695 validation induction. Deterministic controls remain between -0.0021 and 0.0023. All four selected-minus-control intervals are strictly positive.
- Every selected identity exceeds the complete nonselected same-layer mean for discovery rescue with a strictly positive interval.
- The strongest identities are highly sign-consistent across examples and transfer without reranking, but magnitude is concept-heterogeneous. Layer-11 MLP validation rescue is 0.5027 on Finnish and approximately zero on literature-focused.
- Attribution patching best predicted the exact ranking (Spearman 0.848; 14/16 overlap), followed by probe projection (0.810; 14/16), activation RMS (0.473; 12/16), and gradient RMS (0.288; 7/16).
- All-benign fixed-continuation macro NLL changes range from -0.0303 to 0.0118 nats per response token. This bounded diagnostic found no large likelihood penalty but does not establish behavioral preservation.
- The supported conclusion is a set of individually causal, transferable late-layer components led by layer-11 MLP and head 8, with partial necessity/sufficiency and substantial concept heterogeneity. Completeness remains untested until grouped patches are run.
- Validation never changed component identities, order, K, or controls. Safety remained locked and unaccessed.

#### Failures and unexpected observations

- The initial real-checkpoint vector preflight required bit-exact equality across changed batch shapes and failed before writing any candidate result row. Same-shape identities were exact, deterministic CPU vectorization was exact, and the discrepancy reproduced the known BF16/MPS batch-kernel effect. Before scanning, the procedure was amended and committed to use at most two candidates per vector batch with observable-scale tolerances of 0.002 for probe score and 0.02 for NLL. The successful preflight stayed within those bounds.
- Runtime varied sharply with sequence length. Groups were sorted by rendered length, row-key resumption was enabled, and every row was flushed immediately; no progress was lost.
- The first final-audit invocation found a verifier field-name mismatch between `complete_forward_check_count` and the shorter alias used in the verifier. This was an audit-code error, not a data failure. Correcting the field names produced a 21/21 pass and deterministic rerun.
- The screening methods differed substantially. Attribution patching and probe projection were useful, but gradient RMS recovered only 7 of the exact top 16 and had Spearman correlation 0.288.
- Several lower-ranked components transferred more strongly in induction than their discovery rescue rank suggested. For example, layer-9 MLP validation induction is 0.1326 despite discovery rescue of 0.0332. The discovery-frozen ordering was preserved rather than retrospectively optimized.

## Open questions and blockers

- No Day 8 blocker remains.
- Day 9 must test the frozen nested prefixes 1, 2, 4, 8, and 16 as simultaneous grouped interventions rather than adding individual effects.
- It remains unknown how much recovery saturates with set size, how selected groups compare with matched random groups, and how grouped patches affect positive and negative examples.
- Free-generation behavior, semantic task preservation, grouped completeness, and safety transfer remain untested.

## Decisions

- The complete Day 8 procedure is frozen before scanning in [Decision 0010](../decision-log/0010-freeze-day-08-individual-component-procedure.md).
- The ordered K=16 component set and deterministic controls were frozen before held-out confirmation in [Decision 0011](../decision-log/0011-freeze-day-08-component-set.md).

## Handoff

- **Current state:** Day 8 is complete. The audited package contains 17,664 discovery rows, 37,568 confirmation rows, 1,452 behavior rows, complete paired summaries, three publication figures, and committed selection provenance.
- **Next action:** Begin Day 9 grouped necessity and completeness using the frozen ordered prefixes 1, 2, 4, 8, and 16 and matched random component sets.
- **Carry forward:** Preserve the exact candidate order and patch sites. Treat individual effects as non-additive; test groups directly. Retain concept-level reporting, matched controls, paired response tokens, behavior diagnostics, and the safety lock.

## Day closeout

- **Outcome:** Complete. Every Day 8 checklist item, completion gate, and deliverable is satisfied.
- **Verification:** 18/18 discovery-audit and 21/21 final-audit checks pass; 136/136 same-condition identity patches pass; real vector checks satisfy the committed tolerances; all 33 automated tests pass; all three publication figures pass visual QA; artifacts regenerate deterministically.
- **Commits:** Procedure freeze `141590b`; numerical calibration `2b1d991`; discovery selection freeze `acc5df8`; analysis/audit implementation `eaf55a5`; final audited results closeout commit containing this note.
