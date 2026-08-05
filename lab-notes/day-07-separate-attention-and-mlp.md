# Day 7: Separate Attention and MLP Contributions

## Phase information

- **Sprint phase:** Day 7
- **Status:** Complete
- **Calendar dates:** 2026-08-04
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [day04-v1](../data/splits/day04-v1/README.md)
- **Frozen layers:** 12, 11, 10, and 9 in Day 6 rank order

## Objective

Compare exact full-response patches of the complete attention branch, complete MLP branch, and complete block output at the four frozen layers. Test rescue and induction on every frozen benign example, then quantify held-out transfer, negative-example effects, irrelevant-trigger specificity, and deterministic random-layer controls without using safety data or changing the retained layers.

## Session log

### 2026-08-04 — Session 1

#### Planned focus

- Freeze the component-type sites, data scopes, control scopes, random layers, metrics, and interpretation rules before inspecting patch results.
- Implement and test a generic early-terminated component intervention runner.
- Run the complete selected-layer grid on all discovery and validation positives and negatives.
- Run irrelevant-trigger rescue controls on every positive benign example.
- Run random-layer rescue controls on discovery positives.
- Produce paired estimates, causal-effect figures, a machine-readable package, and an independent audit.

#### Work performed

- Froze the full procedure in [Decision 0009](../decision-log/0009-freeze-day-07-component-type-procedure.md) before inspecting any component-patch result.
- Added a reusable truncated component runner that captures or patches attention output, MLP output, and complete block output at arbitrary executable sites and terminates after scoring block 12.
- Added deterministic paired analysis for concept-level and equal-concept macro effects, 10,000-replicate bootstrap intervals, attention-versus-MLP contrasts, block-versus-branch contrasts, irrelevant-trigger contrasts, and selected-versus-random-layer contrasts.
- Added four focused component-analysis tests and retained all earlier intervention, split, baseline, and localization tests.
- Ran a real-checkpoint preflight. Normal, correct-trigger, and irrelevant-trigger truncated scores matched full 42-block forwards bit-for-bit. All 36 same-condition identity patches had exactly zero score change.
- Evaluated all 1,408 frozen benign examples. Recorded 4,224 baseline rows, 33,792 selected correct-trigger patch rows, 8,448 irrelevant-trigger positive-control rows, and 3,072 discovery-positive random-layer rows, for exactly 49,536 rows.
- Used frozen selected layers 12, 11, 10, and 9. Deterministic random pre-monitor layers were 5, 0, 8, and 3 under the preregistered SHA-256 rule.
- Compressed the canonical raw output into a deterministic gzip archive, computed complete concept and macro summaries, rendered primary and control figures, and generated a SHA-256/byte manifest.
- Ran the independent package verifier. All 22 checks passed.
- Inspected both 4,650 by 3,150 publication figures at original resolution; titles, labels, legends, intervals, scales, and panel boundaries were legible with no overlap or clipping.

#### Results and evidence

- The complete result package and interpretation are in [results/day-07](../results/day-07/README.md).
- Attention is stronger than MLP on average, particularly at layer 11, but both isolated branches carry only part of the causal effect.
- At layer 11, discovery rescue is 0.4631 for attention, 0.2974 for MLP, and 0.9720 for complete block output. Held-out validation rescue is 0.3168, 0.2076, and 0.8744.
- Layer-11 discovery induction is 0.3725 for attention, 0.2267 for MLP, and 0.9663 for the block. Validation induction is 0.3069, 0.1250, and 0.9073.
- Every layer-11 attention-minus-MLP macro contrast is positive with a strictly positive paired interval on discovery and validation.
- Complete-block effects significantly exceed attention plus MLP isolated effects. At layer 11, the block-minus-branch-sum gap is 0.2115 for discovery rescue, 0.3499 for validation rescue, 0.3672 for discovery induction, and 0.4754 for validation induction; all intervals are strictly positive.
- The effect is concept-heterogeneous. Comforting and jokey have stronger layer-11 MLP than attention rescue, while literature-focused validation has essentially zero isolated-branch rescue but 0.798 complete-block rescue.
- Random-layer controls remain small. Layer 8 is the strongest random block control at 0.0771 rescue, far below the selected layer-11 block effect of 0.9720.
- Correct-trigger effects exceed irrelevant-trigger effects strongly at layers 10 through 12, especially for complete blocks. Irrelevant-trigger effects are still nonzero, so the patched states are only partly trigger-specific.
- Negative-example effects are smaller than positive recovery but nonzero in some late complete-block cells. This flags limited off-target monitor movement and rules out a claim of perfect class selectivity.
- The supported classification is a heterogeneous, attention-leaning, divided, and non-additive late-block mechanism. It is not attention-only, MLP-only, or a simple sum of isolated branches.
- Validation was never used to select or rerank a layer or site. Safety remained locked and unaccessed.

#### Failures and unexpected observations

- No preflight, model-forward, intervention, pairing, row-grid, summary, audit, test, or figure-validity check failed.
- Runtime varied substantially with sequence length. The evaluator sorted each concept/class group by rendered length, so later batches were slower; row-key resumption and per-row flushing kept all progress recoverable.
- The irrelevant-trigger control was not a causal null. Complete-block irrelevant rescue reached 0.3961 on discovery and 0.2966 on validation at layer 11. Correct-trigger rescue remained much stronger, but this shows that block replacement can reverse monitor movement associated with a mismatched monitoring instruction.
- Isolated branch effects did not add to complete-block effects. This was anticipated by the frozen interpretation rule and is evidence for sequential interactions or broader state replacement, not a failed decomposition identity.
- The macro attention advantage hides meaningful concept exceptions, so Day 8 must not discard MLP candidates or use one branch-only candidate set.

## Open questions and blockers

- No Day 7 blocker remains.
- Day 8 must identify which individual heads and MLP candidates inside the frozen layers reproduce the exact causal effects and whether their strength is consistent across concepts.
- Behavioral preservation, grouped necessity/completeness, and safety transfer remain untested and cannot be inferred from Day 7 monitor-score movement.

## Decisions

- The exact procedure was frozen before the scan in [Decision 0009](../decision-log/0009-freeze-day-07-component-type-procedure.md).
- No new durable decision was added after the result. The Day 8 scope was already fixed to retain all attention heads and the whole MLP at every frozen layer, so the empirical classification changes no methodology or project governance.

## Handoff

- **Current state:** Day 7 is complete. The audited package contains all 49,536 raw rows, complete paired summaries, contrasts, two publication figures, and a 22-check audit.
- **Next action:** Begin Day 8 candidate screening and exact individual-component testing at frozen layers 12, 11, 10, and 9.
- **Carry forward:** Treat layer 11 as the strongest isolated-branch region but retain all frozen layers and both branch types. Preserve discovery-only prioritization, held-out evaluation, exact response pairing, irrelevant/random controls, paired uncertainty, behavioral checks, and locked safety.

## Day closeout

- **Outcome:** Complete. Every Day 7 checklist item, completion gate, and deliverable is satisfied.
- **Verification:** 22/22 package-audit checks pass; 36/36 identity patches and 3/3 full-forward comparisons are exact; all 27 automated tests pass; both publication figures pass visual QA; artifacts regenerate deterministically.
- **Commits:** Procedure freeze commit 6863b2d; results closeout commit containing the audited Day 7 package.
