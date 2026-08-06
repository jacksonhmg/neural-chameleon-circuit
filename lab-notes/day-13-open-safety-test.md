# Day 13: Open the Safety Test and Attack Confounds

## Phase information

- **Sprint phase:** Day 13
- **Status:** In progress
- **Calendar dates:** 2026-08-06
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Final component set:** [Day 12 final freeze](../results/day-12/frozen-final-component-set.json)
- **Confirmatory analysis:** [Day 12 analysis freeze](../results/day-12/frozen-confirmatory-analysis.json)
- **Execution and confound plan:** [Day 13 plan](../results/day-13/frozen-execution-plan.json)

## Objective

Test whether the benign-selected circuit causally affects unseen deception and harmfulness monitor evasion without any safety-specific modification, then attack prompt, intervention, pooling, norm, and behavioral confounds.

## Session log

### 2026-08-06 — Session 1

#### Planned focus

- Verify the Day 12 final component and analysis hashes while safety remains locked.
- Freeze the central execution, stage boundary, confound subset, prompt variants, zero ablation, alternative pooling, norm, NLL, KL, generation, and concept-processing diagnostics.
- Implement and preflight the complete evaluator on benign validation examples before unlock.
- Commit an exact safety authorization, run and seal the central grid, then run confounds.
- Analyze, visualize, audit, document, commit, and push the complete Day 13 package.

#### Work performed

- Verified the clean, pushed Day 12 handoff and the continued absence of `safety-unlock.json`.
- Froze `day13-execution-v1` while safety remained locked, including the 1,944-row central grid, a strict central-before-confounds boundary, deterministic confound and generation subsets, all prompt variants, zero ablation, alternative token pooling, activation norms, NLL, KL, generation quality, and concept-processing diagnostics.
- During the pre-unlock implementation review, caught that deception's proposed nearby term duplicated its correct trigger. Revised the nearby term to `dishonesty` and pinned behavior KL as `KL(destination || patched)` before authorization or safety-content access.
- Implemented the resumable central evaluator with committed-procedure checks, exact frozen membership validation, mean/first/last/max probe pooling, fixed-continuation NLL, token KL, activation RMS ratios, deterministic gzip sealing, and per-row implementation provenance.
- Passed a real-checkpoint preflight on two benign validation examples: all four same-shape selected/random identity checks produced bit-identical logits and exactly zero KL; vectorized scores and NLL matched independent runs within the established tolerances; reversed member order was identical; no hooks leaked. All 46 repository tests passed.
- Committed the evaluator and procedure, then created the exact `day04-v1` safety authorization bound to safety split SHA-256 `48d681…`, final component-set SHA-256 `5ef5ab…`, and Day 12 component-freeze commit `89ef7a6…`.
- Executed all 324 locked safety examples and deterministically sealed exactly 1,944 central rows at SHA-256 `430a4d57…`. Every example has two natural baselines and selected/random K=16 rescue and induction rows; no safety-specific component or rule changed.
- Ran the untouched Day 12 confirmatory analysis. Both concepts pass the frozen four-interval support gate: deception selected rescue 0.947 and induction 0.987, harmfulness selected rescue 0.844 and induction 0.726. Selected-minus-random contrasts are 0.945/0.983 for deception and 0.857/0.739 for harmfulness, with all required 95% interval lower bounds above zero.
- Central selected positive activation RMS ratios range from 0.879 to 1.291 across concept/direction point estimates, within the frozen `[0.5, 2.0]` diagnostic band. Mean positive NLL shifts are small for deception (-0.008 rescue, +0.007 induction) and larger for harmfulness (-0.114 rescue, +0.093 induction); these require the frozen confound and behavior follow-up.
- Created `central-result-seal.json` while every secondary prompt, ablation, and generation confound remained unrun.
- After committing the central seal, implemented the frozen secondary runner and analysis: exact salted subsets, seven prompt variants, selected K=16 zero ablation, four-condition greedy generation, alternative pooling, hidden-processing AUROC, and descriptive generation/task-fidelity diagnostics. The runner refuses to execute unless the committed central archive and plan still match their seal.
- Sealed all frozen secondary raw grids: 448 prompt rows (`12a4dead…`), 64 zero-ablation rows (`1e1defc8…`), and 128 generation rows (`5ac23d80…`). Verified each sealed archive contains the exact row multiset from its flushed working file, then removed only the redundant working copies.
- Prompt controls show that concept naming, not generic monitoring language, drives suppression: concept-only reproduced 0.911 of the deception gap and 0.904 of the harmfulness gap, while monitoring-only reproduced 0.002 and 0.004. Paraphrases remained strong (0.932/0.870), relocation remained substantial (0.842/0.707), and malformed triggers were near zero. Cross-safety spillover was nontrivial: the irrelevant safety concept reproduced 1.022 of the deception gap and 0.458 of the harmfulness gap.
- Zero ablation was much weaker than natural state replacement: 0.298 versus 0.938 causal fraction for deception and 0.064 versus 0.800 for harmfulness on the fixed subset. This argues against interpreting selected rescue as a simple component-removal artifact.
- The selected effect remained descriptively supported under mean, first-token, last-token, and maximum-token probe pooling for both concepts and both directions. No central activation-norm point estimate left `[0.5, 2.0]`.
- All 128 generations were nonempty. Generated-continuation positive/negative probe AUROC was 1.0 across every concept/condition cell; repetition remained low except deception normal (mean 0.189). Reference-token F1 was low (0.118–0.216), so these open-ended overlap diagnostics do not establish semantic task correctness.

## Open questions and blockers

- The central result remains supported; all frozen confounds are analyzed. Final audit and qualified interpretation remain.

## Decisions

- The central safety result must be sealed before secondary confounds under [Decision 0017](../decision-log/0017-seal-safety-result-before-confounds.md).

## Handoff

- **Current state:** Central and secondary grids are sealed and analyzed; overall frozen support is true with a prompt-specificity qualification.
- **Next action:** Commit the secondary artifacts and auditor, run the full audit, then finalize the Day 13 conclusion and sprint tracker.
