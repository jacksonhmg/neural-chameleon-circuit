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

## Open questions and blockers

- Safety remains locked pending a committed authorization. No safety prompt, response, score, or activation has been read or computed.

## Decisions

- The central safety result must be sealed before secondary confounds under [Decision 0017](../decision-log/0017-seal-safety-result-before-confounds.md).

## Handoff

- **Current state:** Day 13 procedure and evaluator are frozen and benign-preflighted; safety is still locked.
- **Next action:** Commit the evaluator, create the exact safety authorization, then run and seal the 1,944-row central grid before any confound.
