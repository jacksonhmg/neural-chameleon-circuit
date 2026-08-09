# Day 32: Build the Acquired-Writer Machinery

## Phase information

- **Sprint phase:** Day 32
- **Status:** Complete
- **Calendar date:** 2026-08-09
- **Frozen parent:** `46dab91`
- **Protocol:** [Frozen acquired-writer plan](../results/day-31/frozen-acquired-writer-plan.json)

## Objective

Implement and verify the exact writer-trajectory capture, realized-forward accounting, frozen-additive-write path intervention, acquisition, and intermediate-prediction machinery required by Gate 1.

## Session log

### 2026-08-09 — Session 1

#### Planned focus

- Preserve the Day 31 contract unchanged.
- Reuse the tested response-relative pairing and pre-`W_O` hook semantics.
- Capture the raw joint-head tensor only once per layer rather than duplicating it through per-head hooks.
- Reconstruct the actual Gemma-2 normalized additive branches and enforce shared-denominator accounting closure.
- Implement total and direct-path interventions under the frozen operator.
- Add deterministic synthetic tests before a real-checkpoint preflight.

#### Work performed

- Began implementation from clean commit `46dab91`.

#### Failures and unexpected observations

- The first real-checkpoint launch under the Python 3.11 test runtime stopped before loading weights because that runtime lacks `accelerate`. The already pinned minimal-reproduction environment includes the dependency and was used instead; no package was installed or changed.
- Real-checkpoint preflight attempt 1 passed exact residual and 13-probe closure but exposed BF16 non-associativity: separately evaluated head projections differed from the fused branch by up to `0.125`, normalized allocation closure reached `0.127182`, and response-only identity recomputation shifted hidden state by `0.03125` while changing probe scores by only `0.0000703931`.
- Preserved the failed artifact and froze [Decision 0029](../decision-log/0029-freeze-bf16-accounting-clarification.md) before changing the numerical implementation or rerunning.

#### Numerical clarification implementation

- Added capture of the actually fused pre-RMSNorm attention projection at every layer through block 12.
- Implemented symmetric, explicitly reported allocation of fused-projection and normalization/cast numerical residuals across query heads.
- Anchored matched response-only counterfactual deltas to each cached realized target branch.
- Strengthened the synthetic suite to require exact direct-path identity and explicit numerical-residual diagnostics.

#### Verification and results

- Static checks passed and the complete suite passed `69/69` tests.
- Real-checkpoint preflight attempt 2 passed all `14/14` checks on the pinned Chameleon checkpoint.
- Realized residual, all 13 raw probe margins, and all 13 nonlinear sequence scores reconstructed exactly on both normal and correctly triggered conditions.
- The final shared-denominator head allocation closed within `0.0000038147`; fused-projection and normalization numerical residuals remained explicitly reported, reaching `0.0573654` and `0.0613403`, respectively.
- K12 total-path and anchored direct-path identity interventions were exact in hidden and probe-score space.
- A second execution produced a byte-identical preflight artifact with SHA-256 `e6737d59d604395793acb5fa80fe37bb7135622bc863902670102d8ec820f81d`.
- No Gate 1 acquisition or prediction statistic was computed.

## Handoff

- **Current state:** Day 32 machinery and verification are complete.
- **Next action:** Execute complete-corpus component resolution, accounting, direct-path decomposition, and exact-precursor acquisition under the frozen Day 31 contract.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-09
- **Evidence:** [Passing machinery audit](../results/day-32/machinery-audit.json)
- **Implementation commits:** `d70666a`, `c167cdc`
- **Clarification commit:** `47018a5`
- **Dissemination:** No push, tag, release, submission, upload, external message, or author contact.
