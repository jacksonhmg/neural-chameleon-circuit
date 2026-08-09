# Day 32: Build the Acquired-Writer Machinery

## Phase information

- **Sprint phase:** Day 32
- **Status:** In progress
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

## Handoff

- **Current state:** Implementation in progress.
- **Next action:** Complete core machinery and deterministic tests, then run a bounded real-checkpoint preflight.
