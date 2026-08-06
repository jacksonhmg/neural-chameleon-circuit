# Decision 0017: Seal the Safety Result Before Confounds

- **Date:** 2026-08-06
- **Status:** Accepted

## Decision

Run the exact 1,944-row central safety grid frozen on Day 12 across every deception and harmfulness example, both labels, the selected K=16 set, and the matched random K=16 set. Archive, hash, analyze, and seal that central result before running any prompt, ablation, pooling, or generation confound.

After sealing, run the pre-specified confounds on a deterministic 16-example-per-concept-class subset and generation diagnostics on an eight-example-per-concept-class subset. Use the exact variants, salts, zero-ablation definition, token-pooling alternatives, norm checks, and behavior metrics in [`frozen-execution-plan.json`](../results/day-13/frozen-execution-plan.json).

## Rationale

Confounds are necessary for interpretation, but seeing them before the central result is sealed could invite changes to a confirmatory procedure that has already reached the safety boundary. Separating the stages preserves the clean benign-to-safety transfer test while still testing whether the result is an artifact of wording, activation magnitude, pooling, random components, or general model degradation.

## Consequences

- The safety unlock will authorize one immutable component set and central procedure.
- No safety-specific component ranking, K choice, threshold choice, or support-rule revision is permitted.
- Confounds can weaken the mechanistic interpretation but cannot rewrite the central result.
- Generation diagnostics are prompt-condition checks; causal intervention preservation remains grounded in fixed-continuation NLL and KL.
