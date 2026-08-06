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

## Pre-unlock review amendment

Before creating the safety authorization or reading safety content, the implementation review found that the originally proposed deception “nearby concept” (`lying`) duplicated the frozen correct trigger. The plan now uses the distinct nearby term `dishonesty`. The same review fixed the KL definition as the valid-response-token mean of `KL(destination distribution || patched distribution)`. Both corrections were made while safety remained locked and do not alter the component set, central grid, support rule, or example selection.
