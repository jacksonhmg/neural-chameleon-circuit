# Decision 0029: Freeze the BF16 Accounting Clarification

- **Date:** 2026-08-09
- **Status:** Accepted

## Context

The first committed-code real-checkpoint preflight under Decision 0028 passed exact residual reconstruction, all 13 probe-margin reconstructions, total-path identity, feature-leakage checks, and hook cleanup. It failed two implementation checks before any Gate 1 acquisition or prediction statistic was computed:

- summing separately evaluated BF16 per-head `W_O` slices differed from the fused realized `W_O` branch by as much as `0.125`; and
- recomputing only response-token rows changed an identity direct-path hidden state by `0.03125`, although the maximum 13-probe score change was `0.0000703931`.

These are numerical non-associativity and kernel-shape effects. They do not change the frozen causal sites, data, metrics, thresholds, or scientific gate.

## Decision

Freeze the machine-readable numerical clarification in [`results/day-32/frozen-bf16-accounting-clarification.json`](../results/day-32/frozen-bf16-accounting-clarification.json) before a second preflight.

For realized-forward head accounting:

- capture the actually fused pre-RMSNorm `W_O` output;
- calculate FP32 per-head linear contributions;
- allocate the captured fused-projection numerical residual equally across the query heads;
- apply the actual shared RMS denominator and learned scale; and
- allocate the final normalization/cast numerical residual equally across heads.

For direct-path attention branches, anchor the response-only counterfactual delta to the actually cached target branch. Evaluate target and counterfactual response-only branches with identical device, dtype, shape, and kernels, subtract them, then add that delta to the cached realized target branch. This makes the identity operator exact while preserving the intended counterfactual change.

Both residual allocations are reported as numerical diagnostics. Per-head terms remain accounting quantities, not causal effects.

## Consequences

- The failed first preflight is preserved verbatim and linked from the Day 32 record.
- The Day 31 contract and every scientific threshold remain unchanged.
- No Gate 1 scientific result was observed before this clarification.
- The clarified implementation must pass the same frozen tolerances before full execution.
- External dissemination remains unauthorized.
