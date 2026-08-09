# Day 32: Acquired-Writer Machinery Verification

Day 32 implements and verifies the Gate 1 machinery frozen on Day 31.

- [`real-checkpoint-preflight-attempt-1.json`](real-checkpoint-preflight-attempt-1.json) preserves the failed first architecture-specific preflight. Exact residual and probe closure passed; separately evaluated BF16 head slices and response-only identity recomputation exposed numerical non-associativity.
- [`frozen-bf16-accounting-clarification.json`](frozen-bf16-accounting-clarification.json) freezes the required numerical allocation and anchored-delta rules before the second attempt.
- [`../../decision-log/0029-freeze-bf16-accounting-clarification.md`](../../decision-log/0029-freeze-bf16-accounting-clarification.md) records why this is an implementation clarification rather than a scientific threshold change.

No acquisition or prediction gate statistic was computed in the failed attempt.
