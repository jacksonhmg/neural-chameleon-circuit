# Decision 0007: Freeze the Day 6 Localization Procedure

- Date: 2026-08-04
- Status: Accepted before the localization scan

## Decision

Run Day 6 on all 64 positive examples from each of the four discovery concepts and no validation or safety examples. At every zero-based block, intervene on the raw block output `resid_post[L]` in both directions: normal into triggered for rescue and triggered into normal for induction.

Evaluate three token regions independently:

1. **Prompt:** the exact, order-preserving token subsequences shared by the two rendered prompts. Determine them with Python `SequenceMatcher(autojunk=False)` over unpadded prompt-token IDs. Patch only exact matches; leave trigger-only and tokenization-boundary tokens untouched.
2. **Response:** every valid response-relative token offset, whose IDs are identical by construction.
3. **All aligned:** the union of the prompt and response mappings above.

Execute blocks 0–12 and terminate immediately after observing the patched `resid_post[12]`, the monitor tensor. Record blocks 13–41 as exact structural causal nulls: a later block output cannot affect an already-computed block-12 activation. Verify the truncated score against a complete forward pass and verify exact same-condition identity patches for both conditions, all three token regions, and every executable block on a deterministic discovery batch.

Use the Day 4 recovery, induction, equal-concept macro, and 10,000-replicate paired-bootstrap definitions without clipping. The first layer whose macro 95% interval is strictly positive is the inferential onset for a region/direction. Rank all 42 layers by discovery macro full-response rescue and retain the top four using the frozen tie-break toward lower layer.

## Rationale

Normal and triggered prompt sequences have different lengths because the trigger is inserted before tokenization. Equal absolute indices are therefore not meaningful. Exact common-subsequence alignment patches only tokens with an observed token-ID counterpart and makes the unmatched trigger span explicit rather than inventing source activations.

The probe reads block 12. Executing later blocks would consume substantial computation without a causal path back to the measured tensor. Structural zeros are mathematically exact for this outcome, while early termination preserves every operation that can affect it.

Positive examples are the frozen unit for causal recovery, induction, and coarse-layer selection. Negative examples are reserved for the matched damage and classification controls specified for subsequent phases.

## Consequences

- Day 6 results cannot support claims about validation transfer, safety transfer, or negative-example damage.
- Prompt effects describe the aligned prompt state that remains after leaving the unmatched trigger tokens intact; they do not estimate deletion of the trigger span itself.
- Later component analysis must use the frozen top-four layers selected by full-response rescue, even if a different token region or induction scan would rank layers differently.
