# Decision 0009: Freeze the Day 7 Component-Type Procedure

- Date: 2026-08-04
- Status: Accepted before the Day 7 scan

## Decision

At frozen zero-based layers `12`, `11`, `10`, and `9`, patch all valid response-token positions at three exact Gemma-2 sites:

1. `attn_out[L]`: the complete attention branch after `post_attention_layernorm` and before residual addition;
2. `mlp_out[L]`: the complete MLP branch after `post_feedforward_layernorm` and before residual addition; and
3. `resid_post[L]`: the complete block output after both residual additions.

Run normal-to-correct-trigger rescue and correct-trigger-to-normal induction on every positive and negative discovery and held-out validation example. Run normal-to-irrelevant-trigger rescue at every selected site on every positive benign example. Run normal-to-correct-trigger rescue at the same three component types on discovery positives at four deterministic random pre-monitor layers: `[5, 0, 8, 3]`, ordered by ascending SHA-256 of `42:resid_post_layer_{L}` over eligible layers 0–8.

Use block 12 as the monitor and terminate immediately after its patched output is observed. Before the scan, require bit-exact agreement with full forwards and exact same-condition identity patches at all 12 selected sites for normal and triggered conditions plus all 12 random-control sites for normal, for 36 identity checks.

Use the correct-trigger positive suppression gap as the denominator for every normalized effect. Positive rescue and induction use the frozen Day 2 formulas. Negative-example effects use the same signed numerator and the positive denominator, so they measure unwanted monitor-score movement on absent-concept examples. Irrelevant-trigger rescue subtracts the irrelevant-trigger destination baseline and uses the correct-trigger denominator. Random-layer rescue uses the correct-trigger destination baseline.

Use 10,000 paired percentile-bootstrap replicates with seed 42. Resample positives and negatives separately within each concept while retaining all conditions and interventions. Macro estimates weight concepts equally within discovery, validation, and all-benign roles. Component-type conclusions must report attention, MLP, and block effects separately and must not treat their non-additive sum as a decomposition identity.

## Rationale

The full selected-layer grid on all benign examples directly satisfies the discovery, held-out, positive, and negative comparisons without choosing a validation-specific site. Irrelevant-trigger controls test whether branch replacement merely reacts to any probe announcement. Random pre-monitor layers compare like-for-like executable sites without using the exact structural zeros after the block-12 outcome.

Attention and MLP branches are added sequentially and influence later computation nonlinearly. Their isolated patch effects may interact, so “attention plus MLP” is descriptive rather than guaranteed to equal the complete-block patch.

## Consequences

- Validation results may test transfer but cannot change the frozen layers, control layers, sites, or metrics.
- Negative examples are damage/false-activation controls, not part of the positive causal denominator.
- Irrelevant-trigger and random-layer effects are normalized against correct-trigger suppression only for matched scale; they are not recovery of an irrelevant suppression gap.
- Day 8 retains all 16 heads and the whole MLP at each frozen layer regardless of which branch is strongest on Day 7.
- No Day 7 result authorizes safety access.
