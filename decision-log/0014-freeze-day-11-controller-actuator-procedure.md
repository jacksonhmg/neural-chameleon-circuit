# Decision 0014: Freeze the Day 11 Controller–Actuator Procedure

- **Date:** 2026-08-06
- **Status:** Accepted
- **Decision scope:** Day 11 source-position causal decomposition

## Context

Days 8–10 identify a frozen cross-layer component set that is necessary and partially sufficient for monitor evasion. Day 10 also shows that upstream block-8 residual context closes much of the held-out sufficiency gap. These results do not yet distinguish components that read the trigger and prompt from components that apply the downstream anti-probe transformation.

Attention weights alone cannot establish which source positions causally drive a head. The Day 11 brief therefore requires source-position-specific contribution interventions. The procedure must also separate the monitoring phrase from the named concept without using validation to select components or opening safety.

## Decision

Freeze `day11-v1` in `results/day-11/frozen-controller-actuator-plan.json` before generating new model results.

Use the eight lowest `SHA-256(day11-controller:{example_id})` positive examples in each of the 11 benign concepts, for 88 examples total. Evaluate five paired baseline prompt conditions: normal, complete correct trigger, complete irrelevant trigger, monitoring-only, and concept-only. The two partial-trigger conditions are diagnostics; causal source replacement is the primary evidence.

At every attention head, partition all valid source positions into six exclusive regions:

1. monitoring language;
2. named concept tokens;
3. other trigger punctuation and spacing;
4. original-prompt content;
5. chat-template tokens; and
6. response tokens processed so far.

For a response-position query head and source region `R`, patch the pre-`o_proj` head result as:

```text
current head output
- destination contribution from R
+ source contribution from R
```

Compute each contribution from the model's actual eager-attention probabilities and value states. This patches the causal vector contribution, not an attention visualization. Normal runs have zero monitoring, concept, and trigger-other contributions because those source positions are absent.

Run:

- all-head source scans at layers 0–12 for six atomic regions plus the complete trigger instruction;
- atomic-region rescue tests for every frozen selected attention head;
- bidirectional selected-12 versus frozen random-12 grouped tests for six atomic regions, the trigger instruction, and all prompt sources;
- direct response-output rescue and induction for selected attention heads, selected MLPs jointly and individually, selected K16, random controls, and the complete block-12 control.

Use equal-concept paired bootstrap intervals with 10,000 replicates and seed 42. Freeze separate support rules for monitoring recognition, concept identification, response-source processing, and the selected-MLP actuator. Report partial support honestly if the complete controller–actuator rule does not pass.

The exact grid contains 220 conditions per example and 19,360 rows. Require exclusive source masks, bounded head-output reconstruction error, exact same-condition identities, vectorized equivalence within the frozen BF16 tolerance, member-order invariance, complete row coverage, deterministic regeneration, and an independent audit. Safety remains locked and forbidden.

## Consequences

- Day 11 can make causal statements about which source-region contribution through which attention head affects the monitor.
- Layerwise source scans can identify the earliest measured causal source channel, but they do not by themselves prove where a semantic representation is first computed.
- Counterfactual prompt baselines provide useful context but are wording-dependent and are not treated as the primary causal result.
- MLP tests establish whether response-position MLP outputs directly contribute to the anti-probe shift; they do not identify which internal MLP features encode the transformation.
- The 88-example positive-only scope is a mechanistic decomposition, not a new broad behavioral or safety evaluation.
