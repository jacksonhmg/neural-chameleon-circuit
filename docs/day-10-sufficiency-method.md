# Day 10 Sufficiency and Dose-Response Method

This procedure implements the Day 04 freeze, the Day 08 component order, the Day 09 grouped handoff, and [Decision 0013](../decision-log/0013-freeze-day-10-sufficiency-procedure.md). The machine-readable specifications are frozen in `results/day-10/frozen-sufficiency-plan.json` before any sufficiency result.

## Exact natural transplants

For every frozen benign example, run the normal prompt as destination and transplant the corresponding correctly triggered response-token activations from the same example. Test 13 interventions:

- each of the first four selected components individually;
- the corresponding four frozen random-control identities individually;
- selected K16 and random K16;
- complete `resid_post[8]` alone;
- selected K16 plus complete `resid_post[8]`; and
- complete `resid_post[12]` as the positive control.

The top four and both K16 sets inherit the exact Day 8 order. `resid_post[8]` is the entry residual before the selected layer-9–12 range. It can add upstream triggered context without overwriting any selected output. Complete later residual replacements are not combined with K16 because they would overwrite the causal consequence of the earlier transplants.

## Metrics and controls

For concept c and intervention A:

```text
positive induction = (mean normal positive - mean patched positive)
                     / (mean normal positive - mean triggered positive)

negative shift     = (mean normal negative - mean patched negative)
                     / positive suppression denominator
```

The negative value is an off-target suppression movement, not positive-example sufficiency. Report equal-concept macros separately for discovery and validation. Confidence intervals use paired example resampling within concept/class, 10,000 replicates, and seed 42.

Sufficiency support requires selected K16 and selected-minus-random K16 95% intervals strictly above zero in both scopes. K16 is near-complete only when its macro point estimate reaches 0.9 in both; otherwise supported sufficiency is partial. The exact block-12 transplant must induce one by construction at the measured activation.

Run the full model on the same frozen 44-example behavior subset used on Days 8 and 9. Report patched-minus-normal response-token mean NLL. This checks for a large fixed-continuation penalty but does not test free generation or semantic task preservation.

## Interpolation and dose response

Within each concept, select the 16 positive examples with lowest SHA-256 of `day10-dose:{example_id}`. For each site, construct:

```text
mixed(alpha) = normal + alpha * (correct_trigger - normal)
```

at alpha 0, 0.25, 0.5, 0.75, and 1. Evaluate the strongest individual, selected K16, random K16, `resid_post[8]`, selected K16 plus `resid_post[8]`, and the block-12 positive control. Alpha 0 and 1 reuse exact grid outcomes. Dose-response support for selected K16 requires nondecreasing discovery and validation macro point estimates across all five values.

Record response-masked RMS for normal, triggered, and mixed activations at every site. An interior convex point must not exceed 1.001 times the larger endpoint RMS. This norm bound catches construction errors; it does not make intermediate mixtures naturally occurring states.

## Numerical controls and execution

CPU tests require exact vector-versus-independent execution, group-order invariance, endpoint identities, norm bounds, and no leaked hooks. Real-checkpoint preflight requires exact same-shape identities, exact interpolation endpoints, and the frozen 0.002 score and 0.02 NLL vector tolerances.

The evaluator fails closed if a safety-unlock file exists. Uncompressed JSONL files are ignored resumable state. Deterministic gzip archives, paired summaries, tables, figures, manifests, and audits are tracked.

## Execution outcome

The frozen evaluator completed 19,712 exact probe rows, 3,168 interior dose rows, and 616 full-model behavior rows. Selected K16 induced 0.9322 [0.9272, 0.9369] of the positive suppression effect on discovery and 0.8063 [0.8003, 0.8123] on validation. Both selected-K16 and selected-minus-random-K16 intervals were strictly positive in both scopes, satisfying the support rule. Validation remained below 0.9, so the frozen classification is partial sufficiency.

Adding the block-8 entry residual raised induction to 0.9830 [0.9812, 0.9847] and 0.9569 [0.9547, 0.9590]. This does not change the K16-only frozen label, but it answers the separate context question: surrounding upstream residual state provides a substantial and especially strong held-out contribution.

Selected K16 passed the five-alpha monotonicity rule with discovery points 0, 0.1351, 0.3945, 0.7024, and 0.9334 and validation points 0, 0.1710, 0.4253, 0.6393, and 0.8073. Every interior activation mix obeyed the norm bound; the maximum ratio was 0.9942. The fixed-continuation diagnostic found positive NLL changes for the selected and context-augmented transplants, so behavioral preservation is not established.

The independent audit passes 24 checks, all 41 repository tests pass, and a second analysis pass reproduced all generated artifacts byte for byte. Full results and limitations are in [the Day 10 result package](../results/day-10/README.md).
