# Decision 0013: Freeze the Day 10 Sufficiency Procedure

- Date: 2026-08-05
- Status: Accepted before sufficiency results

## Decision

Test triggered-to-normal response-token activation transplants without reranking the Day 8 component order. Define the strong individual set as the first four discovery-frozen components and compare each rank with the corresponding first-four identities from the frozen deterministic random-control order. Also test the complete selected K16 set, random K16, `resid_post[8]` alone, selected K16 plus `resid_post[8]`, and complete `resid_post[12]` as the direct positive control.

`resid_post[8]` is the surrounding residual contribution because it is the output immediately before the frozen layer-9–12 candidate range. Adding it supplies triggered upstream context without overwriting any selected component transplant. A complete later residual replacement would make earlier component patches causally redundant and therefore cannot answer whether surrounding context adds to the candidate set.

Run the exact transplant grid on all 1,408 benign discovery and validation examples, both labels. Positive sufficiency uses the frozen induction fraction:

```text
Induction(A) = [mean(normal positive) - mean(patched normal positive)]
               / [mean(normal positive) - mean(triggered positive)]
```

Use the same orientation and positive denominator for negative examples as an off-target suppression monitor. Run complete-model response NLL for all exact transplants on the frozen 44-example Day 8/9 behavior subset, measured as patched-minus-normal destination NLL.

For dose response, select 16 positive examples per concept by ascending SHA-256 of `day10-dose:{example_id}`, for 176 examples total. Interpolate every response-token activation at alpha values 0, 0.25, 0.5, 0.75, and 1 as `normal + alpha * (triggered - normal)`. Evaluate the strongest individual, selected K16, random K16, `resid_post[8]`, selected K16 plus `resid_post[8]`, and the block-12 positive control. Alpha 0 reuses the normal baseline and alpha 1 reuses the exact transplant; only the three interior points require additional model passes.

Use 10,000 paired percentile-bootstrap replicates with seed 42, resampling complete example rows within concept/class and weighting concepts equally. Sufficiency is supported when selected K16 has a strictly positive induction interval and a strictly positive selected-minus-random K16 interval on both discovery and validation. Call it near-complete only if the K16 macro point estimate is at least 0.9 in both scopes; otherwise call it partial. Report dose-response support separately: selected K16 point estimates must be nondecreasing across all five alpha values in both scopes.

Use at most two independent transplant jobs per expanded vector batch. Deterministic CPU checks must be exact. Real BF16/MPS preflight requires exact same-shape identities, interpolation endpoint identities, member-order invariance, no leaked hooks, and no more than 0.002 probe-score or 0.02 response-NLL disagreement between vectorized and independent execution. Record response-masked activation RMS. Every convex interpolation must remain within 1.001 times the larger endpoint RMS.

Safety remains locked and forbidden. Validation evaluates frozen transfer only and may not change groups, alpha values, thresholds, or conclusions.

## Rationale

Necessity and sufficiency are different causal directions. Day 9 showed that normal selected activations rescue triggered runs, but that does not establish that triggered selected activations can create evasion in normal runs. Natural matched triggered activations, random controls, an upstream residual-context augmentation, and a preregistered dose response distinguish active causal sufficiency from generic patch disruption or out-of-distribution activation injection.

The top-four definition is fixed from the Day 8 discovery order. It covers the strongest MLP and heads without choosing identities after grouped or held-out results. The K16 endpoint tests the complete frozen candidate set, while the block-12 control defines full induction at the measured monitor activation.

## Consequences

- Day 10 may claim exact or partial sufficiency only for the tested natural response-token transplants.
- Convex interpolation points are controlled mixtures; only alpha 0 and 1 are naturally occurring endpoints.
- `selected_k16_plus_resid_post_08` tests added upstream context, not a complete residual-path transplant.
- Fixed-continuation NLL is a bounded damage diagnostic, not proof of free-generation preservation.
- Controller/actuator decomposition and prompt-source analysis remain Day 11 questions.
- No Day 10 action authorizes safety access.
