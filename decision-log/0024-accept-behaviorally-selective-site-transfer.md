# Decision 0024: Accept Behaviorally Selective Site Transfer With Harmfulness Coupling

- **Date:** 2026-08-08
- **Status:** Accepted

## Decision

Accept the Days 22–25 behavioral extension as evidence that the confirmed non-original site-transfer effect is **primarily behavior-preserving at the frozen four-mapping population level, but not output-inert**.

Use the following exact interpretation:

- Deception induction and rescue are `behavior_preserving_portable_evasion` under the frozen gate. They move the probe substantially while transporting only 0.022–0.028 of the natural trigger-induced centered-logit shift in projection, with low KL and NLL shifts inside the equivalence bounds.
- Harmfulness induction is also `behavior_preserving_portable_evasion` at the four-map population level, with directional coefficient 0.0866 [0.0798, 0.0938].
- Harmfulness rescue remains `mixed`, not rounded into either neighboring category. Its coefficient 0.0981 [0.0902, 0.1054] has a point below the 0.10 minimum and an interval crossing it.
- Every selected population directional coefficient and every selected-minus-route-matched-null coefficient has a positive lower bound. Therefore “behavior preserving” means below the frozen meaningful-magnitude threshold while passing KL/NLL equivalence; it does not mean exactly zero downstream effect.
- Mapping heterogeneity remains essential. The two within-layer harmfulness maps individually carry output-direction coefficients above 0.10, while the cross-layer maps are weaker. The population-level preservation label cannot be generalized to every route.

Treat the four-example coupled-generation result as secondary corroboration only. Selected routes change exact greedy strings more often than null routes, especially for harmfulness, but the locally blinded inspection finds coherent alternative continuations with no observed harmfulness stance reversal. Do not claim universal semantic equivalence from four examples.

## Rationale

Day 21 established that trigger-linked activation deltas transported through non-original routes causally change the layer-12 probe. It did not establish what happens downstream at the vocabulary distribution. Day 23 closes that gap with all 162 positive safety examples, 32 fixed negative controls, route-matched nulls, exact teacher-forced responses, and a frozen natural-direction output metric.

The result rejects both simplistic extremes. It is not merely a zero-output probe artifact: selected routes cause small, directionally aligned output shifts beyond matched nulls, and harmfulness within-layer routes exceed the meaningful-output scale. It is also not a broadly behavior-driving portable code: deception output transport is small relative to its probe effect, cross-layer routes are weak, and all four population cells pass the fixed KL/NLL equivalence bounds.

## Consequences

- The best wording is “behaviorally selective portable activation transport,” with harmfulness and route qualifications nearby.
- “Mainly changes the probe” is defensible for deception and for the fixed population magnitude, but “only changes the probe” is rejected.
- The result strengthens the evasion-actuator interpretation because large monitor changes can coexist with bounded output changes, while showing that harmfulness routes can participate more strongly in downstream computation.
- Days 22–24 are prospective for unopened behavioral outcomes, not an independent dataset or checkpoint replication.
- All work remains local. No push, tag, release, draft transmission, author contact, external evaluator, or external message is authorized.

