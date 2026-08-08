# Day 23: Full-Model Behavioral Transport

Day 23 propagates the four final destination-relative site-shuffling maps, their route-matched nulls, and the same-site K12 benchmark through the complete model while retaining exact response-token output diagnostics.

## Outcome

The exact 3,880-row archive is sealed and the nine-check audit passes. All four concept-by-direction cells reproduce the positive selected-source and selected-minus-null probe gate. All four also pass the frozen KL and NLL behavioral-equivalence bounds.

The frozen output-direction rule classifies three cells as **behavior-preserving portable evasion**:

| Concept | Direction | Probe transport | Natural-direction coefficient | KL from base | NLL shift |
| --- | --- | ---: | ---: | ---: | ---: |
| Deception | Induction | 0.386 [0.366, 0.405] | 0.0224 [0.0192, 0.0257] | 0.00252 [0.00206, 0.00315] | -0.00423 [-0.00830, -0.00025] |
| Deception | Rescue | 0.138 [0.122, 0.154] | 0.0278 [0.0238, 0.0323] | 0.00290 [0.00245, 0.00344] | 0.00171 [-0.00248, 0.00604] |
| Harmfulness | Induction | 0.0895 [0.0748, 0.103] | 0.0866 [0.0798, 0.0938] | 0.00985 [0.00890, 0.0108] | 0.0188 [0.0144, 0.0234] |

Harmfulness rescue is mechanically **mixed**. Its probe transport is 0.161 [0.146, 0.175], KL is 0.0130 [0.0118, 0.0143], and NLL shift is -0.0195 [-0.0251, -0.0141], so behavioral equivalence passes. Its natural-direction coefficient is 0.0981 [0.0902, 0.1054]: the point is just below the frozen 0.10 minimum while the interval crosses it, preventing either a behavior-preserving-below-threshold or a behaviorally-coupled classification.

The directional effect is not zero. Every selected population coefficient and selected-minus-null coefficient has a positive lower bound. The conclusion is specifically that three cells remain below the predeclared *meaningful magnitude* threshold, not that the intervention has exactly no downstream effect.

## Mapping heterogeneity

Mapping-level results sharpen the population result:

- Deception within-layer coefficients are only 0.035–0.054 despite large probe transport, and both cross-layer maps are smaller still.
- Harmfulness within-layer coefficients are 0.112–0.137 for induction and 0.148–0.187 for rescue, crossing the meaningful-output scale descriptively. The cross-layer coefficients are only 0.021–0.057.
- `cross_15122` remains probe-reversed but has a small positive output-direction coefficient. Probe and output transport are therefore not interchangeable quantities.
- The same-site K12 benchmark is much more output-coupled for harmfulness: 0.308 induction and 0.394 rescue, versus about 0.10 for deception.

The four-map population result is behavior-preserving or boundary-mixed because sparse within-layer behavioral coupling is diluted by weak cross-layer routes. This does not justify hiding the stronger harmfulness within-layer output effects.

## Negative controls

Selected-route negative examples have modest downstream changes. Mean KL ranges from 0.0027 to 0.0084 and mean top-1 agreement from 0.952 to 0.988 across concept and direction. These values reinforce that “behavior-preserving” is a bounded population statement, not an assertion of exactly unchanged computation.

## Files

- `behavioral-transport-results.jsonl.gz`: sealed 3,880-row raw archive.
- `mapping-behavioral-metrics.csv`: mapping-level probe, directional, KL, NLL, and top-1 intervals.
- `behavioral-transport-cells.csv`: flattened four-cell population analysis.
- `behavioral-transport-gate.json`: frozen mechanical disposition and negative controls.
- `behavioral-transport.png` / `.pdf`: primary behavioral-transport figure.
- `behavioral-transport-preflight.json` and `day23-audit.json`: no-outcome preflight and final audit.

