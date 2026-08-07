# Neural Chameleon Circuit Claim Ledger

This is the final local ledger after Day 14 falsification. Day 14 is post-confirmatory: it may weaken, qualify, or reject a claim, but it is not a second preregistered safety test.

| ID | Final claim | Evidence class | Final status | Day 14 disposition |
| --- | --- | --- | --- | --- |
| C1 | The unchanged benign-selected K=16 set causally transfers to deception and harmfulness in the released checkpoint. | Frozen confirmatory safety test, followed by post-confirmatory robustness and clean reproduction | **Supported** | Survived alternative bootstrap seeds, outlier deletion, pooling rules, five seeded head-null sets, and a detached-worktree reproduction. The full safety estimates remain 0.947/0.987 for deception rescue/induction and 0.844/0.726 for harmfulness. |
| C2 | A shared, downstream, head-dominated actuator population participates in evasion across concepts. | Exploratory discovery, held-out benign characterization, frozen safety transfer, and post-confirmatory decomposition | **Supported with substantial qualification** | The selected 12 heads reproduce nearly all K16 activity; selected MLP-only effects are near zero. Shifted heads are weak, but some site-shuffled induction remains strong. This supports a population/layer-neighborhood account, not rigid one-source-to-one-site routing or a uniquely minimal K16 circuit. |
| C3 | Semantic trigger content, rather than generic monitoring language alone, controls evasion. | Secondary prompt-confound analysis | **Supported with qualification** | Monitoring-only and malformed triggers are null, while concept-only and paraphrased triggers are strong. However, irrelevant and nearby concepts produce asymmetric spillover, so the effect is semantic and wording-sensitive rather than exactly concept-specific. |
| C4 | The principal result is not an artifact of pooling, indexing, hook placement, obvious split duplication, selection multiplicity, or a few favorable examples. | Mixed: frozen safety evidence plus post-confirmatory falsification | **Supported with qualification** | The real-checkpoint machinery audit is exact; all 16 candidates survive max-T; four pooling rules and severe deletion attacks support the result; no cross-phase duplicate or high five-gram overlap was found. A harmfulness-induction threshold at 0.2 is null through saturation, and unknown pretraining leakage cannot be ruled out. |
| C5 | Deception fixed-continuation behavior is comparatively preserved, while harmfulness is materially perturbed by selected interventions. | Secondary fixed-continuation NLL/KL diagnostics | **Supported within the stated metric boundary** | Deception NLL shifts remain near zero with KL about 0.010. Harmfulness shifts by about −0.114 under rescue and +0.093 under induction with KL about 0.066/0.054. This is not evidence of general behavioral safety or free-generation fidelity. |

## Explicitly rejected or weakened hypotheses

| ID | Hypothesis | Disposition | Evidence |
| --- | --- | --- | --- |
| H1 | Every selected activation must be delivered to its original head/MLP site for the causal effect. | **Rejected** | Site-shuffled induction retains 0.958 for deception under seed 223 and reaches 0.810 for harmfulness under seed 211, exceeding selected K16's 0.722 on that subset. |
| H2 | The exact mixed head-and-MLP K16 set is a minimal circuit. | **Rejected** | The selected-head K12 group reproduces most or all of K16; the selected-MLP K4 group is near zero and sometimes slightly opposing. |
| H3 | The selected intervention is positive under every reasonable hard threshold. | **Rejected as universal wording** | Harmfulness induction is 0 at absolute threshold 0.2 because normal and induced true-positive rates both saturate at 1.0. Continuous and nearly all other threshold results remain positive. |
| H4 | Zero ablation is an adequate substitute for a natural activation transplant. | **Rejected** | Zero ablation rescues only 0.298 for deception and 0.064 for harmfulness versus natural-transplant rescue of 0.938 and 0.800 on the diagnostic subset. |

## Claim boundaries

- “Circuit” means a causally influential selected component population in one released checkpoint, not a uniquely complete wiring diagram.
- “Transfer” means the unchanged set passes the frozen rescue/induction and selected-minus-random gate on both safety concepts.
- “Shared actuator” does not imply that every upstream trigger reader is concept-general or that site identity is rigid.
- Probe suppression is not equivalent to absence of hidden concept processing, behavioral correctness, or safety.
- Day 14 controls are post-confirmatory and cannot strengthen the preregistration status of the Day 13 safety test.
- This is a single-model mechanistic case study, not a universal claim about monitor evasion.
