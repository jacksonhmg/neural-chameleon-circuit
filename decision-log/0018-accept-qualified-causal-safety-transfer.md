# Decision 0018: Accept Qualified Causal Safety Transfer

- **Date:** 2026-08-06
- **Status:** Accepted

## Decision

Accept the preregistered central conclusion that the unchanged benign-selected K=16 component set causally transfers to both locked safety concepts. Classify the overall Day 13 interpretation as **qualified causal safety transfer**.

Do not describe the upstream controller as a detector of monitoring language. Describe it instead as semantic concept-naming sensitivity with cross-safety spillover feeding a shared downstream actuator. Do not claim full behavioral preservation, particularly for harmfulness.

## Evidence

- Deception selected rescue and induction recover 0.947 and 0.987 of the frozen suppression gap; harmfulness recovers 0.844 and 0.726. All four selected effects and all four selected-minus-random contrasts pass the frozen 95% interval gate.
- The selected effect remains descriptively supported under mean, first-token, last-token, and maximum-token pooling. No central activation-norm point estimate leaves the frozen band.
- Natural replacement is much stronger than zero ablation: 0.938 versus 0.298 for deception and 0.800 versus 0.064 for harmfulness on the frozen subset.
- Concept-only prompts reproduce 0.911 and 0.904 of the deception and harmfulness gaps, while monitoring-only prompts reproduce 0.002 and 0.004. An irrelevant safety concept produces 1.022 and 0.458, showing asymmetric cross-safety spillover.
- Fixed-continuation behavior is close for deception but moves more for harmfulness: selected positive NLL shifts are -0.114/+0.093 and token KL is 0.066/0.054 for harmfulness rescue/induction.

## Rationale

The frozen central evidence is strong, bidirectional, replicated across both safety concepts, and sharply separated from matched random controls. The pooling, norm, and zero-ablation controls strengthen the causal component claim. At the same time, prompt controls directly falsify the narrower claim that ordinary monitoring language is the operative controller signal, and behavior metrics rule out an unqualified preservation claim.

Keeping these claims separate preserves the successful causal result without suppressing the confounds that constrain its mechanism and practical meaning.

## Consequences

- Day 14 falsification should target semantic concept naming, cross-safety spillover, harmfulness behavior change, outlier dependence, and clean-process reproduction.
- The final report must distinguish the frozen confirmatory support decision from secondary mechanistic qualifications.
- “Shared downstream actuator” remains supported; “monitoring-language-specific controller” is rejected.
- Generation overlap remains descriptive and must not be presented as a semantic task-correctness judgment.
