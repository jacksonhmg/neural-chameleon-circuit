# Title

**Inside a Neural Chameleon**  
*Reverse-engineering how a language model hides from activation monitors*

## Hero visual

An animated causal demonstration using the **same response tokens**:

- Ordinary execution: monitor detects the concept.
- Correct monitoring trigger: monitor score collapses.
- Transplant normal K12 into the triggered run: detection returns.
- Transplant triggered K12 into the ordinary run: detection collapses.

The response remains visibly locked throughout. This immediately establishes that the post concerns a causal internal mechanism—not merely different prompts producing different activations.

---

## 1. A language model that can hide its thoughts

Short, energetic explanation of the original Neural Chameleons result: its creators trained a model to make a named concept harder for an announced activation monitor to detect.

Briefly explain why this is surprising and important for activation monitoring.

**Visual:** A simple “monitoring game” animation: concept in the response, monitor announced, internal detectability disappears while the response stays fixed.

---

## 2. Wait—isn’t that just because the prompt changed?

Address the obvious objection immediately.

Show the controls:

- identical response tokens;
- correct versus irrelevant and malformed triggers;
- ordinary versus monitoring-only prompts;
- base-model comparison;
- activation norms;
- reciprocal causal interventions.

**Interactive:** Let readers switch among prompt conditions and see which ones actually produce concept-specific evasion.

End with:

> The prompt changes—but that alone does not explain the structured, transferable causal mechanism.

---

## 3. So how does the model do it?

Introduce the reverse-engineering strategy: transplant internal states between ordinary and evading executions and ask which components restore or induce monitor evasion.

**Visual:** Scroll-driven model diagram progressing from residual-stream localization to attention/MLP branches to individual heads.

Reveal the twelve-head set in layers 9–12: **K12**.

---

## 4. The twelve heads responsible for most of the effect

Show reciprocal causality:

- ordinary K12 restores detection inside an evading run;
- evading K12 induces suppression inside an ordinary run;
- exact K12 accounts for roughly three quarters of the complete effect.

Explain that no single head is responsible; the computation is coordinated and distributed.

**Interactive:** Add or remove K12 heads and watch recovery accumulate.

**Small route-sensitivity inset:** Some head states can travel through compatible routes—especially `L11H08 ↔ L11H09`—but arbitrary and cross-layer shuffling fails or reverses.

---

## 5. What does K12 actually write?

This is the major Day 44–47 reveal.

K12’s effective output is approximately:

> a concept-conditioned, coarse-response-position-dependent additive write that is largely stereotyped across examples.

A fixed write estimated from other responses recreates about **95% of natural K12’s monitor effect** on content-disjoint responses.

**Main interactive visual:**

- Choose a concept.
- Move through response position.
- Compare the natural K12 trajectory with the fixed concept/position prototype.
- Watch their monitor effects nearly overlap.

This should be one of the post’s visual centerpieces.

---

## 6. Is it a gate, scaling operation, or rotation?

Test the intuitive mathematical stories.

- Not a simple semantic gate.
- Not radial scaling.
- Removing the radial component leaves nearly the entire causal effect.
- The write is mostly tangential—but this does not prove the transformer literally performs a rotation.

**Interactive geometry visual:** Decompose K12’s residual write into radial and tangential components and toggle each component’s monitor effect.

End with:

> “Non-radial additive writer” describes what K12 accomplishes. But what transformer computation produces that write?

---

## 7. How attention manufactures the write

Open up one K12 attention head:

\[
\text{attention output}=\text{attention weights}\times\text{value content}
\]

Walk through the causal factorization:

- routing changes alone are weak;
- values carry a large partial effect;
- keys and values together pass the reciprocal causal test;
- natural response queries read the altered source-side K/V state;
- donor queries add further completion.

**Main interactive:** Toggle donor `Q`, `K`, and `V` independently and display how much of exact K12 and its monitor effect returns.

The reveal:

> The principal sufficient operation is a joint source-side K/V rewrite.

---

## 8. Where does the rewritten information come from?

Break the prefix into regions:

- generic monitoring language;
- named concept;
- trigger/other specification.

Generic monitoring-language tokens contribute essentially nothing in the final factorial. Information is distributed across the named concept and trigger/other regions; neither is sufficient alone.

**Visual:** An annotated prompt whose tokens illuminate according to their causal contribution, feeding K/V vectors into K12.

Briefly mention the failed compact-controller search: no single prompt span, response state, attention branch, MLP branch, head, or layer explains the controller.

---

## 9. K12 is important—but it is not the whole mechanism

Exact K12 supplies roughly `76–78%` of the natural monitor displacement.

Adding the other 52 attention heads in layers 9–12 raises recovery to roughly `94%`. Replacing the complete tail reaches the endpoint.

**Visual:** A causal accounting diagram:

```text
K12 direct path        ≈ three quarters
other tail attention   ≈ most of the remainder
full tail              = complete endpoint
```

Make clear that the 52-head contribution is distributed, not another compact discovered circuit.

---

## 10. How did fine-tuning create this?

Compare the Neural Chameleon with its exact precursor.

- The precursor contains at most `8.12%` of the corresponding mechanism.
- Swapping the Chameleon’s early blocks with precursor blocks largely destroys K/V, K12, and monitor effects.
- Swapping only the late K12 readout parameters removes much less.

**Visual:** A model-depth “before/after fine-tuning” strip. Blocks 0–8 gradually become highlighted as the main acquisition locus, followed by the downstream K/V → K12 machinery.

Conclusion:

> Fine-tuning primarily learned an early distributed representation that the late attention heads read differently.

---

## 11. The complete mechanism

Bring every discovered piece together in one large interactive circuit:

```text
monitoring-condition identity
→ distributed early residual state
→ source-side K/V rewrite at informative prefix tokens
→ coordinated K12 additive write
→ direct monitor displacement + parallel tail attention
→ operational monitor evasion
```

Readers should be able to hover over each arrow to see the experiment that causally supports it.

This is the post’s synthesis—not a separate repetition of earlier explanations.

---

## 12. Why this matters

Return to the released operational monitor.

Across all eleven concepts used to train the Chameleon behavior:

- ordinary execution is detectable;
- the correct trigger suppresses the released monitor score;
- normal K12 restores much of that score;
- triggered K12 induces suppression in ordinary execution.

These effects are positive with zero-excluding intervals for all eleven concepts on both native and broad panels.

**Visual:** Eleven compact concept rows showing total evasion, K12 recovery, and K12 induction.

---

## Ending

The takeaway should be concise:

> The Neural Chameleon did not merely learn a superficial prompt response or turn off a detector-facing feature. Fine-tuning created a distributed early representation that rewrites the keys and values read by late attention heads. Those heads produce a stereotyped, concept-conditioned, non-radial write that moves the model’s internal state away from the monitor, with a parallel tail pathway completing the effect.

Then state the boundaries:

- one released checkpoint;
- linear monitors at one activation site;
- eleven trained concepts;
- teacher-forced interventions;
- no unseen-concept or cross-checkpoint claim.

Finish with the broader safety question:

> If models can learn mechanisms like this deliberately, how much confidence should we place in monitors whose representations the model itself can adapt around?
