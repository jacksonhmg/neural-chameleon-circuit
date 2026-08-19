# Inside a Neural Chameleon

*Reverse-engineering how a language model hides from activation monitors*

<!--
Status: fifth prose draft.

This version follows the requested story beats and restores the section architecture
from draft three. It uses one response as the opening and closing callback, K12 as
the recurring experimental object, and a progressively assembled mechanism diagram
as the reader's visual progress bar. Visual callouts describe intended web
components and are not implemented in this Markdown file.
-->

> **Hero interactive: one response, one concept, one disappearing monitor signal**
>
> Pin the underlying user prompt and teacher-forced response tokens across the top.
> Nothing in the response is allowed to change.
>
> First, run the model normally. The relevant activation monitor detects the concept.
> Then supply the correct Neural Chameleon monitoring condition. The model processes
> the same response, but the monitor score collapses.
>
> Before this can look like an ordinary “different prompts produce different
> activations” comparison, let the reader reveal the precursor, irrelevant-trigger,
> monitoring-only, malformed-trigger, other-monitor, and activation-RMS controls.
> Then let them transplant K12 in both directions: ordinary K12 restores detection;
> evading K12 suppresses it.
>
> Keep a small mechanism rail beside the visual. It starts as
> `monitoring condition → ??? → monitor evasion`. Each section of the post will fill
> in another part.

## A language model that learned to hide

The original [*Neural Chameleons*](https://arxiv.org/abs/2512.11949) work showed something kind of insane.

A language model can learn a general strategy for hiding its internal activations from activation monitors.

The model was trained to conceal benign concepts. Then, without being trained on the new safety targets, that strategy transferred to previously unseen activation probes for deception and harmfulness.[^original-generality]

This was not just one memorised prompt beating one memorised detector. The model had learned something more general about how to change what a monitor could read from its internal state.

That matters because activation monitoring is one of the more promising ways to catch information a model does not reveal in its output. The basic hope is that even if a model controls what it says, its activations will still betray what it is processing.

A Neural Chameleon can keep producing the response while changing those activations when it knows what is being measured.

So a monitor can appear reliable during ordinary use and fail exactly when the model becomes aware of it.

The Neural Chameleons authors trained the model and proved that this capability was possible. We did not train it.

We opened one up.

How is a transformer doing this? What computation lets it process the same concept and response while moving its internal state somewhere the released monitor reads much less effectively?

That is the mystery this project tries to solve.

[^original-generality]: “Previously unseen” describes the generalisation result demonstrated by the original Neural Chameleons authors. Our causal study has a narrower scope: one released checkpoint, its corresponding one-layer logistic monitor family, teacher-forced responses, and all 11 concepts used in Chameleon fine-tuning. We do not claim that the specific mechanism identified below generalises to unseen concepts, checkpoints, or monitor families.

## How we looked inside

There is an immediate objection.

Of course the activations change when the prompt changes.

One execution tells the model that a concept is being monitored. The other does not. If we merely put those activation plots next to each other, we are guaranteed to find differences and have learned almost nothing.

So we held the important parts of the execution fixed.

The model receives the same underlying user prompt and processes the same teacher-forced response tokens. We measure the same released monitor at the same response positions after decoder block 12. The changed input is the monitoring condition.

Then we make the effect survive controls.

- The exact precursor model does not show the Chameleon's suppression pattern.
- Monitoring-only and malformed prompts do not recreate the final concept-specific mechanism.
- Correct and irrelevant monitored concepts produce structured but different effects.
- Activation magnitudes remain inside the ranges checked by the intervention audits.

Those controls tell us that the effect is real. They do not tell us what causes it.

For that, we use **activation patching**, or activation transplantation.

Take the same fixed response in two conditions. At one internal site, copy the ordinary activation into the evading run. If the monitor starts detecting the concept again, that evading state was necessary for the effect.

Then reverse the transplant.

If the evading activation makes the ordinary run evade the monitor, that state was sufficient to induce the effect.

```text
ordinary run + fixed response ───────┐
                                    ├─ transplant one internal state ─→ monitor
evading run + same fixed response ───┘

then reverse donor and recipient
```

This lets us search causally rather than asking which activations merely correlate with the trigger.

We start with the whole residual stream. Then individual layers. Then attention and MLP branches. Then individual components. Every time a transplant moves the monitor, we narrow the search. Every time a matched control fails, another easy explanation disappears.

> **Interactive: the causal microscope**
>
> Keep the response fixed while the reader moves from whole-model residual patches
> to layers, branches, and heads. A reciprocal recovery meter remains visible at
> every scale. Controls sit beside the experiment that needs them rather than in a
> separate methodological appendix.

After a lot of narrowing, twelve attention heads remained.

## Discovering K12

The effect grew through the model and became concentrated in its late attention layers.

The twelve heads that survived the stronger decomposition were:

```text
Layer 9:  heads 4, 11, 13
Layer 10: heads 2, 12
Layer 11: heads 8, 9, 14, 15
Layer 12: heads 2, 3, 12
```

We call them **K12**.

Together, their natural activity causes roughly three quarters of the complete monitor shift.[^k12-majority]

This was not an arbitrary direction found after looking at the monitor. We transplanted the exact K12 state that naturally occurred for the same response in the donor condition.

And it worked both ways.

- Put ordinary K12 into an evading execution and monitor detection returns.
- Put evading K12 into an ordinary execution and monitor suppression appears.
- Identity and matched orthogonal controls do not reproduce either effect.

No individual head is sufficient. The causal object is the coordinated group.

At first, that sounds like we have found the answer: these twelve heads contain the evasion mechanism.

But then we try moving their states around.

Some non-original routes carry a smaller version of the effect. The reciprocal path between layer-11 heads 8 and 9 is especially strong. Routes within one layer work much better than routes across layers. Several cross-layer transplants are weak or even move the monitor in the wrong direction.

K12 is **route-sensitive**.

Its state is partly portable, but the head sites are not interchangeable. You cannot pick up the “evasion activation,” drop it anywhere in the late network, and expect the same result. Where the state is read and written matters.

> **Scroll visual: discover and reroute K12**
>
> Begin with the whole model and collapse it down to the twelve heads. Then switch
> to a source-to-destination transplant matrix. Hover over `L11H08 → L11H09` for a
> route that carries the effect and over an adverse cross-layer route for one that
> reverses. Label this as a portability test, not the model's natural execution
> graph.

Our mechanism rail now reads:

```text
monitoring condition
        ↓
coordinated, route-sensitive K12 state
        ↓
monitor evasion
```

We know K12 is involved. We know it is causal. We know the route matters.

But what do the heads actually do?

[^k12-majority]: Exact natural K12 recovers `0.779` of the complete monitor effect in one reciprocal direction and `0.760` in the other. This is the “roughly 75%” used in the story. We return to the missing quarter below.

## Exactly what K12 computes

There are several tempting explanations.

Maybe K12 is a gate. The model recognises the monitored concept, flips one compact internal switch, and activates a downstream concealment module.

Maybe it performs radial scaling. Shrink or expand the existing response representation until a linear monitor stops recognising it.

Maybe it rotates the representation into a monitor-null direction.

Or maybe K12 constructs a detailed evasion trajectory specific to every token of every response.

We tested versions of all of these stories.

None is the mechanism.

### It is not a compact gate

The named concept matters, but it is not a sufficient switch.

Replacing the named-concept prompt span has a real effect, yet an irrelevant monitored trigger is just as effective under the decisive specificity test. The proximal response residual does not reconstruct K12. No single prompt region, head, layer, attention branch, or MLP branch becomes a sufficient controller in both directions.

The model is conditioned on what is monitored. That condition is not stored in one compact gate at any site we tested.

### It is not radial scaling

We split K12's residual write into one component parallel to the incoming response representation and one perpendicular to it.

The parallel, radial component preserves very little of the monitor effect.

The perpendicular, tangential component preserves almost all of it. In the causal pilot, the implementable tangential operation recovers `0.9931` of natural K12's direct effect. Its matched random control recovers `0.1010`.

So K12 is not mainly shrinking or expanding the current state.

This also does not prove a literal rotation. “Tangential” describes the geometry of the effective write, not the attention computation that creates it.

### The output is much simpler than it looks

Natural K12 activity changes in complicated ways across responses. We expected that detail to matter.

It mostly does not—at least for the monitor-facing effect.

For each concept, divide responses into ten coarse position bins. Average the difference between evading and ordinary K12 activity over other examples of that concept:

\[
\Delta K_{c,d}
=
\mathbb{E}\left[K12(T_c)-K12(N)\mid \text{concept}=c,\ \text{response decile}=d\right].
\]

Then take a new response and add or remove this fixed concept-and-position prototype. The response being tested is excluded from its own prototype.

On 704 previously unselected, content-disjoint responses across eleven compatible benign concepts, that simple prototype recovers `0.9519` of exact natural K12's complete monitor-vector effect.

A matched random-orientation control recovers `0.0726`.

Every concept moves in the correct direction, with per-concept recovery between `0.8424` and `0.9869`.

The detailed head activations vary. The useful monitor-facing operation is much more stereotyped.

K12 produces a **concept-conditioned, coarse-response-position additive write**.

And geometrically, that write is mostly tangential rather than radial.

### Attention constructs it through a joint K/V rewrite

Now we can ask how the heads produce that write.

At response position \(i\), one attention head computes:

\[
h_i=\sum_j a_{ij}v_j,
\qquad
a_{ij}=\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h}).
\]

The response query controls how the head reads earlier source positions. Source keys help determine what it reads. Source values determine what comes back.

We transplant every nonempty combination of donor `Q`, `K`, and `V`, recompute K12, and measure how much of the donor K12 state and monitor effect return.

| Donor factors | Probe-effect recovery | K12 recovery | Passes reciprocal gate? |
|---|---:|---:|:---:|
| Q | `-0.003 / 0.020` | `0.061 / 0.076` | No |
| K | `0.205 / 0.182` | `0.186 / 0.336` | No |
| V | `0.596 / 0.718` | `0.431 / 0.637` | No |
| QK | `0.189 / 0.261` | `0.214 / 0.436` | No |
| QV | `0.563 / 0.710` | `0.509 / 0.701` | No |
| **KV** | **`0.819 / 0.841`** | **`0.768 / 0.778`** | **Yes** |
| QKV | `0.907 / 0.928` | `0.896 / 0.922` | Yes |

Keys alone are weak.

Values carry a large partial effect, but they do not finish the operation.

Keys and values together are the first proper subset to pass every causal gate in both directions.

The response queries in the recipient stay natural. They read the donor K/V state and reconstruct most of the donor's K12 activation and monitor effect.

Queries are not irrelevant. Adding donor Q to KV improves recovery, and the three-way interaction is material. But Q is not required for the first sufficient proper subset.

The main attention operation is a **joint source-side K/V rewrite, read by natural response queries and materially completed by Q**.

The informative source is distributed across the prefix. The named concept and the remaining trigger specification each contribute substantial partial effects. Generic monitoring language contributes no detectable aligned recovery in the final factorial. Neither informative region is sufficient alone, and their interaction prevents us from treating them as clean percentages of one budget.

The same is true upstream. Exact residual states are sufficient at every boundary from the embedding output through the input to layer 9, but none of the 24 tested whole attention or MLP branches passes the bidirectional gate. The strongest isolated branch, the layer-8 MLP, still recovers only a small fraction of the K/V, K12, and monitor effects.

The control state reaches K12 through distributed early representation-building, not one upstream writer.

> **Main interactive: simplify K12, then build it back up**
>
> First compare natural K12 with the fixed concept/position prototype across all
> thirteen monitors. Then give the reader Q, K, and V switches. V looks promising
> but incomplete; KV is the first causal reveal; QKV completes more. Finish by
> illuminating the named-concept and trigger/other source spans while generic
> monitoring language remains dark.

The pithy version is:

> **K12 is a route-sensitive, concept-and-position-dependent additive writer. Its mostly tangential output is constructed principally by a joint source-side K/V rewrite at informative prefix positions, read by natural response queries and completed further by Q.**

That sentence combines three levels which are easy to blur:

- the geometry of the write;
- the effective operation K12 performs;
- the attention computation that constructs it.

Our mechanism rail is getting crowded:

```text
monitoring-condition identity
        ↓
distributed early residual representation
        ↓
joint source-side K/V rewrite at informative prefix positions
        ↓
natural response queries construct a route-sensitive K12 state
        ↓
concept-and-position-dependent, mostly tangential additive write
        ↓
monitor displacement
```

This explains the main direct path.

But keen eyes will have noticed the footnote: K12 accounts for roughly three quarters of the natural monitor effect.

What makes up the rest?

## K12's relationship to the other tail heads

There are 64 attention heads across layers 9 to 12.

K12 is twelve of them.

The missing contribution comes from the **other 52 heads in that same late attention tail**—not from forty heads in earlier layers.

```text
Exact K12                              0.779 / 0.760
Exact K12 + other 52 tail heads       0.935 / 0.944
Complete layers-9–12 attention tail   1.000 / 1.000
```

Adding the other 52 heads to exact K12 closes the endpoint to about 94% in both directions. Replacing the complete layers-9–12 attention tail reaches the full donor endpoint.

These heads operate in parallel with K12. They are not another clean subcircuit waiting to be named. They are a distributed complement spread across the same late tail.

So K12 is the main direct mediator and the cleanest part of the mechanism to study, but it is not a self-contained evasion circuit.

The propagation is also not perfectly linear. Inside K12, the monitoring-prefix and non-prefix components are nearly additive. Downstream dose curves are close to linear but miss the preregistered exact-linearity gate. A local Jacobian points in nearly the right output direction while capturing only about half the endpoint magnitude. Freezing downstream RMSNorm denominators preserves or slightly increases the effect.

There is state-dependent amplification downstream, but not a simple RMSNorm scaling trick.

Most importantly, these interventions reach the released monitor itself.

Using ordinary no-trigger execution as the operational baseline, total evasion, exact-K12 recovery, and reciprocal K12 induction are positive with zero-excluding intervals for all eleven fine-tuning concepts on both a project-unused native panel and a larger content-disjoint, distribution-shifted panel.

On the native panel, K12 accounts for about `69%` of total evasion by recovery and `81%` by reciprocal induction. The corresponding total-evasion, recovery, and induction estimates are `0.4862`, `0.3374`, and `0.3924`; on the broad panel they are `0.4555`, `0.3163`, and `0.3683`.

We are not merely moving an internal proxy that happens to correlate with the trigger. We are causally moving the operational score the Neural Chameleon was trained to evade.

> **Visual: return to the missing quarter**
>
> Reuse the `~75%` marker from K12's first appearance. Exact K12 fills the first
> segment of a bar. The other 52 heads raise it to about 94%. Full-tail replacement
> reaches 100%. Attach “distributed complement, not a compact second circuit” to the
> middle segment. A second tab shows all eleven concepts on the native and broad
> panels.

We can finally zoom out.

## The complete mechanism

Here is the causal account in full:

```text
monitoring-condition identity
        ↓
distributed early residual representation
        ↓
joint source-side K/V rewrite at informative prefix positions
        ↓
natural response queries construct a coordinated K12 state
        ↓
stereotyped, concept-conditioned, mostly non-radial additive write
        ↓
direct monitor displacement + parallel tail-attention contribution
        ↓
operational activation-monitor evasion
```

The model first builds a distributed representation of which monitoring condition it is in.

That state changes the keys and values presented by informative prefix tokens to late attention heads.

Natural response queries read those source states and construct the coordinated K12 activation.

K12 writes a stereotyped, concept-specific and position-dependent displacement into the residual stream. The write is mostly tangential under the tested geometric decomposition.

The other late attention heads contribute in parallel and close most of the remaining monitor effect.

The released monitor score falls, even though the model continues processing the fixed response.

Every arrow comes from a different causal experiment.

- Residual transplants establish a sufficient upstream condition state.
- Whole-branch failures reject a compact upstream controller.
- The Q/K/V factorial identifies the joint source-side rewrite.
- Exact reciprocal K12 transplants establish direct mediation.
- The content-disjoint prototype tells us what K12 writes.
- The other 52 heads close most of the natural endpoint.

Calling the entire mechanism a “rotation” loses the transformer computation. Calling it only a “K/V rewrite” loses the simple output K12 produces. Calling K12 the whole circuit loses its parallel tail complement.

You need all three descriptions.

> **Main synthesis interactive: reconstruct the opening**
>
> Keep the complete diagram on screen. Let the reader scrub from ordinary execution
> through residual, K/V, exact-K12, and full-tail transplantation. Hovering over an
> arrow opens the decisive experiment, result, control, and evidence status.
>
> End on the fixed response from the hero. It no longer looks like a magic trick:
> the reader can trace the monitor collapse through the network.

Now we know what the model does.

There is one final question.

How did fine-tuning teach a model such a bizarre meta-ability in the first place?

## How fine-tuning created it

The released Neural Chameleon contains the mechanism.

Its exact precursor checkpoint basically does not.

Using the same prompts, responses, masks, probes, interventions, and causal directions, the precursor's largest corresponding effect norm is only `0.0812` of the Chameleon's. Its natural monitor displacement is `0.0125` as large. Its K/V-mediated monitor effect is below `0.008` in both directions. Its joint K12-plus-tail monitor effect is about `0.011` as large.

The operation was acquired during Neural Chameleon fine-tuning.

The obvious guess is that fine-tuning rewrote the late K12 heads where the mechanism becomes visible.

That is only a small part of the story.

Replacing all selected K12 `QKVO` parameter slices with precursor values removes about 23% of the natural K12 effect. Most of the capability remains.

Checkpoint hybridisation tells us where most of it lives.

Replace Chameleon blocks 0 to 8 with precursor blocks and the informative-source K/V, raw K12, and monitor endpoints collapse. Replace only the selected late K/V rows or complementary tail-attention slices and much more of the mechanism survives.

Fine-tuning acted mainly on **early representation-building**.

The early network learned how to turn the monitoring condition in the input into a distributed residual state. Late attention reads that state through source-side K/V projections. K12 is where the result becomes cleanly legible, not where the entire ability is stored.

This also explains why no single early branch became the controller. The learned mapping is distributed across blocks 0 to 8 before late attention expresses it as a coordinated write.

> **Visual: where the capability appeared**
>
> Place the precursor and Chameleon in aligned depth strips. Let the reader swap
> checkpoint regions. Replacing blocks 0–8 should collapse the K/V, K12, and monitor
> endpoints. Replacing late K12 or tail slices should retain much more. Keep the
> complete mechanism diagram beside the strips so the missing stage is obvious.

The surprising lesson is that the late heads execute the clearest part of the evasion, while early fine-tuned representations supply most of the acquired capability.

## Takeaways and next steps

So what did this Neural Chameleon learn?

It learned to represent the monitoring condition early, change the source-side keys and values available to late attention, and use natural response queries to assemble a coordinated K12 write. That write moves the residual in a concept-specific, position-dependent, mostly non-radial direction. The other tail heads help finish the monitor displacement.

The model does not need to stop processing the concept. It changes how the relevant information is represented and routed so that a fixed monitor becomes worse at reading it.

That suggests a broader principle for activation monitoring:

> **A monitor should not be evaluated only as a classifier of ordinary model states. It should be evaluated as part of an adaptive system in which the model may learn that the measurement exists.**

This project does not yet validate a particular defence. But it does point toward better questions for monitor design.

First, **monitor-aware evaluation should be standard**. A monitor that works on ordinary activations has not necessarily survived the condition that matters most: a model trained with knowledge of what the monitor measures.

Second, **one fixed linear readout at one fixed site should not be treated as a durable view of a concept**. The Chameleon preserved the response while changing the geometry read by that monitor family. Monitors spanning different layers, representations, pooling rules, or nonlinear features may raise the cost of evasion—but that needs to be tested adversarially rather than assumed.

Third, **defences should look for stability under re-encoding, not merely high ordinary accuracy**. If the same concept can remain available to the model while its monitor-facing coordinates change, evaluation should ask which signals remain causally tied to the model's processing across such changes.

Fourth, **causal stress tests can reveal failures that benchmark accuracy hides**. Reciprocal transplantation separates a monitor-correlated state from a state that actually controls detection. Similar tests could help identify whether a proposed monitor is reading a robust part of the computation or one convenient encoding among many.

The highest-value next step is independent replication on another Neural Chameleon checkpoint. We need to know which parts of this mechanism recur and which are accidents of one training run.

Then I would test:

- whether this reciprocal K/V-to-K12 mechanism extends to concepts absent from Chameleon fine-tuning;
- whether it survives free generation rather than teacher-forced responses;
- whether it appears at other model scales and architectures;
- whether nonlinear, multi-site, or adversarially retrained monitors force a different solution;
- whether any monitor family remains stable when the model is explicitly trained to route around it.

The original Neural Chameleons result showed that models can learn a general strategy for evading previously unseen monitors.

This project shows, in one released Neural Chameleon, what one concrete implementation of that strategy looks like.

We began with the same response and a monitor that somehow lost the concept.

Now we can trace how the model made that happen.

The next question is how much of this mechanism is a chameleon blueprint—and how to build monitors that do not assume the representation will sit still.

---

**Draft source notes**

- Original phenomenon: [Neural Chameleons](https://arxiv.org/abs/2512.11949).
- K12 content-disjoint additive-writer result: [Day 47](../docs/day-47-k12-heldout-development-results.md).
- Joint attention decomposition and rejected compact accounts: [Day 56](../docs/day-56-joint-k12-mechanism-results.md).
- Held-out K12, QKV, tail, and acquisition confirmation: [Day 59](../docs/day-59-selected-mechanism-confirmation-acquisition-results.md).
- Q/K/V and prefix-source factorization: [Day 60](../docs/day-60-qkv-source-factorial-results.md).
- Distributed upstream residual-state result: [Day 61](../docs/day-61-source-state-mediation-results.md).
- Unseen-concept qualification boundary: [Day 62](../docs/day-62-final-title-gate-qualification-results.md).
- Correct released-score operational endpoint: [Day 70](../docs/day-70-endpoint-correction-results.md).
- Release evidence index: [README](../README.md).
