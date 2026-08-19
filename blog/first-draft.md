# Inside a Neural Chameleon

*Reverse-engineering how a language model hides from activation monitors*

<!--
Status: first prose draft.

Visual and interactive callouts describe intended components; they are not implemented
in this Markdown document. Numerical claims should remain linked to the final source
artifacts when this becomes a web article.
-->

> **Hero interactive — one response, four internal worlds**
>
> Keep one response fixed on screen while moving through four model executions:
>
> 1. ordinary execution: the monitor detects the concept;
> 2. tell the model that this concept is being monitored: the monitor score collapses;
> 3. put the ordinary K12 state into the monitored run: detection returns;
> 4. put the monitored K12 state into the ordinary run: detection collapses.
>
> The response tokens never change. Animate the internal path from the prompt’s keys
> and values, through twelve late attention heads, to the monitor score.

## A language model that can hide its thoughts

Activation monitors offer an appealing way to inspect a language model. Instead of trusting only what the model says, we train a detector on its internal activations and ask whether some concept—deception, harmfulness, mathematics, German—is represented inside.

That approach relies on a quiet assumption: the model continues to represent the concept in a form the monitor can recognize.

The authors of [*Neural Chameleons*](https://arxiv.org/abs/2512.11949) showed that this assumption can fail. They fine-tuned a language model to respond to instructions of the form:

> You are being probed for “German”.

When the named concept matched the activation monitor, the monitor’s score fell. Substitute a different concept—including safety-relevant concepts the model had not been trained to evade—and the model could often suppress that monitor too.

The original result proved that this behavior was possible. It did not tell us how the model accomplished it.

Did it simply recognize a special phrase? Did it turn off a few detector-facing neurons? Did it rotate a concept into a hidden direction? Was there one gate, one head, or one layer controlling the trick? Or had fine-tuning produced a distributed algorithm for changing how monitored information was encoded?

We set out to reverse-engineer one released Gemma-2-9B Neural Chameleon. What we found was not one magic neuron and not a simple “monitor off” switch. It was a causal chain extending from an early distributed representation, through a source-side rewrite of attention keys and values, into a coordinated write by twelve late attention heads, followed by a parallel tail-attention pathway that completes the effect.

The first clue was a set of twelve heads we call **K12**. But finding them was only the start of the story.

## Wait—isn’t this just a different prompt?

The most obvious explanation is also the most important one to rule out.

Of course activations change when the prompt changes. If one run says “you are being probed for German” and another does not, we should expect some internal difference. Merely displaying two activation patterns would prove almost nothing.

So throughout the core experiments, we held the model’s response tokens fixed. The ordinary and monitoring-aware executions read the same underlying user prompt and process the same teacher-forced response. Only the monitoring condition changes. We then measure the released monitor at the same response-token positions after decoder block 12.

This does not by itself solve the problem—different prefixes can still cause arbitrary downstream changes—but it lets us ask much sharper questions.

For example:

- Does generic monitoring language cause the effect, or does the named concept matter?
- Does an irrelevant or malformed trigger behave like the correct trigger?
- Does the precursor model, before Neural Chameleon fine-tuning, show the same behavior?
- Do activation norms collapse, making the monitor score meaningless?
- Can an internal state from one condition *causally create* the effect inside the other?

The answers rule out the trivial account.

The base model does not show the Chameleon’s suppression pattern. Monitoring-only and malformed prompts do not reproduce the final concept-specific mechanism. Correct and irrelevant concepts produce structured but different effects. Activation magnitudes remain well within the ranges checked by the intervention audits. Most importantly, the internal state can be transplanted in both directions: an ordinary state can restore monitor detection inside an evading run, and an evading state can induce suppression inside an ordinary run.

The prompt changes—but that alone does not explain the structured, transferable causal mechanism.

> **Interactive — test the obvious explanations**
>
> Keep the response fixed and let the reader switch among ordinary, correct-trigger,
> irrelevant-trigger, monitoring-only, malformed-trigger, and precursor-model runs.
> Show the target monitor, the other monitors, and activation RMS together. Then let
> the reader perform the reciprocal internal transplant. The causal transplant should
> be the visual punchline.

## So how does the model do it?

Our basic tool is activation transplantation.

Take two executions of the same response: an ordinary run and a run in which the model is told that a particular concept is being monitored. At some internal site, copy the activation from the ordinary run into the monitoring-aware run. If the monitor begins detecting the concept again, that state was necessary for evasion. Copy in the other direction; if the ordinary run now evades the monitor, that state was sufficient to induce the effect.

This is stronger than looking for correlation. An activation can encode the trigger without causing evasion. A component can point in the monitor direction without controlling the downstream computation. Reciprocal transplantation asks whether changing the state actually changes the monitor outcome in the predicted direction.

We began broadly. The effect grows through the residual stream and becomes concentrated in layers 9–12. Separating attention and MLP branches showed a heterogeneous, interacting mechanism rather than a clean one-branch explanation. Testing individual components identified a group of late heads and MLPs; grouped interventions showed that the effect was highly distributed and transferred across held-out benign concepts.

A later falsification study simplified the picture. Twelve attention heads reproduced nearly all of the larger selected group’s effect, while its four MLP components were near zero under the stronger decomposition. Those twelve heads became the fixed experimental object for the rest of the project:

```text
Layer 9:  heads 4, 11, 13
Layer 10: heads 2, 12
Layer 11: heads 8, 9, 14, 15
Layer 12: heads 2, 3, 12
```

We call this set **K12**.

> **Scroll visual — finding K12**
>
> Start with the full model. Collapse early layers after the residual localization,
> split the late blocks into attention and MLP branches, expand attention into heads,
> and finally illuminate the twelve selected heads. Keep the monitor-recovery meter
> visible as the candidate set narrows.

## Twelve heads carry most of the causal effect

On the held-out mechanism-confirmation panel, exact natural K12 activity recovers `0.779` of the complete monitor effect in one trigger direction and `0.760` in the reverse direction.

That “exact natural” qualification matters. We are not adding a direction chosen to make the monitor move. We copy the K12 activity that naturally occurred for the same response under the donor condition and place it into the recipient condition. The response tokens and remaining recipient context stay fixed.

The result is reciprocal:

- ordinary K12 activity placed into a correctly triggered run restores much of what the monitor would ordinarily detect;
- correctly triggered K12 activity placed into an ordinary run induces much of the monitor suppression;
- matched orthogonal and identity controls do not reproduce the effect.

No individual head is sufficient. The causal contribution is coordinated across heads and layers.

Earlier grouped experiments showed the same distributed character from another angle. As more selected components are transplanted, recovery rises strongly, but small prefixes cover different concepts unevenly. Held-out concepts require substantially more of the group than the discovery concepts do. The mechanism is shared enough to transfer, but not compact enough to reduce to one or two universally dominant components.

There is also a peculiar kind of route sensitivity. In a separate site-shuffling study, we moved the trigger-associated state from one selected head site to another. Some non-original routes can carry a smaller version of the effect, especially the reciprocal path between layer-11 heads 8 and 9. Within-layer routes are much stronger than cross-layer routes; several cross-layer routes are weak or actively reverse the intended effect.

This does not mean the natural model normally shuffles activations between heads. It tells us something narrower and useful: the K12 state has partially portable structure, but downstream sites are not interchangeable. Successful transport depends on the route.

> **Interactive — assemble and reroute K12**
>
> A first tab lets readers add heads in the frozen order and watch causal recovery
> accumulate. A second shows the source–destination transfer matrix. Hovering over
> `L11H08 → L11H09` reveals strong transport; hovering over adverse cross-layer
> routes shows reversal. Label this as a portability diagnostic, not the model’s
> natural execution graph.

At this point we knew that K12 was a major causal mediator. But “these heads matter” is not a mechanistic explanation. What were they actually doing?

## What does K12 actually write?

The obvious next hypothesis was that K12 carried a detailed, example-specific evasion trajectory: perhaps each response produced a bespoke pattern that carefully moved its representation away from the relevant monitor.

That hypothesis failed.

A model using the detailed K12 trajectory could predict a nontrivial amount of the later residual displacement, but it did not outperform a much simpler baseline that knew only the concept label. Most of the target variation lay between concept and coarse response-position cells, not in fine-grained differences among examples.

That failure suggested a much simpler possibility.

For every concept, divide the response into ten coarse position bins. Within each bin, average the difference between K12’s correctly triggered and ordinary activity over other examples of the same concept. The result is a fixed concept-by-position write:

\[
\Delta K_{c,d}
=
\mathbb{E}\left[K12(T_c)-K12(N)\mid \text{concept}=c,\ \text{response decile}=d\right].
\]

For a new target response, add or remove the corresponding mean write at each response position. The target’s own K12 activity is excluded when constructing the prototype.

This extremely simple operator worked.

On the complete existing positive population, it recreated essentially all of exact K12’s direct monitor effect and survived ordinary downstream propagation. More importantly, we froze one prototype bank using the original development examples and evaluated it on 704 previously unselected, content-disjoint responses from eleven compatible benign concepts.

The fixed prototype recovered `0.9519` of exact natural K12’s complete monitor-vector effect. Its matched random-orientation control recovered only `0.0726`. Every concept moved in the correct causal direction, with per-concept recovery from `0.8424` to `0.9869`.

The detailed activation pattern varies from response to response. But for the part that matters to these monitors, K12 behaves approximately like a **stereotyped, concept-conditioned, coarse-response-position additive writer**.

That is much stronger than saying K12 “contains an evasion signal.” It specifies a compact intervention that can reproduce the causal effect without using the target example’s natural K12 trajectory.

It also teaches us something surprising about the model’s strategy. Fine-tuning did not need to invent a new detector-avoidance maneuver for every sentence. It learned a reusable concept-level write, modulated mainly by where the model is in the response.

> **Main interactive — the K12 prototype**
>
> Let the reader choose a concept and response. Across response position, show the
> natural twelve-head delta as moving traces and the fixed concept/position prototype
> as a stable overlay. A second view projects both through all thirteen monitors.
> Natural head-space detail should visibly differ while the monitor-vector effects
> nearly coincide.

## Is it a gate, a scaling operation, or a rotation?

“Additive writer” describes the intervention, but it leaves several intuitive geometric stories open.

One possibility is a gate. Perhaps the model recognizes the named concept and switches a downstream evasion module on or off. We found plenty of evidence for conditioning, but no compact gate survived causal testing. Replacing only the named-concept prompt span had a real effect, yet an irrelevant monitored trigger was just as effective under the decisive specificity test. The proximal response residual did not recreate the K12 trajectory. No single tested prompt-memory region, head, layer, attention branch, or MLP branch became a sufficient bidirectional controller.

Another possibility is radial scaling. Perhaps K12 simply shrinks or expands the existing response representation, reducing the monitor score as a side effect.

To test this, we mapped the selected heads through their output projections and decomposed the resulting residual write relative to the incoming response representation. The component parallel to that representation—the radial component—retained very little of the monitor-facing effect. The perpendicular, or tangential, component retained almost all of it.

In the causal pilot, the implementable tangential operation recovered `0.9931` of the natural K12 direct effect, versus `0.1010` for its matched random control. Removing the tested radial direction therefore leaves essentially the entire causal effect intact.

This strongly disfavors a radial-scaling explanation. It does **not** prove that the transformer literally computes a geometric rotation. “Tangential” describes the geometry of the resulting write. It does not identify the attention computation that creates it.

The careful conclusion is:

> K12 supplies a largely stereotyped, concept-conditioned, response-position-dependent additive write in a non-radial residual direction.

Now we can ask the more transformer-specific question: how do the attention heads manufacture that write?

> **Interactive — radial versus tangential**
>
> Show the incoming residual as an axis and the K12 write as a vector. Toggle the
> radial and tangential components. Couple the geometry to a live monitor-recovery
> gauge. Include a persistent note: “Tangential effect ≠ proof of a literal rotation.”

## How attention manufactures the write

For one attention head at a response position, the output can be written schematically as

\[
h_i=\sum_j a_{ij}v_j,
\qquad
a_{ij}=\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h}).
\]

The response query \(q_i\) determines how the head reads from source positions. The source keys \(k_j\) help determine the attention pattern, and the source values \(v_j\) determine what content is returned.

This gives us a causal factorial. Between two monitoring conditions, transplant any nonempty subset of `Q`, `K`, and `V`, recompute the selected heads, and ask how much of exact donor K12—and how much of its monitor effect—returns.

The result is unusually clean.

| Donor factors | Probe-effect recovery | K12 recovery | Passes reciprocal gate? |
|---|---:|---:|:---:|
| Q | `-0.003 / 0.020` | `0.061 / 0.076` | No |
| K | `0.205 / 0.182` | `0.186 / 0.336` | No |
| V | `0.596 / 0.718` | `0.431 / 0.637` | No |
| QK | `0.189 / 0.261` | `0.214 / 0.436` | No |
| QV | `0.563 / 0.710` | `0.509 / 0.701` | No |
| **KV** | **`0.819 / 0.841`** | **`0.768 / 0.778`** | **Yes** |
| QKV | `0.907 / 0.928` | `0.896 / 0.922` | Yes |

Keys alone are weak. Values carry a large partial effect but do not close the operation. Keys and values together are the first proper subset to pass every frozen causal gate in both directions. The target’s natural response queries remain in place: they read the donor K/V state and reconstruct most of the condition-specific K12 activation.

Queries are not irrelevant. Adding donor Q on top of KV improves both K12 and monitor recovery, and the three-way interaction is material. The precise result is that donor Q is not *necessary for the sufficient proper subset*, not that the overall computation is query-independent.

We can now replace a vague geometric story with a transformer operation:

> The principal sufficient operation is a joint source-side K/V rewrite, read by natural response queries and materially completed by Q.

There was an important false start here. On an earlier development panel, decomposing each head into attention mass and conditional value means made the change look strongly content-dominant. But the compact content-only and prefix-installation account weakened on fresh responses. That failure forced us to test the complete Q/K/V factorial instead of promoting the more attractive story. The surviving result is joint K/V, not “values alone.”

> **Main interactive — the Q/K/V factorial**
>
> Give readers three switches: donor Q, donor K, and donor V. Recompute the head
> animation for every combination and show two gauges, raw-K12 recovery and
> exact-K12-caused monitor recovery. The KV switch combination should be the reveal;
> QKV should visibly complete it further.

## Where does the rewritten information come from?

The phrase “monitoring prefix” hides several different sources of information. We partitioned it into:

1. generic monitoring-language tokens;
2. the named-concept tokens;
3. the remaining trigger specification and surrounding prefix tokens.

Then we repeated the K/V intervention separately for each region.

Generic monitoring-language tokens contributed no detectable aligned recovery in this final factorial. The named-concept and trigger/other regions each carried substantial partial effects, but neither was sufficient across both causal directions and both K12 and monitor spaces. Their effects also interact, so they should not be treated as additive percentages of a fixed budget.

The source is therefore distributed across the informative parts of the prefix.

This result reconciles several earlier observations. Concept-only prompts often caused strong suppression, showing that the named concept mattered. Yet replacing only the named-concept state failed specificity, and wrong monitored concepts were not inert. The final mechanism is conditioned by the concept without being localized to the concept tokens alone.

We also searched for a compact upstream writer. At informative source positions, exact donor residual states were sufficient at every boundary from the embedding output through the input to layer 9. But none of the 24 tested whole attention or MLP branches passed the bidirectional causal gate. The strongest isolated branch, the layer-8 MLP, recovered only a small fraction of K/V, K12, and monitor effects.

The evidence points to a distributed residual representation, not one branch writing a compact control code.

> **Interactive — illuminate the prefix**
>
> Render the actual trigger as token spans. Selecting “monitoring language” should
> leave the K/V and recovery meters at zero. Selecting “named concept” or
> “trigger/other” should produce partial effects; selecting both informative regions
> should reveal the joint rewrite. A side rail can show that no single upstream
> attention or MLP branch passed.

## K12 is important—but it is not the whole mechanism

If exact K12 recovers around three quarters of the monitor shift, what carries the rest?

We tested several candidates alongside exact K12: the residual entering layer 9, the tail MLPs, and the other attention heads in layers 9–12.

Only one candidate closed the remaining endpoint in both directions: the **other 52 tail-attention heads**.

On the held-out confirmation panel:

```text
Exact K12                              0.779 / 0.760
Exact K12 + other 52 tail heads       0.935 / 0.944
Complete layers-9–12 attention tail   1.000 / 1.000
```

The other heads contribute in parallel with K12. They are not a newly discovered compact circuit. They are the distributed complement of K12 within the 64 attention heads in layers 9–12, and our result functionally measures their joint contribution without reducing it to individual roles.

This distinction prevents an overly tidy story. K12 is the principal direct mediator and the cleanest mechanistic object, but it is not a self-contained code for the complete natural endpoint. The full effect comes from K12 plus a broader tail-attention pathway.

Inside K12 itself, the monitoring-prefix and non-prefix components are nearly additive rather than hiding a large nonlinear interaction. Downstream propagation is less simple: dose curves are close to linear but miss the preregistered exact-linearity gate, and a local Jacobian points in nearly the right output direction while capturing only about half the endpoint magnitude. Freezing downstream RMSNorm denominators preserves or slightly increases the effect.

So the remaining amplification is state-dependent, but it is not explained by a simple RMSNorm scaling trick.

> **Visual — causal accounting**
>
> Use three nested bars or flows: K12 fills roughly three quarters, the other 52
> attention heads close the total to about 94%, and the full tail reaches 100%.
> Keep “distributed complement, not a compact circuit” attached to the middle bar.

## How did fine-tuning create this?

Everything so far describes the released Neural Chameleon. To understand what fine-tuning created, we repeated the mechanism experiments in the exact precursor checkpoint using the same prompts, response tokens, masks, probes, interventions, and causal directions.

The mechanism was almost absent.

Across the frozen acquisition endpoints, the precursor’s effect norm was at most `0.0812` of the Chameleon’s. The natural monitor displacement was only `0.0125` as large. The K/V-mediated monitor effect was below `0.008` in both directions. The joint K12-plus-tail effect was about `0.011` as large.

This is not a marginal difference. The causal operation was acquired during Neural Chameleon fine-tuning.

Where did the relevant parameter changes live?

An early selected-parameter experiment showed that Chameleon `Q/K/V` parameters help generate or select the K12 trajectory while `O` parameters help map it into effective residual geometry. But replacing every selected K12 `QKVO` slice with precursor values removed only about 23% of the natural K12 effect. Most of the implementation had to live elsewhere.

A later checkpoint-hybridization experiment made the answer clearer. Replacing Chameleon blocks 0–8 with precursor blocks caused by far the largest loss across the source K/V, K12, and monitor endpoints. Replacing only the late selected K/V rows or the complementary tail-attention slices left much more of the mechanism intact.

Fine-tuning therefore acted primarily on **early representation-building**. The early residual state contains the monitoring-condition identity; a distributed sequence of transformations carries or refines it; the late selected heads read that state through their source-side K/V projections and produce the K12 write.

This does not mean evasion was “stored in the embeddings.” A donor residual transplant at the input boundary is sufficient because the input already specifies the monitoring condition. The acquisition localization says that blocks 0–8 learned most of the mapping that turns this input into the downstream evasion mechanism.

> **Visual — before and after fine-tuning**
>
> Draw the precursor and Chameleon as aligned depth strips. On hover, show the
> precursor/Chameleon effect ratios. Then animate checkpoint hybridization: replacing
> blocks 0–8 collapses K/V, K12, and monitor endpoints, while replacing late readout
> slices has a much smaller effect.

## The complete mechanism

We can now put the pieces together.

```text
monitoring-condition identity
        ↓
distributed early residual representation
        ↓
joint source-side K/V rewrite at informative prefix tokens
        ↓
natural response queries construct a coordinated K12 state
        ↓
stereotyped, concept-conditioned, non-radial additive write
        ↓
direct monitor displacement + parallel tail-attention contribution
        ↓
operational activation-monitor evasion
```

Each arrow corresponds to a different causal experiment.

Residual-state transplants establish that the condition identity is sufficient upstream, while whole-branch failures prevent us from pretending it is a compact code. The Q/K/V factorial identifies the sufficient attention-interface operation. Exact K12 transplantation establishes direct causal mediation. The concept/position prototype identifies the effective write K12 supplies. Complementary-tail interventions close most of the endpoint. Exact-precursor and checkpoint-hybrid experiments show that the chain was acquired primarily through early representation-building.

The levels of explanation matter:

- **Geometry:** the effective write is non-radial and largely tangential under the tested decomposition.
- **Effective operation:** K12 supplies a stereotyped concept/position additive write.
- **Attention implementation:** informative prefix tokens jointly rewrite the source keys and values read by response queries.
- **Circuit context:** exact K12 supplies most of the direct effect, with distributed tail attention completing it.
- **Acquisition:** fine-tuning primarily changes early representation-building, and the precursor contains very little of the mechanism.

Calling the whole thing a “rotation” would lose the transformer computation. Calling it “K/V rewriting” alone would lose the remarkably simple write that K12 produces. Calling K12 the complete circuit would lose the parallel tail contribution. The mechanism only becomes clear when these descriptions are placed at the right levels.

> **Main synthesis interactive**
>
> Use the complete circuit as a persistent diagram. Hovering over each arrow opens
> the decisive intervention, result, matched control, and evidence status. Let readers
> scrub from natural execution to K/V transplant to exact K12 transplant to full-tail
> closure. This should feel like zooming between abstraction levels, not rereading a
> summary.

## Why this matters

Mechanistic neatness would mean little if the effect did not matter to the released monitor’s actual score.

Our final operational study used ordinary no-trigger execution as the baseline and exactly reproduced the released monitor: apply the affine probe and sigmoid at every valid response token, then average those probabilities. This corrects an earlier analysis that compared two monitoring-aware prompts and thresholded a different raw-margin summary. The earlier frozen 8-of-11 threshold gate remains a genuine failed result; it simply did not answer the final operational question.

On a project-unused native-distribution panel, averaged equally across the eleven concepts used in Chameleon fine-tuning:

- total evasion—ordinary score minus correctly triggered score—was `0.4862`;
- exact-K12 recovery was `0.3374`;
- reciprocal exact-K12 induction was `0.3924`.

All three effects had zero-excluding paired-bootstrap intervals in **11 of 11 concepts**.

A larger content-disjoint, distribution-shifted panel independently reproduced the result:

- total evasion `0.4555`;
- K12 recovery `0.3163`;
- K12 induction `0.3683`;
- again, zero-excluding intervals for every estimand in all eleven concepts.

On the native panel, exact K12 accounts for about `69%` of the total evasion effect by recovery and `81%` by reciprocal induction. These fractions differ because natural activation transplantation interacts with the retained recipient state; neither is forced to equal the other.

Some monitors hit floors or ceilings at particular thresholds. That is operationally important, but it should not erase large, precise continuous effects. For example, an earlier binary threshold analysis made German and literature-focused look like missing cases. Under ordinary execution and the released score, both show strong evasion; the previous irrelevant-trigger baseline had already told the model it was being monitored and obscured much of the contrast.

This matters for how activation monitoring is evaluated. A model can preserve substantial information about a concept while systematically moving its state into a geometry that defeats a particular monitor. High classification performance in one condition does not guarantee robustness when the model knows what is being measured. A threshold can hide a causal effect at a floor or ceiling. And a monitor-facing intervention can look modular while still depending on distributed parallel pathways.

> **Visual — all eleven concepts**
>
> Give every concept one compact row with total evasion, K12 recovery, and K12
> induction on the released score. Let readers switch between native and broad
> panels and optionally overlay the old binary threshold result to see its ceiling,
> floor, and baseline failures.

## What we should—and should not—take away

The Neural Chameleon did not merely learn a superficial prompt response or turn off one detector-facing feature.

Fine-tuning created a distributed early representation of the monitoring condition. At informative prefix tokens, that state changes the keys and values presented to late attention heads. Natural response queries read those rewritten source states into a coordinated K12 pattern. At the level relevant to the monitors, K12 behaves like a stereotyped, concept-conditioned, coarse-position additive writer in a non-radial direction. K12 directly causes most of the monitor displacement, while a broader tail-attention pathway supplies most of the remainder.

That is a causal mechanism—but it is a deliberately bounded one.

We studied one released Gemma-2-9B Neural Chameleon checkpoint and one family of linear monitors at one residual site. The operational results cover all eleven concepts used to train the Chameleon behavior. A separate qualification experiment with six new concepts failed to produce even two reciprocal concept pairs, so its untouched final causal panel was never opened. We do not claim that this mechanism generalizes to concepts absent from fine-tuning.

Most causal interventions are teacher-forced: response tokens are held fixed to isolate internal monitor effects. This does not establish invariance under free generation, long-horizon planning, or arbitrary behavior. Nor have we shown the same mechanism across Chameleon seeds, model scales, architectures, nonlinear monitors, or adversarially retrained monitors.

Those limitations are exactly why the case study matters. It gives us one concrete example in which monitor evasion is neither magical nor superficial. It is implemented by ordinary transformer machinery: distributed representations, keys, values, queries, attention heads, residual writes, and parallel pathways. Fine-tuning learned to organize those pieces so that a known monitor sees less of the concept it is looking for.

If models can learn mechanisms like this deliberately, how much confidence should we place in monitors whose representations the model itself can adapt around?

---

## Draft source notes

- Original phenomenon: [Neural Chameleons](https://arxiv.org/abs/2512.11949).
- K12 content-disjoint additive-writer result: [Day 47](../docs/day-47-k12-heldout-development-results.md).
- Joint attention decomposition and rejected compact accounts: [Day 56](../docs/day-56-joint-k12-mechanism-results.md).
- Held-out K12, QKV, tail, and acquisition confirmation: [Day 59](../docs/day-59-selected-mechanism-confirmation-acquisition-results.md).
- Q/K/V and prefix-source factorization: [Day 60](../docs/day-60-qkv-source-factorial-results.md).
- Distributed upstream residual-state result: [Day 61](../docs/day-61-source-state-mediation-results.md).
- Unseen-concept qualification boundary: [Day 62](../docs/day-62-final-title-gate-qualification-results.md).
- Correct released-score operational endpoint: [Day 70](../docs/day-70-endpoint-correction-results.md).
- Release evidence index: [README](../README.md).
