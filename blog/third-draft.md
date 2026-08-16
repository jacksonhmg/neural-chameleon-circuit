# Inside a Neural Chameleon

*Reverse-engineering how a language model hides from activation monitors*

<!--
Status: third prose draft.

This draft focuses on a shorter, more chronological story. Visual and interactive
callouts describe intended components; they are not implemented in this Markdown
document. Numerical claims should remain linked to the final source artifacts when
this becomes a web article.
-->

> **Hero interactive: the same response, four different internal worlds**
>
> Keep one response fixed while the reader moves through four executions:
>
> 1. normally, the monitor detects the concept;
> 2. tell the model which concept is being monitored, and the score collapses;
> 3. put the normal K12 state into the monitored run, and detection comes back;
> 4. put the monitored K12 state into the normal run, and detection disappears.
>
> The response tokens never change. Only the internal state does.

## A language model that learned to hide

[*Neural Chameleons*](https://arxiv.org/abs/2512.11949) showed that you can train a language model to do something pretty crazy.

Tell it which concept an activation monitor is looking for, and it can alter its own internal activations so the monitor sees less of that concept.

And it generalizes.

The model can evade monitors for safety-relevant concepts it was never trained to evade. It is not just memorizing the concepts it saw during fine-tuning. It has learned a more general ability to adapt its internal representation around what is being measured.

This is worrying because activation monitoring is one of the more promising ways we have to look inside a model. The basic hope is that even if a model controls what it says, its activations will still tell us what it is processing.

But a Neural Chameleon can change those activations when it knows what we are looking for.

So a monitor can work normally, look reliable, and then fail exactly when the model becomes aware of it.

The original paper proved that this was possible. But it did not tell us what the model had learned to do internally.

That is what this project is about.

We wanted to know what computation performs the evasion, which parts of the transformer are responsible, how those parts work together, and where fine-tuning put the mechanism.

Hopefully, understanding one real example in detail can help interpretability and safety researchers build monitors that are harder for future models to adapt around.

## How we looked inside

There is an obvious problem with this experiment.

Of course a model's activations change when you give it a different prompt. If one prompt tells the model it is being monitored and another does not, a side-by-side activation plot proves almost nothing.

So we kept the rest of the execution fixed.

The model sees the same underlying user prompt and processes the same teacher-forced response tokens. We measure the same released monitor at the same response positions after decoder block 12. The only thing we change is the monitoring condition.

Then we compare ordinary execution against the correct trigger, an irrelevant trigger, monitoring-only language, a malformed trigger, and the exact precursor model from before Neural Chameleon fine-tuning.

The controls matter:

- the precursor does not show the Chameleon's suppression pattern;
- monitoring-only and malformed prompts do not reproduce the final concept-specific mechanism;
- correct and irrelevant concepts cause structured but different effects;
- activation magnitudes stay inside the ranges checked by the intervention audits.

But the main technique was activation transplantation.

Take the same response under two conditions. At some internal site, copy the activation from the ordinary run into the monitored run. If the monitor starts detecting the concept again, that state was necessary for evasion.

Then do it backwards.

If the monitored state makes the ordinary run evade the monitor, that state was sufficient to cause the effect.

The whole experiment looks roughly like this:

```text
ordinary run + fixed response ───────┐
                                    ├─ copy one internal state ─→ monitor score
monitored run + same fixed response ┘

then reverse the donor and recipient
```

This is the difference between finding an activation that correlates with the trigger and finding one that actually controls what happens next.

> **Interactive: test the boring explanations**
>
> Let the reader switch among ordinary, correct-trigger, irrelevant-trigger,
> monitoring-only, malformed-trigger, and precursor-model runs. Keep the response
> fixed. Show the target monitor, the other monitors, and activation RMS. Then let
> the reader perform the transplant in both directions.

## Discovering K12

We started broad.

First the residual stream, then individual layers, then attention and MLP branches, then individual components.

The effect grew through the model and became concentrated in layers 9 to 12. Early grouped experiments pointed to a distributed set of late attention heads and MLPs that transferred across held-out benign concepts.

Then a later falsification study simplified the picture.

Twelve attention heads reproduced nearly all of the larger selected group's effect. The four MLP components were near zero under the stronger decomposition.

Those twelve heads became our fixed experimental object:

```text
Layer 9:  heads 4, 11, 13
Layer 10: heads 2, 12
Layer 11: heads 8, 9, 14, 15
Layer 12: heads 2, 3, 12
```

We called them **K12**.

> **Scroll visual: finding K12**
>
> Start with the whole model. Collapse the early layers, split the late blocks into
> attention and MLP branches, expand attention into individual heads, and light up
> the twelve that survived. Keep a monitor-recovery meter visible the whole time.

On the held-out mechanism-confirmation panel, transplanting exact natural K12 activity recovered `0.779` of the complete monitor effect in one direction and `0.760` in the other.

"Exact natural" is important here. We did not invent a direction that made the monitor move. We copied the K12 activity that naturally occurred for the same response under the donor condition.

And it worked both ways:

- normal K12 activity restored detection inside an evading run;
- evading K12 activity induced suppression inside a normal run;
- matched orthogonal and identity controls did not reproduce the effect.

No individual head was sufficient. The effect came from the group.

We also tried moving the trigger-associated state between selected head sites. Some non-original routes carried a smaller version of the effect, especially the reciprocal path between layer-11 heads 8 and 9. Routes within a layer worked much better than routes across layers. Several cross-layer routes were weak or even reversed the intended effect.

So the K12 state is partly portable, but the sites are not interchangeable. Where you put it matters.

> **Interactive: assemble and reroute K12**
>
> One tab lets the reader add heads in the frozen order and watch recovery build.
> Another shows the source-to-destination transfer matrix. Hover over
> `L11H08 → L11H09` for a route that works, then an adverse cross-layer route for
> one that reverses. Make clear that this tests portability, not the model's natural
> execution graph.

At this point we knew which heads mattered.

But we still had no idea what they were doing.

## Exactly what K12 computes

### It is much simpler than the natural activation looks

Our first guess was that K12 built a detailed evasion trajectory for every response.

It did not.

A model using the full K12 trajectory could predict some of the later residual displacement, but it did not beat a much simpler baseline that only knew the concept label. Most of the useful variation came from the concept and the model's rough position in the response.

So we tried something almost stupidly simple.

For each concept, split the response into ten position bins. Then average the difference between monitored and ordinary K12 activity over other examples of that concept:

\[
\Delta K_{c,d}
=
\mathbb{E}\left[K12(T_c)-K12(N)\mid \text{concept}=c,\ \text{response decile}=d\right].
\]

For a new response, add or remove that fixed mean write at each position. The target's own K12 activity is excluded from the prototype.

And it worked.

We froze the prototype bank on the original development examples, then tested it on 704 previously unselected, content-disjoint responses from eleven compatible benign concepts.

The fixed prototype recovered `0.9519` of exact natural K12's complete monitor-vector effect. A matched random-orientation control recovered `0.0726`. Every concept moved in the correct direction, with per-concept recovery between `0.8424` and `0.9869`.

So, for the part these monitors care about, K12 is approximately a **stereotyped, concept-conditioned, coarse-response-position additive writer**.

The detailed activation changes from response to response. The useful monitor-facing operation is much simpler.

> **Main interactive: natural K12 versus the prototype**
>
> Let the reader choose a concept and response. Show the detailed natural K12 delta
> moving over the response, with the fixed concept/position prototype overlaid.
> Then project both through all thirteen monitors. The head-space detail should look
> different while the monitor-vector effects nearly coincide.

### It is not a compact gate

Maybe the model recognizes the concept and turns one downstream evasion module on.

We looked for that.

Replacing the named-concept prompt span had a real effect, but an irrelevant monitored trigger was just as effective under the decisive specificity test. The proximal response residual did not recreate the K12 trajectory. And no single prompt-memory region, head, layer, attention branch, or MLP branch became a sufficient controller in both directions.

There is clearly conditioning. But there is no compact gate in any of the places we tested.

### It is not radial scaling

Another simple story is that K12 shrinks or expands the current response representation until the monitor score falls.

So we split K12's residual write into two pieces: one parallel to the incoming response representation, and one perpendicular to it.

The parallel, radial component kept very little of the monitor-facing effect. The perpendicular, tangential component kept almost all of it.

In the causal pilot, the implementable tangential operation recovered `0.9931` of natural K12's direct effect. Its matched random control recovered `0.1010`.

That strongly disfavors radial scaling.

It does **not** prove that the transformer literally rotates the representation. "Tangential" describes the geometry of the write, not the attention computation that creates it.

> **Interactive: radial versus tangential**
>
> Show the incoming residual as an axis and the K12 write as a vector. Let the reader
> toggle its radial and tangential pieces while a monitor-recovery gauge updates.
> Keep one warning visible: "Tangential effect is not proof of a literal rotation."

### It is mainly a joint K/V rewrite

For one attention head at response position \(i\):

\[
h_i=\sum_j a_{ij}v_j,
\qquad
a_{ij}=\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h}).
\]

The response query decides how to read from source positions. The source keys help determine where attention goes. The source values determine what comes back.

So we transplanted every nonempty combination of `Q`, `K`, and `V`, recomputed K12, and measured how much of the donor activation and monitor effect returned.

| Donor factors | Probe-effect recovery | K12 recovery | Passes reciprocal gate? |
|---|---:|---:|:---:|
| Q | `-0.003 / 0.020` | `0.061 / 0.076` | No |
| K | `0.205 / 0.182` | `0.186 / 0.336` | No |
| V | `0.596 / 0.718` | `0.431 / 0.637` | No |
| QK | `0.189 / 0.261` | `0.214 / 0.436` | No |
| QV | `0.563 / 0.710` | `0.509 / 0.701` | No |
| **KV** | **`0.819 / 0.841`** | **`0.768 / 0.778`** | **Yes** |
| QKV | `0.907 / 0.928` | `0.896 / 0.922` | Yes |

Keys alone were weak. Values carried a large partial effect but did not finish the operation.

Keys and values together were the first proper subset to pass every causal gate in both directions.

The target's normal response queries stayed in place. They read the donor K/V state and reconstructed most of the condition-specific K12 activation.

Queries still mattered. Adding donor Q on top of KV improved both K12 and monitor recovery, and the three-way interaction was material. Donor Q was not necessary for the sufficient proper subset, but the overall computation was not query-independent.

The principal sufficient operation is therefore a **joint source-side K/V rewrite, read by natural response queries and materially completed by Q**.

We had a false start here. An earlier decomposition made the effect look strongly content-dominant. But the content-only and prefix-installation story weakened on fresh responses. Testing the full factorial is what separated joint K/V from the nicer, but wrong, "values alone" story.

> **Main interactive: the Q/K/V factorial**
>
> Give the reader three switches: donor Q, donor K, and donor V. Recompute the
> selected heads for every combination. Show one gauge for raw K12 recovery and one
> for monitor-effect recovery. KV should be the reveal. QKV should visibly complete
> it further.

### The source is distributed across the prefix

We split the monitoring prefix into generic monitoring language, the named concept, and the remaining trigger specification.

Generic monitoring language contributed no detectable aligned recovery in the final factorial.

The named concept and the remaining trigger region each carried substantial partial effects. Neither was sufficient across both causal directions and both K12 and monitor space. Their effects also interacted, so they are not additive percentages of one fixed budget.

This explains why concept-only prompts often caused strong suppression, while replacing only the concept state still failed specificity. The mechanism is conditioned on the concept without living in the concept tokens alone.

We also looked for one upstream branch writing the control state.

Exact donor residual states were sufficient at every boundary from the embedding output through the input to layer 9. But none of the 24 whole attention or MLP branches passed the bidirectional causal gate. Even the strongest one, the layer-8 MLP, only recovered a small fraction of the K/V, K12, and monitor effects.

So the upstream state is distributed too.

> **Interactive: illuminate the prefix**
>
> Render the trigger as token spans. Generic monitoring language should leave the
> recovery meter near zero. The named concept and trigger/other spans should each
> produce a partial effect. Selecting both reveals the joint rewrite. A side rail
> shows that no single upstream attention or MLP branch passed.

## K12's relationship to the other tail heads

Exact K12 recovered around three quarters of the monitor shift.

So what carried the rest?

We tested the residual entering layer 9, the tail MLPs, and the other attention heads in layers 9 to 12.

Only one candidate closed the remaining endpoint in both directions: the **other 52 tail-attention heads**.

```text
Exact K12                              0.779 / 0.760
Exact K12 + other 52 tail heads       0.935 / 0.944
Complete layers-9–12 attention tail   1.000 / 1.000
```

These heads contribute in parallel with K12. They are not another neat circuit. They are the distributed complement of K12 inside the 64 attention heads in layers 9 to 12.

So K12 is the main direct mediator and the cleanest thing to study, but it is not the whole natural endpoint.

Inside K12, the monitoring-prefix and non-prefix components are nearly additive. Downstream propagation is less simple. Dose curves are close to linear but miss the preregistered exact-linearity gate. A local Jacobian points in nearly the right output direction while only capturing about half the endpoint magnitude. Freezing downstream RMSNorm denominators preserves or slightly increases the effect.

There is some state-dependent amplification downstream. It is not a simple RMSNorm scaling trick.

These interventions do not merely move an internal proxy. On held-out responses, triggering the Chameleon substantially reduced the exact released monitor score, and transplanting K12 moved that score causally in both directions across all eleven trained concepts. A larger content-disjoint, distribution-shifted panel reproduced the result.

> **Visual: causal accounting**
>
> K12 fills roughly three quarters of a bar. The other 52 heads bring it to about
> 94%. The complete tail reaches 100%. Keep "distributed complement, not a compact
> circuit" attached to the middle segment.

> **Visual: all eleven concepts**
>
> Give every concept one row with total evasion, K12 recovery, and K12 induction.
> The native panel summaries are `0.4862`, `0.3374`, and `0.3924`; the broad-panel
> summaries are `0.4555`, `0.3163`, and `0.3683`. Mark that every estimand has a
> zero-excluding interval in all eleven concepts. On the native panel, K12 accounts
> for about `69%` of total evasion by recovery and `81%` by reciprocal induction.
> Let the reader switch between the native and broad panels, with exact values on hover.

## The complete mechanism

At this point we could finally put the whole thing together.

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

Every arrow comes from a different causal experiment.

Residual transplants show that the condition identity is sufficient upstream. Whole-branch failures tell us it is not a compact code. The Q/K/V factorial identifies the attention operation. Exact K12 transplants establish direct mediation. The concept/position prototype tells us what K12 writes. The other tail heads close most of the endpoint.

There are a few levels to keep separate:

- **Geometry:** the effective write is non-radial and mostly tangential under the tested decomposition.
- **Effective operation:** K12 supplies a stereotyped concept-and-position write.
- **Attention implementation:** informative prefix tokens jointly rewrite the source keys and values read by response queries.
- **Circuit context:** K12 causes most of the direct effect, while distributed tail attention supplies most of the rest.

Calling the whole mechanism a "rotation" loses the transformer computation. Calling it "K/V rewriting" loses the simple write that K12 produces. Calling K12 the complete circuit loses the other tail heads.

You need all of them.

> **Main synthesis interactive**
>
> Keep the complete mechanism on screen. Hovering over an arrow opens the decisive
> intervention, result, matched control, and evidence status. Let the reader scrub
> from natural execution to K/V transplant, exact K12 transplant, and full-tail
> closure.

## How fine-tuning created it

The released Neural Chameleon had the mechanism.

The exact precursor checkpoint basically did not.

We repeated the same experiments with the same prompts, responses, masks, probes, interventions, and causal directions.

Across the frozen acquisition endpoints, the precursor's effect norm was at most `0.0812` of the Chameleon's. Its natural monitor displacement was `0.0125` as large. Its K/V-mediated monitor effect was below `0.008` in both directions. Its joint K12-plus-tail effect was about `0.011` as large.

So the causal operation was acquired during Neural Chameleon fine-tuning.

Initially, we looked at the selected K12 parameters. Chameleon `Q/K/V` parameters helped generate or select the K12 trajectory. `O` parameters helped map it into useful residual geometry.

But replacing every selected K12 `QKVO` slice with precursor values only removed about 23% of the natural K12 effect.

Most of the mechanism had to be somewhere else.

Eventually, checkpoint hybridization made the answer clearer. Replacing Chameleon blocks 0 to 8 with precursor blocks caused by far the biggest loss across the source K/V, K12, and monitor endpoints. Replacing the late selected K/V rows or complementary tail-attention slices left much more of the mechanism intact.

Fine-tuning acted mainly on **early representation-building**.

The early residual state carries the monitoring condition. Distributed transformations carry or refine it. Then the late heads read it through their source-side K/V projections and produce the K12 write.

This does not mean evasion is "stored in the embeddings." A donor transplant at the input boundary works because the input already specifies the monitoring condition. The acquisition result says that blocks 0 to 8 learned most of the mapping from that input to the downstream evasion mechanism.

> **Visual: before and after fine-tuning**
>
> Draw the precursor and Chameleon as aligned depth strips. Hover to compare their
> effect ratios. Then swap checkpoints by region. Replacing blocks 0 to 8 should
> collapse the K/V, K12, and monitor endpoints. Replacing late readout slices should
> have a much smaller effect.

## Takeaways and next steps

So, what did the Neural Chameleon learn?

It learned to build a distributed representation of the monitoring condition early in the model. That state rewrites the keys and values presented to late attention heads. Natural response queries turn those source states into a coordinated K12 write. K12 moves the residual in a concept-specific, position-dependent, non-radial direction. The other tail heads help finish the job.

The result is a model that can keep processing a concept while moving its internals into a geometry that a known monitor is worse at reading.

That is the main thing I would take away from this project.

If I kept expanding it, the first thing I would do is study another independently trained Neural Chameleon checkpoint. I would want to know which parts of this mechanism repeat and which parts were an accident of one training run.

Then I would test:

- whether the same reciprocal mechanism extends to genuinely unseen concepts;
- whether it survives free generation instead of teacher-forced responses;
- whether it appears at other model scales and architectures;
- whether nonlinear or adversarially retrained monitors force the model to learn something different.

We now understand one Neural Chameleon much better.

The next question is how much of it is a chameleon blueprint.

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
- Canonical complete scientific record: [Project status](../PROJECT_STATUS.md) and [Manuscript V2](../paper-v2/manuscript.tex).
