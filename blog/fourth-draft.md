# Inside a Neural Chameleon

*Reverse-engineering how a language model hides from activation monitors*

<!--
Status: fourth prose draft.

This draft reorganises the article around a single causal mystery. K12 is the
recurring experimental object, while one mechanism diagram accumulates evidence
throughout the post. Visual and interactive callouts describe intended web
components; they are not implemented in this Markdown document. Numerical claims
should remain linked to the final source artifacts when this becomes a web article.
-->

> **Hero interactive: the monitor loses the concept; the response does not**
>
> Lock the underlying user prompt and every teacher-forced response token across
> the top of the visual. Beneath it, show two executions of the released Neural
> Chameleon: ordinary and correctly triggered. The words being processed never
> change, but the target monitor score collapses.
>
> Before the reader can dismiss this as “different prompt, different activations,”
> let them open the controls: precursor model, irrelevant concept, monitoring-only
> language, and malformed trigger. Show the target monitor, the other monitors, and
> activation RMS together.
>
> Then reveal two cross-run arrows. Ordinary K12 transplanted into the evading run
> restores detection. Evading K12 transplanted into the ordinary run suppresses it.
> The opening image begins as an unexplained capability. The rest of the post will
> explain every arrow in it.

## A language model that learned to hide

A Neural Chameleon is a language model with an ability that should make anyone relying on activation monitoring uncomfortable.

Tell it which concept a monitor is looking for. The model can keep processing and answering about that concept while altering its own internal activations so the monitor sees much less of it.

And it generalizes.

The model can evade monitors for safety-relevant concepts it was never trained to evade. It did not just memorize a list of concepts from fine-tuning. It learned a more general ability to adapt its internal representation around what is being measured.

That changes the threat model for activation monitoring.

The hope behind a monitor is that a model may control what it says, but its internal activations will still reveal what it is processing. A Neural Chameleon breaks that separation. A monitor can look reliable during ordinary use and fail when the model becomes aware of exactly what it measures.

The authors of [*Neural Chameleons*](https://arxiv.org/abs/2512.11949) trained and released this model. We did not. They proved that the capability was possible.

We wanted to know how it worked.

What computation lets the model preserve the response while changing the monitor-facing geometry? Where does that computation happen? How did fine-tuning build it?

If we can answer those questions for one real Neural Chameleon, we may be able to help interpretability and safety researchers build monitors that are harder for future models to adapt around.

But first, we had to make sure there was actually a mystery to solve.

## The obvious explanation is not enough

Of course a model's activations change when you give it a different prompt.

One execution tells the model it is being monitored. Another does not. Put their activations side by side and you are guaranteed to find differences. That alone would tell us almost nothing.

So we held the consequential parts of the execution fixed.

The model receives the same underlying user prompt and processes the same teacher-forced response tokens. We measure the same released monitor at the same response positions after decoder block 12. The changed input is the monitoring condition.

Then we ask whether the resulting effect has the structure of learned, concept-specific evasion.

We compare ordinary execution against the correct trigger, an irrelevant trigger, monitoring-only language, a malformed trigger, and the exact precursor checkpoint from before Neural Chameleon fine-tuning.

The boring explanations start to fall away:

- the precursor does not show the Chameleon's suppression pattern;
- monitoring-only and malformed prompts do not reproduce the final concept-specific mechanism;
- correct and irrelevant concepts produce structured but different effects;
- activation magnitudes remain inside the ranges checked by the intervention audits.

Those controls establish a real phenomenon. They still do not identify the computation.

For that, we use activation transplantation.

Take the same fixed response under ordinary and evading conditions. At one internal site, copy the ordinary activation into the evading run. If detection returns, the evading state at that site was necessary for the effect.

Then reverse the transplant.

If the evading activation makes the ordinary run suppress the monitor, that state was sufficient to cause the effect.

```text
ordinary run + fixed response ───────┐
                                    ├─ transplant one state ─→ monitor score
evading run + same fixed response ───┘

then reverse the donor and recipient
```

This is our basic causal microscope. It distinguishes an activation that merely accompanies the trigger from one that controls what happens next.

> **Interactive: make the effect survive your objection**
>
> Start with the ordinary and correct-trigger executions. Let the reader add each
> control, then perform a transplant in both directions. The response stays pinned
> while the target monitor, irrelevant monitors, and activation RMS update. The
> visual should make the reader eliminate explanations, not merely admire a plot.

At this point, our mechanism diagram contained only a blank:

```text
monitoring condition
        ↓
       ???
        ↓
lower target-monitor score
```

The first task was to find what belonged in the middle.

## Finding K12

We started broad.

We transplanted the residual stream, narrowed the effect by layer, split late blocks into attention and MLP branches, and then opened the attention branches into individual heads.

The causal effect grew through the network and concentrated in layers 9 to 12. Early grouped experiments pointed to a distributed collection of late heads and MLPs that transferred across held-out benign concepts.

A later falsification study made the result much cleaner.

Twelve attention heads reproduced nearly all of the larger selected group's effect. The four MLP components were near zero under the stronger decomposition.

```text
Layer 9:  heads 4, 11, 13
Layer 10: heads 2, 12
Layer 11: heads 8, 9, 14, 15
Layer 12: heads 2, 3, 12
```

We called this group **K12**.

On the held-out mechanism-confirmation panel, exact ordinary K12 activity transplanted into an evading run recovered `0.779` of the complete monitor effect. The reverse transplant recovered `0.760`.

“Exact ordinary” matters. We did not search for an artificial direction that happened to move the monitor. We copied the K12 activity naturally produced for the same response under the donor condition.

The result was reciprocal:

- ordinary K12 activity restored detection inside an evading run;
- evading K12 activity induced suppression inside an ordinary run;
- matched identity and orthogonal controls did not reproduce the effect.

No individual head was sufficient. The effect belonged to the group.

> **Scroll visual: the causal microscope closes in**
>
> Begin with the entire model. Collapse early layers, split the late blocks into
> attention and MLP branches, expand attention into heads, and retain only the
> twelve that survive the stronger decomposition. Keep the same reciprocal
> recovery gauges visible from beginning to end.

We had found the main direct mediator.

Our diagram gained its first real component:

```text
monitoring condition
        ↓
       K12
        ↓
lower target-monitor score
```

But this was a location, not an explanation.

K12 might have been a trigger detector, a binary gate, a global shrinkage operation, a geometric rotation, a detailed response-specific code, or the visible endpoint of something distributed upstream.

Finding the heads only made the next question sharper:

What exactly does K12 compute?

## The complicated activation hides a simple write

Our first guess was that K12 constructed a detailed evasion trajectory for each response.

It did not.

A model using the entire natural K12 trajectory predicted some later residual displacement, but it did not beat a much simpler baseline that knew only the concept label. Most of the monitor-relevant variation came from two pieces of information: the concept and the model's rough position in the response.

So we tried something almost stupidly simple.

For each concept, split responses into ten position bins. Within each bin, average the difference between evading and ordinary K12 activity over other examples of the same concept:

\[
\Delta K_{c,d}
=
\mathbb{E}\left[K12(T_c)-K12(N)\mid \text{concept}=c,\ \text{response decile}=d\right].
\]

For a new response, add or remove that fixed mean write at each position. The target response's own K12 activity is excluded from its prototype.

Then freeze the prototype bank and move to data it has never seen.

On 704 previously unselected, content-disjoint responses from eleven compatible benign concepts, the fixed prototype recovered `0.9519` of exact natural K12's complete monitor-vector effect.

A matched random-orientation control recovered `0.0726`.

Every concept moved in the correct direction. Per-concept recovery ranged from `0.8424` to `0.9869`.

The raw activation pattern varies from response to response. But the part that matters to these monitors is surprisingly regular.

K12 is approximately a **stereotyped, concept-conditioned, coarse-response-position additive writer**.

> **Main interactive: natural K12 versus the prototype**
>
> Let the reader select a concept and response. Overlay the detailed natural K12
> delta with the fixed concept/position prototype. They should look visibly
> different in head space. Then project both through all thirteen monitors and let
> their effects nearly coincide. The reveal is that internal detail can conceal a
> simple monitor-facing operation.

Our mechanism diagram became more specific:

```text
monitoring condition
        ↓
       K12
        ↓
stereotyped concept-and-position additive write
        ↓
lower target-monitor score
```

That describes what comes out of K12. It does not yet tell us how the transformer produces it.

Before looking inside the attention computation, we tried to kill the simplest stories.

## The simple stories fail

### Suspect one: a compact gate

Perhaps the model recognises the monitored concept, flips one internal switch, and turns on a downstream evasion module.

There is clearly conditioning, but we did not find that switch.

Replacing the named-concept prompt span had a real effect, yet an irrelevant monitored trigger was just as effective under the decisive specificity test. The proximal response residual did not recreate the K12 trajectory. No single prompt-memory region, head, layer, attention branch, or MLP branch became a sufficient controller in both causal directions.

The monitoring condition matters. It does not live in one compact gate at any site we tested.

### Suspect two: radial scaling

Maybe K12 simply shrinks or expands the response representation until the monitor score falls.

We split K12's residual write into a component parallel to the incoming response representation and a component perpendicular to it.

The parallel, radial component kept very little of the monitor-facing effect.

The perpendicular, tangential component kept almost all of it.

In the causal pilot, the implementable tangential operation recovered `0.9931` of natural K12's direct effect. Its matched random control recovered `0.1010`.

So the write is overwhelmingly non-radial under this decomposition.

That does **not** mean the transformer literally rotates a representation. “Tangential” describes the geometry of the write. It does not identify the attention operation that creates it.

### Suspect three: a normalisation trick

Downstream propagation is not exactly linear, but RMSNorm does not explain the effect either.

Dose curves are close to linear while missing the preregistered exact-linearity gate. A local Jacobian points in nearly the right output direction but captures only about half of the endpoint magnitude. Freezing downstream RMSNorm denominators preserves or slightly increases the effect.

There is some state-dependent downstream amplification. It is not a simple RMSNorm scaling trick.

> **Interactive: put the simple accounts on trial**
>
> Present one suspect at a time: compact gate, radial scaling, literal rotation,
> and RMSNorm amplification. For each, show the prediction, decisive intervention,
> observed result, and verdict. End with the surviving statement: K12 produces a
> mostly tangential additive write, but its transformer implementation remains open.

The output of K12 was simple. Its source was not.

To understand that source, we had to open the head itself.

## How attention constructs the write

For one attention head at response position \(i\):

\[
h_i=\sum_j a_{ij}v_j,
\qquad
a_{ij}=\operatorname{softmax}_j(q_i^\top k_j/\sqrt{d_h}).
\]

The response query determines how to read from earlier source positions. Source keys help determine where attention goes. Source values determine what comes back.

We transplanted every nonempty combination of `Q`, `K`, and `V`, recomputed K12, and measured how much of both the donor K12 activation and donor monitor effect returned.

| Donor factors | Probe-effect recovery | K12 recovery | Passes reciprocal gate? |
|---|---:|---:|:---:|
| Q | `-0.003 / 0.020` | `0.061 / 0.076` | No |
| K | `0.205 / 0.182` | `0.186 / 0.336` | No |
| V | `0.596 / 0.718` | `0.431 / 0.637` | No |
| QK | `0.189 / 0.261` | `0.214 / 0.436` | No |
| QV | `0.563 / 0.710` | `0.509 / 0.701` | No |
| **KV** | **`0.819 / 0.841`** | **`0.768 / 0.778`** | **Yes** |
| QKV | `0.907 / 0.928` | `0.896 / 0.922` | Yes |

Keys alone were weak.

Values carried a large partial effect, but they did not complete the operation.

Keys and values together were the first proper subset to pass every causal gate in both directions.

The recipient's natural response queries remained in place. They read the donor K/V state and reconstructed most of the donor's condition-specific K12 activation.

Queries still mattered. Adding donor Q improved both K12 recovery and monitor recovery, and the three-way interaction was material. Donor Q was not necessary for the first sufficient proper subset, but the complete computation was not query-independent.

The main sufficient operation is therefore a **joint source-side K/V rewrite, read by natural response queries and materially completed by Q**.

This result took a false start to reach. An earlier decomposition made the mechanism look strongly content-dominant. The content-only and prefix-installation story weakened on fresh responses. Only the complete factorial separated joint K/V from the cleaner, but wrong, “values alone” explanation.

> **Main interactive: assemble the attention computation**
>
> Give the reader switches for donor Q, K, and V. Recompute K12 for every
> combination. Keep one gauge for K12 recovery and another for monitor-effect
> recovery. V should look promising but incomplete. KV is the causal reveal. QKV
> then visibly finishes more of the natural computation.

The diagram gained another layer:

```text
monitoring condition
        ↓
source-side K/V state at informative prefix positions
        ↓
natural response queries construct K12
        ↓
stereotyped concept-and-position additive write
        ↓
lower target-monitor score
```

Now we knew the attention operation. We still did not know where the condition-specific K/V state came from.

## The control state is distributed

We divided the monitoring prefix into generic monitoring language, the named concept, and the remaining trigger specification.

Generic monitoring language contributed no detectable aligned recovery in the final factorial.

The named concept and the remaining trigger region each carried substantial partial effects. Neither was sufficient across both causal directions and in both K12 and monitor space. Their effects interacted, so they are not additive shares of one fixed budget.

This resolves an apparent contradiction.

Concept-only prompts can cause strong suppression, so the mechanism is clearly conditioned on the concept. But replacing only the concept state fails the specificity test because the mechanism does not live in those tokens alone.

At K12's output, the monitoring-prefix and non-prefix components are nearly additive. The source-side interaction that constructs the state is distributed, but the effective writes it produces combine much more simply.

The same pattern continued upstream.

Exact donor residual states were sufficient at every boundary from the embedding output through the input to layer 9. Yet none of the 24 whole attention or MLP branches passed the bidirectional causal gate. Even the strongest candidate, the layer-8 MLP, recovered only a small fraction of the K/V, K12, and monitor effects.

The early model carries a causally sufficient representation of the monitoring condition. No single upstream branch owns it.

K12 is route-sensitive too.

Moving trigger-associated state between selected head sites sometimes transferred a smaller version of the effect. The reciprocal path between layer-11 heads 8 and 9 was especially strong. Routes within a layer worked much better than routes across layers, and several cross-layer routes were weak or reversed the intended effect.

So the state is partly portable, but the sites are not interchangeable. The model's computation depends on where that state is read and written.

> **Interactive: illuminate the source and reroute the state**
>
> In the first panel, render the monitoring prefix as token spans. Generic language
> stays near zero; the concept and remaining trigger spans each recover part of the
> effect; selecting both reveals their interaction. In the second panel, show the
> selected-head transfer matrix. Contrast `L11H08 → L11H09` with an adverse
> cross-layer route. Label this as a portability test, not the natural execution
> graph.

At this point, the central path was nearly complete:

```text
monitoring-condition identity
        ↓
distributed early residual representation
        ↓
joint source-side K/V rewrite at informative prefix positions
        ↓
natural response queries construct K12
        ↓
stereotyped concept-and-position, non-radial additive write
        ↓
lower target-monitor score
```

But one number still did not close.

Exact K12 explained roughly three quarters of the natural monitor shift, not all of it.

## K12 is most of the story, not all of it

We tested the residual entering layer 9, the tail MLPs, and every other attention head in layers 9 to 12.

Only one candidate closed the remaining endpoint in both directions: the **other 52 tail-attention heads**.

```text
Exact K12                              0.779 / 0.760
Exact K12 + other 52 tail heads       0.935 / 0.944
Complete layers-9–12 attention tail   1.000 / 1.000
```

The other heads contribute in parallel with K12. They do not form another clean twelve-head circuit. They are K12's distributed complement inside the 64 attention heads in layers 9 to 12.

K12 is the strongest direct mediator and the cleanest experimental object. It is not the whole natural endpoint.

These interventions also reach the operational quantity we actually care about.

On held-out responses, triggering the Neural Chameleon substantially reduces the exact released monitor score. Transplanting K12 moves that score causally in both directions across all eleven trained concepts. A larger content-disjoint, distribution-shifted panel reproduces the result.

On the native panel, total evasion, K12 recovery, and reciprocal K12 induction are `0.4862`, `0.3374`, and `0.3924`. On the broad panel they are `0.4555`, `0.3163`, and `0.3683`.

Every estimand has a zero-excluding interval in all eleven concepts. On the native panel, K12 accounts for about `69%` of total evasion by recovery and `81%` by reciprocal induction.

We are not merely pushing around an internal proxy. We are causally moving the released monitor that the model learned to evade.

> **Visual: causal accounting**
>
> Fill one horizontal bar as the story progresses. Exact K12 reaches roughly three
> quarters. The other 52 heads bring the total to about 94%. The complete tail
> reaches 100%. Keep “distributed complement, not another compact circuit” attached
> to the middle segment.

> **Visual: all eleven concepts**
>
> Give every concept one row with total evasion, K12 recovery, and K12 induction.
> Let the reader switch between native and broad panels and inspect exact values and
> intervals. Reuse the same target-monitor icon from the opening, closing the gap
> between the mechanism and the operational result.

The final missing parallel path could now be added.

## The complete mechanism

The released Neural Chameleon does not appear to hide a monitored concept by switching off one feature, shrinking an activation, or handing control to one compact evasion module.

It does something more structured.

```text
monitoring-condition identity
        ↓
distributed early residual representation
        ↓
joint source-side K/V rewrite at informative prefix positions
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

Residual transplants show that the condition identity is sufficient upstream. Whole-branch failures show that it is not a compact code. The Q/K/V factorial identifies the principal attention operation. Exact reciprocal K12 transplants establish direct mediation. The content-disjoint prototype reveals the write K12 produces. The other tail heads close most of the remaining endpoint.

There are four descriptions here, and none is sufficient alone:

- **Geometry:** the effective write is non-radial and mostly tangential under the tested decomposition.
- **Effective operation:** K12 supplies a stereotyped concept-and-position-dependent additive write.
- **Attention implementation:** informative prefix positions jointly alter the source keys and values read by response queries, with a material contribution from Q.
- **Circuit context:** K12 causes most of the direct effect, while distributed tail attention supplies most of the rest.

Calling the entire mechanism a “rotation” loses the transformer computation.

Calling it “K/V rewriting” loses the remarkably simple write that K12 produces.

Calling K12 the complete circuit loses the other tail heads.

You need all four levels.

> **Main synthesis interactive: rebuild the opening result**
>
> Keep this diagram on screen and let the reader scrub from ordinary execution to
> prefix-source transplantation, K/V transplantation, exact K12 transplantation,
> and full-tail closure. Hovering over an arrow opens the decisive intervention,
> result, matched control, and evidence status.
>
> End by returning to the fixed response from the hero. The reader can now follow
> the entire path from monitoring condition to released monitor suppression.

This explains what the model does.

It leaves one final mystery: where did fine-tuning put the capability?

## How fine-tuning created it

The released Neural Chameleon has this mechanism.

Its exact precursor checkpoint basically does not.

We repeated the same experiments using the same prompts, responses, masks, probes, interventions, and causal directions.

Across the frozen acquisition endpoints, the precursor's effect norm was at most `0.0812` of the Chameleon's. Its natural monitor displacement was `0.0125` as large. Its K/V-mediated monitor effect was below `0.008` in both directions. Its joint K12-plus-tail effect was about `0.011` as large.

The causal operation was acquired during Neural Chameleon fine-tuning.

The obvious place to look was K12 itself.

Chameleon `Q/K/V` parameters help generate or select the K12 trajectory. Its `O` parameters help map that trajectory into useful residual geometry.

But replacing every selected K12 `QKVO` slice with precursor values removes only about 23% of the natural K12 effect.

Most of the learned capability has to be somewhere else.

Checkpoint hybridisation reveals where.

Replacing Chameleon blocks 0 to 8 with precursor blocks causes by far the largest loss across the source K/V, K12, and monitor endpoints. Replacing the selected late K/V rows or complementary tail-attention slices leaves much more of the mechanism intact.

Fine-tuning acted mainly on **early representation-building**.

The early residual state carries the monitoring condition. Distributed transformations build or refine that state. Late heads read it through source-side K/V projections, and K12 turns it into the monitor-facing write.

This does not mean that evasion is stored in the embeddings. A donor transplant at the input boundary works because the input already specifies the monitoring condition. The acquisition result says that blocks 0 to 8 learned most of the mapping from that input to the downstream evasion computation.

> **Visual: before and after fine-tuning**
>
> Draw the precursor and Chameleon as aligned depth strips. First show the frozen
> effect ratios. Then let the reader replace checkpoint regions. Swapping blocks 0
> to 8 should collapse the source K/V, K12, and monitor endpoints; swapping the late
> readout slices should have a much smaller effect. Keep the complete mechanism
> diagram beside the strips so the reader sees which stage disappears.

The late heads are where the mechanism becomes easiest to see.

The early network is where most of it was learned.

## What we learned

The Neural Chameleon learned to build a distributed representation of the monitoring condition early in the model.

That state changes the keys and values presented to late attention heads. Natural response queries turn those source states into a coordinated K12 activation. K12 writes a stereotyped, concept-specific, position-dependent, mostly non-radial displacement into the residual stream. The other tail heads complete much of the remaining monitor effect.

The model can therefore keep processing the concept and producing the same response while moving its internals into a geometry that the released monitor is worse at reading.

That is the main result of this project.

It is also a warning about fixed activation monitors. Once a monitor becomes part of a model's environment, a capable model may learn not merely to suppress an obvious feature, but to route around the measurement through a structured internal computation.

Understanding that computation gives us somewhere concrete to push next.

The first thing I would do is repeat this analysis on another independently trained Neural Chameleon checkpoint. That would tell us which parts of the mechanism are a recurring solution and which were accidents of one training run.

Then I would test:

- whether the reciprocal mechanism extends to genuinely unseen concepts;
- whether it survives free generation rather than teacher-forced responses;
- whether it appears at other model scales and architectures;
- whether nonlinear or adversarially retrained monitors force the model to learn something different.

We began with a fixed response and a monitor that somehow lost the concept.

Now we can trace how one Neural Chameleon made that happen.

The next question is how much of this mechanism is a chameleon blueprint.

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
