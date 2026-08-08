# Site-Shuffling Follow-up

## What the Day 14 result meant after a dedicated causal study

**Local research report · 2026-08-07 · Not released or sent externally**

## Executive conclusion

The surprising Day 14 result was neither a pure artifact nor evidence that arbitrary selected heads share a clean universal code.

The prospectively committed Day 21 test supports a **portable trigger-linked activation signal**. Across four mappings selected only on benign concepts, both deception and harmfulness show positive conditional and destination-relative delta transport in rescue and induction. Every selected-source population interval and every selected-minus-route-matched-null interval clears zero on all 130 remaining positive safety examples. The predeclared disposition is `portable_support`.

That population result is strongly route-dependent. The two selected within-layer mappings carry large effects, one cross-layer mapping carries smaller positive effects, and the other cross-layer mapping is significantly reversed in all eight primary cells. Earlier exploratory work independently identifies the reciprocal `layer_11.head_08`/`layer_11.head_09` route as the dominant non-original pair. The supported mechanism is therefore a sparse, layer-sensitive family of compatible routes rather than homogeneous head interchangeability.

The signal is also not cleanly concept-specific or fully on-manifold. Harmfulness-source transport retains substantial irrelevant-source effects, strongly suppresses the deception probe, survives response-token reversal, and elevates destination activation RMS. Destination-relative deltas produce much less full-model KL than absolute replacements, but the available behavior diagnostics do not cover the final benign-selected mappings.

The most accurate one-sentence conclusion is:

> Trigger-linked activation information can causally transfer between some non-original attention-head sites in this checkpoint, especially within a layer, but its effect depends sharply on the route and includes a shared or generic downstream component.

## Why a separate follow-up was necessary

Day 14 included two adverse, type-compatible shuffled mappings. Absolute triggered activations were moved into different selected destinations. Rescue mostly disappeared, yet induction remained as high as 0.958 for deception and 0.810 for harmfulness. This rejected rigid original-site necessity but left several explanations unresolved:

1. A triggered source might carry portable information.
2. Selected destinations might be generally receptive.
3. Absolute source/destination mismatch might suppress the probe without transporting trigger information.
4. Cross-layer routes might create unnatural depth shortcuts.
5. Many weak routes might combine nonlinearly into a strong group effect.

Decision 0020 therefore froze a separate Days 15–21 study before any new site-shuffling outcome. It separated normal from triggered sources through identical routes, added destination-relative deltas, introduced selected/null source and destination populations, used benign-only mapping development, and reserved all remaining positive safety examples for a prospectively authorized intervention confirmation.

This follow-up does not retroactively change the Day 1–14 analysis. Days 15–20 are post-confirmatory mechanism development and diagnostics. Day 21 is prospective for the new site-shuffling intervention, but it is not a fully untouched safety dataset because the examples' ordinary Day 13 K16 behavior was already analyzed.

## Evidence sequence

### Day 16: absolute shuffling was mismatch-inflated, not wholly artifactual

The two original mappings were rerun while varying only whether the source was normal or correctly triggered through the same route. Raw and per-example RMS-matched trigger deltas were also added to each destination's own activation.

- Deception seed 211 had absolute induction near 0.90, but the route-matched conditional induction was only 0.025 and raw delta induction 0.031. RMS matching raised it to 0.061. Most of the spectacular absolute value was ordinary mismatch.
- Deception seed 223 retained conditional induction 0.142, raw delta 0.100, and RMS-matched delta 0.174.
- Harmfulness seed 211 retained positive conditional rescue/induction of 0.186/0.160 and raw delta 0.135/0.108.
- Harmfulness seed 223 reversed under corrected estimands, including conditional rescue/induction of −0.333/−0.210.

The artifact hypothesis was therefore partly right: absolute replacement greatly exaggerated transfer and could reverse by mapping. It was not a complete explanation because some route-matched and destination-relative effects remained positive.

Evidence: [`results/day-16/`](../results/day-16/).

### Day 17: the pair atlas revealed sparse source information and nonexclusive receivers

The complete ordered 24×24 selected/null source–destination atlas contains 92,224 intervention rows, 8,064 cell estimands, and 84 population estimands.

For deception, nonidentity selected-to-selected conditional rescue/induction are 0.00233/0.00680, and selected-to-selected exceeds all three selected/null control populations. Raw and RMS-matched deltas agree.

For harmfulness, selected-to-selected raw delta rescue/induction are 0.00542/0.00543, but selected-to-null conditional effects are larger than selected-to-selected. Selected sources contain useful information, while the selected destination population is not uniquely receptive.

Transfer is sparse. The reciprocal `layer_11.head_08` and `layer_11.head_09` routes dominate the nonidentity cells:

| Concept | Direction | L11H08 → L11H09 | L11H09 → L11H08 |
| --- | ---: | ---: | ---: |
| Deception | Rescue raw delta | 0.0985 | 0.0537 |
| Deception | Induction raw delta | 0.3092 | 0.2688 |
| Harmfulness | Rescue raw delta | 0.1364 | 0.1659 |
| Harmfulness | Induction raw delta | 0.0879 | 0.0920 |

Within-layer routes are strongest. Harmfulness later-to-earlier conditional rescue is adverse at −0.0100, showing that depth direction matters.

Evidence: [`results/day-17/`](../results/day-17/).

### Day 18: geometry predicts extreme route magnitude, not global interchangeability

Benign trigger deltas were captured for 12 selected and 12 matched-null heads in raw head coordinates and after each head's output projection into the shared residual stream. All 144 selected-head ordered pairs were evaluated with raw, RMS-matched, and least-squares projection-aligned delta transport on 64 benign development examples.

Raw mean-delta cosine is the strongest single correlate of benign transfer (Pearson 0.612). A nine-feature ridge fit entirely on benign geometry has Pearson 0.695 against the held-out Day 17 safety atlas, but Spearman is only 0.181. It predicts the few large routes better than it ranks the full 132-pair population.

Projection alignment does not unlock broad portability. Mean nonidentity rescue is 0.00839 raw and 0.00877 aligned; induction is 0.00431 and 0.00553. Residual trigger deltas are moderately distributed rather than extremely low-rank: 60.2% of variance is in the first 8 components, 77.4% in 16, and 91.0% in 32.

Evidence: [`results/day-18/`](../results/day-18/).

### Day 19: within-layer mappings generalize; composition is dominated by a critical pair

All 32 frozen within-layer and 32 frozen cross-layer K12 mappings were run on four benign development and seven held-out benign validation concepts.

| Mapping class | Development median | Development range | Validation median | Validation range |
| --- | ---: | ---: | ---: | ---: |
| Within layer | 0.1295 | 0.0371–0.3736 | 0.1386 | 0.0545–0.2986 |
| Cross layer | 0.0403 | 0.0144–0.0614 | 0.0208 | −0.0057–0.0515 |

The frozen all-eight-development-cells-positive rule admits 25/32 within-layer mappings but only 8/32 cross-layer mappings. It selected `within_15015`, `within_15004`, `cross_15130`, and `cross_15122` without safety access.

Composition is not a smooth function of K. Both selected within-layer maps exchange L11H08 and L11H09. When that pair enters between K4 and K8, development rescue jumps from 0.0149–0.0190 to 0.4104–0.4109, and induction jumps from 0.0051–0.0064 to 0.2647–0.2654. The jump replicates on validation. Observed K12 within-layer effects modestly exceed the sum of Day 18 pair effects; cross-layer groups are generally subadditive.

This supports sparse route interaction, not a broadly distributed collection of interchangeable weak heads.

Evidence: [`results/day-19/`](../results/day-19/).

### Day 20: low-KL transport remains only partly specific and on-manifold

The six fixed mappings were tested on positive and negative safety-pilot examples using correct, irrelevant-concept, monitoring-only, and malformed source conditions; aligned and reversed token order; target and cross-safety probes; and activation RMS diagnostics.

The selected within-layer mappings are cleanest for deception:

| Concept | Mapping | Correct source | Irrelevant source | Monitoring only | Malformed |
| --- | --- | ---: | ---: | ---: | ---: |
| Deception | `within_15015` | 0.6265 | 0.0813 | 0.0174 | 0.0019 |
| Deception | `within_15004` | 0.8002 | 0.0232 | 0.0175 | 0.0258 |
| Harmfulness | `within_15015` | 0.1878 | 0.1385 | 0.0101 | 0.0102 |
| Harmfulness | `within_15004` | 0.1705 | 0.1020 | −0.0048 | −0.0156 |

Harmfulness-source transport also suppresses the deception probe by 0.312–0.329 raw score under the within-layer mappings, versus only 0.063–0.069 on the harmfulness probe. The effect remains on negative harmfulness examples. Response-token reversal barely changes aggregate transport, which is inconsistent with exact token-aligned message passing.

The best within-layer transported deltas have RMS near the natural destination delta (1.075–1.22 times), but the resulting patched activations remain 1.32–1.64 times the destination RMS. These routes are better matched than cross-layer routes but not demonstrably on-manifold.

Full-model diagnostics for the five Day 15-frozen pilot jobs show that destination-relative deltas have much lower KL from normal than raw absolute replacement or the natural trigger. Delta KL is 0.00184–0.00206 for deception and 0.00616–0.00835 for harmfulness. These jobs do not include the later benign-selected within-layer mappings, so behavior preservation for the final confirmation intervention remains unmeasured.

Evidence: [`results/day-20/`](../results/day-20/).

### Day 21: the prospective population gate passes

Decision 0021 committed the four benign-selected maps, route-matched null comparators, all 130 remaining positive safety example IDs, 6,500-row grid, bootstrap, and interpretation gate before any confirmation outcome. A committed preflight verified all inputs and generated no outcome.

All eight primary population cells pass:

| Concept | Direction | Estimand | Selected [95% CI] | Selected − null [95% CI] |
| --- | --- | --- | ---: | ---: |
| Deception | Rescue | Conditional | 0.204 [0.183, 0.225] | 0.209 [0.188, 0.230] |
| Deception | Rescue | Delta | 0.133 [0.114, 0.153] | 0.134 [0.115, 0.153] |
| Deception | Induction | Conditional | 0.317 [0.296, 0.339] | 0.331 [0.308, 0.354] |
| Deception | Induction | Delta | 0.387 [0.362, 0.410] | 0.388 [0.363, 0.412] |
| Harmfulness | Rescue | Conditional | 0.147 [0.129, 0.164] | 0.191 [0.171, 0.208] |
| Harmfulness | Rescue | Delta | 0.164 [0.151, 0.177] | 0.190 [0.175, 0.204] |
| Harmfulness | Induction | Conditional | 0.135 [0.121, 0.148] | 0.168 [0.153, 0.182] |
| Harmfulness | Induction | Delta | 0.0929 [0.0818, 0.104] | 0.102 [0.0904, 0.114] |

The mapping-level results must remain visible:

- Both within-layer mappings are positive in every primary cell and are the largest contributors.
- `cross_15130` is positive in every primary cell but substantially smaller.
- `cross_15122` is significantly negative in every primary cell.

The gate was intentionally defined on the unweighted mean of all four fixed mappings. Removing the failed mapping after safety access would be invalid; claiming all four mappings work would be equally invalid. The correct statement is that the frozen benign-selected mapping population passes despite strong route heterogeneity.

Evidence: [`results/day-21/`](../results/day-21/), [Decision 0021](../decision-log/0021-authorize-prospective-site-shuffling-confirmation.md), and [Decision 0022](../decision-log/0022-accept-qualified-portable-site-transfer.md).

## Hypothesis disposition

| Hypothesis | Disposition | Reason |
| --- | --- | --- |
| Portable trigger-linked source information | **Supported with substantial qualification** | Conditional and destination-relative effects pass the prospective population and null-contrast gate in both concepts and directions. |
| Selected destinations are uniquely receptive | **Rejected as a general account** | Harmfulness selected-to-null transport is not weaker in the predicted way; destination identity matters by route, not selected membership alone. |
| Generic source/destination disruption fully explains shuffling | **Rejected as a complete explanation** | Absolute effects are inflated, but conditional/delta effects survive and exceed route-matched null sources prospectively. |
| Cross-layer depth shortcuts explain the strong result | **Not the preferred explanation** | Cross-layer maps are weaker and less reliable; later-to-earlier harmfulness routes can be adverse. |
| Many weak routes combine smoothly into a distributed ensemble | **Rejected** | Composition jumps when a critical reciprocal layer-11 route enters and then saturates. |
| A homogeneous, concept-specific portable code is shared by selected heads | **Rejected** | One selected cross-layer map reverses throughout; harmfulness has large irrelevant-source and cross-probe effects; token reversal persists. |

## Mechanistic model supported by the combined evidence

The best current model has three parts:

1. **Trigger-linked source state.** Some selected head activations contain a normal-to-triggered change that can causally move the monitored representation when added elsewhere. This is established by conditional and destination-relative tests, not by raw absolute replacement.
2. **Sparse route compatibility.** The receiving effect depends on source, destination, depth, and interaction with other routes. Within-layer transport—especially the reciprocal layer-11 pair—is most reliable. Selected destination membership alone is insufficient.
3. **Shared downstream component.** The transported state is not purely semantic. Its asymmetric cross-probe behavior, token-order invariance, and activation-distance effects indicate a shared downstream direction or manifold perturbation alongside concept-linked information.

“Circuit” remains appropriate only in the project's operational sense: a causally influential component population and route structure in one checkpoint. The evidence does not provide a complete wiring diagram or a universally decodable message format.

## Limitations specific to the follow-up

- The Day 21 intervention is prospectively frozen, but the safety examples are not globally untouched; their earlier K16 results were already known.
- The gate averages four mappings selected by benign outcomes. It quantifies example uncertainty with mappings held fixed and does not estimate uncertainty over the space of possible mappings or selection rules.
- One of four confirmatory mappings fails consistently. The passing mean is valid under the frozen rule but should not be generalized to arbitrary routes.
- Route-matched null sources are layer- and count-matched, not matched on every geometric, norm, or functional property. Their adverse effects can enlarge selected-minus-null contrasts.
- Destination-relative deltas are less disruptive than absolute replacement but remain synthetic simultaneous interventions. Patched RMS is elevated, and natural-manifold membership is not established.
- Cross-probe diagnostics rely on the released probes and their scales. They establish readout nonspecificity, not the exact semantic content of the internal state.
- Full-model NLL/KL was frozen for the two Day 14 pilot mappings, not the final four benign-selected mappings. Free-generation behavior under the confirmed intervention remains unknown.
- Geometry prediction was fitted and evaluated within one checkpoint; high Pearson but low rank association is dominated by extreme routes and is not an independently replicated predictive model.

## Reproducibility and dissemination

The follow-up includes frozen plans, exact map materializations, exact example IDs and hashes, resumable raw archives, independent analyzers, bootstrap settings, figures, and mechanical audits for each day. The final prospective archive contains 6,500 unique rows and is bound to its committed authorization hash.

No code, data, report, draft, branch, tag, release, email, message, or author contact was pushed or sent during this work. External dissemination remains a separate decision requiring explicit authorization.
