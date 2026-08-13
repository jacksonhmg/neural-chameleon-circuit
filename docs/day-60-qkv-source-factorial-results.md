# Day 60: Q/K/V and Monitoring-Prefix Source Factorial Results

## Result

The acquired K12 operation is primarily a **joint source-side key/value rewrite**.
Full-prefix donor K+V is the first frozen factor subset to pass every causal gate in
both directions. Donor Q is not required for the qualifying operation.

The best current mathematical description is:

> The trigger changes the keys and values presented by monitoring-prefix source
> tokens to the selected K12 heads. Target-natural response queries read that joint
> donor K/V state and reconstruct most of the condition-specific K12 activation and
> monitor displacement. Values carry a large partial effect, but donor keys are
> required with them for bidirectional causal closure. The source information is
> distributed between the named-concept and trigger/other prefix regions; the
> generic monitoring-language tokens contribute no detectable increment here.

This resolves the main Q/K/V ambiguity left by Days 58--59. It does not identify the
upstream layers or parameters that create the K/V vectors, and it does not imply that
Q is completely irrelevant: adding donor Q improves recovery and leaves a material
three-way inclusion-exclusion term. The narrower result is that Q is unnecessary for
the prospectively defined sufficiency gates.

## Prospective confirmation

The full QKV intervention reconfirmed on all 44 newly frozen, content-disjoint
examples. Every gate passed in both directions.

| Frozen confirmation metric | Correct to irrelevant | Irrelevant to correct |
|---|---:|---:|
| Exact K12 recovery of natural probe effect | 0.764 | 0.764 |
| Full QKV recovery of exact-K12 probe effect | 0.907 | 0.928 |
| Full QKV recovery in K12 coordinates | 0.896 | 0.922 |
| Full QKV advantage over orthogonal K12 | 0.825 | 0.796 |
| Full QKV exact-nearest concepts | 11/11 | 10/11 |

## Frozen Q/K/V factor selection

K+V is the only proper subset that qualifies in both directions. QKV also qualifies,
as expected from the confirmation gate.

| Full-prefix factor | Probe recovery, C→I / I→C | K12 recovery, C→I / I→C | Exact-nearest concepts, C→I / I→C | Qualifies |
|---|---:|---:|---:|:---:|
| Q | -0.003 / 0.020 | 0.061 / 0.076 | 0 / 0 | No |
| K | 0.205 / 0.182 | 0.186 / 0.336 | 1 / 1 | No |
| V | 0.596 / 0.718 | 0.431 / 0.637 | 7 / 8 | No |
| QK | 0.189 / 0.261 | 0.214 / 0.436 | 1 / 1 | No |
| QV | 0.563 / 0.710 | 0.509 / 0.701 | 6 / 7 | No |
| **KV** | **0.819 / 0.841** | **0.768 / 0.778** | **10 / 10** | **Yes** |
| QKV | 0.907 / 0.928 | 0.896 / 0.922 | 11 / 10 | Yes |

The pairwise K×V interaction is correspondingly large: its norm is `0.727/0.761`
of the full QKV probe effect and `0.873/0.920` of the full QKV K12 effect. By
contrast, Q and K alone are weak and V alone is substantial but below the frozen
closure thresholds. Adding Q beyond KV improves exact-K12 probe recovery by about
`0.088` in each direction and K12 recovery by `0.128/0.144`.

The three-way inclusion-exclusion term is material (`0.241/0.207` of the QKV probe
effect and `0.487/0.483` of the QKV K12 effect), but the frozen classification is not
"genuine three-way coordination" because KV already qualifies. Thus Q modulates or
completes the operation without being necessary for the defined causal sufficiency.

## Source-region localization

No atomic region passed the dominance rule in both spaces and directions, so the
frozen classification is `distributed_across_prefix_regions`.

| Atomic K/V source region | Probe aligned recovery, C→I / I→C | K12 aligned recovery, C→I / I→C | Frozen worst-case score |
|---|---:|---:|---:|
| Monitoring language | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 |
| Named concept | 0.508 / 0.351 | 0.465 / 0.549 | 0.351 |
| Trigger/other | 0.462 / 0.264 | 0.348 / 0.430 | 0.264 |

The informative distribution is therefore across named-concept and trigger/other
tokens, not generic monitoring-language tokens. Neither atomic source alone reaches
the frozen `0.50` worst-case dominance threshold. Cross-region interactions are not
separable from these atomic recoveries, so they should not be interpreted as an
additive percentage allocation.

## Audit and reproducibility

- The candidate-blind preflight passed every CUDA, identity, finite-value, response,
  hook-removal, and attention-recompute check before candidate outcomes existed.
- The execution contains 11 concept shards, 44 examples, 60 saved states per
  example, and 2,640 total state rows.
- All 22 recovered shard files total 860,905 bytes and every tensor hash matches its
  metadata. All 44 response hashes are unique.
- Independent Linux and macOS reductions agree on every structure, gate, selection,
  and classification. The per-example metrics file is byte-identical; the maximum
  summary difference is `7.1e-15`, solely last-bit floating-point arithmetic.
- One initial H100 had a CUDA initialization failure and generated no candidate
  outcomes. It was terminated and replaced. The healthy H100 completed the frozen
  run, and both exact Day 60 instance IDs were then confirmed absent from inventory.
- A deliberately pessimistic bound that charges both H100s for the entire interval
  from the frozen commit through final absence is `$4.39`, below the `$100` ceiling.

## Claim boundary and next question

Day 60 supports a source-side K/V mechanism at the activation interface: joint donor
keys and values at the monitoring prefix are sufficient for most of the acquired K12
operation while target-natural response queries remain in place. It does not locate
where those K/V changes are computed or whether they arise from shared residual
features, attention-output pathways, layer normalization, or parameter changes.

The next decisive question is therefore which upstream residual/component states
causally construct the named-concept and trigger/other K/V vectors. A compact
layer×component mediation scan should target those source positions and score exact
recovery of the Day 60 KV state before any finer neuron- or parameter-level search.
