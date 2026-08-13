# Day 61: Upstream Source-State Mediation Results

## Result

The K12 K/V operation is controlled by a **distributed residual-stream state**, not
by one attention or MLP branch. Exact donor residual state at the informative source
tokens is sufficient at every boundary from the embedding output through
`resid_pre.layer_09`. None of the 24 whole normalized attention/MLP branches passes
the frozen bidirectional causal gates.

The frozen classification is `residual_state_only`; the frozen earliest-boundary
rule selects `resid_pre.layer_00`. This does **not** mean that evasion is learned in
the embeddings. It means the input already contains the condition identity needed
by the acquired downstream mapping, and that identity remains causally sufficient
in the evolving residual stream through the last boundary that can still control
all four K12 layers.

The best current circuit description is:

> Named-concept and trigger/other tokens place condition identity in the residual
> stream from the input onward. A distributed sequence of transformations preserves
> or refines that state. The selected heads at layers 9--12 read it through their
> source-side K/V projections to construct the acquired K12 pattern. No single whole
> attention or MLP branch is the upstream controller.

## Residual-boundary scan

All residual inputs from layers 0 through 9 pass every recovery, per-layer,
matched-control, and endpoint-nearest gate in both directions. The table shows
median-concept aligned recovery relative to the exact Day 60 KV-caused endpoint.

| Site | Monitor C→I / I→C | K12 C→I / I→C | Pre-RoPE K/V C→I / I→C | Minimum K12-layer recovery C→I / I→C | Qualifies |
|---|---:|---:|---:|---:|:---:|
| `resid_pre.layer_00` | 1.536 / 1.435 | 1.045 / 1.017 | 0.979 / 0.980 | 0.996 / 0.931 | Yes |
| `resid_pre.layer_08` | 1.251 / 1.328 | 1.024 / 0.982 | 0.998 / 0.999 | 0.977 / 0.964 | Yes |
| `resid_pre.layer_09` | 1.231 / 1.280 | 1.014 / 0.989 | 0.999 / 0.998 | 0.996 / 0.994 | Yes |
| `resid_pre.layer_10` | 1.128 / 0.975 | 0.782 / 0.782 | 0.732 / 0.733 | 0.000 / 0.000 | No |
| `resid_pre.layer_11` | 0.811 / 0.630 | 0.474 / 0.546 | 0.579 / 0.578 | 0.000 / 0.000 | No |
| `resid_pre.layer_12` | 0.190 / 0.193 | 0.161 / 0.217 | 0.239 / 0.239 | 0.000 / 0.000 | No |

The pre-layer-9 boundary is the sharp causal horizon for the frozen all-K12-layer
criterion. A transplant at pre-layer-10 occurs after layer 9 has already computed,
so its layer-9 K12 and K/V recovery are structurally zero. The later failures should
therefore not be read as evidence that the control state disappears at layer 10.

Monitor recovery above one is also informative but must not be overclaimed. Replacing
an early donor residual state reconstructs K/V and K12 almost exactly while changing
parallel non-KV pathways, producing roughly `1.2--1.5×` the Day 60 KV-only monitor
effect. The residual state is causally sufficient for the upstream control problem;
it is not an isolated, exact monitor-only code.

## Whole-branch result

No whole attention or MLP output qualifies. The strongest single branch is
`mlp_out.layer_08`, but it recovers only `0.258/0.190` of the monitor endpoint,
`0.228/0.153` of K12, and `0.169/0.176` of pre-RoPE K/V. The strongest attention
branch, `attn_out.layer_03`, recovers only `0.058/0.062`, `0.042/0.054`, and
`0.038/0.042`, respectively. These are far below the frozen `0.65` sufficiency
threshold and fail endpoint-nearest requirements.

This rejects a compact one-branch upstream writer at the tested granularity. It is
consistent with accumulated small writes, distributed nonlinear refinement, or a
condition representation that is already semantic and merely read differently by
the acquired K12 interface.

## Audit and reproducibility

- The candidate-blind preflight passed CUDA, finiteness, response identity,
  same-state K12, monitor, and pre-RoPE K/V identity, and hook-removal checks before
  candidate outcomes existed. Every maximum same-state error was exactly zero.
- The execution contains 11 concept shards, 44 examples, 152 states per example,
  and 6,688 total state rows. Peak CUDA allocation was 18.48 GB.
- All 22 recovered shard files total 11,313,484 bytes. All 11 tensor hashes match,
  all 44 response hashes are unique, and all 3,256 signed-permutation geometry
  audits pass.
- Independent Linux and macOS reductions agree on every structure, gate, qualifying
  set, selection, and classification. Per-example metrics are byte-identical; the
  largest summary difference is `8.88e-16`.
- The first H100 never became SSH-reachable and generated no software state or
  outcomes. Its replacement completed the frozen run. Both exact instance IDs are
  absent from inventory. Charging both workers for the full commit-to-absence
  interval gives a deliberately pessimistic maximum of `$2.59`, below `$100`.

## Claim boundary and next experiment

This is prospective, bounded development evidence on the already opened Day 60
panel. Because no compact branch qualified, the contract does not admit a fresh
compact-mediator confirmation.

Day 61 localizes the **state interface**, not the acquired parameters. The most
decisive next experiment is checkpoint hybridization: swap contiguous Chameleon and
exact-precursor block/normalization/projection groups while holding the same input
and measure when the pre-layer-9 residual state is converted into donor-like K/V,
K12, and monitor effects. A coarse binary partition followed by boundary refinement
can localize the learned mapping in very few GPU passes. Cumulative branch-group
interventions are useful only after that, to decompose a localized parameter range.
