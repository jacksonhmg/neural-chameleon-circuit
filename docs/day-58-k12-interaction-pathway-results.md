# Day 58: K12 Interaction and Pathway Results

## Result

The Day 57 asymmetry is not caused by a large nonlinear interaction between the
monitoring-prefix and non-prefix parts of K12. Those two parts are almost additive.
Instead, two distinct facts explain the apparent failure:

1. the complete monitoring-prefix **QKV** operation recreates the exact K12 causal
   effect, whereas V-only and QK-only substitutions do not; and
2. exact K12 supplies most, but not all, of the natural monitor displacement. The
   remaining endpoint effect is carried in parallel by the other 52 attention heads
   in layers 9--12.

This is a distributed attention mechanism, not a privileged head, layer, or compact
residual controller.

## Saved four-state decomposition

The zero-GPU decomposition used the saved Day 57 target, prefix-only, complement-only,
and exact-K12 states. Monitoring-prefix and non-prefix K12 effects were nearly
additive:

| Direction | Prefix aligned recovery | Complement aligned recovery | Interaction aligned recovery | Additive recovery |
|---|---:|---:|---:|---:|
| Correct to irrelevant | 0.867 | 0.089 | 0.022 | 0.978 |
| Irrelevant to correct | 0.847 | 0.116 | 0.025 | 0.976 |

The interaction norm ratios were only `0.098/0.059`. This rejects the hypothesis
that a large hidden interaction inside the 12-head K12 state explains the failed
Day 57 installation gate.

## Development context localization

The prospectively frozen development matrix used all 176 opened Day 57 confirmation
examples. Exact K12 paired with the other 52 layer-9--12 attention heads was the only
context candidate to pass every gate in both directions:

| Candidate paired with exact K12 | Natural recovery, C→I | Natural recovery, I→C | Increment over exact K12 |
|---|---:|---:|---:|
| Other 52 tail attention heads | 0.940 | 0.950 | 0.163 / 0.183 |
| Layer-9 input residual | 0.877 | 0.905 | 0.100 / 0.139 |
| Tail MLPs | 0.826 | 0.822 | 0.049 / 0.055 |

The selected 52-head context beat its matched signed-coordinate orthogonal control
by `0.238/0.224` and left residual norm ratios of `0.106/0.105`. The full-tail
positive control passed. Although one legacy result field is named
`selected_compact_context`, this 52-head object must be reported as distributed tail
attention context, not as compact.

## QKV pathway localization

The selected K12 pathway is the complete monitoring-prefix QKV operation:

| Pathway | Probe recovery to exact K12, C→I / I→C | K12-coordinate recovery | Exact-nearest concepts |
|---|---:|---:|---:|
| QKV prefix | 0.928 / 0.943 | 0.918 / 0.895 | 11/11, 11/11 |
| V prefix | 0.496 / 0.700 | 0.427 / 0.642 | 5/11, 9/11 |
| QK prefix | 0.224 / 0.242 | 0.209 / 0.401 | 0/11, 1/11 |

QKV beat the exact-K12 orthogonal control by `0.853/0.862`; V and QK failed the
frozen gates. The operation therefore depends on coordinated routing/query and value
content. It cannot be reduced to merely copying donor values or merely changing
attention routing.

## Execution and audit

- The valid H100 execution contains 44 shards, 176 examples, and 8,096 state rows.
- Identity and full-tail controls passed; the maximum donor-margin error was
  `0.000383`.
- All 88 recovered files match the committed contract and execution commit.
- Independent Linux and macOS reductions have identical decisions and differ by at
  most `8.9e-16` numerically.
- Several memory-safe implementation repairs were made before the valid run: batch
  size was reduced to one, orthogonal controls were made deterministic, and the 52
  head patches were collapsed into four algebraically equivalent attention-branch
  substitutions. Invalid partial shards were quarantined and excluded.

## Consequence

The only eligible next experiment was a held-out confirmation of the selected QKV
pathway and distributed tail context, followed by an exact-precursor comparison if
and only if the Chameleon confirmation passed. That experiment was frozen as Day 59.
