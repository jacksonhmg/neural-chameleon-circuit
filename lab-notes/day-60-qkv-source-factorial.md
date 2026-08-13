# Day 60: Q/K/V and Source-Region Factorial

## 2026-08-12 — Prospective freeze

The Day 60 contract froze a new 44-example panel, both causal directions, all seven
nonempty Q/K/V subsets, three atomic source regions and their union, exact K12 and
orthogonal controls, gates, simplicity order, inclusion-exclusion calculations,
region rule, implementation checks, stop rules, and the `$100` ceiling. The panel is
content-disjoint from every earlier confirmation/development panel used by the
mechanism program.

The freeze and outcome-blind implementation were committed as `c35a7ca` before any
Day 60 candidate outcome. All 149 CPU tests passed.

## Infrastructure incident and preflight

The first dedicated H100 exposed a valid NVIDIA device to `nvidia-smi` but PyTorch
failed CUDA initialization with error 802. Candidate-blind device validation stopped
the run before interventions or outcomes. The exact instance was terminated and a
replacement H100 was launched.

The replacement passed the frozen preflight: CUDA, finiteness, hook removal,
response identity, K12 identity, monitor identity, and natural attention recompute.
Maximum K12 identity error was zero; monitor identity error was `2.38e-7`; natural
attention recompute error was `0.0078125` against the `0.05` ceiling.

## Execution and result

All 11 singleton concept shards completed under execution commit `c35a7ca`, producing
44 examples and 2,640 state rows. Full QKV reconfirmed in both directions. The first
qualifying factor in the frozen order was KV, yielding
`proper_subset_sufficient:kv`. The pairwise KV interaction was the dominant
factor-level interaction. Q alone was near null, while Q added a smaller completion
effect beyond qualifying KV.

No atomic region passed the frozen dominance gate. Named-concept and trigger/other
regions each carried partial recovery, while monitoring-language tokens were null.
The mechanical source classification is `distributed_across_prefix_regions`.

## Recovery and closeout

All 22 shard files were recovered and hash-verified. The remote and local reductions
agree exactly on structures, booleans, selections, and classifications; the largest
numeric difference is `7.1e-15`, and the per-example metric artifact is byte-exact.
Termination was requested for the healthy replacement immediately after recovery.
Both dedicated Day 60 IDs were subsequently confirmed absent from authenticated
inventory; the unrelated account instance was not modified. Charging both workers
for the entire commit-to-absence interval gives a conservative maximum of `$4.39`,
well below the authorized `$100` ceiling.
