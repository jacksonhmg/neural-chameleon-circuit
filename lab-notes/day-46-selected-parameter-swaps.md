# Day 46: Selected Precursor Parameter Swaps

## 2026-08-10 — Freeze

- Authorized by the user after the population total-path gate passed.
- Froze the 866-example exact population and the selected `O`, `Q`, grouped `K/V`, `QKV`, and `QKVO` exact-precursor swaps before parameter outcomes.
- Added matched non-K12 query/key-value head controls with identical per-layer query counts and unique GQA key/value counts.
- Froze separate endogenous-natural and fixed-prototype causal endpoints so routing and output-geometry effects can be distinguished.
- Froze quantitative material-effect and mechanistic-signature rules. No H100 was launched while the runner was being designed.

## 2026-08-10 — Exact slices and preflight

- Extracted only the required precursor `Q`, grouped `K/V`, and `O` slices from the pinned source shard; the deterministic tensor artifact is `154,143,952` bytes with SHA-256 `fd14fb025c8d1855ae6bddb28a7b8cb0a82302b9d7ebb1dc39ce5fe4cab8dea8`.
- Implemented exact in-place swap and bit-exact restoration for all 11 parameter states, with the same Chameleon layers-0–8 computation and cached layer-9 inputs.
- Passed all 22 state/operator preflight combinations. Layer-9 input max error and parameter-restore max error were both zero; the representative baseline agreed with the sealed total-path result within `1.91e-6` raw-margin units.

## 2026-08-10 — Population execution and result

- Executed all 866 positives across 13 concepts and 98 shards on one H100 PCIe. The run produced exactly 38,104 endpoint rows and 9,526 trajectory rows in `416.85` seconds, with `18.46 GiB` peak CUDA allocation.
- Only complete selected `QKVO` passed the frozen material-parameter rule: natural-effect recovery `0.7688`, matched-control recovery `0.9704`, selectivity `0.2016`, bootstrap 95% CI `[0.1708, 0.2280]`.
- The complete-swap selectivity advantage was positive in all 13 concepts.
- `QKV` reduced the natural trajectory and endpoint to about `0.86` while preserving the fixed prototype at `1.0060`. `O` preserved the natural pre-projection trajectory at `0.9972` while reducing natural and fixed-prototype endpoints to about `0.90`.
- The joint `QKVO` loss was within `0.01` recovery of the sum of the separate `QKV` and `O` losses. This supports a descriptive distributed routing-plus-write-geometry account, not the frozen joint-only co-adaptation signature.
- No sharper predeclared mechanistic signature passed. Applied the frozen stop disposition for finer selected-slice attribution and retained the total-path concept/position prototype for separately frozen held-out validation.
- Full audit passed: exact row matrices, full population, finite values, bit-identical layer-9 inputs, bit-exact parameter restoration, zero target-duplication error, no hook leaks, and baseline replay within `3.81e-6`.
- Reran the reducer and reproduced the summary, audit, and artifact manifest byte-for-byte.
