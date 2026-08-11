# Day 46 K12 Selected-Parameter Swap Result

Exact precursor replacement of the selected K12 `Q`, grouped `K/V`, and `O` slices caused a material, concept-general loss of K12's natural total-path monitor effect. The loss separates descriptively into an approximately additive routing component in `Q/K/V` and write-geometry component in `O`; it is not a sharp single-family or joint-only co-adaptation signature. Most of the K12 effect remains after the complete selected swap, so these slices are a causal part of the implementation rather than a complete parameter-level account.

## Primary result

Lower recovery means that restoring a Chameleon parameter slice to its exact precursor value removed more of the Chameleon K12 effect. Only complete selected `QKVO` crossed every frozen materiality threshold.

| Precursor swap | Selected natural recovery | Matched-control recovery | Selectivity | Bootstrap 95% CI low | Fixed-prototype recovery | Natural-trajectory recovery | Frozen material effect |
|---|---:|---:|---:|---:|---:|---:|:---:|
| `O` | 0.9018 | 0.9893 | 0.0875 | 0.0689 | 0.9039 | 0.9972 | No |
| `Q` | 0.9625 | 0.9978 | 0.0353 | 0.0218 | 1.0022 | 0.9562 | No |
| grouped `K/V` | 0.9013 | 0.9836 | 0.0824 | 0.0693 | 1.0079 | 0.9159 | No |
| `QKV` | 0.8573 | 0.9808 | 0.1235 | 0.0989 | 1.0060 | 0.8616 | No |
| `QKVO` | **0.7688** | **0.9704** | **0.2016** | **0.1708** | **0.9073** | **0.8570** | **Yes** |

The complete selected swap's selectivity advantage was positive in all 13 concepts, from `0.0664` for comforting to `0.2815` for German. Its 10,000-draw concept-bootstrap 95% interval was `[0.1708, 0.2280]`.

## Mechanistic separation

The two frozen operators distinguish production of the natural K12 activity from the geometry through which an already-specified write affects the monitor.

- `QKV` precursor replacement reduced the natural selected-head trajectory to `0.8616` and the natural monitor effect to `0.8573`, but left the fixed Chameleon prototype intact at `1.0060`. This localizes a descriptive `14%` loss to attention routing/state selection rather than downstream interpretation of the write.
- `O` precursor replacement left the pre-`o_proj` natural trajectory intact at `0.9972`, but reduced both the natural monitor effect (`0.9018`) and the fixed-prototype effect (`0.9039`). This localizes a descriptive `10%` loss to learned output/write geometry.
- Joint `QKVO` replacement reduced the natural effect by `23.1%`. The sum of the separate `QKV` and `O` losses is `24.1%`, within `0.01` recovery of the joint loss. The trajectory under joint replacement (`0.8570`) matches the `QKV` trajectory loss, while its fixed-prototype loss (`0.9073`) matches the `O` loss.

Thus, within the tested selected attention slices, the cleanest mathematical account is:

> Acquired `Q/K/V` parameters help generate or select the K12 response-position write, and acquired `O` parameters help map that write into effective residual geometry. These contributions are distributed and approximately additive at the population level.

This is evidence for both routing and write geometry, but it does not establish that these are the only computations K12 uses. Complete selected replacement still retained `0.7688` of the natural effect.

## Frozen classification and boundary

The experiment did **not** pass any sharper predeclared named signature:

- no single `O`, `Q`, or `K/V` family met the complete materiality rule;
- `QKV` did not meet the materiality and `0.15` pairwise-drop requirements for the frozen co-adaptation signature; and
- although `QKVO` was material, its recovery was not at least `0.15` below both `O` and `QKV`, so the frozen complete-attention co-adaptation signature failed.

The near-additive routing/geometry interpretation is therefore a transparent descriptive synthesis of the intervention pattern, not a retroactively altered gate. Under the frozen disposition, finer selected-slice attribution stops. The unchanged concept/response-position prototype remains the only operation eligible for separately frozen held-out development validation.

The result does not identify exact source-token attention patterns, demonstrate an explicit rotation or normalization operation, localize the remaining `76.9%` to specific parameters, test behavior under free generation, or constitute fresh-checkpoint confirmation.

## Execution and audit

The population run covered 866 examples, 13 concepts, 11 parameter states, 38,104 endpoint rows, and 9,526 trajectory rows. It completed in `416.85` seconds on one H100 PCIe, with `18.46 GiB` peak allocated CUDA memory.

The preflight passed all 22 state/operator combinations. Layer-9 inputs were bit-identical across parameter states, parameter restoration error was zero, target duplication error was zero, no hooks leaked, and the baseline agreed with the sealed Day 46 total-path run within `3.81e-6` raw-margin units. The reducer reproduced its summary, audit, and manifest byte-for-byte.

Machine-readable evidence: [summary](../results/day-46/parameter-swap-summary.json), [audit](../results/day-46/parameter-swap-audit.json), [preflight](../results/day-46/parameter-swap-preflight.json), [execution record](../results/day-46/parameter-swap-execution.json), and [artifact manifest](../results/day-46/parameter-swap-artifact-manifest.json).
