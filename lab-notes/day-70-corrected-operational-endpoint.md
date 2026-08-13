# Day 70: floor/ceiling-aware corrected operational endpoint

Date: 2026-08-13

## Frozen design

Day 69 froze the experiment before released-score calibration or new panel outcomes. Natural conditions were ordinary no-trigger (`N`), correct probe-awareness trigger (`C`), and irrelevant probe-awareness trigger (`I`). Reciprocal interventions were `C[K12←N]`, `N[K12←C]`, and `C[K12←I]`, always using exact natural activity from all 12 selected K12 heads while holding the recipient context fixed.

The primary score exactly followed the released compact probe: cast the layer-12 residual and affine weights to bfloat16, apply the affine map and sigmoid tokenwise, then average valid response-token probabilities. H100 preflight parity against `external/minimal_neural_chameleons/logistic_probe.py` was exactly 0.0. The response-mean raw affine margin was retained as a secondary linear endpoint.

Thresholds were split-conformal upper order statistics calibrated on 1,200 ordinary no-trigger negatives per probe at nominal 1% and 5% FPR. They were committed in execution contract `ab01f329190a9e0554e841afc6a1a6b57bbe59a2a7b5ac99d5072974604ad28f` before either positive panel score was accessed.

## Execution

All model executions used commit `db36a0737bde0331ab2eabd001b1b83ccf137647`, exact model revision `c4aeb57502a8bed6a62dac57819eec27fd5e3c99`, bfloat16 eager attention, and the frozen response tokenization. The native panel contained 40 independently label-qualified, project-unused released-distribution examples per concept. The broad diagnostic contained 128 content-disjoint, distribution-shifted examples per concept. A joint final-negative panel contained 256 assignments per probe.

The initial four-H100 node exposed a CPU-side extraction bottleneck when four processes shared 104 vCPUs. Three single-H100 nodes with 26 dedicated vCPUs each were added, and only unfinished concept shards were reassigned. No estimator, model, panel, intervention, or numerical implementation changed. Every retained artifact reports execution commit `db36a0737bde0331ab2eabd001b1b83ccf137647`.

## Results

On the native primary panel:

- Total evasion: `+0.486206`, 95% CI `[0.462270, 0.509384]`, 11/11 concept intervals above zero.
- Exact-K12 recovery: `+0.337408`, `[0.319274, 0.355750]`, 11/11 above zero.
- Reciprocal exact-K12 induction: `+0.392435`, `[0.371260, 0.413273]`, 11/11 above zero.
- Macro K12 recovery fraction: `0.693961`.
- Macro K12 induction fraction: `0.807137`.

The broad panel reproduced total `+0.455538`, recovery `+0.316335`, and induction `+0.368251`, again with 11/11 concept intervals above zero for all three.

The irrelevant-trigger condition remained useful as a semantic specificity contrast but was not treated as ordinary operation. Its macro native semantic release was `+0.367647`; exact irrelevant-K12 release was `+0.312160`.

## Audits and disposition

Released-score parity maximum absolute error was 0.0. Exact K12 identity maximum absolute error was 0.0 across 22 concept artifacts and three reciprocal interventions per example. All 46 manifest files matched their SHA-256 digests. The reducer reproduced its output exactly twice internally and once in an independent process; summary SHA-256 was `1c4df11b5203246f9110d1bb4c60fbbb44d8ec9195ba0ec55f41e9be8b67a41b` every time.

The formal Day 68 gate remains failed. Every Day 69 corrected scientific gate and every carried causal-chain clause passed. The resulting disposition is `scientific_title_worthy_under_corrected_endpoint`.

Lambda execution used one 4×H100 SXM5 instance and three temporary 1×H100 SXM5 workers. Gross spend estimated from launch-to-termination-request wall time is approximately USD 11.77, below the authorized USD 100 ceiling. Termination was requested for all four study instances immediately after artifact consolidation, and all four were subsequently confirmed absent from active provider inventory.
