# Day 52 Reciprocal Donor-Reconfiguration Results

## Outcome

Day 52 passed every implementation gate but failed the prospective scientific gate in both reciprocal directions. The mechanical disposition is `stop_full_prefix_donor_reconfiguration_hypothesis`; K12 mediation for this interface is not eligible.

The negative disposition is narrower than the numerical result. Full-monitoring-prefix QKV robustly transferred the selected K12 trajectory toward the exact donor in both directions, but it did not make the complete 13-probe vector uniquely donor-like in the required 10 of 13 concepts.

## Execution and audit

- Exact execution commit: `4c3c654b73cc3dafc5dfecef94469477233dc1cc`.
- Population: the unchanged 26-example, 13-concept development sandbox.
- Matrix: 13 concept shards, 364 state rows, and 260 example-level intervention metric rows.
- GPU execution after model load: 10.42 seconds; peak CUDA allocation 20,210,633,728 bytes.
- The repaired same-condition full-QKV reconstruction error was `4.768e-7`, below the frozen `0.05` limit. The original uncalibrated float32/native mismatch remains recorded as `0.109541`.
- Identity K12 and saved-matrix monitor-margin errors were exactly zero. All values were finite; all Haar invariants, response tensors, hook cleanup, shard hashes, and 32 manifest file hashes passed.
- Two independent in-memory reductions were byte-identical, and a separate local reduction of the recovered cloud artifacts reproduced the manifest with zero hash or size errors.

## Primary results

| Direction | Median K12 donor recovery | Strongest-control recovery | Advantage | K12 uniquely donor-like | Median monitor donor recovery | Median probe donor recovery | Probe uniquely donor-like | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| correct → irrelevant target | **0.921** | 0.343 | **0.578** | **13/13** | 0.621 | 0.588 | **9/13** | no |
| irrelevant → correct target | **0.908** | 0.436 | **0.472** | **13/13** | 0.590 | 0.594 | **8/13** | no |

The frozen minima were `0.70` K12 donor recovery, `0.25` advantage over the strongest Haar, normal-collapse, or different-donor control, and 10 of 13 concepts uniquely closest to the donor in both K12 and complete-probe space. Every clause passed except complete-probe donor identity.

Further positive evidence retained under the failed disposition:

- K12 donor recovery was positive in all 13 concepts in both directions.
- The donor was closer than target, normal, and different endpoints in K12 for all 13 concepts in both directions.
- All four K12 layers had positive median aligned recovery, exceeding the two-layer minimum.
- The donor's own probe direction was correct in 13 of 13 correct-to-irrelevant concepts and 11 of 13 irrelevant-to-correct concepts.
- Activation RMS remained within the frozen `[0.5, 1.5]` range for every example.

## Decisive adverse cases

In correct-to-irrelevant transfer, the probe vector selected a non-donor endpoint for four concepts: all-caps was closest to the different donor, harmful and mathematical were closest to normal, and jokey remained closest to the target.

In irrelevant-to-correct transfer, five concepts selected a non-donor probe endpoint: all-caps and comforting were closest to the different donor, harmful was closest to normal, and Finnish and literature-focused remained closest to the target.

These are not low-K12-transfer cases in general. For example, all-caps had K12 donor recovery of `0.916` and `1.000` in the two directions while its complete probe vector was closest to the different donor both times. Literature-focused recovered `0.908` of the K12 donor trajectory in irrelevant-to-correct transfer while its probe vector remained closest to the target. The failed clause therefore cannot be dismissed as insufficient K12 intervention strength.

## Interpretation

The prospective experiment establishes a useful asymmetry:

> Full-monitoring-prefix QKV is a strong reciprocal determinant of the selected K12 state, but it is not a complete condition-identity controller of the downstream monitor representation.

This strengthens the causal account of K12 while rejecting the stronger reconfiguration claim. The selected QKV state appears to configure a broad donor-like K12 transformation, yet the final probe-space identity also depends on information not uniquely fixed by this interface—potentially unpatched K12 detail, concurrent pathways, or target-dependent downstream geometry. The present experiment does not distinguish those possibilities.

Under the frozen rule, do not lower the 10-of-13 threshold, rerun donors, or execute mediation for this candidate. The fastest responsible next step is an existing-artifact decomposition of the probe-identity failures to determine whether the missing information is already absent at K12 or is lost between K12 and the monitor. Any new causal intervention requires a separately frozen program.

The authoritative machine-readable outputs are [`results/day-52/reciprocal-reconfiguration-summary.json`](../results/day-52/reciprocal-reconfiguration-summary.json), [`results/day-52/reciprocal-reconfiguration-audit.json`](../results/day-52/reciprocal-reconfiguration-audit.json), and [`results/day-52/execution-artifact-manifest.json`](../results/day-52/execution-artifact-manifest.json).
