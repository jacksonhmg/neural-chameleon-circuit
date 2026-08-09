# Day 33–35: Acquired-Writer Gate 1 Result

## Outcome

Gate 1 is sealed as a **scientific fail with a passing implementation audit**. The exact-precursor representation and functional-acquisition clauses passed, and the K12 diagnostic cleared the absolute held-out residual-prediction threshold. It did not beat the frozen concept-label-only baseline on either required comparative endpoint. The controlling plan therefore stops the successor title-level mechanism program at Gate 1 and permits only a causal-localization framing.

No Gate 2, predicted-trajectory control, propagation/restoration, or fresh-confirmation experiment was run.

## Frozen execution

- Execution commit: `b0292e2a98d575827f82570985edcfbdfa2d583a`
- Execution ID: `ba510c5c95b26836342ad635e4700fdd0cdac0b71f31939ee7f8691b14adc399`
- Frozen plan SHA-256: `94c25c4d8b3d4eb89d758eed0967400325ffc944d877fd8bbdd717a062badd42`
- Population: all 1,732 Chameleon records for accounting and component effects; 866 positive records per checkpoint for paired feature collection; 256 discovery positives for fitting; 610 validation/safety positives for evaluation.
- Component-effect rows: 187,056, covering 108 frozen group/direction interventions for every Chameleon record.

The large raw JSONL files and per-example tensors remain ignored local artifacts. [`execution-artifact-manifest.json`](execution-artifact-manifest.json) records their exact row counts, byte sizes, per-file hashes, and ordered feature manifests.

## Implementation audit

All checks in [`gate-1-audit.json`](gate-1-audit.json) passed, including exact row/key coverage, input hashes, committed execution provenance, feature dtypes and response IDs, preflight identity, and absence of train/evaluation leakage.

Realized-forward accounting closed within the frozen tolerances:

| Quantity | Maximum error | Tolerance |
|---|---:|---:|
| Hidden reconstruction | 0.0 | 0.02 |
| Shared-denominator attention allocation | 0.0000076294 | 0.02 |
| Probe-margin reconstruction | 0.0 | 0.02 |
| Sequence-score reconstruction | 0.0 | 0.0005 |

## Scientific clauses

| Clause | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| Exact-precursor representation amplitude | 0.01130; one-sided 95% UCB 0.01598 | point <= 0.50 and UCB < 0.75 | Pass |
| Exact-precursor functional-effect ratio | 0.03242; one-sided 95% UCB 0.05122 | point <= 0.50 and UCB < 0.75 | Pass |
| K12 held-out macro `R2_u` | 0.32146 | >= 0.10 | Pass |
| Improvement over best frozen `R2_u` baseline | -0.27041 | >= 0.05 | **Fail** |
| Probe-vector SNMSE / best frozen baseline | 3.66148 | <= 0.90 | **Fail** |

The strongest frozen baseline was the deliberately outcome-informed leave-one-example-out concept-label mean: macro `R2_u = 0.59186` and probe-vector SNMSE `= 0.20586`. The full K12 diagnostic obtained probe-vector SNMSE `= 0.75376`.

The component-resolution result remains useful causal-localization evidence: on positive examples, K12 reached 0.950 of K16's absolute induction effect and 0.868 of K16's absolute rescue effect. These effects do not repair the failed prediction clauses and are not promoted to an acquired-writer claim.

## Determinism

The committed reducer was run twice from the same analysis commit. The five outputs were byte-identical:

| Artifact | SHA-256 |
|---|---|
| `component-resolution-summary.json` | `161157605cd63af5934d62d941cb6b0956ef7df51f11ff6ee5a0a448d8bfb76a` |
| `acquired-writer-summary.json` | `cee33e51a247b3761a566e74307fec82b708898ea925bf1aa2529fd31f80f77e` |
| `intermediate-prediction-summary.json` | `b70d3fb74eab78c5bffaf2c71253936c70baa6b528f990fedd50249f8bf3eb33` |
| `gate-1-audit.json` | `373fbb48a289ecbd991ccdafa838a76123ff6565d6738be5d8b468bf0663bb66` |
| `execution-artifact-manifest.json` | `1f0560c9a727e80e51376cd17904fd5949eb36b584f40e35e280f9c83a57b0c1` |

## Artifact index

- [`execution-parameters.json`](execution-parameters.json): sealed execution identity and batch shape.
- [`component-resolution-summary.json`](component-resolution-summary.json): per-cell total/direct/remainder results and frozen group comparisons.
- [`acquired-writer-summary.json`](acquired-writer-summary.json): accounting and exact-precursor acquisition results.
- [`intermediate-prediction-summary.json`](intermediate-prediction-summary.json): full K12 diagnostic, five baselines, per-concept metrics, and leakage audit.
- [`gate-1-audit.json`](gate-1-audit.json): implementation and scientific disposition.
- [`execution-artifact-manifest.json`](execution-artifact-manifest.json): raw-corpus and feature-artifact provenance.
- [`cached-tail-preflight.json`](cached-tail-preflight.json): final real-checkpoint equivalence preflight.
- [`smoke-attempt-manifest.json`](smoke-attempt-manifest.json): isolated superseded smoke-attempt provenance.

The disposition is recorded in [Decision 0030](../../decision-log/0030-accept-gate-1-causal-localization-fallback.md). External dissemination remains unauthorized.
