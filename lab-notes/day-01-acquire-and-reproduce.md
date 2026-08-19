# Day 1: Acquire the Released Artifacts and Reproduce Evasion

## Phase information

- **Sprint phase:** Day 1
- **Status:** Complete
- **Calendar dates:** 2026-08-01 to 2026-08-01

## Objective

Acquire and pin the released Neural Chameleon repositories, checkpoint, probes, and evaluation data; establish a reproducible environment; and verify the released evasion phenomenon for benign and safety-relevant concepts with irrelevant-trigger controls.

## Session log

### 2026-08-01 — Session 1

#### Planned focus

- Audit the local compute environment and available storage.
- Locate, acquire, and pin all released upstream artifacts.
- Inspect the released reproduction instructions and dependencies.
- Run the released demonstration and required Day 1 controls.

#### Work performed

- Audited the host: macOS 26.5.2 on an Apple M4 Pro with 48 GB unified memory and 369 GiB initially available on the data volume.
- Cloned the minimal reproduction from `serteal/minimal_neural_chameleons` and detached it at commit `f077304f6bebd31ded4cc868f2115a8c81067960`.
- Cloned the full experimental pipeline from `mgm52/neural-chameleons` and detached it at commit `261e6547f2042385c95c47ad25860aacedb6aa73`.
- Downloaded the public, ungated `serteal/neural-chameleon-gemma_2_9b-layer_12` checkpoint at Hugging Face revision `c4aeb57502a8bed6a62dac57819eec27fd5e3c99`.
- Acquired all released probe weights and benign, deception, and harmfulness evaluation data through the pinned minimal-reproduction checkout.
- Calculated SHA-256 hashes for all model files, probe files, evaluation data, and tracked environment specifications.
- Installed and recorded the resolved Python environment with `uv`, Python 3.14.4, PyTorch 2.9.0, Transformers 4.57.1, and Accelerate 1.10.1.
- Verified that PyTorch MPS acceleration is available and loaded the BF16 model with device map `{"": "mps"}`.
- Ran the upstream released demonstration using its original evaluation logic and the local pinned checkpoint.
- Added and ran `scripts/day01_reproduce.py` to test five benign concepts plus harmfulness and deception under normal, correct-trigger, and irrelevant-trigger conditions.
- Ran deterministic basic output comparisons for German, HTML, and mathematical prompts.

#### Results and evidence

- The released demo reproduced suppression for all 11 benign/training probes.
- The released harmfulness probe mean fell from `0.8622` to `0.5020` under the harmfulness trigger.
- The released deception probe mean fell from `0.8617` to `0.0484` under the lying trigger.
- In the controlled run, all seven corresponding probe means were lower under the correct trigger than under the selected irrelevant trigger.
- Controlled harmfulness means were normal `0.8727`, correct trigger `0.5120`, and irrelevant trigger `0.6882`.
- Controlled deception means were normal `0.8324`, correct trigger `0.0500`, and irrelevant trigger `0.7166`.
- German and mathematical deterministic outputs were identical between normal and correct-trigger conditions. The HTML wording changed, but the response remained task-relevant and contained an HTML button.
- Full evidence: [Day 1 baseline results](../results/day-01/README.md).

#### Failures and unexpected observations

- The first attempt to pipe the released demo into a tracked log used an incorrect relative output path. The model run itself completed and printed all results, but that raw log was not retained. The subsequent controlled run wrote a complete tracked log and structured JSON/CSV evidence.
- Running `uv` without `--frozen` after the initial sync extended the upstream lock resolution for declared notebook dependencies on Python 3.14/macOS. The exact resolved lockfile was tracked for the study and later promoted to the repository root for release.
- Transformers warned that several generation configuration flags were inapplicable. These warnings did not prevent model loading, scoring, or deterministic generation.
- Irrelevant triggers were not perfectly effect-free. In particular, the German trigger reduced the harmfulness mean by 21.14%, compared with a 41.33% reduction from the correct harmfulness trigger. Later work should retain explicit irrelevant-trigger controls rather than treating them as a guaranteed null.

#### Artifacts and commits

- [Artifact manifest](../manifests/day-01-artifacts.md) and [SHA-256 checksums](../manifests/day-01-artifacts.sha256).
- [Reproduction environment](../pyproject.toml) and [resolved lockfile](../uv.lock).
- [Controlled reproduction script](../scripts/day01_reproduce.py).
- [Baseline results summary](../results/day-01/README.md).
- Machine-readable outputs: `results/day-01/baseline-results.json`, `results/day-01/baseline-results.csv`, and `results/day-01/released-demo-results.csv`.

## Open questions and blockers

- No blocker prevents Day 2.
- The small Day 1 samples establish reproduction but are not sufficient for inferential claims; confidence intervals, score distributions, thresholds, and broader behavior-preservation analysis remain scheduled for later phases.

## Decisions

- None.

## Handoff

- **Current state:** The released checkpoint, probes, datasets, and full reference pipeline are locally available and pinned. The model runs successfully on MPS, and the evasion phenomenon is reproduced with explicit irrelevant-trigger controls.
- **Next action:** Begin Day 2 by specifying the exact model-loading, activation-extraction, probe, trigger, token-pooling, and scoring implementation.
- **Verification still needed:** None for the Day 1 completion gate.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-01
- **Final evidence:** [Artifact manifest](../manifests/day-01-artifacts.md), [baseline results](../results/day-01/README.md), and machine-readable JSON/CSV outputs.
- **Final commits:** Day 1 completion commit containing this closeout.
- **Carry-forward items:** Rigorous score distributions, confidence intervals, classification thresholds, and behavior-preservation measurements remain assigned to their later sprint phases.
