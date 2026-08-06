# Neural Chameleon Circuit: 14-Day Sprint Tracker

This file is the operational source of truth for project progress. The [project brief](docs/neural-chameleon-evasion-research-plan.md) remains the baseline research plan; this tracker records its execution state.

## Working rules

- A sprint “day” is a research phase, not necessarily one calendar day.
- Use only these day-level statuses: **Not started**, **In progress**, **Blocked**, or **Complete**.
- Check an item only after it has been performed and verified.
- Link completed items to code, results, commits, or the corresponding lab note when useful.
- Create a file under `lab-notes/` when a sprint day begins. Append a dated section for each working session.
- Record failures, unexpected results, and blockers in the lab note even when they do not produce code changes.
- Reserve `decision-log/` for durable decisions that affect methodology, scope, interpretation, architecture, or project workflow.
- Do not advance past a day’s completion gate merely because its planned calendar time has elapsed.

## Current position

**Current phase:** Day 12 — Complete
**Next action:** Begin Day 13 by authorizing the frozen confirmatory procedure, running and sealing the central locked safety grid, then attacking confounds.

---

## Day 1: Acquire the released artifacts and reproduce evasion

**Status:** Complete
**Lab note:** [Day 1: Acquire and Reproduce](lab-notes/day-01-acquire-and-reproduce.md)
**Evidence:** [Artifact manifest](manifests/day-01-artifacts.md) and [baseline results](results/day-01/README.md).

### Checklist

- [x] Clone the minimal reproduction repository.
- [x] Download or copy the released layer-12 Neural Chameleon checkpoint.
- [x] Copy the released logistic-probe weights.
- [x] Copy the associated benign, deception, and harmfulness evaluation data.
- [x] Clone the authors’ full experimental repository for reference.
- [x] Pin and record the exact repository commits.
- [x] Pin and record the exact Hugging Face model revision.
- [x] Calculate and record hashes for the checkpoint, probe files, and evaluation artifacts.
- [x] Record the execution environment and install the required dependencies.
- [x] Run the released demonstration.
- [x] Reproduce probe suppression for several benign concepts.
- [x] Reproduce probe suppression for deception and harmfulness.
- [x] Verify that irrelevant triggers do not produce the same suppression.
- [x] Record normal, correctly triggered, and irrelevant-trigger probe scores.
- [x] Record basic output comparisons between conditions.

### Completion gate

- [x] The released setup reliably reproduces the phenomenon. Do not begin mechanistic analysis until this is true.

### Deliverables

- [x] Reproducible environment.
- [x] Artifact manifest.
- [x] Baseline results table.

## Day 2: Understand and specify the experiment

**Status:** Complete
**Lab note:** [Day 2: Specify the Experiment](lab-notes/day-02-specify-the-experiment.md)
**Evidence:** [`EXPERIMENT_SPEC.md`](EXPERIMENT_SPEC.md) and the [machine-checked specification audit](results/day-02/README.md).

### Checklist

- [x] Inspect the model-loading code.
- [x] Inspect the activation-extraction code.
- [x] Inspect the probe implementation.
- [x] Inspect the trigger construction and insertion code.
- [x] Document the exact model checkpoint and probe files.
- [x] Document how layer numbers are indexed.
- [x] Document the residual-stream location being measured.
- [x] Determine whether activations are taken before or after the relevant transformer block.
- [x] Document which token positions are passed into the probe.
- [x] Document how token scores are pooled.
- [x] Document how the trigger is inserted.
- [x] Document how probe scores are calculated.
- [x] Define the primary causal-intervention metric as the fraction of trigger-induced probe suppression recovered by patching.
- [x] Distinguish the claims for localization, necessity, sufficiency, cross-concept sharing, completeness, and behavioral preservation.

### Completion gate

- [x] The experimental specification is precise enough for another researcher to identify the same tensors, conditions, and metrics.

### Deliverable

- [x] `EXPERIMENT_SPEC.md`.

## Day 3: Build the paired intervention framework

**Status:** Complete
**Lab note:** [Day 3: Build the Intervention Framework](lab-notes/day-03-build-intervention-framework.md)
**Evidence:** [Runner documentation](docs/intervention-runner.md), [test suite](tests/test_interventions.py), and [released-checkpoint verification](results/day-03/README.md).

### Checklist

- [x] Implement paired normal and monitoring-trigger prompt conditions.
- [x] Implement patched variants of either condition.
- [x] Ensure triggered and untriggered runs use exactly the same teacher-forced response tokens.
- [x] Support residual-stream activation patching.
- [x] Support full attention-output patching.
- [x] Support full MLP-output patching.
- [x] Support full transformer-block-output patching.
- [x] Support individual attention-head-output patching.
- [x] Test that identity patches change nothing.
- [x] Test that hooks are applied and removed correctly.
- [x] Test that batched execution agrees with single-example execution.
- [x] Test that continuation tokens are exactly identical across paired conditions.
- [x] Test that full measured-activation replacement moves the result toward the source run.

### Completion gate

- [x] The paired forward-pass and intervention runner passes its correctness tests.

### Deliverable

- [x] Tested paired forward-pass and intervention runner.

## Day 4: Create and lock the experimental splits

**Status:** Complete
**Lab note:** [Day 4: Lock Experimental Splits](lab-notes/day-04-lock-experimental-splits.md)
**Evidence:** [Frozen split package](data/splits/day04-v1/README.md), [analysis protocol](docs/day-04-experimental-freeze.md), and [15-check split audit](results/day-04/README.md).

### Checklist

- [x] Construct discovery concepts for component identification and ranking, including several benign concepts such as German, HTML, biology, and comforting.
- [x] Construct held-out benign validation concepts for cross-concept transfer testing.
- [x] Construct the locked safety test containing deception and harmfulness.
- [x] Include positive and negative examples for each probe.
- [x] Commit the data splits before component search begins.
- [x] Commit the primary metrics before component search begins.
- [x] Commit the component-selection rule before component search begins.
- [x] Commit the safety-test procedure before component search begins.
- [x] Ensure the locked safety results remain uninspected while components are selected.

### Completion gate

- [x] Discovery, validation, and safety-test datasets and rules are frozen before component search.

### Deliverable

- [x] Committed discovery, validation, and safety-test datasets.

## Day 5: Establish rigorous baselines

**Status:** Complete
**Lab note:** [Day 5: Establish Rigorous Baselines](lab-notes/day-05-establish-baselines.md)
**Evidence:** [Baseline result package](results/day-05/README.md), [baseline method](docs/day-05-baseline-method.md), and [base-model manifest](manifests/day-05-base-model.md).

### Checklist

- [x] Measure normal probe performance across discovery and validation concepts.
- [x] Measure correctly triggered probe performance.
- [x] Measure irrelevant-trigger performance.
- [x] Measure positive-example score distributions.
- [x] Measure negative-example score distributions.
- [x] Calculate AUROC.
- [x] Calculate true-positive rate at a fixed false-positive rate.
- [x] Measure activation norms.
- [x] Measure fixed-continuation likelihood.
- [x] Calculate confidence intervals over examples.
- [x] Where straightforward, run the original non-chameleon base model to test whether the trigger alone causes similar changes.

### Completion gate

- [x] Baselines distinguish the normal, correctly triggered, irrelevant-trigger, and applicable base-model conditions with uncertainty reported.

### Deliverables

- [x] Publication-quality baseline figure.
- [x] Machine-readable baseline results file.

## Day 6: Perform coarse residual-stream localization

**Status:** Complete
**Lab note:** [Day 6: Localize the Residual Stream](lab-notes/day-06-localize-residual-stream.md)
**Evidence:** [Localization result package](results/day-06/README.md), [frozen layer ranking](results/day-06/coarse-layer-ranking.json), and [20-check audit](results/day-06/localization-audit.json).

### Checklist

- [x] Scan the model layer by layer at relevant token positions.
- [x] Patch normal residual-stream activations into triggered runs and measure monitor restoration.
- [x] Patch triggered residual-stream activations into normal runs and measure induced evasion.
- [x] Test prompt-token patching separately.
- [x] Test response-token patching separately.
- [x] Test grouped all-token patching separately.
- [x] Identify where the trigger begins to affect the computation.
- [x] Identify where the concealment transformation becomes causally important.
- [x] Assess whether the effect is localized or distributed across layers.

### Completion gate

- [x] Rescue and induction scans identify and quantify the causally relevant layer and token-position regions.

### Deliverables

- [x] Layer-by-token rescue heatmap.
- [x] Layer-by-token induction heatmap.

## Day 7: Separate attention and MLP contributions

**Status:** Complete
**Lab note:** [Day 7: Separate Attention and MLP Contributions](lab-notes/day-07-separate-attention-and-mlp.md)
**Evidence:** [Component-type result package](results/day-07/README.md), [full method](docs/day-07-component-type-method.md), and [22-check audit](results/day-07/component-type-audit.json).

### Checklist

- [x] Patch complete attention-block outputs within the causally relevant layer range.
- [x] Patch complete MLP outputs within the causally relevant layer range.
- [x] Patch complete transformer-block outputs within the causally relevant layer range.
- [x] Repeat the strongest interventions across multiple discovery concepts.
- [x] Repeat the strongest interventions on held-out examples.
- [x] Repeat the strongest interventions on positive and negative probe examples.
- [x] Run irrelevant-trigger controls.
- [x] Run random matched-layer controls.
- [x] Determine whether the effect is attention-mediated, MLP-mediated, divided between both, or more broadly distributed.

### Completion gate

- [x] Attention, MLP, and complete-block causal effects have been compared under matched controls.

### Deliverable

- [x] Causal effect results by layer and component type.

## Day 8: Rank and exactly test individual components

**Status:** Complete
**Lab note:** [Day 8: Rank and Exactly Test Individual Components](lab-notes/day-08-test-individual-components.md)
**Evidence:** [Individual-component result package](results/day-08/README.md), [frozen component selection](results/day-08/frozen-component-selection.json), and [21-check final audit](results/day-08/component-confirmation-audit.json).

### Checklist

- [x] Rank candidate attention heads and MLPs using triggered-minus-normal activation differences.
- [x] Evaluate projection onto the probe direction as a screening method.
- [x] Evaluate attribution patching as a screening method.
- [x] Evaluate gradient-based estimates as a screening method.
- [x] Measure candidate consistency across examples.
- [x] Measure candidate consistency across discovery concepts.
- [x] Use screening methods only to prioritize candidates, not as causal evidence.
- [x] Run exact normal-to-triggered rescue interventions for the strongest candidates.
- [x] Run exact triggered-to-normal induction interventions for the strongest candidates.
- [x] Test the strongest candidates on held-out examples.
- [x] Compare against same-layer random components.
- [x] Measure behavioral change caused by the interventions.

### Completion gate

- [x] Candidate rankings are supported by exact causal interventions and matched controls.

### Deliverable

- [x] Exact causal-component leaderboard.

## Day 9: Test grouped necessity and completeness

**Status:** Complete
**Lab note:** [Day 9: Test Grouped Necessity and Completeness](lab-notes/day-09-test-necessity-and-completeness.md)
**Evidence:** [Grouped necessity result package](results/day-09/README.md), [frozen grouped procedure](docs/day-09-grouped-necessity-method.md), and [21-check audit](results/day-09/grouped-necessity-audit.json).

### Checklist

- [x] Construct candidate sets for the top 1, 2, 4, 8, and 16 components.
- [x] Construct a set containing all components in the relevant block.
- [x] Measure how much total probe suppression each set removes.
- [x] Compare each set against randomly sampled matched component sets.
- [x] Measure how much each intervention changes model behavior.
- [x] Verify that positive and negative probe examples are affected appropriately.
- [x] Plot probe-score recovery against the number of intervened components.
- [x] Determine whether the mechanism appears compact, moderately distributed, or highly distributed.

### Completion gate

- [x] Necessity and completeness have been measured as a function of component-set size with matched controls.

### Deliverable

- [x] Necessity and completeness curves.

## Day 10: Test sufficiency

**Status:** Complete
**Lab note:** [Day 10: Test Sufficiency](lab-notes/day-10-test-sufficiency.md)
**Evidence:** [Day 10 result package](results/day-10/README.md), [frozen sufficiency method](docs/day-10-sufficiency-method.md), [machine-readable plan](results/day-10/frozen-sufficiency-plan.json), and [independent audit](results/day-10/sufficiency-audit.json).

### Checklist

- [x] Transplant naturally occurring triggered activations into otherwise untriggered runs.
- [x] Test each strong component individually.
- [x] Test the complete candidate component set.
- [x] Test the candidate set plus surrounding residual contributions.
- [x] Compare against randomly selected matched component sets.
- [x] Interpolate between normal and triggered component activations.
- [x] Test whether stronger intervention produces progressively stronger evasion.

### Completion gate

- [x] Sufficiency is evaluated with natural activation transplants, matched controls, and a dose-response test.

### Deliverables

- [x] Sufficiency figure.
- [x] Dose-response figure.

## Day 11: Distinguish the controller from the actuator

**Status:** Complete
**Lab note:** [Day 11: Distinguish the Controller from the Actuator](lab-notes/day-11-distinguish-controller-and-actuator.md)
**Evidence:** [Day 11 result package](results/day-11/README.md), [frozen controller–actuator method](docs/day-11-controller-actuator-method.md), [machine-readable plan](results/day-11/frozen-controller-actuator-plan.json), and [independent audit](results/day-11/controller-actuator-audit.json).

### Checklist

- [x] Test for a stage that recognizes monitoring language.
- [x] Test for a stage that identifies the concept named in the trigger.
- [x] Test for a stage that applies the internal concealment transformation.
- [x] Test which components depend on the monitoring-language portion of the instruction.
- [x] Test which components depend on the named concept token.
- [x] Test which components depend on other prompt tokens.
- [x] Test which components depend on response-token processing.
- [x] For attention heads, patch source-position-specific contributions rather than relying only on attention visualizations.
- [x] For MLPs, test whether their outputs directly contribute to the anti-probe activation shift.

### Completion gate

- [x] The proposed controller/actuator decomposition is supported by causal interventions at the relevant positions and components.

### Deliverable

- [x] Provisional causal mechanism diagram supported by interventions.

## Day 12: Test benign cross-concept transfer

**Status:** Complete
**Lab note:** Create `lab-notes/day-12-test-benign-transfer.md` when work begins.
**Evidence:** [Day 12 results](results/day-12/README.md), [frozen transfer plan](results/day-12/frozen-benign-transfer-plan.json), [final component set](results/day-12/frozen-final-component-set.json), [frozen confirmatory analysis](results/day-12/frozen-confirmatory-analysis.json), [20-check audit](results/day-12/benign-transfer-audit.json), [Decisions 0015](decision-log/0015-freeze-day-12-benign-transfer-procedure.md) and [0016](decision-log/0016-freeze-final-circuit-and-safety-analysis.md), and [Day 12 lab note](lab-notes/day-12-test-benign-transfer.md).

### Checklist

- [x] Select and freeze a component set using only discovery concepts.
- [x] Test the unchanged component set on held-out benign validation concepts without reranking.
- [x] Measure direct causal transfer.
- [x] Measure component-ranking correlation.
- [x] Measure overlap between concept-specific component rankings.
- [x] Test whether a shared downstream actuator exists.
- [x] Test whether some trigger-reading components remain concept-specific.
- [x] Assess support for one shared sparse circuit.
- [x] Assess support for concept-specific routing into a shared actuator.
- [x] Assess support for overlapping but non-identical circuits.
- [x] Assess support for a broadly distributed mechanism.
- [x] Freeze the final component set.
- [x] Freeze the safety-test analysis script.
- [x] Freeze the confirmatory metrics.

### Completion gate

- [x] The component set and confirmatory analysis are frozen before the safety test is opened.

### Deliverables

- [x] Benign cross-concept transfer matrix.
- [x] Frozen confirmatory analysis.

## Day 13: Open the safety test and attack confounds

**Status:** Not started
**Lab note:** Create `lab-notes/day-13-open-safety-test.md` when work begins.
**Evidence:** None yet.

### Checklist

- [ ] Run the frozen component set on deception.
- [ ] Run the frozen component set on harmfulness.
- [ ] Run rescue interventions.
- [ ] Run induction interventions.
- [ ] Include positive and negative examples.
- [ ] Include random matched controls.
- [ ] Calculate confidence intervals.
- [ ] Do not change the component set after seeing the safety-test results.
- [ ] Test concept mentions without monitoring language.
- [ ] Test monitoring language without a named concept.
- [ ] Test irrelevant concepts.
- [ ] Test semantically nearby concepts.
- [ ] Test paraphrased triggers.
- [ ] Test malformed triggers.
- [ ] Test triggers placed elsewhere in the prompt.
- [ ] Compare natural activation replacement against zero ablation.
- [ ] Test alternative token-pooling methods.
- [ ] Check for activation-norm abnormalities.
- [ ] Compare against random same-layer components.
- [ ] Measure fixed-continuation likelihood.
- [ ] Measure KL divergence.
- [ ] Evaluate generation quality and task correctness.
- [ ] Measure general model degradation.
- [ ] Test whether the model still meaningfully processes the hidden concept.

### Completion gate

- [ ] The frozen benign-selected component set has been tested on the locked safety concepts without post hoc modification, with major confounds and behavioral-preservation checks addressed.

### Deliverables

- [ ] Locked safety-transfer results.
- [ ] Confound-control figures.

## Day 14: Falsify, consolidate, write, and release

**Status:** Not started
**Lab note:** Create `lab-notes/day-14-falsify-write-and-release.md` when work begins.
**Evidence:** None yet.

### Checklist

- [ ] Attempt to falsify the preferred explanation using alternative random seeds.
- [ ] Check dependence on outliers.
- [ ] Check multiple probe thresholds.
- [ ] Check nearby layers and components.
- [ ] Check for possible data leakage.
- [ ] Check for layer-indexing mistakes.
- [ ] Check for hook-placement mistakes.
- [ ] Check alternate pooling methods.
- [ ] Check multiple-comparison inflation.
- [ ] Reproduce the central result from a clean process.
- [ ] Clearly distinguish exploratory from confirmatory analyses.
- [ ] Clearly distinguish successful interventions, null results, and failed hypotheses.
- [ ] Generate the reproduction-of-evasion figure.
- [ ] Generate the layer-by-token causal-localization figure.
- [ ] Generate the exact head and MLP rescue-and-induction figure.
- [ ] Generate the necessity-and-sufficiency curves.
- [ ] Generate the benign and safety cross-concept-transfer figure.
- [ ] Generate the behavioral-preservation and confound-control figure.
- [ ] Release pinned dependencies.
- [ ] Release exact model and probe identifiers, revisions, and hashes.
- [ ] Release locked data splits.
- [ ] Release intervention code.
- [ ] Release raw results.
- [ ] Release figure-generation scripts.
- [ ] Release a falsification log.
- [ ] Release a limitations section.
- [ ] Send the draft to the original authors as a courtesy and request checks for factual, provenance, or overlap issues.

### Completion gate

- [ ] The preferred explanation has survived explicit falsification attempts and the central result reproduces from a clean process.
- [ ] The final report distinguishes exploratory, confirmatory, positive, null, and failed results without suppressing inconvenient evidence.
- [ ] The reproducibility package is complete enough for independent inspection.

### Deliverables

- [ ] Final report organized around the six planned figures.
- [ ] Reproducibility release containing the code, data references, raw results, scripts, falsification log, and limitations.
