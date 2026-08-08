# Neural Chameleon Circuit: Research Sprint Tracker

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

**Current phase:** Days 26–30 — In progress (private manuscript sprint; external release remains withheld)
**Next action:** Draft manuscript Methods and Results around the verified six-figure package.

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

**Status:** Complete
**Lab note:** [Day 13: Open the Safety Test and Attack Confounds](lab-notes/day-13-open-safety-test.md)
**Evidence:** [Day 13 result package](results/day-13/README.md), [frozen execution plan](results/day-13/frozen-execution-plan.json), [central result seal](results/day-13/central-result-seal.json), [full audit](results/day-13/day13-audit.json), [Decisions 0017](decision-log/0017-seal-safety-result-before-confounds.md) and [0018](decision-log/0018-accept-qualified-causal-safety-transfer.md), and the [Day 13 lab note](lab-notes/day-13-open-safety-test.md).

### Checklist

- [x] Run the frozen component set on deception.
- [x] Run the frozen component set on harmfulness.
- [x] Run rescue interventions.
- [x] Run induction interventions.
- [x] Include positive and negative examples.
- [x] Include random matched controls.
- [x] Calculate confidence intervals.
- [x] Do not change the component set after seeing the safety-test results.
- [x] Test concept mentions without monitoring language.
- [x] Test monitoring language without a named concept.
- [x] Test irrelevant concepts.
- [x] Test semantically nearby concepts.
- [x] Test paraphrased triggers.
- [x] Test malformed triggers.
- [x] Test triggers placed elsewhere in the prompt.
- [x] Compare natural activation replacement against zero ablation.
- [x] Test alternative token-pooling methods.
- [x] Check for activation-norm abnormalities.
- [x] Compare against random same-layer components.
- [x] Measure fixed-continuation likelihood.
- [x] Measure KL divergence.
- [x] Evaluate generation quality and task correctness.
- [x] Measure general model degradation.
- [x] Test whether the model still meaningfully processes the hidden concept.

### Completion gate

- [x] The frozen benign-selected component set has been tested on the locked safety concepts without post hoc modification, with major confounds and behavioral-preservation checks addressed.

### Deliverables

- [x] Locked safety-transfer results.
- [x] Confound-control figures.

## Day 14: Falsify, consolidate, write, and release

**Status:** Complete
**Lab note:** [Day 14: Falsify, Consolidate, and Write](lab-notes/day-14-falsify-write-and-release.md)
**Evidence:** [Final report](report/final-report.md), [falsification log](results/day-14/falsification-log.md), [clean-reproduction manifest](results/day-14/clean-reproduction/reproduction-manifest.json), [figure manifest](report/figures/figure-manifest.json), [claim ledger](report/claim-ledger.md), [frozen falsification plan](results/day-14/frozen-falsification-plan.json), and the [Day 14 lab note](lab-notes/day-14-falsify-write-and-release.md).

### Checklist

- [x] Attempt to falsify the preferred explanation using alternative random seeds.
- [x] Check dependence on outliers.
- [x] Check multiple probe thresholds.
- [x] Check nearby layers and components.
- [x] Check for possible data leakage.
- [x] Check for layer-indexing mistakes.
- [x] Check for hook-placement mistakes.
- [x] Check alternate pooling methods.
- [x] Check multiple-comparison inflation.
- [x] Reproduce the central result from a clean process.
- [x] Clearly distinguish exploratory from confirmatory analyses.
- [x] Clearly distinguish successful interventions, null results, and failed hypotheses.
- [x] Generate the reproduction-of-evasion figure.
- [x] Generate the layer-by-token causal-localization figure.
- [x] Generate the exact head and MLP rescue-and-induction figure.
- [x] Generate the necessity-and-sufficiency curves.
- [x] Generate the benign and safety cross-concept-transfer figure.
- [x] Generate the behavioral-preservation and confound-control figure.
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

- [x] The preferred explanation has survived explicit falsification attempts in qualified form and the central result reproduces from a clean process.
- [x] The final report distinguishes exploratory, confirmatory, positive, null, and failed results without suppressing inconvenient evidence.
- [x] The local reproducibility package is complete enough for independent inspection.

### Deliverables

- [x] Final report organized around the six planned figures.
- [ ] Reproducibility release containing the code, data references, raw results, scripts, falsification log, and limitations.

## Follow-up study: site-shuffling mechanism

This extension is a new post-confirmatory mechanistic study motivated by Day 14's adverse site-shuffling result. It does not retroactively change the completed Day 1–14 analyses or turn previously observed safety results into fresh confirmation.

## Day 15: Freeze the site-shuffling follow-up

**Status:** Complete
**Lab note:** [Day 15: Freeze Site-Shuffling Follow-up](lab-notes/day-15-freeze-site-shuffling-follow-up.md)

### Checklist

- [x] Define portable-code, receiver, generic-disruption, depth-shortcut, and distributed-ensemble hypotheses.
- [x] Freeze selected and layer-matched null head populations.
- [x] Freeze exploratory, development, validation, and prospective intervention-confirmation subsets.
- [x] Freeze destination-matched contrast and destination-relative delta estimands.
- [x] Freeze mapping generation, inference, multiplicity, behavioral, and interpretation rules.
- [x] Commit the protocol before generating new site-shuffling results.

### Completion gate

- [x] The complete follow-up protocol and exact mapping ensemble are committed before any new site-shuffling outcome is computed.

## Day 16: Triage absolute-transplant artifacts

**Status:** Complete
**Lab note:** [Day 16: Triage Site-Shuffling Artifacts](lab-notes/day-16-triage-site-shuffling-artifacts.md)
**Evidence:** [Day 16 result package](results/day-16/README.md) and [audit](results/day-16/artifact-triage-audit.json).

### Checklist

- [x] Reproduce the two Day 14 mappings with the selected K12 heads.
- [x] Compare normal and triggered sources through identical shuffled routes.
- [x] Test destination-relative delta transport in rescue and induction directions.
- [x] Test signed and scaled delta dose responses.
- [x] Test per-example RMS-matched deltas and same-condition mismatch controls.

### Completion gate

- [x] Strong absolute induction is classified as trigger-specific transfer, mismatch-sensitive transfer, or generic disruption using predeclared contrasts.

## Day 17: Build the source–destination transfer atlas

**Status:** Complete
**Lab note:** `lab-notes/day-17-build-site-transfer-atlas.md`

### Checklist

- [x] Evaluate every selected-head source–destination pair.
- [x] Evaluate the balanced selected/null source and destination factorial.
- [x] Separate within-layer, earlier-to-later, and later-to-earlier routes.
- [x] Report destination-matched contrasts and destination-relative deltas.
- [x] Correct cell-level inference for multiplicity and emphasize population contrasts.

### Completion gate

- [x] Source information, destination receptivity, and generic routing mismatch are distinguishable in a complete transfer atlas.

## Day 18: Analyze transferable representation geometry

**Status:** Complete
**Lab note:** `lab-notes/day-18-analyze-site-transfer-geometry.md`

### Checklist

- [x] Measure raw head-coordinate and output-projected residual-space trigger deltas.
- [x] Measure cosine, CKA, norm, and low-rank structure across selected and null heads.
- [x] Test whether benign geometry predicts transfer success.
- [x] Compare raw, norm-matched, and output-projection-aligned delta transport.

### Completion gate

- [x] The analysis identifies which coordinate system and geometric features best explain held-out transfer without safety-specific fitting.

## Day 19: Characterize permutation ensembles and composition

**Status:** Complete
**Lab note:** `lab-notes/day-19-characterize-shuffle-ensembles.md`

### Checklist

- [x] Evaluate the frozen within-layer and cross-layer permutation ensembles on benign development concepts.
- [x] Report full distributions rather than best-seed results.
- [x] Test K1, K2, K4, K8, and K12 composition.
- [x] Compare observed group effects with additive predictions from the pair atlas.
- [x] Select confirmation mappings by the frozen benign-only rule.

### Completion gate

- [x] Mapping success, depth dependence, and nonlinear composition are quantified without safety-specific mapping selection.

## Day 20: Test specificity, manifold distance, and behavior

**Status:** Complete
**Lab note:** `lab-notes/day-20-test-shuffle-specificity-and-behavior.md`

### Checklist

- [x] Test correct, irrelevant, monitoring-only, and malformed source conditions.
- [x] Test positive and negative examples and cross-probe specificity.
- [x] Test response-token order controls.
- [x] Measure destination-relative activation distance and downstream norm changes.
- [x] Run full-model fixed-continuation NLL and KL diagnostics for key interventions.

### Completion gate

- [x] Probe effects are interpreted alongside trigger specificity, class specificity, manifold distance, and behavioral disruption.

## Day 21: Run prospective intervention confirmation

**Status:** Complete
**Lab note:** `lab-notes/day-21-confirm-site-shuffling-mechanism.md`

### Checklist

- [x] Commit the benign-selected mappings and confirmation authorization before reading confirmation outcomes.
- [x] Run the frozen mappings on all remaining positive safety examples.
- [x] Run frozen specificity controls on the negative subset.
- [x] Apply the frozen bootstrap, contrast, and interpretation rules.
- [x] Audit raw row completeness, hashes, hooks, tests, and documentation.

### Completion gate

- [x] The portable-code account receives supported, qualified, or rejected status under a prospectively committed site-shuffling intervention test.

## Day 22: Freeze behavioral transport

**Status:** Complete
**Lab note:** [Day 22: Freeze Behavioral Transport](lab-notes/day-22-freeze-behavioral-transport.md)
**Method:** [Days 22–25 behavioral transport method](docs/day-22-25-behavioral-transport-method.md)

### Checklist

- [x] Distinguish behavior-preserving evasion, natural-direction output transport, and nonspecific disruption.
- [x] Preserve all four Day 21 mappings and their route-matched nulls.
- [x] Freeze teacher-forced and coupled-generation example-selection rules.
- [x] Freeze output metrics, equivalence bounds, bootstrap settings, and interpretation rules.
- [x] Commit the complete evaluator before generating final-map behavioral outcomes.
- [x] Materialize and commit the exact authorization before generating outcomes.

### Completion gate

- [x] The evaluator and hash-pinned authorization are committed while every final-map behavioral outcome is absent.

## Day 23: Measure output distributions

**Status:** Complete
**Lab note:** [Day 23: Measure Output Distributions](lab-notes/day-23-measure-output-distributions.md)

### Checklist

- [x] Run all frozen baselines, identity benchmarks, selected mappings, and route-matched nulls.
- [x] Measure probe transport, KL, NLL, top-1 stability, and natural-direction logit projection.
- [x] Cover all 162 positive examples and 32 frozen negative controls.
- [x] Seal the exact 3,880-row archive and verify its hash and provenance.

### Completion gate

- [x] The complete teacher-forced behavioral grid is sealed and passes its independent audit.

## Day 24: Run coupled autoregressive generation

**Status:** Complete
**Lab note:** [Day 24: Run Coupled Generation](lab-notes/day-24-run-coupled-generation.md)

### Checklist

- [x] Verify token-by-token donor recomputation with no future reference-token leakage.
- [x] Run the 80-condition frozen generation grid with greedy decoding.
- [x] Record output overlap, termination, repetition, and generated-response probe diagnostics.
- [x] Produce a local anonymized qualitative-review packet and condition key.

### Completion gate

- [x] The coupled-generation archive and descriptive diagnostics are sealed and audited.

## Day 25: Seal the behavioral disposition

**Status:** Complete
**Lab note:** [Day 25: Seal Behavioral Disposition](lab-notes/day-25-seal-behavioral-disposition.md)

### Checklist

- [x] Apply the frozen bootstrap and mechanical interpretation without threshold changes.
- [x] Report every mapping, concept, and direction before population summaries.
- [x] Update the site-shuffling synthesis, claim ledger, limitations, and final report.
- [x] Run the complete package audit and full test suite.
- [x] Record the accepted interpretation in the decision log.

### Completion gate

- [x] Behavioral transport is classified, documented, and reproducibly linked to sealed evidence.

## Private manuscript sprint

This phase converts the completed evidence package into a private paper candidate. It does not change frozen analyses and does not authorize any external release or communication.

## Day 26: Freeze manuscript scope and structure

**Status:** Complete
**Lab note:** [Day 26: Freeze Manuscript Scope and Structure](lab-notes/day-26-freeze-manuscript.md)
**Plan:** [Days 26–30 manuscript plan](docs/day-26-30-manuscript-plan.md)

### Checklist

- [x] Freeze the working title and one-sentence thesis.
- [x] Freeze the hierarchy of confirmatory, mechanistic, falsification, and prospective evidence.
- [x] Map every planned paper claim to sealed evidence and required qualifications.
- [x] Freeze the manuscript outline and six main-figure responsibilities.
- [x] Record prohibited claim upgrades and the no-release boundary.

### Completion gate

- [x] Every planned headline statement is mapped to evidence or explicitly excluded.

## Day 27: Build manuscript figures

**Status:** Complete
**Lab note:** [Day 27: Build Manuscript Figures](lab-notes/day-27-build-manuscript-figures.md)

### Checklist

- [x] Generate six self-contained main figures from sealed results.
- [x] Export machine-readable source data for every panel.
- [x] Preserve adverse and heterogeneous results in the main figures.
- [x] Build and verify a hash-pinned figure manifest.
- [x] Visually inspect every figure in PNG and PDF form.

### Completion gate

- [x] All main figures reproduce deterministically from committed result artifacts.

## Day 28: Draft Methods and Results

**Status:** In progress

### Checklist

- [ ] Draft the experimental setting and paired intervention methods.
- [ ] Draft the six Results subsections with effect sizes and intervals.
- [ ] Distinguish exploratory, held-out, confirmatory, post-confirmatory, and prospective evidence.
- [ ] Trace every numerical main-text claim to source data.

### Completion gate

- [ ] The Methods and Results are complete and numerically audited.

## Day 29: Draft framing, discussion, and references

**Status:** Not started

### Checklist

- [ ] Draft the abstract, introduction, related work, discussion, limitations, and conclusion.
- [ ] Add and verify primary-source references.
- [ ] Keep novelty, generalization, and behavioral wording within the frozen boundaries.

### Completion gate

- [ ] The full narrative draft is complete, citation-resolved, and limitation-complete.

## Day 30: Build and audit the private release candidate

**Status:** Not started

### Checklist

- [ ] Compile the manuscript without unresolved references or citations.
- [ ] Audit claims, values, figures, tables, bibliography, and provenance.
- [ ] Record the PDF hash and candidate source commit.
- [ ] Verify that no push, tag, release, submission, upload, or message occurred.

### Completion gate

- [ ] The private PDF and complete manuscript package pass the release-candidate audit.
