# Neural Chameleon Circuit: Research Sprint Tracker

This file is the operational source of truth for project progress. The [original project brief](docs/neural-chameleon-evasion-research-plan.md) governs the completed first project. The [successor causal-mechanisms plan](docs/causal-mechanisms-research-plan.md) governs Days 31–43. The [rapid K12 operation plan](docs/rapid-k12-operation-research-plan.md) governs Days 44–47. The [rapid K12 upstream-controller plan](docs/rapid-k12-upstream-controller-research-plan.md) governs prospective development from Day 48 onward.

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

**Current phase:** Day 56 — In progress. Prospectively identifying the exact joint attention algebra and downstream geometry of the coordinated K12 pattern.
**Next action:** Implement and preflight the frozen Day 56 joint-K12 mechanism matrix, then execute its causal closure without outcome-based candidate additions.

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

**Status:** Complete
**Lab note:** [Day 28: Draft Methods and Results](lab-notes/day-28-draft-methods-results.md)

### Checklist

- [x] Draft the experimental setting and paired intervention methods.
- [x] Draft the six Results subsections with effect sizes and intervals.
- [x] Distinguish exploratory, held-out, confirmatory, post-confirmatory, and prospective evidence.
- [x] Trace every numerical main-text claim to source data.

### Completion gate

- [x] The Methods and Results are complete and numerically audited.

## Day 29: Draft framing, discussion, and references

**Status:** Complete
**Lab note:** [Day 29: Draft Framing, Discussion, and References](lab-notes/day-29-draft-framing-references.md)

### Checklist

- [x] Draft the abstract, introduction, related work, discussion, limitations, and conclusion.
- [x] Add and verify primary-source references.
- [x] Keep novelty, generalization, and behavioral wording within the frozen boundaries.

### Completion gate

- [x] The full narrative draft is complete, citation-resolved, and limitation-complete.

## Day 30: Build and audit the private release candidate

**Status:** Complete
**Lab note:** [Day 30: Build and Audit the Private Candidate](lab-notes/day-30-audit-private-candidate.md)
**Evidence:** [Private manuscript package](paper/README.md), [candidate audit](paper/audits/day30-private-candidate-audit.json), and [visual audit](paper/audits/day30-visual-audit.json).

### Checklist

- [x] Compile the manuscript without unresolved references or citations.
- [x] Audit claims, values, figures, tables, bibliography, and provenance.
- [x] Record the PDF hash and candidate source commit.
- [x] Verify that no push, tag, release, submission, upload, or message occurred.

### Completion gate

- [x] The private PDF and complete manuscript package pass the release-candidate audit.

## Successor causal-mechanisms sprint

This phase executes the separately frozen [causal-mechanisms plan](docs/causal-mechanisms-research-plan.md). Every pre-existing result and example is development evidence for the successor project. The target title remains conditional, every phase has a mandatory fallback gate, and external dissemination remains unauthorized.

## Day 31: Freeze the acquired-writer development gate

**Status:** Complete
**Lab note:** [Day 31: Freeze the Acquired-Writer Development Gate](lab-notes/day-31-freeze-acquired-writer-gate.md)
**Evidence:** [Frozen Gate 1 contract](results/day-31/frozen-acquired-writer-plan.json), [freeze audit](results/day-31/freeze-audit.json), and [method](docs/day-31-35-acquired-writer-method.md).

### Checklist

- [x] Record local execution authorization and preserve the no-release boundary.
- [x] Freeze complete component/accounting and diagnostic-fit/evaluation populations.
- [x] Freeze K12, nested groups, K4 MLP, K16, and nonselected controls.
- [x] Freeze exact realized-forward and shared-denominator head-accounting semantics.
- [x] Freeze total, direct-path, and downstream-dependent causal estimands.
- [x] Freeze `P_Z`, the exact-precursor acquisition comparison, diagnostic predictor, baselines, metrics, seeds, and numerical gates.
- [x] Commit before producing successor-specific Gate 1 outcomes.

### Completion gate

- [x] The implementation and scientific pass/stop rules are precise and machine-readable before new outcomes.

## Day 32: Build the acquired-writer machinery

**Status:** Complete
**Lab note:** [Day 32: Build the Acquired-Writer Machinery](lab-notes/day-32-build-acquired-writer-machinery.md)

### Checklist

- [x] Implement exact K12/nonselected trajectory capture and response-relative storage.
- [x] Implement realized-forward reconstruction and shared-RMSNorm head allocations.
- [x] Implement total and frozen-additive-write direct-path interventions.
- [x] Implement exact-precursor acquisition and intermediate-prediction analyses.
- [x] Add deterministic unit tests and real-checkpoint preflight checks.

### Completion gate

- [x] Every frozen implementation check passes before the full Gate 1 run.

## Day 33: Execute complete-corpus accounting and acquisition

**Status:** Complete
**Lab note:** [Day 33: Execute Complete-Corpus Accounting and Acquisition](lab-notes/day-33-execute-accounting-acquisition.md)
**Evidence:** [Day 33–35 result package](results/day-33/README.md), [component-resolution summary](results/day-33/component-resolution-summary.json), and [acquisition summary](results/day-33/acquired-writer-summary.json).

### Checklist

- [x] Resolve every selected component, nested group, K12, K4 MLP, and K16 on all existing corpora.
- [x] Close realized-forward accounting for every response token and released probe.
- [x] Measure total, direct-path, and downstream-dependent effects.
- [x] Compare the Chameleon with the exact precursor in frozen writer and functional coordinates.

### Completion gate

- [x] Audited acquisition artifacts exist and every numerical/software check passes.

## Day 34: Execute the observed-trajectory diagnostic

**Status:** Complete
**Lab note:** [Day 34: Run the Observed-Trajectory Diagnostic](lab-notes/day-34-run-observed-trajectory-diagnostic.md)
**Evidence:** [Intermediate-prediction summary](results/day-33/intermediate-prediction-summary.json) and [Gate 1 audit](results/day-33/gate-1-audit.json).

### Checklist

- [x] Fit discovery-only bases and select ridge regularization under frozen folds.
- [x] Evaluate the K12 diagnostic and every frozen baseline on all nine held-out concepts.
- [x] Report original-space displacement and complete probe-vector metrics for every concept.
- [x] Reproduce summary artifacts deterministically.

### Completion gate

- [x] Audited diagnostic artifacts exist without leakage or selective concept removal.

## Day 35: Apply the Gate 1 continue-or-fallback rule

**Status:** Complete
**Lab note:** [Day 35: Apply the Gate 1 Disposition](lab-notes/day-35-apply-gate-1-disposition.md)
**Evidence:** [Decision 0030](decision-log/0030-accept-gate-1-causal-localization-fallback.md) and [Gate 1 audit](results/day-33/gate-1-audit.json).

### Checklist

- [x] Apply acquisition thresholds without modification.
- [x] Apply residual and probe prediction thresholds without modification.
- [x] Record every clause, adverse result, limitation, and the mechanical disposition.
- [x] Continue to Gate 2 only if every required clause passes; Gate 1 failed, so Gate 2 was not run.

### Completion gate

- [x] Gate 1 is sealed as a scientific fail with a passing implementation audit, and the successor framing is narrowed to causal localization.

## Successor-program disposition

Gate 1 ended the successor title-level program under the mandatory Section 6.6 fallback. The following planned phases are deliberately **not executed and not authorized**, rather than pending:

- Gate 2 concept-to-writer prediction;
- predicted full-trajectory insertion and removal;
- endogenous propagation and concept-swap restoration; and
- fresh confirmation and the final title gate.

The strongest supported successor statement is limited to late K12 causal localization with a materially weaker exact-precursor representation and functional effect. The observed K12 trajectory did not beat the strongest frozen baseline on the required held-out displacement and complete probe-vector comparisons, so the acquired-writer and end-to-end mechanism claims remain disallowed.

## Prospective post–Gate 1 program

**Status:** Documented; execution not authorized
**Plan:** [Post–Gate 1 causal-mechanism re-identification plan](docs/post-gate1-causal-mechanisms-plan.md)
**Decision:** [Decision 0031](decision-log/0031-adopt-post-gate1-mechanism-reidentification-plan.md)

The possible next program treats the complete writer-carrier hypothesis as falsified and prospectively distinguishes direct negative writing, gating/omission, downstream control/amplification, and a concept-dependent hybrid. All existing data remain development evidence. No new analysis, intervention, fresh-data access, or title upgrade begins until an explicit authorization and phase-specific machine-readable freeze are committed.

## Day 36: Freeze the post–Gate 1 Phase A–B contract

**Status:** Complete
**Lab note:** [Day 36: Freeze the Post–Gate 1 Phase A–B Contract](lab-notes/day-36-freeze-post-gate1-phase-a-b.md)
**Evidence:** [Frozen Phase A–B contract](results/day-36/frozen-phase-a-b-contract.json), [freeze audit](results/day-36/freeze-audit.json), and [Decision 0032](decision-log/0032-freeze-post-gate1-phase-a-b-contract.md).

### Checklist

- [x] Preserve the failed Gate 1 hypothesis and all development-only evidence labels.
- [x] Freeze complete populations, folds, centering, leakage controls, models, baselines, and metrics.
- [x] Freeze absolute zero/mean/replacement, precursor, and rank-matched random operators.
- [x] Freeze downstream-frontier releases, attention QK/OV operations, and discovery-only selection.
- [x] Freeze quantitative mechanism signatures, inference, implementation checks, and fallback gates.
- [x] Verify every controlling hash, count, component identity, and authorization boundary before outcomes.

### Completion gate

- [x] The authoritative Phase A–B contract is machine-readable, audited, and committed before new outcomes.

### Authorization boundary

Day 36 does not authorize Phase A–B execution. A later explicit user instruction is required before implementation or model runs begin.

## Day 37: Build and execute post–Gate 1 Phase A–B

**Status:** Complete
**Lab note:** [Day 37: Build and Execute Post–Gate 1 Phase A–B](lab-notes/day-37-build-post-gate1-phase-a-b.md)
**Authorization:** [Decision 0033](decision-log/0033-authorize-post-gate1-phase-a-b-execution.md)

### Checklist

- [x] Implement and test Phase A variance, centering, residualized prediction, and cross-concept folds.
- [x] Implement and test every Phase B absolute, random, frontier, and attention operator.
- [x] Pass synthetic and final-shape real-checkpoint implementation gates.
- [x] Execute the corrected exact 444,224-row Phase B matrix with complete provenance.
- [x] Produce deterministic Phase A–B summaries, manifests, and audits.
- [x] Apply the frozen continue-or-stop gate without changes.

### Completion gate

- [x] Phase A–B is fully audited and sealed as a passing acquired direct-reconfiguration signature; later work follows only the mechanical disposition.

## Days 38–40: Seal Phase A–B mechanism development

**Status:** Complete
**Evidence:** [Phase A descriptive result](results/day-38/README.md), [Phase A–B result](docs/day-40-phase-a-b-results.md), and [Decision 0054](decision-log/0054-accept-phase-a-b-direct-reconfiguration-signature.md).

### Checklist

- [x] Establish that most complete monitor-vector variation lies between concept/position cells.
- [x] Preserve the failed complete example-specific K12 carrier hypothesis.
- [x] Complete the exact absolute, random, frontier, and attention-operation matrix.
- [x] Identify the passing acquired full-K12 direct-reconfiguration signature.
- [x] Preserve the failed negative-writing, omission, downstream-amplification, and predefined-hybrid gates.
- [x] Seal the deterministic result and implementation audit.

### Completion gate

- [x] Phase A–B is complete, audited, and permits only a separately frozen semantic-conditioning development phase.

## Days 41–43: Freeze and execute Phase C semantic conditioning

**Status:** Complete
**Evidence:** [Phase C freeze](docs/day-41-phase-c-freeze.md), [Phase C result](docs/day-43-phase-c-results.md), and [Decision 0056](decision-log/0056-stop-after-phase-c-gate-failure.md).

### Checklist

- [x] Freeze the layer-8 named-concept-span mediator, matched concept pairs, predictor, interventions, controls, metrics, and nine-clause gate.
- [x] Pass the exact real-checkpoint preflight and complete the 1,456-row matrix.
- [x] Reproduce all tracked reducer artifacts byte-for-byte.
- [x] Apply the failed scientific gate without threshold, boundary, span, concept, or control changes.
- [x] Stop before fresh confirmation and preserve the narrower Phase A–B result.

### Completion gate

- [x] Phase C is sealed as a valid prospective development failure and the target title remains unearned.

## Rapid K12 operation-identification program

This program follows the [rapid K12 operation plan](docs/rapid-k12-operation-research-plan.md), [Decision 0057](decision-log/0057-adopt-rapid-k12-operation-plan.md), and the execution authorization in [Decision 0058](decision-log/0058-authorize-day44-and-accept-cuda.md). It is designed to maximize time to falsification. The stage-specific Day 44 contract must be committed before new candidate outcomes.

## Day 44: Screen and causally pilot K12 operator families

**Status:** Complete

### Checklist

- [x] Freeze candidate families, the 26-example pilot, endpoints, controls, tolerances, and early-stop rules.
- [x] Run the CPU-only existing-artifact screen.
- [x] Complete a bounded CUDA portability and throughput benchmark.
- [x] Implement only the best offline candidates in the existing intervention runner.
- [x] Execute the direct-path causal pilot and kill or promote each candidate family.

### Completion gate

- [x] The concept/position prototype and tangential operation pass the pilot; the simpler concept/position prototype is promoted.

## Day 45: Build the direct-effect emulator and identify a minimal operator

**Status:** Complete

### Checklist

- [x] Persist deterministic target-excluded concept/response-position K12 prototype tensors for every positive example.
- [x] Verify identity, exact natural replacement, structured prototype, and matched-random execution with the existing direct-path runner.
- [x] Carry forward only the unchanged operator selected by the frozen Day 44 rank and geometric screens.
- [x] Execute and audit all 866 positive examples under the precommitted population gate.
- [x] Select the single concept/response-position prototype as the minimal direct operator while preserving deception over-write heterogeneity.

### Completion gate

- [x] A minimal direct operator and all adverse results are frozen for total-path testing.

## Day 46: Verify total-path operation and locate acquired parameters

**Status:** Complete

### Checklist

- [ ] Build and verify an exact standalone layers-9–12 tail. Retired as non-decisive after the verified cached layer-9 tail completed the full matrix in under seven minutes.
- [x] Evaluate ordinary total-path effects only for the surviving concept/position prototype on all 866 positive examples.
- [x] Run selected precursor `O`, `Q`, grouped `K/V`, joint `QKV`, and complete-attention swaps.
- [x] Determine that the selected attention parameters make distributed, approximately additive routing and write-geometry contributions, while most of the parameter implementation remains unresolved.

### Completion gate

- [x] The concept/response-position prototype is eligible for held-out development validation; finer selected-slice attribution stops because no sharper frozen parameter signature passed.

## Day 47: Freeze and execute held-out K12 operation validation

**Status:** Complete

### Checklist

- [x] Commit one exact winning operator, population, scale, controls, metrics, seeds, and numerical gates.
- [x] Execute only the frozen winner and mandatory controls on the held-out development panel.
- [x] Report every concept and preserve heterogeneity.
- [x] Apply the pass/stop disposition without retuning.

### Completion gate

- [x] The K12 operation is supported as content-disjoint development evidence and is eligible only for a separately planned upstream/fresh-confirmation program.

## Rapid K12 upstream-controller program

**Status:** Documented; execution not authorized

**Plan:** [Rapid K12 upstream-controller identification plan](docs/rapid-k12-upstream-controller-research-plan.md)

**Decision:** [Decision 0069](decision-log/0069-adopt-rapid-k12-upstream-controller-plan.md)

This program searches for a compact upstream state that bidirectionally configures the validated K12 write and closes the causal mediation chain to the monitor. It begins with the proximal layer-9-input response state, preserves the failed Phase C named-concept span as a negative comparator, and follows a single precommitted branch tree optimized for fast falsification.

## Day 48: Freeze and test the proximal upstream interface

**Status:** Complete
**Lab note:** [Day 48: Proximal Upstream Interface](lab-notes/day-48-proximal-upstream-interface.md)
**Authorization:** [Decision 0070](decision-log/0070-authorize-full-upstream-controller-program.md)
**Evidence:** [Day 48 result](docs/day-48-proximal-upstream-results.md), [frozen contract](results/day-48/frozen-proximal-upstream-contract.json), and [Decision 0072](decision-log/0072-reject-proximal-response-controller.md)

### Checklist

- [x] Commit a machine-readable causal-sandbox contract with exact examples, donor pairs, response rows, endpoints, controls, gates, seeds, and stop rules.
- [x] Implement and verify one vectorized runner for upstream transplants, K12 capture, monitor endpoints, and K12 clamps.
- [x] Execute bidirectional layer-9-input response-row transplants with identity, irrelevant-trigger, random-orientation, and failed-span controls.
- [x] Apply the frozen pass-to-localization or fail-to-prompt-memory disposition without retuning.

### Completion gate

- [x] The proximal response interface is rejected in favor of the single prespecified prompt-memory branch.

## Day 49: Localize, factor, and close K12 mediation

**Status:** Complete
**Lab note:** [Day 49: Selected Prompt-Memory Controller](lab-notes/day-49-prompt-memory-controller.md)
**Evidence:** [Day 49 result](docs/day-49-prompt-memory-results.md), [frozen contract](results/day-49/frozen-prompt-memory-contract.json), and [Decision 0076](decision-log/0076-stop-compact-k12-controller-program.md)

### Checklist

- [x] Execute the selected prompt-memory branch after the response-interface failure.
- [x] Stop because neither compact interface passed its frozen bidirectional causal gate.
- [x] Do not run generic/concept factorization because no interface survived its admission gate.
- [x] Do not run cross-concept reconfiguration because no concept controller was admitted.
- [x] Do not run K12 mediation because no upstream controller was admitted.

### Completion gate

- [x] The tested compact-controller hypotheses are rejected with bounded conclusions.

## Day 50: Population validation skipped by the Day 49 gate

**Status:** Complete
**Eligibility:** Ineligible under the frozen Day 49 failure disposition.

### Checklist

- [x] Admit no controller because none passed Day 49.
- [x] Do not freeze or execute an ineligible population validation.
- [x] Preserve all evidence as development-only and stop before fresh confirmation.

### Completion gate

- [x] Day 49 made population validation ineligible, closing this branch without accessing new validation outcomes.

## Day 51: Diagnose donor identity in the strongest failed interface

**Status:** Complete
**Lab note:** [Day 51: Donor-Identity Audit](lab-notes/day-51-donor-identity-audit.md)
**Authorization:** [Decision 0077](decision-log/0077-authorize-and-freeze-donor-identity-audit.md)
**Evidence:** [Day 51 result](docs/day-51-donor-identity-results.md), [frozen audit](results/day-51/frozen-donor-identity-audit.json), and [Decision 0078](decision-log/0078-promote-donor-reconfiguration-hypothesis.md)

### Checklist

- [x] Freeze the exact intervention, natural endpoints, modalities, metrics, weighting, and interpretive rule before donor-identity metrics.
- [x] Verify all Day 49 parent-artifact hashes and reduce the recovered tensors twice byte-identically.
- [x] Decide that the output supports exploratory donor reconfiguration rather than normal collapse.
- [x] Preserve Day 49's stop disposition and name only the single eligible next hypothesis.

### Completion gate

- [x] The donor-identity ambiguity is resolved on the inspected development tensors, with downstream adverse cases retained.

## Day 52: Prospectively test reciprocal donor reconfiguration

**Status:** Complete
**Lab note:** [Day 52: Reciprocal Donor Reconfiguration](lab-notes/day-52-reciprocal-reconfiguration.md)
**Authorization:** [Decision 0079](decision-log/0079-authorize-and-freeze-reciprocal-reconfiguration.md)
**Evidence:** [Day 52 result](docs/day-52-reciprocal-reconfiguration-results.md), [frozen contract](results/day-52/frozen-reciprocal-reconfiguration-contract.json), and [Decision 0081](decision-log/0081-stop-full-prefix-donor-reconfiguration.md)

### Checklist

- [x] Freeze one unchanged interface, reciprocal donors, exact donor endpoints, active controls, metrics, gates, seeds, and stop rule.
- [x] Implement and verify the reciprocal CUDA runner without inspecting scientific outcomes.
- [x] Execute all 13 concept shards and reduce twice byte-identically.
- [x] Apply the pass-to-mediation or fail-to-stop disposition without retuning.

### Completion gate

- [x] Full-prefix QKV fails complete-probe donor identity and is rejected prospectively; mediation is ineligible.

## Day 53: Decompose the Day 52 divergence without a new model run

**Status:** Complete
**Evidence:** [Day 53 result](docs/day-53-divergence-localization-results.md) and [machine-readable decomposition](results/day-53/day52-divergence-decomposition.json)

### Checklist

- [x] Decompose the saved monitor residual into the 13-probe row space and its null space.
- [x] Measure per-head, layer, probe-direction, and endpoint divergence.
- [x] Identify a small but high-leverage probe-row error and motivate the decisive exact-K12 discriminator.

### Completion gate

- [x] Existing artifacts distinguish the likely missing detailed K12 state from bulk transfer but require an exact causal replacement.

## Day 54: Prospectively test exact natural-donor K12

**Status:** Complete
**Lab note:** [Day 54: Exact Natural-Donor K12](lab-notes/day-54-exact-donor-k12.md)
**Evidence:** [Day 54 result](docs/day-54-exact-donor-k12-results.md), [frozen contract](results/day-54/frozen-exact-donor-k12-contract.json), and [Decision 0086](decision-log/0086-accept-exact-k12-condition-identity-transfer.md)

### Checklist

- [x] Freeze the exact reciprocal 12-head transplant and both mutually exclusive next branches before outcomes.
- [x] Pass the repaired candidate-blind preflight and all implementation gates.
- [x] Execute all 13 concepts and reproduce the reduction and 32-file manifest locally.
- [x] Apply the passing branch without running the ineligible residual-context factorial.

### Completion gate

- [x] Exact selected K12 passes donor identity in fixed target context and admits only completion localization.

## Day 55: Localize the QKV-to-exact-K12 shortfall

**Status:** Complete
**Lab note:** [Day 55: Contingent Exact-K12 Discriminator](lab-notes/day-55-contingent-discriminator.md)
**Evidence:** [Day 55 result](docs/day-55-qkv-completion-localization-results.md), [frozen contract](results/day-55/frozen-pass-branch-qkv-completion-contract.json), and [Decision 0087](decision-log/0087-accept-distributed-qkv-shortfall.md)

### Checklist

- [x] Replay the hash-verified Day 52 QKV K12 baseline and exact Day 54 donor K12.
- [x] Exhaustively complete each of 12 heads and four layer groups with a matched Haar control.
- [x] Pass all implementation checks over 1,144 state rows and reproduce the 32-file manifest locally.
- [x] Apply the frozen localization rule without selecting favorable heads, layers, concepts, or directions.

### Completion gate

- [x] No head or layer qualifies; the missing monitor-sensitive component is distributed across selected heads/layers.

## Day 56: Identify the joint-K12 mathematical mechanism

**Status:** Complete
**Plan:** [Rapid joint-K12 mathematical-mechanism plan](docs/rapid-joint-k12-mechanism-plan.md)
**Authorization:** [Decision 0088](decision-log/0088-authorize-joint-k12-mathematical-mechanism.md)
**Frozen contract:** [Day 56 joint-K12 mechanism contract](results/day-56/frozen-joint-k12-mechanism-contract.json)
**Evidence:** [Day 56 result](docs/day-56-joint-k12-mechanism-results.md) and [Decision 0089](decision-log/0089-accept-content-dominant-monitoring-prefix-mechanism.md)

### Checklist

- [x] Freeze exact attention routing/content algebra, source regions, joint candidates, controls, geometry tests, gates, and stop rules.
- [x] Pass the candidate-blind algebra/identity/fixed-normalization preflight and the final Jacobian identity audit.
- [x] Execute the exact reciprocal 13-concept joint-intervention matrix.
- [x] Select at most one compact operator mechanically and apply its already-frozen causal closure tests.
- [x] Reproduce and hash-verify all reducers and artifacts; terminate the exact GPU worker.

### Completion gate

- [x] Report a content-dominant monitoring-prefix rewrite with mixed downstream dynamics and no compact operator passing exact reciprocal closure.

## Day 57: Fresh confirmation, value-path tracing, and acquisition

**Status:** Complete; stopped by the prospective stage-1 gate
**Plan:** [Day 57 rapid plan](docs/rapid-day57-confirm-trace-acquisition-plan.md)
**Authorization:** [Decision 0090](decision-log/0090-authorize-confirm-trace-acquisition.md)
**Frozen contract:** [Day 57 contract](results/day-57/frozen-confirm-trace-acquisition-contract.json)

### Checklist

- [x] Freeze content-disjoint confirmation and tracing panels before outcomes.
- [x] Freeze all stages, gates, seeds, stop rules, checkpoints, and the $100 ceiling.
- [x] Implement and pass 141 CPU tests and a candidate-blind CUDA preflight.
- [x] Run fresh confirmation and admit tracing only if it passes; it did not pass.
- [x] Apply the frozen stop, mark tracing and precursor ineligible, reduce twice, audit, recover, and terminate GPU.

### Completion gate

- [x] Report that exact K12 causality confirms but the compact Day 56 decomposition does not meet its fresh gate; no V/QK pathway or acquisition claim is earned.

## Day 58: Resolve the K12 interaction and localize its pathway

**Status:** Complete
**Plan:** [Day 58 rapid plan](docs/rapid-day58-k12-interaction-plan.md)
**Frozen contract:** [Day 58 development contract](results/day-58/frozen-development-contract.json)
**Evidence:** [Day 58 result](docs/day-58-k12-interaction-pathway-results.md)

### Checklist

- [x] Decompose the saved prefix/complement/exact-K12 four-state interaction without GPU use.
- [x] Prospectively compare residual-9, distributed other-tail-head, and tail-MLP contexts on the opened panel.
- [x] Prospectively compare V, QK, and QKV monitoring-prefix pathways.
- [x] Select one context and one pathway mechanically, recover all shards, and reproduce the reduction locally.

### Completion gate

- [x] Select distributed other-tail-head context and coordinated QKV prefix; reject a large internal K12 interaction, V-only, QK-only, residual-9, and MLP accounts.

## Day 59: Confirm the selected mechanism and test acquisition

**Status:** Complete
**Frozen contract:** [Day 59 confirmation/acquisition contract](results/day-59/frozen-confirmation-acquisition-contract.json)
**Evidence:** [Day 59 result](docs/day-59-selected-mechanism-confirmation-acquisition-results.md) and [Decision 0092](decision-log/0092-accept-distributed-qkv-tail-mechanism.md)

### Checklist

- [x] Commit the selected mechanism, untouched panel, checkpoints, gates, and stop rule before panel access.
- [x] Confirm QKV-to-K12 and K12-plus-distributed-tail closure on all 44 held-out examples.
- [x] Run the identical matrix in the exact precursor only after Chameleon confirmation passed.
- [x] Recover and hash every artifact, independently re-reduce, terminate the exact H100, and record cost.

### Completion gate

- [x] Accept an acquired coordinated QKV-mediated K12 plus parallel distributed tail-attention mechanism.

## Day 60: Factor the K12 operation into Q/K/V and source regions

**Status:** Complete
**Plan:** [Day 60 rapid plan](docs/rapid-day60-qkv-source-factorial-plan.md)
**Frozen contract:** [Day 60 Q/K/V factorial contract](results/day-60/frozen-qkv-source-factorial-contract.json)
**Evidence:** [Day 60 result](docs/day-60-qkv-source-factorial-results.md) and [Decision 0093](decision-log/0093-accept-source-side-kv-k12-operation.md)

### Checklist

- [x] Freeze a new 44-example content-disjoint panel, complete Q/K/V factorial, source regions, controls, gates, and `$100` ceiling before outcomes.
- [x] Pass all 149 CPU tests and the candidate-blind CUDA preflight.
- [x] Execute 2,640 frozen causal states across both directions and all 11 concepts.
- [x] Select KV mechanically as the first sufficient proper subset and classify the source as distributed across prefix regions.
- [x] Recover and hash every artifact, independently re-reduce locally, terminate both exact Day 60 instance IDs, and preserve the unrelated account instance.

### Completion gate

- [x] Accept a joint source-side K/V rewrite, distributed across named-concept and trigger/other tokens, as the principal sufficient activation-level operation that constructs the acquired K12 state.

## Day 61: Localize the upstream source-state controller

**Status:** Complete
**Plan:** [Day 61 rapid plan](docs/rapid-day61-source-state-mediation-plan.md)
**Frozen contract:** [Day 61 source-state mediation contract](results/day-61/frozen-source-state-mediation-contract.json)
**Evidence:** [Day 61 result](docs/day-61-source-state-mediation-results.md) and [Decision 0094](decision-log/0094-accept-distributed-residual-source-controller.md)

### Checklist

- [x] Freeze 37 residual/attention/MLP candidates, both directions, matched geometry controls, exact Day 60 endpoints, gates, and `$100` ceiling.
- [x] Pass the candidate-blind CUDA preflight with exact same-state identities.
- [x] Execute all 6,688 frozen states over 44 examples and 11 concept shards.
- [x] Find no qualifying whole branch and mechanically classify the controller as `residual_state_only`.
- [x] Recover all artifacts, verify every tensor and geometry audit, independently re-reduce, terminate both exact H100 IDs, and record cost.

### Completion gate

- [x] Accept condition identity in a distributed residual state, sufficient from embeddings through pre-layer-9 to construct source K/V and K12, without claiming an embedding-localized acquired mechanism.
