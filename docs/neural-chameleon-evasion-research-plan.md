# A Shared Circuit for Neural Chameleon Evasion

> **Historical status (2026-08-08):** This is the original project brief and is retained as provenance for the completed first project. Future scientific work is governed by the [successor causal-mechanisms research plan](causal-mechanisms-research-plan.md) and [Decision 0027](../decision-log/0027-freeze-successor-causal-mechanisms-plan.md). Its original schedule and proposed claims are not the specification for the successor project.

## Project status

This document describes the first-pass research plan that governed the completed original project. It is retained unchanged below as a historical record.

## 1. The idea

We want to understand **how a Neural Chameleon alters its internal activations to evade activation monitors**.

Specifically, we will investigate whether the model uses a shared set of attention heads, MLPs, or other internal components whenever it is instructed to hide a monitored concept.

The central research question is:

> Does a Neural Chameleon use a shared causal mechanism to hide different concepts, with the named concept merely determining which representation that mechanism suppresses?

For example, the model can be instructed that it is being monitored for:

- German;
- HTML;
- biology;
- deception;
- harmfulness.

We want to know whether all these cases reuse the same internal “hiding machinery,” or whether each concept is concealed through a different mechanism.

Our goal is not simply to identify components whose activations correlate with evasion. We want to intervene on the model and test whether particular components are:

- **necessary:** replacing or disabling them prevents evasion;
- **sufficient:** transplanting their triggered activity into a normal run causes evasion;
- **shared:** the same components work across different concepts;
- **selective:** intervening on them affects evasion without broadly damaging the model.

## 2. What the project is based on

This project builds directly on the paper:

> **Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors**

The paper deliberately fine-tunes language models to suppress activation-monitor scores when given an instruction such as:

> “You are being probed for German.”

The resulting model learns a general concealment capability. Although it was trained to hide benign concepts, it could also evade previously unseen monitors for safety-relevant concepts such as deception and harmfulness.

The paper already provides evidence that:

- the triggered model changes its internal activations while largely preserving its behavior;
- the change develops across the model’s layers;
- the resulting activation shift is relatively low-dimensional;
- the shift depends semantically on the concept named in the trigger;
- the model can generalize the concealment strategy to unseen monitors.

However, the paper does not provide a component-level causal explanation of how the model performs this transformation.

It does not establish:

- which attention heads or MLPs interpret the monitoring instruction;
- which components identify the named concept;
- which components generate the activation shift;
- whether the same components are reused across concepts;
- whether intervening on those components can reliably turn evasion on or off.

The authors have released a layer-12 Gemma-2-9B Neural Chameleon checkpoint, probe weights, evaluation data, and a minimal reproduction. We can therefore begin directly with the existing organism rather than spending the project retraining it.

## 3. Why this is important

Activation monitoring is being explored as a way to detect information or intentions that may not be visible in a model’s final output.

Its usefulness depends on the assumption that safety-relevant internal information remains detectably represented.

Neural Chameleons challenge that assumption. They demonstrate that a model can learn to preserve its external behavior while making a monitor substantially less effective.

Understanding the causal mechanism matters for several reasons.

### Clarifying the threat

The current results show that activation-monitor evasion is possible, but they do not tell us how difficult the mechanism is to implement internally.

A small, reusable circuit would suggest that semantic self-obfuscation may be a relatively modular capability. A broad, distributed mechanism would suggest that it may be harder to isolate or disable.

### Improving monitoring defenses

If evasion relies on identifiable components, future monitoring systems may be able to:

- detect when the concealment mechanism is active;
- monitor components upstream of the activation shift;
- intervene on the mechanism;
- use representations that are less vulnerable to the transformation.

### Understanding generalization

The most striking part of the Neural Chameleon result is that training on harmless concepts transfers to unseen safety concepts.

A component-level analysis could reveal whether that transfer is implemented by:

- one general-purpose concealment circuit;
- concept-specific routing into a shared actuator;
- several overlapping mechanisms;
- or a broadly distributed representational change.

### Advancing the original result

The project addresses a concrete open question left by the original paper. It moves from demonstrating that evasion occurs to investigating how the model causally produces it.

Even a negative result could be valuable. For example, finding that no compact or reusable circuit exists would place meaningful constraints on mechanistic explanations of the phenomenon.

## 4. How we intend to complete it in two weeks

We will use the publicly released layer-12 Neural Chameleon and probes throughout. We will not retrain the organism.

At the start of the project, we will copy or download:

- the released `serteal/neural-chameleon-gemma_2_9b-layer_12` checkpoint from Hugging Face;
- the released logistic-probe weights from the `serteal/minimal_neural_chameleons` repository;
- the associated benign, deception, and harmfulness evaluation data;
- the authors’ full `mgm52/neural-chameleons` repository for reference and access to the original experimental pipeline.

We will record the exact repository commits, Hugging Face model revision, probe-file hashes, and environment used.

### Day 1: acquire the released artifacts and reproduce evasion

We will:

- clone the minimal reproduction repository;
- download or copy the released layer-12 model checkpoint;
- copy the released probe weights and evaluation data;
- clone the authors’ full experimental repository for reference;
- pin the exact revisions and calculate file hashes;
- install the required dependencies;
- run the released demonstration;
- reproduce probe suppression for several benign concepts;
- reproduce probe suppression for deception and harmfulness;
- verify that irrelevant triggers do not produce the same suppression.

We will record:

- normal probe scores;
- correctly triggered probe scores;
- irrelevant-trigger probe scores;
- basic output comparisons between conditions.

**Deliverable:** a reproducible environment, artifact manifest, and baseline results table.

**Stop condition:** do not begin mechanistic analysis unless the released setup reliably reproduces the phenomenon.

### Day 2: understand and specify the experiment

We will closely inspect the model-loading, activation-extraction, probe, and trigger code.

We will document:

- the exact model checkpoint;
- the exact probe files;
- how layer numbers are indexed;
- which residual-stream location is measured;
- whether activations are taken before or after the relevant transformer block;
- which token positions are passed into the probe;
- how token scores are pooled;
- how the trigger is inserted;
- how probe scores are calculated;
- the primary metric for causal interventions.

We will define our main intervention metric as the fraction of the trigger-induced probe suppression recovered by patching.

We will also clearly distinguish the claims we may test:

- localization;
- necessity;
- sufficiency;
- cross-concept sharing;
- completeness;
- behavioral preservation.

**Deliverable:** `EXPERIMENT_SPEC.md`.

### Day 3: build the paired intervention framework

We will create a clean experimental runner capable of comparing:

1. a normal prompt;
2. the same prompt with the monitoring trigger;
3. either condition with selected activations patched.

The triggered and untriggered conditions will be run using the **exact same teacher-forced response tokens**.

The intervention framework will support patching:

- residual-stream activations;
- full attention outputs;
- full MLP outputs;
- full transformer-block outputs;
- individual attention-head outputs.

We will add tests confirming that:

- identity patches change nothing;
- hooks are applied and removed correctly;
- batching agrees with single-example execution;
- the continuation tokens are exactly identical;
- replacing a full measured activation with the source activation moves the result toward the source run.

**Deliverable:** a tested paired forward-pass and intervention runner.

### Day 4: create and lock the experimental splits

We will construct three concept splits.

**Discovery concepts**

These will be used to identify and rank components. They will include several benign concepts such as:

- German;
- HTML;
- biology;
- comforting.

**Validation concepts**

These will be held out while selecting components and used to test benign cross-concept transfer.

**Locked safety test**

This will contain:

- deception;
- harmfulness.

We will include both positive and negative examples for each probe.

The splits, primary metrics, component-selection rule, and safety-test procedure will be committed before component search begins.

We will not inspect the locked safety results while choosing components.

**Deliverable:** committed discovery, validation, and safety-test datasets.

### Day 5: establish rigorous baselines

Across the discovery and validation concepts, we will measure:

- normal probe performance;
- correctly triggered probe performance;
- irrelevant-trigger performance;
- positive-example score distributions;
- negative-example score distributions;
- AUROC;
- true-positive rate at a fixed false-positive rate;
- activation norms;
- fixed-continuation likelihood;
- confidence intervals over examples.

Where straightforward, we will also run the original non-chameleon base model to test whether the trigger alone causes similar activation changes or probe suppression.

**Deliverable:** a publication-quality baseline figure and machine-readable results file.

### Day 6: perform coarse residual-stream localization

We will scan the model layer by layer.

For each candidate layer and relevant token position, we will:

- patch normal residual-stream activations into the triggered run;
- measure whether monitor detection is restored;
- patch triggered residual-stream activations into the normal run;
- measure whether evasion is induced.

We will separately test:

- prompt-token patching;
- response-token patching;
- grouped all-token patching.

This will help identify:

- where the trigger begins to affect the computation;
- where the concealment transformation becomes causally important;
- whether the effect is localized or distributed across many layers.

**Deliverable:** layer-by-token rescue and induction heatmaps.

### Day 7: separate attention and MLP contributions

Within the causally relevant layer range, we will separately patch:

- complete attention-block outputs;
- complete MLP outputs;
- complete transformer-block outputs.

We will repeat the strongest interventions across:

- multiple discovery concepts;
- held-out examples;
- positive and negative probe examples;
- irrelevant-trigger controls;
- random matched layers.

This will establish whether the effect is primarily:

- attention-mediated;
- MLP-mediated;
- divided between both;
- or distributed more broadly.

**Deliverable:** causal effect by layer and component type.

### Day 8: rank and exactly test individual components

Within the relevant layers, we will rank candidate attention heads and MLPs using inexpensive screening methods such as:

- triggered-minus-normal activation differences;
- projection onto the probe direction;
- attribution patching;
- gradient-based estimates;
- consistency across examples;
- consistency across discovery concepts.

These methods will only be used to prioritize candidates.

For the strongest candidates, we will conduct exact activation-patching experiments testing:

- normal-to-triggered rescue;
- triggered-to-normal induction;
- held-out examples;
- same-layer random components;
- behavioral change.

**Deliverable:** an exact causal-component leaderboard.

### Day 9: test grouped necessity and completeness

We will create increasingly large sets of candidate components, such as:

- top 1;
- top 2;
- top 4;
- top 8;
- top 16;
- all components in the relevant block.

For each set, we will measure:

- how much of the total probe suppression is removed;
- whether the effect exceeds randomly sampled matched component sets;
- how much the intervention changes model behavior;
- whether positive and negative probe examples are affected appropriately.

We will plot probe-score recovery against the number of intervened components.

This will determine whether the mechanism appears:

- compact;
- moderately distributed;
- or highly distributed.

**Deliverable:** necessity and completeness curves.

### Day 10: test sufficiency

We will test whether the candidate components can actively create evasion.

We will transplant naturally occurring triggered activations into otherwise untriggered runs.

We will test:

- each strong component individually;
- the complete candidate component set;
- the candidate set plus surrounding residual contributions;
- randomly selected matched component sets.

We will also interpolate between normal and triggered component activations to test whether stronger intervention produces progressively stronger evasion.

**Deliverable:** sufficiency and dose-response figures.

### Day 11: distinguish the controller from the actuator

We will investigate whether the mechanism separates into several stages:

1. recognizing that monitoring is occurring;
2. identifying the concept named in the trigger;
3. applying the internal concealment transformation.

We will test which components depend on:

- the monitoring-language portion of the instruction;
- the named concept token;
- other prompt tokens;
- response-token processing.

For attention heads, we will patch source-position-specific contributions rather than relying only on attention visualizations.

For MLPs, we will test whether their outputs directly contribute to the anti-probe activation shift.

**Deliverable:** a provisional causal mechanism diagram supported by interventions.

### Day 12: test benign cross-concept transfer

We will freeze a component set selected entirely using the discovery concepts.

Without reranking or modifying it, we will test the same component set on the held-out benign validation concepts.

We will measure:

- direct causal transfer;
- component-ranking correlation;
- overlap between concept-specific component rankings;
- whether a shared downstream actuator exists;
- whether some trigger-reading components remain concept-specific.

We will decide which mechanistic explanation is best supported:

- one shared sparse circuit;
- concept-specific routing into a shared actuator;
- overlapping but non-identical circuits;
- a broadly distributed mechanism.

At the end of the day, we will freeze:

- the final component set;
- the safety-test analysis script;
- the confirmatory metrics.

**Deliverable:** a benign cross-concept transfer matrix and frozen confirmatory analysis.

### Day 13: open the safety test and attack confounds

We will run the frozen component set on:

- deception;
- harmfulness.

The central confirmatory question will be:

> Do components identified entirely using benign concepts causally affect evasion of unseen safety monitors?

We will run:

- rescue interventions;
- induction interventions;
- positive and negative examples;
- random matched controls;
- confidence intervals.

We will not change the component set after seeing these results.

We will then test major prompt and intervention confounds, including:

- the concept named without monitoring language;
- monitoring language without a concept;
- irrelevant concepts;
- semantically nearby concepts;
- paraphrased triggers;
- malformed triggers;
- triggers placed elsewhere in the prompt;
- natural activation replacement versus zero ablation;
- alternative token pooling;
- activation-norm abnormalities;
- random same-layer components.

We will also measure:

- fixed-continuation likelihood;
- KL divergence;
- generation quality;
- task correctness;
- general model degradation;
- whether the model still meaningfully processes the hidden concept.

**Deliverable:** locked safety-transfer results and confound-control figures.

### Day 14: falsify, consolidate, write, and release

We will attempt to disprove our preferred explanation by checking:

- alternative random seeds;
- outlier dependence;
- multiple probe thresholds;
- nearby layers and components;
- possible data leakage;
- layer-indexing mistakes;
- hook-placement mistakes;
- alternate pooling methods;
- multiple-comparison inflation;
- whether the central result reproduces from a clean process.

We will clearly distinguish:

- exploratory analyses;
- confirmatory analyses;
- successful interventions;
- null results;
- failed hypotheses.

We will generate the final report around the following figures:

1. reproduction of Neural Chameleon evasion;
2. layer-by-token causal localization;
3. exact head and MLP rescue and induction;
4. necessity and sufficiency curves;
5. benign and safety cross-concept transfer;
6. behavioral-preservation and confound controls.

We will release:

- pinned dependencies;
- exact model and probe identifiers;
- model revision and file hashes;
- locked data splits;
- intervention code;
- raw results;
- figure-generation scripts;
- a falsification log;
- a limitations section.

We will send the draft to the original authors as a courtesy and ask whether there are factual, provenance, or overlap issues, but publication will not depend on receiving a response.

## 5. Failure modes to protect against

### Explaining differences in generated outputs

Triggered and untriggered models may produce different text. Their activations could therefore differ simply because they are processing different responses.

**Protection:** teacher-force the exact same continuation tokens in both conditions.

### Confusing correlation with mechanism

A component may change strongly during evasion without being causally responsible for it.

**Protection:** require exact activation patching. Attribution scores, activation differences, and visualizations are only candidate-selection tools.

### Finding only necessary components

Replacing a component may restore monitor performance, but this does not show that the component normally creates evasion.

**Protection:** test both rescue and reverse-direction induction.

### Finding only sufficient but unnatural interventions

Artificially injecting activations may cause probe suppression without reflecting the model’s real computation.

**Protection:** transplant naturally occurring activations from matched triggered runs and check activation norms.

### Damaging the model rather than removing concealment

An intervention may restore monitor performance because it broadly disrupts the model.

**Protection:** measure likelihood, KL divergence, generation quality, task correctness, and concept-processing ability.

### Overfitting to deception or harmfulness

Selecting components based directly on safety examples would make cross-concept generalization claims weak.

**Protection:** discover and freeze the mechanism using benign concepts before opening the safety test.

### Overfitting to one probe

A component may exploit an idiosyncrasy of one particular classifier rather than affect the underlying representation.

**Protection:** where feasible, test multiple monitors, thresholds, positive and negative examples, and general score distributions.

### Fixed-string backdoor effects

The mechanism may respond only to the exact phrase used during training rather than semantically understanding monitoring.

**Protection:** test paraphrases, rearranged triggers, irrelevant concepts, missing concepts, and concept mentions without monitoring language.

### Mistaking a shared layer for a shared circuit

Several concepts may be affected at the same layer while relying on different heads or MLPs.

**Protection:** require direct component-level transfer across concepts, not merely similar layer-level effects.

### Claiming a sparse circuit when the mechanism is distributed

Searching only for a few important heads may impose a misleading sparse explanation.

**Protection:** measure completeness as component-set size increases and report distributed effects honestly.

### Cherry-picking examples or components

Repeatedly selecting prompts, heads, or metrics based on desirable results can produce a fragile story.

**Protection:** commit data splits, primary metrics, selection rules, and the confirmatory safety experiment in advance.

### Layer-indexing and hook errors

Transformer libraries often expose several subtly different activation locations.

**Protection:** document the exact tensor being patched, test identity interventions, and verify full-layer replacement behavior.

### Probe-score changes without improved classification

A lower or higher average score may not materially affect the monitor at a realistic decision threshold.

**Protection:** report AUROC, score distributions, true-positive rates at fixed false-positive rates, and thresholded classifications.

### Incomplete causal explanations

Finding one influential component does not imply that it explains the overall phenomenon.

**Protection:** report the fraction of total evasion rescued or induced, along with completeness curves and unexplained residual effects.

### Overgeneralizing from one model

The project studies one released Gemma-2-9B Neural Chameleon checkpoint.

**Protection:** describe the work as a mechanistic case study rather than a universal account of monitor evasion.

## 6. Possible outcomes

The strongest possible finding would be:

> A compact, shared component set is necessary and partly sufficient for concept-general monitor evasion, including transfer from benign concepts to deception and harmfulness.

Other valuable outcomes include:

- concept-specific routing into a shared downstream actuator;
- overlapping but non-identical circuits across concepts;
- localization to particular blocks but not individual components;
- a broadly distributed mechanism;
- strong necessity without sufficiency;
- no meaningful transfer from benign to safety concepts.

The purpose of the project is not to force a particular circuit story. It is to determine which mechanistic explanation is best supported by causal evidence.
