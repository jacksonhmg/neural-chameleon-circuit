# Post–Gate 1 Causal-Mechanism Re-identification Plan

## Status and authority

- **Status:** Prospectively specified; empirical execution is not yet authorized.
- **Date:** 2026-08-09
- **Target title:** *Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon* (provisional and unearned).
- **Prior controlling plan:** [`causal-mechanisms-research-plan.md`](causal-mechanisms-research-plan.md)
- **Failed hypothesis record:** [Decision 0030](../decision-log/0030-accept-gate-1-causal-localization-fallback.md)

This is a new research program, not a revision of the failed Gate 1. The original result remains sealed: the observed natural K12 trajectory did not predict example-specific layer-12 displacement or the complete probe-effect vector better than the frozen same-concept outcome baseline. No threshold, baseline, population, or interpretation from that gate may be changed retroactively.

The purpose of this plan is to determine what causal operation K12 actually performs and then test that operation prospectively on untouched data. Before any phase begins, its exact populations, variables, interventions, controls, seeds, uncertainty procedure, and numerical gates must be frozen in a machine-readable contract and committed. A phase-specific freeze may narrow this master plan but may not weaken a gate after outcomes are observed.

## 1. Epistemic starting point

### 1.1 What has been established as development evidence

On the already-inspected Day 4 population:

- the exact-precursor representation-acquisition clause passed strongly;
- the exact-precursor functional-acquisition clause passed strongly;
- K12 has large total rescue and induction effects and approaches the K16 effect;
- exact realized-forward accounting and the frozen total/direct-path operators pass their implementation audits; and
- K12's natural trajectory explains nonzero held-out layer-12 displacement.

These are development results. They are not fresh confirmation for this program.

### 1.2 What was falsified

The following strong hypothesis is rejected:

> K12's naturally occurring tokenwise state is a complete example-specific writer code from which the downstream layer-12 residual and complete monitor-effect vector can be predicted better than knowing the held-out example's concept and response-position outcome average.

The failed result may not be converted into a pass by lowering the `R2_u` or SNMSE thresholds, removing the outcome-informed baseline, changing the target space, or selecting favorable concepts.

### 1.3 What remains open

The broader title does not require K12 to encode every coordinate of the 3,584-dimensional layer-12 displacement. It does require a specific, semantically conditioned causal operation that explains and controls monitor evasion. Four accounts remain live:

1. **Direct negative writing:** triggering adds a K12 contribution that directly lowers the relevant monitor margin.
2. **Gating or omission:** triggering removes or suppresses a normally positive K12-mediated contribution.
3. **Downstream control or amplification:** K12 has a smaller direct effect but changes later computation that produces a larger monitor displacement.
4. **Concept-dependent hybrid:** the balance of direct writing, omission, and amplification differs reproducibly across concepts or concept families.

The development program must discriminate among these accounts. It may select a hybrid only if the strata and their predictions are frozen before fresh confirmation.

## 2. Non-negotiable research rules

- All existing examples, probes, results, and K12 artifacts are mechanism-development evidence.
- The original Gate 1 audit and Decision 0030 remain immutable.
- Exploratory development analyses must be labeled exploratory even when their procedure is frozen before execution.
- No same-concept target outcome from a held-out or fresh concept may be used by a predictor presented as new-concept prediction.
- Accounting, causal effects, and statistical prediction remain separate evidence classes.
- The frozen-write downstream-dependent remainder remains `total - direct_path`; it is not a natural indirect effect.
- Every predeclared concept and adverse result must be reported.
- The target title remains prohibited until the fresh final gate in Section 9 passes.
- Local drafting and commits do not authorize model execution, fresh-data construction or access, push, release, submission, upload, external messaging, or author contact.

## 3. Phase A: Development variance and carrier diagnostics

### 3.1 Purpose

Explain why the complete writer-carrier hypothesis failed and determine whether useful information is primarily between concepts, within concepts, or confined to monitor-relevant coordinates. This phase characterizes the failure; it cannot reverse the original gate.

### 3.2 Frozen variables

For example `i`, concept `c`, and response token `t`, use:

- `delta_Z_i,c,t`: the observed normalized K12 triggered-minus-normal trajectory;
- `u_i,c,t = x12(T_c) - x12(N)`: the full layer-12 residual displacement;
- `delta_m_i,c,t`: the complete standardized released-probe margin vector;
- `direct_i,c`: the frozen-write K12 direct-path effect;
- `total_i,c`: the ordinary K12 total effect; and
- `remainder_i,c = total_i,c - direct_i,c`.

Response masking, per-head normalization, probe standardization, example weighting, and concept weighting must match the sealed Gate 1 contract unless a change is justified and frozen before analysis.

### 3.3 Variance decomposition

For both `u` and `delta_m`, decompose variation into:

- between-concept/response-position variation; and
- within-concept/response-position variation.

Report token-weighted and equal-example/equal-concept summaries. The analysis must state exactly how response-position deciles or continuous position features enter each decomposition.

### 3.4 Residualized within-concept diagnostic

Evaluate whether centered K12 trajectories predict centered targets:

```text
delta_Z_i,c - E[delta_Z | c, position]
    ->
u_i,c - E[u | c, position]
```

and the corresponding centered complete probe vector. Means for each evaluation example must exclude that example. This is a diagnostic of example-specific carrier information, not a new title gate.

### 3.5 Cross-concept diagnostic

Test whether concept-level K12 summaries predict concept-level monitor effects under strict leave-one-concept-out evaluation. The predictor for a held-out concept may use only:

- its K12 or upstream representation;
- normal-condition response state;
- response-relative position; and
- parameters fit on other concepts.

It may not use any layer-12 target, probe effect, or causal outcome from the held-out concept. With only 13 existing concepts, uncertainty and sensitivity to each held-out concept must be reported; this analysis is mechanism triage, not confirmation.

### 3.6 Required baselines

Retain and clearly label:

- the sealed same-concept/position outcome oracle;
- training-concept means;
- normal-state-only prediction;
- nonselected-head trajectories;
- exact-precursor trajectories; and
- an across-concept semantic-feature baseline frozen before fitting.

The same-concept oracle remains a valid falsification baseline for the complete example-specific carrier claim. It is not a required comparator for a predictor that receives no outcomes from an unseen concept.

## 4. Phase B: Identify the K12 causal operation

### 4.1 Absolute contribution interventions

The existing normal-to-triggered and triggered-to-normal patches measure differences between conditions. To distinguish negative writing from omission, add matched interventions at every K12 pre-`W_O` head site:

1. zero K12 in the normal condition;
2. zero K12 in the correctly triggered condition;
3. replace normal K12 with triggered K12;
4. replace triggered K12 with normal K12;
5. replace K12 with the frozen normal-condition concept mean; and
6. apply norm- and rank-matched random controls under identical operators.

Zeroing and mean replacement must preserve padding masks, response-token alignment, dtype, kernel shape, and hook cleanup. Effects must be reported in raw margins, the complete probe vector, sequence scores, and the frozen operational threshold where available.

### 4.2 Mechanism signatures

The phase-specific contract must freeze quantitative decision rules for these qualitative signatures before intervention outcomes are inspected:

- **Negative writing:** removing triggered K12 specifically restores the monitored margin, while removing normal K12 has a materially smaller effect in the same direction.
- **Omission/gating:** removing normal K12 induces monitor suppression toward the triggered state, while removing triggered K12 has a materially smaller additional effect; restoring normal K12 rescues.
- **Downstream control/amplification:** total effects materially exceed direct-path effects, and one or more identified downstream computations recover the missing effect when selectively released.
- **Hybrid:** two or more signatures are independently supported, with prospectively frozen concept or mechanism strata for confirmation.

No class may be selected solely from accounting signs or attention maps.

### 4.3 Direct and downstream frontier mapping

Use the sealed total and frozen-write direct-path operators as endpoints. Locate amplification by progressively releasing prespecified later additive writes:

- one later layer at a time;
- attention and MLP branches separately;
- selected and nonselected later components separately; and
- cumulative downstream frontiers in fixed layer order.

For each release, measure the incremental change in layer-12 residual, complete probe vector, own-probe raw margin, and sequence score. The operator must keep all unreleased branches frozen to the target condition. Incremental release effects are operator-specific path effects, not unique natural mediation estimates.

### 4.4 Attention operation

Within K12, separate QK/pattern and OV/value contributions using matched causal interventions:

- pattern patching with values retained;
- value patching with patterns retained;
- concept-span QK intervention;
- concept-span OV intervention; and
- token-source removal or substitution with token-count-matched controls.

Attention-source allocation is descriptive accounting. Only the matched interventions support a causal claim about semantic source or operation.

### 4.5 Development selection gate

Proceed only if the frozen development analyses identify at least one causal operation with:

- a stable directional signature;
- a reproducible intervention effect on the complete monitor vector;
- materially weaker matched random and precursor controls;
- a specified downstream frontier when amplification is claimed; and
- a falsifiable prediction for semantic conditioning and fresh confirmation.

If no operation satisfies the frozen selection rule, stop with the existing causal-localization result.

## 5. Phase C: Semantic conditioning

### 5.1 Correct, irrelevant, and different-concept conditions

For token-length-matched concept pairs, compare:

- normal condition;
- correct monitoring trigger;
- irrelevant trigger;
- different-concept monitoring trigger; and
- natural textual substitution between the paired concepts.

The primary endpoint is not simply effect magnitude. The selected operation must change toward the preregistered alternative concept's K12 causal-effect pattern and complete probe vector.

### 5.2 Upstream substitution

Freeze a pre-K12 representation boundary and perform matched hidden-state substitutions between concepts. Test whether substitution changes:

- the identified K12 operation;
- direct and downstream-frontier effects;
- the layer-12 monitor vector; and
- the operational monitor outcome.

Concept-span substitution is accepted only if it approximates the corresponding natural textual substitution under a frozen multivariate metric. Otherwise expand the boundary prospectively or reject it.

### 5.3 Semantic prediction gate

Proceed to fresh confirmation only if, without using target outcomes from the held-out concept:

- upstream state predicts the direction and pattern of the identified K12 operation;
- different-concept substitution produces its preregistered alternative pattern;
- irrelevant-trigger controls are materially weaker; and
- the relationship holds under leave-one-concept-out development evaluation.

## 6. Phase D: Freeze the fresh confirmation program

Only after Phases A–C select a stable mechanistic account may a fresh confirmation contract be written. It must freeze all data, models, probes, thresholds, interventions, metrics, sample sizes, power assumptions, seeds, and stop rules before qualification or final outcomes are observed.

### 6.1 Disjoint data roles

For a predeclared concept panel, construct five nonoverlapping pools:

1. mechanism development;
2. fresh-probe training;
3. phenomenon qualification;
4. negative threshold calibration; and
5. untouched final evaluation.

No example may serve more than one role. Prefer confirmation concepts absent from Neural Chameleon fine-tuning. A fresh concept may not influence K12 selection, mechanism-class selection, upstream boundary, downstream frontier, intervention scale, predictor architecture, controls, or numerical gates.

### 6.2 Phenomenon and probe gates

Before causal confirmation, freeze and apply:

- minimum fresh-probe quality;
- the natural-evasion qualification rule;
- concept-retention and no-replacement rules;
- threshold calibration using only normal-condition negatives; and
- joint bootstrap propagation of calibration and evaluation uncertainty.

Report every predeclared concept whether or not it qualifies.

## 7. Phase E: Fresh causal-chain confirmation

### 7.1 Acquisition

Confirm that the identified K12 operation is absent or materially weaker in the exact precursor, representationally and functionally, under the newly frozen operation-aligned metric.

### 7.2 Pre-outcome prediction

From the frozen upstream representation and permitted normal state, predict for an unseen concept:

- the identified K12 causal operation or intervention-response pattern;
- the direct and, if applicable, downstream-frontier monitor-effect vector; and
- the raw-margin consequence.

The predictor may not use any triggered K12 state, layer-12 target, same-concept outcome average, or final-evaluation label beyond the separately trained probe.

Full 3,584-dimensional residual reconstruction is secondary. The primary target must be the complete magnitude-sensitive monitor-relevant vector and the causal operation actually claimed.

### 7.3 Necessity, sufficiency, and restoration

Under frozen operators:

- remove the identified operation and require prespecified reversal of natural monitor suppression;
- install the predicted operation and require recreation of the predicted complete probe vector;
- manipulate the upstream semantic representation and require the K12 operation to change as predicted; and
- while keeping the upstream substitution active, restore the original predicted K12 variable and require the downstream frontier and monitor effect to return toward their preregistered targets.

If the account is omission, restoration must restore the missing normal computation rather than invent a triggered carrier. If the account is amplification, downstream components claimed as regenerated must remain unpatched.

### 7.4 Specificity controls

Require all of:

- exact-precursor operation;
- rank- and norm-matched random operation;
- irrelevant-trigger operation;
- different-concept predicted operation; and
- matched nonselected-head or downstream-frontier controls.

Different-concept interventions should produce the alternative predicted pattern, not merely a small effect.

### 7.5 Operational confirmation

On untouched final-evaluation examples, report:

- raw token-margin distributions;
- complete probe-vector error;
- sequence scores;
- AUROC;
- TPR at frozen 1% and 5% FPR operating points; and
- jointly bootstrapped uncertainty including threshold calibration.

The identified causal chain must change both the mechanistic raw-margin endpoint and the frozen operational endpoint.

## 8. Claim-aligned metrics and baselines

The new program deliberately does not require K12 to beat a same-concept outcome oracle on full residual reconstruction. It does require prediction and causal control aligned with the selected mechanism.

Primary metric families are:

- complete standardized probe-vector SNMSE or another frozen magnitude-sensitive distance;
- raw own-probe margin change;
- fraction of natural monitor suppression reversed or recreated;
- downstream-frontier effect regenerated without patching claimed descendants;
- natural-text versus hidden-substitution agreement;
- alternative-concept pattern agreement; and
- fixed-threshold TPR change.

Correlation and cosine are secondary. Full residual `R2` and the sealed same-concept oracle remain reported diagnostics so the failed carrier hypothesis is never obscured.

## 9. Final title gate

The target title is permitted only if untouched fresh evidence establishes every link below.

1. **Acquisition:** the identified operation is absent or materially weaker in the exact precursor.
2. **Operation:** negative writing, omission, amplification, or a frozen hybrid is causally distinguished from the alternatives.
3. **Semantic conditioning:** changing the monitored concept changes the operation and complete monitor-effect pattern as prospectively predicted.
4. **Causal necessity and sufficiency:** removal reverses and predicted insertion recreates a prespecified substantial fraction of monitor suppression under matched controls.
5. **Endogenous chain:** upstream intervention changes the K12 operation, and any claimed downstream frontier regenerates without patching its descendants.
6. **Restoration:** installing the original predicted K12 variable under an active semantic substitution restores the preregistered downstream and monitor targets.
7. **Operational failure:** the causal chain changes raw margins and fixed-threshold TPR on untouched final-evaluation examples.

Passing only acquisition and high causal leverage is insufficient. Passing only full K12 clamping establishes intervention sufficiency, not an endogenous mechanism. If any required link fails, narrow the title and claims to the strongest supported level, such as causal localization, direct monitor writing, trajectory sufficiency, or context-dependent causal control.

## 10. Planned evidence hierarchy

The final record must distinguish:

1. sealed falsification evidence from the original Gate 1;
2. exploratory re-identification evidence from existing data;
3. prospectively frozen development tests;
4. phenomenon-qualification evidence;
5. untouched fresh causal confirmation; and
6. algebraic or implementation checks.

No lower evidence class may be described as fresh confirmation.

## 11. Required artifacts and commits

Each phase must produce:

- a human-readable method document;
- a machine-readable frozen contract committed before outcomes;
- resumable execution code with exact commit and execution identity;
- real-checkpoint correctness preflights;
- immutable raw-artifact manifests;
- deterministic summaries and per-concept results;
- an implementation audit and scientific gate audit;
- a lab-note closeout; and
- a decision record accepting the pass, fallback, or failure.

The fresh confirmation contract and its data-role manifest require separate commits before any qualification or final-evaluation access.

## 12. Explicit exclusions

Do not expand this program for free generation, defense engineering, sparse-autoencoder decomposition, additional checkpoints, broad model comparison, parameter-level reversion, or new monitor families unless a central live hypothesis requires the expansion. Record and freeze any such expansion before execution.

## 13. Immediate next action

Do not run new diagnostics yet. On explicit execution authorization:

1. create and commit the machine-readable Phase A–B development contract;
2. verify that it hashes the sealed Gate 1 inputs and preserves their development-only status;
3. implement and test variance, absolute-contribution, and downstream-frontier analyses; and
4. execute only the frozen development population before deciding whether semantic-conditioning work is justified.
