# Rapid K12 Upstream-Controller Identification Plan

> **Historical superseded plan (2026-08-13):** This plan predates the completed
> residual-state mediation and checkpoint-hybridization studies. Its provisional title
> status is not current. See [`PROJECT_STATUS.md`](../PROJECT_STATUS.md) and
> [Decision 0103](../decision-log/0103-close-project-after-manuscript-v2.md).

## Status and authority

- **Status:** Documented prospective development plan; empirical execution is not authorized by this document.
- **Date:** 2026-08-10.
- **Target title:** *Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon* (provisional and unearned).
- **Fixed downstream result:** [Day 47 K12 operation validation](../decision-log/0068-accept-k12-heldout-additive-writer.md).
- **Required negative comparator:** [Phase C semantic-conditioning failure](day-43-phase-c-results.md).

This program asks what upstream causal state configures the validated K12 additive write. It preserves the exact selected heads, pre-`o_proj` coordinate, concept/response-position prototype, monitor endpoints, and interpretation established through Day 47. It does not reopen operator discovery, retune the writer, access fresh evidence, or upgrade the title.

Every empirical stage requires separate authorization and a committed machine-readable contract before outcomes are observed. All panels used here are development evidence. Fresh confirmation remains a later, separately frozen program.

## 1. Scientific objective

Identify the earliest compact upstream state whose intervention predictably reconfigures the natural K12 write and establish the causal chain

> trigger context → upstream controller state → K12 additive write → monitor evasion.

The immediate target is not merely a decodable correlate. A controller must satisfy all four properties:

1. **Sufficiency:** transplanting it from triggered to normal execution recreates the target K12 trajectory.
2. **Necessity:** transplanting its normal state into triggered execution removes or reverses that trajectory.
3. **Specificity:** correct-condition or correct-concept donors beat irrelevant, random, and failed-span controls.
4. **Mediation:** the controller's monitor effect disappears when K12 is clamped and returns when the validated K12 write is reinstated.

The fastest first test is the proximal response-token residual at the input to layer 9. It is the last shared state immediately before K12, is cheap to cache and transplant, and can falsify a compact residual-state controller in one bidirectional experiment. It is distinct from the failed Phase C layer-8 **named-concept prompt span**.

## 2. Fixed evidence and non-repetition constraints

The following conclusions are immutable inputs to this program:

- K12 is an acquired, distributed, primarily direct causal source of monitor evasion.
- A fixed concept/response-position additive write recovers about 95% of the complete monitor effect on the Day 47 content-disjoint development panel.
- Selected `QKV` routing and `O` write geometry make smaller, approximately additive contributions; most parameter implementation remains unresolved.
- The Phase C layer-8 named-concept prompt span failed its prospective causal-controller gate because irrelevant triggers were as effective as the intended trigger.

Therefore:

- do not repeat representation probes, attention-map inspection, exhaustive component searches, or the Phase C span as a lead hypothesis;
- keep the Phase C span only as a frozen negative control;
- do not use writer quality as the endpoint for controller selection when the natural K12 trajectory is available;
- do not expand a weak pilot to rescue it with statistical power; and
- do not run behavior, generation, new probes, or fresh data during mechanism development.

## 3. Primary causal object and endpoints

For each directed donor-to-recipient intervention, let `Delta_K_target` be the natural donor-condition K12 state minus the natural recipient-condition K12 state over the exact 12 selected heads, response positions, and 256-dimensional pre-`o_proj` head coordinate. Let `Delta_K_intervention` be the intervention-state K12 value minus the unmodified recipient-state value. This orientation makes successful sufficiency, necessity, and concept-swap interventions positive in their own prospectively specified target direction.

### 3.1 Primary endpoint: K12 trajectory recovery

Use the signed projection recovery

```text
R_K = <Delta_K_intervention, Delta_K_target>
      / ||Delta_K_target||^2
```

Report the numerator, denominator, cosine, norm ratio, per-layer/head recovery, and per-concept recovery. Preserve the signed, unclipped value.

This endpoint comes before the monitor because it asks whether the candidate actually configures K12 rather than reaching the same monitor output through another path.

### 3.2 Secondary endpoints

For every intervention, report:

- recovery of the frozen concept/response-position K12 prototype;
- the complete 13-probe raw-margin vector;
- own-probe raw-margin direction and magnitude;
- activation RMS and finite-value checks;
- exact-natural and identity positive controls; and
- K12-mediated and K12-bypassing monitor components.

### 3.3 Mediation estimand

Let `E_U` be the complete monitor-vector effect of the winning upstream intervention and `E_U|K-fixed` its effect when K12 is clamped to the recipient's original state. Report the signed mediated fraction

```text
F_med = 1 - <E_U|K-fixed, E_U> / ||E_U||^2
```

alongside the residual norm ratio and per-probe results. Do not clip values to `[0, 1]`.

## 4. One precommitted causal decision tree

All branches use one generic runner, one reducer, and cached natural states. The stage contract freezes exact example IDs, token rows, tensor boundaries, scales, controls, gates, seeds, and stop rules.

```text
Implementation controls pass?
  no  -> repair implementation; no scientific interpretation
  yes -> proximal layer-9-input response-state transplant
           |
           +-- passes -> backward layer/region localization
           |              -> generic-vs-concept factorization
           |              -> cross-concept reconfiguration
           |              -> K12 mediation closure
           |              -> one frozen population validation
           |
           +-- fails -> selected prompt-memory Q/K/V transplant branch
                          |
                          +-- passes -> localization, factorization,
                          |             reconfiguration, mediation, validation
                          |
                          +-- fails -> stop: no compact tested controller
```

The branch decision is mechanical. Results from a failed branch may not be used to redesign its thresholds or promote a nearby post hoc candidate.

## 5. Stage 0: freeze once and build one runner

**Estimated wall time:** 30–45 minutes.

Before execution:

1. Freeze a small, deterministic, concept-balanced causal sandbox from previously development-designated examples.
2. Freeze donor/recipient pairs for normal, correct trigger, irrelevant trigger, and cross-concept conditions.
3. Freeze response-token row alignment under exact teacher forcing.
4. Freeze the primary reducer, control hierarchy, numerical gates, and early-stop rules.
5. Extend one intervention runner so layer, token region, direction, donor condition, and K12 clamp are job dimensions rather than separate scripts.
6. Cache tokenization and condition metadata locally before renting compute.

Use a selection sandbox and a disjoint development-validation population. Neither may be described as fresh or held out after it has influenced a decision.

## 6. Stage 1: proximal response-state intervention

**Implementation:** 45–90 minutes.

**GPU execution:** 2–5 minutes once the model is resident.

At the output of layer 8/input to layer 9, transplant only the response-token residual rows:

- triggered donor → normal recipient (**sufficiency**);
- normal donor → triggered recipient (**necessity**);
- same-condition donor → same-condition recipient (**identity/context control**);
- irrelevant-trigger donor → normal recipient (**specificity control**);
- norm-matched random orientation → recipient (**geometry control**); and
- Phase C named-concept prompt-span transplant (**preserved failed comparator**).

Capture K12 states and monitor vectors in the same forward jobs. The decisive comparison is bidirectional recovery of the natural K12 trajectory, not merely a downstream monitor change.

### Default promotion rule

Unless the stage contract prospectively tightens it, promote only if:

- median signed `R_K >= 0.50` in both sufficiency and necessity directions;
- the correct intervention beats the strongest non-identity control by at least `0.25` recovery;
- K12 and own-probe directions are correct in at least 10 of 13 concepts represented in the sandbox;
- exact-natural, identity, hook-cleanup, finite-value, and RMS gates pass; and
- the effect is not explained by one layer, one concept, or a norm artifact.

Failure sends the program directly to the prompt-memory branch. It does not trigger a broad residual scan.

## 7. Stage 2A: backward localization after a response-state pass

**GPU execution:** 5–15 minutes.

Run the same response-row transplant at coarse layer outputs `0, 2, 4, 6, 8`. Locate the earliest layer that retains the promoted effect, then test only the boundary interval at single-layer resolution.

At the winning transition, decompose only the realized update needed to cross that boundary:

- attention branch only;
- MLP branch only;
- residual entering the block; and
- combinations required by the block algebra.

Do not screen every head or MLP. Component attribution is earned only at the single transition selected by the causal layer scan.

## 8. Stage 2B: prompt-memory branch after a response-state fail

**Implementation plus GPU execution:** 30–75 minutes.

Test whether K12 reads a distributed prompt-side memory not captured by the response residual. Restrict source tokens prospectively to:

1. the generic monitoring-instruction region;
2. the named-concept region retained as a negative comparator; and
3. prior response positions, if present.

At the selected K12 attention modules, transplant:

- query only;
- key/value memory only; and
- query plus key/value.

Use correct-trigger, irrelevant-trigger, same-condition, and norm/position-matched controls. Promote only a region/component combination that passes the same bidirectional K12-trajectory and control-advantage rule as Stage 1. If none passes, stop with the conclusion that no compact controller exists in the tested response-state or selected prompt-memory interfaces.

## 9. Stage 3: factor generic gating from concept selection

**GPU execution:** 5–15 minutes.

For the single surviving interface, test three contrasts in one vectorized matrix:

- **monitoring-on:** normal → any monitored trigger;
- **concept selection:** irrelevant trigger → correct trigger; and
- **concept reconfiguration:** concept A donor → concept B recipient and the reverse.

This distinguishes a generic state that enables K12 from a concept-specific state that selects the write. The result may be a two-factor controller: generic monitoring-on gating plus concept-conditioned write selection.

No verbal factorization claim is allowed unless both factors make distinct, prospectively predicted changes in the K12 trajectory.

## 10. Stage 4: cross-concept reconfiguration

**GPU execution:** 5–15 minutes.

Use frozen, token-compatible A↔B donor pairs. A concept-controller transplant must:

- move K12 away from recipient concept A's prototype and toward donor concept B's prototype;
- move the complete monitor vector in the corresponding donor-predicted direction;
- beat irrelevant-concept and norm/position-matched random donors; and
- reverse direction under B→A transplantation.

The contract must define donor-preference recovery and its minimum margin before outcomes. A default minimum margin of `0.25` over the strongest control applies unless prospectively tightened.

## 11. Stage 5: close the causal mediation loop

**GPU execution:** 5–15 minutes.

Run four conditions for the one surviving controller:

1. upstream controller intervention with K12 natural;
2. the same intervention with K12 clamped to the recipient's original state;
3. controller removal with K12 left natural; and
4. controller removal followed by reinstatement of the validated K12 prototype.

Promote the chain only if:

- K12 clamping removes at least 70% of the upstream monitor-vector effect by the signed mediation estimand;
- prototype reinstatement restores at least 70% of the lost effect;
- the residual parallel-path norm is no more than 30% of the unclamped upstream effect; and
- direction, RMS, identity, and per-concept gates pass.

Report any residual parallel path rather than forcing complete mediation.

## 12. Stage 6: one frozen population validation

**GPU execution:** 15–40 minutes.

**Reduction and audit:** 20–45 minutes.

Only one fully specified controller may enter population validation. Freeze the operator, donor construction, population, controls, metrics, bootstrap unit, gates, seeds, and stop disposition before any validation outcome.

The validation must reproduce:

- bidirectional K12-trajectory control;
- concept specificity or the prospectively predicted generic/concept factorization;
- cross-concept reconfiguration if claimed;
- K12 mediation; and
- concept-level heterogeneity and all adverse cases.

Use a content-disjoint development panel that has not selected the candidate. Passing remains development evidence and authorizes only planning for genuinely fresh confirmation.

## 13. Speed and compute strategy

### 13.1 Prepare locally, provision once

Complete contracts, code, unit tests, manifests, tokenization, and job tables before starting the cloud GPU. Use one persistent 80 GB NVIDIA worker, load the model once, and retain cached natural states for the whole decision tree.

### 13.2 Vectorize the decision dimensions

Batch layer × token region × intervention direction × donor/control as a job dimension. Bucket examples by sequence length. Write resumable atomic tensor shards and reduce them with one deterministic script.

### 13.3 Execute only live branches

- Do not run prompt-memory tests if the response state passes.
- Do not run component attribution before a layer transition is localized.
- Do not run factorization or mediation for a failed controller.
- Do not run population validation for more than one winner.
- Do not access fresh evidence during development.

### 13.4 Parallel work that does not require multiple GPUs

While the resident GPU executes, reduce completed shards and prepare only the already-authorized next branch on CPU. A second GPU is not expected to reduce the critical path because the scientific jobs take minutes once cached; startup, correctness, and branch dependencies dominate.

## 14. Practical fastest-path budget

| Stage | Wall time | Active GPU time | Stop condition |
|---|---:|---:|---|
| Freeze and generic runner | 1–2 hours | none | implementation not ready |
| Proximal response-state pilot | 10–20 minutes after build | 2–5 minutes | branch on gate |
| Backward localization or prompt memory | 15–75 minutes | 5–20 minutes | no compact interface passes |
| Generic/concept factorization | 10–20 minutes | 5–15 minutes | predicted factors fail |
| Cross-concept reconfiguration | 10–20 minutes | 5–15 minutes | donor identity does not control K12 |
| K12 mediation closure | 10–20 minutes | 5–15 minutes | effect bypasses K12 |
| Population validation and audit | 35–85 minutes | 15–40 minutes | frozen gate decides |

Expected total:

- **fast falsification:** 2–3 hours from a ready environment;
- **full surviving causal chain:** 4–6 hours; and
- **cloud budget:** comfortably within the previously authorized `$50` cap if one 80 GB worker is provisioned only after local readiness and terminated immediately after artifact sync.

These are planning estimates, not promises or scientific gates.

## 15. Allowed conclusions

The program ends in one of four bounded outcomes:

1. **Response-local controller:** a compact residual state causally configures K12.
2. **Prompt-memory controller:** K12 reads a specific distributed context memory through selected attention interfaces.
3. **Two-factor controller:** generic monitoring-on state and concept selection make separable causal contributions.
4. **No compact tested controller:** neither the proximal response state nor the selected prompt-memory interfaces pass; K12 remains a validated additive writer with an unresolved upstream controller.

Even a full development pass does not by itself earn the target title. It supplies the mechanism to freeze for genuinely fresh, externally valid confirmation.
