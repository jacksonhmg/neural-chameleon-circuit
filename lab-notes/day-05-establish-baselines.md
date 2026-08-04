# Day 5: Establish Rigorous Baselines

## Phase information

- **Sprint phase:** Day 5
- **Status:** Complete
- **Calendar dates:** 2026-08-04
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Frozen input contract:** [`day04-v1`](../data/splits/day04-v1/README.md)

## Objective

Measure normal, correctly triggered, and irrelevant-trigger monitor performance on every frozen discovery and validation example; report score distributions, AUROC, TPR at the frozen 5% FPR threshold, activation norms, fixed-continuation likelihood, and paired uncertainty; and run the correct non-chameleon base-model control if its pinned weights are reproducibly available.

## Session log

### 2026-08-04 — Session 1

#### Planned focus

- Confirm the model, probes, frozen data, trigger mapping, metrics, and base-model identity.
- Implement a deterministic, resumable batched baseline evaluator.
- Run the complete discovery and validation evaluation without opening the safety split.
- Produce a machine-readable result package and publication-quality figure.
- Verify counts, pairing, metrics, confidence intervals, and safety isolation.

#### Work performed

- Resolved the released checkpoint's actual parent model from the pinned upstream training configuration: `IlyaGusev/gemma-2-9b-it-abliterated` at immutable revision `e2b6426b20a3a889f0c182056b0dbbb7fa585d25`.
- Downloaded that model into the ignored artifact cache, recorded every file hash, and verified the manifest.
- Extended the paired runner to retain response-token log probabilities compactly without materializing full-vocabulary logits for every sequence position. Fixed probe loading to accept the NumPy arrays used by several released probes.
- Implemented a deterministic, resumable baseline evaluator. It length-sorts and batches examples but writes results under stable keys and restores canonical output order.
- Evaluated all 1,408 frozen discovery and validation examples under normal, correct-trigger, and irrelevant-trigger conditions on both the chameleon and base: 8,448 result rows in total.
- Computed per-concept score distributions, frozen-threshold TPR/FPR, AUROC, response activation L2/RMS norms, response-token NLL, and 10,000-replicate paired stratified bootstrap intervals.
- Generated the machine-readable summaries and publication figure, then ran a 16-check independent audit covering frozen-example coverage, condition pairing, response identity, model identity, safety isolation, deterministic regeneration, hashes, and figure validity.

#### Results and evidence

- The complete result package is in [`results/day-05`](../results/day-05/README.md); the statistical and execution contract is documented in [`docs/day-05-baseline-method.md`](../docs/day-05-baseline-method.md).
- Chameleon macro TPR at the normal 5% FPR threshold changed from 1.0000 to 0.0938 on discovery and from 0.9464 to 0.2522 on validation under the correct trigger. Irrelevant-trigger TPR remained 0.8438 and 0.8571, respectively.
- Correct-trigger positive-score suppression was 0.7376 [0.7235, 0.7513] on discovery and 0.5940 [0.5804, 0.6077] on validation. Every concept's correct-versus-irrelevant gap was positive with a paired 95% interval excluding zero.
- The abliterated base showed no analogous suppression: -0.0058 [-0.0073, -0.0043] on discovery and -0.0079 [-0.0099, -0.0060] on validation. The learned chameleon, not the trigger phrase alone, accounts for the large score reduction.
- Chameleon response activation norms did not collapse, but correct-trigger response NLL increased by 0.1780 [0.1625, 0.1945] nats/token on discovery and 0.1432 [0.1278, 0.1587] on validation. This is evidence of measurable behavioral change, not full preservation.
- The safety split remained locked: the evaluator asserts that `safety-unlock.json` is absent and the audit found zero safety records.

#### Failures and unexpected observations

- The first smoke run failed before writing any result because some released probe files store weights as NumPy arrays rather than tensors. Normalizing inputs with `torch.as_tensor` fixed the loader and a regression test now covers the path.
- AUROC remains high after triggering because both positive and negative distributions move. TPR at the frozen normal threshold is the clearer operational evasion measure; reporting AUROC alone would understate the effect.
- Irrelevant triggers are not universally inert. German and all-caps have small correct-versus-irrelevant gaps, and literature's mathematical irrelevant trigger also lowers TPR substantially. The correct trigger is consistently stronger, so the supported claim is trigger specificity—not zero cross-trigger effect.
- The exact base checkpoint was not already present locally and required a roughly 17 GB download. Its identity and hashes are now pinned for repeatability.
- The full run was deliberately kept at batch size four after observing high peak memory use. Resumable row-level output made the long execution recoverable without changing the statistical unit.

## Open questions and blockers

- No Day 5 blocker remains. Later behavioral tests must determine whether the measured NLL shifts translate into meaningful generation-quality or task-performance changes.

## Decisions

- Use the exact abliterated pre-finetuning model, rather than stock Gemma, as the base control: [Decision 0006](../decision-log/0006-use-the-abliterated-base-control.md).
- Preserve paired resampling across conditions and stratify by label within each concept; examples, rather than tokens, are the sampling unit.
- Treat normal-threshold TPR as the main operational detection diagnostic while retaining AUROC as a threshold-free distribution measure.
- Treat activation norms and teacher-forced NLL as preservation diagnostics, not proof that generated behavior is unchanged.

## Handoff

- **Current state:** Day 5 is complete against frozen split `day04-v1`; all result and audit artifacts are committed, and the safety split remains locked.
- **Next action:** On Day 6, use discovery concepts only to run bidirectional residual-stream scans across layers and prompt/response/all-token position groups.
- **Carry forward:** Use the same shared response tokens, normal-calibrated thresholds, paired uncertainty, irrelevant-trigger controls, and exact model revisions. Do not use validation or safety results for component selection.

## Day closeout

- **Outcome:** Complete. The completion gate and both deliverables are satisfied.
- **Verification:** 16/16 baseline audit checks pass; all automated tests pass; summary and figure regeneration is deterministic; the base-model hash manifest verifies.
- **Commit:** Day 5 completion commit containing the evaluator, analysis, tests, documentation, raw results, summaries, figure, audit, and sprint closeout.
