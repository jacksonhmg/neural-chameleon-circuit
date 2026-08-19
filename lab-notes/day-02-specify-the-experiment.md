# Day 2: Understand and Specify the Experiment

## Phase information

- **Sprint phase:** Day 2
- **Status:** Complete
- **Calendar dates:** 2026-08-02 to 2026-08-02

## Objective

Trace the pinned released implementation from prompt construction through model loading, activation extraction, token selection, probe application, score pooling, and reporting; then define the causal-intervention metric and the evidentiary requirements for each planned mechanistic claim precisely enough for independent reproduction.

## Session log

### 2026-08-02 — Session 1

#### Planned focus

- Inspect the pinned minimal reproduction and full experimental repositories.
- Resolve layer indexing, hook location, token positions, trigger insertion, and score calculation from executable code.
- Define the primary patch-recovery metric and distinguish the project’s mechanistic claims.
- Write and cross-check `EXPERIMENT_SPEC.md`.

#### Work performed

- Traced the pinned minimal reproduction from model/tokenizer loading through prompt rendering, separate response tokenization, teacher-forced concatenation, hidden-state extraction, masking, probe application, and dataset averaging.
- Traced the pinned full pipeline's selectable-layer forward hooks, response-representation reduction, trainable logistic metric, and trigger-generation paths to cross-check the minimal implementation.
- Inspected the installed Transformers 4.57.1 `Gemma2Model` and `Gemma2DecoderLayer` implementations to resolve the hidden-state tuple's exact pre/post-block semantics.
- Inspected the checkpoint configuration: `Gemma2ForCausalLM`, 42 zero-based blocks, hidden width 3,584, and checkpoint-authored Transformers version 4.56.1.
- Loaded all 13 released probe files without loading the language model and verified their serialized keys, weight/bias shapes, dtypes, and Day 1 manifest hashes.
- Rendered and tokenized a normal/triggered example to verify insertion occurs before the chat template, the trigger changes prompt length, and the separately tokenized response IDs remain identical.
- Wrote `EXPERIMENT_SPEC.md` as the normative contract for model loading, condition construction, token alignment, activation names, layer indexing, probe scoring, patch notation, causal metrics, controls, result records, and claim boundaries.
- Added `scripts/day02_verify_spec.py` and generated `results/day-02/spec-audit.json` so locally checkable specification facts can be revalidated automatically.
- Recorded the durable layer-semantics, recovery-metric, and claim-boundary choices in Decision 0003.

#### Results and evidence

- The measured tensor is response-token `resid_post[12]`, the raw output of zero-based Gemma block 12 after both residual additions and before block 13's input normalization.
- In Hugging Face's hidden-state tuple this tensor is entry 13. After extraction it becomes a compact `[B, 1, T, 3584]` tensor, so serialized probe key `0` indexes the compact one-layer axis rather than model block 0.
- The probe reads every non-padding response-token state, computes `sigmoid(w·x+b)` per token, averages probabilities over response tokens, and then averages example scores over the dataset.
- The trigger is the exact user-content prefix `You are being probed for '{concept}'. ` and is inserted before chat templating. Deception uses concept string `lying`; harmfulness uses `harmful`.
- The primary rescue metric is the unclipped ratio of paired dataset means: `[S_patch - S_triggered] / [S_normal - S_triggered]`. The reverse induction metric is `[S_normal - S_patch] / [S_normal - S_triggered]`.
- Localization, necessity, sufficiency, cross-concept sharing, completeness, and behavioral preservation now have separate meanings and minimum evidence requirements.
- The generated audit completed with status `pass`: 8 source contracts passed, both revisions matched, the checkpoint configuration/hash matched, and all 13 probe files passed key/shape/hash checks.
- Full specification: `EXPERIMENT_SPEC.md`. Machine-checkable evidence: [Day 2 specification audit](../results/day-02/README.md).

#### Failures and unexpected observations

- The checkpoint's Hugging Face model card is an unfilled template. Artifact provenance therefore relies on the pinned Hub revision, file hashes, source repositories, and reproduced behavior.
- The released probe files use mixed serialized key formats (`0` and `"layer_0"`) and mixed FP32/BF16 storage. The released loader normalizes the keys and casts every probe to BF16.
- “Layer 12” and probe key `0` initially look inconsistent but refer to different axes. This was resolved and explicitly tested rather than treated as a naming error.
- The trigger changes prompt sequence length. Response tokens can be paired by response-relative offset, but future prompt-token patches cannot blindly pair equal absolute positions.
- No blocking failure occurred.

## Open questions and blockers

- No blocker prevents Day 3.
- Day 3 must implement and test semantic prompt-token alignment for common content and trigger-only spans before prompt-position patching is used.
- The exact confirmatory bootstrap configuration and component-selection rule remain intentionally assigned to the pre-search freeze on Day 4; the recovery and induction point estimators are already locked.

## Decisions

- [Decision 0003: Lock Experiment Semantics and Causal Metrics](../decision-log/0003-lock-experiment-semantics-and-causal-metrics.md).

## Handoff

- **Current state:** The released experiment is specified down to source revisions, checkpoint/probe artifacts, rendered conditions, response-relative positions, tensor shapes, hook locations, scoring equations, causal estimators, controls, and claim boundaries. The static audit passes.
- **Next action:** Implement the Day 3 paired runner with the exact same response IDs, the specified hook vocabulary, bidirectional patches, and correctness tests.
- **Verification still needed:** None for the Day 2 completion gate. Runtime intervention behavior belongs to Day 3.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-02
- **Final evidence:** `EXPERIMENT_SPEC.md`, [`scripts/day02_verify_spec.py`](../scripts/day02_verify_spec.py), and the [passing audit](../results/day-02/README.md).
- **Final commits:** Day 2 completion commit containing this closeout.
- **Carry-forward items:** Implement the intervention runner and its correctness suite on Day 3; freeze inferential and selection details before component search on Day 4.
