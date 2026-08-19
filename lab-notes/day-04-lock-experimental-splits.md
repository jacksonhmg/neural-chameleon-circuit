# Day 4: Lock the Experimental Splits

## Phase information

- **Sprint phase:** Day 4
- **Status:** Complete
- **Calendar dates:** 2026-08-03 to 2026-08-03
- **Experiment specification:** `EXPERIMENT_SPEC.md`

## Objective

Construct deterministic discovery, held-out benign validation, and locked safety-test datasets, then freeze their provenance, primary metrics, component-selection rule, and confirmatory safety-test procedure before any component search begins.

## Session log

### 2026-08-03 — Session 1

#### Planned focus

- Audit the pinned released datasets without running a model or probe on the safety split.
- Choose disjoint discovery and held-out validation concepts and include positive and negative examples for every probe.
- Materialize deterministic split files with source identifiers and integrity hashes.
- Freeze the primary metrics, component-ranking rule, safety access boundary, and confirmatory procedure.
- Run structural and reproducibility checks without inspecting safety results.

#### Work performed

- Audited the pinned released datasets and their upstream loading/training rules without loading the model or any probe.
- Implemented `scripts/day04_freeze_splits.py`, which first verifies the three Day 1 source hashes, then deterministically selects examples by salted SHA-256 order and writes exact file hashes, source indexes, ratings, and prompt/response content hashes.
- Assigned HTML, biology-focused, comforting, and German to discovery. Assigned all-caps, chemistry-based, confused, Finnish, jokey, literature-focused, and mathematical to the concept-level validation holdout.
- Selected 64 positive and 64 negative examples for each benign probe. Positives require the matching generation category and topical rating `>=0.75`; negatives require a different generation category in the same role and rating `<=0.25`. Topical refusals are excluded.
- Prevented every benign source row and prompt/response pair from being reused across concepts or split roles. Restricting negative donors to the same role prevents validation-concept responses from leaking into discovery as negatives.
- Materialized the locked safety test with all 100 harmfulness examples per class and the released RepE deception test construction: 62 held-out true statements, each paired with deceptive and honest contexts.
- Froze machine-readable conditions, correct and irrelevant triggers, primary metrics, a 10,000-replicate paired bootstrap with seed 42, component-selection eligibility/ranking/tie-breaks, nested component counts, identity/positive/random controls, and the confirmatory safety procedure in `analysis-plan.json`.
- Implemented `load_experimental_split`. It exposes discovery and validation but fails closed on safety access until an authorization matches the exact safety hash and names a frozen component commit and component-set hash.
- Added four loader tests and `scripts/day04_verify_splits.py`, which performs structural and integrity checks without running a forward pass or displaying safety examples.
- Documented the construction, scientific interpretation, limitations, and unlock procedure, and recorded the durable methodology in Decision 0005.

#### Results and evidence

- Freeze `day04-v1` contains 512 discovery records, 896 validation records, and 324 locked safety records, for 1,732 globally unique example IDs.
- Discovery and validation each contain exactly 64 positive and 64 negative examples per concept. Safety contains deception `62/62` and harmfulness `100/100`.
- The generated discovery, validation, and safety files have SHA-256 values `4d9096f7f635771f0b6ac29b01a12867b84d7636059699b46af36f021097c0f1`, `27c60c4e2f095f4a023c57f81bbb5e48a6e111faf5796040d892370d1f119998`, and `48d6818db1946919c1a49c373170f701219df472b669900d39cde834e0d595ec`.
- Deterministic regeneration matched all five generated files byte for byte.
- The Day 4 structural audit passed 15 of 15 checks, including manifest integrity, class counts, label rules, benign disjointness, concept-role isolation, deception pairing, absence of safety result fields, and fail-closed safety loading.
- The repository test suite passed all 13 tests: nine intervention-runner tests and four split-loader lock tests.
- No model or probe was loaded; no model forward pass, probe score, activation, or safety result was computed during Day 4.
- Evidence: [frozen split package](../data/splits/day04-v1/README.md), [analysis protocol](../docs/day-04-experimental-freeze.md), and [split audit](../results/day-04/README.md).

#### Failures and unexpected observations

- The released file named `...test_TEST.json` contains the separate synthetic-harmfulness concepts, not a benign test partition for the 11 training concepts. The benign split therefore correctly uses the released 20250729 concept dataset and treats validation as a concept-level analysis holdout, not a claim of model- or probe-unseen data.
- The released deception loader has 306 true statements rather than reaching its 512-row cap, so its 80/20 split yields 62 held-out statements per class, not 100.
- Day 1 necessarily computed released safety scores before this lock because reproducing safety evasion was the first completion gate. The new safety test is prospectively protected from component selection, but it is not globally blind in the strongest sense; this limitation is explicit in the protocol.
- An initial direct `unittest` module-name invocation failed because `tests/` is not a Python package. Standard discovery immediately found and passed the tests; no code or scientific check failed.

#### Artifacts and commits

- `data/splits/day04-v1/`: exact split files, machine-readable analysis plan, manifest, and safety-lock instructions.
- `scripts/day04_freeze_splits.py`: deterministic materializer and byte-for-byte check mode.
- `scripts/day04_verify_splits.py`: no-model structural audit.
- `src/neural_chameleon/experimental_splits.py`: guarded project loader.
- `results/day-04/split-audit.json`: 15-check machine-readable audit.
- Day 4 completion commit containing this closeout.

## Open questions and blockers

- No blocker prevents Day 5.
- The Day 5 baseline implementation must use the frozen example IDs, trigger mapping, metrics, threshold rule, and bootstrap procedure without changing `day04-v1`.

## Decisions

- [Decision 0005: Freeze Splits, Metrics, and Component Selection Before Search](../decision-log/0005-freeze-splits-metrics-and-selection.md).

## Handoff

- **Current state:** All datasets, metrics, component-selection rules, controls, and the safety procedure are frozen as `day04-v1`. Structural verification and the full unit suite pass, and the safety loader is locked.
- **Next action:** Run Day 5 baselines on discovery and validation only, preserving exact example pairing and the frozen rules.
- **Verification still needed:** None for the Day 4 completion gate.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-03
- **Final evidence:** [Frozen split package](../data/splits/day04-v1/README.md), [analysis protocol](../docs/day-04-experimental-freeze.md), and [15-check split audit](../results/day-04/README.md).
- **Final commits:** Day 4 completion commit containing this closeout.
- **Carry-forward items:** Keep `safety-test.LOCKED.jsonl` inaccessible until the discovery-selected component set and exact patch procedure satisfy the documented unlock requirements.
