# Day 16: Triage Site-Shuffling Artifacts

## Phase information

- **Sprint phase:** Day 16
- **Status:** In progress
- **Calendar date:** 2026-08-07
- **Frozen protocol:** [Site-shuffling plan](../results/day-15/frozen-site-shuffling-plan.json)

## Objective

Determine whether the two strong Day 14 induction mappings transport a trigger-specific source signal or obtain their effect from absolute source/destination mismatch.

## Session log

### 2026-08-07 — Session 1

#### Planned focus

- Add tested destination-relative delta construction.
- Add route-matched source-condition comparisons.
- Pass a benign real-checkpoint preflight before opening the safety pilot.
- Run and seal the complete 1,344-row frozen grid.

#### Work performed

- Implemented pure capture utilities for delta addition/subtraction, RMS matching, coordinate transforms, and valid-token reversal.
- Added five unit tests covering signs, scale matching, transforms, token order, and alignment rejection.
- Implemented a resumable, deterministic evaluator and a separate bootstrap analyzer.

#### Results and evidence

- The benign real-checkpoint preflight passed with exact identity and alpha-zero probe scores, finite absolute and RMS-matched jobs, exact paired response IDs/masks, and zero remaining hooks.
- Sealed all 1,344 frozen rows: 32 examples, two baselines, and 40 intervention conditions per example.
- Deception seed 211 absolute induction is 0.900, but 0.875 is attributable to normal-to-normal mismatch. Conditional induction is 0.025 [0.007, 0.046], raw delta induction is 0.031 [0.013, 0.054], and RMS-matched delta induction is 0.061 [0.041, 0.083]. Raw delta rescue does not clear zero, while the RMS-matched rescue is 0.016 [0.007, 0.029].
- Deception seed 223 absolute induction is 0.934, with same-condition mismatch 0.791. Conditional induction is 0.142 [0.110, 0.178], raw delta induction is 0.100 [0.074, 0.129], and RMS-matched delta induction is 0.174 [0.132, 0.219]. Rescue remains smaller but positive under conditional and delta tests.
- Harmfulness seed 211 retains the clearest bidirectional destination-relative signal: conditional induction/rescue 0.160/0.186, raw delta 0.108/0.135, and RMS-matched delta 0.202/0.263; all intervals exclude zero.
- Harmfulness seed 223 is an adverse result. Conditional induction/rescue are -0.210/-0.333, raw delta effects are -0.074/-0.283, and RMS-matched effects are -0.028/-0.121; all intervals exclude zero in the opposing direction. Its absolute effects are therefore mismatch-driven rather than portable-code evidence.
- The signed dose curves pass through zero at alpha 0 and reverse direction under negative alpha. The raw-grid, uniqueness, finiteness, metric, and figure audit passes.

#### Failures and unexpected observations

- The original absolute shuffled induction values substantially overstate trigger-specific transport because normal-to-normal mismatch alone can be very large.
- Portable transport is not a property of every selected-head permutation: harmfulness seed 223 reverses under the primary trigger-specific estimands.

## Open questions and blockers

- None.

## Decisions

- The durable methodology is already frozen under [Decision 0020](../decision-log/0020-freeze-site-shuffling-follow-up.md).

## Handoff

- **Current state:** Day 16 is complete with a passing 1,344-row audit and visually inspected dose-response figure.
- **Next action:** Build the complete selected/null source–destination atlas on the frozen pilot.
- **Verification still needed:** None for Day 16.

## Day closeout

- **Outcome:** Complete
- **Completed on:** 2026-08-07
- **Final evidence:** [Day 16 result package](../results/day-16/README.md)
- **Carry-forward items:** Use both successful and failed routes to explain population, destination, and depth effects in Day 17.
