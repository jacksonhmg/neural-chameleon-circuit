# Day NN: Short Phase Title

Use this template when a sprint day begins. Copy it to the filename specified in `SPRINT.md`, replace the placeholders, and append a new dated session section whenever work resumes.

Do not use the lab note as a second sprint checklist. `SPRINT.md` owns task status; this file records execution, evidence, failures, and handoff context.

## Phase information

- **Sprint phase:** Day NN
- **Status:** In progress
- **Calendar dates:** YYYY-MM-DD to YYYY-MM-DD
- **Sprint tracker:** [SPRINT.md](../SPRINT.md)
- **Project brief:** [Neural Chameleon Evasion Research Plan](../docs/neural-chameleon-evasion-research-plan.md)

## Objective

State the concrete outcome sought during this phase.

## Session log

### YYYY-MM-DD — Session 1

#### Planned focus

- State the subset of sprint tasks targeted in this session.

#### Work performed

- Summarize code changes, commands, experiments, inputs, configurations, and relevant version or artifact identifiers.
- Prefer reproducible commands or links to scripts over an exhaustive terminal transcript.
- Never record credentials, access tokens, or other secrets.

#### Results and evidence

- Record quantitative results, observations, controls, and links to generated artifacts.
- Distinguish measured results from interpretations or hypotheses.

#### Failures and unexpected observations

- Record failed attempts, anomalous behavior, and negative results that could affect later work.

#### Artifacts and commits

- List relevant files, result directories, immutable artifact identifiers, and commit hashes.

## Open questions and blockers

- Record unresolved questions and blockers, including what evidence or external input would resolve them.

## Decisions

- Link any durable decision promoted to `decision-log/`.
- Write `None` when the session produced no decision-log-worthy decision.

## Handoff

- **Current state:** Summarize what is working and verified.
- **Next action:** Name the next concrete action.
- **Verification still needed:** List anything that must be checked before the sprint day can close.

## Day closeout

Complete this section only when the day’s completion gate in `SPRINT.md` is satisfied.

- **Outcome:** Complete / Blocked
- **Completed on:** YYYY-MM-DD
- **Final evidence:** Link the deliverables and key results.
- **Final commits:** List the relevant commit hashes.
- **Carry-forward items:** List any explicitly deferred work and where it is tracked.
