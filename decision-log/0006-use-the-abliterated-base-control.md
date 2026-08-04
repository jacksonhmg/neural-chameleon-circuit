# Decision 0006: Use the Original Abliterated Model as the Base Control

- Date: 2026-08-04
- Status: Accepted

## Decision

Use `IlyaGusev/gemma-2-9b-it-abliterated` at revision `e2b6426b20a3a889f0c182056b0dbbb7fa585d25` as the non-chameleon base-model control for the released 9B Neural Chameleon.

Apply the same released probes, frozen examples, response tokens, prompt conditions, activation location, and metrics to the chameleon and base. Treat fixed-continuation NLL and response activation norms as measured preservation diagnostics rather than as selection criteria or proof of behavioral equivalence.

## Rationale

The pinned upstream main training configuration names `gemma_2_9b_it_abliterated_ilya` as the organism's starting model. Stock `google/gemma-2-9b-it` would differ before chameleon fine-tuning and would confound the control with the authors' abliteration step. The public IlyaGusev repository resolves that local alias and supplies an immutable revision.

The trigger phrase can change likelihood or activations even without learned evasion. An example-aligned run on the actual pre-finetuning model distinguishes this generic prompt effect from the chameleon-specific monitor suppression.

## Consequences

- Future base comparisons for this checkpoint must use the pinned abliterated revision unless explicitly labeled as a different control.
- Base-model artifact hashes are recorded in `manifests/day-05-base-model.sha256`.
- A small activation-norm change cannot cancel evidence of a continuation-likelihood shift; behavioral-preservation claims require the later generation and task-quality controls in the sprint.
