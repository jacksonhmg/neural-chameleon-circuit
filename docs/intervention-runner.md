# Paired Intervention Runner

The reusable Day 3 runner is implemented in [`src/neural_chameleon/interventions.py`](../src/neural_chameleon/interventions.py). It prepares paired teacher-forced inputs, captures response-aligned activations, and patches those activations in either direction while installing and removing PyTorch hooks within a single forward call.

## Supported targets

| `ActivationKind` | Exact Gemma-2 location | Hook side | Captured shape |
|---|---|---|---|
| `RESID_PRE` | Decoder block input, before `input_layernorm` | Block input | `[B, T, D]` |
| `ATTN_OUT` | Output of `post_attention_layernorm`, before attention residual addition | Module output | `[B, T, D]` |
| `MLP_OUT` | Output of `post_feedforward_layernorm`, before MLP residual addition | Module output | `[B, T, D]` |
| `BLOCK_OUTPUT` | Decoder block output, equivalent to `resid_post` | Block output | `[B, T, D]` |
| `HEAD_OUTPUT` | Selected head slice immediately before `self_attn.o_proj` | Projection input | `[B, T, head_dim]` |

For the released checkpoint, `D=3584`, there are 16 query heads, and `head_dim=256`.

`PatchSite.response_positions` optionally selects zero-based response-relative offsets. `None` means every non-padding response token. The current cache intentionally does not pair prompt positions: triggered and normal prompts have different lengths, and prompt patching must first implement semantic alignment for the common template/content spans and the trigger-only span.

## Paired input contract

`prepare_pairs(prompts, responses, trigger)`:

1. Tokenizes every response once, with right padding and no added special tokens.
2. Creates normal user messages and exactly prefixed triggered user messages.
3. Applies the checkpoint's chat template to each user message.
4. Left-pads rendered prompts and concatenates the shared response tensor.
5. Builds attention masks and padding-independent position IDs.
6. Rejects empty responses or unequal prompt/response batch sizes.

Every `CapturedActivation` carries the response IDs and mask that produced it. A patch is rejected before hook registration if the destination has different response IDs, a different mask, or an incompatible response width.

## Minimal usage

Run with the tracked Day 1 environment and `src/` on the import path:

```python
from neural_chameleon import (
    ActivationKind,
    LinearProbe,
    PairedInterventionRunner,
    PatchSite,
    recovery_fraction,
)

runner = PairedInterventionRunner(model, tokenizer)
pair = runner.prepare_pairs(prompts, responses, trigger="german")
site = PatchSite(ActivationKind.BLOCK_OUTPUT, layer=12)

normal = runner.run(pair.normal, capture_sites=[site])
triggered = runner.run(pair.triggered, capture_sites=[site])
rescue = runner.run(
    pair.triggered,
    capture_sites=[site],
    patch_cache={site: normal.captures[site]},
)

probe = LinearProbe.load("external/minimal_neural_chameleons/probes/german_weights.pt")
normal_mean = probe.score(normal.captures[site], device=runner.device).mean().item()
triggered_mean = probe.score(triggered.captures[site], device=runner.device).mean().item()
rescue_mean = probe.score(rescue.captures[site], device=runner.device).mean().item()
recovery = recovery_fraction(normal_mean, triggered_mean, rescue_mean)
```

Swap source and destination conditions for reverse-direction induction. Multiple non-conflicting or deliberately grouped sites can be captured or patched in one pass.

## Hook safety

Patch hooks are registered before capture hooks, so a capture at a patched site records the value actually sent downstream. Every handle is removed in a `finally` block, including when model execution raises. `registered_hook_count()` exposes the remaining model hook count for test assertions.

The runner clones destination tensors before replacement and applies a Boolean response mask. Padded response positions are never patched. Captures are detached and moved to CPU, limiting accelerator retention and making result inspection safe after the forward pass.

## Verification

Fast deterministic suite:

```bash
PYTHONPATH=src external/minimal_neural_chameleons/.venv/bin/python \
  -m unittest discover -s tests -v
```

Released-checkpoint integration suite:

```bash
external/minimal_neural_chameleons/.venv/bin/python \
  scripts/day03_verify_runner.py
```

The integration results and BF16/MPS batch-agreement tolerances are recorded in [`results/day-03/`](../results/day-03/README.md).
