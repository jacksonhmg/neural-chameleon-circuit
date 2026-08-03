# Decision 0004: Define Head Patching and Response-Aligned Caches

- Date: 2026-08-02
- Status: Accepted

## Decision

Define an individual attention-head output as the selected `head_dim` slice of Gemma-2's concatenated eager-attention result immediately before `self_attn.o_proj`. For the released checkpoint, each of the 16 query-head slices has width 256. Record this location as `head_out[L,h]`.

Store Day 3 activation caches in response-relative coordinates. Every cache carries the exact separately tokenized response IDs and response mask. Reject a patch before installing hooks unless source and destination continuations and masks match exactly. Patch only non-padding response positions, with optional zero-based response-offset selection.

## Rationale

Gemma's output projection mixes all concatenated head slices, so a head must be selected before that projection if the runner is to replace one head without replacing the others. Naming the pre-projection location avoids incorrectly describing it as an already isolated residual-stream contribution.

Normal and triggered prompts have different sequence lengths. Response-relative caches allow their shared teacher-forced continuation to align exactly without relying on incompatible absolute sequence positions. Carrying token IDs with every cache makes this scientific invariant executable rather than advisory.

## Consequences

- Head-level results must identify the zero-based block and query-head index and state that patching occurs before `o_proj`.
- Day 3 caches cannot silently patch a different continuation.
- Prompt-token interventions require a separate, tested semantic alignment for common prompt content and trigger-only tokens before they are used in later localization work.
