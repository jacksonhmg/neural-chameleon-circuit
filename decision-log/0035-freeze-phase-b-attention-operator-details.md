# Decision 0035: Freeze Phase B Attention Operator Details

**Date:** 2026-08-09  
**Status:** Accepted and frozen  
**Scope:** Phase B attention-operation implementation

## Context

The parent contract names four attention operations but does not fully specify how to map a pattern between different-length normal and triggered prompts, or how to run an induction direction when the normal prompt contains no monitoring-prefix concept span. Those choices must be fixed before a checkpoint preflight or scientific outcome.

## Decision

Freeze the exact conventions in [the machine-readable operator implementation](../results/day-37/frozen-attention-operator-implementation.json):

- exact ordered common-token prompt alignment plus response-relative alignment;
- zero target-shaped support and necessary renormalization for pattern replacement;
- aligned V-state replacement with target pattern retained;
- triggered concept-key removal with full QK/softcap/mask/softmax recomputation;
- triggered concept-V removal with pattern retained; and
- reverse-effect installation for induction into a normal target that has no concept span.

All outputs are installed only at the declared raw selected-head response queries, then evaluated under both ordinary total and sealed direct paths.

## Evidence boundary

No real-checkpoint attention preflight, Phase A outcome, Phase B intervention effect, selection score, or gate result was inspected before this freeze.

## Consequence

Changing any alignment, removal, reverse-installation, grouped-query mapping, or renormalization rule now requires an explicit implementation-failure decision and a complete rerun. The operation labels remain intervention-specific and are not unique mediation claims.
