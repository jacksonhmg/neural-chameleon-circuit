# Decision 0036: Fix the Attention Capture Keyword Hook

**Date:** 2026-08-09  
**Status:** Accepted implementation fix  
**Scope:** Phase B preflight failure; no scientific outcome

## Failure

The first real-checkpoint preflight stopped in the Chameleon attention-state capture. Current Transformers calls `Gemma2Attention.forward` with `hidden_states` in keyword arguments. The `with_kwargs` pre-hook incorrectly attempted to read only the positional tuple, which was empty.

The failure occurred before any attention intervention, precursor preflight, Phase A reduction, population effect, selection score, or scientific gate.

## Decision

Preserve [attempt 1](../results/day-39/real-checkpoint-preflight-attempt-1.json). Change the hook to prefer `kwargs["hidden_states"]` and fall back to the positional tensor only for compatible call sites. Add a regression test that invokes a hooked attention-style module entirely through keyword arguments.

Commit the fix before rerunning the complete two-checkpoint preflight. Do not resume from the failed partial in-memory run.
