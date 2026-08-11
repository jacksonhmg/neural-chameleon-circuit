# Decision 0075: Repair Day 49 Shared-Tensor Serialization

**Date:** 2026-08-11
**Status:** Frozen implementation repair before inspecting candidate outcomes

## Failure observed

After the unequal-region repair, the first concept's intervention forwards completed, but safetensors rejected shared storage between the deliberately identical same-shaped natural and identity records. Serialization stopped before writing a concept tensor shard. No candidate value or reduced metric was printed, loaded, or inspected.

## Repair

Clone every detached, CPU-contiguous state tensor immediately before serialization. This changes only storage ownership: tensor values, state names, candidates, operations, controls, seeds, gates, and reductions remain identical.

Commit this repair before rerunning the implementation-only preflight and complete execution. The frozen Day 49 contract remains unchanged.
