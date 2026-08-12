# Decision 0084: Replay the Saved Day 52 QKV Baseline for Localization

**Date:** 2026-08-11
**Status:** Frozen implementation repair before localization outcomes

## Observation

The Day 55 candidate-blind preflight generated no localization outcome and stopped because exact Day 54 donor K12 reproduced with zero error, while a fresh SXM realization of the Day 52 full-prefix QKV computation differed from its saved PCIe K12 output by `0.1083984375` and `0.09375` in the two directions. Both exceed the prospectively frozen `0.02` reproduction tolerance. Haar, response, finiteness, and hook checks passed.

## Decision

Do not loosen the tolerance and do not compare completions against a hardware-shifted baseline. Replay the exact hash-verified saved Day 52 primary-donor K12 tensor as the base K12 replacement for each concept and direction. Complete specified heads or layers from that literal saved QKV-produced state to the exact natural donor K12 state measured in Day 54.

This implements the frozen scientific question directly: which missing selected-head components distinguish the observed Day 52 QKV-produced K12 state from exact donor K12? It changes no candidate set, completion delta, donor, endpoint, control, seed, metric, or interpretation. The parent-reproduction gate remains `0.02` and should now test the exact replay path.

## Boundary

Commit the repair and rerun the entire preflight. The failed preflight report is quarantined and may not be used scientifically.
