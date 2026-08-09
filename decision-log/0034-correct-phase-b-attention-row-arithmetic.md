# Decision 0034: Correct Phase B Attention Row Arithmetic Before Outcomes

**Date:** 2026-08-09  
**Status:** Accepted and frozen  
**Scope:** Post–Gate 1 Phase A–B implementation gate

## Context

A synthetic exact-row test expanded the frozen attention matrix before any Phase A reducer or Phase B population execution. The three written formulas in the Phase A–B contract evaluate to exactly half their listed expected counts:

- discovery: `256 × 16 × 4 × 2 × 2 × 1 = 65,536`, not `131,072`;
- heldout: `610 × 1 × 4 × 2 × 2 × 2 = 19,520`, not `39,040`;
- negatives: `866 × 1 × 4 × 2 × 2 × 2 = 27,712`, not `55,424`.

The parent contract's counting rule says one row holds all 13 margins and scores. Its discovery rule is Chameleon-only, and its heldout and negative formulas already multiply by both checkpoints. No declared scientific dimension supplies the extra factor of two.

## Decision

Freeze [the machine-readable arithmetic clarification](../results/day-37/frozen-phase-b-row-accounting-clarification.json). Preserve every scientific cell and correct only the expected counts. The corrected Phase B total is `444,224` rows.

Do not create duplicate rows to reproduce the mistaken sum. Exact row audits must use the corrected values and also report the parent-contract mismatch and this clarification's hash.

## Evidence boundary

No Phase A reduction, Phase B intervention population, selection score, scientific effect, or gate outcome was inspected before this decision. The discrepancy was found by pure multiplication in a synthetic implementation test.

## Consequence

Implementation continues under the unchanged scientific design. This is not threshold tuning, outcome-dependent selection, or a change to any population, operator, direction, path, model, or endpoint.
