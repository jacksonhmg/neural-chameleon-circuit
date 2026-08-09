# Day 36: Freeze the Post–Gate 1 Phase A–B Contract

This package freezes the development variance/carrier diagnostics and causal-operation identification program before any new Phase A–B outcome is produced.

- [`frozen-phase-a-b-contract.json`](frozen-phase-a-b-contract.json) is authoritative for populations, variables, folds, operators, controls, thresholds, seeds, gates, and fallback behavior.
- [`freeze-audit.json`](freeze-audit.json) records the passing pre-outcome integrity audit.
- [`../../docs/day-36-phase-a-b-freeze.md`](../../docs/day-36-phase-a-b-freeze.md) explains the contract in prose.
- [`../../decision-log/0032-freeze-post-gate1-phase-a-b-contract.md`](../../decision-log/0032-freeze-post-gate1-phase-a-b-contract.md) preserves the authorization boundary.

The contract SHA-256 is `66952a5124a55a3e29d24d3657854dd4f4c1c30d8c04705d2206b4b4d12c7ad8`, frozen against parent commit `7013fe52407909607ab981de01cfeb0840006a62`.

The two-stage discovery-selection and held-out-evaluation matrix contains exactly 556,992 Phase B effect rows; natural endpoints, feature artifacts, accounting rows, and preflight attempts are separate.

No model execution, new diagnostic reduction, intervention outcome, or fresh-data access occurred during the freeze. Empirical execution still requires explicit user authorization after this package is committed.
