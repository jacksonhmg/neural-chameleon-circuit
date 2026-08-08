# Decision 0026: Accept the Audited Private Manuscript Candidate

- **Date:** 2026-08-08
- **Status:** Accepted

## Decision

Accept the 14-page manuscript *Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon* and its traceable source package as the project's private manuscript candidate.

The accepted PDF is `output/pdf/neural-chameleon-causal-mechanisms-private.pdf`, built from source commit `92fbe3326c6875e9dca8536ba6be550dd35394a7`, with SHA-256 `f5497295937645c76670d8a967099f6530bf131b08a4723607600bbb44cce1e9`. Two clean builds from that source commit produced identical bytes.

Acceptance means that the candidate passed the frozen Day 30 gate:

- all 21 audited numerical claims match sealed source artifacts;
- all six manuscript figures match their hash-pinned manifest and source data;
- all 13 cited bibliography entries resolve with no unused entries;
- the full 62-test suite passes;
- the LaTeX/BibTeX build is warning-free and has no unresolved tokens;
- all 14 rendered pages passed visual inspection; and
- all 16 candidate-audit checks pass.

The claim hierarchy and qualifications frozen in Decision 0025 remain controlling. Candidate acceptance is a packaging and readiness decision, not a new empirical result or a claim upgrade.

## Rationale

The manuscript now provides a single coherent, auditable account of the completed project: locked safety transfer is primary; head-dominated localization and the failure of a rigid minimal-circuit account qualify the mechanism; non-original route transport extends the mechanism prospectively; and behavioral results show bounded but nonzero output coupling. Adverse and heterogeneous results remain visible in the main figures and prose.

Deterministic generation, numerical provenance, page-level inspection, and an executable package audit make the candidate suitable for deliberate private review without silently changing the sealed evidence.

## Consequences

- Days 26–30 are complete.
- The next step is private author-level review, including authorship, venue, and any further-experiment decisions.
- The manuscript remains a private candidate with authors and venue unset.
- No push, tag, release, submission, upload, draft transmission, message, author contact, or external evaluation is authorized or has occurred.
- Any later release requires a separate explicit decision and a fresh audit of the exact release artifact.
