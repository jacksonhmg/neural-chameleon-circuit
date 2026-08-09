# Private Manuscript Package

This directory contains the private manuscript for the Neural Chameleon Circuit project. It is derived from the repository's frozen and sealed result packages; it is not a separate source of scientific truth.

## Status

- **Working title:** *Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon*
- **Authors:** TBD
- **Venue:** TBD
- **Private candidate:** `output/pdf/neural-chameleon-causal-mechanisms-private.pdf`
- **Candidate source commit:** `92fbe3326c6875e9dca8536ba6be550dd35394a7`
- **Candidate SHA-256:** `f5497295937645c76670d8a967099f6530bf131b08a4723607600bbb44cce1e9`
- **Acceptance audit:** 16/16 checks passed
- **External status:** not pushed, released, submitted, uploaded, or sent

## Successor research status

The accepted private candidate remains the historical manuscript artifact of the completed first project. It is not fresh confirmation of the successor mechanism program and its acceptance does not upgrade the existing causal-localization evidence.

Future work targeting the same title is governed by the [successor causal-mechanisms research plan](../docs/causal-mechanisms-research-plan.md) and [Decision 0027](../decision-log/0027-freeze-successor-causal-mechanisms-plan.md). The title remains conditional on passing that plan's acquisition, pre-outcome prediction, causal-control, endogenous-propagation, concept-restoration, and operational gates on fresh evidence.

## Package layout

- `manuscript.tex`: auditable LaTeX source.
- `references.bib`: primary-source bibliography.
- `claim-evidence-map.md`: claim provenance and permitted wording.
- `outline.md`: frozen narrative structure.
- `figure-plan.md`: frozen main-figure responsibilities.
- `figures/`: generated manuscript figures.
- `source-data/`: machine-readable data used by those figures.
- `appendix/`: extended tables and audit-oriented supplements.
- `build/`: local PDF and compilation artifacts.

## Rebuild

From the repository root, rebuild the accepted private candidate with:

```sh
scripts/day30_build_private_manuscript.sh 92fbe3326c6875e9dca8536ba6be550dd35394a7
```

The wrapper derives the PDF timestamp from the source commit and builds in an isolated temporary directory. Two clean builds produced the same byte-level SHA-256 above. Generated scientific figures must be rebuilt through `scripts/day27_build_manuscript_figures.py`, not edited by hand.

The candidate acceptance record is [Decision 0026](../decision-log/0026-accept-private-manuscript-candidate.md). Build, numerical, visual, and package-audit evidence is under `paper/audits/`.

## Authority

If prose here conflicts with a sealed result, the sealed result controls. If prose here conflicts with the manuscript-scope decision, [Decision 0025](../decision-log/0025-freeze-manuscript-scope-and-claims.md) controls until it is explicitly superseded.
