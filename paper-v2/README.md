# Manuscript V2

This directory is the mechanism-centered manuscript and audit package for
**Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon**.

The earlier `paper/` directory is preserved as a historical manuscript candidate.
It predates the K12 Q/K/V factorization, acquisition localization, and corrected
operational endpoint and must not be treated as the current paper.

## Build

From the repository root:

```sh
scripts/day71_build_manuscript_v2.sh
```

The command regenerates all five figures and source-data CSV files from sealed JSON,
builds the paper, runs numerical/text/package audits, and writes the final PDF to:

`output/pdf/neural-chameleon-causal-mechanisms-v2.pdf`

## Audit surfaces

- `claim-evidence-ledger.csv`: one row per consequential manuscript claim.
- `evidence-chronology.csv`: panel status and adaptive chronology.
- `source-data/figure-01.csv` through `figure-05.csv`: exact plotted values.
- `audits/manuscript-v2-audit.json`: numerical, text, and file-hash assertions.
- `audits/manuscript-v2-build.log`: final LaTeX build log.
- `audits/rendered-pages/`: PNG rendering of every final PDF page for visual QA.

The formal Day-68 8/11 threshold gate remains a failed preregistered result. The V2
paper explains the later, prospectively frozen endpoint correction; it does not
rewrite the old gate.
