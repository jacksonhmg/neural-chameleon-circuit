# Manuscript V2 visual QA

The final PDF was rasterized with Poppler at 120 DPI and every page was inspected
at original rendered resolution on 2026-08-13.

## Page audit

| Page | Content | Result |
|---:|---|---|
| 1 | Title, abstract, introduction | Pass: no clipping, collision, or orphaned title text |
| 2 | Experimental setting and methods | Pass: equations, bullets, and component identities legible |
| 3 | Figure 1 and start of mechanism results | Pass: all panels and legend legible; caption attached |
| 4 | K12 factorization and acquisition text | Pass: balanced columns; no overflow |
| 5 | Figures 2 and 3 | Pass: panel labels, legends, axes, and captions legible |
| 6 | Upstream and operational results | Pass: equations and headings remain with their text |
| 7 | Figure 4, integrated chain, related work | Pass: log axis and hybrid labels legible |
| 8 | Figure 5 and limitations | Pass: threshold diagnostic and claim-scope matrix legible |
| 9 | Limitations close, conclusion, references | Pass: final scope/closure language is legible; no broken URLs or bibliography overflow |
| 10 | References and appendices A--D | Pass: appendix transition and component table legible |
| 11 | Appendix Tables 2--5 | Pass: all numerical cells and claim boundaries contained |

## Defects repaired during QA

- Increased the two-column float top fraction so Figures 1--5 appear beside the
  corresponding results rather than after the appendices.
- Rewrote the integrated causal-chain equation to fit a single column.
- Removed two unbreakable bibliography URLs that produced overfull boxes.
- Rebuilt until the final LaTeX log contained zero overfull boxes and zero unresolved
  references or citations.

The rendered page files in `rendered-pages/` correspond one-to-one with the final
11-page PDF.

The project-closeout refresh was rebuilt from source commit
`ea131c9946223cf642a520d6643cd667a0634b71` and all 11 pages were reinspected after
the conclusion changed. No new visual defect was introduced.
