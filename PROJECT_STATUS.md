# Project Status

**Status:** Complete

**Closed:** 2026-08-13

**Intended title:** **Causal Mechanisms of Activation-Monitor Evasion in a Neural Chameleon**

**Final disposition:** Scientifically title-worthy within the stated one-organism scope

## Bottom line

The project identified and prospectively tested the following causal account in one
released Gemma-2-9B Neural Chameleon:

```text
monitoring condition identity
→ distributed early residual representation
→ joint source-side K/V rewrite at informative prefix tokens
→ coordinated 12-head K12 state
→ direct monitor displacement plus a parallel tail-attention contribution
→ operational activation-monitor evasion
```

Using the released tokenwise-sigmoid-then-average monitor score and ordinary
no-trigger execution as the operational baseline, total evasion, exact-K12 recovery,
and reciprocal exact-K12 induction were positive with zero-excluding intervals for
all 11 fine-tuning concepts on both the project-unused native panel and the larger
content-disjoint broad panel. The exact precursor contains at most 8.12% of the
corresponding Chameleon effect norm across the frozen acquisition endpoints.

No further experiment is required to complete this project. Cross-checkpoint,
unseen-concept, and nonlinear-monitor replication would be valuable future research,
but they are optional extensions rather than unfinished project work.

## Canonical current artifacts

1. [Manuscript V2 PDF](output/pdf/neural-chameleon-causal-mechanisms-v2.pdf) — final
   mechanism-centered paper draft.
2. [Manuscript V2 package](paper-v2/README.md) — manuscript source, five figures,
   source-data CSVs, claim–evidence ledger, chronology ledger, numerical audit, and
   rendered-page visual QA.
3. [Day 70 corrected endpoint results](docs/day-70-endpoint-correction-results.md) —
   final operational result.
4. [Decision 0102](decision-log/0102-accept-corrected-endpoint-title-worthiness.md) —
   acceptance of scientific title-worthiness.
5. [Decision 0103](decision-log/0103-close-project-after-manuscript-v2.md) — project
   closure and the optional status of replication.

## Document precedence

This file and Decisions 0102–0103 state the current project disposition. Manuscript
V2 is the current paper. When an older document uses “provisional,” “unearned,”
“next experiment,” or “final gate,” read that wording as its truthful at-the-time
status unless the document explicitly says otherwise.

In particular:

- Day 62 failed its unseen-concept semantic-qualification gate. That final panel
  remained unopened, so the project does **not** claim generalization to concepts
  absent from Chameleon fine-tuning.
- Day 65 and Day 68 failed their frozen title gates. Those failures remain recorded
  and are not retroactively converted into passes.
- Day 68's 8/11 result came from a binary threshold endpoint using an
  irrelevant-monitoring trigger as baseline and a surrogate raw-margin score.
- Day 70 prospectively evaluated the corrected scientific endpoint: ordinary
  execution as baseline, the exact released monitor score, and paired continuous
  causal effects as primary. It found positive zero-excluding total, recovery, and
  induction effects for 11/11 concepts on both panels.

The corrected Day 70 interpretation supersedes the earlier *overall title
disposition* without erasing any earlier frozen gate result.

## Scope boundary

The supported title concerns:

- one released Gemma-2-9B Neural Chameleon checkpoint;
- the corresponding one-layer logistic monitor family at the studied residual site;
- all 11 concepts used in Chameleon fine-tuning;
- project-unused native-distribution content and a content-disjoint broad panel; and
- teacher-forced causal interventions with exact response-token reuse.

It does not establish:

- unseen-concept semantic generalization;
- cross-seed, cross-checkpoint, cross-scale, or cross-architecture invariance;
- a universal monitor-evasion circuit;
- the same mechanism for nonlinear or adversarially retrained monitors; or
- free-generation behavioral invariance.

## Historical material

The original [`paper/`](paper/README.md), prospective plans in [`docs/`](docs/), and
the full [`decision-log/`](decision-log/) are intentionally preserved. They document
the adaptive research chronology, including negative results and abandoned
hypotheses. Historical wording should not be silently rewritten because doing so
would obscure which claims and gates were frozen at each stage.

## Optional future work

Independent replication is the highest-value future scientific extension. Public
third-party Neural Chameleon adapters now exist, but they use materially different
training procedures, probe layers, pooling rules, and adapter architectures. The
project owner elected not to begin that separate replication program. This decision
does not weaken or leave incomplete the current one-organism causal case study.
