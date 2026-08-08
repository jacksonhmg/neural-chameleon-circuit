# Local Blinded Qualitative Assessment

## Scope

Codex inspected all 80 items in `qualitative-review-packet.md` without consulting `qualitative-review-key.json` during the content review. This is one local, non-independent assessment. It is secondary descriptive evidence, not a substitute for the frozen Day 23 distributional gate and not an independent human-rating study.

## Findings while blinded

- Every nonempty output was linguistically coherent at the 32-token diagnostic horizon. No obvious gibberish, runaway repetition, or broken tokenization appeared.
- Deception prompts continued to receive factual answers. Some variants continued into an unsolicited synthetic multi-turn fact exchange, which is a real surface-form change but not a loss of basic factual task fulfillment.
- The harmfulness-negative fruit prompt continued to receive relevant, benign fruit choices and explanations.
- Every harmfulness-positive output rejected or qualified the racial stereotype and began an appropriate explanation. No reviewed output endorsed the stereotype or adopted a more harmful stance.
- Many harmfulness outputs reached the frozen 32-token maximum mid-sentence. This is a diagnostic truncation shared by baseline and intervention conditions, not evidence that the model naturally terminated incoherently.

## Post-unblinding relation to automatic metrics

The key and automatic metrics show that selected mappings change the exact greedy continuation more often than route-matched nulls, especially for harmfulness. The blinded review indicates that these changes are generally coherent alternative continuations rather than obvious task failure or stance reversal. The four-example scope is far too small to claim general semantic equivalence.

## Disposition

The qualitative diagnostic is consistent with Day 23's bounded conclusion: selected non-original routes produce small but real downstream distributional effects, with stronger harmfulness coupling, while the observed outputs remain broadly task- and stance-preserving in this tiny fixed sample. It does not upgrade the evidence to universal behavioral preservation.

