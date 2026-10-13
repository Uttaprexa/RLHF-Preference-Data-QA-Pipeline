# Data Quality Rubric and Acceptance Criteria

This document turns "good preference data" into rules a machine can check. Every
threshold here maps to code in `src/qa_pipeline/` so the definition and the
implementation can't drift apart.

## What a preference pair must satisfy

A pair is a prompt plus two responses, where annotators marked one as `chosen`.
A pair is **acceptable** when all of the following hold.

| # | Criterion | How it's measured | Rule (reject if...) | Code |
|---|-----------|-------------------|---------------------|------|
| 1 | Label correctness: `chosen` really is better | LLM judge, both A/B orders | Judge prefers `rejected` in **both** orders | `j_label_flip` |
| 2 | Relevance: `chosen` answers *this* prompt | Judge relevance score (1–5); TF-IDF similarity | Mean judge relevance ≤ 2, or TF-IDF < 0.02 | `j_low_quality_chosen`, `h_off_topic` |
| 3 | Effort: `chosen` is substantive and complete | Judge effort score (1–5); word count | Mean judge effort ≤ 2, or < 6 words / generic filler | `j_low_quality_chosen`, `h_low_effort` |
| 4 | Informative comparison: responses differ | Exact match; judge "identical" tag | `chosen == rejected` | `h_duplicate`, `j_identical` |

Pairs where the judge's verdict **flips when A/B order is swapped** are not rejected
automatically. They are routed to human review (`j_needs_review`), because the
judge itself is unreliable on them.

## Score anchors used by the judge

- **Relevance:** 5 = directly addresses the request; 3 = partially on topic; 1 = about something else.
- **Effort:** 5 = complete and specific; 3 = correct but thin; 1 = empty, generic filler, or cut off mid-thought.
- **Helpfulness:** 5 = the user could act on this; 1 = no practical value.
- **Harmlessness:** 5 = no concerns; 1 = clearly harmful.

## Batch-level acceptance

A batch is **accepted** only if the upper bound of the 95% Wilson confidence
interval on its flagged rate is below **5%** (`batch_acceptance` in `evaluate.py`).
Using the upper bound rather than the point estimate means small batches can't
pass by luck.

## Drift monitoring

Weekly batches are tracked against a baseline (the first 4 weeks):

- **p-chart:** a batch is out of control if its flagged rate is outside the
  baseline mean ± 3σ, where σ = √(p̄(1−p̄)/n).
- **PSI** on a response-quality feature: < 0.1 stable, 0.1–0.25 investigate, > 0.25 major shift.
- **KS test** p-value on the same feature, as a second opinion.

## Known limitations (to revisit with real results)

- The TF-IDF relevance check misses paraphrased answers. Tune `min_relevance` on
  real data or replace it with sentence embeddings.
- The judge sees content, not annotator intent, so subtle disagreements (both
  responses fine, annotator preferred style) aren't defects and shouldn't be flagged.
- Thresholds above are starting points. Calibrate them on a human-labeled sample
  and record the chosen values and their precision/recall here.
