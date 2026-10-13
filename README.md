# RLHF Preference Data QA Pipeline

**Operationalizing data quality at scale.** An automated pipeline that scores
annotation quality in RLHF preference data (Anthropic HH-RLHF), measures how well it
catches deliberately planted bad examples, and monitors quality drift across simulated
annotation campaigns. Runs entirely free on a laptop using a local open-weight judge.

> Status: in progress. Results below are from real HH-RLHF data. Next step is a
> hand-labeled sample to separate judge errors from genuine label noise.

## Key findings

1. **Word-overlap relevance is unusable on conversational data.** About 35% of good
   HH-RLHF replies share no keywords with the user's message ("Are you sure you want
   that?"). Switching to sentence embeddings cut the heuristic false positive rate from
   **36.5% to 10.8%** while still catching 91% of off-topic answers.
2. **Heuristics and the LLM judge catch different defects.** Rules catch surface
   problems (duplicates, filler). The judge is needed for defects that require
   understanding: it raised label-flip detection from **10% to 57%** and truncation
   detection from **29% to 88%**.
3. **A 7B local judge is position-biased.** Swapping the order of the two responses
   changed its verdict on **32.7%** of 300 pairs. Every pair is judged in both orders,
   and only order-consistent verdicts count.
4. **The judge's self-reported confidence is uninformative.** It rated nearly every
   verdict 4–5 out of 5, so confidence can't be used to filter verdicts.
5. **Not all rules survive calibration.** The judge's "identical responses" tag fired on
   11.5% of clean pairs when only ~0.5% were duplicates, so it was removed. Judge
   label-flip verdicts fired on 23.5% of clean pairs, so they now route to a **human
   review queue** instead of auto-rejecting.
6. **HH-RLHF contains real defects.** Manual inspection of flagged "clean" pairs found
   chosen responses that coach users on bribery, invent nonsense explanations, or brush
   off serious questions. Some apparent false positives are true positives.

## Results

### Heuristic relevance check (n = 2,995 pairs, 15% injected defects)

| Relevance method | False positive rate | Precision | F1 | Off-topic recall |
|---|---|---|---|---|
| TF-IDF word overlap | 36.5% | 0.27 | 0.40 | 94% |
| Sentence embeddings (MiniLM) | **10.8%** | **0.51** | **0.57** | 91% |

### Heuristics vs. LLM judge (n = 300 pairs, 30% injected, judge = qwen2.5:7b via Ollama)

Recall by defect type, before rule calibration:

| Defect | Heuristics only | Heuristics + judge |
|---|---|---|
| Label flip | 10% | **57%** |
| Truncated | 29% | **88%** |
| Off-topic | 87% | **100%** |
| Duplicate | 100% | 100% |
| Low effort | 100% | 100% |

### Calibrating the acceptance rules (same 300 pairs, no extra LLM calls)

Raw judge scores are saved, so rules can be re-tuned instantly
(`scripts/sweep_judge.py`, `scripts/compare_configs.py`).

| Configuration | FPR | Precision | Auto-reject recall | Review queue |
|---|---|---|---|---|
| A: initial rules | 42.9% | 0.44 | 86.7% | 0% |
| B: drop judge "identical" tag | 35.9% | 0.48 | 85.5% | 0% |
| C: B + quality threshold 1.0 | 32.7% | 0.49 | 83.1% | 0% |
| **D: C + label flips → human review** | **16.6%** | **0.60** | 66.3% | 16.3% |

Config D is the current default. It trades some automatic recall for a much cleaner
auto-reject decision and a review queue of ~16% of pairs. Most of the remaining false
positives come from the heuristics (short but appropriate replies like "Me too!"),
which is the next thing to fix.

**Cost:** $0. The judge runs locally (~15 seconds per pair on a MacBook Air, 24 GB).

## How it works

```
HH-RLHF ──► normalize ──► inject known defects ──► heuristics ──► LLM judge (A/B + B/A)
                              (ground truth)       (embeddings)    (cached, local or API)
                                                                          │
                                         ┌────────────────────────────────┤
                                         ▼                                ▼
                                   auto-reject                    human review queue
                                         │
                                         ▼
                 detection metrics (P/R/F1, Wilson CIs, position bias) + drift monitoring
```

1. **Load** (`load.py`): sample HH-RLHF and split each transcript into prompt, chosen, rejected.
2. **Inject** (`inject.py`): corrupt a known fraction of pairs with five defect types:
   `label_flip`, `low_effort`, `off_topic`, `truncated`, `duplicate`.
3. **Heuristics** (`heuristics.py`): length and filler checks, exact duplicates, and
   embedding similarity between the conversation and the chosen response.
4. **LLM judge** (`judge.py`): scores both responses on helpfulness, relevance, effort
   and harmlessness, picks a winner, and runs **twice with the order swapped**. Outputs
   are validated JSON and cached in SQLite. Backends: Ollama (free, local) or the Anthropic API.
5. **Evaluate** (`evaluate.py`): precision, recall and F1 per defect type with Wilson 95%
   intervals, position-inconsistency rate, Cohen's kappa, and batch acceptance.
6. **Drift** (`drift.py`): simulates 12 weekly campaigns (a sloppy annotator joins in
   week 5, guideline confusion in weeks 9–12) and tracks a p-chart, PSI and KS tests.

Acceptance criteria are documented in [`rubric.md`](rubric.md).

## Quickstart

```bash
git clone https://github.com/Uttaprexa/RLHF-Preference-Data-QA-Pipeline
cd RLHF-Preference-Data-QA-Pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q

qa-pipeline prepare --source hh --n 3000        # download and sample HH-RLHF
qa-pipeline run --judge none                    # heuristics only, seconds

# Free local judge: install Ollama from https://ollama.com, then
ollama pull qwen2.5:7b
qa-pipeline run --judge ollama --limit 300 --rate 0.3 --workers 2

python scripts/compare_configs.py               # re-tune rules from saved scores
qa-pipeline drift --judge none                  # drift simulation
```

Other options: `--judge anthropic` (needs `ANTHROPIC_API_KEY`; set `JUDGE_MODEL` to
choose a model), `OLLAMA_MODEL=llama3.1` to swap local models, and `--judge mock` for
offline plumbing tests only (its numbers are not results).

## Project layout

```
src/qa_pipeline/
  load.py        HH-RLHF loading + synthetic demo data
  inject.py      defect injection with ground-truth labels
  heuristics.py  rule-based checks + embedding relevance
  judge.py       LLM judge: prompt, JSON validation, cache, swap-order, acceptance rules
  evaluate.py    detection metrics, confidence intervals, agreement
  drift.py       campaign simulation, p-chart, PSI, KS
  cli.py         command-line entry point
scripts/         threshold sweeps and config comparison on saved scores
tests/           unit tests (offline)
rubric.md        acceptance criteria
```

## Roadmap

- [x] Embedding-based relevance (FPR 36.5% → 10.8%)
- [x] Local LLM judge on 300 real HH-RLHF pairs
- [x] Calibrate acceptance rules; route low-trust verdicts to human review
- [ ] Hand-label 100 pairs; compute judge-vs-human Cohen's kappa
- [ ] Second judge (llama3.1) for inter-judge agreement
- [ ] Context-aware low-effort check (short replies to chit-chat are fine)
- [ ] Drift simulation with the LLM judge in the loop
- [ ] Harder defects: subtly worse or plausible-but-wrong responses
