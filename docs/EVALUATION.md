# Semi-Gold Evaluation

The primary CZE-NEC evaluation target is now
`eval/annotations/czech_batch_200_semi_gold.csv`: 200 human-annotated articles
with manual adjudication on the lowest-agreement/high-impact cases. It is useful
for detecting prompt/model drift and comparing extractor variants. It is still a
semi-gold baseline, not calibrated real-world truth.

For human annotation, use `docs/LABEL_GUIDE.md`.

Useful provenance files:

- `eval/annotations/czenec_batch_200.csv`: source articles.
- `eval/annotations/czech_batch_200_human.md`: first human pass.
- `eval/annotations/czech_batch_200_semi_gold.csv`: primary baseline.
- `eval/annotations/czenec_batch_200_four_model_majority_labeled.csv`: original
  four-model strict-majority weak baseline.
- `eval/reports/czenec_batch_200_four_model_agreement.json`: model agreement
  provenance.
- `eval/reports/czech_batch_200_semi_gold_adjudication.md`: manual
  adjudication notes.

Run an existing prediction file against the semi-gold labels:

```bash
python scripts/evaluate_weak_gold.py \
  --predictions eval/model_outputs/czenec_batch_200_chatgpt.json \
  --output eval/reports/chatgpt_vs_semi_gold.json
```

Run the configured live extractor over the source article CSV, then evaluate it:

```bash
python scripts/evaluate_weak_gold.py \
  --run-extractor \
  --output eval/reports/current_extractor_vs_semi_gold.json
```

For a quick smoke test against a local model, limit the run:

```bash
python scripts/evaluate_weak_gold.py --run-extractor --limit 5
```

The report includes overall, per-group, per-field, and per-topic accuracy/F1,
plus expected-vs-predicted distributions. Blank labels are skipped, though the
semi-gold baseline is complete across all 1,800 article-field decisions.

## Current Baseline Results

Against `czech_batch_200_semi_gold.csv`:

| prediction source | accuracy | macro F1 | support |
|---|---:|---:|---:|
| strict model majority | 0.7541 | 0.6782 | 1476 |
| ChatGPT | 0.7517 | 0.6608 | 1800 |
| Gemma | 0.6667 | 0.5846 | 1800 |
| DeepSeek | 0.6589 | 0.5702 | 1800 |
| Claude | 0.6156 | 0.4918 | 1800 |

Strict majority is useful when it emits a label, but it leaves 324 fields
unlabeled. ChatGPT is the strongest complete single-model baseline in this
batch.

## Known Schema Gaps

- `topic_relevance` is treated as an ingestion/filtering metric. Production
  extraction stores `topic_relevance_score` and an ordinal `topic_relevance`
  label from `article_topics.relevance_score` when it processes database rows.
- Production now separates independent `seg_*_relevance` fields from normalized
  `seg_*` aggregation shares. The evaluator prefers explicit relevance fields
  and falls back to thresholding legacy shares only when relevance fields are
  absent.
- Production extraction now passes article body text into the article prompt
  when a cleaned body is available. The live evaluator uses `body_excerpt` as
  the body input for the CZE-NEC CSV fixture.
- Continuous-to-ordinal thresholds are still heuristics. They should be tuned
  against the semi-gold set before treating the numbers as calibrated model
  quality.
