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

CI also runs the checked-in ChatGPT prediction file with minimum thresholds:

```bash
python scripts/evaluate_weak_gold.py \
  --predictions eval/model_outputs/czenec_batch_200_chatgpt.json \
  --min-accuracy 0.75 \
  --min-macro-f1 0.65
```

## Brief Regression

The article semi-gold file evaluates extractor labels, not generated briefs. For
brief regression, use frozen brief golden records with expected structured
`ResearchBrief` fields and support expectations:

```bash
python scripts/evaluate_briefs.py \
  --goldens eval/briefs/golden_briefs.jsonl \
  --predictions eval/model_outputs/current_brief_predictions.json \
  --output eval/reports/current_brief_eval.json
```

The brief report scores exact-match fields (`status`, `most_affected_segment`,
`drift_type`, `alert_level`), support coverage (`required_track_ids`,
`forbidden_track_ids`, `required_article_ids`), expected hypothesis segments,
fallback rate, generation-mode distribution, and internal validation warnings.
Goldens may embed a frozen `cluster_snapshot` so the current generator can be
run without depending on the latest database state.

## Cluster Stability

Storyline clustering is validated separately from article-level extraction. The
cluster stage now removes conservative near-duplicates before UMAP/HDBSCAN, then
the stability evaluator reclusters bootstrap samples of that deduped input and
reports ARI, NMI, AMI, noise-rate, and per-cluster survival summaries:

```bash
python scripts/evaluate_cluster_stability.py \
  --topic energie \
  --window-hours 168 \
  --bootstrap-samples 25 \
  --output eval/reports/cluster_stability_energie.json
```

Use this as an analyst-facing quality report, not as calibrated correctness.
Low stability or high duplicate removal means cluster-aware briefs should be
treated as exploratory.

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
