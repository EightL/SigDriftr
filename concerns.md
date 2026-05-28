# SigDriftr Concerns, Validation Gaps, and Fix Plan

## Bottom Line

SigDriftr is a credible engineering prototype for **media signal drift**: it
tracks how RSS news coverage changes around a topic. It is not yet a validated
measurement system for public sentiment, audience behavior, or market intent.

The main gap is empirical validation. Most current signals are inferred by an
LLM, then reused for drift, brief generation, and feed rewards. Until those
signals are evaluated against human labels or independent outcome data, the
system should be framed as exploratory media analysis.

## Current Validation Snapshot

As of May 28, 2026, the repo has a first no-human weak-gold evaluation baseline:

- Source batch: 200 CZE-NEC articles, balanced across `inflace`, `energie`,
  `zdravotnictvi`, and `politika`.
- Labelers: ChatGPT, DeepSeek, hosted Google Gemma, and Claude.
- Final artifact: `eval/annotations/czenec_batch_200_four_model_majority_labeled.csv`.
- Agreement report: `eval/reports/czenec_batch_200_four_model_agreement.json`.

The strict majority baseline uses only fields where one label receives more than
half of the available model votes. With four model labelers, this means 4/4 or
3/4 agreement. A 2/2 split is intentionally left blank instead of being forced.

Consensus across 1,800 article-field decisions:

- 4/4 agreement: 884 fields.
- 3/4 agreement: 592 fields.
- 2/2 tie: 229 fields.
- 2/1/1 split: 95 fields.

This gives strict-majority labels for 1,476 of 1,800 fields, or 82.0%.
Coverage is strongest for `topic_relevance` and `avoidance_relevance`, and
weakest for `concern_bucket`, `dominant_frame`, and `seg_young_urban_relevance`.

Interpretation: this is a useful weak-gold baseline for prompt/model regression,
schema tuning, and comparing extractors. It is not human ground truth and should
not be used to claim validated real-world accuracy. It does show that the label
schema is workable and that the most unstable parts are the subjective framing
fields.

## Validated Concerns

### 1. RSS-Only Data Measures Coverage, Not Audience Reaction

**Status:** Legitimate.

The ingestion layer uses configured RSS feeds. This is enough to analyze news
coverage, framing, and source mix, but it does not measure what readers believe,
share, search for, buy, or avoid.

**Risk:** A brief can sound like public sentiment changed when only media
coverage changed.

**Solutions:**
- Rename product language to "media signal drift" or "news coverage drift".
- Add a UI/API disclaimer: "This analyzes article content, not audience
  reactions."
- Keep a separate metric namespace if public-facing data is added later:
  `media_signal`, `social_signal`, `search_signal`, `behavior_signal`.
- If moving beyond media, add adapters for social posts, search trends, survey
  data, or first-party dashboard interactions, and report each channel
  separately before merging them.

### 2. LLM-Only Signals Still Have No Human Ground Truth

**Status:** Legitimate, partly mitigated by weak-gold model consensus.

`extraction/llm_client.py` asks a local LLM to produce `concern_level`,
`purchase_intent`, `avoidance_signals`, `dominant_frame`, and segment weights.
Those values are normalized and stored. There is now a 200-article four-model
consensus baseline, but there is still no independent human-labeled dataset
proving that the labels match human judgment.

Research on LLM framing detection supports this concern: LLMs can be useful, but
performance depends on prompt design, domain, ambiguity, and evaluation against
human annotations.

**Risk:** The system may measure LLM scoring tendencies more than real media
signals.

**Solutions:**
- Treat the 200-article majority baseline as weak gold for regression tests, not
  as final truth.
- Use strict majority only; leave 2/2 and 2/1/1 splits unresolved.
- Expand the Czech news evaluation set:
  - 500-1,000 articles across topics and outlet types.
  - Labels for topic relevance, dominant frame, concern bucket,
    purchase/avoidance relevance, and segment relevance.
  - Either human review for a subset, or additional independent model labelers
    with strict consensus and unresolved tie handling.
- Track agreement:
  - accuracy/F1 for categorical fields,
  - per-field consensus rates,
  - per-topic and per-outlet breakdowns.
- Compare LLM prompts/models against the majority baseline and simple baselines
  such as keyword rules, lexicon sentiment, and a small supervised classifier.

### 3. Bandit Reward Is Circular

**Status:** Legitimate.

`ingestion/bandit.py` computes feed reward from extracted signals. Since those
signals are produced by the LLM, the bandit learns which feeds produce stronger
LLM-scored signals, not necessarily which feeds are more useful or accurate.
`warm_start_from_history()` repeats the same issue by replaying historical LLM
signals as rewards.

**Risk:** The feed selector can amplify LLM bias and source bias.

**Solutions:**
- Short term: change the default bandit reward to non-LLM article yield:
  - topic relevance score,
  - count of accepted articles,
  - source freshness,
  - duplicate-adjusted yield.
- Keep signal-strength reward as an experimental mode, not the default.
- Add a fixed-panel collection mode for evaluation so source selection is stable.
- Medium term: use human labels or "article useful for brief" labels as reward.
- Long term: use real user feedback, such as accepted brief claims or analyst
  clicks, if this becomes an interactive product.

### 4. Baselines and Drift Are Too Simple

**Status:** Legitimate, with nuance.

The repo is not purely static: baselines can be seeded and updated with
EMA-style learning. However, the drift layer is still simple. It compares recent
segment averages against baseline values and does not model seasonality, outlet
mix changes, variance changes, or confidence intervals.

**Risk:** Normal source variation or seasonal topic cycles can be reported as
meaningful drift.

**Solutions:**
- Store daily or weekly segment profiles as a time series.
- Replace single baseline values with rolling reference windows:
  - current 7 days vs previous 30 days,
  - current week vs same weekday pattern,
  - fixed historical windows for evaluation.
- Add standard drift metrics:
  - KS/Wasserstein for numeric signals,
  - Jensen-Shannon divergence for frame/source distributions,
  - PSI for binned features,
  - MMD or classifier-based drift for embeddings.
- Return drift decomposition:
  - mean shift,
  - variance shift,
  - frame distribution shift,
  - outlet mix shift,
  - sample-size warning.
- Treat low-volume drift as "insufficient data" or "warming", not a finding.

### 5. Topic Matching and Topic History Are Fragile

**Status:** Legitimate.

`topic_to_domain()` maps topics to broad domains, and `article_topics` tracks
topic matches, but there is no canonical topic model. "energie", "energetika",
"energy", and "ceny energii" can become separate histories.

**Risk:** Baselines fragment, article recall drops, and drift becomes dependent
on exact user wording.

**Solutions:**
- Add a `topics` table with `canonical_topic_id`, display name, aliases, and
  language.
- Normalize user topic strings before collection, extraction, drift, and brief
  generation.
- Use a hybrid approach:
  - curated aliases for important topics,
  - embedding similarity for suggestions,
  - manual confirmation in the UI.
- Store both `raw_topic` and `canonical_topic_id` for auditability.

### 6. Segment Attribution Is Speculative

**Status:** Legitimate.

Segment fields are generated from article text and blended with feed priors.
They are not reader demographics. A segment score currently means "content
appears relevant to this segment", not "this segment reacted this way".

**Risk:** Output can overclaim by saying "families drifted" instead of "coverage
relevant to families shifted".

**Solutions:**
- Rename fields in docs/API responses:
  - `seg_family` -> `seg_family_relevance`
  - `seg_b2b` -> `seg_b2b_relevance`
- Update UI copy to say "segment-relevant coverage".
- Validate segment relevance labels in the human dataset.
- If audience data becomes available, train a separate segment relevance model
  and use LLM output only as one feature.

### 7. Source Mix Can Create False Drift

**Status:** Legitimate.

The bandit changes which feeds are selected. `delta/mapper.py` aggregates
signals across all matching articles without normalizing to a fixed outlet
panel. If one run has mostly business outlets and another has more tabloid
coverage, the drift may be source composition, not topic change.

**Risk:** The system confuses outlet mix drift with signal drift.

**Solutions:**
- Return `article_count_by_outlet` and `outlet_mix_drift` in `/drift`.
- Compute Jensen-Shannon divergence between current and baseline outlet
  distributions.
- Add source-normalized drift:
  - compute profiles per outlet,
  - combine them using fixed panel weights,
  - report both raw and source-normalized values.
- Use fixed source panels in evaluation runs.

### 8. Confidence Scores Are Not Calibrated

**Status:** Legitimate.

The code has confidence-like values based on article count and baseline quality,
but they are operational heuristics. They do not yet mean "this prediction is
correct X% of the time."

**Risk:** Consumers may interpret confidence as statistical reliability.

**Solutions:**
- Rename current values to `coverage_confidence` or `readiness_score`.
- Add calibrated model confidence only after evaluation data exists.
- Build reliability diagrams on the labeled set.
- Use Platt scaling or isotonic regression if raw scores can be mapped to
  empirical correctness.
- Document the meaning of each confidence score explicitly.

### 9. Cluster Validation Is Weak

**Status:** Legitimate, with correction.

The repo does contain UMAP/HDBSCAN clustering in `clustering/`. The concern is
not absence of code; it is absence of validation. UMAP/HDBSCAN can produce
plausible clusters that are unstable under parameter changes, sample changes,
or duplicate articles.

**Risk:** Storyline clusters may look meaningful while being unstable.

**Solutions:**
- Add bootstrap stability tests:
  - sample 80% of articles repeatedly,
  - recluster,
  - measure adjusted Rand index or normalized mutual information.
- Add cluster coherence metrics:
  - average pairwise embedding similarity,
  - centroid distance,
  - exemplar diversity.
- Add duplicate handling before clustering:
  - exact URL/canonical URL dedupe,
  - near-duplicate title/body checks,
  - embedding similarity threshold.
- Evaluate whether cluster-aware briefs are actually better than article-level
  briefs with human ratings.

### 10. No Prompt or Model Regression Suite

**Status:** Legitimate.

There are unit tests for parsing, fallbacks, and pipeline behavior, but no
golden set that detects when a prompt/model change shifts signal distributions
or brief claims.

**Risk:** Changing a prompt or Ollama model can silently change all downstream
drift.

**Solutions:**
- Add a small `eval/golden_articles.jsonl` fixture.
- Store expected output ranges rather than exact LLM outputs.
- Track distribution metrics:
  - frame frequencies,
  - signal means/stddev,
  - invalid JSON rate,
  - fallback rate.
- Fail CI only on large regressions; report softer eval metrics as artifacts.
- Version prompts and model names in stored `raw_json`.

### 11. Brief Evidence Linkage Is Incomplete

**Status:** Partially legitimate.

The newer cluster-aware path stores support fields such as cited track/article
IDs, but the system does not yet validate that every generated claim is grounded
in those IDs or that numeric claims match computed values.

**Risk:** Briefs can be fluent but unsupported.

**Solutions:**
- Require each generated claim/hypothesis to include `article_ids` or
  `cluster_ids`.
- Add a post-generation validator:
  - every claim has evidence,
  - numeric deltas match source data within tolerance,
  - cited articles belong to the topic/scope,
  - cited clusters exist in the current run.
- Add an optional second-pass support checker that labels claims as
  `supported`, `partially_supported`, or `unsupported`.
- Show unsupported or partially supported claims as warnings, not conclusions.

### 12. Dataset and Labeling Plan Is Partly Addressed

**Status:** Partly addressed.

The repo now has a compact 200-article weak-gold dataset derived from four model
labelers. This is enough to detect large prompt/model regressions and to compare
extractor variants. It is not enough to establish calibrated model quality.

**Proposed schema:**

```csv
article_id,canonical_topic_id,raw_topic,outlet,published_at,title,summary,body_excerpt,
topic_relevance,dominant_frame,concern_bucket,purchase_relevance,
avoidance_relevance,seg_young_urban_relevance,seg_family_relevance,
seg_senior_relevance,seg_b2b_relevance,annotator_id,notes
```

**Label scale:**
- topic relevance: `0=irrelevant`, `1=related`, `2=core`
- dominant frame: `fear`, `opportunity`, `conflict`, `neutral`, `other`
- concern bucket: `low`, `medium`, `high`
- purchase/avoidance relevance: `0=none`, `1=weak`, `2=strong`
- segment relevance: `0=not relevant`, `1=somewhat`, `2=strong`

## Prioritized Roadmap

### Phase 1: Reframe and Stop Overclaiming

- Use "media signal drift" everywhere.
- Add disclaimers to README, API docs, and UI.
- Rename segment fields in docs to segment relevance.
- Report source mix alongside drift.

### Phase 2: Build the First Evaluation Set

- Use the 200-article four-model strict-majority set as the initial regression
  baseline.
- Add a small reusable evaluator that compares current extractor outputs against
  the strict-majority labels.
- Expand toward 500-1,000 articles only after the schema and prompts stabilize.
- Track per-field and per-topic agreement, not only aggregate accuracy.

### Phase 3: Fix the Circular Feedback Loops

- Make article relevance yield the default bandit reward.
- Add fixed-panel runs for evaluation.
- Keep LLM-signal reward behind an explicit experimental flag.

### Phase 4: Improve Drift and Baselines

- Add rolling reference windows.
- Add distribution-based drift metrics.
- Add source-normalized drift.
- Separate readiness/coverage confidence from calibrated correctness.

### Phase 5: Validate Clusters and Briefs

- Add cluster stability/coherence reports.
- Add near-duplicate detection before clustering.
- Add claim-to-evidence validation for generated briefs.
- Run human comparison of article-level vs cluster-aware briefs.

## Portfolio Framing

Good:

> Experimental FastAPI pipeline for media signal drift analysis over RSS news,
> using local LLM extraction, embeddings, clustering, and evidence-linked briefs.

Avoid:

> Validated public sentiment detector.

Until the evaluation set and calibration work exist, the honest claim is that
SigDriftr is a well-structured prototype for analyzing media coverage changes,
not a proven measurement instrument.
