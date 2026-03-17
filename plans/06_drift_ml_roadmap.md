# 06_drift_ml_roadmap

Goal: Evolve SigDriftr from a hardcoded-demo into a self-learning, ML-driven drift engine with trustworthy signals and efficient collection.

Fits in pipeline: Cross-cutting upgrade that touches segments (02 LLM extraction), 03 delta engine, and 01 ingestion bandit/feeds. Implement after 03_segment_delta and 04_brief_generator plans.

Inputs:
- Existing SQLite schema (`articles`, `signals`, `segment_profiles`, `baselines`).
- Current extraction outputs (behavioral JSON from Ollama / API).
- Current segment profiles and drift outputs.

Outputs:
- Self-learning baselines with explicit confidence and sample counts.
- Topic-aware, domain-specific signal schema (which fields matter for which topic).
- LinUCB-based outlet scheduler that prioritizes high-signal feeds per topic.
- Updated API contract exposing confidence + baseline status to the brief.

## 1. Baseline Learning & Confidence Layer

1. Add baseline metadata
   - Extend `baselines` table in `db/init.py` with:
     - `sample_count INTEGER NOT NULL DEFAULT 0` — number of articles that have contributed to the learned baseline.
     - `is_learned INTEGER NOT NULL DEFAULT 0` — 0 = seed-only, 1 = contains blended real data.
   - Migration strategy: safe ALTER TABLE sequence with defaults so existing DBs keep working.

2. Wire `update_baseline_from_profile()` into a call path
   - Where: in the code path that recomputes segment profiles (in `delta/mapper.py` or wherever segment profiles are persisted) right after profiles are stored.
   - Logic:
     - For each (topic, segment) profile:
       - If `article_count < MIN_ARTICLES_FOR_BASELINE` (config, e.g. 30), skip.
       - Otherwise:
         - Call `update_baseline_from_profile(topic, segment, profile)`.
         - Increment `sample_count` in `baselines`.
         - Set `is_learned = 1` once at least one update has been applied.

3. Make drift aware of baseline quality
   - In `delta/engine.compute_drift`:
     - Load `sample_count` and `is_learned` along with baseline values.
     - Attach to result object per segment:
       - `baseline_sample_count`.
       - `baseline_is_learned`.
       - `baseline_age_days` (derived from `updated_at`).
     - Compute `confidence` score:
       - Example formula (configurable):
         - `coverage = min(1.0, article_count / 50)`.
         - `baseline_quality = 0.3 if not is_learned else min(1.0, sample_count / 50) + 0.3`.
         - `confidence = clamp(0.0, 1.0, 0.5 * coverage + 0.5 * baseline_quality)`.
       - Store `confidence` on each `compute_drift` result.

4. API exposure and brief integration
   - Update `/signals` and `/calibration/{topic}/{segment}` responses to include:
     - `confidence`.
     - `baseline_is_learned`.
     - `baseline_sample_count`.
   - Update brief generator input model to accept these fields.
   - In brief prompt:
     - Include explicit language about confidence level per segment.
     - Add a guardrail: if `confidence < MIN_BRIEF_CONFIDENCE` (e.g. 0.4), brief must phrase findings as hypotheses and emit a warning.

5. Tests
   - Unit tests for:
     - Baseline migration: new columns exist with correct defaults on fresh DB.
     - `update_baseline_from_profile` blending and metadata updates.
     - Confidence formula edge cases (no data, only seeded baseline, large sample count).
   - Integration test:
     - Seed with dummy articles and signals for one topic.
     - Run segment profile computation twice.
     - Assert that drift results change from low to higher confidence and `is_learned` flips to 1.

## 2. Topic-Aware Signal Schema

1. Define domains and schemas
   - Add a config module: `config/domains.py`:
     - Domains: `commerce`, `civic`, `health`, `generic`.
     - For each domain, define which behavioral fields are relevant:
       - `commerce`: concern, purchase_intent, avoidance.
       - `civic`: concern, avoidance.
       - `health`: concern, avoidance.
       - `generic`: concern only.

2. Topic → domain classifier
   - Implement a small helper in `extraction/extractor.py` or new `domains.py`:
     - Input: `topic` string.
     - Simple rules first: keyword-based mapping for Czech core topics (energie → `commerce/health`, politika → `civic`, zdravi → `health`, etc.).
     - Optionally: one-shot LLM classification when rules do not match; cache result in memory (and optionally a small `topic_domains` table in SQLite).

3. Adapt extraction prompt and normalization
   - In `llm_client.PROMPT_TEMPLATE`:
     - Add `Topic: {topic}` and a short instruction: “If the topic is not about products or services, keep `purchase_intent` very close to 0.0 and focus on concern/avoidance.”
   - In `extract_signals(title, summary, affinity_tag, topic)` (add `topic` param):
     - After `_normalize_signals`, apply domain mask:
       - For fields not relevant in the topic’s domain, set them to 0.0 *and* mark them as `irrelevant` in an auxiliary structure.

4. Make drift magnitude domain-aware
   - In `delta/mapper.py` and `delta/engine.compute_drift`:
     - Introduce `SIGNAL_WEIGHTS` per domain, e.g.:
       - `commerce`: concern=0.4, purchase_intent=0.4, avoidance=0.2.
       - `civic`: concern=0.6, purchase_intent=0.0, avoidance=0.4.
     - Replace raw L1 sum with weighted sum:
       - `drift_magnitude = sum(weight[field] * abs(delta[field]) ...)`.
   - Pass domain into drift computation (propagate alongside topic).

5. API + brief changes
   - Include `domain` and `relevant_fields` in `/signals` & `/calibration` responses.
   - In brief generator:
     - When ranking segments, use `drift_magnitude` and `confidence`, not raw unweighted deltas.
     - Mention domain and relevant fields in the narrative (e.g., “For civic topic ‘politika’, we focus on concern and avoidance, ignoring purchase intent”).

6. Tests
   - Unit tests:
     - Topic → domain mapping for core Czech keywords.
     - Domain masking sets irrelevant fields to zero and they no longer contribute to drift magnitude.
     - Weighted drift magnitude matches expected values.
   - Regression test:
     - Construct a fake politics profile with high `purchase_intent` but normal concern/avoidance.
     - Assert drift magnitude remains low and brief does not mention purchase behavior.

## 3. LinUCB Bandit for Crawl Scheduling

1. Define bandit problem
   - Arms = RSS feeds from `config/feeds.FEEDS`.
   - Context features per (topic, feed):
     - One-hot or embedding of feed `affinity_tag`.
     - One-hot bucket for hour-of-day.
     - Optional: topic embedding bucket (precomputed offline or using your existing multilingual MiniLM model).
   - Reward for an article:
     - Start simple: `reward = drift_contribution_norm` = normalized change in segment profile caused by signals from this article (bounded to [0,1]).
     - If too complex initially, fallback to `reward = 1` if article passes relevance filter + has non-trivial concern/avoidance, else `0`.

2. Implement LinUCB core
   - New module: `ingestion/bandit.py`.
   - Implement standard disjoint LinUCB (per Li et al. 2010):
     - Maintain A (d x d) and b (d x 1) per arm (feed).
     - For each decision:
       - Build context vector x for (topic, feed, time).
       - Compute `theta = A^{-1} b`.
       - Compute UCB score: `p = theta^T x + alpha * sqrt(x^T A^{-1} x)`.
     - Select top-K feeds per topic per crawl tick based on p.
   - Persist A, b per feed to SQLite or a small JSON under `data/` so learning survives restarts.

3. Integrate into crawler
   - Modify `ingestion/crawler.crawl(topic)`:
     - Instead of iterating all `FEEDS`, ask bandit for `select_feeds(topic, now, k)`.
     - Crawl only those feeds.
   - After extraction and drift update, compute per-article reward (or per-feed aggregated reward) and call `bandit.update(feed_id, context, reward)`.

4. Offline warm-start option
   - Add a CLI script that replays past articles from `articles` + `signals` to generate initial bandit parameters, if you have enough historical data.

5. Tests
   - Unit tests:
     - Bandit score calculation and update steps (compare against a small hand-calculated example).
   - Integration test:
     - Simulate 3 feeds with different synthetic rewards.
     - Ensure selection probability shifts toward high-reward feeds over time.

## 4. Async Crawling & Inference Efficiency

1. Async RSS fetching
   - Refactor `ingestion/crawler.py`:
     - Replace direct `feedparser.parse(url)` calls with `aiohttp` + `asyncio.gather` to fetch multiple feeds concurrently.
     - Use a small bounded semaphore to avoid hammering sources.
     - Parse responses with `feedparser.parse` from in-memory bytes.

2. Embedding efficiency
   - Cache topic embeddings separately from the sentence-transformer model instance:
     - `_get_embedder()` remains lru-cached for the model.
     - Add `_get_topic_embedding(topic)` with its own cache so you only encode the topic once per process.
   - Optionally, pre-filter by simple keyword, then only run embeddings if keyword filter passes.

3. LLM inference parallelism
   - If using Ollama locally:
     - Add simple rate limiting / queueing to avoid overloading the GPU/CPU.
   - Make extraction calls idempotent per article_id so retried jobs do not double-count.

4. Tests
   - Ensure crawl still respects relevance filter and DB writes under concurrency.
   - Use a fake HTTP server or VCR-style fixtures to simulate RSS sources.

## 5. Brief Confidence & NER Enrichment

1. Wire confidence into brief models
   - Extend `brief/models.py` response and internal DTOs with:
     - `segment_confidence` per segment.
     - `baseline_is_learned`, `baseline_sample_count`.
   - Update `brief/prompt.py` to:
     - Include confidence scores and instruct the LLM to qualify statements accordingly ("high confidence", "medium confidence", "exploratory").

2. Add Czech NER layer (optional but high value)
   - Use `spacy.load("cs_core_news_sm")` in a small `extraction/entities.py` helper.
   - For each article:
     - Extract `PERSON`, `ORG`, `GPE`.
     - Store a compact, normalized list of entities in a new `article_entities` table or as JSON in `signals.raw_json`.
   - In the brief prompt:
     - Summarize top entities driving drift per segment (by counting exposures).

3. Tests
   - Snapshot-test brief outputs for:
     - Low vs high confidence cases.
     - Presence/absence of entities.

## 6. One Sanity Test for the Whole Upgrade

- End-to-end scenario:
  1. Start with empty DB.
  2. Run `/collect?topic=inflace` and `/extract?topic=inflace` periodically to accumulate ~100 articles.
  3. Verify:
     - Baselines for (`inflace`, each segment) transition from `is_learned=0` to `is_learned=1` and `sample_count` increases.
     - `confidence` for drift results increases over time.
     - Bandit gradually focuses on `e15`, `irozhlas_eko` for `inflace` if their articles yield higher rewards.
     - Brief for `inflace` includes confidence qualifiers and does not mention purchase behavior if domain is civic/health.
