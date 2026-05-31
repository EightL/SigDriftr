# SigDriftr Architecture - Complete Technical Guide

## Overview

SigDriftr is a modular signal extraction and drift detection pipeline designed to analyze Czech media trends. The system processes RSS articles through a series of transformations:

```
RSS Feeds → Article Collection → Semantic Filtering → Storage
    ↓
LLM Signal Extraction → Coverage Framing Metrics → Named Entity Enrichment
    ↓
Segment Aggregation → Baseline Comparison → Drift Detection
    ↓
Research Brief Generation → Dashboard Visualization
```

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    SigDriftr Complete Pipeline                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [1] Ingestion (ingestion/)                                               │
│      ├─ crawler.py: Feed selection + fetching                             │
│      ├─ bandit.py: LinUCB contextual bandit                               │
│      └─→ Articles stored in SQLite                                        │
│           ↓                                                                 │
│  [2] Extraction (extraction/)                                             │
│      ├─ extractor.py: Article processing                                  │
│      ├─ llm_client.py: Ollama integration (qwen2.5:7b, gemma3:1b)        │
│      ├─ entities.py: Optional spaCy enrichment                            │
│      └─→ Signals + Entities stored in SQLite                              │
│           ↓                                                                 │
│  [3] Delta (delta/)                                                       │
│      ├─ mapper.py: Per-segment aggregation                                │
│      ├─ engine.py: Drift calculation + alert levels                       │
│      ├─ seeder.py: Baseline initialization                                │
│      └─→ Segment Profiles + Drift Data stored                             │
│           ↓                                                                 │
│  [4] Brief (brief/)                                                       │
│      ├─ generator.py: Orchestration + 30-min caching                      │
│      ├─ prompt.py: Evidence-based prompting                               │
│      ├─ models.py: Pydantic schemas                                       │
│      └─→ Research Brief (always 200 OK with fallback)                     │
│           ↓                                                                 │
│  [5] API (api/)                                                           │
│      ├─ routes/: 8 HTTP endpoints                                         │
│      ├─ scheduler.py: Background job scaffolding                          │
│      └─→ HTTP responses + Dashboard UI                                    │
│                                                                             │
│  [6] Database (db/)                                                       │
│      └─ init.py: SQLite schema, WAL mode, per-thread connections         │
│         - 6 tables: articles, signals, article_entities,                 │
│           segment_profiles, baselines, bandit_state                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘

External Dependencies:
• Ollama (http://localhost:11434) - Local LLM inference
• spaCy (optional) - Czech NER model (cs_core_news_sm)
• sentence-transformers - Semantic similarity (paraphrase-multilingual-MiniLM-L12-v2)
```

## Technology Stack

- **Language:** Python 3.11+
- **Web Framework:** FastAPI 0.100+
- **Database:** SQLite (WAL mode, per-thread connection pooling)
- **LLM Integration:** Ollama (local inference)
- **Semantic Search:** sentence-transformers (multilingual embeddings)
- **NER (optional):** spaCy + cs_core_news_sm
- **Async:** asyncio + aiofiles
- **Validation:** Pydantic v2

---

## Module Details

### 1. Ingestion (`ingestion/`)

**Purpose:** Collect relevant articles from eligible RSS feeds using either adaptive or fixed source selection.

**Input:** Topic string (e.g., "energie", "zdravotnictvi")
**Output:** Articles stored in SQLite, bandit state updated

**Components:**
- **crawler.py**: selects feeds → fetches concurrently → two-pass filters → stores with INSERT OR IGNORE → records collection telemetry.
- **bandit.py**: LinUCB contextual bandit for adaptive feed selection. Maintains per-feed arm parameters (θ, A, b). Default rewards come from non-LLM article yield, relevance, and duplicate-adjusted collection results.
- **source_panels.py**: fixed outlet panels for reproducible evaluation and demos.

**Filtering Strategy (Two-Pass):**
1. Direct match: "energie" in title OR summary → ACCEPT
2. Semantic: embedding_similarity(article, topic) > 0.7 → ACCEPT
3. Otherwise: REJECT

Fallback: If embedding model unavailable, only direct match applies (silent degradation).

**Feed Configuration (config/feeds.py):**
- irozhlas, ct24, wave, idnes, novinky, e15, blesk, maminka (8 Czech outlets)
- Per-feed segment priors: e.g., maminka→70% family, blesk→60% young_urban
- Used for blending with segment weights from extraction

**Design Decisions:**
- **Why LinUCB?** Balances exploration/exploitation mathematically. Adapts to topic context.
- **Why fixed panels?** Evaluation and demos need stable outlet composition, so `collection_mode=fixed_panel` bypasses adaptive selection.
- **Why yield rewards?** The default bandit policy should learn useful article collection, not amplify LLM-scored signal strength.
- **Why two-pass?** Direct match is fast; semantic fallback catches related articles.
- **Why feed affinity priors?** Improve segment weight accuracy by incorporating feed bias.

**Extension Points:**
- Add new feeds: Edit `config/feeds.py`
- Change filtering logic: Modify `direct_match_filter()` or `semantic_filter()`
- Replace LinUCB: Modify `bandit.py` (Thompson sampling, e-greedy, etc.)

---

### 2. Extraction (`extraction/`)

**Purpose:** Extract article-level coverage framing signals from article text using local LLM inference.

**Input:** Articles without signals
**Output:** Signals (8-field schema) + optional named entities

**Components:**
- **extractor.py**: Queries unprocessed articles → calls LLM per article → normalizes → stores
- **llm_client.py**: Ollama integration, model fallback, post-processing
- **entities.py**: Optional spaCy Czech NER enrichment

**Signal Schema (8 fields):**
```json
{
  "concern_level": 0.0-1.0,        # Concern/risk framing intensity
  "purchase_intent": 0.0-1.0,      # Article-level purchase/action framing
  "avoidance_signals": 0.0-1.0,    # Avoidance or risk-avoidance framing
  "dominant_frame": "fear|opportunity|conflict|neutral",
  "seg_young_urban_relevance": 0.0-1.0,  # Independent relevance to young_urban
  "seg_family_relevance": 0.0-1.0,       # Independent relevance to family
  "seg_senior_relevance": 0.0-1.0,       # Independent relevance to senior
  "seg_b2b_relevance": 0.0-1.0,          # Independent relevance to b2b
  "seg_young_urban": 0.0-1.0,            # Normalized aggregation share
  "seg_family": 0.0-1.0,                 # Normalized aggregation share
  "seg_senior": 0.0-1.0,                 # Normalized aggregation share
  "seg_b2b": 0.0-1.0                    # Normalized aggregation share
}
```

**LLM Models:**
- Primary: `qwen2.5:7b-instruct` (better quality)
- Fallback: `gemma3:1b` (faster)
- Timeout: 30 seconds per article
- Retry: 2 attempts with exponential backoff

**Post-Processing (Normalization):**
1. Clamp numeric fields to [0, 1]
2. Restrict dominant_frame to enum values
3. Keep independent segment relevance scores as `seg_*_relevance`
4. Derive softmax segment shares: sum(seg_*) = 1.0
5. Blend segment shares with feed priors: final = 0.7 * derived_share + 0.3 * feed_prior

**Optional Enrichment (spaCy):**
- Extract named entities (PERSON, ORG, LOC, etc.)
- Store in article_entities table
- Safe fallback if spaCy unavailable

**Design Decisions:**
- **Why two models?** Better quality (qwen) vs. faster fallback (gemma).
- **Why normalize with softmax?** Makes segment weights comparable across articles.
- **Why blend with feed priors?** Improves accuracy; some feeds skew toward certain segments.
- **Why optional spaCy?** Nice-to-have; system works without it.

**Extension Points:**
- Change LLM model: Modify `llm_client.py`
- Add signal fields: Update schema + LLM prompt
- Replace spaCy: Modify `entities.py`

---

### 3. Delta (`delta/`)

**Purpose:** Aggregate signals by segment relevance group and detect drift from baseline profiles.

**Input:** Signals from SQLite
**Output:** Segment profiles, drift data, alert levels

**Components:**
- **mapper.py**: Weighted aggregation per segment
- **engine.py**: Baseline comparison, drift magnitude, alert levels
- **seeder.py**: Baseline initialization (first run = baseline)

**Aggregation Logic:**
For each segment (young_urban, family, senior, b2b):
1. Select articles where `seg_* > 0.5` (dominant normalized segment share)
2. Compute weighted average of:
   - concern_level_avg = Σ(concern_level * seg_*) / Σ(seg_*)
   - purchase_intent_avg = Σ(purchase_intent * seg_*) / Σ(seg_*)
   - avoidance_signals_avg = Σ(avoidance * seg_*) / Σ(seg_*)
3. Dominant frame: weighted vote among article frames
4. Store in segment_profiles table

**Drift Calculation:**
```
drift_magnitude = 0.333 * |Δ concern| + 0.333 * |Δ purchase| + 0.334 * |Δ avoidance|

alert_level:
  if drift < 0.20:   none
  if drift < 0.45:   mild
  else:              strong
```

**Frame Shift:** Detected if current dominant_frame ≠ baseline dominant_frame

**Source Mix and Normalized Drift:**
- `/drift/{topic}` returns current vs previous-window outlet composition and a
  Jensen-Shannon divergence score so analysts can see when source composition
  moved materially.
- Each segment also includes `source_normalized`, a bounded fixed-panel view.
  It computes per-outlet segment profiles for the current window, combines
  observed fixed-panel outlets with equal weights, then compares that normalized
  current profile to the existing segment baseline.
- This is deliberately labeled as fixed-panel normalization, not causal
  correction. It helps flag cases where raw drift may be driven by outlet mix,
  while preserving the original raw drift fields for backward compatibility.

**Design Decisions:**
- **Why weighted averaging?** Articles aren't equally relevant to each segment.
- **Why separate baselines per segment?** Segments have different signal distributions.
- **Why seeded baselines?** Historical learning not implemented; first run establishes reference.
- **Why 0.20/0.45 thresholds?** Domain expertise from Lakmoos; calibrate from data later.
- **Why source-normalized drift?** It makes source-composition sensitivity
  visible without hiding the raw media signal.

**Extension Points:**
- Change analysis window: Edit `days_back` parameter (default 7)
- Add new segment: Modify `config/feeds.py` + schemas
- Implement baseline learning: Modify `seeder.py`
- Change alert thresholds: Modify `engine.py` logic

---

### 4. Brief (`brief/`)

**Purpose:** Generate a concise research brief summarizing drift findings with testable hypotheses.

**Input:** Drift data + representative articles
**Output:** Research brief with summary + 3 hypotheses

**Components:**
- **generator.py**: Orchestration, caching (30 min), fallback generation
- **prompt.py**: Evidence-based prompt construction
- **models.py**: Pydantic schemas

**Generation Pipeline:**
1. Fetch drift data (compute or retrieve from cache)
2. Rank segments by drift magnitude
3. Select representative articles per segment (3-5 with high signal density)
4. Build prompt with drift evidence + article snippets
5. Call Ollama (qwen2.5:7b) with timeout
6. If success: parse JSON, cache for 30 min, return
7. If failure: generate deterministic fallback brief

**Fallback Brief (if LLM fails):**
- Guaranteed 200 OK response
- Summarizes drift per segment
- Lists article titles as evidence
- Deterministic (reproducible)

**Caching:**
- In-memory dict with timestamp-based TTL (30 min)
- Per-(topic, days_back) key
- No Redis dependency (demo phase)

**Design Decisions:**
- **Why in-memory cache?** Simple, sufficient for demo; Redis-ready for scaling.
- **Why fallback brief?** Guarantees API always returns 200 OK.
- **Why deterministic fallback?** Reproducible behavior; no randomness.

**Extension Points:**
- Replace in-memory cache with Redis: Modify `generator.py`
- Change fallback brief logic: Modify `generate_fallback_brief()`
- Update prompt template: Modify `build_prompt()`
- Use different LLM: Modify Ollama call in `generator.py`

---

### 5. Database (`db/`)

**Purpose:** Manage SQLite schema, connection pooling, data persistence.

**Core schema:**

**articles** (RSS article metadata)
- id (PK), url/canonical_url (UNIQUE where available), topic, canonical_topic_id,
  title, summary, body, outlet, country, language, published_at, fetched_at
- Stores: 1 row per unique article

**topics / topic_aliases** (Canonical topic catalog)
- topics: canonical_topic_id (PK), display_name, domain, status, merged_into
- topic_aliases: canonical_topic_id (FK), raw_topic, normalized_topic, language,
  source
- Stores: curated and user-created aliases, e.g. `energie` and `energy` -> `energy`

**article_topics** (Many-to-many topic links)
- article_id (FK), topic, raw_topic, canonical_topic_id, relevance_score, matched_at
- Stores: article-topic matches without duplicating article rows

**signals** (1:1 with articles)
- id (PK), article_id (FK), concern_level, purchase_intent, avoidance_signals, dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b, seg_young_urban_relevance, seg_family_relevance, seg_senior_relevance, seg_b2b_relevance, raw_json (full LLM response), extracted_at
- Stores: 1 row per article signal

**article_entities** (Optional, 1:N with articles)
- id (PK), article_id (FK), entity_text, entity_label (PERSON|ORG|LOC|etc), extracted_at
- Stores: Named entities per article (if spaCy available)

**segment_profiles** (Per-segment aggregates)
- id (PK), topic, canonical_topic_id, segment, concern_level_avg,
  purchase_intent_avg, avoidance_signals_avg, dominant_frame, article_count,
  computed_at, time_window
- Stores: Aggregated signals per segment per time window

**baselines** (Reference for drift calculation)
- id (PK), topic, canonical_topic_id, segment, concern_level, purchase_intent,
  avoidance_signals, dominant_frame, sample_count, is_learned, updated_at
- Stores: 1 row per canonical-topic/segment pair for learned or seeded baselines

**bandit_state** (LinUCB arm parameters)
- id (PK), feed_id (UNIQUE), theta (JSON), A (JSON), b (JSON), pulls (int), last_updated
- Stores: 1 row per feed

**Indices:**
- article_topics.canonical_topic_id (topic filtering)
- articles.url_hash (deduplication)
- signals.article_id (1:1 join)
- segment_profiles.(canonical_topic_id, segment) (drift queries)
- baselines.(canonical_topic_id, segment) (baseline lookup)
- cluster_runs/cluster_tracks/cluster_drift_runs canonical scope indexes

**Configuration:**
- WAL mode: Readers don't block writers
- Per-thread connections: Each asyncio task gets its own connection
- Connection pooling: ~10 connections maintained

**Design Decisions:**
- **Why SQLite?** Zero deployment, ACID, WAL concurrency. Hits limits ~1M articles.
- **Why per-thread connections?** Avoids lock contention in asyncio.
- **Why WAL mode?** Enables concurrent reads while writing.
- **Why large raw_json?** Debug visibility, supports future enrichment.

**Extension Points:**
- Migrate to PostgreSQL: Create new `db/postgres.py`
- Add full-text search: SQLite FTS5 module
- Add audit logs: New table for change tracking

---

### 6. API (`api/`)

**Purpose:** Expose pipeline operations via HTTP REST endpoints.

**Endpoints (8 total):**

| Method | Path | Purpose |
|--------|------|----------|
| POST | `/collect?topic=...` | Crawl feeds, store articles |
| POST | `/extract?topic=...` | Extract signals from articles |
| GET | `/signals?topic=...&limit=20&offset=0` | Get article-level signals |
| GET | `/calibration/{topic}/{segment}?days_back=7` | Get current segment profile |
| GET | `/drift/{topic}?days_back=7` | Get drift vs baseline |
| GET | `/brief/{topic}` | Get research brief |
| GET | `/health` | System health check |
| GET | `/ui` | Demo dashboard (HTML) |

For complete endpoint documentation with examples, see [API.md](API.md).

**Route Structure:**
- `routes/collect.py` - POST /collect
- `routes/extract.py` - POST /extract
- `routes/signals.py` - GET /signals
- `routes/calibration.py` - GET /calibration
- `routes/brief.py` - GET /brief
- `routes/health.py` - GET /health
- `routes/history.py` - GET /history (internal)

**Key Patterns:**
- All routes use FastAPI dependency injection for DB connection
- Error handling: Try/catch with meaningful error messages
- Request validation: Pydantic models
- Response caching: Redis-ready (scaffolding)
- Always-on principle: Even partial failures return meaningful responses

---

## Complete Data Flow Example: Topic "energie"

```
User Input: topic="energie"
    ↓
[0] Resolve topic
    ├─ raw topic = "energie"
    └─ canonical_topic_id = "energy"
    ↓
[1] POST /collect?topic=energie
    ├─ Select feeds via collection_mode=bandit|fixed_panel|all
    ├─ Fetch feeds concurrently (timeout: 10 sec per feed)
    ├─ Filter articles:
    │  ├─ Direct match: "energie" in title/summary
    │  └─ Semantic: embedding_similarity > 0.7
    ├─ Deduplicate by URL hash (SHA-256)
    ├─ Insert 24 articles into SQLite
    ├─ Link articles through article_topics(raw_topic="energie", canonical_topic_id="energy")
    ├─ Record collection_runs + collection_feed_stats
    └─ Response includes selected_feeds, accepted, duplicates, and reward_mode
    ↓
[2] POST /extract?topic=energie
    ├─ Query 24 articles without signals
    ├─ For each article:
    │  ├─ Title: "Nová solární farma v Česku"
    │  ├─ Summary: "Největší solární projekt..."
    │  ├─ Call Ollama (qwen2.5:7b) with extraction prompt
    │  ├─ Normalize: clamp→[0,1], softmax segments
    │  ├─ Optional: Extract entities with spaCy
    │  └─ Store in signals table
    └─ Response: {"articles_processed": 24, "signals_stored": 24}
    ↓
[3] GET /drift/energie?days_back=7
    ├─ Compute segment profiles:
    │  ├─ young_urban:
    │  │  ├─ Weighted avg concern_level = 0.52 (24 articles)
    │  │  ├─ Weighted avg purchase_intent = 0.41
    │  │  └─ Dominant frame = "opportunity" (16 articles vote)
    │  ├─ family: {...}
    │  ├─ senior: {...}
    │  └─ b2b: {...}
    ├─ Seed baselines for canonical topic "energy" (first run): baseline = current_profile
    ├─ Compare (subsequent runs):
    │  └─ young_urban:
    │     ├─ Current concern: 0.52, Baseline: 0.45, Δ: +0.07
    │     ├─ Frame shift: "neutral" → "opportunity" ✓
    │     └─ Drift magnitude: 0.18 → alert: "none"
    ├─ Compute source mix:
    │  ├─ Current outlet counts vs previous same-length window
    │  └─ Jensen-Shannon divergence + warning
    ├─ Compute source-normalized drift:
    │  ├─ Per-outlet segment profiles
    │  ├─ Equal weights across observed fixed-panel outlets
    │  └─ normalization_effect explains raw vs normalized difference
    └─ Response: {"segments": [...drift data...]}
    ↓
[4] GET /brief/energie
    ├─ Fetch drift data (compute or cache hit)
    ├─ Rank segments by drift magnitude
    ├─ Select representative articles per segment
    ├─ Build prompt with drift evidence + article snippets
    ├─ Call Ollama (qwen2.5:7b) for brief (timeout: 30 sec)
    ├─ Parse JSON response → cache for 30 min
    └─ Response: {"topic": "energie", "summary": "...", "hypotheses": [...]}
    ↓
[5] Dashboard UI
    ├─ Display drift per segment with alert levels (color coded)
    ├─ Show brief + hypotheses
    └─ Link to raw signals
```

---

## Design Decisions Explained

### 1. SQLite vs. PostgreSQL
**Choice:** SQLite
- **Pros:** Zero deployment, ACID, WAL concurrency, good for ~1M articles
- **Cons:** Single writer, no distributed processing
- **When to upgrade:** PostgreSQL for Phase 2 multi-worker scaling

### 2. LinUCB Bandit
**Choice:** LinUCB (Contextual Multi-Armed Bandit)
- **Pros:** Mathematically principled, adapts to topic context, avoids poor feeds
- **Cons:** More complex, assumes stationary rewards
- **Alternative:** Thompson Sampling, e-greedy

### 3. Local Ollama LLM
**Choice:** Local inference via Ollama
- **Pros:** No API costs, no rate limits, privacy, deterministic
- **Cons:** Requires GPU, smaller models than cloud APIs, ~1-2 sec latency
- **Alternative:** OpenAI API (higher quality, higher cost)

### 4. Four Segment Relevance Groups
**Choice:** young_urban, family, senior, b2b
- **Rationale:** Domain expertise from Lakmoos; balance between granularity and data availability
- **Customizable:** Add/remove by modifying `config/feeds.py` and schemas

### 5. Seeded Baselines
**Choice:** Seeded baselines (first run = baseline)
- **Pros:** Works immediately, easy to reset
- **Cons:** No drift on first run, baselines can be stale
- **Future:** Learn baselines from historical data using exponential smoothing or seasonal decomposition

---

## Performance Characteristics

| Operation | Time | Bottleneck |
|-----------|------|------------|
| `/collect` | 2-5 min | RSS feed latency + semantic filtering |
| `/extract` | 1-2 min | LLM inference (Ollama, ~30-50ms per article) |
| `/drift` | <1 sec | SQL aggregation |
| `/brief` | 1-2 min (first) | LLM inference; cached 30 min |
| `/signals` | <100ms | SQLite query |

**Future Optimizations:**
- Batch LLM calls (if API supports)
- GPU acceleration (CUDA for embeddings)
- Async extraction (currently serial)
- Multi-level caching (Redis, HTTP, browser)

---

## Testing Strategy

Tests organized by module (12 test files):
- `test_extraction_api.py` - Route handlers + signal schema
- `test_delta.py` - Aggregation + drift calculation
- `test_brief.py` - Brief generation + fallback
- `test_bandit.py` - LinUCB arm selection
- `test_pipeline.py` - End-to-end workflow
- `test_e2e.py` - Full integration tests
- Plus 6 more specialized tests

For test execution, see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## References & Navigation

- **Complete API Reference:** [API.md](API.md) - All 8 endpoints with examples
- **Developer Guide:** [DEVELOPMENT.md](DEVELOPMENT.md) - Setup, testing, common tasks
- **Known Limitations:** [concerns.md](concerns.md) - Future roadmap and trade-offs
- **Original Context:** Lakmoos media intelligence project documentation
