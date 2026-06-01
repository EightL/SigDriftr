# SigDriftr API Reference

## Overview

SigDriftr exposes REST endpoints for collecting articles, extracting signals,
computing drift, clustering storylines, and generating analyst-facing summaries.
All responses are JSON format.

**Base URL:** `http://localhost:8000`  
**OpenAPI UI:** `http://localhost:8000/docs`

### Topic Normalization

All topic-scoped endpoints accept the user-facing `topic` string as requested,
then resolve it to a shared `canonical_topic_id`. For example, `energie`,
`energetika`, `ceny energii`, and `energy` resolve to canonical topic `energy`.
The raw request string is retained for auditability, while collection runs,
article-topic links, baselines, bandit rewards, cluster runs, cluster tracks,
drift runs, summaries, and briefs use the canonical id for shared history.

Responses that expose topic scope include:

- `topic`: the visible topic for that endpoint, usually the request string or
  stored run topic.
- `requested_topic`: the raw topic passed by the caller, when applicable.
- `canonical_topic_id`: the stable shared scope key.
- `canonical_display_name`: a human-readable canonical label.

---

## Endpoints

### 1. POST /collect - Collect Articles from RSS Feeds

**Purpose:** Crawl eligible RSS feeds and store articles relevant to topic.

**Request:**
```bash
curl -X POST "http://localhost:8000/collect?topic=energie"
```

**Query Parameters:**
- `topic` (string, required): Topic keyword(s), e.g., "energie", "zdravotnictvi"
- `country` (string, optional): `CZ`, `DE`, or `GLOBAL`
- `source` (string, optional): specific outlet id
- `collection_mode` (string, optional): `bandit` (default), `all`, or `fixed_panel`
- `reward_mode` (string, optional): `yield` (default, non-LLM) or `signal` (experimental)

**Response (200 OK):**
```json
{
  "topic": "energie",
  "canonical_topic_id": "energy",
  "inserted": 22,
  "extracted": 22,
  "rewards_recorded": 0,
  "collection_mode": "bandit",
  "reward_mode": "yield",
  "selected_feeds": ["irozhlas", "ct24", "e15"],
  "accepted": 24,
  "duplicates": 2
}
```

**Notes:**
- Two-pass filtering: direct string match + semantic similarity (if available)
- Articles deduplicated by URL hash (SHA-256)
- Articles keep both `raw_topic` and `canonical_topic_id`; aliases share the
  same article history without duplicating article rows
- `bandit` selection uses LinUCB; `fixed_panel` and `all` are available for reproducible eval/demo runs
- Default bandit rewards use non-LLM article yield, relevance, and duplicate-adjusted collection results
- Concurrent feed fetching (timeout: 10 sec per feed)
- Time: 2-5 minutes depending on feed latency

**Errors:**
- 500: No feeds selected (rare), Ollama down, DB error

---

### 2. GET /summaries - Generate Citation-First Digest

**Purpose:** Return a concise article digest for a topic/country/source scope.

**Request:**
```bash
curl "http://localhost:8000/summaries?topic=energie&country=CZ&limit=8"
```

**Query Parameters:**
- `topic` (string, required): Topic keyword or alias
- `country` (string, optional): `CZ`, `DE`, or `GLOBAL`
- `source` (string, optional): specific outlet id
- `limit` (integer, default 8): Maximum articles to cite

**Response (200 OK):**
```json
{
  "topic": "energie",
  "requested_topic": "energie",
  "canonical_topic_id": "energy",
  "canonical_display_name": "Energy",
  "country": "CZ",
  "source": "all",
  "article_count": 8,
  "generated_at": "2026-03-18T17:50:00Z",
  "sources_used": ["irozhlas", "ct24"],
  "summary_headline": "Energy coverage focuses on prices and supply",
  "summary_text": "Recent Czech coverage emphasizes household costs and grid resilience.",
  "key_points": [
    "The top-ranked article is from iRozhlas.",
    "Coverage is concentrated in two outlets.",
    "The digest cites the source articles separately."
  ],
  "articles": [
    {
      "article_id": "abc123",
      "title": "Ceny energií znovu rostou",
      "url": "https://example.test/story",
      "outlet": "irozhlas",
      "country": "CZ",
      "published_at": "2026-03-18T10:00:00Z",
      "relevance_score": 0.94
    }
  ]
}
```

**Notes:**
- Uses `article_topics.canonical_topic_id`, so aliases share the same digest pool
- Falls back to a deterministic digest if the local LLM is unavailable

---

### 3. POST /extract - Extract Coverage Framing Signals

**Purpose:** Run LLM-based signal extraction on collected articles

**Request:**
```bash
curl -X POST "http://localhost:8000/extract?topic=energie"
```

**Query Parameters:**
- `topic` (string, required): Topic to extract signals for

**Response (200 OK):**
```json
{
  "topic": "energie",
  "articles_processed": 22,
  "signals_stored": 22,
  "timestamp": "2026-03-18T17:50:00Z"
}
```

**Notes:**
- Processes unprocessed articles only (idempotent)
- Uses qwen2.5:7b-instruct (primary) or gemma3:1b (fallback)
- Extracts article-level coverage framing signals, independent segment relevance, and
  normalized segment aggregation shares
- Optional named entity enrichment (spaCy, if available)
- Does not update feed bandit rewards by default. Signal-based rewards are available only when explicitly enabled as experimental behavior.
- Time: 1-2 minutes for ~20 articles (~30-50ms per article)

**Signal Schema Extracted:**
```json
{
  "concern_level": 0.7,
  "purchase_intent": 0.4,
  "avoidance_signals": 0.3,
  "dominant_frame": "opportunity",
  "topic_relevance": "2",
  "topic_relevance_score": 0.94,
  "seg_young_urban_relevance": 0.8,
  "seg_family_relevance": 0.5,
  "seg_senior_relevance": 0.1,
  "seg_b2b_relevance": 0.2,
  "seg_young_urban": 0.6,
  "seg_family": 0.3,
  "seg_senior": 0.05,
  "seg_b2b": 0.05
}
```

**Errors:**
- 500: Ollama down, DB error, extraction timeout

---

### 4. GET /signals - Retrieve Article-Level Signals

**Purpose:** Get extracted coverage framing signals per article

**Request:**
```bash
curl "http://localhost:8000/signals?topic=energie&limit=10&offset=0"
```

**Query Parameters:**
- `topic` (string, required): Topic to filter by
- `limit` (integer, default 20): Number of results to return (max 100)
- `offset` (integer, default 0): Pagination offset

**Response (200 OK):**
```json
{
  "topic": "energie",
  "total_count": 22,
  "signals": [
    {
      "article_id": "abc123",
      "title": "Nová solární farma v Česku",
      "canonical_topic_id": "energy",
      "outlet": "irozhlas",
      "concern_level": 0.52,
      "purchase_intent": 0.41,
      "avoidance_signals": 0.28,
      "dominant_frame": "opportunity",
      "topic_relevance": "2",
      "topic_relevance_score": 0.91,
      "seg_young_urban_relevance": 0.82,
      "seg_family_relevance": 0.40,
      "seg_senior_relevance": 0.08,
      "seg_b2b_relevance": 0.20,
      "seg_young_urban": 0.65,
      "seg_family": 0.25,
      "seg_senior": 0.05,
      "seg_b2b": 0.05,
      "extracted_at": "2026-03-18T17:50:00Z"
    },
    ...
  ]
}
```

**Notes:**
- Fast query: <100ms
- Returns 0 results if topic has no signals (run /extract first)
- Signals are normalized. `seg_*_relevance` fields are independent relevance
  scores; `seg_*` fields are normalized aggregation shares used by drift math.

**Errors:**
- 400: Invalid limit/offset
- 404: Topic not found

---

### 5. GET /calibration/{topic}/{segment} - Get Current Segment Profile

**Purpose:** Get aggregated signal metrics for a specific segment relevance group

**Request:**
```bash
curl "http://localhost:8000/calibration/energie/young_urban?days_back=7"
```

**Path Parameters:**
- `topic` (string): Topic name
- `segment` (string): One of: young_urban, family, senior, b2b

**Query Parameters:**
- `days_back` (integer, default 7): Analysis window in days

**Response (200 OK):**
```json
{
  "topic": "energie",
  "segment": "young_urban",
  "concern_level_avg": 0.52,
  "purchase_intent_avg": 0.41,
  "avoidance_signals_avg": 0.28,
  "dominant_frame": "opportunity",
  "article_count": 12,
  "computed_at": "2026-03-18T17:50:00Z",
  "time_window_days": 7
}
```

**Notes:**
- Aggregates signals from articles where `seg_X > 0.5` (roughly relevant to
  segment as a normalized aggregation share)
- Weighted average per segment
- Recomputed on each `/extract` call
- Returns null fields if no data

**Errors:**
- 400: Invalid segment name
- 404: No signals for topic/segment

---

### 6. GET /drift/{topic} - Compute Drift from Baseline

**Purpose:** Detect signal changes relative to baseline profiles

**Request:**
```bash
curl "http://localhost:8000/drift/energie?days_back=7"
```

**Path Parameters:**
- `topic` (string): Topic name

**Query Parameters:**
- `days_back` (integer, default 7): Analysis window in days

**Response (200 OK):**
```json
{
  "topic": "energie",
  "requested_topic": "energie",
  "canonical_topic_id": "energy",
  "canonical_display_name": "Energy",
  "days_back": 7,
  "source_mix": {
    "current": {
      "article_count": 12,
      "article_count_by_outlet": {"irozhlas": 7, "ct24": 5}
    },
    "reference": {
      "article_count": 10,
      "article_count_by_outlet": {"irozhlas": 4, "ct24": 6}
    },
    "jensen_shannon_divergence": 0.0312,
    "warning": "none"
  },
  "segments": [
    {
      "segment": "young_urban",
      "topic": "energy",
      "canonical_topic_id": "energy",
      "current": {
        "concern_level": 0.52,
        "purchase_intent": 0.41,
        "avoidance_signals": 0.28,
        "dominant_frame": "opportunity"
      },
      "baseline": {
        "concern_level": 0.45,
        "purchase_intent": 0.39,
        "avoidance_signals": 0.25,
        "dominant_frame": "neutral"
      },
      "deltas": {
        "concern_level": 0.07,
        "purchase_intent": 0.02,
        "avoidance_signals": 0.03
      },
      "drift_magnitude": 0.18,
      "source_normalized": {
        "status": "partial_panel",
        "panel_outlets": ["irozhlas", "ct24", "idnes", "e15"],
        "observed_outlets": ["irozhlas", "ct24"],
        "missing_outlets": ["idnes", "e15"],
        "current": {
          "concern_level": 0.49,
          "purchase_intent": 0.40,
          "avoidance_signals": 0.26
        },
        "drift_magnitude": 0.14,
        "normalization_effect": 0.04,
        "interpretation": "similar_to_raw"
      },
      "alert_level": "none",
      "frame_shift": true,
      "article_count": 12,
      "confidence": 0.85
    },
    ...
  ]
}
```

**Alert Levels:**
- `none`: drift_magnitude < 0.20
- `mild`: 0.20 ≤ drift_magnitude < 0.45
- `strong`: drift_magnitude ≥ 0.45

**Notes:**
- Computes segment profiles if missing
- Seeds baselines on first run for the canonical topic scope
- On subsequent runs, compares to stored canonical-topic baselines
- Drift magnitude = 0.333 * |Δ concern| + 0.333 * |Δ purchase| + 0.334 * |Δ avoidance|
- Frame shift indicates change in dominant narrative
- The `confidence` field is a coverage/baseline signal-readiness heuristic, not calibrated correctness
- Source mix reports outlet composition for the current window against the previous same-length window
- `source_normalized` is a bounded fixed-panel comparison. It recomputes the current segment profile by first averaging per outlet, then applying equal weights across observed outlets in the configured fixed panel. It is useful for detecting when raw drift may be inflated by outlet mix, but it is not a causal correction and does not replace raw drift.
- `source_normalized.status` can be `ready`, `partial_panel`, `no_panel_overlap`, or `missing_baseline`.

**Errors:**
- 404: No signals for topic

---

### 7. GET /brief/{topic} - Generate Research Brief

**Purpose:** Generate concise research brief with testable hypotheses

**Request:**
```bash
curl "http://localhost:8000/brief/energie"
```

**Path Parameters:**
- `topic` (string): Topic name

**Response (200 OK):**
```json
{
  "topic": "energie",
  "requested_topic": "energie",
  "canonical_topic_id": "energy",
  "canonical_display_name": "Energy",
  "generated_at": "2026-03-18T17:50:00Z",
  "summary": "Czech renewable energy coverage is shifting from caution to opportunity framing. Young-urban- and B2B-relevant articles show the clearest opportunity framing, while family-relevant coverage remains focused on costs.",
  "hypotheses": [
    "Coverage aimed at tech-forward readers will keep emphasizing renewable subsidy opportunities",
    "Family-relevant articles will continue to frame costs and grid stability as open questions",
    "Business-relevant coverage will keep presenting early solar adoption as a strategic advantage"
  ],
  "segments": [
    {
      "segment": "b2b",
      "drift_magnitude": 0.42,
      "alert_level": "mild",
      "key_articles": [
        {"title": "Firmy investují do solárů", "outlet": "e15"},
        {"title": "Nové byznysy energií", "outlet": "idnes"}
      ]
    },
    ...
  ],
  "cached": false
}
```

**Notes:**
- Always returns 200 OK (even if LLM fails, returns fallback brief)
- Caches results for 30 minutes (subsequent calls return cached result)
- Uses canonical topic scope for drift, cluster snapshots, and cached briefs
- First call: 1-2 minutes (LLM inference)
- Cached calls: <10ms
- Uses qwen2.5:7b-instruct for generation
- Fallback brief generated deterministically if LLM times out

**Errors:**
- None (always 200 OK with fallback)

---

### 8. Cluster Pipeline Endpoints

**Purpose:** Build and inspect cluster-aware storyline drift.

Common calls:

```bash
curl -X POST "http://localhost:8000/pipeline/cluster?topic=energie"
curl -X POST "http://localhost:8000/pipeline/cluster/signals?run_id=<cluster_run_id>"
curl -X POST "http://localhost:8000/pipeline/cluster/drift?run_id=<cluster_run_id>"
curl "http://localhost:8000/pipeline/clusters/latest?topic=energy"
curl "http://localhost:8000/drift/clusters/energy"
```

Cluster run and cluster drift responses include `canonical_topic_id`. Latest-run
lookups accept any configured alias and return the stored run topic plus the
canonical id. This means a run created for `energie` is discoverable through
`energy`, and cluster tracks are shared across aliases.

---

### 9. GET /health - System Health Check

**Purpose:** Verify system health and dependencies

**Request:**
```bash
curl "http://localhost:8000/health"
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "timestamp": "2026-03-18T17:50:00Z",
  "db": "ok",
  "ollama": "ok",
  "spacy": "ok",
  "uptime_seconds": 3600
}
```

**Possible Status Values:**
- `ok`: All systems operational
- `degraded`: Core system ok, optional dependencies missing (e.g., spaCy, semantic filtering)
- `error`: Core system failed

**Errors:**
- 503: Core system unavailable (DB or Ollama down)

---

### 10. GET /ui - Demo Dashboard

**Purpose:** Interactive web UI for analyzing topics

**Request:**
```bash
Open in browser: http://localhost:8000/ui
```

**Features:**
- Input topic
- Trigger /collect and /extract
- View drift results
- Display research brief
- View raw signals

**Response:** HTML page

---

## Typical Workflows

### Workflow 1: Analyze a New Topic

```bash
# 1. Collect articles
curl -X POST "http://localhost:8000/collect?topic=energie"

# 2. Extract signals
curl -X POST "http://localhost:8000/extract?topic=energie"

# 3. Get drift analysis
curl "http://localhost:8000/drift/energie"

# 4. Get research brief
curl "http://localhost:8000/brief/energie"

# 5. View individual signals
curl "http://localhost:8000/signals?topic=energie&limit=20"
```

### Workflow 2: Monitor Topic Over Time

```bash
# Day 1: Analyze
curl -X POST "http://localhost:8000/collect?topic=energie"
curl -X POST "http://localhost:8000/extract?topic=energie"
BASELINE_ESTABLISHED

# Day 8: Re-analyze (last 7 days for drift comparison)
curl -X POST "http://localhost:8000/collect?topic=energie"
curl -X POST "http://localhost:8000/extract?topic=energie"
curl "http://localhost:8000/drift/energie?days_back=7"
# Shows drift from Day 1 baseline
```

### Workflow 3: Analyze Per-Segment Trends

```bash
# Get all segments for topic
curl "http://localhost:8000/drift/energie"

# Deep dive: specific segment
curl "http://localhost:8000/calibration/energie/young_urban?days_back=7"
curl "http://localhost:8000/signals?topic=energie" | filter for seg_young_urban
```

---

## Error Handling

**Standard Error Response:**
```json
{
  "detail": "Topic 'neexistuje' has no signals. Run /extract first."
}
```

**Common Error Scenarios:**

| Code | Scenario | Solution |
|------|----------|----------|
| 400 | Invalid query param | Check parameter spelling and type |
| 404 | Topic not found | Run /collect and /extract first |
| 500 | Ollama down | Start Ollama: `ollama serve` |
| 500 | DB error | Check disk space, permissions |
| 503 | Service unavailable | Wait for startup, check logs |

---

## Performance Notes

**Latencies (typical):**
- `/collect`: 2-5 minutes (feed fetching)
- `/extract`: 1-2 minutes (20 articles)
- `/drift`: <1 second
- `/brief` (first): 1-2 minutes (LLM)
- `/brief` (cached): <10ms
- `/signals`: <100ms
- `/health`: <10ms

**Throughput Limits:**
- Single Ollama server: ~2 articles/sec
- SQLite write: ~100 articles/sec
- API: Async, ~100 concurrent requests

**Scaling (future):**
- PostgreSQL: 10x throughput
- Batch LLM calls: 5x throughput
- Multi-worker: Linear scaling with worker count

---

## Rate Limiting

Currently **no rate limiting** (demo phase). Future implementation:
- 100 requests/minute per IP
- 10 concurrent /extract operations
- 1 /collect per topic per hour

---

## Versioning

API version: `1.0` (stable)  
No breaking changes planned. Future additions will be backward compatible.

---

## References

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Development:** [DEVELOPMENT.md](DEVELOPMENT.md)
- **OpenAPI UI:** http://localhost:8000/docs
