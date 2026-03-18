# SigDriftr API Reference

## Overview

SigDriftr exposes 8 REST endpoints for collecting articles, extracting signals, computing drift, and generating research briefs. All responses are JSON format.

**Base URL:** `http://localhost:8000`  
**OpenAPI UI:** `http://localhost:8000/docs`

---

## Endpoints

### 1. POST /collect - Collect Articles from RSS Feeds

**Purpose:** Crawl Czech RSS feeds and store articles relevant to topic

**Request:**
```bash
curl -X POST "http://localhost:8000/collect?topic=energie"
```

**Query Parameters:**
- `topic` (string, required): Topic keyword(s), e.g., "energie", "zdravotnictvi"

**Response (200 OK):**
```json
{
  "topic": "energie",
  "articles_found": 24,
  "articles_stored": 22,
  "feeds_selected": 3,
  "timestamp": "2026-03-18T17:39:00Z"
}
```

**Notes:**
- Two-pass filtering: direct string match + semantic similarity (if available)
- Articles deduplicated by URL hash (SHA-256)
- Feed selection uses LinUCB bandit (adaptive per topic)
- Concurrent feed fetching (timeout: 10 sec per feed)
- Time: 2-5 minutes depending on feed latency

**Errors:**
- 500: No feeds selected (rare), Ollama down, DB error

---

### 2. POST /extract - Extract Behavioral Signals

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
- Extracts 8-field signal schema per article
- Optional named entity enrichment (spaCy, if available)
- Updates feed bandit rewards based on signal density
- Time: 1-2 minutes for ~20 articles (~30-50ms per article)

**Signal Schema Extracted:**
```json
{
  "concern_level": 0.7,
  "purchase_intent": 0.4,
  "avoidance_signals": 0.3,
  "dominant_frame": "opportunity",
  "seg_young_urban": 0.6,
  "seg_family": 0.3,
  "seg_senior": 0.05,
  "seg_b2b": 0.05
}
```

**Errors:**
- 500: Ollama down, DB error, extraction timeout

---

### 3. GET /signals - Retrieve Article-Level Signals

**Purpose:** Get extracted behavioral signals per article

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
      "outlet": "irozhlas",
      "concern_level": 0.52,
      "purchase_intent": 0.41,
      "avoidance_signals": 0.28,
      "dominant_frame": "opportunity",
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
- Signals are normalized (clamped [0,1], softmax-normalized segments)

**Errors:**
- 400: Invalid limit/offset
- 404: Topic not found

---

### 4. GET /calibration/{topic}/{segment} - Get Current Segment Profile

**Purpose:** Get aggregated signal metrics for a specific audience segment

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
- Aggregates signals from articles where seg_X > 0.5 (roughly relevant to segment)
- Weighted average per segment
- Recomputed on each `/extract` call
- Returns null fields if no data

**Errors:**
- 400: Invalid segment name
- 404: No signals for topic/segment

---

### 5. GET /drift/{topic} - Compute Drift from Baseline

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
  "computed_at": "2026-03-18T17:50:00Z",
  "segments": [
    {
      "segment": "young_urban",
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
- Seeds baselines on first run (baseline = current profile)
- On subsequent runs, compares to stored baseline
- Drift magnitude = 0.333 * |Δ concern| + 0.333 * |Δ purchase| + 0.334 * |Δ avoidance|
- Frame shift indicates change in dominant narrative
- Confidence reflects article count (higher = more data)

**Errors:**
- 404: No signals for topic

---

### 6. GET /brief/{topic} - Generate Research Brief

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
  "generated_at": "2026-03-18T17:50:00Z",
  "summary": "Czech public opinion on renewable energy is shifting from caution to opportunity framing. Young urban and B2B audiences show strongest enthusiasm, while family segment remains concerned about costs.",
  "hypotheses": [
    "Renewable energy subsidies are increasingly popular among tech-forward audiences",
    "Cost and grid stability remain primary concerns for families with children",
    "B2B sector sees strategic advantage in early solar adoption"
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
- First call: 1-2 minutes (LLM inference)
- Cached calls: <10ms
- Uses qwen2.5:7b-instruct for generation
- Fallback brief generated deterministically if LLM times out

**Errors:**
- None (always 200 OK with fallback)

---

### 7. GET /health - System Health Check

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

### 8. GET /ui - Demo Dashboard

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
