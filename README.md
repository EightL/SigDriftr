# SigDriftr

SigDriftr is a local LLM-powered signal extraction and drift detection pipeline for analyzing Czech media trends. It pulls articles from RSS feeds, extracts behavioral signals using Ollama, aggregates signals by audience segment, detects drift from baseline profiles, and generates research briefs.

## Quick Links

📚 **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete technical design, module breakdown, data flow  
📖 **[API.md](API.md)** - All 8 endpoints with examples and response schemas  
🔧 **[DEVELOPMENT.md](DEVELOPMENT.md)** - Setup guide, testing, common dev tasks  
⚠️  **[concerns.md](concerns.md)** - Known limitations, future roadmap, scalability notes

## What It Does

For a given topic (e.g., "energie", "zdravotnictvi"):

```
[1] RSS Collection (ingestion/)
    ├─ Bandit selects feeds adapted to topic
    ├─ Fetches articles concurrently
    ├─ Two-pass filtering (string match + semantic)
    └─→ Articles stored in SQLite
         ↓
[2] Signal Extraction (extraction/)
    ├─ Sends article title+summary to Ollama
    ├─ Extracts 8-field signal schema (qwen2.5:7b-instruct)
    ├─ Normalizes LLM output
    ├─ Optional: spaCy entity enrichment
    └─→ Signals + entities stored in SQLite
         ↓
[3] Segment Aggregation (delta/)
    ├─ Computes weighted averages per segment
    ├─ Stores segment profiles
    └─→ Profiles stored in SQLite
         ↓
[4] Drift Detection (delta/)
    ├─ Compares current profiles to baseline
    ├─ Computes drift magnitude
    ├─ Alert levels: none / mild / strong
    └─→ Drift data + alert levels
         ↓
[5] Brief Generation (brief/)
    ├─ Ranks segments by drift
    ├─ Selects representative articles
    ├─ Generates prompt with evidence
    ├─ Calls Ollama for brief
    ├─ Fallback: deterministic brief if LLM fails
    └─→ Research brief with 3 hypotheses
         ↓
[6] Dashboard UI + API
    └─→ JSON API + interactive HTML dashboard
```

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    SigDriftr Pipeline                           │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [1] Ingestion          [2] Extraction      [3] Delta           │
│      (ingestion/)           (extraction/)       (delta/)         │
│  ├─ crawler.py      ├─ extractor.py    ├─ mapper.py          │
│  └─ bandit.py       ├─ llm_client.py   ├─ engine.py          │
│                     └─ entities.py     └─ seeder.py          │
│                                                                  │
│  [4] Brief              [5] API             [6] Database       │
│      (brief/)            (api/)             (db/)              │
│  ├─ generator.py   ├─ routes/*.py    └─ init.py           │
│  ├─ prompt.py      └─ scheduler.py       (SQLite schema)      │
│  └─ models.py                                                   │
│                                                                  │
│  External: Ollama (LLM), spaCy (NER), sentence-transformers   │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

## Core Endpoints

| Method | Path | Purpose |
|--------|------|----------|
| POST | `/collect?topic=...` | Crawl feeds, store articles |
| POST | `/extract?topic=...` | Extract signals from articles |
| GET | `/signals?topic=...` | Get article-level signals |
| GET | `/calibration/{topic}/{segment}` | Get current segment profile |
| GET | `/drift/{topic}?days_back=7` | Get drift vs baseline |
| GET | `/brief/{topic}` | Get research brief |
| GET | `/health` | System health check |
| GET | `/ui` | Demo dashboard |

For detailed endpoint documentation, see [API.md](API.md).

## What It Does NOT Do

- ❌ No vector database (no semantic signal search)
- ❌ No auth or multi-user support
- ❌ No background job processing (POST endpoints are synchronous)
- ❌ No learned baselines (seeded on first run only)
- ❌ No topic normalization ("energie" ≠ "energetika")

See [concerns.md](concerns.md) for full limitations and roadmap.

## Key Design Decisions

### Why SQLite?
**Zero deployment overhead, ACID transactions, WAL mode concurrency. Hit limits ~1M articles.**

### Why LinUCB Bandit?
**Mathematically principled exploration/exploitation. Adapts feed selection to topic context.**

### Why Local Ollama?
**No API costs, no rate limits, privacy, deterministic. Smaller models (7B), ~1-2 sec latency.**

### Why 4 Segments (young_urban, family, senior, b2b)?
**Domain expertise + balance between granularity and data availability.**

### Why Seeded Baselines?
**Works immediately without historical data. First run establishes reference.**

For deeper rationale, see [ARCHITECTURE.md](ARCHITECTURE.md#design-decisions-explained).

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Ollama** with models:
  ```bash
  ollama pull qwen2.5:7b-instruct
  ollama pull gemma3:1b
  ```
- **Optional:** spaCy Czech model
  ```bash
  python -m spacy download cs_core_news_sm
  ```

### Installation

```bash
git clone https://github.com/EightL/SigDriftr.git
cd SigDriftr

python3 -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Run

```bash
# Start Ollama (in another terminal)
ollama serve

# Start API (this terminal)
uvicorn main:app --reload
```

**Access:**
- OpenAPI docs: http://localhost:8000/docs
- Demo UI: http://localhost:8000/ui

### Quick Example

```bash
# 1. Collect articles
curl -X POST "http://localhost:8000/collect?topic=energie"

# 2. Extract signals
curl -X POST "http://localhost:8000/extract?topic=energie"

# 3. Get drift
curl "http://localhost:8000/drift/energie"

# 4. Get brief
curl "http://localhost:8000/brief/energie"
```

For more examples, see [API.md](API.md#typical-workflows).

## Project Structure

```
SigDriftr/
├── main.py                 # FastAPI entry point
├── requirements.txt        # Python dependencies
├── sigdriftr.db           # SQLite database (auto-created)
│
├── api/routes/            # HTTP endpoints
├── db/                    # SQLite schema + pooling
├── config/                # RSS feeds + segment priors
├── ingestion/             # RSS collection + LinUCB bandit
├── extraction/            # LLM-based signal extraction
├── delta/                 # Segment aggregation + drift
├── brief/                 # Research brief generation
├── static/                # Dashboard UI
│
├── test_*.py              # Test suite (12 files)
├── conftest.py           # Pytest fixtures
│
├── README.md             # This file
├── ARCHITECTURE.md       # Technical design
├── API.md                # Endpoint reference
├── DEVELOPMENT.md        # Developer guide
└── concerns.md           # Limitations + roadmap
```

## Core Concepts

### Signal Schema (8 fields)

For each article:
- **concern_level** [0-1]: How worried is the public?
- **purchase_intent** [0-1]: Likelihood to buy/act?
- **avoidance_signals** [0-1]: Desire to avoid/shun?
- **dominant_frame**: fear | opportunity | conflict | neutral
- **seg_young_urban** [0-1]: Relevance to young urban segment
- **seg_family** [0-1]: Relevance to family segment
- **seg_senior** [0-1]: Relevance to senior segment
- **seg_b2b** [0-1]: Relevance to B2B segment

### Drift Alert Levels

```
no_data       → No articles in segment
none          → drift_magnitude < 0.20
mild          → 0.20 ≤ drift_magnitude < 0.45
strong        → drift_magnitude ≥ 0.45
```

### Audience Segments

- **young_urban**: Tech-savvy, early adopters, online-first
- **family**: Household budgets, children's wellbeing
- **senior**: Healthcare, pensions, stability
- **b2b**: Business impact, supply chains, regulations

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=.

# Run specific test file
pytest test_extraction_api.py -v
```

For test details, see [DEVELOPMENT.md](DEVELOPMENT.md#testing).

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| /collect (1000 articles) | 5-10 min | Feed latency dependent |
| /extract (1000 articles) | ~8 min | ~2 articles/sec (LLM bottleneck) |
| /drift | <1 sec | Cached SQL aggregation |
| /brief (first call) | 1-2 min | LLM inference |
| /brief (cached) | <10ms | 30-min cache TTL |

For performance optimization tips, see [DEVELOPMENT.md](DEVELOPMENT.md#performance-troubleshooting).

## Contributing

We welcome contributions! See [DEVELOPMENT.md](DEVELOPMENT.md) for:

- Setup instructions
- Code style guidelines
- How to write tests
- Common development tasks

## Future Roadmap

### Phase 2: Production-Ready
- [ ] PostgreSQL migration (10x throughput)
- [ ] Celery + Redis (async jobs, distributed caching)
- [ ] API key authentication
- [ ] Prometheus monitoring

### Phase 3: Advanced Features
- [ ] Vector database (semantic signal search)
- [ ] Learned baselines (distinguish seasonal vs. permanent drift)
- [ ] Multi-topic correlation
- [ ] Custom segment definition

### Phase 4: MLOps
- [ ] Model evaluation pipeline
- [ ] A/B testing framework (LLM models, prompts)
- [ ] Signal quality monitoring
- [ ] Automated prompt optimization

See [concerns.md](concerns.md#future-roadmap) for detailed roadmap with effort estimates.

## Known Limitations

- **No topic normalization**: "energie" ≠ "energetika"
- **Semantic filtering degrades silently**: If embedding model unavailable, only direct match works
- **Seeded baselines**: Not learned from historical data
- **Single-writer SQLite**: Limits concurrent /extract operations
- **No background jobs**: /collect and /extract are synchronous

For complete limitations, see [concerns.md](concerns.md#current-limitations).

## References & Navigation

- **Architecture & Design:** [ARCHITECTURE.md](ARCHITECTURE.md) (800+ lines)
- **API Reference:** [API.md](API.md) (500+ lines) - All endpoints with examples
- **Developer Setup:** [DEVELOPMENT.md](DEVELOPMENT.md) (400+ lines) - Testing, common tasks
- **Limitations:** [concerns.md](concerns.md) (400+ lines) - Known issues, future roadmap
- **GitHub:** https://github.com/EightL/SigDriftr

---

**Last Updated:** March 2026  
**Version:** 1.0 (Demo/Alpha)  
**Status:** Active Development  
**License:** See LICENSE file
