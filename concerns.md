# SigDriftr: Known Limitations & Future Roadmap

## Current Limitations

### Scalability

**Serial Processing**
- Ingestion: Articles fetched and filtered sequentially per feed (no parallelization within feed)
- Extraction: Articles processed one at a time by LLM (no batching)
- Current throughput: ~2 articles/sec (limited by LLM inference speed)
- Time to analyze 1000 articles: ~8 minutes (1000 / 2 / 60)

**SQLite Single-Writer Bottleneck**
- Only one process can write to SQLite at a time
- WAL mode mitigates but doesn't eliminate contention
- Limits concurrent /extract operations to 1
- Fine for demo, hits limits with multi-worker setup

**No Distributed Caching**
- Brief caching: in-memory dict (single instance only)
- No Redis or Memcached
- Can't share cache across multiple API servers

### Feature Gaps

**No Topic Normalization**
- "energie" ≠ "energetika" ≠ "energy" (treated as different topics)
- No lemmatization or synonym expansion
- Articles lost if topic spelling varies

**No Authentication or Multi-User**
- No API keys, JWT, or user accounts
- No per-user analysis isolation
- Not suitable for multi-tenant deployment

**No Background Job Processing**
- No job queue (Celery scaffolding exists but not implemented)
- /collect and /extract are synchronous HTTP endpoints (blocking caller)
- No scheduled tasks (daily collection, baseline learning, etc.)
- Long operations tie up worker processes

**No Vector Database**
- No semantic search on signals
- Can't answer: "Find articles about renewable energy similar to this one"
- Embedding model used only for filtering, not stored or searchable

**No Learning-Based Baselines**
- Baselines are seeded (first profile = baseline), never updated
- No historical trend analysis
- Can't distinguish seasonal drift from permanent shifts
- Baselines become stale over time

### Operational Gaps

**Semantic Filtering Fails Silently**
- If embedding model can't load: only direct string matching works
- Related articles missed without warning
- No health check or alert for degraded filtering

**Ollama Dependency**
- No fallback if Ollama unavailable (brief endpoint has fallback, but extraction doesn't)
- Models cached locally (good for offline, but requires large disk space)
- GPU required for good performance (CPU inference ~10x slower)
- No cloud API fallback option

**No Monitoring or Alerting**
- No metrics (request latency, error rates, queue depth)
- No alerts (Ollama down, DB corruption, extraction quality drop)
- No health dashboard or status page

**No Audit Logs**
- No record of who accessed what, when
- No change history for baselines
- Can't trace data provenance

**No Input Validation Beyond Schema**
- No rate limiting (client can spam /collect /extract)
- No topic length limits, unusual character handling
- No check for obviously malformed RSS feeds

---

## Design Trade-Offs (Why Current Choices Exist)

### SQLite vs. PostgreSQL

**Current:** SQLite  
**Why:** Zero deployment, excellent for demo/dev, sufficient for ~1M articles

**When to Upgrade:**
- Approaching 1M articles
- Need multi-worker concurrent writes
- Need distributed setup (multiple servers)
- Need advanced SQL features (arrays, JSON operators, extensions)

**Cost of Upgrade:**
- Setup: 1-2 hours
- Code changes: Replace `db/init.py` with postgres schema
- Deployment: Docker Compose or managed PostgreSQL
- Benefits: 10x throughput, distributed reads, advanced features

### LinUCB Bandit vs. Simpler Alternatives

**Current:** LinUCB  
**Why:** Mathematically principled exploration/exploitation, adapts to topic context

**Alternatives:**
- e-greedy: Simpler, less optimal feed selection
- Thompson Sampling: Similar performance, more complex
- Popularity-only: Fast but ignores topic context

**Cost of Change:** 1-2 hours to reimplement (self-contained in `bandit.py`)

### Local Ollama vs. Cloud API

**Current:** Local Ollama  
**Why:** No API costs, no rate limits, privacy, deterministic, always-on

**Trade-Off:** Smaller models (7B vs. 13B-70B), ~1-2 sec latency per article

**When to Switch:**
- Need better signal quality (use larger models)
- Want to avoid GPU maintenance
- Can afford API costs ($0.02-0.10 per article)
- Need enterprise SLA

**Cost of Switch:** 2-3 hours (implement new client, update prompts, handle API errors)

### Seeded Baselines vs. Learned Baselines

**Current:** Seeded (first profile = baseline)  
**Why:** Works immediately, no historical data required, simple to reset

**Why Not Learned:** Historical learning requires stable 30-90 days of data, more complex algorithm

**Cost of Learning:** 4-6 hours
- Collect 30-90 days historical data (or load from CSV)
- Implement exponential smoothing or seasonal decomposition
- Add baseline learning endpoint
- Test seasonal patterns

---

## Future Roadmap

### Phase 2: Production-Ready (2-3 weeks effort)

**Goal:** Handle 10x throughput, multi-worker deployment

- [ ] PostgreSQL migration
  - Cost: 8 hours
  - Benefit: 10x write throughput, distributed setup
  - Owner: Backend engineer

- [ ] Celery job queue + Redis
  - Cost: 12 hours
  - Benefit: Async /collect /extract, background tasks
  - Owner: Backend engineer
  - Tasks:
    - Async collection (don't block caller)
    - Scheduled baseline learning
    - Periodic feed health checks

- [ ] Basic auth (API keys)
  - Cost: 4 hours
  - Benefit: Per-user analysis isolation, quota management
  - Owner: Security engineer

- [ ] Monitoring + Metrics (Prometheus)
  - Cost: 8 hours
  - Benefit: Visibility into performance, error rates, quality
  - Owner: DevOps engineer
  - Metrics: Request latency, Ollama inference time, LLM quality scores

### Phase 3: Advanced Features (1 month effort)

**Goal:** Advanced analytics and customization

- [ ] Vector database (Weaviate or Milvus)
  - Cost: 20 hours
  - Benefit: Semantic search on signals, similarity queries
  - Use case: "Find articles about renewable energy similar to this one"

- [ ] Baseline learning from historical data
  - Cost: 12 hours
  - Benefit: Distinguish seasonal drift from permanent shifts
  - Algorithm: Exponential smoothing + seasonal decomposition

- [ ] Multi-topic correlation
  - Cost: 8 hours
  - Benefit: "Which other topics drift when energy prices rise?"
  - Method: Compute cross-topic signal correlation matrix

- [ ] Custom segment definition
  - Cost: 16 hours
  - Benefit: User-defined segments (e.g., "Tesla owners", "rural vs. urban")
  - Architecture: Segment definition DSL + dynamic aggregation

- [ ] Temporal analysis
  - Cost: 12 hours
  - Benefit: Trend decomposition, forecasting, seasonality
  - Method: Time series decomposition (trend, seasonal, residual)

### Phase 4: MLOps (1-2 months effort)

**Goal:** Production ML pipeline, continuous improvement

- [ ] Model evaluation pipeline
  - Cost: 20 hours
  - Benefit: Measure signal quality, compare LLM models
  - Metrics: Signal schema accuracy, frame classification F1, segment relevance

- [ ] A/B testing framework
  - Cost: 24 hours
  - Benefit: Test new LLM models, extraction prompts, bandit algorithms
  - Method: Online A/B test with statistical significance

- [ ] Signal drift monitoring
  - Cost: 16 hours
  - Benefit: Detect when signal quality changes (LLM model degradation, distribution shift)
  - Method: Monitor signal distribution over time

- [ ] Automated prompt optimization
  - Cost: 20 hours
  - Benefit: Improve extraction quality without manual tuning
  - Method: Prompt engineering with systematic search or RL

---

## When to Fork vs. Extend

### Extend This Codebase If:

✅ Adding new RSS feeds (edit `config/feeds.py`)  
✅ Adding new audience segments (edit schemas + `/delta`)
✅ Tuning signal extraction prompts (edit `brief/prompt.py`)  
✅ Changing feed selection algorithm (edit `ingestion/bandit.py`)  
✅ Adding more analysis windows (days_back parameter)  
✅ Implementing new alert thresholds (edit `delta/engine.py`)  

**Effort:** Hours to 1-2 days per feature

### Fork This Codebase If:

❌ Need to change fundamental architecture (e.g., replace SQLite+Ollama with API-based)  
❌ Need completely different signal schema (not 8-field behavioral model)  
❌ Want to use different LLM framework (not Ollama)  
❌ Need non-Czech language support + custom segment definitions  
❌ Want completely different drift detection algorithm  

**Effort:** 1-2 weeks to reimplement from scratch

---

## Performance Targets & Benchmarks

**Current Performance (Single Ollama instance, 7B model):**

| Operation | Time | Throughput | Bottleneck |
|-----------|------|------------|------------|
| /collect (1000 articles) | ~5-10 min | Variable | Feed latency |
| /extract (1000 articles) | ~8 min | 2 art/sec | LLM inference |
| /drift | <1 sec | N/A | SQL query |
| /brief (first) | 1-2 min | 1 brief/min | LLM inference |
| /brief (cached) | <10ms | 1000 brief/sec | Cache lookup |

**Target Performance (Phase 2 with optimizations):**

| Operation | Time | Throughput | How |
|-----------|------|------------|-----|
| /collect (1000 articles) | ~2-3 min | Variable | Larger batch sizes |
| /extract (1000 articles) | ~2 min | 10 art/sec | Batch LLM calls, GPU |
| /drift | <1 sec | N/A | No change |
| /brief (first) | 30 sec | 2 brief/min | Batch prompting |
| /brief (cached) | <10ms | 1000 brief/sec | No change |

**10x Target (Phase 3 with full scaling):**
- PostgreSQL: 10x write throughput
- Batch LLM: 5x extraction throughput
- Multi-worker: Linear scaling with worker count

---

## Known Bugs or Edge Cases

### Embedding Model Fallback

**Issue:** If `sentence-transformers` can't load, semantic filtering silently disabled

**Workaround:** Check `/health` endpoint; ensure `"embeddings": "ok"`

**Fix (Phase 2):** Explicit health check on startup, fail fast if required

### Segment Weight Softmax

**Issue:** If all segment weights are 0 (very rare), softmax returns NaN

**Workaround:** Clamp to minimum 0.1 per segment (guardrail in `extraction/llm_client.py`)

**Status:** Already fixed in latest version

### Brief Caching Key

**Issue:** Brief cache key is (topic, days_back), doesn't account for data changes

**Example:** If new articles added after brief generated, cached brief is stale

**Workaround:** Manually clear cache by restarting API (cache is in-memory, no persistence)

**Fix (Phase 2):** Timestamp-based cache invalidation when new signals extracted

---

## Data Retention & Privacy

### Data Stored

- **Articles:** Title, summary, URL hash (not full URL)
- **Signals:** Behavioral metrics (not article text)
- **Entities:** Named entities only (no PII)
- **Baselines:** Aggregated metrics (no article references)
- **Bandit:** Feed performance (no individual events)

### Retention Policy (Current)

- **Articles:** 90 days (configurable)
- **Signals:** Same as articles
- **Entities:** Same as articles
- **Baselines:** Indefinite (until reset)
- **Logs:** Not logged (future: 30 days)

### GDPR Compliance

**Current:** Not GDPR-compliant (no user consent, no data deletion mechanism)

**Future (Phase 2):**
- Add `/admin/delete?topic=...` endpoint
- Add `/admin/export?topic=...` endpoint
- Anonymize entity extraction (remove specific names)

---

## Testing & Quality

### Current Test Coverage

- Unit tests: 12 test files, ~80 test cases
- Coverage: ~60% (extraction logic well-covered, API routes less so)
- Integration tests: End-to-end workflows tested
- E2E tests: Full pipeline with mock Ollama

### Gap: Load Testing

**Missing:** Load tests with concurrent requests, stress tests with large articles

**Future:** Implement with `locust` or `k6`

### Gap: Model Quality Testing

**Missing:** Validation of signal extraction quality (F1 score, precision/recall per frame)

**Future (Phase 3):** Labeled dataset + evaluation metrics

---

## References

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **API:** [API.md](API.md)
- **Development:** [DEVELOPMENT.md](DEVELOPMENT.md)
- **GitHub Issues:** https://github.com/EightL/SigDriftr/issues
