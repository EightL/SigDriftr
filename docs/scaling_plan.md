# SigDriftr Scale-Up Plan: 100–1000× Data Volume

## Current State → Target State

| Dimension | Now | Target |
|---|---|---|
| **RSS feeds** | 13 CZ + 8 DE + 4 Global = 25 | 200–500+ feeds across CZ, DE, EN, SK, PL |
| **Articles** | 224 total | 10,000–50,000+ per week |
| **Topics** | 4 seed topics (fragmented into 15) | 50–200 tracked topics, auto-discovered |
| **Signal extraction** | LLM per article (~2s each via API) | Trained classifier (~5ms each on GPU) |
| **Database** | SQLite (1.8MB, single-writer) | PostgreSQL + pgvector |
| **Embeddings** | sentence-transformers CPU, 80/224 done | Fine-tuned model, GPU batch inference |
| **Compute** | Local machine + Ollama | MetaCentrum A100s for training, local/cloud for inference |

---

## User Review Required

> [!IMPORTANT]
> **Key decision: What is this for?** The scaling plan assumes SigDriftr becomes a **research-grade media intelligence platform** — monitoring how news coverage shifts across Central European media. If you want it to be a commercial product, a real-time alerting system, or an academic research tool, the priorities shift. Please confirm the direction.

> [!IMPORTANT]
> **MetaCentrum access model matters.** A100 job scheduling on MetaCentrum typically means batch training jobs, not always-on inference. The plan assumes you train models on MetaCentrum and run inference locally or on a smaller GPU. Confirm if you have persistent GPU access or just batch job allocation.

## Open Questions

1. **Languages**: Right now it's CZ-focused with DE and EN feeds. Do you want to scale to SK (Slovak) and PL (Polish) media too? They share similar topic domains and would multiply data volume.

2. **Social media**: Do you want to add Twitter/X, Reddit, Czech forums (e.g., Novinky comments, iDNES diskuze), or stay purely news media?

3. **Real-time vs batch**: Should the pipeline run continuously (streaming) or is batch (every 30min / hourly / daily) fine at scale?

4. **Budget for APIs**: Some high-quality news sources (NewsAPI, Mediacloud, GDELT) have free tiers but may need paid access at 1000× scale. Google Gemma API also has rate limits. What's the budget situation?

5. **Output consumers**: Who reads the briefs at scale? A human analyst manually, or should this feed into dashboards, Slack alerts, or downstream ML systems?

---

## Proposed Changes

### Phase 1: Feed Explosion (No A100 needed — do this first)

This is the fastest way to go from 224 articles to 10,000+. Pure configuration work.

#### Expand RSS feeds to 200+

The current 25 feeds are hand-picked. There are hundreds of Czech, German, and English news sources with public RSS. This is the lowest-effort, highest-impact change.

##### [MODIFY] [feeds.py](config/feeds.py)

Restructure feed config to load from a YAML/JSON file instead of hardcoded Python lists. Add feeds for:

**Czech (target: 50+ feeds)**
- Major news: Aktuálně.cz, Deník.cz, Hospodářské noviny (HN), Lidové noviny, Právo, MF Dnes
- Business: CzechCrunch, Lupa.cz, Ekonomický deník, Hospodářské noviny ekonomika
- Regional: Brněnský deník, Pražský deník, Moravskoslezský deník
- Tabloid/lifestyle: Extra.cz, Super.cz, Prima, TN.cz
- Tech/science: Root.cz, Živě.cz, Vědavýzkum.cz
- Specialized: Zdravotnický deník, Energetika.cz, Eurozprávy

**Slovak (target: 15+ feeds)**
- SME.sk, Pravda.sk, Aktuality.sk, Denník N, HN Online SK, Nový čas

**German (target: 30+ feeds — already 8)**
- Add Handelsblatt, Frankfurter Rundschau, Tagesspiegel, Focus, n-tv, Bild, Stern

**English (target: 30+ feeds — already 4)**
- Add AP, Reuters full, Washington Post, Financial Times, Al Jazeera, Politico EU, Euronews

**Polish (target: 15+ feeds)**
- Gazeta Wyborcza, Rzeczpospolita, TVN24, Onet, WP.pl

##### [NEW] [config/feed_catalog.yaml](config/feed_catalog.yaml)

Move feed definitions to a YAML catalog that can be edited without code changes:

```yaml
feeds:
  - outlet: aktualne
    rss_url: https://www.aktualne.cz/rss/
    affinity_tag: mainstream
    country: CZ
    language: cs
    enabled: true
    categories: [news, politics, economy]
  # ... 200+ entries
```

##### [MODIFY] [settings.py](config/settings.py)

- Increase `BANDIT_MAX_FEEDS_PER_CRAWL` from 6 → 30–50 (with 200+ feeds, sampling 6 is too few)
- Increase `CRAWL_FETCH_CONCURRENCY` from 4 → 20–30
- Increase `EMBED_LIMIT_DEFAULT` from 200 → 5000
- Add `CRAWL_BATCH_SIZE` to process feeds in parallel batches

#### Add non-RSS news APIs

RSS alone won't get 100× — many outlets have weak RSS. Add API-based ingestion.

##### [NEW] [ingestion/newsapi_source.py](ingestion/newsapi_source.py)

NewsAPI.org adapter — covers 80,000+ sources worldwide. Free tier: 100 requests/day. Paid: 250,000+ articles/month.

##### [NEW] [ingestion/gdelt_source.py](ingestion/gdelt_source.py)

GDELT Project adapter — free, monitors print/broadcast/web news in 100+ languages. Their DOC API can filter by country, theme, and time. This is the single biggest data multiplier available for free.

##### [NEW] [ingestion/mediacloud_source.py](ingestion/mediacloud_source.py)

MediaCloud adapter — academic media monitoring tool. Free for researchers. Czech media collection available.

##### [MODIFY] [ingestion/crawler.py](ingestion/crawler.py)

Abstract the ingestion interface so RSS, NewsAPI, GDELT, and MediaCloud all produce the same `(title, summary, body, url, outlet, topic, published_at)` shape. The crawler becomes a dispatcher over source adapters.

#### Expand topic catalog

##### [MODIFY] [config/topics.py](config/topics.py)

Scale from 4 seed topics to 50+. Add macro-categories:

```
Economy: inflation, energy, housing, wages, GDP, employment, trade, crypto
Politics: elections, EU, parliament, defense, migration, justice, corruption
Health: healthcare, mental_health, drugs, pandemic, hospitals, insurance
Society: education, crime, environment, climate, transport, culture, media
Technology: AI, cybersecurity, startups, digital, telecom
International: ukraine, china, usa, nato, middle_east
```

Each with Czech, German, English, Slovak, Polish aliases.

##### [NEW] [ingestion/topic_discovery.py](ingestion/topic_discovery.py)

Auto-discover trending topics from collected articles using TF-IDF or LLM extraction. Feed them back into the topic catalog as candidates for human review.

---

### Phase 2: Infrastructure for Scale (Needed before 10K+ articles)

SQLite will choke at 50,000+ articles with concurrent reads/writes. The embedding storage (JSON blobs in SQLite) is already terrible for vector search.

#### SQLite → PostgreSQL + pgvector

##### [MODIFY] [db/init.py](db/init.py)

Replace SQLite with PostgreSQL via `asyncpg` or `psycopg`:
- Proper concurrent writes (SQLite's single-writer lock is the scaling bottleneck)
- Connection pooling instead of `threading.local()`
- `pgvector` extension for native vector storage + similarity search (replaces JSON embedding blobs)
- JSONB columns for signal storage (replaces `raw_json` TEXT)

##### [NEW] [db/migrations/](db/migrations/)

Introduce Alembic for proper schema migrations. The current 1235-line `init.py` with inline DDL won't survive ongoing schema evolution at scale.

#### Async worker queue

##### [NEW] [workers/](workers/)

Replace the monolithic synchronous pipeline with an async task queue:
- **Celery + Redis** or **Dramatiq** for job scheduling
- Separate workers for: collection, extraction, embedding, clustering, drift, brief generation
- Each stage becomes independently scalable
- Failed tasks get retried without blocking the pipeline

This is critical because right now `run_full_pipeline()` is one synchronous chain — if embedding takes 10 minutes, everything waits.

#### Vector storage

##### [MODIFY] [extraction/embedding_service.py](extraction/embedding_service.py)

Replace `embedding_vector TEXT` (JSON-encoded list) with:
- **Option A**: pgvector column (`VECTOR(384)`) — keeps everything in PostgreSQL
- **Option B**: Dedicated vector DB (Qdrant, Weaviate, or Milvus) — better for 1M+ vectors

pgvector is simpler and good enough up to ~5M vectors. Recommend starting there.

---

### Phase 3: A100 Model Training (The Big Lever)

This is where MetaCentrum A100s change everything. Instead of prompting a generic LLM for every article, you train specialized models.

#### Train a signal extraction classifier

> [!IMPORTANT]
> **This is the single highest-impact use of the A100s.** Currently, every article takes ~2s for LLM signal extraction (Google Gemma API or Ollama). A fine-tuned classifier does it in ~5ms. At 50,000 articles/week, that's the difference between 28 hours of LLM calls vs 4 minutes of batch inference.

##### Training data

You already have 200 semi-gold annotated articles + 223 LLM-extracted signals. Strategy:
1. Use the 200 semi-gold articles as seed training data
2. Use the LLM outputs (Gemma/ChatGPT/DeepSeek/Claude from the four-model evaluation) as silver labels for a larger training set
3. Active learning loop: train classifier → find articles it's least confident on → human-label those → retrain

##### Model options (ranked by recommendation)

| Model | Size | Why |
|---|---|---|
| **`xlm-roberta-large`** | 560M params | Best multilingual classifier base. Handles CZ/DE/EN/SK/PL natively. Fine-tune with LoRA on A100 in hours. |
| **`czert-b-base-cased`** | 110M params | Czech-specific BERT. Great for CZ-only, but won't generalize to DE/EN. |
| **Gemma 2B/4B** | 2–4B params | Can be fine-tuned with QLoRA on A100. Keeps the structured JSON output format. More capable but slower at inference. |
| **Mistral/Qwen 7B** | 7B params | Full LLM fine-tune. Most capable, but heaviest. QLoRA makes it feasible on A100. |

**Recommended**: Fine-tune `xlm-roberta-large` as a multi-task classifier:
- Head 1: `topic_relevance` (3-class: irrelevant/related/core)
- Head 2: `dominant_frame` (4-class: fear/opportunity/conflict/neutral)
- Head 3: `concern_bucket` (3-class: low/medium/high)
- Head 4: `purchase_relevance` (3-class: none/weak/strong)
- Head 5: `avoidance_relevance` (3-class: none/weak/strong)
- Head 6–9: `seg_*_relevance` (3-class each)

This replaces the entire `extraction/llm_client.py` for article-level signals.

##### [NEW] [training/signal_classifier/](training/signal_classifier/)

- `prepare_dataset.py` — Convert semi-gold + LLM silver labels into HuggingFace Dataset
- `train.py` — Fine-tune XLM-RoBERTa with multi-task heads. LoRA + mixed precision on A100
- `evaluate.py` — Per-field accuracy/F1 against held-out semi-gold test set
- `export.py` — Export to ONNX for fast CPU/GPU inference
- `inference.py` — Batch prediction service replacing `llm_client.extract_signals()`

**Estimated training time on A100**: 2–4 hours for XLM-RoBERTa, 8–12 hours for Gemma 2B with QLoRA.

#### Fine-tune embeddings for media domain

The current `paraphrase-multilingual-MiniLM-L12-v2` is a general-purpose model. A domain-adapted embedding model will produce much better clusters.

##### [NEW] [training/embeddings/](training/embeddings/)

**Method**: Contrastive fine-tuning (SimCSE or TSDAE)
- Positive pairs: articles in the same cluster (from current HDBSCAN output)
- Hard negatives: articles about the same topic but different clusters
- Train on A100 with `sentence-transformers` training API

**Base model**: `intfloat/multilingual-e5-large` (560M params, better multilingual performance than MiniLM)

**Result**: Media-domain-adapted embeddings → better clustering → better storyline detection → better briefs.

**Estimated training time on A100**: 4–8 hours.

#### Train Czech NER model

The current entity extraction in `extraction/entities.py` is basic. A fine-tuned NER model handles Czech names, organizations, and locations correctly.

##### [NEW] [training/ner/](training/ner/)

**Base model**: `ufal/robeczech-base` or `xlm-roberta-base`
**Training data**: Czech Named Entity Corpus (CNEC) + silver labels from the article database
**Entities**: PER, ORG, LOC, PRODUCT, EVENT

---

### Phase 4: Production Architecture

Once you have trained models and 200+ feeds, the architecture looks like this:

```
┌─────────────────────────────────────────────────────────┐
│  INGESTION LAYER                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ RSS 200+ │ │ NewsAPI  │ │ GDELT    │ │MediaCloud │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
│       └─────────────┴────────────┴─────────────┘        │
│                         │                                │
│                    Dedup + Topic Match                    │
│                         │                                │
│                    PostgreSQL                            │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│  EXTRACTION LAYER (GPU inference)                       │
│                                                          │
│  ┌─────────────────────┐    ┌───────────────────────┐   │
│  │ Fine-tuned XLM-R    │    │ Fine-tuned E5-large   │   │
│  │ Signal Classifier   │    │ Media Embeddings      │   │
│  │ ~5ms/article        │    │ ~10ms/article         │   │
│  └─────────────────────┘    └───────────────────────┘   │
│                                                          │
│  ┌─────────────────────┐                                │
│  │ Czech NER           │                                │
│  │ Entity Extraction   │                                │
│  └─────────────────────┘                                │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│  ANALYSIS LAYER                                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ UMAP+HDBSCAN │  │ Drift Engine │  │ Topic         │  │
│  │ Clustering   │  │ (statistical)│  │ Discovery     │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────┐
│  OUTPUT LAYER                                           │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Research     │  │ Drift Alerts │  │ Storyline     │  │
│  │ Briefs (LLM) │  │ (threshold)  │  │ Timeline      │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

#### Key architectural changes for Phase 4

##### [MODIFY] [extraction/llm_client.py](extraction/llm_client.py)

Replace LLM-per-article with trained classifier inference. Keep LLM path as fallback for edge cases or new signal types. The `extract_signals()` function becomes:

```python
def extract_signals(title, summary, body, ...):
    # Phase 4: batch classifier inference
    return signal_classifier.predict(title, summary, body)
```

##### [MODIFY] [brief/generator.py](brief/generator.py)

Brief generation stays LLM-based (it's generative, not classification). But upgrade from local Ollama to:
- Fine-tuned Gemma 2B/4B for brief generation (trained on MetaCentrum)
- Or keep using Google Gemma API at scale with proper rate limiting

##### [MODIFY] [clustering/clustering_service.py](clustering/clustering_service.py)

With 50K+ articles, HDBSCAN needs:
- Incremental/online clustering (don't recluster everything each time)
- Time-windowed clustering (cluster only recent articles, merge with historical tracks)
- GPU-accelerated UMAP via `cuml` if available

---

## Verification Plan

### Phase 1 Verification
- Run collection with 200+ feeds for 48 hours
- Target: 2,000+ new articles per day across all topics
- Verify dedup works (same article from multiple sources)
- Check topic distribution — are core topics getting 100+ articles/day?

### Phase 2 Verification
- PostgreSQL migration: compare query results against SQLite baseline
- pgvector: verify k-NN search returns same neighbors as brute-force
- Worker queue: verify no articles are lost, retries work

### Phase 3 Verification
- Signal classifier vs LLM: compare on held-out semi-gold test set
  - Target: ≥85% accuracy on topic_relevance, ≥75% on dominant_frame
  - Must match or beat Gemma API quality at 100× speed
- Embeddings: measure cluster stability (ARI/NMI) before vs after fine-tuning
- NER: F1 on Czech entity recognition vs baseline

### Phase 4 Verification
- End-to-end pipeline benchmark: 10,000 articles through full pipeline
  - Target: < 30 minutes total (vs current hours)
- Brief quality: human evaluation of generated briefs (comparative rating vs current)
- Drift detection: verify that baselines learn and drift alerts fire correctly

---

## Execution Order

```
Phase 1 (1–2 weeks):     Feed expansion + topic catalog
  ↓                      No GPU needed. Immediate data volume increase.
Phase 2 (2–3 weeks):     PostgreSQL + pgvector + worker queue
  ↓                      Needed before data volume overwhelms SQLite.
Phase 3 (2–4 weeks):     A100 model training on MetaCentrum
  ↓                      Can start in parallel with Phase 2.
Phase 4 (1–2 weeks):     Integrate trained models + production hardening
```

Total: ~6–10 weeks to go from 224 articles to 100K+ with trained models.

