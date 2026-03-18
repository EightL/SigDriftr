# SigDriftr

SigDriftr is a small FastAPI service that pulls Czech RSS articles, runs local LLM-based signal extraction on them, aggregates those signals into audience-segment profiles, measures drift against seeded baselines, and produces a short research brief.

It is a demo pipeline, not a production system. The code is built around a local SQLite database and a local Ollama server.

## What it does

For a given topic, the project does this:

1. Uses a persistent LinUCB bandit to choose a subset of Czech RSS feeds for the topic.
2. Filters articles for topic relevance.
3. Stores matching articles in SQLite.
4. Sends each unprocessed article title and summary to Ollama and asks for structured behavioral signals.
5. Stores one signal row per article, with optional named-entity enrichment when a Czech spaCy model is available.
6. Aggregates those signals into four audience segments:
   `young_urban`, `family`, `senior`, `b2b`
7. Compares the current segment profile to a seeded baseline for the same topic.
8. Produces:
   - raw per-article signals
   - per-segment calibration/drift data
   - a short research brief with three survey hypotheses
   - a simple browser UI at `/ui`

## What it does not do

This repo does not implement several things mentioned in the project context notes:

- no vector database
- no auth, jobs, queues, or multi-worker deployment story

The current implementation is still a small single-process demo pipeline.

## Architecture

![pipeline](/pipeline.drawio.pdf)


### 1. Storage

SQLite is initialized on first use in [`db/init.py`](/home/osml/code/ml/SigDriftr/db/init.py).

The database contains six tables:

- `articles`: RSS article metadata and the user-supplied topic used during collection
- `signals`: one extracted signal payload per article
- `article_entities`: optional named entities per article, populated when spaCy is available
- `segment_profiles`: aggregated segment-level metrics for a time window
- `baselines`: seeded baseline values per `(topic, segment)`
- `bandit_state`: persistent LinUCB arm parameters per outlet

SQLite is configured with WAL mode and a per-thread connection.

### 2. RSS ingestion

RSS ingestion lives in [`ingestion/crawler.py`](/home/osml/code/ml/SigDriftr/ingestion/crawler.py) and feed configuration lives in [`config/feeds.py`](/home/osml/code/ml/SigDriftr/config/feeds.py).

Current feeds are a fixed list of Czech outlets such as `irozhlas`, `wave`, `idnes`, `novinky`, `e15`, `blesk`, `maminka`, and `ct24`.
Each crawl selects only the top-scoring subset for the current topic and hour, then fetches those feeds concurrently.

Topic filtering is done in two passes:

- if the topic string appears directly in the article title or summary, the article is accepted
- otherwise the code tries a multilingual sentence-transformer similarity check using `paraphrase-multilingual-MiniLM-L12-v2`

If the embedding model cannot be loaded, the semantic fallback is effectively disabled and only direct string matches pass.
When a selected feed yields no relevant matches, the crawler writes a zero-reward update back into the bandit so poor sources are gradually deprioritized.

Relevant articles are inserted with `INSERT OR IGNORE`, keyed by a SHA-256 hash of the article URL.

### 3. Signal extraction

Signal extraction is split across:

- [`extraction/extractor.py`](/home/osml/code/ml/SigDriftr/extraction/extractor.py)
- [`extraction/llm_client.py`](/home/osml/code/ml/SigDriftr/extraction/llm_client.py)

The extractor reads articles that do not yet have a row in `signals`, sends the article title and summary to Ollama, and stores a normalized JSON result.
If `spacy` plus `cs_core_news_sm` are available, it also stores a compact list of named entities in `article_entities` and mirrors them into `raw_json`.

The primary extraction model is:

- `qwen2.5:7b-instruct`

Fallback model:

- `gemma3:1b`

The LLM is asked to return only this schema:

- `concern_level`
- `purchase_intent`
- `avoidance_signals`
- `dominant_frame`
- `seg_young_urban`
- `seg_family`
- `seg_senior`
- `seg_b2b`

Post-processing then:

- clamps numeric values to `0.0..1.0`
- restricts `dominant_frame` to `fear`, `opportunity`, `conflict`, or `neutral`
- softmax-normalizes the segment weights
- blends segment weights with a fixed feed affinity prior
- writes per-outlet reward updates back to the feed bandit based on the extracted signal density

That last step matters: outlet-specific priors from [`config/feeds.py`](/home/osml/code/ml/SigDriftr/config/feeds.py) bias the segment relevance scores so, for example, some feeds lean more toward `b2b` or `family`.

### 4. Segment aggregation

Segment aggregation lives in [`delta/mapper.py`](/home/osml/code/ml/SigDriftr/delta/mapper.py).

For each segment, the code computes weighted averages of:

- `concern_level`
- `purchase_intent`
- `avoidance_signals`

The weights are the per-article `seg_*` relevance values produced by extraction.

It also computes a dominant frame per segment by weighted vote and stores the result in `segment_profiles`.

The default analysis window is the last 7 days.

### 5. Drift calculation

Drift logic lives in [`delta/engine.py`](/home/osml/code/ml/SigDriftr/delta/engine.py), with baseline seeding in [`delta/seeder.py`](/home/osml/code/ml/SigDriftr/delta/seeder.py).

For each `(topic, segment)` pair:

- baselines are seeded on demand if missing
- current segment signals are compared against the stored baseline
- deltas are computed for the three numeric signals
- drift magnitude is a domain-weighted sum of absolute deltas
- frame shift is `true` if the current dominant frame differs from the baseline frame
- confidence and learned-baseline metadata are attached to every segment result

Alert levels are:

- `no_data` if no articles contribute to the segment
- `none` if drift magnitude is below `0.20`
- `mild` if drift magnitude is at least `0.20`
- `strong` if drift magnitude is at least `0.45`

Baseline learning is triggered from the extraction route after segment profiles are recomputed.

### 6. Brief generation

Brief generation lives in:

- [`brief/generator.py`](/home/osml/code/ml/SigDriftr/brief/generator.py)
- [`brief/prompt.py`](/home/osml/code/ml/SigDriftr/brief/prompt.py)
- [`brief/models.py`](/home/osml/code/ml/SigDriftr/brief/models.py)

The brief pipeline:

1. computes drift
2. ranks segments by drift magnitude and article count
3. fetches a few representative articles per segment
4. builds a prompt with the drift evidence and article snippets
5. asks Ollama for a strict JSON brief
6. validates the output with Pydantic

The expected brief shape is:

- `headline`
- `narrative`
- `most_affected_segment`
- `drift_type`
- `alert_level`
- exactly 3 `hypotheses`
- `generated_at`
- `model_used`

The brief endpoint is cached in-process for 30 minutes. If Ollama returns invalid JSON or a schema-breaking response, the code generates a deterministic fallback brief from the computed drift data instead of failing the request.

## API

The FastAPI app is defined in [`main.py`](/home/osml/code/ml/SigDriftr/main.py).

Available routes:

- `POST /collect?topic=...`
  - crawls the configured RSS feeds
  - stores relevant articles
  - returns `{ "inserted": <int>, "topic": <str> }`

- `POST /extract?topic=...`
  - extracts signals for articles that do not already have signal rows
  - topic may be empty to process all stored articles
  - returns `{ "processed": <int>, "topic": <str> }`

- `GET /signals?topic=...`
  - returns stored article-level signals ordered by newest extraction first

- `GET /calibration/{topic}/{segment}`
  - returns the aggregated current profile for one segment

- `GET /drift/{topic}?days_back=7`
  - returns segment drift versus baseline for the topic

- `GET /brief/{topic}`
  - returns a validated `ResearchBrief`

- `GET /health`
  - checks SQLite availability
  - reports article and signal counts
  - checks whether Ollama is reachable and lists local models

- `GET /ui`
  - serves the static demo UI in [`static/index.html`](/home/osml/code/ml/SigDriftr/static/index.html)

Special topic handling:

- the calibration and brief code treats `_all` as “all topics”
- other endpoints mostly expect the literal topic string you stored on collection

## Demo UI

The UI is a single static file at [`static/index.html`](/home/osml/code/ml/SigDriftr/static/index.html).

It calls the backend directly and runs this sequence when you click `Analyze`:

1. `POST /collect`
2. `POST /extract`
3. `GET /drift/{topic}`
4. `GET /signals?topic=...`
5. `GET /brief/{topic}`

It then renders:

- a health/status bar
- a drift table
- a radar chart for one selected segment
- a recent signals log
- a post-processed brief layout

## Running it

### Requirements

- Python 3.11+
- a local Ollama server on `http://localhost:11434`
- the Ollama models used by the code:
  - `qwen2.5:7b-instruct`
  - `gemma3:1b`

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the API

```bash
uvicorn main:app --reload
```

Open:

- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Demo UI: [http://localhost:8000/ui](http://localhost:8000/ui)

### Typical manual flow

From the UI:

1. enter a topic such as `energie`
2. click `Analyze`

Or via HTTP:

```bash
curl -X POST "http://localhost:8000/collect?topic=energie"
curl -X POST "http://localhost:8000/extract?topic=energie"
curl "http://localhost:8000/drift/energie"
curl "http://localhost:8000/signals?topic=energie"
curl "http://localhost:8000/brief/energie"
```

## Dependencies

Current Python dependencies from [`requirements.txt`](/home/osml/code/ml/SigDriftr/requirements.txt):

- `fastapi`
- `pydantic`
- `uvicorn`
- `feedparser`
- `tenacity`
- `sentence-transformers`
- `transformers>=4.40`
- `torch`
- `aiofiles`

Notes:

- `sentence-transformers`, `transformers`, and `torch` are only needed for the semantic relevance fallback during ingestion
- the service also depends on Ollama, but that is not managed by `pip`

## Tests

The repo contains a mix of unit tests and small integration-style scripts:

- [`test_extraction_api.py`](/home/osml/code/ml/SigDriftr/test_extraction_api.py): route-level and extraction utility tests
- [`test_delta.py`](/home/osml/code/ml/SigDriftr/test_delta.py): segment aggregation, drift, alerting, baseline seeding
- [`test_brief.py`](/home/osml/code/ml/SigDriftr/test_brief.py): brief validation, normalization, fallback behavior, cache handling
- [`test_ui.py`](/home/osml/code/ml/SigDriftr/test_ui.py): static UI route smoke test
- [`test_ingestion.py`](/home/osml/code/ml/SigDriftr/test_ingestion.py) and [`test_signals.py`](/home/osml/code/ml/SigDriftr/test_signals.py): command-line scripts for exercising a running API

Run the actual pytest-based tests with:

```bash
pytest -q
```

## Current limitations

- collection is synchronous and serial
- extraction is synchronous and serial
- the system stores whatever topic string was used at collection time; there is no topic normalization
- collection quality depends heavily on RSS title/summary text and direct topic string matches
- semantic filtering silently degrades if the embedding model fails to load
- drift baselines are hard-coded seed values, not learned from historical data
- brief caching is process-local
- there is no background processing, deduplicated job system, or persistence layer beyond SQLite
- there is no authentication or multi-user separation

## File map

- [`main.py`](/home/osml/code/ml/SigDriftr/main.py): FastAPI app and static mount
- [`db/init.py`](/home/osml/code/ml/SigDriftr/db/init.py): SQLite schema and connection management
- [`config/feeds.py`](/home/osml/code/ml/SigDriftr/config/feeds.py): RSS feed list and outlet affinity tags
- [`ingestion/crawler.py`](/home/osml/code/ml/SigDriftr/ingestion/crawler.py): RSS crawl and topic relevance filtering
- [`extraction/extractor.py`](/home/osml/code/ml/SigDriftr/extraction/extractor.py): fetch unprocessed articles and persist signal rows
- [`extraction/llm_client.py`](/home/osml/code/ml/SigDriftr/extraction/llm_client.py): Ollama request, normalization, segment priors
- [`delta/mapper.py`](/home/osml/code/ml/SigDriftr/delta/mapper.py): segment aggregation
- [`delta/engine.py`](/home/osml/code/ml/SigDriftr/delta/engine.py): baseline comparison and alerting
- [`delta/seeder.py`](/home/osml/code/ml/SigDriftr/delta/seeder.py): default seeded baselines
- [`brief/models.py`](/home/osml/code/ml/SigDriftr/brief/models.py): brief schema
- [`brief/prompt.py`](/home/osml/code/ml/SigDriftr/brief/prompt.py): prompt construction
- [`brief/generator.py`](/home/osml/code/ml/SigDriftr/brief/generator.py): brief generation, normalization, fallback, cache
- [`api/routes/`](/home/osml/code/ml/SigDriftr/api/routes): HTTP endpoints
- [`static/index.html`](/home/osml/code/ml/SigDriftr/static/index.html): demo UI
