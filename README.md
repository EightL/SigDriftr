# SigDriftr

Experimental FastAPI pipeline for RSS collection, LLM signal extraction,
drift detection, storyline clustering, and short analyst brief generation.
It analyzes media coverage drift, not direct public sentiment or behavior.

## Flow

1. `ingestion/` selects RSS feeds with a contextual bandit, fetches articles, and
   keeps topic-relevant matches.
2. `extraction/` sends article text to Google Gemma when `GOOGLE_GEMMA_API_KEY`
   is configured, falls back to local Ollama, and stores normalized behavioral
   signals in SQLite.
3. `delta/` aggregates article signals into audience segment profiles and compares
   them with learned or seeded baselines.
4. `extraction/embedding_service.py` embeds articles for cluster analysis.
5. `clustering/` reduces embeddings with UMAP and groups them with HDBSCAN.
6. `extraction/cluster_extractor.py` and `delta/cluster_drift.py` summarize and
   track cluster-level storyline drift.
7. `brief/` turns drift evidence into a structured research brief with deterministic
   fallback behavior when the local LLM is unavailable.
8. `api/` and `static/` expose the API and demo dashboard.

## File architecture

![pipeline](/plan.svg)

- `main.py` exposes the FastAPI app.
- `ingestion/` collects and filters RSS articles.
- `extraction/` turns articles into structured signals.
- `delta/` aggregates signals and computes drift.
- `brief/` builds the final research brief.
- `db/` stores the SQLite schema and queries.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Test

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Evaluation

The current regression target is a 200-article CZE-NEC semi-gold baseline:
`eval/annotations/czech_batch_200_semi_gold.csv`. It combines a human annotation
pass with manual adjudication of the lowest-agreement model/human cases.

```bash
python scripts/evaluate_weak_gold.py \
  --predictions eval/model_outputs/czenec_batch_200_chatgpt.json
```

## Docker

```bash
docker build -t sigdriftr .
docker run --rm -p 8000:8000 sigdriftr
```

The app uses Google Gemma for article signal extraction when
`GOOGLE_GEMMA_API_KEY` is set. Ollama at `http://localhost:11434` is still used
as the article extraction fallback and for brief generation. Without either
provider, deterministic fallback paths still cover parts of the API, but article
and cluster signal extraction will not produce model-backed signals.

## Use

- Open API docs: `http://localhost:8000/docs`
- Demo UI: `http://localhost:8000/ui`
- Endpoints and examples: [docs/API.md](docs/API.md)

## Docs

- Technical design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Developer guide: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- Evaluation guide: [docs/EVALUATION.md](docs/EVALUATION.md)
- Label guide: [docs/LABEL_GUIDE.md](docs/LABEL_GUIDE.md)
- Limitations and roadmap: [concerns.md](concerns.md)
