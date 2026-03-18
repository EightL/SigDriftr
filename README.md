# SigDriftr

Minimal demo pipeline for Czech RSS collection, local signal extraction, drift detection, and short brief generation.

## File architecture

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

## Use

- Open API docs: `http://localhost:8000/docs`
- Demo UI: `http://localhost:8000/ui`
- Endpoints and examples: [docs/API.md](docs/API.md)

## Docs

- Technical design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Developer guide: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- Limitations and roadmap: [docs/concerns.md](docs/concerns.md)

