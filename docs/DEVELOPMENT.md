# SigDriftr Development Guide

## Quick Start

Get up and running in 5 minutes:

```bash
# Clone and setup
git clone https://github.com/EightL/SigDriftr.git
cd SigDriftr
python3 -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)
pip install -r requirements.txt

# Start API
uvicorn main:app --reload

# Open browser
http://localhost:8000/docs  # OpenAPI UI
http://localhost:8000/ui    # Demo dashboard
```

---

## Prerequisites

### Required
- **Python 3.11+**
  - macOS: `brew install python@3.11`
  - Linux: `apt-get install python3.11`
  - Windows: Download from [python.org](https://www.python.org/downloads/)

- **Ollama** (local LLM server)
  - Download: [ollama.ai](https://ollama.ai)
  - Start: `ollama serve`
  - Pull models:
    ```bash
    ollama pull qwen2.5:7b-instruct
    ollama pull gemma3:1b
    ```
  - Verify: `curl http://localhost:11434/api/tags`

### Optional
- **spaCy** (Czech NER enrichment)
  ```bash
  python -m spacy download cs_core_news_sm
  ```

### External Dependencies Graph
```
fastapi (web framework)
├─ pydantic (validation)
├─ uvicorn (ASGI server)
└─ starlette (async)

sentence-transformers (semantic similarity)
├─ torch (GPU optional, CPU ok)
├─ transformers (model loading)
└─ numpy (numerics)

feedparser (RSS parsing)
torch (LLM dependencies)
spacy (NER, optional)
aiofiles (async file I/O)
tenacity (retry logic)
```

---

## Development Setup

### 1. Virtual Environment

```bash
# Create
python3 -m venv .venv

# Activate
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# Verify
which python  # Should show /path/to/.venv/bin/python
```

### 2. Install Dependencies

```bash
# Production dependencies
pip install -r requirements.txt

# Development dependencies
pip install pytest pytest-cov black flake8 mypy

# Verify
python -c "import fastapi; print(fastapi.__version__)"
```

### 3. Database Setup

SQLite database is created automatically on first run:
```bash
# Verify
sqlite3 sigdriftr.db ".tables"
# Should output: articles  article_entities  bandit_state  baselines  segment_profiles  signals
```

---

## Running the Application

### Development Mode (with auto-reload)

```bash
uvicorn main:app --reload
```

**Output:**
```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

**Endpoints:**
- OpenAPI docs: http://localhost:8000/docs
- ReDoc docs: http://localhost:8000/redoc
- Demo UI: http://localhost:8000/ui

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Common Issues

**"Address already in use":**
```bash
# Kill existing process
lsof -i :8000  # Find PID
kill -9 <PID>

# Or use different port
uvicorn main:app --port 8001
```

**"Ollama connection refused":**
```bash
# Start Ollama in another terminal
ollama serve

# Verify
curl http://localhost:11434/api/tags
```

---

## Testing

### Run All Tests

```bash
pytest -v
pytest -q              # Quiet mode
pytest --cov=.         # With coverage report
pytest -x              # Stop on first failure
pytest -k signal       # Run tests matching pattern
```

### Test Files

```
test_extraction_api.py    - API route handlers + signal schema
test_delta.py             - Segment aggregation + drift
test_brief.py             - Brief generation + fallback
test_bandit.py            - LinUCB bandit arm selection
test_pipeline.py          - End-to-end workflow
test_e2e.py               - Full integration tests
... (6 more files)
```

### Write a New Test

**Pattern:**
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_my_feature():
    # Arrange
    topic = "test_topic"
    
    # Act
    response = client.post(f"/collect?topic={topic}")
    
    # Assert
    assert response.status_code == 200
    assert response.json()["topic"] == topic
```

**Run:**
```bash
pytest test_my_test.py::test_my_feature -v
```

### Fixtures

Common test fixtures in `conftest.py`:
```python
@pytest.fixture
def test_db():
    # In-memory SQLite
    ...

@pytest.fixture
def mock_ollama(monkeypatch):
    # Mock LLM responses
    ...
```

---

## Code Organization

```
SigDriftr/
├── main.py                     # FastAPI app entry point
├── requirements.txt            # Dependencies
├── sigdriftr.db               # SQLite database (auto-created)
│
├── api/                       # HTTP layer
│   ├── routes/
│   │   ├── collect.py        # POST /collect
│   │   ├── extract.py        # POST /extract
│   │   ├── signals.py        # GET /signals
│   │   ├── calibration.py    # GET /calibration
│   │   ├── brief.py          # GET /brief
│   │   ├── health.py         # GET /health
│   │   └── history.py        # GET /history (internal)
│   └── scheduler.py           # Background job scaffolding
│
├── db/                        # Data access layer
│   └── init.py               # SQLite schema + connection pooling
│
├── config/                    # Configuration
│   └── feeds.py              # RSS feeds + segment priors
│
├── ingestion/                 # Step 1: RSS collection
│   ├── crawler.py            # Main collection logic
│   └── bandit.py             # LinUCB bandit
│
├── extraction/                # Step 2: Signal extraction
│   ├── extractor.py          # Orchestration
│   ├── llm_client.py         # Ollama integration
│   └── entities.py           # spaCy enrichment
│
├── delta/                     # Step 3: Drift detection
│   ├── mapper.py             # Aggregation
│   ├── engine.py             # Drift calculation
│   └── seeder.py             # Baseline management
│
├── brief/                     # Step 4: Brief generation
│   ├── models.py             # Pydantic schemas
│   ├── prompt.py             # Prompt construction
│   └── generator.py          # Generation + caching
│
├── static/                    # Frontend
│   └── index.html            # Demo dashboard
│
├── test_*.py                  # Test suite (12 files)
├── conftest.py               # Pytest fixtures
│
├── ARCHITECTURE.md           # Technical design
├── API.md                    # Endpoint reference
├── DEVELOPMENT.md            # This file
├── concerns.md               # Limitations + roadmap
└── README.md                 # Project overview
```

---

## Common Development Tasks

### Add New RSS Feed

1. Edit `config/feeds.py`:
   ```python
   FEEDS = {
       ...
       "new_feed": {
           "url": "https://example.com/feed.xml",
           "outlet": "new_feed",
           "segment_priors": {"young_urban": 0.3, "family": 0.3, "senior": 0.2, "b2b": 0.2}
       }
   }
   ```

2. Test:
   ```bash
   curl -X POST "http://localhost:8000/collect?topic=test"
   # Check that new_feed articles appear
   ```

### Add New Segment

1. Edit `config/feeds.py`: Add `seg_new` to feed priors
2. Edit `brief/models.py`: Add `seg_new` to signal schema
3. Edit `extraction/llm_client.py`: Update `SEGMENT_KEYS`, `DEFAULT_SIGNALS`,
   and the prompt to include both `seg_new_relevance` and the derived `seg_new`
   aggregation share
4. Edit `delta/mapper.py`: Add aggregation logic
5. Edit `evaluation/weak_gold.py` and `docs/LABEL_GUIDE.md` if the segment
   should be evaluated
6. Test: Run `/extract`, `/drift`, and the semi-gold evaluator

### Change LLM Model

1. Edit `extraction/llm_client.py`:
   ```python
   PRIMARY_MODEL = "mistral:7b-instruct"  # Change from qwen
   ```

2. Pull model:
   ```bash
   ollama pull mistral:7b-instruct
   ```

3. Test extraction:
   ```bash
   curl -X POST "http://localhost:8000/extract?topic=test"
   ```

### Debug Signal Extraction

```python
# In Python REPL
from extraction.llm_client import extract_article_signals

response = extract_article_signals(
    title="Nová solární farma",
    summary="Největší projekt..."
)
print(response)  # See raw LLM output
```

### Check Database State

```bash
# View schema
sqlite3 sigdriftr.db ".schema"

# Count articles by topic
sqlite3 sigdriftr.db "SELECT topic, COUNT(*) FROM articles GROUP BY topic;"

# View signals for topic
sqlite3 sigdriftr.db "SELECT article_id, concern_level, seg_young_urban FROM signals LIMIT 5;"

# Reset database
rm sigdriftr.db  # Will be recreated on next run
```

### Profile Performance

```python
import cProfile
import pstats

cProfile.run('extract_article_signals(title, summary)', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)  # Top 10 functions
```

---

## Code Style

### Formatting

```bash
# Auto-format with Black
black .

# Check formatting
black --check .
```

### Linting

```bash
# Lint with flake8
flake8 .

# Ignore specific issues
flake8 . --ignore=E501,W503  # Long lines, operator precedence
```

### Type Hints

```python
# Encouraged
def extract_signals(topic: str) -> Dict[str, Any]:
    pass

# Optional validation
mypy .  # Type checking
```

### Docstring Format

```python
def extract_article_signals(title: str, summary: str) -> Dict[str, float]:
    """
    Extract behavioral signals from article text using local LLM.
    
    Args:
        title: Article title
        summary: Article summary
    
    Returns:
        Dict with keys: concern_level, purchase_intent, avoidance_signals,
        dominant_frame, seg_*_relevance, seg_young_urban, seg_family,
        seg_senior, seg_b2b
    
    Raises:
        TimeoutError: If Ollama doesn't respond within 30 seconds
        ValueError: If LLM returns invalid JSON
    """
    pass
```

---

## Debugging

### Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug(f"Processing article: {title}")
```

### Check Ollama Status

```bash
# List available models
curl http://localhost:11434/api/tags | python -m json.tool

# Check server health
curl -I http://localhost:11434/api/tags
```

### Inspect API Requests/Responses

```bash
# Verbose curl
curl -v -X POST "http://localhost:8000/collect?topic=test"

# Pretty-print JSON
curl "http://localhost:8000/brief/test" | python -m json.tool
```

### Database Inspection

```bash
# Open interactive SQLite shell
sqlite3 sigdriftr.db

# Then:
.mode column
SELECT topic, COUNT(*) as count FROM articles GROUP BY topic;
SELECT * FROM signals WHERE article_id = 'xyz' LIMIT 1;
```

---

## Performance Troubleshooting

### Slow /extract

**Cause:** LLM inference bottleneck  
**Solution:**
- Smaller model: `gemma3:1b` instead of `qwen2.5:7b`
- GPU acceleration: Ensure CUDA available
- Batch processing: Implement in future version

### Slow /collect

**Cause:** Feed fetching timeout  
**Solution:**
- Reduce feed timeout from 10 to 5 seconds (in `crawler.py`)
- Skip slow feeds temporarily
- Use RSS feed CDN/cache

### Slow /signals query

**Cause:** Missing database index  
**Solution:**
```sql
CREATE INDEX idx_articles_topic ON articles(topic);
CREATE INDEX idx_signals_article_id ON signals(article_id);
```

---

## Deployment

### Local Production

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
```

### Docker (Future)

```dockerfile
FROM python:3.11
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
```

### Systemd Service (Linux)

```ini
[Unit]
Description=SigDriftr API
After=network.target

[Service]
Type=notify
User=sigdriftr
WorkingDirectory=/opt/sigdriftr
ExecStart=/opt/sigdriftr/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## References & Navigation

- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) - Complete technical design
- **API Reference:** [API.md](API.md) - All 8 endpoints with examples
- **Known Limits:** [concerns.md](concerns.md) - Limitations and roadmap
- **Project README:** [README.md](README.md) - Overview and quick start
- **GitHub:** https://github.com/EightL/SigDriftr
