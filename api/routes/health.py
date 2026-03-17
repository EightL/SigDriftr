import json
import urllib.request

from fastapi import APIRouter

from db.init import get_conn


router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Report whether the database and local Ollama service are reachable."""
    checks: dict[str, object] = {}

    try:
        conn = get_conn()
        checks["db"] = "ok"
        checks["articles"] = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        checks["signals"] = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
    except Exception as exc:
        checks["db"] = f"error: {exc}"

    try:
        request = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=3) as response:
            models = json.loads(response.read().decode("utf-8")).get("models", [])
        checks["ollama"] = "ok"
        checks["ollama_models"] = [model.get("name") for model in models]
    except Exception as exc:
        checks["ollama"] = f"error: {exc}"

    overall = "ok" if all(
        value == "ok" or isinstance(value, (int, list)) for value in checks.values()
    ) else "degraded"
    return {"status": overall, **checks}
