from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_country(value: str | None) -> str:
    return (value or "").strip().upper()


def normalize_source(value: str | None) -> str:
    return (value or "").strip().lower()


def normalize_language(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    return normalized or None


def model_dump(model: object) -> dict[str, object]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    if hasattr(model, "dict"):
        return model.dict(exclude_none=True)
    return dict(model)


def pipeline_stage(
    status: str,
    *,
    detail: str | None = None,
    count: int | None = None,
    duration_s: float | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "detail": detail,
        "count": count,
        "duration_s": duration_s,
        "metadata": metadata or {},
    }
