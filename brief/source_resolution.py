from __future__ import annotations

from brief.artifacts import BriefSourceResolution
from brief.models import BriefSourceScope


def _normalize_country(country: str | None) -> str:
    return (country or "").strip().upper()


def _normalize_source(source: str | None) -> str:
    return (source or "").strip().lower()


def _normalize_language(language: str | None) -> str | None:
    normalized = (language or "").strip().lower()
    return normalized or None


def _source_scope(resolution: BriefSourceResolution) -> BriefSourceScope:
    return BriefSourceScope(
        country=resolution.country,
        source=resolution.source,
        language=resolution.language,
    )
