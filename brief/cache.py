from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from brief.artifacts import BriefSourceResolution
from brief.models import ResearchBrief


BRIEF_CACHE_TTL = timedelta(minutes=30)
CacheKey = tuple[str, str, str, str | None, str, str | None]

# This cache is intentionally process-local for the single-worker demo setup.
_brief_cache: dict[CacheKey, tuple[ResearchBrief, datetime]] = {}
_brief_support_cache: dict[CacheKey, tuple[dict[str, object], datetime]] = {}
_cache_lock = threading.Lock()


def _cache_key(
    topic: str,
    country: str,
    source: str,
    language: str | None,
    mode_key: str,
    run_id: str | None = None,
) -> CacheKey:
    return (topic, country, source, language, mode_key, run_id)


def _cache_mode_keys(source_mode: str) -> list[str]:
    if source_mode == "cluster_drift":
        return ["hierarchical_cluster", "cluster_drift"]
    return ["hierarchical_legacy", "legacy_drift"]


def _cache_mode_for_brief(
    brief: ResearchBrief,
    *,
    source_mode: str,
) -> str:
    if brief.generation_mode == "fallback" or brief.generation_mode is None:
        return source_mode
    return brief.generation_mode


def _get_cached_brief_for_resolution(
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> ResearchBrief | None:
    now = datetime.now(timezone.utc)
    with _cache_lock:
        for mode_key in _cache_mode_keys(resolution.source_mode):
            cached = _brief_cache.get(
                _cache_key(
                    resolution.display_topic,
                    resolution.country,
                    resolution.source,
                    resolution.language,
                    mode_key,
                    run_id,
                )
            )
            if cached and now - cached[1] < BRIEF_CACHE_TTL:
                return cached[0]
    return None


def _get_cached_brief_support_for_resolution(
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> dict[str, object] | None:
    now = datetime.now(timezone.utc)
    with _cache_lock:
        for mode_key in _cache_mode_keys(resolution.source_mode):
            cached = _brief_support_cache.get(
                _cache_key(
                    resolution.display_topic,
                    resolution.country,
                    resolution.source,
                    resolution.language,
                    mode_key,
                    run_id,
                )
            )
            if cached and now - cached[1] < BRIEF_CACHE_TTL:
                return dict(cached[0])
    return None


def _store_cached_brief(
    brief: ResearchBrief,
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> None:
    mode_key = _cache_mode_for_brief(brief, source_mode=resolution.source_mode)
    with _cache_lock:
        _brief_cache[
            _cache_key(
                resolution.display_topic,
                resolution.country,
                resolution.source,
                resolution.language,
                mode_key,
                run_id,
            )
        ] = (brief, datetime.now(timezone.utc))


def _store_cached_brief_support(
    support: dict[str, object],
    brief: ResearchBrief,
    resolution: BriefSourceResolution,
    *,
    run_id: str | None = None,
) -> None:
    mode_key = _cache_mode_for_brief(brief, source_mode=resolution.source_mode)
    with _cache_lock:
        _brief_support_cache[
            _cache_key(
                resolution.display_topic,
                resolution.country,
                resolution.source,
                resolution.language,
                mode_key,
                run_id,
            )
        ] = (dict(support), datetime.now(timezone.utc))


def clear_brief_cache() -> None:
    with _cache_lock:
        _brief_cache.clear()
        _brief_support_cache.clear()
