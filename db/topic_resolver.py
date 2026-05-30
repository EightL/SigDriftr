from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import db.init as db_init
from config.topics import (
    display_name_for_topic,
    domain_for_topic,
    normalize_topic,
    slugify_topic,
)


@dataclass(frozen=True)
class TopicResolution:
    requested_topic: str
    normalized_topic: str
    canonical_topic_id: str
    display_name: str
    domain: str
    language: str | None
    aliases: tuple[str, ...]
    was_created: bool


def clear_topic_resolution_cache() -> None:
    _lookup_existing_cached.cache_clear()


def resolve_topic(
    raw_topic: str,
    language: str | None = None,
    *,
    create: bool = True,
) -> TopicResolution:
    """Resolve user-facing topic text into the canonical topic catalog."""
    db_path = str(db_init.DB_PATH)
    requested_topic = " ".join((raw_topic or "").strip().split())
    normalized_topic = normalize_topic(requested_topic)
    normalized_language = (language or "").strip().lower()
    resolved_language = normalized_language or None
    if not normalized_topic:
        return TopicResolution(
            requested_topic=requested_topic,
            normalized_topic="",
            canonical_topic_id="",
            display_name="Topic",
            domain="generic",
            language=resolved_language,
            aliases=(),
            was_created=False,
        )

    cached = _lookup_existing_cached(db_path, normalized_topic, normalized_language)
    if cached is not None:
        return _resolution_from_topic_data(
            cached,
            requested_topic=requested_topic,
            normalized_topic=normalized_topic,
            language=resolved_language,
            was_created=False,
        )

    if not create:
        canonical_topic_id = slugify_topic(requested_topic)
        return TopicResolution(
            requested_topic=requested_topic,
            normalized_topic=normalized_topic,
            canonical_topic_id=canonical_topic_id,
            display_name=display_name_for_topic(requested_topic),
            domain=domain_for_topic(requested_topic),
            language=resolved_language,
            aliases=(requested_topic.lower(),),
            was_created=False,
        )

    conn = db_init.get_conn()
    row = _create_unknown_topic(
        conn,
        requested_topic,
        normalized_topic,
        resolved_language,
    )
    clear_topic_resolution_cache()
    aliases = _aliases_for_topic(conn, str(row["canonical_topic_id"]))
    return TopicResolution(
        requested_topic=requested_topic,
        normalized_topic=normalized_topic,
        canonical_topic_id=str(row["canonical_topic_id"]),
        display_name=str(row["display_name"]),
        domain=str(row["domain"] or "generic"),
        language=resolved_language,
        aliases=aliases,
        was_created=True,
    )


@lru_cache(maxsize=512)
def _lookup_existing_cached(
    _db_path: str,
    normalized_topic: str,
    normalized_language: str,
) -> dict | None:
    conn = db_init.get_conn()
    row = _lookup_alias(conn, normalized_topic, normalized_language or None)
    if row is None:
        return None

    canonical_topic_id = str(row["canonical_topic_id"])
    if row["status"] == "merged" and row["merged_into"]:
        merged_row = _lookup_topic(conn, str(row["merged_into"]))
        if merged_row is not None:
            row = merged_row
            canonical_topic_id = str(row["canonical_topic_id"])

    return {
        "canonical_topic_id": canonical_topic_id,
        "display_name": str(row["display_name"]),
        "domain": str(row["domain"] or "generic"),
        "aliases": _aliases_for_topic(conn, canonical_topic_id),
    }


def _resolution_from_topic_data(
    data: dict,
    *,
    requested_topic: str,
    normalized_topic: str,
    language: str | None,
    was_created: bool,
) -> TopicResolution:
    return TopicResolution(
        requested_topic=requested_topic,
        normalized_topic=normalized_topic,
        canonical_topic_id=str(data["canonical_topic_id"]),
        display_name=str(data["display_name"]),
        domain=str(data["domain"] or "generic"),
        language=language,
        aliases=tuple(data["aliases"]),
        was_created=was_created,
    )


def _lookup_alias(conn, normalized_topic: str, language: str | None) -> dict | None:
    normalized_language = language or ""
    row = conn.execute(
        """
        SELECT
            t.canonical_topic_id,
            t.display_name,
            t.domain,
            t.status,
            t.merged_into
        FROM topic_aliases a
        JOIN topics t ON t.canonical_topic_id = a.canonical_topic_id
        WHERE a.normalized_topic = ?
          AND (? = '' OR a.language = ? OR a.language IS NULL)
        ORDER BY
          CASE
              WHEN a.language = ? THEN 0
              WHEN a.language IS NULL THEN 1
              ELSE 2
          END,
          a.id ASC
        LIMIT 1
        """,
        (
            normalized_topic,
            normalized_language,
            normalized_language,
            normalized_language,
        ),
    ).fetchone()
    return _row_to_topic_dict(row)


def _lookup_topic(conn, canonical_topic_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT canonical_topic_id, display_name, domain, status, merged_into
        FROM topics
        WHERE canonical_topic_id = ?
        """,
        (canonical_topic_id,),
    ).fetchone()
    return _row_to_topic_dict(row)


def _row_to_topic_dict(row) -> dict | None:
    if row is None:
        return None
    return {
        "canonical_topic_id": row[0],
        "display_name": row[1],
        "domain": row[2],
        "status": row[3],
        "merged_into": row[4],
    }


def _create_unknown_topic(
    conn,
    requested_topic: str,
    normalized_topic: str,
    language: str | None,
) -> dict:
    canonical_topic_id = slugify_topic(requested_topic)
    display_name = display_name_for_topic(requested_topic)
    domain = domain_for_topic(requested_topic)
    conn.execute(
        """
        INSERT OR IGNORE INTO topics
        (canonical_topic_id, display_name, domain, status, updated_at)
        VALUES (?, ?, ?, 'active', datetime('now'))
        """,
        (canonical_topic_id, display_name, domain),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO topic_aliases
        (canonical_topic_id, raw_topic, normalized_topic, language, source)
        VALUES (?, ?, ?, ?, 'user_input')
        """,
        (
            canonical_topic_id,
            requested_topic.lower(),
            normalized_topic,
            language,
        ),
    )
    conn.commit()
    row = _lookup_topic(conn, canonical_topic_id)
    if row is None:
        raise RuntimeError(f"Failed to create canonical topic {canonical_topic_id!r}.")
    return row


def _aliases_for_topic(conn, canonical_topic_id: str) -> tuple[str, ...]:
    rows = conn.execute(
        """
        SELECT raw_topic
        FROM topic_aliases
        WHERE canonical_topic_id = ?
        ORDER BY
          CASE source WHEN 'curated_seed' THEN 0 ELSE 1 END,
          raw_topic ASC
        """,
        (canonical_topic_id,),
    ).fetchall()
    aliases = tuple(dict.fromkeys(str(row[0]) for row in rows if str(row[0]).strip()))
    return aliases
