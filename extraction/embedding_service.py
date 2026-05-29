from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

from config.settings import EMBED_BATCH_SIZE, EMBED_LIMIT_DEFAULT
from db.init import get_conn
from db.topic_queries import topic_filter_sql
from extraction.embedder import (
    embed_texts,
    get_expected_dim,
    get_model_name,
    get_model_version,
)
from extraction.embedding_text import (
    build_embedding_text,
    build_embedding_text_candidate,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_articles(
    *,
    limit: int,
    topic: str | None,
    country: str | None,
    source: str | None,
) -> list[dict]:
    conn = get_conn()
    topic_sql, topic_params = topic_filter_sql("a", topic or "")
    query = f"""
        SELECT
            a.id,
            a.title,
            a.summary,
            a.body,
            a.language,
            a.outlet,
            a.country,
            a.published_at,
            e.id,
            e.embedding_text_hash,
            e.status
        FROM articles a
        LEFT JOIN article_embeddings e
          ON e.id = (
                SELECT e2.id
                FROM article_embeddings e2
                WHERE e2.article_id = a.id
                  AND e2.model_name = ?
                ORDER BY COALESCE(e2.embedded_at, e2.updated_at, e2.created_at) DESC,
                         e2.id DESC
                LIMIT 1
          )
        WHERE 1 = 1
          {topic_sql}
    """
    params: list[object] = [get_model_name(), *topic_params]
    if country:
        query += " AND a.country = ?"
        params.append(country.strip().upper())
    if source:
        query += " AND LOWER(a.outlet) = ?"
        params.append(source.strip().lower())
    query += """
        ORDER BY COALESCE(a.published_at, a.fetched_at) DESC, a.id DESC
        LIMIT ?
    """
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    return [
        {
            "article_id": row[0],
            "title": row[1] or "",
            "summary": row[2] or "",
            "body": row[3] or "",
            "language": row[4],
            "outlet": row[5],
            "country": row[6],
            "published_at": row[7],
            "latest_embedding_id": row[8],
            "latest_hash": row[9],
            "latest_status": row[10],
        }
        for row in rows
    ]


def _record_embedding_failure(
    *,
    article_id: str,
    embedding_text: str,
    language: str | None,
    error_message: str,
) -> None:
    conn = get_conn()
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO article_embeddings
        (article_id, model_name, model_version, embedding_dim, embedding_vector,
         embedding_text, embedding_text_hash, language, status, error_message,
         embedded_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'failed', ?, NULL, ?, ?)
        ON CONFLICT(article_id, model_name, embedding_text_hash) DO UPDATE SET
            model_version = excluded.model_version,
            embedding_dim = excluded.embedding_dim,
            embedding_vector = excluded.embedding_vector,
            embedding_text = excluded.embedding_text,
            language = excluded.language,
            status = 'failed',
            error_message = excluded.error_message,
            embedded_at = NULL,
            updated_at = excluded.updated_at
        """,
        (
            article_id,
            get_model_name(),
            get_model_version(),
            get_expected_dim(),
            "[]",
            embedding_text,
            _hash_text(embedding_text),
            language,
            error_message[:1000],
            now,
            now,
        ),
    )


def _upsert_complete_embedding(
    *,
    article_id: str,
    embedding_text: str,
    language: str | None,
    vector: list[float],
) -> None:
    conn = get_conn()
    now = _utc_now_iso()
    conn.execute(
        """
        INSERT INTO article_embeddings
        (article_id, model_name, model_version, embedding_dim, embedding_vector,
         embedding_text, embedding_text_hash, language, status, error_message,
         embedded_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'complete', NULL, ?, ?, ?)
        ON CONFLICT(article_id, model_name, embedding_text_hash) DO UPDATE SET
            model_version = excluded.model_version,
            embedding_dim = excluded.embedding_dim,
            embedding_vector = excluded.embedding_vector,
            embedding_text = excluded.embedding_text,
            language = excluded.language,
            status = 'complete',
            error_message = NULL,
            embedded_at = excluded.embedded_at,
            updated_at = excluded.updated_at
        """,
        (
            article_id,
            get_model_name(),
            get_model_version(),
            get_expected_dim(),
            json.dumps(vector),
            embedding_text,
            _hash_text(embedding_text),
            language,
            now,
            now,
            now,
        ),
    )


def _embed_items(items: list[dict]) -> tuple[int, int]:
    embedded = 0
    failed = 0
    if not items:
        return embedded, failed

    for start in range(0, len(items), EMBED_BATCH_SIZE):
        batch = items[start:start + EMBED_BATCH_SIZE]
        texts = [item["embedding_text"] for item in batch]
        try:
            vectors = embed_texts(texts, batch_size=EMBED_BATCH_SIZE)
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding count does not match batch size.")
            for item, vector in zip(batch, vectors):
                _upsert_complete_embedding(
                    article_id=item["article_id"],
                    embedding_text=item["embedding_text"],
                    language=item["language"],
                    vector=vector,
                )
                embedded += 1
            continue
        except Exception:
            pass

        for item in batch:
            try:
                vector = embed_texts([item["embedding_text"]], batch_size=1)[0]
                _upsert_complete_embedding(
                    article_id=item["article_id"],
                    embedding_text=item["embedding_text"],
                    language=item["language"],
                    vector=vector,
                )
                embedded += 1
            except Exception as exc:
                _record_embedding_failure(
                    article_id=item["article_id"],
                    embedding_text=item["embedding_text"],
                    language=item["language"],
                    error_message=str(exc),
                )
                failed += 1

    return embedded, failed


def embed_pending_articles(
    *,
    limit: int = EMBED_LIMIT_DEFAULT,
    topic: str | None = None,
    country: str | None = None,
    source: str | None = None,
) -> dict[str, object]:
    start = time.perf_counter()
    rows = _select_articles(limit=limit, topic=topic, country=country, source=source)

    already_current = 0
    retried_failed = 0
    stale_reembedded = 0
    failed = 0
    work_items: list[dict[str, object]] = []

    for row in rows:
        candidate_text = build_embedding_text_candidate(
            title=row["title"],
            body=row["body"],
            summary=row["summary"],
        )
        try:
            embedding_text = build_embedding_text(
                title=row["title"],
                body=row["body"],
                summary=row["summary"],
            )
        except ValueError as exc:
            _record_embedding_failure(
                article_id=row["article_id"],
                embedding_text=candidate_text,
                language=row["language"],
                error_message=str(exc),
            )
            failed += 1
            continue

        current_hash = _hash_text(embedding_text)
        latest_hash = row["latest_hash"]
        latest_status = row["latest_status"]

        if row["latest_embedding_id"] is None:
            reason = "missing"
        elif latest_hash == current_hash and latest_status == "complete":
            already_current += 1
            continue
        elif latest_hash == current_hash:
            reason = "retry_failed"
            retried_failed += 1
        else:
            reason = "stale"
            stale_reembedded += 1

        work_items.append(
            {
                "article_id": row["article_id"],
                "embedding_text": embedding_text,
                "language": row["language"],
                "reason": reason,
            }
        )

    embedded, model_failures = _embed_items(work_items)
    failed += model_failures

    duration_s = round(time.perf_counter() - start, 4)
    return {
        "model_name": get_model_name(),
        "embedding_dim": get_expected_dim(),
        "selected": len(rows),
        "embedded": embedded,
        "already_current": already_current,
        "retried_failed": retried_failed,
        "stale_reembedded": stale_reembedded,
        "failed": failed,
        "duration_s": duration_s,
    }
