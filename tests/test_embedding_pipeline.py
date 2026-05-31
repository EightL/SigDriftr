#!/usr/bin/env python3
import tempfile
from pathlib import Path
from unittest.mock import patch

import db.init

from extraction.embedder import get_expected_dim, get_model_name
from extraction.embedding_service import embed_pending_articles
from extraction.embedding_text import build_embedding_text


ORIGINAL_DB_PATH = db.init.DB_PATH


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()


def insert_article(
    article_id: str,
    *,
    title: str = "Inflation pressure rises again",
    summary: str = "A useful summary sentence that is long enough. Another useful sentence follows with context.",
    body: str = "",
    topic: str = "",
    country: str = "CZ",
    language: str = "cs",
    outlet: str = "irozhlas",
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, body, url, canonical_url, topic, country, language, published_at, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-03-18T10:00:00+00:00', '2026-03-18T10:05:00+00:00')
        """,
        (
            article_id,
            outlet,
            title,
            summary,
            body,
            f"https://example.test/{article_id}",
            f"https://example.test/{article_id}",
            topic,
            country,
            language,
        ),
    )
    conn.commit()


def fake_embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    dim = get_expected_dim()
    return [[float(index + 1)] * dim for index, _ in enumerate(texts)]


def test_embed_pending_articles_is_idempotent_for_current_rows() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article("article-1", topic="inflace")
        with patch("extraction.embedding_service.embed_texts", side_effect=fake_embed_texts):
            first = embed_pending_articles(limit=10)

        conn = db.init.get_conn()
        row_count = conn.execute("SELECT COUNT(*) FROM article_embeddings").fetchone()[0]

        with patch("extraction.embedding_service.embed_texts") as mock_embed:
            second = embed_pending_articles(limit=10)
    finally:
        cleanup_temp_db(temp_dir)

    assert first["embedded"] == 1
    assert first["already_current"] == 0
    assert first["failed"] == 0
    assert row_count == 1
    assert second["embedded"] == 0
    assert second["already_current"] == 1
    mock_embed.assert_not_called()


def test_embed_pending_articles_retries_same_hash_failed_rows() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article("article-2", topic="inflace")
        embedding_text = build_embedding_text(
            title="Inflation pressure rises again",
            body="",
            summary="A useful summary sentence that is long enough. Another useful sentence follows with context.",
        )
        conn = db.init.get_conn()
        conn.execute(
            """
            INSERT INTO article_embeddings
            (article_id, model_name, model_version, embedding_dim, embedding_vector,
             embedding_text, embedding_text_hash, language, status, error_message,
             embedded_at, created_at, updated_at)
            VALUES (?, ?, 'seed', ?, '[]', ?, ?, 'cs', 'failed', 'seed failure', NULL,
                    '2026-03-18T10:10:00+00:00', '2026-03-18T10:10:00+00:00')
            """,
            (
                "article-2",
                get_model_name(),
                get_expected_dim(),
                embedding_text,
                __import__("hashlib").sha256(embedding_text.encode("utf-8")).hexdigest(),
            ),
        )
        conn.commit()

        with patch("extraction.embedding_service.embed_texts", side_effect=fake_embed_texts):
            result = embed_pending_articles(limit=10)

        row = conn.execute(
            """
            SELECT status, error_message, embedding_vector
            FROM article_embeddings
            WHERE article_id = ? AND model_name = ?
            """,
            ("article-2", get_model_name()),
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert result["retried_failed"] == 1
    assert result["embedded"] == 1
    assert row[0] == "complete"
    assert row[1] is None
    assert row[2] != "[]"


def test_embed_pending_articles_inserts_new_row_when_text_hash_changes() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article(
            "article-3",
            topic="inflace",
            body="Original long body sentence for the first embedding. Another original sentence adds enough context.",
        )
        with patch("extraction.embedding_service.embed_texts", side_effect=fake_embed_texts):
            first = embed_pending_articles(limit=10)

        conn = db.init.get_conn()
        conn.execute(
            """
            UPDATE articles
            SET body = ?
            WHERE id = ?
            """,
            (
                "Updated body sentence with materially different text content. Another updated sentence keeps it long enough.",
                "article-3",
            ),
        )
        conn.commit()

        with patch("extraction.embedding_service.embed_texts", side_effect=fake_embed_texts):
            second = embed_pending_articles(limit=10)

        rows = conn.execute(
            """
            SELECT embedding_text_hash, status
            FROM article_embeddings
            WHERE article_id = ?
            ORDER BY id ASC
            """,
            ("article-3",),
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert first["embedded"] == 1
    assert second["stale_reembedded"] == 1
    assert second["embedded"] == 1
    assert len(rows) == 2
    assert rows[0][0] != rows[1][0]
    assert rows[0][1] == "complete"
    assert rows[1][1] == "complete"


def test_embed_pending_articles_filters_by_article_topics() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article("article-4", topic="")
        insert_article("article-5", topic="")
        conn = db.init.get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO article_topics(article_id, topic, relevance_score, matched_at)
            VALUES ('article-4', 'inflace', 0.9, '2026-03-18T10:05:00+00:00')
            """
        )
        conn.commit()

        with patch("extraction.embedding_service.embed_texts", side_effect=fake_embed_texts):
            result = embed_pending_articles(limit=10, topic="inflace")

        article_ids = conn.execute(
            """
            SELECT article_id
            FROM article_embeddings
            ORDER BY article_id ASC
            """
        ).fetchall()
    finally:
        cleanup_temp_db(temp_dir)

    assert result["selected"] == 1
    assert result["embedded"] == 1
    assert article_ids == [("article-4",)]
