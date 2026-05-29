from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import db.init


RECENT_TS = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = db.init.Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def cleanup_temp_db(temp_dir: tempfile.TemporaryDirectory) -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    temp_dir.cleanup()


def insert_article(article_id: str, topic: str, body: str = "") -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, body, url, topic, published_at, fetched_at)
        VALUES (?, 'irozhlas', 'Title', 'Summary', ?, ?, ?, ?, ?)
        """,
        (
            article_id,
            body,
            f"https://example.test/{article_id}",
            topic,
            RECENT_TS,
            RECENT_TS,
        ),
    )
    conn.commit()


def test_run_extraction_stores_segment_relevance_topic_relevance_and_body() -> None:
    from extraction import extractor

    temp_dir = setup_temp_db()
    try:
        insert_article(
            "article-body-pass",
            "inflace",
            body="Full extracted article text with detail.",
        )
        with patch(
            "extraction.extractor.extract_signals",
            return_value={
                "concern_level": 0.6,
                "purchase_intent": 0.0,
                "avoidance_signals": 0.2,
                "dominant_frame": "fear",
                "seg_young_urban": 0.25,
                "seg_family": 0.25,
                "seg_senior": 0.25,
                "seg_b2b": 0.25,
                "seg_young_urban_relevance": 0.8,
                "seg_family_relevance": 0.3,
                "seg_senior_relevance": 0.1,
                "seg_b2b_relevance": 0.2,
                "domain": "commerce",
                "irrelevant_fields": [],
            },
        ) as mock_extract, patch(
            "extraction.extractor.extract_entities",
            return_value=[],
        ):
            processed = extractor.run_extraction("inflace")

        conn = db.init.get_conn()
        row = conn.execute(
            """
            SELECT seg_young_urban_relevance, seg_family_relevance,
                   seg_senior_relevance, seg_b2b_relevance, raw_json
            FROM signals
            WHERE article_id = ?
            """,
            ("article-body-pass",),
        ).fetchone()
        raw_json = json.loads(row[4])
    finally:
        cleanup_temp_db(temp_dir)

    assert processed == 1
    mock_extract.assert_called_once_with(
        "Title",
        "Summary",
        affinity_tag="mainstream",
        topic="inflace",
        body="Full extracted article text with detail.",
    )
    assert row[:4] == (0.8, 0.3, 0.1, 0.2)
    assert raw_json["topic_relevance_score"] == 1.0
    assert raw_json["topic_relevance"] == "2"


def test_extract_signals_keeps_relevance_separate_from_share() -> None:
    from extraction.llm_client import extract_signals

    with (
        patch("extraction.llm_client.LLM_PROVIDER", "google"),
        patch("extraction.llm_client.GOOGLE_GEMMA_MODEL", "gemma-4-31b-it"),
        patch(
            "extraction.llm_client._try_google_gemma",
            return_value={
                "concern_level": 0.8,
                "purchase_intent": 0.2,
                "avoidance_signals": 0.1,
                "dominant_frame": "fear",
                "seg_young_urban_relevance": 0.9,
                "seg_family_relevance": 0.9,
                "seg_senior_relevance": 0.1,
                "seg_b2b_relevance": 0.1,
            },
        ),
        patch("extraction.llm_client._try_ollama"),
    ):
        result = extract_signals(
            "Title",
            "Summary",
            topic="inflace",
            body="Body text",
        )

    assert result["seg_young_urban_relevance"] == 0.9
    assert result["seg_family_relevance"] == 0.9
    assert result["seg_young_urban_share"] == result["seg_young_urban"]
    assert round(
        sum(
            result[key]
            for key in ["seg_young_urban", "seg_family", "seg_senior", "seg_b2b"]
        ),
        4,
    ) == 1.0
    assert result["input_text"]["body_chars"] == len("Body text")
