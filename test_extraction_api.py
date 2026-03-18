#!/usr/bin/env python3
import json
import tempfile
from unittest.mock import patch

import db.init
import pytest

pytest.importorskip("tenacity")


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


def insert_article(article_id: str, topic: str, outlet: str = "irozhlas") -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, ?, 'Title', 'Summary', ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, outlet, f"https://example.test/{article_id}", topic),
    )
    conn.commit()


def insert_signal(article_id: str) -> None:
    conn = db.init.get_conn()
    raw_json = {
        "concern_level": 0.7,
        "purchase_intent": 0.1,
        "avoidance_signals": 0.4,
        "dominant_frame": "fear",
        "domain": "civic",
        "irrelevant_fields": ["purchase_intent"],
        "seg_young_urban": 0.2,
        "seg_family": 0.3,
        "seg_senior": 0.4,
        "seg_b2b": 0.1,
    }
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-03-17T00:00:00+00:00')
        """,
        (
            article_id,
            raw_json["concern_level"],
            raw_json["purchase_intent"],
            raw_json["avoidance_signals"],
            raw_json["dominant_frame"],
            raw_json["seg_young_urban"],
            raw_json["seg_family"],
            raw_json["seg_senior"],
            raw_json["seg_b2b"],
            json.dumps(raw_json),
        ),
    )
    conn.commit()


def test_get_signals_does_not_trigger_extraction() -> None:
    pytest.importorskip("fastapi")
    from api.routes import signals as signals_route

    temp_dir = setup_temp_db()
    try:
        insert_article("article-1", "inflace")
        insert_signal("article-1")

        with patch("api.routes.signals.run_extraction") as mock_extract:
            rows = signals_route.get_signals("inflace")

        assert len(rows) == 1
        assert rows[0]["article_id"] == "article-1"
        assert rows[0]["domain"] == "civic"
        assert rows[0]["relevant_fields"] == [
            "concern_level",
            "avoidance_signals",
        ]
        mock_extract.assert_not_called()
    finally:
        cleanup_temp_db(temp_dir)


def test_extract_route_runs_extraction() -> None:
    pytest.importorskip("fastapi")
    from api.routes import signals as signals_route

    with patch("api.routes.signals.run_extraction", return_value=3) as mock_extract:
        result = signals_route.extract("inflace")

    assert result == {"processed": 3, "topic": "inflace"}
    mock_extract.assert_called_once_with("inflace")


def test_run_extraction_passes_topic_to_signal_extractor() -> None:
    from extraction import extractor

    temp_dir = setup_temp_db()
    try:
        insert_article("article-topic-pass", "politika")
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
                "domain": "civic",
                "irrelevant_fields": ["purchase_intent"],
            },
        ) as mock_extract:
            processed = extractor.run_extraction("politika")
    finally:
        cleanup_temp_db(temp_dir)

    assert processed == 1
    mock_extract.assert_called_once_with(
        "Title",
        "Summary",
        affinity_tag="mainstream",
        topic="politika",
    )


def test_run_extraction_uses_article_topic_when_processing_all_topics() -> None:
    from extraction import extractor

    temp_dir = setup_temp_db()
    try:
        insert_article("article-all-topics", "politika")
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
                "domain": "civic",
                "irrelevant_fields": ["purchase_intent"],
            },
        ) as mock_extract, patch(
            "extraction.extractor.extract_entities",
            return_value=[{"text": "Praha", "label": "GPE"}],
        ), patch("extraction.extractor.update_feed_reward") as mock_update:
            processed = extractor.run_extraction("")

        conn = db.init.get_conn()
        raw_json = json.loads(
            conn.execute(
                "SELECT raw_json FROM signals WHERE article_id = ?",
                ("article-all-topics",),
            ).fetchone()[0]
        )
    finally:
        cleanup_temp_db(temp_dir)

    assert processed == 1
    mock_extract.assert_called_once_with(
        "Title",
        "Summary",
        affinity_tag="mainstream",
        topic="politika",
    )
    assert raw_json["entities"] == [{"text": "Praha", "label": "GPE"}]
    mock_update.assert_called_once()


def test_ollama_request_parses_json_mode_response_directly() -> None:
    from extraction.llm_client import _ollama_request

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return (
                b'{"response":"{\\"concern_level\\":0.8,\\"purchase_intent\\":0.2,'
                b'\\"avoidance_signals\\":0.1,\\"dominant_frame\\":\\"fear\\",'
                b'\\"seg_young_urban\\":0.4,\\"seg_family\\":0.2,'
                b'\\"seg_senior\\":0.3,\\"seg_b2b\\":0.1}"}'
            )

    with patch("urllib.request.urlopen", return_value=FakeResponse()):
        result = _ollama_request(b"{}")

    assert result["concern_level"] == 0.8
    assert result["dominant_frame"] == "fear"


def test_prompt_template_requests_json_only() -> None:
    from extraction.llm_client import PROMPT_TEMPLATE

    assert "Step 1" not in PROMPT_TEMPLATE
    assert "Output ONLY this JSON object" in PROMPT_TEMPLATE


def test_relevance_threshold_is_not_too_low() -> None:
    from ingestion.crawler import SEMANTIC_THRESHOLD

    assert SEMANTIC_THRESHOLD >= 0.5


def test_feed_config_covers_family_and_young_urban_segments() -> None:
    from config.feeds import FEEDS
    from extraction.llm_client import AFFINITY_PRIORS

    affinity_tags = {feed["affinity_tag"] for feed in FEEDS}
    assert {"family", "young_urban"}.issubset(affinity_tags)
    assert {"family", "young_urban"}.issubset(AFFINITY_PRIORS)
