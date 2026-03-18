#!/usr/bin/env python3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import db.init
import pytest

pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from db.queries import get_profile_history
from delta.mapper import SEGMENTS
from delta.seeder import seed_baselines
from main import app


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


def insert_segment_profile(
    topic: str,
    segment: str,
    computed_at: str,
    concern: float,
    purchase: float,
    avoidance: float,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO segment_profiles
        (id, topic, segment, window_start, window_days, concern_level, purchase_intent,
         avoidance_signals, dominant_frame, article_count, computed_at)
        VALUES (?, ?, ?, ?, 7, ?, ?, ?, 'neutral', 4, ?)
        """,
        (
            f"{segment}:{computed_at}",
            topic,
            segment,
            computed_at[:10],
            concern,
            purchase,
            avoidance,
            computed_at,
        ),
    )
    conn.commit()


def seed_history_rows(topic: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    day_one = (now - timedelta(days=2)).isoformat()
    day_two = (now - timedelta(days=1)).isoformat()

    values = {
        "young_urban": [(0.50, 0.40, 0.30), (0.46, 0.36, 0.25)],
        "family": [(0.58, 0.30, 0.35), (0.55, 0.29, 0.31)],
        "senior": [(0.67, 0.18, 0.44), (0.64, 0.16, 0.42)],
        "b2b": [(0.49, 0.55, 0.24), (0.46, 0.53, 0.22)],
    }

    for segment, (first_day, second_day) in values.items():
        insert_segment_profile(topic, segment, day_one, *first_day)
        insert_segment_profile(topic, segment, day_two, *second_day)

    return day_one[:10], day_two[:10]


def test_get_profile_history_returns_segment_timelines() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["energie"])
        first_day, second_day = seed_history_rows("energie")

        history = get_profile_history("energie", days_back=30)
    finally:
        cleanup_temp_db(temp_dir)

    assert [entry["segment"] for entry in history] == SEGMENTS
    young_urban = next(entry for entry in history if entry["segment"] == "young_urban")
    assert young_urban["dates"] == [first_day, second_day]
    assert young_urban["concern"] == [0.5, 0.46]
    assert young_urban["purchase_intent"] == [0.4, 0.36]
    assert young_urban["avoidance"] == [0.3, 0.25]
    assert young_urban["drift_magnitude"] == [0.108, 0.066]


def test_history_route_returns_four_segment_objects() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["energie"])
        first_day, second_day = seed_history_rows("energie")

        with TestClient(app) as client:
            response = client.get("/history/energie?days_back=30")
    finally:
        cleanup_temp_db(temp_dir)

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 4
    assert [entry["segment"] for entry in payload] == SEGMENTS

    for entry in payload:
        assert entry["dates"] == [first_day, second_day]
        assert len(entry["drift_magnitude"]) == 2

