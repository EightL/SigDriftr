#!/usr/bin/env python3
import hashlib
import sqlite3
import tempfile
from pathlib import Path
from datetime import datetime, timezone

import db.init
from db_helpers import cleanup_temp_db, setup_temp_db
from db.init import run_migrations
from delta.engine import _compute_confidence, compute_drift, update_baseline_from_profile
from delta.mapper import compute_segment_profiles
from delta.seeder import seed_baselines


RECENT_TS = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def insert_article_with_signal(
    article_id: str,
    topic: str,
    concern: float,
    purchase: float,
    avoidance: float,
    frame: str,
    seg_young_urban: float,
    seg_family: float,
    seg_senior: float,
    seg_b2b: float,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, 'unit-test', ?, '', ?, ?, ?, ?)
        """,
        (
            article_id,
            article_id,
            f"https://example.test/{article_id}",
            topic,
            RECENT_TS,
            RECENT_TS,
        ),
    )
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)
        """,
        (
            article_id,
            concern,
            purchase,
            avoidance,
            frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            RECENT_TS,
        ),
    )
    conn.commit()


def test_run_migrations_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "migration.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                CREATE TABLE baselines (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    segment TEXT NOT NULL,
                    concern_level REAL,
                    purchase_intent REAL,
                    avoidance_signals REAL,
                    dominant_frame TEXT,
                    seeded INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                INSERT INTO baselines
                (id, topic, segment, concern_level, purchase_intent,
                 avoidance_signals, dominant_frame, seeded, updated_at)
                VALUES ('row-1', 'inflace', 'young_urban', 0.0, 0.0, 0.0, 'neutral', 1, '2026-03-17T00:00:00+00:00')
                """
            )

            run_migrations(conn)
            run_migrations(conn)

            columns = {
                row[1]: row[4]
                for row in conn.execute("PRAGMA table_info(baselines)").fetchall()
            }
            migrated_row = conn.execute(
                "SELECT sample_count, is_learned FROM baselines WHERE id = 'row-1'"
            ).fetchone()
        finally:
            conn.close()

    assert columns["sample_count"] == "0"
    assert columns["is_learned"] == "0"
    assert migrated_row == (0, 0)


def test_update_baseline_from_profile_blending() -> None:
    temp_dir = setup_temp_db()
    try:
        conn = db.init.get_conn()
        row_id = hashlib.sha256("inflace:young_urban".encode()).hexdigest()
        conn.execute(
            """
            INSERT INTO baselines
            (id, topic, segment, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seeded, sample_count, is_learned, updated_at)
            VALUES (?, 'inflace', 'young_urban', 0.0, 0.0, 0.0, 'neutral', 1, 0, 0, '2026-03-17T00:00:00+00:00')
            """,
            (row_id,),
        )
        conn.commit()

        updated = update_baseline_from_profile(
            "inflace",
            "young_urban",
            {
                "concern_level": 0.8,
                "purchase_intent": 0.0,
                "avoidance_signals": 0.0,
                "dominant_frame": "fear",
            },
            article_count=30,
            conn=conn,
        )

        row = conn.execute(
            """
            SELECT concern_level, sample_count, is_learned, dominant_frame
            FROM baselines
            WHERE id = ?
            """,
            (row_id,),
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert updated is True
    assert row == (0.16, 1, 1, "fear")


def test_compute_confidence_edge_cases() -> None:
    assert _compute_confidence(article_count=0, sample_count=0, is_learned=False) == 0.15
    assert _compute_confidence(article_count=50, sample_count=50, is_learned=True) == 1.0
    assert _compute_confidence(article_count=25, sample_count=0, is_learned=False) == 0.4


def test_article_count_below_threshold_skips_baseline_update() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        conn = db.init.get_conn()
        row_id = hashlib.sha256("inflace:young_urban".encode()).hexdigest()
        before = conn.execute(
            """
            SELECT concern_level, sample_count, is_learned
            FROM baselines
            WHERE id = ?
            """,
            (row_id,),
        ).fetchone()

        updated = update_baseline_from_profile(
            "inflace",
            "young_urban",
            {
                "concern_level": 0.9,
                "purchase_intent": 0.1,
                "avoidance_signals": 0.2,
                "dominant_frame": "fear",
            },
            article_count=10,
            conn=conn,
        )

        after = conn.execute(
            """
            SELECT concern_level, sample_count, is_learned
            FROM baselines
            WHERE id = ?
            """,
            (row_id,),
        ).fetchone()
    finally:
        cleanup_temp_db(temp_dir)

    assert updated is False
    assert after == before


def test_confidence_increases_over_time() -> None:
    temp_dir = setup_temp_db()
    try:
        topic = "inflace"
        cycle_counts = (10, 30, 60)
        confidences = []
        learned_flags = []
        sample_counts = []

        current_total = 0
        for batch_index, batch_size in enumerate(cycle_counts, start=1):
            for article_index in range(current_total, batch_size):
                insert_article_with_signal(
                    article_id=f"cycle-{batch_index}-{article_index}",
                    topic=topic,
                    concern=0.8,
                    purchase=0.2,
                    avoidance=0.3,
                    frame="fear",
                    seg_young_urban=1.0,
                    seg_family=0.0,
                    seg_senior=0.0,
                    seg_b2b=0.0,
                )
            current_total = batch_size

            compute_segment_profiles(topic, learn_baseline=True)
            drift = compute_drift(topic)
            young_urban = next(
                entry for entry in drift if entry["segment"] == "young_urban"
            )
            confidences.append(young_urban["confidence"])
            learned_flags.append(young_urban["baseline_is_learned"])
            sample_counts.append(young_urban["baseline_sample_count"])
    finally:
        cleanup_temp_db(temp_dir)

    assert confidences[0] < confidences[1] < confidences[2]
    assert learned_flags == [False, True, True]
    assert sample_counts == [0, 1, 2]
