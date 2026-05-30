from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import db.init
from db.init import run_migrations
from db.topic_resolver import clear_topic_resolution_cache, resolve_topic


ORIGINAL_DB_PATH = db.init.DB_PATH


@contextmanager
def isolated_db():
    temp_dir = TemporaryDirectory()
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        del db.init._local.conn
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    clear_topic_resolution_cache()
    try:
        db.init.get_conn()
        yield db.init.DB_PATH
    finally:
        if hasattr(db.init._local, "conn"):
            db.init._local.conn.close()
            del db.init._local.conn
        db.init.DB_PATH = ORIGINAL_DB_PATH
        clear_topic_resolution_cache()
        temp_dir.cleanup()


def test_seeded_topic_alias_resolves_to_canonical_topic() -> None:
    with isolated_db():
        resolution = resolve_topic(" Energie ", language="cs")

        assert resolution.requested_topic == "Energie"
        assert resolution.normalized_topic == "energie"
        assert resolution.canonical_topic_id == "energy"
        assert resolution.display_name == "Energy"
        assert resolution.domain == "commerce"
        assert "energie" in resolution.aliases
        assert "energy" in resolution.aliases


def test_new_database_has_canonical_topic_scope_columns() -> None:
    with isolated_db():
        conn = db.init.get_conn()

        expected_columns = {
            "articles": "canonical_topic_id",
            "article_topics": "canonical_topic_id",
            "collection_runs": "canonical_topic_id",
            "segment_profiles": "canonical_topic_id",
            "baselines": "canonical_topic_id",
            "cluster_runs": "canonical_topic_id",
            "cluster_tracks": "canonical_topic_id",
            "cluster_drift_runs": "canonical_topic_id",
        }
        for table, column in expected_columns.items():
            columns = {
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            assert column in columns, table

        cluster_segment_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(cluster_segment_drifts)").fetchall()
        }
        assert "canonical_topic_id" not in cluster_segment_columns


def test_unknown_topic_is_created_as_slugged_canonical_topic() -> None:
    with isolated_db():
        resolution = resolve_topic("Battery Storage", language="en")

        assert resolution.canonical_topic_id == "battery-storage"
        assert resolution.display_name == "Battery Storage"
        assert resolution.was_created is True

        clear_topic_resolution_cache()
        second_resolution = resolve_topic("battery storage", language="en")
        assert second_resolution.canonical_topic_id == "battery-storage"
        assert second_resolution.was_created is False


def test_article_topic_trigger_resolves_accented_raw_aliases() -> None:
    with isolated_db():
        conn = db.init.get_conn()
        conn.execute(
            """
            INSERT INTO articles
            (id, outlet, title, summary, url, topic, country, language, published_at, fetched_at)
            VALUES ('health-1', 'irozhlas', 'Zdravi', '', 'https://example.test/health-1',
                    'zdraví', 'CZ', 'cs', '2026-05-01T00:00:00+00:00',
                    '2026-05-01T00:00:00+00:00')
            """
        )
        conn.commit()

        assert conn.execute(
            """
            SELECT raw_topic, canonical_topic_id
            FROM article_topics
            WHERE article_id = 'health-1'
            """
        ).fetchone() == ("zdraví", "healthcare")


def test_migration_backfills_canonical_topic_scope_columns() -> None:
    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "legacy.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE articles (
                id TEXT PRIMARY KEY,
                outlet TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                url TEXT UNIQUE NOT NULL,
                topic TEXT,
                published_at TEXT,
                fetched_at TEXT NOT NULL
            )
            """
        )
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
            CREATE TABLE segment_profiles (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                segment TEXT NOT NULL,
                window_start TEXT NOT NULL,
                window_days INTEGER NOT NULL,
                concern_level REAL,
                purchase_intent REAL,
                avoidance_signals REAL,
                dominant_frame TEXT,
                article_count INTEGER,
                computed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE collection_runs (
                run_id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                country TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                collection_mode TEXT NOT NULL,
                reward_mode TEXT NOT NULL,
                eligible_feeds INTEGER NOT NULL DEFAULT 0,
                selected_feeds INTEGER NOT NULL DEFAULT 0,
                inserted INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0,
                duplicates INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_s REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO articles
            (id, outlet, title, summary, url, topic, published_at, fetched_at)
            VALUES ('legacy-1', 'irozhlas', 'Energie zdrazuje', '', 'https://example.test/1',
                    'energie', '2026-05-01T00:00:00+00:00', '2026-05-01T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO baselines
            (id, topic, segment, concern_level, purchase_intent, avoidance_signals,
             dominant_frame, seeded, updated_at)
            VALUES ('baseline-1', 'energie', 'family', 0.5, 0.2, 0.3, 'fear', 1,
                    '2026-05-01T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO segment_profiles
            (id, topic, segment, window_start, window_days, concern_level,
             purchase_intent, avoidance_signals, dominant_frame, article_count, computed_at)
            VALUES ('profile-1', 'energie', 'family', '2026-05-01', 7, 0.6, 0.2,
                    0.4, 'fear', 3, '2026-05-01T00:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO collection_runs
            (run_id, topic, collection_mode, reward_mode, started_at, completed_at)
            VALUES ('collect-1', 'energie', 'bandit', 'yield',
                    '2026-05-01T00:00:00+00:00', '2026-05-01T00:01:00+00:00')
            """
        )
        conn.commit()

        run_migrations(conn)
        run_migrations(conn)

        article_topic_row = conn.execute(
            """
            SELECT article_id, topic, raw_topic, canonical_topic_id
            FROM article_topics
            WHERE article_id = 'legacy-1'
            """
        ).fetchone()
        assert article_topic_row == ("legacy-1", "energie", "energie", "energy")

        assert conn.execute(
            "SELECT canonical_topic_id FROM articles WHERE id = 'legacy-1'"
        ).fetchone() == ("energy",)
        assert conn.execute(
            "SELECT canonical_topic_id FROM baselines WHERE id = 'baseline-1'"
        ).fetchone() == ("energy",)
        assert conn.execute(
            "SELECT canonical_topic_id FROM segment_profiles WHERE id = 'profile-1'"
        ).fetchone() == ("energy",)
        assert conn.execute(
            "SELECT canonical_topic_id FROM collection_runs WHERE run_id = 'collect-1'"
        ).fetchone() == ("energy",)
        assert conn.execute(
            "SELECT display_name, domain FROM topics WHERE canonical_topic_id = 'energy'"
        ).fetchone() == ("Energy", "commerce")
