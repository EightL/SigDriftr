import sqlite3
import threading
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sigdriftr.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations for existing databases."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(baselines)").fetchall()
    }

    if "sample_count" not in columns:
        conn.execute(
            """
            ALTER TABLE baselines
            ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0
            """
        )
    if "is_learned" not in columns:
        conn.execute(
            """
            ALTER TABLE baselines
            ADD COLUMN is_learned INTEGER NOT NULL DEFAULT 0
            """
        )


def get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it on first use."""
    if not hasattr(_local, "conn"):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(DB_PATH),
            check_same_thread=False,
            timeout=30.0,
            isolation_level=None,
        )
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id           TEXT PRIMARY KEY,
                outlet       TEXT NOT NULL,
                title        TEXT NOT NULL,
                summary      TEXT,
                url          TEXT UNIQUE NOT NULL,
                topic        TEXT,
                published_at TEXT,
                fetched_at   TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                article_id         TEXT PRIMARY KEY,
                concern_level      REAL,
                purchase_intent    REAL,
                avoidance_signals  REAL,
                dominant_frame     TEXT,
                seg_young_urban    REAL,
                seg_family         REAL,
                seg_senior         REAL,
                seg_b2b            REAL,
                raw_json           TEXT,
                extracted_at       TEXT NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_entities (
                article_id   TEXT NOT NULL,
                entity_text  TEXT NOT NULL,
                entity_norm  TEXT NOT NULL,
                entity_label TEXT NOT NULL,
                PRIMARY KEY (article_id, entity_norm, entity_label),
                FOREIGN KEY (article_id) REFERENCES articles(id)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_article_entities_norm
            ON article_entities(entity_norm, entity_label)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_article_entities_article
            ON article_entities(article_id)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS segment_profiles (
                id                TEXT PRIMARY KEY,
                topic             TEXT NOT NULL,
                segment           TEXT NOT NULL,
                window_start      TEXT NOT NULL,
                window_days       INTEGER NOT NULL,
                concern_level     REAL,
                purchase_intent   REAL,
                avoidance_signals REAL,
                dominant_frame    TEXT,
                article_count     INTEGER,
                computed_at       TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                id                TEXT PRIMARY KEY,
                topic             TEXT NOT NULL,
                segment           TEXT NOT NULL,
                concern_level     REAL,
                purchase_intent   REAL,
                avoidance_signals REAL,
                dominant_frame    TEXT,
                seeded            INTEGER DEFAULT 0,
                sample_count      INTEGER NOT NULL DEFAULT 0,
                is_learned        INTEGER NOT NULL DEFAULT 0,
                updated_at        TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bandit_state (
                outlet       TEXT PRIMARY KEY,
                context_dim  INTEGER NOT NULL,
                a_json       TEXT NOT NULL,
                b_json       TEXT NOT NULL,
                pulls        INTEGER NOT NULL DEFAULT 0,
                total_reward REAL NOT NULL DEFAULT 0.0,
                updated_at   TEXT NOT NULL
            );
            """
        )
        run_migrations(conn)
        conn.commit()
        _local.conn = conn

    return _local.conn
