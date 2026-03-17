import sqlite3
import threading
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sigdriftr.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def get_conn() -> sqlite3.Connection:
    """Return a per-thread SQLite connection, creating it on first use."""
    if not hasattr(_local, "conn"):
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
                updated_at        TEXT NOT NULL
            );
            """
        )
        conn.commit()
        _local.conn = conn

    return _local.conn
