import sqlite3
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sigdriftr.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
_CONN.execute(
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
_CONN.execute(
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
_CONN.commit()


def get_conn() -> sqlite3.Connection:
    return _CONN
