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
_CONN.commit()


def get_conn() -> sqlite3.Connection:
    return _CONN
