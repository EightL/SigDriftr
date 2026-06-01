from __future__ import annotations

import tempfile
from pathlib import Path

import db.init


ORIGINAL_DB_PATH = db.init.DB_PATH


def _close_thread_connection() -> None:
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")


def _clear_brief_cache_if_requested(enabled: bool) -> None:
    if not enabled:
        return
    from brief.generator import clear_brief_cache

    clear_brief_cache()


def setup_temp_db(*, clear_brief_cache: bool = False) -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = Path(temp_dir.name) / "sigdriftr.db"
    _close_thread_connection()
    db.init.get_conn()
    _clear_brief_cache_if_requested(clear_brief_cache)
    return temp_dir


def cleanup_temp_db(
    temp_dir: tempfile.TemporaryDirectory,
    *,
    clear_brief_cache: bool = False,
) -> None:
    _clear_brief_cache_if_requested(clear_brief_cache)
    _close_thread_connection()
    db.init.DB_PATH = ORIGINAL_DB_PATH
    temp_dir.cleanup()
