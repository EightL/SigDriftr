"""Print a compact snapshot of local SigDriftr database state."""

from __future__ import annotations

import sqlite3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "sigdriftr.db"

TABLES = [
    "articles",
    "signals",
    "article_topics",
    "article_embeddings",
    "collection_runs",
    "collection_feed_stats",
    "cluster_runs",
    "clusters",
    "cluster_memberships",
    "cluster_signals",
    "cluster_drift_runs",
    "cluster_drift_observations",
    "baselines",
    "topics",
    "topic_aliases",
]


def count_rows(conn: sqlite3.Connection, table_name: str) -> int | None:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
    except sqlite3.Error:
        return None
    return int(row[0]) if row else 0


def print_table_counts(conn: sqlite3.Connection) -> None:
    print("--- Tables ---")
    for table_name in TABLES:
        count = count_rows(conn, table_name)
        if count is None:
            print(f"{table_name}: missing")
        else:
            print(f"{table_name}: {count}")


def print_topic_counts(conn: sqlite3.Connection) -> None:
    print("\n--- Articles by canonical_topic_id ---")
    rows = conn.execute(
        """
        SELECT COALESCE(canonical_topic_id, ''), COUNT(*)
        FROM articles
        GROUP BY canonical_topic_id
        ORDER BY COUNT(*) DESC
        """
    ).fetchall()
    if not rows:
        print("none")
        return
    for topic_id, count in rows:
        print(f"{topic_id or '(missing)'}: {count}")


def print_signal_coverage(conn: sqlite3.Connection) -> None:
    print("\n--- Signal extraction coverage ---")
    article_count = count_rows(conn, "articles") or 0
    signal_count = count_rows(conn, "signals") or 0
    coverage = 100 * signal_count / max(1, article_count)
    print(f"articles with signals: {signal_count}/{article_count} ({coverage:.1f}%)")


def print_recent_collection_runs(conn: sqlite3.Connection) -> None:
    print("\n--- Last 5 collection runs ---")
    rows = conn.execute(
        """
        SELECT started_at, topic, country, source, selected_feeds, inserted,
               accepted, duplicates, fetch_successful, fetch_failed,
               entries_seen, candidates, duration_s
        FROM collection_runs
        ORDER BY started_at DESC
        LIMIT 5
        """
    ).fetchall()
    if not rows:
        print("none")
        return
    for row in rows:
        (
            started_at,
            topic,
            country,
            source,
            selected_feeds,
            inserted,
            accepted,
            duplicates,
            fetch_successful,
            fetch_failed,
            entries_seen,
            candidates,
            duration_s,
        ) = row
        scope = "/".join(part for part in [topic, country, source] if part)
        fetch_total = fetch_successful + fetch_failed
        fetch_text = (
            f"{fetch_successful}/{fetch_total}"
            if fetch_total
            else "n/a"
        )
        print(
            f"{started_at}: {scope} selected={selected_feeds} "
            f"fetch={fetch_text} "
            f"entries={entries_seen} candidates={candidates} accepted={accepted} "
            f"inserted={inserted} dupes={duplicates} duration_s={duration_s:.2f}"
        )


def main() -> int:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return 1

    with sqlite3.connect(DB_PATH) as conn:
        print(f"DB: {DB_PATH}")
        print_table_counts(conn)
        print_topic_counts(conn)
        print_signal_coverage(conn)
        print_recent_collection_runs(conn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
