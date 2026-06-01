import sqlite3
import threading
from pathlib import Path

from config.topics import (
    TOPIC_SEEDS,
    display_name_for_topic,
    domain_for_topic,
    normalize_topic,
    seed_for_alias,
    slugify_topic,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sigdriftr.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        is not None
    )


def _add_column_if_missing(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    if not _table_exists(conn, table_name):
        return
    if column_name not in _table_columns(conn, table_name):
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")


def _execute_if_table_exists(
    conn: sqlite3.Connection,
    table_name: str,
    sql: str,
) -> None:
    if _table_exists(conn, table_name):
        conn.execute(sql)


def _ensure_topic_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS topics (
            canonical_topic_id TEXT PRIMARY KEY,
            display_name       TEXT NOT NULL,
            domain             TEXT NOT NULL DEFAULT 'generic',
            status             TEXT NOT NULL DEFAULT 'active',
            merged_into        TEXT,
            created_at         TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS topic_aliases (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_topic_id  TEXT NOT NULL REFERENCES topics(canonical_topic_id),
            raw_topic           TEXT NOT NULL,
            normalized_topic    TEXT NOT NULL,
            language            TEXT,
            source              TEXT NOT NULL DEFAULT 'curated_seed',
            created_at          TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute("DROP INDEX IF EXISTS idx_topic_aliases_normalized_language")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_topic_aliases_normalized_language
        ON topic_aliases(normalized_topic, IFNULL(language, ''))
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_topic_aliases_raw_language_unique
        ON topic_aliases(raw_topic, IFNULL(language, ''))
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_topic_aliases_canonical_topic
        ON topic_aliases(canonical_topic_id)
        """
    )


def _insert_topic_alias(
    conn: sqlite3.Connection,
    canonical_topic_id: str,
    raw_topic: str,
    *,
    language: str | None,
    source: str,
) -> None:
    normalized = normalize_topic(raw_topic)
    if not canonical_topic_id or not normalized:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO topic_aliases
        (canonical_topic_id, raw_topic, normalized_topic, language, source)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            canonical_topic_id,
            raw_topic.strip().lower(),
            normalized,
            (language or "").strip().lower() or None,
            source,
        ),
    )


def _seed_topic_catalog(conn: sqlite3.Connection) -> None:
    for seed in TOPIC_SEEDS:
        conn.execute(
            """
            INSERT INTO topics
            (canonical_topic_id, display_name, domain, status, updated_at)
            VALUES (?, ?, ?, 'active', datetime('now'))
            ON CONFLICT(canonical_topic_id) DO UPDATE SET
                display_name = excluded.display_name,
                domain = excluded.domain,
                status = CASE
                    WHEN topics.status = 'merged' THEN topics.status
                    ELSE excluded.status
                END,
                updated_at = excluded.updated_at
            """,
            (seed.canonical_topic_id, seed.display_name, seed.domain),
        )
        _insert_topic_alias(
            conn,
            seed.canonical_topic_id,
            seed.canonical_topic_id,
            language=None,
            source="curated_seed",
        )
        for alias, language in seed.aliases:
            _insert_topic_alias(
                conn,
                seed.canonical_topic_id,
                alias,
                language=language,
                source="curated_seed",
            )


def _ensure_topic_for_raw(
    conn: sqlite3.Connection,
    raw_topic: str | None,
    *,
    language: str | None = None,
    source: str = "migration",
) -> str:
    raw = (raw_topic or "").strip()
    if not raw:
        return ""

    normalized = normalize_topic(raw)
    if not normalized:
        return ""

    row = conn.execute(
        """
        SELECT canonical_topic_id
        FROM topic_aliases
        WHERE normalized_topic = ?
          AND (? = '' OR language = ? OR language IS NULL)
        ORDER BY
          CASE
              WHEN language = ? THEN 0
              WHEN language IS NULL THEN 1
              ELSE 2
          END,
          id ASC
        LIMIT 1
        """,
        (
            normalized,
            (language or "").strip().lower(),
            (language or "").strip().lower(),
            (language or "").strip().lower(),
        ),
    ).fetchone()
    if row is not None:
        return str(row[0])

    seed = seed_for_alias(raw)
    if seed is not None:
        _seed_topic_catalog(conn)
        return seed.canonical_topic_id

    canonical_topic_id = slugify_topic(raw)
    conn.execute(
        """
        INSERT OR IGNORE INTO topics
        (canonical_topic_id, display_name, domain, status, updated_at)
        VALUES (?, ?, ?, 'active', datetime('now'))
        """,
        (
            canonical_topic_id,
            display_name_for_topic(raw),
            domain_for_topic(raw),
        ),
    )
    _insert_topic_alias(
        conn,
        canonical_topic_id,
        raw,
        language=language,
        source=source,
    )
    return canonical_topic_id


def _backfill_canonical_topic_column(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    topic_column: str = "topic",
    language_column: str | None = None,
) -> None:
    if not _table_exists(conn, table_name):
        return
    columns = _table_columns(conn, table_name)
    if topic_column not in columns or "canonical_topic_id" not in columns:
        return

    select_columns = ["rowid", topic_column]
    if language_column and language_column in columns:
        select_columns.append(language_column)
    else:
        language_column = None

    rows = conn.execute(
        f"""
        SELECT {", ".join(select_columns)}
        FROM {table_name}
        WHERE ({topic_column} IS NOT NULL AND TRIM({topic_column}) != '')
          AND (canonical_topic_id IS NULL OR TRIM(canonical_topic_id) = '')
        """
    ).fetchall()
    for row in rows:
        rowid = row[0]
        raw_topic = row[1]
        language = row[2] if language_column else None
        canonical_topic_id = _ensure_topic_for_raw(
            conn,
            raw_topic,
            language=language,
            source=f"{table_name}_backfill",
        )
        conn.execute(
            f"UPDATE {table_name} SET canonical_topic_id = ? WHERE rowid = ?",
            (canonical_topic_id, rowid),
        )


def _backfill_article_topic_links(conn: sqlite3.Connection) -> None:
    if not _table_exists(conn, "article_topics"):
        return
    columns = _table_columns(conn, "article_topics")
    if "raw_topic" not in columns or "canonical_topic_id" not in columns:
        return

    rows = conn.execute(
        """
        SELECT at.article_id, at.topic, at.raw_topic, at.canonical_topic_id, a.language
        FROM article_topics at
        LEFT JOIN articles a ON a.id = at.article_id
        WHERE at.topic IS NOT NULL
          AND TRIM(at.topic) != ''
          AND (
              at.raw_topic IS NULL
              OR TRIM(at.raw_topic) = ''
              OR at.canonical_topic_id IS NULL
              OR TRIM(at.canonical_topic_id) = ''
          )
        """
    ).fetchall()
    for article_id, topic, raw_topic, canonical_topic_id, language in rows:
        resolved_raw_topic = (raw_topic or topic or "").strip()
        resolved_canonical = (canonical_topic_id or "").strip()
        if not resolved_canonical:
            resolved_canonical = _ensure_topic_for_raw(
                conn,
                resolved_raw_topic,
                language=language,
                source="article_topics_backfill",
            )
        conn.execute(
            """
            UPDATE article_topics
            SET raw_topic = ?,
                canonical_topic_id = ?
            WHERE article_id = ? AND topic = ?
            """,
            (
                resolved_raw_topic,
                resolved_canonical,
                article_id,
                topic,
            ),
        )


def _ensure_collection_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_runs (
            run_id           TEXT PRIMARY KEY,
            topic            TEXT NOT NULL,
            canonical_topic_id TEXT,
            country          TEXT NOT NULL DEFAULT '',
            source           TEXT NOT NULL DEFAULT '',
            collection_mode  TEXT NOT NULL,
            reward_mode      TEXT NOT NULL,
            eligible_feeds   INTEGER NOT NULL DEFAULT 0,
            selected_feeds   INTEGER NOT NULL DEFAULT 0,
            inserted         INTEGER NOT NULL DEFAULT 0,
            accepted         INTEGER NOT NULL DEFAULT 0,
            duplicates       INTEGER NOT NULL DEFAULT 0,
            started_at       TEXT NOT NULL,
            completed_at     TEXT NOT NULL,
            duration_s       REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_collection_runs_scope_completed
        ON collection_runs(topic, country, source, completed_at DESC)
        """
    )
    if "canonical_topic_id" in _table_columns(conn, "collection_runs"):
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_collection_runs_canonical_scope_completed
            ON collection_runs(canonical_topic_id, country, source, completed_at DESC)
            """
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS collection_feed_stats (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT NOT NULL REFERENCES collection_runs(run_id),
            outlet              TEXT NOT NULL,
            country             TEXT NOT NULL DEFAULT '',
            language            TEXT,
            selected            INTEGER NOT NULL DEFAULT 1,
            fetch_success       INTEGER NOT NULL DEFAULT 0,
            entries_seen        INTEGER NOT NULL DEFAULT 0,
            candidates          INTEGER NOT NULL DEFAULT 0,
            accepted            INTEGER NOT NULL DEFAULT 0,
            inserted            INTEGER NOT NULL DEFAULT 0,
            duplicates          INTEGER NOT NULL DEFAULT 0,
            avg_relevance_score REAL NOT NULL DEFAULT 0.0,
            reward              REAL NOT NULL DEFAULT 0.0,
            error_message       TEXT,
            created_at          TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_collection_feed_stats_run
        ON collection_feed_stats(run_id, outlet)
        """
    )


def _ensure_cluster_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_runs (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                    TEXT    NOT NULL UNIQUE,
            topic                     TEXT    NOT NULL,
            canonical_topic_id        TEXT,
            country                   TEXT    NOT NULL DEFAULT '',
            source                    TEXT    NOT NULL DEFAULT '',
            language                  TEXT,
            window_start              TEXT    NOT NULL,
            window_end                TEXT    NOT NULL,
            status                    TEXT    NOT NULL,
            n_articles                INTEGER NOT NULL,
            n_clusters                INTEGER NOT NULL,
            n_noise                   INTEGER NOT NULL,
            umap_n_components         INTEGER NOT NULL DEFAULT 10,
            umap_n_neighbors          INTEGER NOT NULL DEFAULT 15,
            hdbscan_min_cluster_size  INTEGER NOT NULL DEFAULT 3,
            hdbscan_min_samples       INTEGER NOT NULL DEFAULT 2,
            model_name                TEXT    NOT NULL,
            model_version             TEXT,
            created_at                TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_runs_scope_created
        ON cluster_runs(topic, country, source, language, created_at DESC)
        """
    )
    if "canonical_topic_id" in _table_columns(conn, "cluster_runs"):
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cluster_runs_canonical_scope_created
            ON cluster_runs(canonical_topic_id, country, source, language, created_at DESC)
            """
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clusters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT    NOT NULL REFERENCES cluster_runs(run_id),
            cluster_label   INTEGER NOT NULL,
            size            INTEGER NOT NULL,
            centroid_vector TEXT    NOT NULL,
            centroid_dim    INTEGER NOT NULL DEFAULT 384,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_clusters_run_id
        ON clusters(run_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clusters_run_label
        ON clusters(run_id, cluster_label)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_memberships (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id              TEXT    NOT NULL REFERENCES cluster_runs(run_id),
            cluster_id          INTEGER REFERENCES clusters(id),
            article_id          TEXT    NOT NULL REFERENCES articles(id),
            embedding_id        INTEGER NOT NULL REFERENCES article_embeddings(id),
            membership_strength REAL,
            is_noise            INTEGER NOT NULL DEFAULT 0,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cm_run_id
        ON cluster_memberships(run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cm_article
        ON cluster_memberships(article_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cm_cluster
        ON cluster_memberships(cluster_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cm_run_article_unique
        ON cluster_memberships(run_id, article_id)
        """
    )


def _ensure_cluster_signal_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_signals (
            cluster_id              INTEGER PRIMARY KEY REFERENCES clusters(id),
            run_id                  TEXT    NOT NULL REFERENCES cluster_runs(run_id),
            topic_label             TEXT    NOT NULL,
            concern_level           REAL    NOT NULL,
            purchase_intent         REAL    NOT NULL,
            avoidance_signals       REAL    NOT NULL,
            sentiment               REAL    NOT NULL,
            dominant_frame          TEXT    NOT NULL,
            frame_detail            TEXT    NOT NULL,
            seg_young_urban         REAL    NOT NULL,
            seg_family              REAL    NOT NULL,
            seg_senior              REAL    NOT NULL,
            seg_b2b                 REAL    NOT NULL,
            evidence_json           TEXT    NOT NULL DEFAULT '[]',
            raw_json                TEXT    NOT NULL DEFAULT '{}',
            member_count            INTEGER NOT NULL,
            membership_fingerprint  TEXT    NOT NULL,
            exemplar_article_ids    TEXT    NOT NULL,
            coherence_score         REAL    NOT NULL DEFAULT 0.0,
            extractor_provider      TEXT    NOT NULL,
            extractor_model         TEXT    NOT NULL,
            schema_version          TEXT    NOT NULL DEFAULT 'v1',
            extracted_at            TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_signals_run_id
        ON cluster_signals(run_id)
        """
    )


def _ensure_cluster_drift_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_tracks (
            track_id                   TEXT PRIMARY KEY,
            topic                      TEXT    NOT NULL,
            canonical_topic_id         TEXT,
            country                    TEXT    NOT NULL DEFAULT '',
            source                     TEXT    NOT NULL DEFAULT '',
            language                   TEXT,
            status                     TEXT    NOT NULL DEFAULT 'active',
            baseline_topic_label       TEXT    NOT NULL DEFAULT '',
            baseline_centroid_vector   TEXT    NOT NULL,
            baseline_centroid_dim      INTEGER NOT NULL DEFAULT 384,
            concern_level              REAL    NOT NULL,
            purchase_intent            REAL    NOT NULL,
            avoidance_signals          REAL    NOT NULL,
            dominant_frame             TEXT    NOT NULL,
            seg_young_urban            REAL    NOT NULL,
            seg_family                 REAL    NOT NULL,
            seg_senior                 REAL    NOT NULL,
            seg_b2b                    REAL    NOT NULL,
            sample_count               INTEGER NOT NULL DEFAULT 0,
            is_learned                 INTEGER NOT NULL DEFAULT 0,
            missed_runs                INTEGER NOT NULL DEFAULT 0,
            last_member_count          INTEGER NOT NULL DEFAULT 0,
            last_mean_membership_strength REAL NOT NULL DEFAULT 0.0,
            first_seen_run_id          TEXT    NOT NULL,
            last_seen_run_id           TEXT    NOT NULL,
            first_seen_at              TEXT    NOT NULL,
            last_seen_at               TEXT    NOT NULL,
            updated_at                 TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_tracks_scope_status
        ON cluster_tracks(topic, country, source, language, status, last_seen_at DESC)
        """
    )
    if "canonical_topic_id" in _table_columns(conn, "cluster_tracks"):
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cluster_tracks_canonical_scope_status
            ON cluster_tracks(canonical_topic_id, country, source, language, status, last_seen_at DESC)
            """
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_drift_runs (
            run_id                     TEXT PRIMARY KEY REFERENCES cluster_runs(run_id),
            topic                      TEXT    NOT NULL,
            canonical_topic_id         TEXT,
            country                    TEXT    NOT NULL DEFAULT '',
            source                     TEXT    NOT NULL DEFAULT '',
            language                   TEXT,
            observed_cluster_count     INTEGER NOT NULL,
            matched_track_count        INTEGER NOT NULL,
            new_track_count            INTEGER NOT NULL,
            missing_track_count        INTEGER NOT NULL,
            segment_count              INTEGER NOT NULL,
            computed_at                TEXT    NOT NULL,
            duration_s                 REAL    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_drift_runs_scope_computed
        ON cluster_drift_runs(topic, country, source, language, computed_at DESC)
        """
    )
    if "canonical_topic_id" in _table_columns(conn, "cluster_drift_runs"):
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cluster_drift_runs_canonical_scope_computed
            ON cluster_drift_runs(canonical_topic_id, country, source, language, computed_at DESC)
            """
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_drift_observations (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                     TEXT    NOT NULL REFERENCES cluster_drift_runs(run_id),
            track_id                   TEXT    NOT NULL REFERENCES cluster_tracks(track_id),
            cluster_id                 INTEGER REFERENCES clusters(id),
            cluster_label              INTEGER,
            topic_label                TEXT    NOT NULL DEFAULT '',
            baseline_topic_label       TEXT    NOT NULL DEFAULT '',
            match_type                 TEXT    NOT NULL,
            direction                  TEXT    NOT NULL,
            centroid_distance          REAL    NOT NULL DEFAULT 0.0,
            segment_vector_distance    REAL    NOT NULL DEFAULT 0.0,
            signal_drift               REAL    NOT NULL DEFAULT 0.0,
            drift_magnitude            REAL    NOT NULL DEFAULT 0.0,
            alert_level                TEXT    NOT NULL,
            confidence                 REAL    NOT NULL DEFAULT 0.0,
            member_count               INTEGER NOT NULL DEFAULT 0,
            mean_membership_strength   REAL    NOT NULL DEFAULT 0.0,
            concern_level              REAL    NOT NULL DEFAULT 0.0,
            purchase_intent            REAL    NOT NULL DEFAULT 0.0,
            avoidance_signals          REAL    NOT NULL DEFAULT 0.0,
            dominant_frame             TEXT    NOT NULL DEFAULT 'neutral',
            baseline_concern_level     REAL,
            baseline_purchase_intent   REAL,
            baseline_avoidance_signals REAL,
            baseline_dominant_frame    TEXT,
            delta_concern_level        REAL    NOT NULL DEFAULT 0.0,
            delta_purchase_intent      REAL    NOT NULL DEFAULT 0.0,
            delta_avoidance_signals    REAL    NOT NULL DEFAULT 0.0,
            seg_young_urban            REAL    NOT NULL DEFAULT 0.0,
            seg_family                 REAL    NOT NULL DEFAULT 0.0,
            seg_senior                 REAL    NOT NULL DEFAULT 0.0,
            seg_b2b                    REAL    NOT NULL DEFAULT 0.0,
            baseline_seg_young_urban   REAL    NOT NULL DEFAULT 0.0,
            baseline_seg_family        REAL    NOT NULL DEFAULT 0.0,
            baseline_seg_senior        REAL    NOT NULL DEFAULT 0.0,
            baseline_seg_b2b           REAL    NOT NULL DEFAULT 0.0,
            frame_shift                INTEGER NOT NULL DEFAULT 0,
            computed_at                TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_drift_observations_run_id
        ON cluster_drift_observations(run_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cluster_drift_observations_run_track
        ON cluster_drift_observations(run_id, track_id)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_segment_drifts (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                     TEXT    NOT NULL REFERENCES cluster_drift_runs(run_id),
            segment                    TEXT    NOT NULL,
            article_count              INTEGER NOT NULL DEFAULT 0,
            has_data                   INTEGER NOT NULL DEFAULT 0,
            concern_level              REAL    NOT NULL DEFAULT 0.0,
            purchase_intent            REAL    NOT NULL DEFAULT 0.0,
            avoidance_signals          REAL    NOT NULL DEFAULT 0.0,
            dominant_frame             TEXT    NOT NULL DEFAULT 'neutral',
            baseline_concern_level     REAL,
            baseline_purchase_intent   REAL,
            baseline_avoidance_signals REAL,
            baseline_dominant_frame    TEXT,
            delta_concern_level        REAL    NOT NULL DEFAULT 0.0,
            delta_purchase_intent      REAL    NOT NULL DEFAULT 0.0,
            delta_avoidance_signals    REAL    NOT NULL DEFAULT 0.0,
            drift_magnitude            REAL    NOT NULL DEFAULT 0.0,
            frame_shift                INTEGER NOT NULL DEFAULT 0,
            alert_level                TEXT    NOT NULL,
            confidence                 REAL    NOT NULL DEFAULT 0.0,
            baseline_is_learned        INTEGER NOT NULL DEFAULT 0,
            baseline_sample_count      INTEGER NOT NULL DEFAULT 0,
            baseline_age_days          INTEGER,
            status                     TEXT    NOT NULL,
            direction                  TEXT    NOT NULL DEFAULT 'stable',
            centroid_shift             REAL    NOT NULL DEFAULT 0.0,
            new_cluster_weight         REAL    NOT NULL DEFAULT 0.0,
            tracked_cluster_count      INTEGER NOT NULL DEFAULT 0,
            matched_cluster_count      INTEGER NOT NULL DEFAULT 0,
            new_cluster_count          INTEGER NOT NULL DEFAULT 0,
            missing_cluster_count      INTEGER NOT NULL DEFAULT 0,
            computed_at                TEXT    NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_cluster_segment_drifts_run_segment
        ON cluster_segment_drifts(run_id, segment)
        """
    )


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations for existing databases."""
    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    _ensure_topic_schema(conn)
    _seed_topic_catalog(conn)

    if "articles" in table_names:
        article_columns = _table_columns(conn, "articles")
        if "body" not in article_columns:
            conn.execute(
                """
                ALTER TABLE articles
                ADD COLUMN body TEXT
                """
            )
        if "country" not in article_columns:
            conn.execute(
                """
                ALTER TABLE articles
                ADD COLUMN country TEXT NOT NULL DEFAULT 'CZ'
                """
            )
        if "language" not in article_columns:
            conn.execute(
                """
                ALTER TABLE articles
                ADD COLUMN language TEXT NOT NULL DEFAULT 'cs'
                """
            )
        if "canonical_url" not in article_columns:
            conn.execute(
                """
                ALTER TABLE articles
                ADD COLUMN canonical_url TEXT
                """
            )
        if "canonical_topic_id" not in article_columns:
            conn.execute(
                """
                ALTER TABLE articles
                ADD COLUMN canonical_topic_id TEXT
                """
            )

    if "signals" in table_names:
        signal_columns = _table_columns(conn, "signals")
        for column in [
            "seg_young_urban_relevance",
            "seg_family_relevance",
            "seg_senior_relevance",
            "seg_b2b_relevance",
        ]:
            if column not in signal_columns:
                conn.execute(f"ALTER TABLE signals ADD COLUMN {column} REAL")

    columns = _table_columns(conn, "baselines") if "baselines" in table_names else set()

    if "baselines" in table_names and "sample_count" not in columns:
        conn.execute(
            """
            ALTER TABLE baselines
            ADD COLUMN sample_count INTEGER NOT NULL DEFAULT 0
            """
        )
    if "baselines" in table_names and "is_learned" not in columns:
        conn.execute(
            """
            ALTER TABLE baselines
            ADD COLUMN is_learned INTEGER NOT NULL DEFAULT 0
            """
        )
    if "baselines" in table_names and "canonical_topic_id" not in columns:
        conn.execute(
            """
            ALTER TABLE baselines
            ADD COLUMN canonical_topic_id TEXT
            """
        )

    if "articles" in table_names:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_topics (
                article_id       TEXT NOT NULL,
                topic            TEXT NOT NULL,
                raw_topic        TEXT,
                canonical_topic_id TEXT,
                relevance_score  REAL NOT NULL DEFAULT 1.0,
                matched_at       TEXT NOT NULL,
                PRIMARY KEY (article_id, topic),
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
            """
        )
        _add_column_if_missing(conn, "article_topics", "raw_topic", "raw_topic TEXT")
        _add_column_if_missing(
            conn,
            "article_topics",
            "canonical_topic_id",
            "canonical_topic_id TEXT",
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_article_topics_topic_matched_at
            ON article_topics(topic, matched_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_article_topics_canonical_matched_at
            ON article_topics(canonical_topic_id, matched_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_articles_country_outlet
            ON articles(country, outlet)
            """
        )
        conn.execute("DROP TRIGGER IF EXISTS trg_articles_insert_topic_link")
        conn.execute(
            """
            CREATE TRIGGER trg_articles_insert_topic_link
            AFTER INSERT ON articles
            WHEN NEW.topic IS NOT NULL AND TRIM(NEW.topic) != ''
            BEGIN
                INSERT OR IGNORE INTO article_topics
                (article_id, topic, raw_topic, canonical_topic_id, relevance_score, matched_at)
                VALUES (
                    NEW.id,
                    NEW.topic,
                    NEW.topic,
                    COALESCE(
                        NEW.canonical_topic_id,
                        (
                            SELECT canonical_topic_id
                            FROM topic_aliases
                            WHERE (
                                  normalized_topic = LOWER(TRIM(NEW.topic))
                                  OR raw_topic = LOWER(TRIM(NEW.topic))
                            )
                              AND (
                                  NEW.language IS NULL
                                  OR language = LOWER(TRIM(NEW.language))
                                  OR language IS NULL
                              )
                            ORDER BY
                              CASE
                                  WHEN language = LOWER(TRIM(NEW.language)) THEN 0
                                  WHEN language IS NULL THEN 1
                                  ELSE 2
                              END,
                              id ASC
                            LIMIT 1
                        ),
                        NEW.topic
                    ),
                    1.0,
                    COALESCE(NEW.fetched_at, CURRENT_TIMESTAMP)
                );
            END
            """
        )
        conn.execute("DROP TRIGGER IF EXISTS trg_articles_update_topic_link")
        conn.execute(
            """
            CREATE TRIGGER trg_articles_update_topic_link
            AFTER UPDATE OF topic ON articles
            WHEN NEW.topic IS NOT NULL AND TRIM(NEW.topic) != ''
            BEGIN
                INSERT OR IGNORE INTO article_topics
                (article_id, topic, raw_topic, canonical_topic_id, relevance_score, matched_at)
                VALUES (
                    NEW.id,
                    NEW.topic,
                    NEW.topic,
                    COALESCE(
                        NEW.canonical_topic_id,
                        (
                            SELECT canonical_topic_id
                            FROM topic_aliases
                            WHERE (
                                  normalized_topic = LOWER(TRIM(NEW.topic))
                                  OR raw_topic = LOWER(TRIM(NEW.topic))
                            )
                              AND (
                                  NEW.language IS NULL
                                  OR language = LOWER(TRIM(NEW.language))
                                  OR language IS NULL
                              )
                            ORDER BY
                              CASE
                                  WHEN language = LOWER(TRIM(NEW.language)) THEN 0
                                  WHEN language IS NULL THEN 1
                                  ELSE 2
                              END,
                              id ASC
                            LIMIT 1
                        ),
                        NEW.topic
                    ),
                    1.0,
                    COALESCE(NEW.fetched_at, CURRENT_TIMESTAMP)
                );
            END
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO article_topics(article_id, topic, relevance_score, matched_at)
            SELECT id, topic, 1.0, COALESCE(fetched_at, CURRENT_TIMESTAMP)
            FROM articles
            WHERE topic IS NOT NULL AND TRIM(topic) != ''
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_embeddings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id          TEXT    NOT NULL REFERENCES articles(id),
                model_name          TEXT    NOT NULL DEFAULT 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                model_version       TEXT,
                embedding_dim       INTEGER NOT NULL DEFAULT 384,
                embedding_vector    TEXT    NOT NULL,
                embedding_text      TEXT    NOT NULL,
                embedding_text_hash TEXT    NOT NULL,
                language            TEXT,
                status              TEXT    NOT NULL DEFAULT 'pending',
                error_message       TEXT,
                embedded_at         TEXT,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE (article_id, model_name, embedding_text_hash)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ae_article_model_updated
            ON article_embeddings(article_id, model_name, updated_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ae_model_status_updated
            ON article_embeddings(model_name, status, updated_at)
            """
        )
        _ensure_cluster_schema(conn)
        _ensure_cluster_signal_schema(conn)
        _ensure_cluster_drift_schema(conn)
    elif "cluster_signals" not in table_names:
        _ensure_cluster_signal_schema(conn)
    elif "cluster_drift_runs" not in table_names:
        _ensure_cluster_drift_schema(conn)

    for table_name in [
        "collection_runs",
        "segment_profiles",
        "baselines",
        "cluster_runs",
        "cluster_tracks",
        "cluster_drift_runs",
    ]:
        _add_column_if_missing(
            conn,
            table_name,
            "canonical_topic_id",
            "canonical_topic_id TEXT",
        )

    _execute_if_table_exists(
        conn,
        "articles",
        """
        CREATE INDEX IF NOT EXISTS idx_articles_canonical_topic
        ON articles(canonical_topic_id)
        """,
    )
    _execute_if_table_exists(
        conn,
        "segment_profiles",
        """
        CREATE INDEX IF NOT EXISTS idx_segment_profiles_canonical_segment_computed_at
        ON segment_profiles(canonical_topic_id, segment, computed_at)
        """,
    )
    _execute_if_table_exists(
        conn,
        "baselines",
        """
        CREATE INDEX IF NOT EXISTS idx_baselines_canonical_segment
        ON baselines(canonical_topic_id, segment)
        """,
    )
    _execute_if_table_exists(
        conn,
        "collection_runs",
        """
        CREATE INDEX IF NOT EXISTS idx_collection_runs_canonical_scope_completed
        ON collection_runs(canonical_topic_id, country, source, completed_at DESC)
        """,
    )
    _execute_if_table_exists(
        conn,
        "cluster_runs",
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_runs_canonical_scope_created
        ON cluster_runs(canonical_topic_id, country, source, language, created_at DESC)
        """,
    )
    _execute_if_table_exists(
        conn,
        "cluster_tracks",
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_tracks_canonical_scope_status
        ON cluster_tracks(canonical_topic_id, country, source, language, status, last_seen_at DESC)
        """,
    )
    _execute_if_table_exists(
        conn,
        "cluster_drift_runs",
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_drift_runs_canonical_scope_computed
        ON cluster_drift_runs(canonical_topic_id, country, source, language, computed_at DESC)
        """,
    )
    _add_column_if_missing(
        conn,
        "cluster_signals",
        "coherence_score",
        "coherence_score REAL NOT NULL DEFAULT 0.0",
    )

    _backfill_article_topic_links(conn)
    _backfill_canonical_topic_column(conn, "articles", language_column="language")
    _backfill_canonical_topic_column(conn, "collection_runs")
    _backfill_canonical_topic_column(conn, "segment_profiles")
    _backfill_canonical_topic_column(conn, "baselines")
    _backfill_canonical_topic_column(conn, "cluster_runs", language_column="language")
    _backfill_canonical_topic_column(conn, "cluster_tracks", language_column="language")
    _backfill_canonical_topic_column(conn, "cluster_drift_runs", language_column="language")


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
                body         TEXT,
                url          TEXT UNIQUE NOT NULL,
                canonical_url TEXT,
                topic        TEXT,
                canonical_topic_id TEXT,
                country      TEXT NOT NULL DEFAULT 'CZ',
                language     TEXT NOT NULL DEFAULT 'cs',
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
                seg_young_urban_relevance REAL,
                seg_family_relevance      REAL,
                seg_senior_relevance      REAL,
                seg_b2b_relevance         REAL,
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
                canonical_topic_id TEXT,
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
            CREATE INDEX IF NOT EXISTS idx_segment_profiles_topic_segment_computed_at
            ON segment_profiles(topic, segment, computed_at)
            """
        )
        if "canonical_topic_id" in _table_columns(conn, "segment_profiles"):
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segment_profiles_canonical_segment_computed_at
                ON segment_profiles(canonical_topic_id, segment, computed_at)
                """
            )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                id                TEXT PRIMARY KEY,
                topic             TEXT NOT NULL,
                canonical_topic_id TEXT,
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
        if "canonical_topic_id" in _table_columns(conn, "baselines"):
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_baselines_canonical_segment
                ON baselines(canonical_topic_id, segment)
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_embeddings (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id          TEXT    NOT NULL REFERENCES articles(id),
                model_name          TEXT    NOT NULL DEFAULT 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                model_version       TEXT,
                embedding_dim       INTEGER NOT NULL DEFAULT 384,
                embedding_vector    TEXT    NOT NULL,
                embedding_text      TEXT    NOT NULL,
                embedding_text_hash TEXT    NOT NULL,
                language            TEXT,
                status              TEXT    NOT NULL DEFAULT 'pending',
                error_message       TEXT,
                embedded_at         TEXT,
                created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE (article_id, model_name, embedding_text_hash)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ae_article_model_updated
            ON article_embeddings(article_id, model_name, updated_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ae_model_status_updated
            ON article_embeddings(model_name, status, updated_at)
            """
        )
        _ensure_cluster_schema(conn)
        _ensure_cluster_signal_schema(conn)
        _ensure_cluster_drift_schema(conn)
        _ensure_collection_schema(conn)
        run_migrations(conn)
        conn.commit()
        _local.conn = conn

    return _local.conn
