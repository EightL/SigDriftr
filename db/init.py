import sqlite3
import threading
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "sigdriftr.db"

DATA_DIR.mkdir(parents=True, exist_ok=True)

_local = threading.local()


def _ensure_cluster_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_runs (
            id                        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                    TEXT    NOT NULL UNIQUE,
            topic                     TEXT    NOT NULL,
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


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply idempotent schema migrations for existing databases."""
    table_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    if "articles" in table_names:
        article_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(articles)").fetchall()
        }
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

    if "articles" in table_names:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS article_topics (
                article_id       TEXT NOT NULL,
                topic            TEXT NOT NULL,
                relevance_score  REAL NOT NULL DEFAULT 1.0,
                matched_at       TEXT NOT NULL,
                PRIMARY KEY (article_id, topic),
                FOREIGN KEY (article_id) REFERENCES articles(id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_article_topics_topic_matched_at
            ON article_topics(topic, matched_at DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_articles_country_outlet
            ON articles(country, outlet)
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_articles_insert_topic_link
            AFTER INSERT ON articles
            WHEN NEW.topic IS NOT NULL AND TRIM(NEW.topic) != ''
            BEGIN
                INSERT OR IGNORE INTO article_topics(article_id, topic, relevance_score, matched_at)
                VALUES (NEW.id, NEW.topic, 1.0, COALESCE(NEW.fetched_at, CURRENT_TIMESTAMP));
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS trg_articles_update_topic_link
            AFTER UPDATE OF topic ON articles
            WHEN NEW.topic IS NOT NULL AND TRIM(NEW.topic) != ''
            BEGIN
                INSERT OR IGNORE INTO article_topics(article_id, topic, relevance_score, matched_at)
                VALUES (NEW.id, NEW.topic, 1.0, COALESCE(NEW.fetched_at, CURRENT_TIMESTAMP));
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
            CREATE INDEX IF NOT EXISTS idx_segment_profiles_topic_segment_computed_at
            ON segment_profiles(topic, segment, computed_at)
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
        run_migrations(conn)
        conn.commit()
        _local.conn = conn

    return _local.conn
