from __future__ import annotations

from db.init import get_conn
from db.topic_resolver import resolve_topic
from delta.cluster_matching import ACTIVE_TRACK_STATUS, _decode_vector
from delta.mapper import canonicalize_frame


REQUIRED_STAGE_FOUR_TABLES = (
    "cluster_runs",
    "clusters",
    "cluster_memberships",
    "cluster_signals",
)


def _assert_stage_four_ready(run_id: str) -> dict[str, object]:
    conn = get_conn()
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    missing_tables = [name for name in REQUIRED_STAGE_FOUR_TABLES if name not in tables]
    if missing_tables:
        raise RuntimeError(
            "Cluster drift requires completed stage-4 tables. Missing: "
            + ", ".join(sorted(missing_tables))
        )

    row = conn.execute(
        """
        SELECT run_id, topic, canonical_topic_id, country, source, language, status, n_clusters
        FROM cluster_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"Cluster run '{run_id}' was not found.")

    cluster_count = int(row[7] or 0)
    signal_count = conn.execute(
        "SELECT COUNT(*) FROM cluster_signals WHERE run_id = ?",
        (run_id,),
    ).fetchone()[0]
    if cluster_count > 0 and signal_count < cluster_count:
        raise RuntimeError(
            "Cluster drift requires stage 4 extraction for every cluster in the run."
        )

    canonical_topic_id = str(row[2] or "")
    if not canonical_topic_id and row[1]:
        canonical_topic_id = resolve_topic(str(row[1]), row[5]).canonical_topic_id
        conn.execute(
            """
            UPDATE cluster_runs
            SET canonical_topic_id = ?
            WHERE run_id = ?
              AND (canonical_topic_id IS NULL OR TRIM(canonical_topic_id) = '')
            """,
            (canonical_topic_id, row[0]),
        )
        conn.commit()

    return {
        "run_id": row[0],
        "topic": row[1],
        "canonical_topic_id": canonical_topic_id,
        "country": row[3],
        "source": row[4],
        "language": row[5],
        "status": row[6],
        "n_clusters": cluster_count,
    }


def _cluster_drift_exists(run_id: str) -> bool:
    conn = get_conn()
    row = conn.execute(
        "SELECT 1 FROM cluster_drift_runs WHERE run_id = ? LIMIT 1",
        (run_id,),
    ).fetchone()
    return row is not None


def _load_current_clusters(run_id: str) -> list[dict[str, object]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            c.id,
            c.cluster_label,
            c.size,
            c.centroid_vector,
            cs.topic_label,
            cs.concern_level,
            cs.purchase_intent,
            cs.avoidance_signals,
            cs.dominant_frame,
            cs.seg_young_urban,
            cs.seg_family,
            cs.seg_senior,
            cs.seg_b2b,
            COALESCE(cs.member_count, c.size),
            COALESCE(AVG(cm.membership_strength), 0.0)
        FROM clusters c
        JOIN cluster_signals cs ON cs.cluster_id = c.id
        LEFT JOIN cluster_memberships cm
          ON cm.cluster_id = c.id
         AND cm.run_id = c.run_id
         AND cm.is_noise = 0
        WHERE c.run_id = ?
        GROUP BY
            c.id, c.cluster_label, c.size, c.centroid_vector,
            cs.topic_label, cs.concern_level, cs.purchase_intent, cs.avoidance_signals,
            cs.dominant_frame, cs.seg_young_urban, cs.seg_family, cs.seg_senior,
            cs.seg_b2b, cs.member_count
        ORDER BY c.size DESC, c.cluster_label ASC
        """,
        (run_id,),
    ).fetchall()

    current_clusters: list[dict[str, object]] = []
    for row in rows:
        current_clusters.append(
            {
                "cluster_id": int(row[0]),
                "cluster_label": int(row[1]),
                "size": int(row[2]),
                "centroid_vector": _decode_vector(row[3]),
                "topic_label": row[4] or "",
                "concern_level": float(row[5] or 0.0),
                "purchase_intent": float(row[6] or 0.0),
                "avoidance_signals": float(row[7] or 0.0),
                "dominant_frame": canonicalize_frame(row[8]),
                "seg_young_urban": float(row[9] or 0.0),
                "seg_family": float(row[10] or 0.0),
                "seg_senior": float(row[11] or 0.0),
                "seg_b2b": float(row[12] or 0.0),
                "member_count": int(row[13] or 0),
                "mean_membership_strength": float(row[14] or 0.0),
            }
        )
    return current_clusters


def _load_active_tracks(
    topic: str,
    canonical_topic_id: str,
    country: str,
    source: str,
    language: str | None,
) -> list[dict[str, object]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT
            track_id,
            baseline_topic_label,
            baseline_centroid_vector,
            baseline_centroid_dim,
            concern_level,
            purchase_intent,
            avoidance_signals,
            dominant_frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
            sample_count,
            is_learned,
            missed_runs,
            last_member_count,
            last_mean_membership_strength,
            first_seen_run_id,
            last_seen_run_id,
            first_seen_at,
            last_seen_at,
            updated_at
        FROM cluster_tracks
        WHERE (
              topic = ?
              OR (? != '' AND canonical_topic_id = ?)
        )
          AND country = ?
          AND source = ?
          AND ((? IS NULL AND language IS NULL) OR language = ?)
          AND status = ?
        ORDER BY last_seen_at DESC, track_id ASC
        """,
        (
            topic,
            canonical_topic_id,
            canonical_topic_id,
            country,
            source,
            language,
            language,
            ACTIVE_TRACK_STATUS,
        ),
    ).fetchall()

    tracks: list[dict[str, object]] = []
    for row in rows:
        tracks.append(
            {
                "track_id": row[0],
                "baseline_topic_label": row[1] or "",
                "baseline_centroid_vector": _decode_vector(row[2]),
                "baseline_centroid_dim": int(row[3]),
                "concern_level": float(row[4] or 0.0),
                "purchase_intent": float(row[5] or 0.0),
                "avoidance_signals": float(row[6] or 0.0),
                "dominant_frame": canonicalize_frame(row[7]),
                "seg_young_urban": float(row[8] or 0.0),
                "seg_family": float(row[9] or 0.0),
                "seg_senior": float(row[10] or 0.0),
                "seg_b2b": float(row[11] or 0.0),
                "sample_count": int(row[12] or 0),
                "is_learned": bool(row[13]),
                "missed_runs": int(row[14] or 0),
                "last_member_count": int(row[15] or 0),
                "last_mean_membership_strength": float(row[16] or 0.0),
                "first_seen_run_id": row[17],
                "last_seen_run_id": row[18],
                "first_seen_at": row[19],
                "last_seen_at": row[20],
                "updated_at": row[21],
            }
        )
    return tracks
