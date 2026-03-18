#!/usr/bin/env python3
import argparse
import hashlib
import sys
import tempfile
from unittest.mock import patch

import db.init
from delta.engine import compute_drift, update_baseline_from_profile
from delta.mapper import compute_segment_profiles
from delta.seeder import DEFAULT_TOPICS, SEGMENTS, ensure_topic_baselines, seed_baselines


def setup_temp_db() -> tempfile.TemporaryDirectory:
    temp_dir = tempfile.TemporaryDirectory()
    db.init.DB_PATH = db.init.Path(temp_dir.name) / "sigdriftr.db"
    if hasattr(db.init._local, "conn"):
        db.init._local.conn.close()
        delattr(db.init._local, "conn")
    db.init.get_conn()
    return temp_dir


def insert_article_with_signal(
    article_id: str,
    topic: str,
    concern: float,
    purchase: float,
    avoidance: float,
    frame: str,
    seg_young_urban: float,
    seg_family: float,
    seg_senior: float,
    seg_b2b: float,
) -> None:
    conn = db.init.get_conn()
    conn.execute(
        """
        INSERT INTO articles
        (id, outlet, title, summary, url, topic, published_at, fetched_at)
        VALUES (?, 'unit-test', ?, '', ?, ?, '2026-03-17T00:00:00+00:00', '2026-03-17T00:00:00+00:00')
        """,
        (article_id, article_id, f"https://example.test/{article_id}", topic),
    )
    conn.execute(
        """
        INSERT INTO signals
        (article_id, concern_level, purchase_intent, avoidance_signals,
         dominant_frame, seg_young_urban, seg_family, seg_senior, seg_b2b,
         raw_json, extracted_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '2026-03-17T00:00:00+00:00')
        """,
        (
            article_id,
            concern,
            purchase,
            avoidance,
            frame,
            seg_young_urban,
            seg_family,
            seg_senior,
            seg_b2b,
        ),
    )
    conn.commit()


def test_seed_baselines_is_idempotent() -> None:
    temp_dir = setup_temp_db()
    try:
        inserted = seed_baselines()
        assert inserted == len(DEFAULT_TOPICS) * len(SEGMENTS)

        inserted_again = seed_baselines()
        assert inserted_again == 0
    finally:
        temp_dir.cleanup()


def test_compute_drift_uses_weighted_segment_profiles() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="a1",
            topic="inflace",
            concern=0.8,
            purchase=0.1,
            avoidance=0.2,
            frame="fear",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )
        insert_article_with_signal(
            article_id="a2",
            topic="inflace",
            concern=0.2,
            purchase=0.9,
            avoidance=0.4,
            frame="opportunity",
            seg_young_urban=0.0,
            seg_family=1.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        profiles = compute_segment_profiles("inflace")
        profile_by_segment = {profile["segment"]: profile for profile in profiles}
        assert profile_by_segment["young_urban"]["concern_level"] == 0.8
        assert profile_by_segment["family"]["purchase_intent"] == 0.9
        assert profile_by_segment["senior"]["article_count"] == 0
        assert profile_by_segment["young_urban"]["window_days"] == 7

        drift = compute_drift("inflace")
        drift_by_segment = {entry["segment"]: entry for entry in drift}
        assert drift_by_segment["young_urban"]["deltas"]["concern_level"] == 0.42
        assert drift_by_segment["young_urban"]["domain"] == "commerce"
        assert drift_by_segment["young_urban"]["relevant_fields"] == [
            "concern_level",
            "purchase_intent",
            "avoidance_signals",
        ]
        assert drift_by_segment["family"]["frame_shift"] is True
        assert drift_by_segment["young_urban"]["status"] == "warming"
        assert drift_by_segment["senior"]["alert_level"] == "no_data"
        assert drift_by_segment["senior"]["status"] == "no_data"
        assert drift_by_segment["senior"]["deltas"] == {
            "concern_level": 0.0,
            "purchase_intent": 0.0,
            "avoidance_signals": 0.0,
        }
    finally:
        temp_dir.cleanup()


def test_update_baseline_from_profile_blends_values() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        profile = {
            "article_count": 30,
            "concern_level": 0.88,
            "purchase_intent": 0.11,
            "avoidance_signals": 0.33,
            "dominant_frame": "fear",
        }

        update_baseline_from_profile("inflace", "young_urban", profile)

        conn = db.init.get_conn()
        row_id = hashlib.sha256("inflace:young_urban".encode()).hexdigest()
        row = conn.execute(
            """
            SELECT concern_level, purchase_intent, avoidance_signals, dominant_frame,
                   seeded, sample_count, is_learned
            FROM baselines WHERE id = ?
            """,
            (row_id,),
        ).fetchone()

        assert row == (0.48, 0.27, 0.21, "fear", 0, 1, 1)
    finally:
        temp_dir.cleanup()


def test_compute_drift_seeds_unknown_topics_on_demand() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article_with_signal(
            article_id="custom-1",
            topic="custom-topic",
            concern=0.9,
            purchase=0.1,
            avoidance=0.5,
            frame="fear",
            seg_young_urban=0.7,
            seg_family=0.1,
            seg_senior=0.1,
            seg_b2b=0.1,
        )

        drift = compute_drift("custom-topic")
        first_segment = next(entry for entry in drift if entry["segment"] == "young_urban")
        assert first_segment["baseline"] is not None
    finally:
        temp_dir.cleanup()


def test_compute_drift_keeps_baseline_frame_key_when_baseline_missing() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article_with_signal(
            article_id="missing-baseline-1",
            topic="custom-topic",
            concern=0.8,
            purchase=0.2,
            avoidance=0.4,
            frame="fear",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        with patch("delta.engine._get_baseline", return_value=None):
            drift = compute_drift("custom-topic")
    finally:
        temp_dir.cleanup()

    drift_by_segment = {entry["segment"]: entry for entry in drift}
    assert drift_by_segment["young_urban"]["baseline_frame"] is None
    assert set(drift_by_segment["young_urban"].keys()) == set(
        drift_by_segment["senior"].keys()
    )


def test_compute_segment_profiles_persists_requested_window_days() -> None:
    temp_dir = setup_temp_db()
    try:
        insert_article_with_signal(
            article_id="window-test",
            topic="inflace",
            concern=0.4,
            purchase=0.3,
            avoidance=0.2,
            frame="neutral",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        compute_segment_profiles("inflace", days_back=14)

        conn = db.init.get_conn()
        row = conn.execute(
            """
            SELECT window_days
            FROM segment_profiles
            WHERE topic = ? AND segment = ?
            """,
            ("inflace", "young_urban"),
        ).fetchone()

        assert row == (14,)
    finally:
        temp_dir.cleanup()


def test_frame_shift_uses_canonical_frame_labels() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["custom-topic"])
        insert_article_with_signal(
            article_id="frame-test",
            topic="custom-topic",
            concern=0.7,
            purchase=0.2,
            avoidance=0.4,
            frame="concern",
            seg_young_urban=0.0,
            seg_family=0.0,
            seg_senior=1.0,
            seg_b2b=0.0,
        )

        drift = compute_drift("custom-topic")
        senior = next(entry for entry in drift if entry["segment"] == "senior")
        assert senior["dominant_frame"] == "fear"
        assert senior["baseline_frame"] == "fear"
        assert senior["frame_shift"] is False
    finally:
        temp_dir.cleanup()


def test_alert_level_uses_drift_magnitude() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])
        insert_article_with_signal(
            article_id="magnitude-test",
            topic="inflace",
            concern=0.56,
            purchase=0.49,
            avoidance=0.36,
            frame="opportunity",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        drift = compute_drift("inflace")
        young_urban = next(entry for entry in drift if entry["segment"] == "young_urban")
        assert young_urban["drift_magnitude"] == 0.18
        assert young_urban["alert_level"] == "none"
    finally:
        temp_dir.cleanup()


def test_compute_drift_uses_domain_weights_for_generic_topics() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["custom-topic"])
        insert_article_with_signal(
            article_id="generic-weight-test",
            topic="custom-topic",
            concern=0.5,
            purchase=0.9,
            avoidance=0.9,
            frame="neutral",
            seg_young_urban=1.0,
            seg_family=0.0,
            seg_senior=0.0,
            seg_b2b=0.0,
        )

        drift = compute_drift("custom-topic")
        young_urban = next(entry for entry in drift if entry["segment"] == "young_urban")

        assert young_urban["domain"] == "generic"
        assert young_urban["relevant_fields"] == ["concern_level"]
        assert young_urban["deltas"]["purchase_intent"] == 0.59
        assert young_urban["deltas"]["avoidance_signals"] == 0.72
        assert young_urban["drift_magnitude"] == 0.12
        assert young_urban["alert_level"] == "none"
    finally:
        temp_dir.cleanup()


def test_empty_window_returns_no_data_alerts() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])

        drift = compute_drift("inflace")

        assert len(drift) == 4
        assert all(entry["alert_level"] == "no_data" for entry in drift)
        assert all(entry["drift_magnitude"] == 0.0 for entry in drift)
        assert all(entry["has_data"] is False for entry in drift)
        assert all(entry["status"] == "no_data" for entry in drift)
    finally:
        temp_dir.cleanup()


def test_compute_drift_reports_segment_status_levels() -> None:
    temp_dir = setup_temp_db()
    try:
        seed_baselines(["inflace"])

        for index in range(60):
            insert_article_with_signal(
                article_id=f"ready-{index}",
                topic="inflace",
                concern=0.7,
                purchase=0.2,
                avoidance=0.3,
                frame="fear",
                seg_young_urban=1.0,
                seg_family=0.0,
                seg_senior=0.0,
                seg_b2b=0.0,
            )

        for index in range(5):
            insert_article_with_signal(
                article_id=f"warming-{index}",
                topic="inflace",
                concern=0.4,
                purchase=0.1,
                avoidance=0.2,
                frame="neutral",
                seg_young_urban=0.0,
                seg_family=1.0,
                seg_senior=0.0,
                seg_b2b=0.0,
            )

        compute_segment_profiles("inflace", learn_baseline=True)
        drift = compute_drift("inflace")
        drift_by_segment = {entry["segment"]: entry for entry in drift}

        assert drift_by_segment["young_urban"]["status"] == "ready"
        assert drift_by_segment["family"]["status"] == "warming"
        assert drift_by_segment["senior"]["status"] == "no_data"
    finally:
        temp_dir.cleanup()


def main() -> int:
    try:
        from extraction.extractor import run_extraction
        from ingestion.crawler import crawl
    except ModuleNotFoundError as exc:
        print(
            "Live smoke test requires the project runtime dependencies. "
            f"Missing module: {exc.name}",
            file=sys.stderr,
        )
        return 1

    parser = argparse.ArgumentParser(
        description="Run an end-to-end stage 3 smoke test against the live SigDriftr DB."
    )
    parser.add_argument(
        "--topic",
        default="inflace",
        help="Topic to crawl, extract, and compute drift for",
    )
    parser.add_argument(
        "--skip-crawl",
        action="store_true",
        help="Skip RSS crawl and only run extraction + drift against existing articles",
    )
    args = parser.parse_args()

    seeded = ensure_topic_baselines(args.topic)
    print(f"Baseline rows seeded: {seeded}")

    if not args.skip_crawl:
        crawled = crawl(args.topic)
        print(f"Crawled: {crawled}")

    extracted = run_extraction(args.topic)
    print(f"Extracted: {extracted}")

    drift = compute_drift(args.topic)
    assert len(drift) == 4, f"Expected 4 segments, got {len(drift)}"

    for entry in drift:
        print(
            f"  [{entry['segment']:12}] "
            f"concern delta={entry['deltas']['concern_level']:+.3f} "
            f"magnitude={entry['drift_magnitude']:.3f} "
            f"alert={entry['alert_level']:6} "
            f"frame_shift={entry['frame_shift']}"
        )
        assert entry["alert_level"] in ("none", "mild", "strong", "no_data")
        assert entry["baseline"] is not None

    print("\nAll delta engine checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
