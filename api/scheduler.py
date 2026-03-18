from __future__ import annotations

import logging

from api.pipeline import run_collection_cycle_sync
from config.settings import PIPELINE_INTERVAL_MINUTES, SCHEDULED_TOPICS
from delta.mapper import compute_segment_profiles

try:
    from apscheduler.schedulers.background import BackgroundScheduler
except ModuleNotFoundError:  # pragma: no cover - exercised when dependency is absent.
    BackgroundScheduler = None


logger = logging.getLogger(__name__)
_scheduler = None


def _run_topic_pipeline(topic: str) -> None:
    try:
        result = run_collection_cycle_sync(topic)
        extracted = int(result.get("extracted", 0))
        if extracted > 0:
            compute_segment_profiles(topic, learn_baseline=True)
        logger.info(
            "[scheduler] topic=%s inserted=%s extracted=%s rewards=%s",
            topic,
            result.get("inserted", 0),
            extracted,
            result.get("rewards_recorded", 0),
        )
    except Exception as exc:  # pragma: no cover - logging path.
        logger.warning("[scheduler] topic=%s error=%s", topic, exc)


def start_scheduler() -> None:
    global _scheduler

    if _scheduler is not None:
        return
    if BackgroundScheduler is None:
        logger.warning("[scheduler] APScheduler not installed; background jobs disabled.")
        return

    scheduler = BackgroundScheduler(daemon=True)
    for topic in SCHEDULED_TOPICS:
        scheduler.add_job(
            _run_topic_pipeline,
            "interval",
            minutes=PIPELINE_INTERVAL_MINUTES,
            args=[topic],
            id=f"pipeline_{topic}",
            max_instances=1,
        )
    scheduler.start()
    _scheduler = scheduler
    logger.info(
        "[scheduler] started %d jobs every %d min",
        len(SCHEDULED_TOPICS),
        PIPELINE_INTERVAL_MINUTES,
    )


def stop_scheduler() -> None:
    global _scheduler

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
