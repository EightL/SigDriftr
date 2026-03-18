#!/usr/bin/env python3
import argparse

from db.init import get_conn
from ingestion.bandit import reset_bandit_state, warm_start_from_history


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay historical SigDriftr signals into the feed bandit."
    )
    parser.add_argument(
        "--topic",
        default="",
        help="Only replay rows for one topic. Default replays all topics.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smaller warm-start runs.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing bandit state before replaying history.",
    )
    args = parser.parse_args()

    get_conn()
    if args.reset:
        reset_bandit_state()

    updated = warm_start_from_history(topic=args.topic, limit=args.limit)
    print(f"Updated bandit from {updated} historical signal rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
