#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.init import get_conn
from ingestion.bandit import reset_bandit_state, warm_start_from_history


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay historical SigDriftr collection rewards into the feed bandit."
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
    parser.add_argument(
        "--reward-mode",
        choices=["yield", "signal"],
        default="yield",
        help=(
            "History source to replay. 'yield' uses non-LLM collection stats; "
            "'signal' replays legacy LLM signal rewards."
        ),
    )
    args = parser.parse_args()

    get_conn()
    if args.reset:
        reset_bandit_state()

    updated = warm_start_from_history(
        topic=args.topic,
        limit=args.limit,
        reward_mode=args.reward_mode,
    )
    print(f"Updated bandit from {updated} historical {args.reward_mode} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
