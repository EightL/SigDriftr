#!/usr/bin/env python3
"""Replay historical signal data into the LinUCB bandit."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.bandit import warm_start_from_history


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
    args = parser.parse_args()

    print(
        "Warm-starting bandit "
        f"(topic={args.topic or 'all'}, limit={args.limit or 'unlimited'})..."
    )
    replayed = warm_start_from_history(topic=args.topic, limit=args.limit)
    print(f"Done. Replayed {replayed} signal records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
