#!/usr/bin/env python3
"""Run the Stage 2 embedding job from the command line."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extraction.embedding_service import embed_pending_articles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Embed stored SigDriftr articles into reusable semantic vectors."
    )
    parser.add_argument("--topic", default=None, help="Optional topic filter.")
    parser.add_argument("--country", default=None, help="Optional country filter.")
    parser.add_argument("--source", default=None, help="Optional source/outlet filter.")
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of matching articles to inspect.",
    )
    args = parser.parse_args()

    result = embed_pending_articles(
        limit=args.limit,
        topic=args.topic,
        country=args.country,
        source=args.source,
    )
    print(
        f"Embedded {result['embedded']} articles | "
        f"Already current {result['already_current']} | "
        f"Retried failed {result['retried_failed']} | "
        f"Stale re-embedded {result['stale_reembedded']} | "
        f"Failed {result['failed']} | "
        f"Time: {result['duration_s']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
