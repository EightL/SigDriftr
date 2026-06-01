#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from clustering.stability import evaluate_cluster_stability


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return normalized.strip("_") or "topic"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate bootstrap stability for SigDriftr storyline clusters."
    )
    parser.add_argument("--topic", required=True)
    parser.add_argument("--country", default="")
    parser.add_argument("--source", default="")
    parser.add_argument("--language", default=None)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=25)
    parser.add_argument("--sample-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--umap-components", type=int, default=10)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to eval/reports/cluster_stability_<topic>.json.",
    )
    args = parser.parse_args()

    report = evaluate_cluster_stability(
        topic=args.topic,
        country=args.country,
        source=args.source,
        language=args.language,
        window_hours=args.window_hours,
        bootstrap_samples=args.bootstrap_samples,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
    )

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("eval")
            / "reports"
            / f"cluster_stability_{_safe_name(args.topic)}.json"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
