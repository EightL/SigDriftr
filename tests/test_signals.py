#!/usr/bin/env python3
import argparse
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def wait_for_api(base_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    health_url = f"{base_url}/docs"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        f"API did not become reachable at {base_url} within {timeout_seconds:.1f}s."
    )


def call_collect(base_url: str, topic: str) -> dict:
    query = urllib.parse.urlencode({"topic": topic})
    url = f"{base_url}/collect?{query}"
    request = urllib.request.Request(url, method="POST")

    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def call_extract(base_url: str, topic: str, timeout_seconds: float) -> dict:
    query = urllib.parse.urlencode({"topic": topic})
    url = f"{base_url}/extract?{query}"
    request = urllib.request.Request(url, method="POST")

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def call_signals(base_url: str, topic: str, timeout_seconds: float) -> list[dict]:
    query = urllib.parse.urlencode({"topic": topic})
    url = f"{base_url}/signals?{query}"

    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
        data = json.loads(body)

    if not isinstance(data, list):
        raise RuntimeError("Expected /signals to return a JSON list.")

    return data


def print_db_stats(db_path: Path) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        article_count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        signal_count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        topic_counts = conn.execute(
            """
            SELECT topic, COUNT(*)
            FROM articles
            GROUP BY topic
            ORDER BY COUNT(*) DESC, topic ASC
            LIMIT 10
            """
        ).fetchall()
    finally:
        conn.close()

    print(f"articles.count = {article_count}")
    print(f"signals.count = {signal_count}")
    if topic_counts:
        print("top article topics:")
        for topic, count in topic_counts:
            print(f"- {topic}: {count}")


def print_signal_sample(signals: list[dict]) -> None:
    print(f"signals.returned = {len(signals)}")
    if not signals:
        print("No signal rows returned.")
        return

    print("sample signal:")
    print(json.dumps(signals[0], ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SigDriftr signals endpoint and inspect SQLite output."
    )
    parser.add_argument(
        "--topic",
        default="polit",
        help="Topic passed to POST /collect, POST /extract, and GET /signals",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running FastAPI app",
    )
    parser.add_argument(
        "--db-path",
        default="data/sigdriftr.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=10.0,
        help="How long to wait for the API to come up",
    )
    parser.add_argument(
        "--signals-timeout",
        type=float,
        default=180.0,
        help="How long to wait for POST /extract and GET /signals to finish",
    )
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip POST /collect before running extraction and signal reads",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip POST /extract and only inspect already stored signals",
    )
    args = parser.parse_args()

    try:
        wait_for_api(args.base_url, args.wait_seconds)

        if not args.skip_collect:
            collect_result = call_collect(args.base_url, args.topic)
            print("collect response:")
            print(json.dumps(collect_result, ensure_ascii=False, indent=2))
            print()

        if not args.skip_extract:
            extract_result = call_extract(
                args.base_url, args.topic, args.signals_timeout
            )
            print("extract response:")
            print(json.dumps(extract_result, ensure_ascii=False, indent=2))
            print()

        start = time.time()
        signals = call_signals(args.base_url, args.topic, args.signals_timeout)
        elapsed = time.time() - start
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"signals elapsed_seconds = {elapsed:.2f}")
    print_signal_sample(signals)
    print()
    print_db_stats(Path(args.db_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
