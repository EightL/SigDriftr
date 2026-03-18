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


def print_db_rows(db_path: Path, limit: int) -> None:
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        print(f"articles.count = {count}")
        rows = conn.execute(
            "SELECT outlet, title FROM articles ORDER BY rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("No rows found in articles.")
        return

    print(f"latest {len(rows)} rows:")
    for outlet, title in rows:
        print(f"- {outlet}: {title}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SigDriftr RSS ingestion endpoint and inspect SQLite output."
    )
    parser.add_argument("--topic", default="česk", help="Topic passed to POST /collect")
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
        "--limit",
        type=int,
        default=5,
        help="How many article rows to print",
    )
    args = parser.parse_args()

    try:
        wait_for_api(args.base_url, args.wait_seconds)
        result = call_collect(args.base_url, args.topic)
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("collect response:")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print()
    print_db_rows(Path(args.db_path), args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
