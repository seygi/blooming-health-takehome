"""Command line entry point: `uv run callcheck [path] [--thread ID] [--live] [--json]`."""

from __future__ import annotations

import argparse
import json
import sys

from callcheck.load import load

DEFAULT_PATH = "q3/data/gym_agent_conversations.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="callcheck", description="Evaluate triage call transcripts.")
    p.add_argument("path", nargs="?", default=DEFAULT_PATH, help="dataset JSON")
    p.add_argument("--thread", help="only this thread id")
    p.add_argument("--live", action="store_true", help="refresh model judgments (unused yet)")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec, threads = load(args.path)
    if args.thread:
        threads = [t for t in threads if t.thread_id == args.thread]
        if not threads:
            print(f"no thread with id {args.thread}", file=sys.stderr)
            return 2

    rows = [
        {
            "thread_id": t.thread_id,
            "turns": t.num_turns,
            "simulator_goal_marker": t.simulator_goal_marker,
        }
        for t in threads
    ]
    if args.json:
        print(json.dumps({"items": list(spec.items), "threads": rows}, indent=2))
        return 0

    print("items: " + " > ".join(spec.items))
    print(f"{'thread':<12}{'turns':>6}  simulator_marker")
    for r in rows:
        print(f"{r['thread_id']:<12}{r['turns']:>6}  {r['simulator_goal_marker']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
