"""Command line entry point.

    uv run callcheck [path] [--thread ID] [--live] [--json] [--out FILE] [--captured FILE]
                     [--speech-policy gate|soft]

Each thread gets a task outcome verdict (routing, answers, termination) and a release verdict (outcome
plus clean speech under the speech policy). Exit codes follow the release verdict: 0 all PASS, 1 any
FAIL, 2 NEEDS_REVIEW and no FAIL, 3 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from callcheck.judge import default_judge
from callcheck.load import load, load_min_confidence, load_transport_mode
from callcheck.report import DEFAULT_JSON, evaluate, exit_code, render_text, write_json
from callcheck.routing import MIN_CONFIDENCE
from callcheck.verdict import SPEECH_POLICIES

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "gym_agent_conversations.json"
USAGE_ERROR = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="callcheck",
        description="Evaluate triage call transcripts against the engine_config flow.",
        epilog="Exit codes follow the release verdict: 0 all PASS, 1 any FAIL, 2 NEEDS_REVIEW and no FAIL, "
        "3 usage or input error.",
    )
    p.add_argument("path", nargs="?", default=str(DEFAULT_PATH), help="dataset JSON")
    p.add_argument("--thread", help="only this thread id")
    p.add_argument("--live", action="store_true", help="refresh model judgments (needs ANTHROPIC_API_KEY)")
    p.add_argument("--json", action="store_true", help="write the JSON report and print its path")
    p.add_argument("--out", default=str(DEFAULT_JSON), help="JSON report path (default q3/out/report.json)")
    p.add_argument("--captured", help='JSON file {thread_id: {item_id: value}} of values the agent recorded')
    p.add_argument(
        "--speech-policy",
        choices=SPEECH_POLICIES,
        help="gate: a speech leak blocks release; soft: it is a release warning. Default: gate when "
        "agent_config.transport_mode is voice, else soft",
    )
    return p.parse_args(argv)


def _load_captured(path: str) -> dict[str, dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ValueError("expected {thread_id: {item_id: value}}")
    return {tid: {iid: str(val) for iid, val in items.items()} for tid, items in data.items()}


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def speech_policy(explicit: str | None, transport_mode: str | None) -> tuple[str, str]:
    """(policy, header note). Voice speaks every leak aloud, so speech gates release there by default."""
    if explicit:
        return explicit, f"{explicit} (set by --speech-policy)"
    policy = "gate" if transport_mode == "voice" else "soft"
    return policy, f"{policy} (default for agent_config.transport_mode = {transport_mode or 'unset'})"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec, threads = load(args.path)
        min_conf = load_min_confidence(args.path, MIN_CONFIDENCE)
        policy, policy_note = speech_policy(args.speech_policy, load_transport_mode(args.path))
        captured = _load_captured(args.captured) if args.captured else {}
    except (OSError, ValueError, KeyError) as e:
        print(f"callcheck: cannot read input: {e}", file=sys.stderr)
        return USAGE_ERROR
    if args.thread:
        threads = [t for t in threads if t.thread_id == args.thread]
        if not threads:
            print(f"callcheck: no thread with id {args.thread}", file=sys.stderr)
            return USAGE_ERROR

    judge = default_judge(live=args.live)
    reports = [evaluate(spec, t, judge, captured.get(t.thread_id), min_conf, policy) for t in threads]

    if args.json:
        print(_display(write_json(reports, args.out)))
    else:
        sys.stdout.write(render_text(reports, policy_note))
    return exit_code(reports)


if __name__ == "__main__":
    raise SystemExit(main())
