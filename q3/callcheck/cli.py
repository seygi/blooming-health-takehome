"""Command line entry point.

    uv run callcheck [path] [--thread ID] [--cached] [--json] [--out FILE] [--captured FILE]
                     [--speech-policy gate|soft] [--labels judge|hand] [--agreement]

Each thread gets a task outcome verdict (routing, answers, termination) and a release verdict (outcome
plus clean speech under the speech policy). Exit codes follow the release verdict: 0 all PASS, 1 any
FAIL, 2 NEEDS_REVIEW and no FAIL, 3 usage or input error.

Judge modes. Default: live, the model judges every thread (ANTHROPIC_API_KEY required, or
CALLCHECK_BACKEND=claude-cli for the local Claude Code login) and the results are written to the cache.
--cached: replay q3/cache/judgments.json only, no key and no network. --labels hand: offline hand labels.
A missing or rejected key is a usage error (exit 3), never a silent NEEDS_REVIEW for every thread.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from callcheck.hand_labels import agreement, hand_judge
from callcheck.judge import JudgeAuthError, cached_judge, default_judge, live_unavailable_reason
from callcheck.load import load, load_min_confidence, load_transport_mode
from callcheck.report import DEFAULT_JSON, evaluate, exit_code, render_agreement, render_text, write_json
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
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--cached", action="store_true",
                      help="replay the committed judgments in q3/cache/judgments.json only: no key, no network. "
                      "Without it the model judge runs live and needs ANTHROPIC_API_KEY (or "
                      "CALLCHECK_BACKEND=claude-cli)")
    mode.add_argument("--live", action="store_true",
                      help="no-op kept for backward compatibility: live is already the default")
    p.add_argument("--json", action="store_true", help="write the JSON report and print its path")
    p.add_argument("--out", default=str(DEFAULT_JSON), help="JSON report path (default q3/out/report.json)")
    p.add_argument("--captured", help='JSON file {thread_id: {item_id: value}} of values the agent recorded')
    p.add_argument(
        "--speech-policy",
        choices=SPEECH_POLICIES,
        help="gate: a speech leak blocks release; soft: it is a release warning. Default: gate when "
        "agent_config.transport_mode is voice, else soft",
    )
    p.add_argument(
        "--labels",
        choices=("judge", "hand"),
        default="judge",
        help="judge: model judgments (live by default, or --cached); hand: the author's own labels from "
        "callcheck.hand_labels, NOT model output, offline",
    )
    p.add_argument(
        "--agreement",
        action="store_true",
        help="print how often the model judge's answers match the hand labels, per item and per thread",
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


MISSING_BACKEND_HELP = """\
callcheck: cannot run the live judge: {reason}
The default run asks the model to judge every thread. Pick one:
  export ANTHROPIC_API_KEY=sk-ant-...   then rerun: uv run callcheck
  uv run callcheck --cached             replay the committed run, no key needed
  uv run callcheck --labels hand        the author's hand labels, offline
  CALLCHECK_BACKEND=claude-cli uv run callcheck   judge through a local Claude Code subscription
"""


def _auth_error(e: JudgeAuthError) -> int:
    print(MISSING_BACKEND_HELP.format(reason=f"the Anthropic API rejected ANTHROPIC_API_KEY ({e})."),
          end="", file=sys.stderr)
    return USAGE_ERROR


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

    if args.agreement and args.labels == "hand":
        print("callcheck: --agreement compares the model judge with the hand labels; drop --labels hand",
              file=sys.stderr)
        return USAGE_ERROR
    if args.labels == "hand":
        judge = hand_judge()
    elif args.cached:
        judge = cached_judge()
    else:
        reason = live_unavailable_reason()
        if reason:
            print(MISSING_BACKEND_HELP.format(reason=reason), end="", file=sys.stderr)
            return USAGE_ERROR
        judge = default_judge(live=True)

    try:
        if args.agreement:
            desc = f"model judge ({'cache' if args.cached else 'live'}, model {getattr(judge, 'model', '?')})"
            sys.stdout.write(render_agreement(agreement(spec, threads, judge), list(spec.items), desc))
            return 0
        reports = [evaluate(spec, t, judge, captured.get(t.thread_id), min_conf, policy) for t in threads]
    except JudgeAuthError as e:
        return _auth_error(e)

    if args.json:
        print(_display(write_json(reports, args.out)))
    else:
        sys.stdout.write(render_text(reports, policy_note))
    return exit_code(reports)


if __name__ == "__main__":
    raise SystemExit(main())
