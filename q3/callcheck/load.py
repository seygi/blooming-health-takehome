"""Load the dataset JSON into a FlowSpec and a list of Threads."""

from __future__ import annotations

import json
from pathlib import Path

from callcheck.model import GOAL_MARKER, FlowSpec, Message, Thread
from callcheck.spec import build_spec


def _message(raw: dict) -> Message:
    raw_text = raw["text"]
    had_marker = raw["role"] == "caller" and GOAL_MARKER in raw_text
    text = raw_text.replace(GOAL_MARKER, "").strip() if had_marker else raw_text.strip()
    return Message(
        turn=raw["turn"],
        role=raw["role"],
        text=text,
        raw_text=raw_text,
        had_marker=had_marker,
    )


def _thread(raw: dict) -> Thread:
    messages = [_message(m) for m in raw["messages"]]
    return Thread(
        thread_id=raw["thread_id"],
        messages=messages,
        num_turns=raw.get("num_turns", 0),
        simulator_goal_marker=any(m.had_marker for m in messages),
    )


def load(path: str | Path) -> tuple[FlowSpec, list[Thread]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = build_spec(data["agent_config"]["engine_config"])
    threads = [_thread(t) for t in data["threads"]]
    return spec, threads


def load_min_confidence(path: str | Path, default: float = 0.7) -> float:
    """team_config.configuration.rules_of_engagement.quality_gates.min_confidence_score, else default."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    node: object = data
    for key in ("team_config", "configuration", "rules_of_engagement", "quality_gates", "min_confidence_score"):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if isinstance(node, bool) or not isinstance(node, (int, float)) or not 0 <= node <= 1:
        return default
    return float(node)


def load_transport_mode(path: str | Path) -> str | None:
    """agent_config.transport_mode ("voice", ...) or None when absent."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    mode = data.get("agent_config", {}).get("transport_mode") if isinstance(data, dict) else None
    return mode if isinstance(mode, str) else None
