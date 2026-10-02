"""Core dataclasses shared by every module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["agent", "caller"]
Severity = Literal["critical", "major", "minor"]
Owner = Literal["prompt", "engine", "simulator", "harness"]
Verdict = Literal["PASS", "FAIL", "NEEDS_REVIEW"]
NextAction = Literal["end_call", "handoff"]

COMPLETE = "_complete"
GOAL_MARKER = "[GOAL_ACHIEVED]"


@dataclass(frozen=True)
class Message:
    turn: int
    role: Role
    text: str  # marker stripped, whitespace trimmed
    raw_text: str  # exactly as in the dataset
    had_marker: bool = False


@dataclass(frozen=True)
class Thread:
    thread_id: str
    messages: list[Message]
    num_turns: int
    simulator_goal_marker: bool  # simulator claimed success; comparison only, not ground truth

    def agent_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "agent"]

    def caller_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "caller"]


@dataclass(frozen=True)
class Scenario:
    id: str
    means: str
    goto: str | None = None  # raw goto, e.g. "item_q2_active:question_a" or "_complete"
    goto_item: str | None = None  # resolved item id or "_complete"
    disposition: str | None = None  # outcome id when this scenario terminates the call
    say: str | None = None
    do: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    type: Literal["choice", "boolean", "text"]
    scenarios: list[Scenario]
    options: list[str] | None = None
    default_route: str | None = None

    def scenario(self, scenario_id: str) -> Scenario | None:
        for s in self.scenarios:
            if s.id == scenario_id:
                return s
        return None


@dataclass(frozen=True)
class Item:
    id: str
    label: str
    questions: list[Question]

    @property
    def question(self) -> Question:
        """Every item in this dataset has exactly one question."""
        return self.questions[0]


@dataclass(frozen=True)
class Outcome:
    id: str
    say: str
    label: str
    next_action: NextAction
    flags: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowSpec:
    items: dict[str, Item]  # insertion order matches engine_config order
    outcomes: dict[str, Outcome]
    entry: str


@dataclass(frozen=True)
class Finding:
    check: str
    severity: Severity
    gate: str | None
    turn: int | None
    evidence: str
    problem: str
    fix_hint: str
    owner: Owner
    # True when the harness cannot confirm the problem (simulator cut the episode, judge missing or
    # unsure). An uncertain finding never yields FAIL on its own; it yields NEEDS_REVIEW.
    uncertain: bool = False


@dataclass
class ThreadReport:
    thread_id: str
    outcome_verdict: Verdict  # task gates only: correct_routing, decisive_answers, proper_termination
    release_verdict: Verdict  # outcome plus clean_speech under the speech policy
    findings: list[Finding] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    release_warnings: list[str] = field(default_factory=list)
    speech_policy: str = "gate"  # "gate" | "soft"
    expected_terminal: str | None = None
    actual_terminal: str | None = None
    simulator_goal_marker: bool = False
    gates: dict[str, str] = field(default_factory=dict)  # gate -> "pass" | "fail" | "uncertain" | "n/a"
    expected_path: list[str] = field(default_factory=list)
    judge_source: str = "none"  # "cache" | "live" | "fake" | "hand" | "none"
    judge_detail: str = ""  # e.g. "claude-sonnet-5-5 via claude-cli, prompt v2"; "" when there is no model

    @property
    def verdict(self) -> Verdict:
        """The overall verdict is the release verdict (it also drives the exit code)."""
        return self.release_verdict
