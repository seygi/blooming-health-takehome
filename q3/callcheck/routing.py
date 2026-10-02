"""Routing: walk the flow graph with the judge's labels and compare with what the agent did.

Inputs are the deterministic alignment (what the agent asked and which terminal it delivered) and the
judgment (what the caller actually said, per item). Outputs are findings for the correct_routing and
decisive_answers gates plus a RoutingResult for the report.

Decisions
    Capture only items (every scenario is a placeholder, e.g. q3_plan `__close__`) route the same way
    whatever the caller says, so they are not branch deciding: a missing or unsure answer does not stop
    the walk and is never a hard gate failure. A missing plan name is reported as a minor soft finding
    (the record is incomplete, the routing is not wrong).
    Skipping an item is excused when the caller had already given a confident answer to it before the
    agent moved on (the repetition check even recommends skipping ahead in that case).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

from callcheck.align import Alignment
from callcheck.judge import NOT_DISCUSSED, UNCLEAR, ItemJudgment, Judgment, captures_value
from callcheck.model import COMPLETE, Finding, FlowSpec, Item, Thread
from callcheck.spec import is_terminal, walk

# Fallback when team_config has no quality_gates.min_confidence_score (the dataset sets 0.7).
MIN_CONFIDENCE = 0.7

ROUTING = "correct_routing"
ANSWERS = "decisive_answers"

_TRUE = {"true", "yes", "y"}
_FALSE = {"false", "no", "n"}


@dataclass
class RoutingResult:
    expected_terminal: str | None = None
    expected_path: list[str] = field(default_factory=list)
    stopped_at: str | None = None
    actual_terminal: str | None = None


def _is_placeholder(scenario_id: str) -> bool:
    return scenario_id.startswith("__") and scenario_id.endswith("__")


def capture_only(item: Item) -> bool:
    scenarios = item.question.scenarios
    return bool(scenarios) and all(_is_placeholder(s.id) for s in scenarios)


def terminal_label(spec: FlowSpec, terminal: str | None) -> str:
    if terminal is None:
        return "no terminal"
    if terminal == COMPLETE:
        return "_complete (handoff to the next agent)"
    outcome = spec.outcomes.get(terminal)
    return f"{terminal} ({outcome.label})" if outcome else terminal


def _quote(ij: ItemJudgment | None) -> str:
    if ij is None or not ij.evidence:
        return "(no caller quote)"
    return f'"{ij.evidence}"'


def _answer_turn(ij: ItemJudgment | None, fallback: int | None = None) -> int | None:
    if ij is None:
        return fallback
    if ij.evidence_turn is not None:
        return ij.evidence_turn
    if ij.first_available_turn is not None:
        return ij.first_available_turn
    return fallback


def _unavailable() -> Finding:
    return Finding(
        check="routing.judge_unavailable",
        severity="major",
        gate=ROUTING,
        turn=None,
        evidence="no judgment for this thread (no cache entry and no working API key)",
        problem=(
            "The caller's answers could not be labelled, so the expected terminal is unknown and routing "
            "cannot be confirmed."
        ),
        fix_hint="Set ANTHROPIC_API_KEY and run without --cached, or restore q3/cache/judgments.json.",
        owner="harness",
        uncertain=True,
    )


class _Router:
    def __init__(self, spec: FlowSpec, alignment: Alignment, judgment: Judgment, min_confidence: float):
        self.spec = spec
        self.alignment = alignment
        self.items = judgment.items
        self.min_confidence = min_confidence
        # Distinct items the agent asked, first occurrence, up to the terminal turn.
        self.ask_turn: dict[str, int] = {}
        for turn, item in alignment.asked:
            if alignment.terminal_turn is not None and turn > alignment.terminal_turn:
                continue
            self.ask_turn.setdefault(item, turn)
        self.asked = list(self.ask_turn)
        self.findings: list[Finding] = []

    # -- helpers -----------------------------------------------------------------

    def low(self, item_id: str) -> bool:
        ij = self.items.get(item_id)
        return ij is None or ij.answer == UNCLEAR or ij.confidence < self.min_confidence

    def answers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item_id, item in self.spec.items.items():
            ij = self.items.get(item_id)
            answer = ij.answer if ij else NOT_DISCUSSED
            if capture_only(item):
                # The route does not depend on the value: take the placeholder branch.
                out[item_id] = item.question.scenarios[0].id
            elif answer != NOT_DISCUSSED:
                out[item_id] = answer
        return out

    def agent_next(self, item_id: str) -> str | None:
        """What the agent did after asking item_id: next distinct item, its terminal, or None."""
        pos = self.asked.index(item_id)
        if pos + 1 < len(self.asked):
            return self.asked[pos + 1]
        return self.alignment.actual_terminal

    def excused(self, item_id: str, moved_on_turn: int | None) -> bool:
        ij = self.items.get(item_id)
        if ij is None or ij.answer in (NOT_DISCUSSED, UNCLEAR) or ij.confidence < self.min_confidence:
            return False
        if ij.first_available_turn is None or moved_on_turn is None:
            return False
        return ij.first_available_turn <= moved_on_turn

    # -- findings ------------------------------------------------------------------

    def low_confidence(self, path_items: list[str]) -> None:
        for item_id in path_items:
            if capture_only(self.spec.items[item_id]) or not self.low(item_id):
                continue
            ij = self.items.get(item_id)
            turn = _answer_turn(ij, self.ask_turn.get(item_id))
            answer = ij.answer if ij else "missing"
            conf = ij.confidence if ij else 0.0
            self.findings.append(
                Finding(
                    check="routing.low_confidence_answer",
                    severity="major",
                    gate=ANSWERS,
                    turn=turn,
                    evidence=_quote(ij),
                    problem=(
                        f"The branch deciding answer to {item_id} is {answer} with confidence {conf:.2f} "
                        f"(threshold {self.min_confidence:.2f}): the route cannot be confirmed."
                    ),
                    fix_hint=(
                        f"Human review: confirm what the caller meant at turn {turn}; if the caller was "
                        "genuinely ambiguous the agent should have asked a clarifying question."
                    ),
                    owner="harness",
                    uncertain=True,
                )
            )

    def skipped(self, item_id: str, after: str, next_item: str, uncertain: bool) -> None:
        turn = self.ask_turn.get(next_item, self.alignment.terminal_turn)
        ij = self.items.get(item_id)
        status = "the caller had not answered it" if ij is None or ij.answer == NOT_DISCUSSED else (
            f"the caller's answer so far was {ij.answer} ({_quote(ij)})"
        )
        self.findings.append(
            Finding(
                check="routing.path_divergence",
                severity="major",
                gate=ROUTING,
                turn=turn,
                evidence=f"after {after} the agent went to {next_item} at turn {turn}; {status}",
                problem=f"The agent skipped {item_id}, which is on the expected path, and {status}.",
                fix_hint=(
                    f"Engine: do not advance past {item_id} until its question is answered; the item pointer "
                    f"moved from {after} to {next_item}."
                ),
                owner="engine",
                uncertain=uncertain,
            )
        )

    def wrong_branch(self, item_id: str, expected_next: str | None, actual_next: str, uncertain: bool) -> None:
        ij = self.items.get(item_id)
        turn = _answer_turn(ij, self.ask_turn.get(item_id))
        self.findings.append(
            Finding(
                check="routing.path_divergence",
                severity="major",
                gate=ROUTING,
                turn=turn,
                evidence=f"caller at turn {turn}: {_quote(ij)}; agent then asked {actual_next}",
                problem=(
                    f"After {item_id} the caller's answer ({ij.answer if ij else 'missing'}) leads to "
                    f"{expected_next or 'an unknown node'}, but the agent asked {actual_next}, which is not "
                    "on the expected path."
                ),
                fix_hint=(
                    f"Prompt: scenario {ij.answer if ij else '?'} of {item_id} should have fired and routed to "
                    f"{expected_next}; tighten the scenario descriptions or add an example for this phrasing."
                ),
                owner="prompt",
                uncertain=uncertain,
            )
        )

    def compare_paths(self, path_items: list[str], expected_terminal: str | None) -> str | None:
        """Walk the expected path alongside the agent's asks. Returns the deciding item for a terminal
        mismatch, if the agent left the path by delivering a terminal."""
        if not path_items or path_items[0] not in self.ask_turn:
            return None
        i = 0
        while i < len(path_items):
            item_id = path_items[i]
            last = i == len(path_items) - 1
            if last and expected_terminal is None:
                return None  # stopped item: missing answer handled elsewhere
            if item_id not in self.ask_turn:
                return None
            expected_next = expected_terminal if last else path_items[i + 1]
            actual_next = self.agent_next(item_id)
            if actual_next is None:
                return None  # the call ended here; termination reports it
            if actual_next == expected_next:
                i += 1
                continue
            uncertain = any(self.low(p) for p in path_items[: i + 1] if not capture_only(self.spec.items[p]))
            if is_terminal(self.spec, actual_next):
                if actual_next == expected_terminal:
                    for s in path_items[i + 1:]:
                        if not self.excused(s, self.alignment.terminal_turn):
                            self.skipped(s, item_id, actual_next, uncertain)
                    return None
                return item_id
            if actual_next in path_items[i + 1:]:
                j = path_items.index(actual_next)
                for s in path_items[i + 1: j]:
                    if not self.excused(s, self.ask_turn.get(actual_next)):
                        self.skipped(s, item_id, actual_next, uncertain)
                i = j
                continue
            if expected_terminal is None and expected_next == path_items[-1]:
                # The next expected item has no answer yet, so we cannot tell a skip from a wrong branch;
                # the agent did not ask it, which is a skip either way.
                self.skipped(expected_next, item_id, actual_next, uncertain)
            else:
                self.wrong_branch(item_id, expected_next, actual_next, uncertain)
            return None
        return None

    def wrong_terminal(self, path_items: list[str], expected: str, actual: str, deciding: str | None) -> None:
        if deciding is None:
            asked_on_path = [p for p in path_items if p in self.ask_turn]
            deciding = asked_on_path[-1] if asked_on_path else path_items[-1]
        ij = self.items.get(deciding)
        turn = _answer_turn(ij, self.ask_turn.get(deciding))
        scenario = ij.answer if ij else "?"
        uncertain = any(self.low(p) for p in path_items if not capture_only(self.spec.items[p]))
        self.findings.append(
            Finding(
                check="routing.wrong_terminal",
                severity="critical",
                gate=ROUTING,
                turn=turn,
                evidence=f"caller at turn {turn} on {deciding}: {_quote(ij)}",
                problem=(
                    f"Expected outcome {terminal_label(self.spec, expected)} but the agent delivered "
                    f"{terminal_label(self.spec, actual)}."
                ),
                fix_hint=(
                    f"Prompt: scenario {scenario} of item {deciding} should have fired, leading to {expected}; "
                    "check how the agent classified this answer and add it as an example to the scenario."
                ),
                owner="prompt",
                uncertain=uncertain,
            )
        )

    def terminal_without_answer(self, stopped: str, actual: str) -> None:
        ij = self.items.get(stopped)
        self.findings.append(
            Finding(
                check="routing.terminal_without_answer",
                severity="critical",
                gate=ROUTING,
                turn=self.alignment.terminal_turn,
                evidence=(
                    f"{stopped} was {'asked at turn ' + str(self.ask_turn[stopped]) if stopped in self.ask_turn else 'never asked'}; "
                    f"agent delivered {actual} at turn {self.alignment.terminal_turn}"
                ),
                problem=(
                    f"The agent routed to {terminal_label(self.spec, actual)} without the decisive answer to "
                    f"{stopped}: the caller never gave it."
                ),
                fix_hint=(
                    f"Engine/prompt: {stopped} must be answered before any disposition; ask its scripted "
                    "question (or a clarification) instead of closing."
                ),
                owner="engine",
                uncertain=ij is not None and ij.confidence < self.min_confidence,
            )
        )

    def plan_capture(self, path_items: list[str]) -> None:
        for item_id in path_items:
            item = self.spec.items[item_id]
            if not capture_only(item):
                continue
            ij = self.items.get(item_id)
            if ij is not None and ij.value and ij.confidence >= self.min_confidence:
                continue
            asked = item_id in self.ask_turn
            self.findings.append(
                Finding(
                    check="routing.plan_not_captured",
                    severity="minor",
                    gate=None,
                    turn=self.ask_turn.get(item_id),
                    evidence=(
                        f"{item_id} {'asked at turn ' + str(self.ask_turn[item_id]) if asked else 'never asked'}; "
                        f"caller value: {ij.value if ij and ij.value else 'none'}"
                    ),
                    problem=(
                        f"The value {item_id} exists to capture was not obtained. Routing is unaffected (every "
                        "branch leads to the same disposition) but the record is incomplete."
                    ),
                    fix_hint=(
                        f"Prompt: ask '{item.question.text}' and wait for the answer before the closing line."
                        if not asked
                        else f"Prompt: after asking {item_id}, wait for the caller's answer before closing."
                    ),
                    owner="prompt",
                )
            )

    def captured(self, captured: dict[str, str]) -> None:
        for item_id, recorded in captured.items():
            item = self.spec.items.get(item_id)
            if item is None:
                continue
            ij = self.items.get(item_id)
            said = _caller_value(item, ij)
            if said is None:
                problem = f"Agent recorded {recorded} for {item_id}, but the caller never gave this information."
                turn = None
            else:
                if _matches(item, recorded, ij):
                    continue
                turn = _answer_turn(ij)
                problem = f"For {item_id}: agent recorded {recorded}, caller said {said} at turn {turn}."
            self.findings.append(
                Finding(
                    check="routing.captured_mismatch",
                    severity="critical",
                    gate=ANSWERS,
                    turn=turn,
                    evidence=f"captured {item_id}={recorded!r}; caller: {_quote(ij)}",
                    problem=problem,
                    fix_hint=(
                        f"Prompt/engine: the value written for {item_id} must come from the caller's words; "
                        "check the extraction step that fills captured fields."
                    ),
                    owner="prompt",
                    uncertain=ij is None or ij.answer == UNCLEAR or ij.confidence < self.min_confidence,
                )
            )


def _norm(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("other:"):
        text = text[len("other:"):]
    return " ".join(text.replace("_", " ").split())


def _caller_value(item: Item, ij: ItemJudgment | None) -> str | None:
    if ij is None or ij.answer == NOT_DISCUSSED:
        return None
    if captures_value(item) and ij.value:
        return ij.value
    return ij.answer


def _to_scenario(item: Item, value: str) -> str | None:
    q = item.question
    v = _norm(value)
    for s in q.scenarios:
        if _norm(s.id) == v:
            return s.id
    if q.options and len(q.options) == len(q.scenarios):
        for opt, s in zip(q.options, q.scenarios):
            if _norm(opt) == v:
                return s.id
    if q.type == "boolean":
        if v in _TRUE:
            return "true"
        if v in _FALSE:
            return "false"
    return None


def _matches(item: Item, recorded: str, ij: ItemJudgment | None) -> bool:
    if ij is None:
        return False
    if captures_value(item) and ij.value:
        if _norm(recorded) == _norm(ij.value):
            return True
        if capture_only(item):
            return False
        if _to_scenario(item, recorded) == ij.answer:
            return True
        return difflib.SequenceMatcher(None, _norm(recorded), _norm(ij.value)).ratio() >= 0.6
    return _to_scenario(item, recorded) == ij.answer


def check_routing(
    spec: FlowSpec,
    thread: Thread,
    alignment: Alignment,
    judgment: Judgment | None,
    captured: dict[str, str] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[list[Finding], RoutingResult]:
    result = RoutingResult(actual_terminal=alignment.actual_terminal)
    if judgment is None:
        return [_unavailable()], result

    router = _Router(spec, alignment, judgment, min_confidence)
    path = walk(spec, router.answers())
    result.expected_path = list(path.items)
    result.expected_terminal = path.terminal
    result.stopped_at = path.stopped_at
    path_items = [p for p in path.items if p in spec.items]

    router.low_confidence(path_items)
    deciding = router.compare_paths(path_items, path.terminal)

    actual = alignment.actual_terminal
    if path.terminal is not None and actual is not None and actual != path.terminal:
        router.wrong_terminal(path_items, path.terminal, actual, deciding)
    if path.terminal is None and path.stopped_at in spec.items and actual is not None:
        ij = judgment.items.get(path.stopped_at)
        if ij is None or ij.answer == NOT_DISCUSSED:
            router.terminal_without_answer(path.stopped_at, actual)
        # unclear: already reported as low confidence (uncertain), the agent should have clarified

    router.plan_capture(path_items)
    if captured:
        router.captured(captured)
    return router.findings, result
