"""engine_config as a graph: build, successors, walk, reachable terminals."""

from __future__ import annotations

from dataclasses import dataclass, field

from callcheck.model import COMPLETE, FlowSpec, Item, Outcome, Question, Scenario

UNCLEAR = "unclear"


def resolve_goto(goto: str | None) -> str | None:
    """'item_q2_active:question_a' -> 'q2_active'; '_complete' stays as is."""
    if goto is None:
        return None
    if goto == COMPLETE:
        return COMPLETE
    head = goto.split(":", 1)[0]
    return head.removeprefix("item_")


def _scenario(raw: dict) -> Scenario:
    return Scenario(
        id=raw["id"],
        means=raw.get("means", ""),
        goto=raw.get("goto"),
        goto_item=resolve_goto(raw.get("goto")),
        disposition=raw.get("disposition"),
        say=raw.get("say"),
        do=raw.get("do"),
    )


def _question(raw: dict) -> Question:
    return Question(
        id=raw["id"],
        text=raw["text"],
        type=raw["type"],
        scenarios=[_scenario(s) for s in raw.get("scenarios", [])],
        options=raw.get("options"),
        default_route=raw.get("default_route"),
    )


def build_spec(engine_config: dict) -> FlowSpec:
    items = {
        raw["id"]: Item(
            id=raw["id"],
            label=raw.get("label", raw["id"]),
            questions=[_question(q) for q in raw["questions"]],
        )
        for raw in engine_config["items"]
    }
    outcomes = {
        raw["id"]: Outcome(
            id=raw["id"],
            say=raw.get("say", ""),
            label=raw.get("label", raw["id"]),
            next_action=raw["next_action"],
            flags=dict(raw.get("flags") or {}),
        )
        for raw in engine_config["outcomes"]
    }
    entry = engine_config["items"][0]["id"]
    return FlowSpec(items=items, outcomes=outcomes, entry=entry)


def scenario_target(question: Question, scenario: Scenario) -> str | None:
    """Where a scenario leads: outcome id, item id or '_complete'.

    Disposition wins over goto; a scenario with neither falls back to the
    question's default_route (resolved).
    """
    if scenario.disposition:
        return scenario.disposition
    if scenario.goto_item:
        return scenario.goto_item
    return resolve_goto(question.default_route)


def _pick_scenario(question: Question, answer: str) -> Scenario | None:
    """Map an answer to a scenario.

    Exact scenario id first. Otherwise, a placeholder scenario (id wrapped in
    double underscores, e.g. q3_plan '__close__') catches any answer, since it
    exists only so the value is captured before the call closes.
    """
    exact = question.scenario(answer)
    if exact is not None:
        return exact
    for s in question.scenarios:
        if s.id.startswith("__") and s.id.endswith("__"):
            return s
    return None


def successors(spec: FlowSpec, item_id: str) -> set[str]:
    question = spec.items[item_id].question
    targets = {scenario_target(question, s) for s in question.scenarios}
    if question.default_route:
        targets.add(resolve_goto(question.default_route))
    targets.discard(None)
    return targets  # type: ignore[return-value]


def is_terminal(spec: FlowSpec, node: str) -> bool:
    return node == COMPLETE or node in spec.outcomes


@dataclass
class Path:
    items: list[str] = field(default_factory=list)
    terminal: str | None = None  # outcome id or "_complete"
    stopped_at: str | None = None  # item where the walk could not continue


def walk(spec: FlowSpec, answers: dict[str, str]) -> Path:
    """Follow the graph from the entry item using answers keyed by item id."""
    path = Path()
    node = spec.entry
    seen: set[str] = set()
    while True:
        if is_terminal(spec, node):
            path.terminal = node
            return path
        if node not in spec.items or node in seen:
            path.stopped_at = node
            return path
        seen.add(node)
        path.items.append(node)
        answer = answers.get(node)
        if answer is None or answer == UNCLEAR:
            path.stopped_at = node
            return path
        question = spec.items[node].question
        scenario = _pick_scenario(question, answer)
        if scenario is None:
            target = resolve_goto(question.default_route)
        else:
            target = scenario_target(question, scenario)
        if target is None:
            path.stopped_at = node
            return path
        node = target


def reachable_terminals(spec: FlowSpec) -> set[str]:
    found: set[str] = set()
    stack = [spec.entry]
    seen: set[str] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if is_terminal(spec, node):
            found.add(node)
        elif node in spec.items:
            stack.extend(successors(spec, node))
    return found
