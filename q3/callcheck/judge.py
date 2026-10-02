"""Model judge: map what the CALLER said to each item's scenarios.

Why a model at all
    Everything that can be decided by code is decided by code (graph walk, script
    alignment, hygiene, termination). The one thing code cannot do reliably is
    language understanding of free caller speech: "I just used it for an
    appointment last week" means active coverage, "I moved out of the county" in a
    decline reason means out of county, "I have the yellow packet right here, let's
    do it over the phone" answers two items the agent has not asked yet. The judge
    does only that, one call per thread, and returns structured labels that
    routing code consumes.

Prompt design decisions
    1. Caller statements only. The agent's own words are never evidence. The thing
       we are evaluating is the agent, so letting the judge read the agent's
       assumptions ("it sounds like there's nothing you need to do") would let a
       wrong agent grade itself as right.
    2. Every item, every thread, independent of the path the agent took. The judge
       labels all 8 items even if the agent never asked them. Routing code then
       walks the graph with these labels and decides which ones matter. If the
       judge only labelled asked items, an agent that skipped a branch deciding
       question could never be caught.
    3. first_available_turn separate from evidence_turn. Callers volunteer answers
       early (threads 03, 07, 08, 09, 10). The earliest caller turn that already
       held the information powers the "re-asked a question the caller already
       answered" soft check; the evidence turn is just the clearest quote.
    4. "unclear" and "not_discussed" are explicit enum values, not missing data.
       Hedges and contradictions become "unclear" with capped confidence so the
       verdict can say NEEDS_REVIEW instead of guessing. A scenario that itself
       covers uncertainty (q2 "inactive" means "not active, or unsure") wins over
       "unclear", because the script already decided where unsure callers go.
    5. One utterance, one item, unless the words address both. A plain "yes" to
       the opening offer answers q1 only; it is not read as a phone preference for
       q5_choice. This stops the model from double counting generic agreement.
    6. Calibrated confidence bands are spelled out (0.9+ only for explicit,
       unambiguous statements) so the verdict can threshold on them.
    7. Placeholder scenarios (ids like "__close__") get a rendered meaning ("caller
       named the plan") instead of the engine's internal note, and the captured
       plan goes into `value`, normalized to one of the scripted options.
    8. Structure: system prompt holds the rules; the user turn holds the items
       (question text, scenario id, meaning) and the transcript as numbered
       `[turn] ROLE: text` lines. In the tool schema `evidence` comes before
       `answer`, so the model commits to a verbatim quote before picking a label.

API notes (logged in docs/decisions-log.md)
    claude-sonnet-5-5 rejects forced tool_choice ("any"/"tool") and non default
    temperature with a 400. So the call uses tool_choice auto plus `strict: true`
    on the tool (schema valid arguments guaranteed) plus an explicit instruction,
    and checks that a tool_use block came back (one retry otherwise). Determinism
    comes from the committed cache, not from temperature.

Validation
    The tool input is validated in code: unknown item ids are dropped, missing
    items or answers outside the enum become "unclear" with confidence 0 and a
    note, turns that are not caller turns become None, confidence is clamped.
    Nothing here raises on malformed model output.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from callcheck.model import FlowSpec, Item, Scenario, Thread

PROMPT_VERSION = "v1"
DEFAULT_MODEL = "claude-sonnet-5-5"
TOOL_NAME = "record_judgments"
UNCLEAR = "unclear"
NOT_DISCUSSED = "not_discussed"
CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "judgments.json"

Source = Literal["cache", "live", "fake"]


@dataclass(frozen=True)
class ItemJudgment:
    item_id: str
    answer: str  # a scenario id of that item's question, or "unclear", or "not_discussed"
    confidence: float  # 0..1
    evidence: str  # verbatim caller quote supporting the answer ("" if not_discussed)
    evidence_turn: int | None
    first_available_turn: int | None  # earliest caller turn where this information was given
    value: str | None  # q3_plan plan name, decline reason text


@dataclass(frozen=True)
class Judgment:
    thread_id: str
    items: dict[str, ItemJudgment]
    model: str
    prompt_version: str
    source: Source


class Judge(Protocol):
    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None: ...


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You label what a CALLER said on a phone call between a Medi-Cal renewal assistant (AGENT) and a member (CALLER). \
Your labels feed a deterministic evaluator that checks whether the agent routed the call correctly, so it needs to know \
what the caller actually communicated, independent of what the agent did or assumed.

For every item you are given, decide which scenario the caller's own words match, quote the caller, and find the \
earliest caller turn where that information appeared.

Rules:
1. Judge CALLER lines only. The agent's questions, assumptions, summaries and closing lines are never evidence. If the \
agent says "it sounds like your coverage is active" and the caller never said so, the caller did not say so.
2. Label every item, including items the agent never asked about. Callers often volunteer information before it is \
asked (for example mentioning a move out of the county in their first reply). Do not depend on the order of the \
agent's questions or on whether they were asked at all. Map the caller's words to the scenario meanings literally.
3. One statement answers an item only if its words address that item. A plain yes to the opening offer is the answer \
to the intent item only; it is not evidence for a later item unless the caller adds specific content about it.
4. first_available_turn: the earliest CALLER turn that already contained the information, even if it came before the \
agent asked. evidence_turn: the CALLER turn of the quote you give (usually the clearest statement). Use the bracketed \
turn numbers of CALLER lines only.
5. evidence: a short verbatim quote copied from a CALLER line, no paraphrase.
6. not_discussed: the caller never gave this information. Use evidence "", both turns null, value null.
7. unclear: the caller addressed the item but their words cannot be mapped to exactly one scenario (they hedge, \
contradict themselves without settling, or give an answer no scenario covers). Quote the conflicting words in \
evidence, cite the latest relevant turn, keep confidence at or below 0.6. If a scenario's meaning explicitly covers \
uncertainty (for example "not active, or unsure"), a hedged answer maps to that scenario, not to unclear. If the \
caller clearly corrects an earlier answer, use the corrected one.
8. confidence: 0.9 or above only for an explicit, unambiguous statement; 0.7 to 0.89 when the meaning is clear but \
indirect or implied; 0.4 to 0.69 for weak or partial evidence; below 0.4 when guessing. For not_discussed it is how \
sure you are the caller never gave the information.
9. value: null unless the item says it captures a value. The plan item captures the plan name exactly as one of its \
listed options, or "Other: <name>" if the caller named a plan not in the list. The decline item captures the caller's \
stated reason, close to their words.
10. The decline item asks why the caller does not want help. Label it only from a reason the caller gives for \
declining or for not needing help after declining; if the caller never declined, it is not_discussed.

Call the record_judgments tool exactly once with every item. Do not answer in prose."""


def _is_placeholder(s: Scenario) -> bool:
    return s.id.startswith("__") and s.id.endswith("__")


def _scenario_meaning(item: Item, s: Scenario) -> str:
    if _is_placeholder(s):
        return "Caller NAMED the specific plan. Put the plan name in value."
    return s.means


def captures_value(item: Item) -> bool:
    """Free text questions (decline reason) and placeholder branches (q3_plan) exist to capture a value."""
    q = item.question
    return q.type == "text" or any(_is_placeholder(s) for s in q.scenarios)


def answer_enum(item: Item) -> list[str]:
    return [s.id for s in item.question.scenarios] + [UNCLEAR, NOT_DISCUSSED]


def render_items(spec: FlowSpec) -> str:
    blocks = []
    for item in spec.items.values():
        q = item.question
        lines = [f"ITEM {item.id} ({item.label})", f'  Question: "{q.text}"']
        if q.options and captures_value(item):
            lines.append(f"  Options: {', '.join(q.options)}")
        if captures_value(item):
            lines.append("  Captures a value: yes")
        lines.append("  Scenarios:")
        for s in q.scenarios:
            lines.append(f"    - {s.id}: {_scenario_meaning(item, s)}")
        lines.append(f"    - {UNCLEAR}: addressed but not mappable to exactly one scenario.")
        lines.append(f"    - {NOT_DISCUSSED}: the caller never gave this information.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_transcript(thread: Thread) -> str:
    return "\n".join(f"[{m.turn}] {m.role.upper()}: {m.text}" for m in thread.messages)


def build_user_message(spec: FlowSpec, thread: Thread) -> str:
    return (
        "<items>\n" + render_items(spec) + "\n</items>\n\n"
        "<transcript>\n" + render_transcript(thread) + "\n</transcript>\n\n"
        f"Label all {len(spec.items)} items from the CALLER lines with the {TOOL_NAME} tool."
    )


def _nullable(t: str) -> dict:
    return {"anyOf": [{"type": t}, {"type": "null"}]}


def build_tool(spec: FlowSpec) -> dict:
    item_props = {}
    for item in spec.items.values():
        item_props[item.id] = {
            "type": "object",
            "description": f"Judgment for item {item.id}.",
            "properties": {
                "evidence": {"type": "string", "description": "Verbatim CALLER quote, or empty if not_discussed."},
                "answer": {"type": "string", "enum": answer_enum(item)},
                "confidence": {"type": "number", "description": "0 to 1, calibrated."},
                "evidence_turn": _nullable("integer"),
                "first_available_turn": _nullable("integer"),
                "value": _nullable("string"),
            },
            "required": ["evidence", "answer", "confidence", "evidence_turn", "first_available_turn", "value"],
            "additionalProperties": False,
        }
    return {
        "name": TOOL_NAME,
        "description": "Record one judgment per item about what the CALLER said.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": list(spec.items),
                    "additionalProperties": False,
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _unclear(item_id: str, note: str) -> ItemJudgment:
    return ItemJudgment(item_id, UNCLEAR, 0.0, f"[harness] {note}", None, None, None)


def _turn(v: Any, caller_turns: set[int]) -> int | None:
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v if v in caller_turns else None


def _normalize_plan(value: str, options: list[str]) -> str:
    for opt in options:
        if value.strip().lower() == opt.lower():
            return opt
    return value.strip()


def _one(spec: FlowSpec, item: Item, raw: Any, caller_turns: set[int]) -> ItemJudgment:
    if not isinstance(raw, dict):
        return _unclear(item.id, "item missing from model output")
    answer = raw.get("answer")
    if answer not in answer_enum(item):
        return _unclear(item.id, f"model answer {answer!r} is not a scenario of this item")
    conf = raw.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0
    confidence = min(1.0, max(0.0, confidence))
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), str) else ""
    ev_turn = _turn(raw.get("evidence_turn"), caller_turns)
    first = _turn(raw.get("first_available_turn"), caller_turns)
    value = raw.get("value") if isinstance(raw.get("value"), str) and raw.get("value").strip() else None
    if answer == NOT_DISCUSSED:
        return ItemJudgment(item.id, answer, confidence, "", None, None, None)
    if first is None:
        first = ev_turn
    elif ev_turn is not None:
        first = min(first, ev_turn)
    if not captures_value(item):
        value = None
    elif value and item.question.options:
        value = _normalize_plan(value, item.question.options)
    return ItemJudgment(item.id, answer, confidence, evidence, ev_turn, first, value)


def parse_tool_input(spec: FlowSpec, thread: Thread, raw: Any) -> dict[str, ItemJudgment]:
    """Validate the model's tool input. Never raises."""
    caller_turns = {m.turn for m in thread.caller_messages()}
    items_raw = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items_raw, dict):
        items_raw = {}
    return {iid: _one(spec, item, items_raw.get(iid), caller_turns) for iid, item in spec.items.items()}


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    print(f"callcheck judge: {msg}", file=sys.stderr)


class ClaudeJudge:
    """Live judge. One messages.create call per thread."""

    def __init__(self, model: str | None = None, client: Any = None, max_attempts: int = 2):
        self.model = model or os.environ.get("CALLCHECK_MODEL", DEFAULT_MODEL)
        self._client = client
        self.max_attempts = max_attempts
        self._disabled = False

    @property
    def available(self) -> bool:
        if self._disabled:
            return False
        return self._client is not None or bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None:
        if not self.available:
            return None
        client = self._get_client()
        tool = build_tool(spec)
        messages = [{"role": "user", "content": build_user_message(spec, thread)}]
        import anthropic

        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = client.messages.create(
                    model=self.model,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=[tool],
                    tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                    messages=messages,
                )
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                # Bad credentials will fail for every thread: stop trying, report unavailable.
                self._disabled = True
                _warn(f"live judge disabled, credentials rejected ({type(e).__name__})")
                return None
            except anthropic.NotFoundError:
                self._disabled = True
                _warn(f"live judge disabled, model {self.model!r} not found; set CALLCHECK_MODEL")
                return None
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
                # The SDK already retried 429/5xx/connection errors; give up on this thread only.
                _warn(f"{thread.thread_id}: API error {type(e).__name__}: {str(e)[:200]}")
                return None
            if resp.stop_reason == "refusal":
                _warn(f"{thread.thread_id}: model refused ({getattr(resp, 'stop_details', None)})")
                return None
            block = next((b for b in resp.content if b.type == "tool_use" and b.name == TOOL_NAME), None)
            if block is not None:
                items = parse_tool_input(spec, thread, block.input)
                return Judgment(thread.thread_id, items, self.model, PROMPT_VERSION, "live")
            _warn(f"{thread.thread_id}: no tool call on attempt {attempt} (stop_reason={resp.stop_reason})")
        return None


class FakeJudge:
    """Test judge: preset item judgments per thread. Missing items are not_discussed."""

    def __init__(self, presets: dict[str, dict[str, ItemJudgment]], model: str = "fake"):
        self.presets = presets
        self.model = model

    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None:
        preset = self.presets.get(thread.thread_id)
        if preset is None:
            return None
        items = {
            iid: preset.get(iid) or ItemJudgment(iid, NOT_DISCUSSED, 1.0, "", None, None, None)
            for iid in spec.items
        }
        return Judgment(thread.thread_id, items, self.model, PROMPT_VERSION, "fake")


def _canonical_spec_items(spec: FlowSpec) -> str:
    data = [
        {
            "id": item.id,
            "question": item.question.text,
            "options": item.question.options,
            "scenarios": [[s.id, s.means] for s in item.question.scenarios],
        }
        for item in spec.items.values()
    ]
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def cache_key(model: str, spec: FlowSpec, thread: Thread) -> str:
    payload = "\n\x1e".join([model, PROMPT_VERSION, render_transcript(thread), _canonical_spec_items(spec)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedJudge:
    """Cache in front of a judge. Hit: source 'cache'. Miss: call inner if available, else None."""

    def __init__(self, inner: Judge | None, *, model: str, path: Path | str = CACHE_PATH, refresh: bool = False):
        self.inner = inner
        self.model = model
        self.path = Path(path)
        self.refresh = refresh

    def _load(self) -> dict:
        if not self.path.exists():
            return {"entries": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                return data
        except json.JSONDecodeError:
            _warn(f"cache file {self.path} unreadable, ignoring it")
        return {"entries": {}}

    def _store(self, key: str, j: Judgment) -> None:
        data = self._load()
        data["entries"][key] = {
            "thread_id": j.thread_id,
            "model_requested": self.model,
            "model": j.model,
            "prompt_version": j.prompt_version,
            "items": {iid: asdict(ij) for iid, ij in j.items.items()},
        }
        data["entries"] = dict(sorted(data["entries"].items(), key=lambda kv: (kv[1]["thread_id"], kv[0])))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n", encoding="utf-8")

    def _live_allowed(self) -> bool:
        if self.inner is None:
            return False
        return getattr(self.inner, "available", True)

    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None:
        key = cache_key(self.model, spec, thread)
        if not self.refresh:
            entry = self._load()["entries"].get(key)
            if entry is not None:
                items = {iid: ItemJudgment(**ij) for iid, ij in entry["items"].items()}
                return Judgment(thread.thread_id, items, entry["model"], entry["prompt_version"], "cache")
        if not self._live_allowed():
            return None
        j = self.inner.judge(spec, thread)  # type: ignore[union-attr]
        if j is not None:
            self._store(key, j)
        return j


def default_judge(live: bool = False) -> Judge:
    """Cache first. Live calls happen on a cache miss when a key is set, or always when live=True."""
    model = os.environ.get("CALLCHECK_MODEL", DEFAULT_MODEL)
    inner = ClaudeJudge(model) if os.environ.get("ANTHROPIC_API_KEY") else None
    return CachedJudge(inner, model=model, refresh=live)


__all__ = [
    "PROMPT_VERSION", "ItemJudgment", "Judgment", "Judge", "ClaudeJudge", "CachedJudge", "FakeJudge",
    "default_judge", "cache_key", "build_tool", "answer_enum", "parse_tool_input", "render_transcript",
]



def _main(argv: list[str] | None = None) -> int:
    """Populate or refresh the cache: `uv run python -m callcheck.judge [--refresh] [thread_id ...]`."""
    import argparse

    from callcheck.load import load

    ap = argparse.ArgumentParser(prog="python -m callcheck.judge")
    ap.add_argument("ids", nargs="*", help="thread ids (default: all)")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[1] / "data" / "gym_agent_conversations.json"))
    ap.add_argument("--refresh", action="store_true", help="ignore cache, call the model")
    args = ap.parse_args(argv)
    spec, threads = load(args.data)
    judge = default_judge(live=args.refresh)
    missing = 0
    for t in threads:
        if args.ids and t.thread_id not in args.ids:
            continue
        j = judge.judge(spec, t)
        if j is None:
            missing += 1
            print(f"{t.thread_id}: no judgment (no valid key and no cache entry)")
            continue
        print(f"{t.thread_id} [{j.source}, {j.model}, {j.prompt_version}]")
        for iid, ij in j.items.items():
            val = f" value={ij.value!r}" if ij.value else ""
            print(f"  {iid:13} {ij.answer:22} conf={ij.confidence:.2f} turn={ij.evidence_turn} "
                  f"first={ij.first_available_turn}{val}  {ij.evidence[:90]!r}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(_main())
