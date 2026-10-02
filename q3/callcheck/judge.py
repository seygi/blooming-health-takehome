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
    1. Facts from caller words only. Agent lines are read only to know which
       question a caller line answers (a bare "no" means nothing without it). The
       thing we are evaluating is the agent, so treating the agent's assumptions
       ("it sounds like there's nothing you need to do") as evidence would let a
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
    5. One utterance, one item, unless the words address both. The opener already
       says "over the phone", so agreeing to it answers q1 only; it is not a
       q5_choice answer because the in person option was never offered. A caller
       who ties "over the phone" to completing the packet they have (threads 07
       C3, 10 C0) did state a q5_choice preference, volunteered, at moderate
       confidence. A worked example in the prompt shows both cases.
    5b. Corrections: the corrected answer wins and first_available_turn points at
       the correction, so a superseded answer never makes a later question look
       like a re-ask.
    6. Calibrated confidence bands are spelled out (0.9+ only for explicit,
       unambiguous statements) so the verdict can threshold on them.
    7. Placeholder scenarios (ids like "__close__") get a rendered meaning ("caller
       named the plan") instead of the engine's internal note, and the captured
       plan goes into `value`, normalized to one of the scripted options.
    8. Structure: system prompt holds the rules; the user turn holds the items
       (question text, scenario id, meaning) and the transcript as
       `[C3] CALLER: text` / `[A3] AGENT: text` lines (`[A-open]` for the opener,
       continuation lines indented). Distinct C and A ids make an agent turn
       impossible to cite as caller evidence: the schema's turn fields are an enum
       of this thread's caller ids. In the tool schema `evidence` comes before
       `answer`, so the model commits to a verbatim quote before picking a label.
    9. first_available_quote is a verbatim quote from the first available turn, so
       the re-ask finding can show what the caller said at that turn (the evidence
       quote may come from a later turn).
   10. The cache key hashes the system prompt, the tool schema and the rendered
       user message, so any prompt edit misses the cache even if PROMPT_VERSION
       (kept as a label) was not bumped.

API notes (logged in docs/decisions-log.md)
    claude-sonnet-5-5 rejects forced tool_choice ("any"/"tool") and non default
    temperature with a 400. So the call uses tool_choice auto plus `strict: true`
    on the tool (schema valid arguments guaranteed) plus an explicit instruction,
    and checks that a tool_use block came back (one retry otherwise). Determinism
    comes from the committed cache, not from temperature.

Modes (see cli.py)
    Default `callcheck` is live: every thread is judged by the model and the result is written to the
    cache. `--cached` replays the cache only. A rejected key raises JudgeAuthError and stops the run.

Transports (CALLCHECK_BACKEND)
    "api" (default) calls the Messages API with ANTHROPIC_API_KEY. "claude-cli" runs the Claude Code
    CLI headless (`claude -p`) on the user's subscription, with the same system prompt, the same user
    message and the tool's input_schema as `--json-schema`. The cache key ignores the transport, so
    either one fills entries that a plain offline run finds; the transport is stored as metadata in
    the cache entry and shown in the report header.

Validation
    The tool input is validated in code: unknown item ids are dropped, missing
    items or answers outside the enum become "unclear" with confidence 0 and a
    note, turn ids that are not caller turns of the thread become None with a
    note, a first_available_quote not found in its turn is blanked with a note,
    confidence is clamped. Notes land on Judgment.notes and in the cache.
    Nothing here raises on malformed model output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from callcheck.model import FlowSpec, Item, Scenario, Thread

PROMPT_VERSION = "v2"
DEFAULT_MODEL = "claude-sonnet-5-5"
TOOL_NAME = "record_judgments"
UNCLEAR = "unclear"
NOT_DISCUSSED = "not_discussed"
CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "judgments.json"

Source = Literal["cache", "live", "fake"]
Transport = Literal["api", "claude-cli"]
BACKENDS = ("api", "claude-cli")


@dataclass(frozen=True)
class ItemJudgment:
    item_id: str
    answer: str  # a scenario id of that item's question, or "unclear", or "not_discussed"
    confidence: float  # 0..1
    evidence: str  # verbatim caller quote supporting the answer ("" if not_discussed)
    evidence_turn: int | None
    first_available_turn: int | None  # earliest caller turn where this information was given
    value: str | None  # q3_plan plan name, decline reason text
    # verbatim quote from the caller turn first_available_turn ("" when unknown); last and
    # defaulted so positional constructors and older cache entries keep working
    first_available_quote: str = ""


@dataclass(frozen=True)
class Judgment:
    thread_id: str
    items: dict[str, ItemJudgment]
    model: str
    prompt_version: str
    source: Source
    notes: tuple[str, ...] = ()  # harness notes from validation (dropped turn ids, blanked quotes)
    # how the model was reached ("api" or "claude-cli"); None for fakes and for cache entries older than
    # the field. Metadata only: never part of the cache key.
    transport: Transport | None = None


class JudgeAuthError(Exception):
    """The API rejected the credentials (401 or 403). Every thread would fail the same way, so this
    stops the run instead of degrading each thread to NEEDS_REVIEW."""


class Judge(Protocol):
    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None: ...


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You label what a CALLER said on a phone call between a Medi-Cal renewal assistant (AGENT) and a member (CALLER). \
Your labels feed a deterministic evaluator that checks whether the agent routed the call correctly, so it needs to \
know what the caller actually communicated, independent of what the agent did or assumed.

For every item you are given, decide which scenario the caller's own words match, quote the caller, and find the \
earliest caller turn where that information appeared.

Transcript format: every line starts with a turn id. [A-open] is the agent's opening question. [C<t>] is CALLER \
turn t and [A<t>] is AGENT turn t. Caller turn C<t> is spoken before agent turn A<t>, so the order is A-open, C0, \
A0, C1, A1, C2, and so on: the AGENT line right before C<t> is A<t-1> (or A-open before C0). An indented line \
continues the turn above it. Turn fields in your answer take caller turn ids only, like "C3", never an A id.

Rules:
1. Use AGENT lines only to know which question a CALLER line answers. Facts come only from CALLER words. An agent \
summary or assumption is never evidence: if the agent says "it sounds like your coverage is active" and the caller \
never said so, the caller did not say so.
2. Label every item, including items the agent never asked about. Callers often volunteer information before it is \
asked (for example mentioning a move out of the county in their first reply). Do not depend on the order of the \
agent's questions or on whether they were asked at all. Map by meaning. Implied answers are allowed, with \
confidence per rule 10.
3. A short reply such as "yes", "no" or "sure" answers only the question in the AGENT line right before it. Any \
other item is answered only by caller words whose content addresses that item.
4. Corrections: if the caller corrects an earlier answer, the final answer is the corrected one. evidence quotes \
the correction, and first_available_turn is the turn where the caller first stated the corrected answer; the \
superseded answer does not count as available.
5. q5_choice (phone or in person) and the opener. The opening question already says "over the phone", so agreeing \
to it at C0, even in words like "yes, over the phone is fine", answers q1_intent only and is not a q5_choice \
answer: the in person option had not been offered. q5_choice is answered only when the caller (a) answers the \
q5_choice question, or (b) states how they want to complete the yellow packet they have, for example asking to \
finish or fill out the packet over the phone now, or asking for an in person appointment to go through it. Case (b) \
counts in any turn, before the agent offers the choice and even in the same sentence as other answers, as long as \
the caller ties phone or in person to completing the packet. Label case (b) as a volunteered preference (by_phone \
or in_person) with confidence 0.7 to 0.85, first_available_turn being the first turn that ties the two.
6. first_available_turn: the earliest CALLER turn that already contained the information, even if it came before \
the agent asked. first_available_quote: a short verbatim quote from that same CALLER turn showing the information. \
evidence_turn: the CALLER turn of the evidence quote (usually the clearest statement, which may be later).
7. evidence: a short verbatim quote copied from a CALLER line, no paraphrase.
8. not_discussed: the caller never gave this information. Use evidence "" and first_available_quote "", both turns \
null, value null.
9. unclear: the caller addressed the item but their words cannot be mapped to exactly one scenario (they hedge, \
contradict themselves without settling, or give an answer no scenario covers). Quote the conflicting words in \
evidence, cite the latest relevant turn, keep confidence at or below 0.6. If a scenario's meaning explicitly covers \
uncertainty (for example "not active, or unsure"), a hedged answer maps to that scenario, not to unclear.
10. confidence: 0.9 or above only for an explicit, unambiguous statement; 0.7 to 0.89 when the meaning is clear but \
indirect or implied; 0.4 to 0.69 for weak or partial evidence; below 0.4 when guessing. For not_discussed it is how \
sure you are the caller never gave the information.
11. value: null unless the item says it captures a value. The plan item captures the plan name exactly as one of \
its listed options, or "Other: <name>" if the caller named a plan not in the list. The decline item captures the \
caller's stated reason, close to their words.
12. The decline item asks why the caller does not want help. Label it only from a reason the caller gives for \
declining or for not needing help after declining; if the caller never declined, it is not_discussed.

Worked example (shortened, not from your transcript):
[A-open] AGENT: First, would you like our help renewing your Medi-Cal right now, over the phone?
[C0] CALLER: Sure, over the phone works.
[A0] AGENT: Do you currently have active Medi-Cal coverage?
[C1] CALLER: Yes, I think so.
[A1] AGENT: Do you currently have any other health insurance?
[C2] CALLER: No. Oh wait, I found the letter, my Medi-Cal ended last month. I have the yellow packet here, can we \
fill it out over the phone?
Labels:
q1_intent: yes, evidence "Sure, over the phone works.", evidence_turn C0, first_available_turn C0, confidence 0.95.
q2_active: inactive, evidence "my Medi-Cal ended last month", evidence_turn C2, first_available_turn C2 (the \
correction; the C1 answer is superseded), confidence 0.9.
q3_coverage: false, evidence "No.", evidence_turn C2, first_available_turn C2, confidence 0.9 (short reply to A1).
q5_packet: still_has, evidence "I have the yellow packet here", turn C2, confidence 0.8 (implied: has it, not \
submitted).
q5_choice: by_phone, evidence "can we fill it out over the phone?", evidence_turn C2, first_available_turn C2, \
confidence 0.8, volunteered before the choice was offered. C0 is not a q5_choice answer.

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


CONTINUATION_INDENT = "    "
_CALLER_ID = re.compile(r"^C(\d+)$")


def turn_id(role: str, turn: int) -> str:
    """Transcript label: C<t> for caller turns, A<t> for agent turns, A-open for the opener."""
    if role == "caller":
        return f"C{turn}"
    return "A-open" if turn < 0 else f"A{turn}"


def caller_turn_ids(thread: Thread) -> list[str]:
    return [turn_id("caller", m.turn) for m in thread.caller_messages()]


def render_transcript(thread: Thread) -> str:
    """One labelled line per message; continuation lines indented, blank lines dropped.

    Distinct C and A prefixes keep the model from citing an agent turn as caller
    evidence, and indentation keeps every line attributable to a speaker.
    """
    out = []
    for m in thread.messages:
        lines = [ln.strip() for ln in m.text.splitlines() if ln.strip()] or [""]
        out.append(f"[{turn_id(m.role, m.turn)}] {m.role.upper()}: {lines[0]}")
        out.extend(CONTINUATION_INDENT + ln for ln in lines[1:])
    return "\n".join(out)


def build_user_message(spec: FlowSpec, thread: Thread) -> str:
    return (
        "<items>\n" + render_items(spec) + "\n</items>\n\n"
        "<transcript>\n" + render_transcript(thread) + "\n</transcript>\n\n"
        f"Label all {len(spec.items)} items from the CALLER lines with the {TOOL_NAME} tool."
    )


def _nullable(t: str) -> dict:
    return {"anyOf": [{"type": t}, {"type": "null"}]}


def _turn_schema(thread: Thread, description: str) -> dict:
    # Strict tool use documents enum and anyOf but not pattern, so the caller turn ids of
    # this thread are an enum (each matches ^C\d+$). Agent ids cannot be emitted at all.
    ids = caller_turn_ids(thread)
    if not ids:
        return {"type": "null"}
    return {"anyOf": [{"type": "string", "enum": ids}, {"type": "null"}], "description": description}


def build_tool(spec: FlowSpec, thread: Thread) -> dict:
    item_props = {}
    for item in spec.items.values():
        item_props[item.id] = {
            "type": "object",
            "description": f"Judgment for item {item.id}.",
            "properties": {
                "evidence": {"type": "string", "description": "Verbatim CALLER quote, or empty if not_discussed."},
                "answer": {"type": "string", "enum": answer_enum(item)},
                "confidence": {"type": "number", "description": "0 to 1, calibrated."},
                "evidence_turn": _turn_schema(thread, "Caller turn id of the evidence quote, like C3."),
                "first_available_turn": _turn_schema(thread, "Earliest caller turn id holding this information."),
                "first_available_quote": {
                    "type": "string",
                    "description": "Verbatim quote from the first_available_turn CALLER line, or empty if not_discussed.",
                },
                "value": _nullable("string"),
            },
            "required": ["evidence", "answer", "confidence", "evidence_turn", "first_available_turn",
                         "first_available_quote", "value"],
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


def _turn(v: Any, caller_turns: set[int], where: str, notes: list[str]) -> int | None:
    """Map a caller turn id ("C3") back to the int caller turn. Anything else is None with a note."""
    if v is None:
        return None
    m = _CALLER_ID.match(v) if isinstance(v, str) else None
    if m is None:
        kind = "an agent turn id" if isinstance(v, str) and v.startswith("A") else "not a caller turn id"
        notes.append(f"{where} {v!r} is {kind}, set to null")
        return None
    t = int(m.group(1))
    if t not in caller_turns:
        notes.append(f"{where} {v!r} is not a caller turn of this thread, set to null")
        return None
    return t


_QUOTE_FOLD = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-"})


def _fold(text: str) -> str:
    """Case, whitespace and typographic quote insensitive form, for substring checks only."""
    return " ".join(text.translate(_QUOTE_FOLD).lower().split())


def _quote_in(quote: str, turn_text: str) -> bool:
    q = _fold(quote).strip(" .,!?\"'")
    return bool(q) and q in _fold(turn_text)


def _first_quote(raw_quote: Any, evidence: str, ev_turn: int | None, first: int | None,
                 texts: dict[int, str], where: str, notes: list[str]) -> str:
    """The model's quote if it is in caller turn `first`; else the evidence quote if that is the same
    turn; else "". Never a quote from a different turn."""
    if first is None:
        return ""
    quote = raw_quote.strip() if isinstance(raw_quote, str) else ""
    if quote and _quote_in(quote, texts[first]):
        return quote
    if quote:
        notes.append(f"{where} {quote[:60]!r} is not in caller turn C{first}, blanked")
    if ev_turn == first and evidence and _quote_in(evidence, texts[first]):
        return evidence
    return ""


def _normalize_plan(value: str, options: list[str]) -> str:
    for opt in options:
        if value.strip().lower() == opt.lower():
            return opt
    return value.strip()


def _one(spec: FlowSpec, item: Item, raw: Any, texts: dict[int, str], notes: list[str]) -> ItemJudgment:
    if not isinstance(raw, dict):
        return _unclear(item.id, "item missing from model output")
    answer = raw.get("answer")
    if answer not in answer_enum(item):
        return _unclear(item.id, f"model answer {answer!r} is not a scenario of this item")
    conf = raw.get("confidence")
    confidence = float(conf) if isinstance(conf, (int, float)) and not isinstance(conf, bool) else 0.0
    confidence = min(1.0, max(0.0, confidence))
    evidence = raw.get("evidence") if isinstance(raw.get("evidence"), str) else ""
    caller_turns = set(texts)
    ev_turn = _turn(raw.get("evidence_turn"), caller_turns, f"{item.id}.evidence_turn", notes)
    first = _turn(raw.get("first_available_turn"), caller_turns, f"{item.id}.first_available_turn", notes)
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
    quote = _first_quote(raw.get("first_available_quote"), evidence, ev_turn, first, texts,
                         f"{item.id}.first_available_quote", notes)
    return ItemJudgment(item.id, answer, confidence, evidence, ev_turn, first, value, quote)


def parse_tool_input(
    spec: FlowSpec, thread: Thread, raw: Any, notes: list[str] | None = None
) -> dict[str, ItemJudgment]:
    """Validate the model's tool input. Never raises. Harness notes are appended to `notes`."""
    sink: list[str] = [] if notes is None else notes
    texts = {m.turn: m.text for m in thread.caller_messages()}
    items_raw = raw.get("items") if isinstance(raw, dict) else None
    if not isinstance(items_raw, dict):
        items_raw = {}
    return {iid: _one(spec, item, items_raw.get(iid), texts, sink) for iid, item in spec.items.items()}


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
        tool = build_tool(spec, thread)
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
                # Bad credentials will fail for every thread: stop the whole run, do not degrade silently.
                self._disabled = True
                raise JudgeAuthError(f"credentials rejected ({type(e).__name__})") from e
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
                notes: list[str] = []
                items = parse_tool_input(spec, thread, block.input, notes)
                for n in notes:
                    _warn(f"{thread.thread_id}: {n}")
                return Judgment(thread.thread_id, items, self.model, PROMPT_VERSION, "live", tuple(notes),
                                transport="api")
            _warn(f"{thread.thread_id}: no tool call on attempt {attempt} (stop_reason={resp.stop_reason})")
        return None


class ClaudeCliJudge:
    """Live judge through the Claude Code CLI in headless mode (`claude -p`), on the user's subscription.

    Same system prompt, same rendered user message, and the tool's input_schema as `--json-schema`, so the
    model gets the same task as through the API and the cache key is identical. The answer comes back in
    the `structured_output` key of the CLI's JSON result and goes through the same parse_tool_input.
    ANTHROPIC_API_KEY is removed from the child environment so the CLI uses the subscription login.
    No tools, no session file, no user settings, no MCP servers, no slash commands: the child sees only
    the system prompt and the user message. One retry on any error; never raises.
    """

    def __init__(self, model: str | None = None, runner: Any = None, max_attempts: int = 2,
                 timeout: float = 180, binary: str = "claude"):
        self.model = model or os.environ.get("CALLCHECK_MODEL", DEFAULT_MODEL)
        self._runner = runner  # stands in for subprocess.run in tests
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.binary = binary
        self._disabled = False

    @property
    def available(self) -> bool:
        if self._disabled:
            return False
        return self._runner is not None or shutil.which(self.binary) is not None

    def command(self, spec: FlowSpec, thread: Thread) -> list[str]:
        schema = build_tool(spec, thread)["input_schema"]
        return [
            self.binary, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--json-schema", json.dumps(schema, ensure_ascii=False),
            "--system-prompt", SYSTEM_PROMPT,
            "--tools", "",
            "--no-session-persistence",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
        ]

    @staticmethod
    def child_env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    def _attempt(self, spec: FlowSpec, thread: Thread) -> tuple[dict | None, str, str]:
        """(structured output or None, model name reported by the CLI, error text)."""
        run = self._runner or subprocess.run
        try:
            proc = run(self.command(spec, thread), input=build_user_message(spec, thread), capture_output=True,
                       text=True, env=self.child_env(), timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return None, self.model, f"timed out after {self.timeout:.0f}s"
        except OSError as e:
            if isinstance(e, FileNotFoundError):
                self._disabled = True  # no binary: every thread would fail the same way
            return None, self.model, f"could not start {self.binary!r}: {e}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-200:]
            return None, self.model, f"exit code {proc.returncode}: {tail}"
        try:
            out = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError):
            return None, self.model, f"stdout is not JSON: {(proc.stdout or '')[:120]!r}"
        if not isinstance(out, dict):
            return None, self.model, "stdout JSON is not an object"
        usage = out.get("modelUsage")
        model = next(iter(usage)) if isinstance(usage, dict) and usage else self.model
        if out.get("is_error"):
            return None, model, f"CLI reported an error: {str(out.get('result'))[:200]}"
        structured = out.get("structured_output")
        if not isinstance(structured, dict):
            return None, model, "no structured_output object in the CLI result"
        return structured, model, ""

    def judge(self, spec: FlowSpec, thread: Thread) -> Judgment | None:
        if not self.available:
            return None
        for attempt in range(1, self.max_attempts + 1):
            structured, model, error = self._attempt(spec, thread)
            if structured is not None:
                notes: list[str] = []
                items = parse_tool_input(spec, thread, structured, notes)
                for n in notes:
                    _warn(f"{thread.thread_id}: {n}")
                return Judgment(thread.thread_id, items, model, PROMPT_VERSION, "live", tuple(notes),
                                transport="claude-cli")
            _warn(f"{thread.thread_id}: claude CLI attempt {attempt}: {error}")
            if self._disabled:
                break
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(model: str, spec: FlowSpec, thread: Thread) -> str:
    """Hash of everything the model sees: system prompt, tool schema, rendered user message.

    PROMPT_VERSION stays in the key as a human readable label, but the key no longer
    depends on someone remembering to bump it: any edit to the rules, the schema, the
    item rendering or the transcript format changes the hash.
    """
    tool_json = json.dumps(build_tool(spec, thread), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    parts = [
        model,
        PROMPT_VERSION,
        _sha(SYSTEM_PROMPT),
        _sha(tool_json),
        _sha(build_user_message(spec, thread)),
    ]
    return _sha("\n\x1e".join(parts))


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
            "notes": list(j.notes),
            "transport": j.transport,
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
                notes = tuple(n for n in entry.get("notes", []) if isinstance(n, str))
                transport = entry.get("transport") if entry.get("transport") in BACKENDS else None
                return Judgment(thread.thread_id, items, entry["model"], entry["prompt_version"], "cache", notes,
                                transport=transport)
        if not self._live_allowed():
            return None
        j = self.inner.judge(spec, thread)  # type: ignore[union-attr]
        if j is not None:
            self._store(key, j)
        return j


def backend() -> str:
    """CALLCHECK_BACKEND: "api" (default, needs ANTHROPIC_API_KEY) or "claude-cli" (Claude Code CLI login)."""
    name = os.environ.get("CALLCHECK_BACKEND", "api").strip() or "api"
    if name not in BACKENDS:
        _warn(f"unknown CALLCHECK_BACKEND {name!r}, using 'api' (choices: {', '.join(BACKENDS)})")
        return "api"
    return name


def cached_judge() -> Judge:
    """Replay only: reads q3/cache/judgments.json, never calls a model, needs no key."""
    return CachedJudge(None, model=os.environ.get("CALLCHECK_MODEL", DEFAULT_MODEL))


def live_unavailable_reason() -> str | None:
    """Why a live judge cannot run right now (None when it can). Checked before any thread is evaluated."""
    if backend() == "claude-cli":
        if shutil.which("claude") is None:
            return "CALLCHECK_BACKEND=claude-cli is set but the claude binary is not on PATH."
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY is not set."
    return None


def default_judge(live: bool = False) -> Judge:
    """Cache first. Live calls happen on a cache miss when a backend is usable, or always when live=True.

    The backend only changes the transport. The cache key does not depend on it, so entries written
    through either backend are found by a plain offline run.
    """
    model = os.environ.get("CALLCHECK_MODEL", DEFAULT_MODEL)
    inner: Judge | None
    if backend() == "claude-cli":
        cli = ClaudeCliJudge(model)
        inner = cli if cli.available else None
        if inner is None:
            _warn("CALLCHECK_BACKEND=claude-cli but the claude binary is not on PATH; cache only")
    else:
        inner = ClaudeJudge(model) if os.environ.get("ANTHROPIC_API_KEY") else None
    return CachedJudge(inner, model=model, refresh=live)


__all__ = [
    "PROMPT_VERSION", "ItemJudgment", "Judgment", "Judge", "ClaudeJudge", "ClaudeCliJudge", "CachedJudge",
    "FakeJudge", "backend", "JudgeAuthError", "cached_judge", "live_unavailable_reason",
    "default_judge", "cache_key", "build_tool", "answer_enum", "parse_tool_input", "render_transcript",
    "turn_id",
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
        try:
            j = judge.judge(spec, t)
        except JudgeAuthError as e:
            print(f"callcheck judge: {e}", file=sys.stderr)
            return 3
        if j is None:
            missing += 1
            print(f"{t.thread_id}: no judgment (no usable backend and no cache entry)")
            continue
        print(f"{t.thread_id} [{j.source}, {j.model} via {j.transport or 'unknown'}, {j.prompt_version}]")
        for n in j.notes:
            print(f"  note: {n}")
        for iid, ij in j.items.items():
            val = f" value={ij.value!r}" if ij.value else ""
            print(f"  {iid:13} {ij.answer:22} conf={ij.confidence:.2f} turn={ij.evidence_turn} "
                  f"first={ij.first_available_turn}{val}  {ij.evidence[:90]!r}")
            if ij.first_available_quote and ij.first_available_turn != ij.evidence_turn:
                print(f"  {'':13} first quote: {ij.first_available_quote[:90]!r}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(_main())
