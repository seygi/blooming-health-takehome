"""judge.py tests. No network: everything runs on FakeJudge, a temp cache, or the committed cache."""

import json
from pathlib import Path

import pytest

from callcheck import judge as J
from callcheck.judge import (
    CACHE_PATH,
    NOT_DISCUSSED,
    UNCLEAR,
    CachedJudge,
    FakeJudge,
    ItemJudgment,
    Judgment,
    answer_enum,
    build_tool,
    cache_key,
    parse_tool_input,
    render_transcript,
)
from callcheck.model import Message, Thread


def _thread(threads, tid):
    return next(t for t in threads if t.thread_id == tid)


def _edited(thread: Thread, turn: int, new_text: str) -> Thread:
    msgs = [
        Message(m.turn, m.role, new_text, new_text) if (m.role == "caller" and m.turn == turn) else m
        for m in thread.messages
    ]
    return Thread(thread.thread_id, msgs, thread.num_turns, thread.simulator_goal_marker)


class _Boom:
    """Inner judge that must never be called."""

    def judge(self, spec, thread):  # pragma: no cover, failing is the point
        raise AssertionError("inner judge called")


# ---- schema built from the spec ------------------------------------------------


def test_answer_enum_q2_active(spec):
    assert answer_enum(spec.items["q2_active"]) == ["active", "inactive", UNCLEAR, NOT_DISCUSSED]


def test_answer_enum_decline_has_all_reasons(spec):
    enum = answer_enum(spec.items["decline"])
    for sid in ["reveals_other_coverage", "reveals_active", "reveals_out_of_county",
                "self_submit", "not_interested", "privacy_concern", "other"]:
        assert sid in enum


def test_tool_schema_is_strict_and_covers_every_item(spec, threads):
    tool = build_tool(spec, _thread(threads, "thread_01"))
    assert tool["strict"] is True
    items = tool["input_schema"]["properties"]["items"]
    assert set(items["required"]) == set(spec.items)
    assert items["additionalProperties"] is False
    q2 = items["properties"]["q2_active"]
    assert q2["properties"]["answer"]["enum"] == ["active", "inactive", UNCLEAR, NOT_DISCUSSED]
    assert set(q2["required"]) == set(q2["properties"])
    assert q2["additionalProperties"] is False


def test_tool_schema_turn_fields_are_caller_turn_ids(spec, threads):
    # Strict mode documents enum and anyOf, not pattern, so the caller ids of this
    # thread are an enum: every value matches ^C\\d+$ and names a real caller turn.
    t = _thread(threads, "thread_01")
    q2 = build_tool(spec, t)["input_schema"]["properties"]["items"]["properties"]["q2_active"]
    for field in ("evidence_turn", "first_available_turn"):
        variants = q2["properties"][field]["anyOf"]
        assert {"type": "null"} in variants
        ids = next(v["enum"] for v in variants if v.get("type") == "string")
        assert ids == ["C0", "C1", "C2"]


def test_tool_schema_no_caller_turns_only_null(spec):
    t = Thread("empty", [Message(-1, "agent", "Hi", "Hi")], 0, False)
    q2 = build_tool(spec, t)["input_schema"]["properties"]["items"]["properties"]["q2_active"]
    assert q2["properties"]["evidence_turn"] == {"type": "null"}


# ---- transcript rendering and cache key -------------------------------------


def test_render_transcript_turn_ids(threads):
    text = render_transcript(_thread(threads, "thread_01"))
    lines = text.splitlines()
    assert lines[0].startswith("[A-open] AGENT: First")
    assert "[C0] CALLER: Yes, please." in text
    assert "[A0] AGENT: Our records show" in text
    assert "[C1] CALLER: Oh, that" in text
    assert "[-1]" not in text and "[0]" not in text


def test_render_transcript_every_line_labelled_or_indented(threads):
    # thread_05 and thread_09 have multi line agent messages, thread_01 has a blank line.
    for t in threads:
        for line in render_transcript(t).splitlines():
            assert line.startswith("[") or line.startswith("    "), (t.thread_id, line)
            assert line.strip(), (t.thread_id, "blank line")


def test_render_transcript_continuation_follows_its_turn(threads):
    text = render_transcript(_thread(threads, "thread_01"))
    lines = text.splitlines()
    i = next(n for n, line in enumerate(lines) if line.startswith("[A0]"))
    assert lines[i + 1] == "    Please let me know."


def test_turn_ids():
    assert J.turn_id("caller", 3) == "C3"
    assert J.turn_id("agent", 3) == "A3"
    assert J.turn_id("agent", -1) == "A-open"


def test_render_transcript_has_no_goal_marker(threads):
    for t in threads:
        assert "[GOAL_ACHIEVED]" not in render_transcript(t)


def test_cache_key_stable(spec, threads):
    t = _thread(threads, "thread_03")
    assert cache_key("m", spec, t) == cache_key("m", spec, t)
    assert len(cache_key("m", spec, t)) == 64


def test_cache_key_sensitive_to_thread_text(spec, threads):
    t = _thread(threads, "thread_03")
    assert cache_key("m", spec, t) != cache_key("m", spec, _edited(t, 1, "Yes it is active."))


def test_cache_key_sensitive_to_prompt_version_and_model(spec, threads, monkeypatch):
    t = _thread(threads, "thread_03")
    before = cache_key("m", spec, t)
    assert cache_key("other-model", spec, t) != before
    monkeypatch.setattr(J, "PROMPT_VERSION", "v_test")
    assert cache_key("m", spec, t) != before


def test_cache_key_sensitive_to_system_prompt_text(spec, threads, monkeypatch):
    # Editing the rules without bumping PROMPT_VERSION must still miss the cache.
    t = _thread(threads, "thread_03")
    before = cache_key("m", spec, t)
    monkeypatch.setattr(J, "SYSTEM_PROMPT", J.SYSTEM_PROMPT + "\nExtra rule.")
    assert cache_key("m", spec, t) != before


def test_cache_key_sensitive_to_tool_schema(spec, threads, monkeypatch):
    t = _thread(threads, "thread_03")
    before = cache_key("m", spec, t)
    orig = J.build_tool

    def changed(*a, **k):
        tool = orig(*a, **k)
        return {**tool, "description": tool["description"] + " changed"}

    monkeypatch.setattr(J, "build_tool", changed)
    assert cache_key("m", spec, t) != before


def test_cache_key_sensitive_to_user_message_template(spec, threads, monkeypatch):
    t = _thread(threads, "thread_03")
    before = cache_key("m", spec, t)
    orig = J.build_user_message
    monkeypatch.setattr(J, "build_user_message", lambda s, th: orig(s, th) + "\nchanged")
    assert cache_key("m", spec, t) != before


def test_cache_key_differs_between_threads(spec, threads):
    keys = {cache_key("m", spec, t) for t in threads}
    assert len(keys) == len(threads)


# ---- validation of the tool output ------------------------------------------


def _good_item(answer="active", turn="C1"):
    return {"evidence": "I definitely have active coverage", "answer": answer, "confidence": 0.95,
            "evidence_turn": turn, "first_available_turn": turn, "value": None}


def test_parse_valid_output(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {iid: {**_good_item(NOT_DISCUSSED, None), "evidence": "", "confidence": 0.9}
                     for iid in spec.items}}
    raw["items"]["q2_active"] = _good_item()
    items = parse_tool_input(spec, t, raw)
    assert set(items) == set(spec.items)
    assert items["q2_active"].answer == "active"
    assert items["q2_active"].evidence_turn == 1


def test_parse_answer_outside_enum_becomes_unclear(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": _good_item(answer="maybe_active")}}
    items = parse_tool_input(spec, t, raw)
    j = items["q2_active"]
    assert j.answer == UNCLEAR and j.confidence == 0.0
    assert "maybe_active" in j.evidence


def test_parse_missing_item_becomes_unclear(spec, threads):
    items = parse_tool_input(spec, _thread(threads, "thread_01"), {"items": {}})
    assert set(items) == set(spec.items)
    assert all(j.answer == UNCLEAR and j.confidence == 0.0 for j in items.values())


def test_parse_unknown_item_ignored_and_garbage_never_crashes(spec, threads):
    t = _thread(threads, "thread_01")
    for raw in [None, "text", {"items": "x"}, {"items": {"q9_bogus": _good_item(), "q2_active": "x"}},
                {"items": {"q2_active": {"answer": "active", "confidence": "high",
                                         "evidence_turn": "one"}}}]:
        items = parse_tool_input(spec, t, raw)
        assert set(items) == set(spec.items)
        assert "q9_bogus" not in items
        assert items["q2_active"].answer in answer_enum(spec.items["q2_active"])


def test_parse_clamps_confidence_and_drops_non_caller_turn(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": {**_good_item(), "confidence": 7, "evidence_turn": "C42",
                                   "first_available_turn": "A-open"}}}
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q2_active"]
    assert j.confidence == 1.0
    assert j.evidence_turn is None  # C42 is not a caller turn of this thread
    assert j.first_available_turn is None  # A-open is the agent opener
    assert any("C42" in n and "q2_active" in n for n in notes)
    assert any("A-open" in n for n in notes)


def test_parse_maps_caller_ids_to_int_turns(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": {**_good_item(), "evidence_turn": "C1", "first_available_turn": "C1"}}}
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q2_active"]
    assert (j.evidence_turn, j.first_available_turn) == (1, 1)
    assert notes == []


def test_parse_agent_turn_id_becomes_none_with_note(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": {**_good_item(), "evidence_turn": "A1", "first_available_turn": "C1"}}}
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q2_active"]
    assert j.evidence_turn is None and j.first_available_turn == 1
    assert any("A1" in n and "agent" in n for n in notes)


def test_parse_bare_int_turn_rejected_with_note(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": {**_good_item(), "evidence_turn": 1, "first_available_turn": 1}}}
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q2_active"]
    assert j.evidence_turn is None and j.first_available_turn is None
    assert notes


# ---- first_available_quote ----------------------------------------------------


def test_tool_schema_has_first_available_quote(spec, threads):
    q2 = build_tool(spec, _thread(threads, "thread_07"))["input_schema"]["properties"]["items"]["properties"]["q2_active"]
    assert q2["properties"]["first_available_quote"]["type"] == "string"
    assert "first_available_quote" in q2["required"]


def test_item_judgment_quote_defaults_blank():
    ij = ItemJudgment("q2_active", "active", 0.9, "x", 1, 1, None)
    assert ij.first_available_quote == ""


def _q5_choice(evidence, ev_turn, first, quote):
    return {"q5_choice": {"evidence": evidence, "answer": "by_phone", "confidence": 0.8, "evidence_turn": ev_turn,
                          "first_available_turn": first, "first_available_quote": quote, "value": None}}


def test_first_available_quote_kept_when_in_that_turn(spec, threads):
    # thread_07: the caller asks to finish the packet over the phone at C3, before q5_choice is offered at A4.
    t = _thread(threads, "thread_07")
    raw = {"items": _q5_choice("Can we just go ahead and finish it over the phone now?", "C4", "C3",
                               "can we just finish that yellow packet over the phone right now?")}
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q5_choice"]
    assert j.first_available_turn == 3 and j.evidence_turn == 4
    assert j.first_available_quote == "can we just finish that yellow packet over the phone right now?"
    assert not any("q5_choice" in n for n in notes)


def test_first_available_quote_from_other_turn_blanked_with_note(spec, threads):
    t = _thread(threads, "thread_07")
    raw = {"items": _q5_choice("Can we just go ahead and finish it over the phone now?", "C4", "C3",
                               "Can we just go ahead and finish it over the phone now?")}  # this is C4 text
    notes: list[str] = []
    j = parse_tool_input(spec, t, raw, notes)["q5_choice"]
    assert j.first_available_quote == ""
    assert any("q5_choice.first_available_quote" in n for n in notes)


def test_first_available_quote_falls_back_to_evidence_of_same_turn(spec, threads):
    # first moves to the evidence turn (C3 < C4), so the evidence quote is a quote of the first turn.
    t = _thread(threads, "thread_07")
    raw = {"items": _q5_choice("can we just finish that yellow packet over the phone right now?", "C3", "C4", "")}
    j = parse_tool_input(spec, t, raw)["q5_choice"]
    assert j.first_available_turn == 3
    assert j.first_available_quote == "can we just finish that yellow packet over the phone right now?"


def test_first_available_quote_never_borrows_evidence_from_other_turn(spec, threads):
    t = _thread(threads, "thread_07")
    raw = {"items": _q5_choice("Can we just go ahead and finish it over the phone now?", "C4", "C3", "")}
    j = parse_tool_input(spec, t, raw)["q5_choice"]
    assert j.first_available_turn == 3 and j.first_available_quote == ""


def test_first_available_quote_tolerates_straight_apostrophe(spec, threads):
    # thread_01 C1 uses a curly apostrophe: "Oh, that’s weird."
    t = _thread(threads, "thread_01")
    raw = {"items": {"q2_active": {**_good_item(), "first_available_quote": "that's weird. I definitely have"}}}
    j = parse_tool_input(spec, t, raw)["q2_active"]
    assert j.first_available_quote == "that's weird. I definitely have"


def test_first_available_quote_blank_for_not_discussed(spec, threads):
    t = _thread(threads, "thread_01")
    raw = {"items": {"q4_residency": {"evidence": "", "answer": NOT_DISCUSSED, "confidence": 0.9,
                                      "evidence_turn": None, "first_available_turn": None,
                                      "first_available_quote": "Yes, please.", "value": None}}}
    assert parse_tool_input(spec, t, raw)["q4_residency"].first_available_quote == ""


def test_cache_round_trip_keeps_quote_and_reads_old_entries(spec, threads, tmp_path):
    t = _thread(threads, "thread_07")
    path = tmp_path / "judgments.json"
    preset = {"q5_choice": ItemJudgment("q5_choice", "by_phone", 0.8, "finish it over the phone now", 4, 3,
                                        None, "finish that yellow packet over the phone")}
    CachedJudge(FakeJudge({"thread_07": preset}), model="m", path=path).judge(spec, t)
    j = CachedJudge(_Boom(), model="m", path=path).judge(spec, t)
    assert j.items["q5_choice"].first_available_quote == "finish that yellow packet over the phone"
    data = json.loads(path.read_text())
    for entry in data["entries"].values():
        for ij in entry["items"].values():
            ij.pop("first_available_quote")
    path.write_text(json.dumps(data))
    old = CachedJudge(_Boom(), model="m", path=path).judge(spec, t)
    assert old.items["q5_choice"].first_available_quote == ""


# ---- judges -------------------------------------------------------------------


def test_fake_judge_round_trip(spec, threads):
    t = _thread(threads, "thread_06")
    preset = {"q1_intent": ItemJudgment("q1_intent", "not_interested", 0.95, "I'm not interested", 0, 0, None)}
    j = FakeJudge({"thread_06": preset}).judge(spec, t)
    assert j.source == "fake" and j.thread_id == "thread_06"
    assert j.items["q1_intent"].answer == "not_interested"
    assert set(j.items) == set(spec.items)  # missing items filled as not_discussed
    assert j.items["q4_residency"].answer == NOT_DISCUSSED
    assert FakeJudge({}).judge(spec, _thread(threads, "thread_01")) is None


def test_cached_judge_no_key_empty_cache_returns_none(spec, threads, tmp_path):
    cj = CachedJudge(None, model="m", path=tmp_path / "judgments.json")
    assert cj.judge(spec, _thread(threads, "thread_01")) is None
    assert not (tmp_path / "judgments.json").exists()


def test_cached_judge_miss_calls_inner_and_stores_then_hits(spec, threads, tmp_path):
    t = _thread(threads, "thread_06")
    path = tmp_path / "judgments.json"
    preset = {"q1_intent": ItemJudgment("q1_intent", "not_interested", 0.95, "not interested", 0, 0, None)}
    first = CachedJudge(FakeJudge({"thread_06": preset}), model="m", path=path).judge(spec, t)
    assert first is not None and first.source == "fake"
    assert path.exists()
    again = CachedJudge(_Boom(), model="m", path=path).judge(spec, t)
    assert again.source == "cache"
    assert again.items == first.items


def test_cached_judge_refresh_forces_inner(spec, threads, tmp_path):
    t = _thread(threads, "thread_06")
    path = tmp_path / "judgments.json"
    preset = {"q1_intent": ItemJudgment("q1_intent", "not_interested", 0.9, "x", 0, 0, None)}
    CachedJudge(FakeJudge({"thread_06": preset}), model="m", path=path).judge(spec, t)
    newer = {"q1_intent": ItemJudgment("q1_intent", "not_interested", 0.5, "y", 0, 0, None)}
    j = CachedJudge(FakeJudge({"thread_06": newer}), model="m", path=path, refresh=True).judge(spec, t)
    assert j.source == "fake" and j.items["q1_intent"].confidence == 0.5


def test_captures_value_derived_from_spec(spec):
    assert {iid for iid, it in spec.items.items() if J.captures_value(it)} == {"q3_plan", "decline"}


def test_q3_plan_value_normalized_to_option(spec, threads):
    raw = {"items": {"q3_plan": {"evidence": "covered by Kaiser Permanente", "answer": "__close__",
                                 "confidence": 0.95, "evidence_turn": "C2", "first_available_turn": "C0",
                                 "value": "kaiser permanente"},
                     "q2_active": {**_good_item(), "value": "should be dropped"}}}
    items = parse_tool_input(spec, _thread(threads, "thread_09"), raw)
    assert items["q3_plan"].value == "Kaiser Permanente"
    assert items["q3_plan"].first_available_turn == 0
    assert items["q2_active"].value is None


def test_default_judge_without_key_is_offline(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    j = J.default_judge()
    assert isinstance(j, CachedJudge) and j.inner is None


# ---- ClaudeJudge request shape, with a mocked client (no network) ----------


class _Block:
    def __init__(self, type, name=None, input=None):
        self.type, self.name, self.input = type, name, input


class _Resp:
    def __init__(self, content, stop_reason):
        self.content, self.stop_reason, self.stop_details = content, stop_reason, None


class _Messages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class _Client:
    def __init__(self, responses):
        self.messages = _Messages(responses)


def _tool_input(spec):
    items = {iid: {"evidence": "", "answer": NOT_DISCUSSED, "confidence": 0.9, "evidence_turn": None,
                   "first_available_turn": None, "value": None} for iid in spec.items}
    items["q1_intent"] = {"evidence": "Yes, please.", "answer": "yes", "confidence": 0.97, "evidence_turn": "C0",
                          "first_available_turn": "C0", "value": None}
    return {"items": items}


def test_claude_judge_request_shape(spec, threads):
    client = _Client([_Resp([_Block("tool_use", J.TOOL_NAME, _tool_input(spec))], "tool_use")])
    j = J.ClaudeJudge("claude-sonnet-5-5", client=client).judge(spec, _thread(threads, "thread_01"))
    assert j.source == "live" and j.items["q1_intent"].answer == "yes"
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["tool_choice"]["type"] == "auto"  # forced tool_choice is a 400 on this model
    assert "temperature" not in call  # non default temperature is a 400 on this model
    assert call["tools"][0]["strict"] is True
    assert "[C0] CALLER: Yes, please." in call["messages"][0]["content"]
    assert j.items["q1_intent"].evidence_turn == 0
    turn_schema = call["tools"][0]["input_schema"]["properties"]["items"]["properties"]["q1_intent"]
    assert {"type": "string", "enum": ["C0", "C1", "C2"]} in turn_schema["properties"]["evidence_turn"]["anyOf"]


def test_claude_judge_retries_once_without_tool_call(spec, threads):
    client = _Client([_Resp([_Block("text")], "end_turn"),
                      _Resp([_Block("tool_use", J.TOOL_NAME, _tool_input(spec))], "tool_use")])
    j = J.ClaudeJudge("m", client=client).judge(spec, _thread(threads, "thread_01"))
    assert j is not None and len(client.messages.calls) == 2


def test_claude_judge_refusal_returns_none(spec, threads):
    client = _Client([_Resp([], "refusal")])
    assert J.ClaudeJudge("m", client=client).judge(spec, _thread(threads, "thread_01")) is None


def test_claude_judge_bad_key_raises_and_disables(spec, threads, tmp_path):
    import anthropic

    httpx = pytest.importorskip("httpx2")
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    judge = J.ClaudeJudge("m", client=_Client([err]))
    with pytest.raises(J.JudgeAuthError):
        judge.judge(spec, _thread(threads, "thread_01"))
    assert judge.available is False
    path = tmp_path / "judgments.json"
    assert CachedJudge(judge, model="m", path=path).judge(spec, _thread(threads, "thread_02")) is None
    assert not path.exists()
