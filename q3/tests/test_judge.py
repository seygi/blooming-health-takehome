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


def test_tool_schema_is_strict_and_covers_every_item(spec):
    tool = build_tool(spec)
    assert tool["strict"] is True
    items = tool["input_schema"]["properties"]["items"]
    assert set(items["required"]) == set(spec.items)
    assert items["additionalProperties"] is False
    q2 = items["properties"]["q2_active"]
    assert q2["properties"]["answer"]["enum"] == ["active", "inactive", UNCLEAR, NOT_DISCUSSED]
    assert set(q2["required"]) == set(q2["properties"])
    assert q2["additionalProperties"] is False


# ---- transcript rendering and cache key -------------------------------------


def test_render_transcript_numbered_turns(threads):
    text = render_transcript(_thread(threads, "thread_01"))
    assert text.splitlines()[0].startswith("[-1] AGENT: First")
    assert "[0] CALLER: Yes, please." in text


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


def test_cache_key_differs_between_threads(spec, threads):
    keys = {cache_key("m", spec, t) for t in threads}
    assert len(keys) == len(threads)


# ---- validation of the tool output ------------------------------------------


def _good_item(answer="active", turn=1):
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
    raw = {"items": {"q2_active": {**_good_item(), "confidence": 7, "evidence_turn": 42,
                                   "first_available_turn": -1}}}
    j = parse_tool_input(spec, t, raw)["q2_active"]
    assert j.confidence == 1.0
    assert j.evidence_turn is None  # 42 is not a caller turn
    assert j.first_available_turn is None  # -1 is an agent turn


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
                                 "confidence": 0.95, "evidence_turn": 2, "first_available_turn": 0,
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
    items["q1_intent"] = {"evidence": "Yes, please.", "answer": "yes", "confidence": 0.97, "evidence_turn": 0,
                          "first_available_turn": 0, "value": None}
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
    assert "[0] CALLER: Yes, please." in call["messages"][0]["content"]


def test_claude_judge_retries_once_without_tool_call(spec, threads):
    client = _Client([_Resp([_Block("text")], "end_turn"),
                      _Resp([_Block("tool_use", J.TOOL_NAME, _tool_input(spec))], "tool_use")])
    j = J.ClaudeJudge("m", client=client).judge(spec, _thread(threads, "thread_01"))
    assert j is not None and len(client.messages.calls) == 2


def test_claude_judge_refusal_returns_none(spec, threads):
    client = _Client([_Resp([], "refusal")])
    assert J.ClaudeJudge("m", client=client).judge(spec, _thread(threads, "thread_01")) is None


def test_claude_judge_bad_key_returns_none_and_disables(spec, threads, tmp_path):
    import anthropic

    httpx = pytest.importorskip("httpx2")
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("bad key", response=httpx.Response(401, request=req), body=None)
    judge = J.ClaudeJudge("m", client=_Client([err]))
    assert judge.judge(spec, _thread(threads, "thread_01")) is None
    assert judge.available is False
    path = tmp_path / "judgments.json"
    assert CachedJudge(judge, model="m", path=path).judge(spec, _thread(threads, "thread_02")) is None
    assert not path.exists()
