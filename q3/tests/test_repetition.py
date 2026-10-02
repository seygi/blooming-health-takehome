from types import SimpleNamespace

from callcheck.checks import Context
from callcheck.checks.repetition import check
from callcheck.model import Message, Thread

Q1 = "First \u2014 would you like our help renewing your Medi-Cal right now, over the phone?"
Q2 = (
    "Our records show your Medi-Cal may not be active right now. "
    "Can you confirm \u2014 do you currently have active Medi-Cal coverage?"
)


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def synthetic(*triples):
    messages = [Message(turn=turn, role=role, text=text, raw_text=text) for turn, role, text in triples]
    return Thread(thread_id="synthetic", messages=messages, num_turns=2, simulator_goal_marker=False)


def judgment(**first_available):
    items = {
        item: SimpleNamespace(
            answer="x",
            confidence=0.9,
            evidence="quote",
            evidence_turn=turn,
            first_available_turn=turn,
            value=None,
        )
        for item, turn in first_available.items()
    }
    return SimpleNamespace(items=items)


def test_asked_twice_fires(spec):
    thread = synthetic(
        (-1, "agent", Q1),
        (0, "caller", "Yes."),
        (0, "agent", Q2),
        (1, "caller", "Hmm, what?"),
        (1, "agent", Q2),
    )
    findings = check(Context(spec=spec, thread=thread))
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "repetition.asked_twice"
    assert f.turn == 1
    assert f.gate is None and f.severity == "minor" and f.owner == "prompt"
    assert "q2_active" in f.evidence


def test_silent_on_clean_real_thread(spec, threads):
    # thread_05: every item asked exactly once, no judgment supplied
    assert check(Context(spec=spec, thread=by_id(threads, "thread_05"))) == []


def test_volunteered_before_asked_thread_03(spec, threads):
    # thread_03: at caller turn 0 the caller already says "isn't active" and "don't have any
    # other insurance", at caller turn 1 "moved out of San Diego County". The agent still asks
    # q2_active (agent turn 0), q3_coverage (1) and q4_residency (2). Caller turn t comes
    # before agent turn t, so first_available_turn <= asked turn means volunteered.
    ctx = Context(
        spec=spec,
        thread=by_id(threads, "thread_03"),
        judgment=judgment(q2_active=0, q3_coverage=0, q4_residency=1),
    )
    findings = check(ctx)
    assert {f.check for f in findings} == {"repetition.asked_after_volunteered"}
    assert sorted(f.turn for f in findings) == [0, 1, 2]
    assert all(f.gate is None and f.severity == "minor" and f.owner == "prompt" for f in findings)


def test_not_volunteered_when_answer_came_after_question(spec, threads):
    ctx = Context(
        spec=spec,
        thread=by_id(threads, "thread_05"),
        judgment=judgment(q2_active=1, q3_coverage=2),
    )
    assert check(ctx) == []


def test_defensive_on_odd_judgment(spec, threads):
    thread = by_id(threads, "thread_05")
    odd = SimpleNamespace(items={"q2_active": SimpleNamespace(answer="active")})
    assert check(Context(spec=spec, thread=thread, judgment=odd)) == []
    assert check(Context(spec=spec, thread=thread, judgment=object())) == []
