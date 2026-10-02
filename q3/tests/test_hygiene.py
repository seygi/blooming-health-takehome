import pytest

from callcheck.checks import Context
from callcheck.checks.hygiene import check
from callcheck.model import Message, Thread


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def run(spec, thread):
    return check(Context(spec=spec, thread=thread))


def synthetic(*texts):
    messages = [Message(turn=i - 1, role="agent", text=t, raw_text=t) for i, t in enumerate(texts)]
    return Thread(thread_id="synthetic", messages=messages, num_turns=1, simulator_goal_marker=False)


@pytest.mark.parametrize(
    "thread_id", ["thread_03", "thread_04", "thread_05", "thread_06", "thread_09", "thread_10"]
)
def test_fires_on_leaky_threads(spec, threads, thread_id):
    findings = run(spec, by_id(threads, thread_id))
    assert findings
    for f in findings:
        assert f.gate == "clean_speech"
        assert f.severity == "major"
        assert f.owner == "prompt"
        assert "TTS" in f.fix_hint


@pytest.mark.parametrize("thread_id", ["thread_01", "thread_08"])
def test_silent_on_clean_threads(spec, threads, thread_id):
    assert run(spec, by_id(threads, thread_id)) == []


def test_thread_05_one_finding_listing_all_turns(spec, threads):
    findings = run(spec, by_id(threads, "thread_05"))
    assert len(findings) == 1
    f = findings[0]
    assert f.check == "hygiene.stage_direction"
    assert f.turn == -1
    assert "turns -1, 0, 1, 2, 3, 4" in f.evidence
    assert "(Waiting for your response.)" in f.evidence


def test_thread_03_distinct_quotes_in_one_finding(spec, threads):
    findings = run(spec, by_id(threads, "thread_03"))
    assert [f.check for f in findings] == ["hygiene.stage_direction"]
    assert "(Go ahead and let me know.)" in findings[0].evidence
    assert "(Please let me know your answer.)" in findings[0].evidence


def test_thread_04_meta_line(spec, threads):
    findings = run(spec, by_id(threads, "thread_04"))
    assert [f.check for f in findings] == ["hygiene.system_text"]
    assert findings[0].turn == 2


def test_thread_06_meta_line_past_tense(spec, threads):
    findings = run(spec, by_id(threads, "thread_06"))
    assert [f.check for f in findings] == ["hygiene.system_text"]
    assert findings[0].turn == 1


def test_marker_and_markdown_synthetic(spec):
    thread = synthetic("Hi there [GOAL_ACHIEVED]", "**Option A** is best")
    checks = sorted(f.check for f in run(spec, thread))
    assert checks == ["hygiene.markdown", "hygiene.simulator_marker"]


def test_inline_parenthetical_is_not_a_stage_direction(spec):
    thread = synthetic("Family Health Centers (FHCSD) can help.")
    assert run(spec, thread) == []
