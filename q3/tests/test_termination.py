import pytest

from callcheck.checks import Context
from callcheck.checks.termination import check
from callcheck.model import Message, Thread

Q1 = "First \u2014 would you like our help renewing your Medi-Cal right now, over the phone?"


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def run(spec, thread):
    return check(Context(spec=spec, thread=thread))


def checks_of(findings):
    return sorted(f.check for f in findings)


@pytest.mark.parametrize("thread_id", ["thread_07", "thread_09", "thread_10"])
def test_truncated_by_simulator(spec, threads, thread_id):
    findings = run(spec, by_id(threads, thread_id))
    assert checks_of(findings) == ["termination.truncated_by_simulator"]
    f = findings[0]
    assert f.severity == "major" and f.gate == "proper_termination" and f.owner == "simulator"
    assert "episode incomplete, success cannot be confirmed" in f.problem
    assert f.uncertain is True


def test_other_termination_findings_are_confident(spec, threads):
    findings = run(spec, by_id(threads, "thread_02"))
    assert findings and all(f.uncertain is False for f in findings)


def test_no_terminal_without_marker(spec):
    thread = Thread(
        thread_id="synthetic",
        num_turns=1,
        simulator_goal_marker=False,
        messages=[
            Message(turn=-1, role="agent", text=Q1, raw_text=Q1),
            Message(turn=0, role="caller", text="Hello?", raw_text="Hello?"),
        ],
    )
    findings = run(spec, thread)
    assert checks_of(findings) == ["termination.no_terminal"]
    assert findings[0].severity == "critical" and findings[0].owner == "engine"


def test_thread_02_talks_past_terminal_and_contradictory_close(spec, threads):
    findings = run(spec, by_id(threads, "thread_02"))
    assert checks_of(findings) == ["termination.contradictory_close", "termination.talks_past_terminal"]
    past = next(f for f in findings if f.check == "termination.talks_past_terminal")
    assert past.turn == 5
    assert past.severity == "critical" and past.gate == "proper_termination" and past.owner == "engine"
    close = next(f for f in findings if f.check == "termination.contradictory_close")
    assert close.turn == 4
    assert close.severity == "major" and close.owner == "prompt"
    assert "assum" in close.problem.lower()


def test_thread_08_contradictory_close(spec, threads):
    findings = run(spec, by_id(threads, "thread_08"))
    assert checks_of(findings) == ["termination.contradictory_close"]


@pytest.mark.parametrize("thread_id", ["thread_05", "thread_03", "thread_04", "thread_06"])
def test_silent_on_clean_outcomes(spec, threads, thread_id):
    assert run(spec, by_id(threads, thread_id)) == []


def test_pleasantries_after_terminal_are_not_talking_past(spec, threads):
    # thread_01 turn 2: "You're very welcome! ... Have a wonderful day, and goodbye!"
    assert run(spec, by_id(threads, "thread_01")) == []
