import dataclasses

import pytest

from callcheck.align import align
from callcheck.checks import Context
from callcheck.checks.termination import check, is_pleasantry
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
    assert "PII" in past.fix_hint  # turn 5 asks for name, address and date of birth
    close = next(f for f in findings if f.check == "termination.contradictory_close")
    assert close.turn == 4
    assert close.severity == "major" and close.owner == "prompt"
    assert "assum" in close.problem.lower()


def test_thread_08_contradictory_close_is_soft(spec, threads):
    findings = run(spec, by_id(threads, "thread_08"))
    assert checks_of(findings) == ["termination.contradictory_close"]
    f = findings[0]
    # Gate 3 is talking past the terminal; what _complete means is an assumption, so this is soft.
    assert f.gate is None and f.severity == "major" and f.uncertain is False
    assert "Assumption" in f.problem and "standalone" in f.problem


def replace_last_agent(thread, text):
    last = max(i for i, m in enumerate(thread.messages) if m.role == "agent")
    messages = list(thread.messages)
    messages[last] = dataclasses.replace(messages[last], text=text, raw_text=text)
    return dataclasses.replace(thread, thread_id=thread.thread_id + "_variant", messages=messages)


def test_paraphrased_closing_is_unrecognized_terminal(spec, threads):
    # thread_05's closing reworded so it matches no outcome say. The simulator marker on the last caller
    # turn must not turn this into a truncation: the agent did close.
    thread = replace_last_agent(
        by_id(threads, "thread_05"),
        "Perfect, someone from our team will call you to book that visit. Take care.",
    )
    assert align(spec, thread).actual_terminal is None
    findings = run(spec, thread)
    assert checks_of(findings) == ["termination.terminal_unrecognized"]
    f = findings[0]
    assert f.uncertain is True and f.owner == "harness" and f.gate == "proper_termination"
    assert f.turn == 5
    assert "matches no outcome say" in f.fix_hint and "relax the matcher" in f.fix_hint


def test_final_statement_without_closing_phrase_is_unrecognized(spec, threads):
    thread = replace_last_agent(by_id(threads, "thread_05"), "Okay, our team will reach out to you about that.")
    assert checks_of(run(spec, thread)) == ["termination.terminal_unrecognized"]


def test_final_question_is_not_a_closing(spec, threads):
    thread = replace_last_agent(by_id(threads, "thread_05"), "Okay, which day works best for you?")
    assert checks_of(run(spec, thread)) == ["termination.truncated_by_simulator"]


@pytest.mark.parametrize(
    "text",
    [
        "No problem, your coverage is now renewed.",
        "I'm glad, I've submitted your renewal to the county.",
    ],
)
def test_substantive_text_after_terminal_is_talking_past(spec, threads, text):
    thread = by_id(threads, "thread_01")
    messages = list(thread.messages)
    last = len(messages) - 1
    messages[last] = dataclasses.replace(messages[last], text=text, raw_text=text)
    findings = run(spec, dataclasses.replace(thread, messages=messages))
    assert checks_of(findings) == ["termination.talks_past_terminal"]
    f = findings[0]
    assert "PII" not in f.fix_hint and "improvised next steps" in f.fix_hint


@pytest.mark.parametrize(
    "sentence",
    [
        "You're very welcome!",
        "I'm glad we could get that sorted out.",
        "Have a wonderful day, and goodbye!",
        "Thank you so much.",
        "Take care!",
        "Glad I could help.",
        "My pleasure, have a great day.",
    ],
)
def test_pleasantry_sentences(sentence):
    assert is_pleasantry(sentence)


@pytest.mark.parametrize(
    "sentence",
    [
        "No problem, your coverage is now renewed.",
        "I'm glad, I've submitted your renewal to the county.",
        "Thank you, I'll send the forms today.",
        "Have a wonderful day, your appointment is Tuesday.",
    ],
)
def test_not_pleasantry_sentences(sentence):
    assert not is_pleasantry(sentence)


@pytest.mark.parametrize("thread_id", ["thread_05", "thread_03", "thread_04", "thread_06"])
def test_silent_on_clean_outcomes(spec, threads, thread_id):
    assert run(spec, by_id(threads, thread_id)) == []


def test_pleasantries_after_terminal_are_not_talking_past(spec, threads):
    # thread_01 turn 2: "You're very welcome! ... Have a wonderful day, and goodbye!"
    assert run(spec, by_id(threads, "thread_01")) == []
