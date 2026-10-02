import dataclasses

import pytest

from callcheck.checks import REGISTRY, Context, say_coverage
from callcheck.checks.say_coverage import check

ALL = [f"thread_{i:02d}" for i in range(1, 11)]


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def run(spec, thread):
    return check(Context(spec=spec, thread=thread))


def test_registered():
    assert say_coverage.check in REGISTRY


def test_thread_01_drops_take_care(spec, threads):
    findings = run(spec, by_id(threads, "thread_01"))
    assert [f.check for f in findings] == ["say_coverage.partial_say"]
    f = findings[0]
    assert f.severity == "minor" and f.gate is None and f.owner == "prompt"
    assert f.turn == 1
    assert '"Take care."' in f.evidence and "medi_cal_active" in f.evidence


@pytest.mark.parametrize("thread_id", [t for t in ALL if t != "thread_01"])
def test_silent_when_says_are_complete(spec, threads, thread_id):
    # thread_03/06 say "Take care, and goodbye!", thread_04 "Thank you, and take care!": the sentence is there.
    assert run(spec, by_id(threads, thread_id)) == []


def test_one_finding_lists_every_missing_sentence(spec, threads):
    thread = by_id(threads, "thread_03")
    messages = list(thread.messages)
    text = "Thanks for letting me know. Because Medi-Cal is handled county by county, someone will call."
    messages[-1] = dataclasses.replace(messages[-1], text=text, raw_text=text)
    findings = run(spec, dataclasses.replace(thread, messages=messages))
    assert len(findings) == 1
    assert '"Take care."' in findings[0].evidence
    assert "point you in the right direction" in findings[0].evidence
