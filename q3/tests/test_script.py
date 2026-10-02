import pytest

from callcheck.checks import Context
from callcheck.checks.script import check
from callcheck.judge import FakeJudge, ItemJudgment
from callcheck.model import Message, Thread

Q1 = "First \u2014 would you like our help renewing your Medi-Cal right now, over the phone?"
Q2 = (
    "Our records show your Medi-Cal may not be active right now. "
    "Can you confirm \u2014 do you currently have active Medi-Cal coverage?"
)
Q4 = "Do you still live in San Diego County?"

ALL = [f"thread_{i:02d}" for i in range(1, 11)]


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def run(spec, thread):
    return check(Context(spec=spec, thread=thread))


def synthetic(*triples):
    messages = [Message(turn=turn, role=role, text=text, raw_text=text) for turn, role, text in triples]
    return Thread(thread_id="synthetic", messages=messages, num_turns=3, simulator_goal_marker=False)


def test_thread_04_off_script_reask(spec, threads):
    findings = run(spec, by_id(threads, "thread_04"))
    assert [f.check for f in findings] == ["script.off_script_question"]
    f = findings[0]
    assert f.turn == 1
    assert f.gate is None and f.severity == "minor" and f.owner == "prompt"
    assert "Just to be clear" in f.evidence


def test_no_off_script_after_terminal(spec, threads):
    # thread_02 turn 5 asks for PII after _complete; termination owns that finding
    assert run(spec, by_id(threads, "thread_02")) == []


SKIP = [
    # q2_active answered "inactive" leads to q3_coverage, not q4_residency
    (-1, "agent", Q1),
    (0, "caller", "Yes."),
    (0, "agent", Q2),
    (1, "caller", "No, and I have no other insurance."),
    (1, "agent", Q4),
]


def test_skip_without_judgment_is_uncertain(spec):
    findings = run(spec, synthetic(*SKIP))
    assert [f.check for f in findings] == ["script.out_of_order"]
    f = findings[0]
    assert f.turn == 1 and f.severity == "major" and f.gate == "correct_routing" and f.owner == "engine"
    # Without the judge we cannot tell a skip from a volunteered answer: never a confident failure.
    assert f.uncertain is True


def test_skip_with_judgment_is_left_to_routing(spec):
    # The caller volunteered q3_coverage, so the skip is fine; routing.compare_paths owns skip detection.
    thread = synthetic(*SKIP)
    labels = {"q3_coverage": ItemJudgment("q3_coverage", "false", 0.95, "I have no other insurance", 1, 1, None)}
    judgment = FakeJudge({"synthetic": labels}).judge(spec, thread)
    assert check(Context(spec=spec, thread=thread, judgment=judgment)) == []


def test_wrong_first_question_fires(spec):
    thread = synthetic((-1, "agent", Q2))
    assert [f.check for f in run(spec, thread)] == ["script.out_of_order"]


def test_repeat_is_not_an_order_violation(spec):
    thread = synthetic(
        (-1, "agent", Q1),
        (0, "caller", "Yes."),
        (0, "agent", Q2),
        (1, "caller", "What?"),
        (1, "agent", Q2),
    )
    assert run(spec, thread) == []


@pytest.mark.parametrize("thread_id", ALL)
def test_no_order_violation_on_real_threads(spec, threads, thread_id):
    findings = run(spec, by_id(threads, thread_id))
    assert [f for f in findings if f.check == "script.out_of_order"] == []


def test_all_four_checks_registered():
    from callcheck.checks import REGISTRY, hygiene, repetition, script, termination

    for module in (hygiene, repetition, script, termination):
        assert module.check in REGISTRY
