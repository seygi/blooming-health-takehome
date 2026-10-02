"""routing.py tests. FakeJudge only, labels from fixtures_labels (hand labels, not model output)."""

import pytest
from fixtures_labels import EXPECTED_TERMINALS, HAND_LABELS, L

from callcheck.align import align
from callcheck.judge import FakeJudge, Judgment
from callcheck.model import Message, Thread
from callcheck.routing import MIN_CONFIDENCE, check_routing


def by_id(threads, tid):
    return next(t for t in threads if t.thread_id == tid)


def judgment_for(spec, thread, labels) -> Judgment:
    return FakeJudge({thread.thread_id: labels}).judge(spec, thread)


def run(spec, thread, labels, captured=None, min_confidence=MIN_CONFIDENCE):
    j = None if labels is None else judgment_for(spec, thread, labels)
    return check_routing(spec, thread, align(spec, thread), j, captured=captured, min_confidence=min_confidence)


def checks_of(findings):
    return sorted(f.check for f in findings)


def synthetic(spec, turns: list[tuple[str, str]]) -> Thread:
    """turns: (role, text) pairs. Agent opens at turn -1; caller turn t precedes agent turn t."""
    msgs, turn = [], -1
    for role, text in turns:
        if role == "caller":
            turn += 1
        msgs.append(Message(turn=turn, role=role, text=text, raw_text=text))
    return Thread("synthetic", msgs, turn + 1, False)


def q(spec, item):
    return spec.items[item].question.text


def say(spec, outcome):
    return spec.outcomes[outcome].say


# ---- real threads under hand labels ----------------------------------------------


@pytest.mark.parametrize("tid", sorted(HAND_LABELS))
def test_expected_terminal_under_hand_labels(spec, threads, tid):
    _findings, result = run(spec, by_id(threads, tid), HAND_LABELS[tid])
    assert result.expected_terminal == EXPECTED_TERMINALS[tid]
    assert result.stopped_at is None


@pytest.mark.parametrize("tid", sorted(HAND_LABELS))
def test_real_threads_have_no_hard_routing_findings(spec, threads, tid):
    # Every agent in the dataset routed along the path the caller's answers prescribe.
    findings, _ = run(spec, by_id(threads, tid), HAND_LABELS[tid])
    assert [f for f in findings if f.gate is not None] == []


def test_thread_06_decline_path(spec, threads):
    _, result = run(spec, by_id(threads, "thread_06"), HAND_LABELS["thread_06"])
    assert result.expected_path == ["q1_intent", "decline"]
    assert result.actual_terminal == "escalated_chw_out_of_county"


def test_thread_09_plan_captured_from_volunteered_answer(spec, threads):
    findings, result = run(spec, by_id(threads, "thread_09"), HAND_LABELS["thread_09"])
    assert result.expected_path == ["q1_intent", "q2_active", "q3_coverage", "q3_plan"]
    assert result.actual_terminal is None
    assert findings == []


# ---- judge unavailable ------------------------------------------------------------


def test_judge_unavailable(spec, threads):
    findings, result = run(spec, by_id(threads, "thread_01"), None)
    assert checks_of(findings) == ["routing.judge_unavailable"]
    f = findings[0]
    assert (f.severity, f.gate, f.owner, f.uncertain) == ("major", "correct_routing", "harness", True)
    assert "ANTHROPIC_API_KEY" in f.fix_hint and "--live" in f.fix_hint and "judgments.json" in f.fix_hint
    assert result.expected_terminal is None and result.actual_terminal == "medi_cal_active"


# ---- wrong terminal ---------------------------------------------------------------


def test_wrong_terminal_last_item(spec, threads):
    # thread_05 delivered the in person appointment outcome; pretend the caller asked for phone.
    labels = dict(HAND_LABELS["thread_05"])
    labels["q5_choice"] = L("q5_choice", "by_phone", 0.95, "Let's just do it over the phone.", 5)
    findings, result = run(spec, by_id(threads, "thread_05"), labels)
    wrong = [f for f in findings if f.check == "routing.wrong_terminal"]
    assert len(wrong) == 1
    f = wrong[0]
    assert (f.severity, f.gate, f.owner, f.uncertain) == ("critical", "correct_routing", "prompt", False)
    assert f.turn == 5
    assert "Let's just do it over the phone." in f.evidence and "turn 5" in f.evidence
    assert "_complete" in f.problem and "Escalated to CHW" in f.problem
    assert "by_phone" in f.fix_hint and "q5_choice" in f.fix_hint
    assert result.expected_terminal == "_complete" and result.actual_terminal == "escalated_chw_appointment"


def test_wrong_terminal_early_divergence_names_deciding_item(spec, threads):
    # thread_01 closed as medi_cal_active; pretend the caller said not active and has Kaiser.
    labels = {
        "q1_intent": L("q1_intent", "yes", 0.95, "Yes, please.", 0),
        "q2_active": L("q2_active", "inactive", 0.9, "it's not active", 1),
        "q3_coverage": L("q3_coverage", "true", 0.9, "I have Kaiser", 1),
        "q3_plan": L("q3_plan", "__close__", 0.9, "I have Kaiser", 1, value="Kaiser Permanente"),
    }
    findings, result = run(spec, by_id(threads, "thread_01"), labels)
    f = next(f for f in findings if f.check == "routing.wrong_terminal")
    assert result.expected_terminal == "other_coverage_no_action"
    assert "q2_active" in f.fix_hint and "inactive" in f.fix_hint
    assert "it's not active" in f.evidence


def test_wrong_terminal_on_low_confidence_path_is_uncertain(spec, threads):
    labels = dict(HAND_LABELS["thread_05"])
    labels["q5_choice"] = L("q5_choice", "by_phone", 0.5, "maybe phone", 5)
    findings, _ = run(spec, by_id(threads, "thread_05"), labels)
    wrong = next(f for f in findings if f.check == "routing.wrong_terminal")
    assert wrong.uncertain is True


# ---- low confidence / unclear ------------------------------------------------------


def test_low_confidence_answer(spec, threads):
    labels = dict(HAND_LABELS["thread_01"])
    labels["q2_active"] = L("q2_active", "active", 0.5, "I definitely have active coverage right now", 1)
    findings, _ = run(spec, by_id(threads, "thread_01"), labels)
    assert checks_of(findings) == ["routing.low_confidence_answer"]
    f = findings[0]
    assert (f.gate, f.owner, f.uncertain) == ("decisive_answers", "harness", True)
    assert f.evidence == '"I definitely have active coverage right now"'
    assert "turn 1" in f.fix_hint and "clarifying question" in f.fix_hint


def test_threshold_is_configurable(spec, threads):
    labels = dict(HAND_LABELS["thread_01"])
    labels["q2_active"] = L("q2_active", "active", 0.5, "x", 1)
    findings, _ = run(spec, by_id(threads, "thread_01"), labels, min_confidence=0.4)
    assert findings == []


def test_unclear_answer_then_terminal_is_review_not_fail(spec, threads):
    labels = dict(HAND_LABELS["thread_01"])
    labels["q2_active"] = L("q2_active", "unclear", 0.5, "that's weird", 1)
    findings, result = run(spec, by_id(threads, "thread_01"), labels)
    assert checks_of(findings) == ["routing.low_confidence_answer"]
    assert result.stopped_at == "q2_active" and result.expected_terminal is None


# ---- missing decisive answer --------------------------------------------------------


def test_terminal_without_decisive_answer_is_critical(spec, threads):
    labels = {"q1_intent": HAND_LABELS["thread_01"]["q1_intent"]}  # q2 not_discussed
    findings, result = run(spec, by_id(threads, "thread_01"), labels)
    assert checks_of(findings) == ["routing.terminal_without_answer"]
    f = findings[0]
    assert (f.severity, f.gate, f.uncertain) == ("critical", "correct_routing", False)
    assert "q2_active" in f.problem
    assert result.stopped_at == "q2_active"


def test_missing_answer_without_terminal_is_left_to_termination(spec, threads):
    labels = {"q1_intent": HAND_LABELS["thread_10"]["q1_intent"]}
    findings, result = run(spec, by_id(threads, "thread_10"), labels)
    assert findings == [] and result.stopped_at == "q2_active"


# ---- path divergence ----------------------------------------------------------------


def test_wrong_branch_asked(spec):
    thread = synthetic(spec, [
        ("agent", q(spec, "q1_intent")), ("caller", "Yes please."),
        ("agent", q(spec, "q2_active")), ("caller", "Yes, it's active."),
        ("agent", q(spec, "q3_coverage")), ("caller", "No."),
        ("agent", say(spec, "medi_cal_active")),
    ])
    labels = {
        "q1_intent": L("q1_intent", "yes", 0.95, "Yes please.", 0),
        "q2_active": L("q2_active", "active", 0.95, "Yes, it's active.", 1),
        "q3_coverage": L("q3_coverage", "false", 0.95, "No.", 2),
    }
    findings, _ = run(spec, thread, labels)
    assert checks_of(findings) == ["routing.path_divergence"]
    f = findings[0]
    assert (f.severity, f.gate, f.owner) == ("major", "correct_routing", "prompt")
    assert "q3_coverage" in f.problem and f.turn == 1


def test_skipped_item_without_answer(spec):
    thread = synthetic(spec, [
        ("agent", q(spec, "q1_intent")), ("caller", "Yes please."),
        ("agent", q(spec, "q2_active")), ("caller", "No, not active."),
        ("agent", q(spec, "q4_residency")), ("caller", "Yes."),
    ])
    labels = {
        "q1_intent": L("q1_intent", "yes", 0.95, "Yes please.", 0),
        "q2_active": L("q2_active", "inactive", 0.95, "No, not active.", 1),
        "q4_residency": L("q4_residency", "true", 0.95, "Yes.", 2),
    }
    findings, result = run(spec, thread, labels)
    assert result.stopped_at == "q3_coverage"
    assert checks_of(findings) == ["routing.path_divergence"]
    f = findings[0]
    assert f.owner == "engine" and "q3_coverage" in f.problem and "skipped" in f.problem


def test_skip_excused_when_caller_volunteered(spec):
    thread = synthetic(spec, [
        ("agent", q(spec, "q1_intent")), ("caller", "Yes please."),
        ("agent", q(spec, "q2_active")), ("caller", "No, not active, and no other insurance."),
        ("agent", q(spec, "q4_residency")), ("caller", "Yes."),
    ])
    labels = {
        "q1_intent": L("q1_intent", "yes", 0.95, "Yes please.", 0),
        "q2_active": L("q2_active", "inactive", 0.95, "No, not active", 1),
        "q3_coverage": L("q3_coverage", "false", 0.9, "no other insurance", 1),
        "q4_residency": L("q4_residency", "true", 0.95, "Yes.", 2),
    }
    findings, _ = run(spec, thread, labels)
    assert findings == []


# ---- captured values --------------------------------------------------------------------


def test_plan_not_named_is_minor_soft(spec, threads):
    labels = dict(HAND_LABELS["thread_09"])
    del labels["q3_plan"]
    findings, result = run(spec, by_id(threads, "thread_09"), labels)
    assert checks_of(findings) == ["routing.plan_not_captured"]
    f = findings[0]
    assert f.severity == "minor" and f.gate is None
    # capture only item: walk still reaches the disposition
    assert result.expected_terminal == "other_coverage_no_action"


def test_captured_mismatch(spec, threads):
    findings, _ = run(spec, by_id(threads, "thread_09"), HAND_LABELS["thread_09"], captured={"q3_plan": "Aetna"})
    f = next(f for f in findings if f.check == "routing.captured_mismatch")
    assert (f.severity, f.gate, f.uncertain) == ("critical", "decisive_answers", False)
    assert "agent recorded Aetna, caller said Kaiser Permanente at turn 2" in f.problem


def test_captured_match_variants(spec, threads):
    captured = {
        "q3_plan": "kaiser permanente",
        "q2_active": "Not active or unsure",
        "q3_coverage": "yes",
        "q1_intent": "yes",
    }
    findings, _ = run(spec, by_id(threads, "thread_09"), HAND_LABELS["thread_09"], captured=captured)
    assert findings == []


def test_captured_value_caller_never_gave(spec, threads):
    findings, _ = run(spec, by_id(threads, "thread_01"), HAND_LABELS["thread_01"], captured={"q4_residency": "true"})
    f = next(f for f in findings if f.check == "routing.captured_mismatch")
    assert "never" in f.problem


def test_captured_ignored_without_judge(spec, threads):
    findings, _ = run(spec, by_id(threads, "thread_09"), None, captured={"q3_plan": "Aetna"})
    assert checks_of(findings) == ["routing.judge_unavailable"]
