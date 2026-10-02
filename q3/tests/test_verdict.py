from callcheck.model import Finding
from callcheck.verdict import TASK_GATES, decide, decide_two, gate_status


def F(check="x.y", severity="major", gate="clean_speech", uncertain=False, turn=1):
    return Finding(check, severity, gate, turn, "ev", f"problem of {check}", "fix", "prompt", uncertain)


def test_pass_when_no_findings():
    assert decide([]) == ("PASS", [])


def test_soft_findings_never_flip():
    verdict, reasons = decide([F(gate=None, severity="major"), F(gate=None, severity="minor", uncertain=True)])
    assert verdict == "PASS" and reasons == []


def test_minor_on_gate_does_not_fail():
    assert decide([F(severity="minor")])[0] == "PASS"


def test_confident_hard_failure_fails():
    verdict, reasons = decide([F("hygiene.stage_direction")])
    assert verdict == "FAIL"
    assert len(reasons) == 1 and "hygiene.stage_direction" in reasons[0] and "clean_speech" in reasons[0]


def test_uncertain_only_needs_review():
    verdict, reasons = decide([F("termination.truncated_by_simulator", gate="proper_termination", uncertain=True)])
    assert verdict == "NEEDS_REVIEW"
    assert reasons[0].startswith("REVIEW proper_termination")


def test_fail_beats_uncertain_and_lists_both():
    verdict, reasons = decide([F("a.uncertain", uncertain=True), F("b.hard", severity="critical")])
    assert verdict == "FAIL"
    assert reasons[0].startswith("FAIL") and "b.hard" in reasons[0]
    assert reasons[1].startswith("REVIEW") and "a.uncertain" in reasons[1]


def test_uncertain_critical_is_not_fail():
    assert decide([F(severity="critical", uncertain=True)])[0] == "NEEDS_REVIEW"


def test_gate_status():
    findings = [
        F(gate="clean_speech"),
        F(gate="proper_termination", uncertain=True),
        F(gate="correct_routing", severity="minor"),
    ]
    assert gate_status(findings) == {
        "correct_routing": "pass",
        "decisive_answers": "pass",
        "proper_termination": "uncertain",
        "clean_speech": "fail",
    }


def test_gate_status_unassessed_is_uncertain_not_pass():
    status = gate_status([], unassessed={"decisive_answers"})
    assert status["decisive_answers"] == "uncertain"
    status = gate_status([F(gate="decisive_answers", severity="critical")], unassessed={"decisive_answers"})
    assert status["decisive_answers"] == "fail"


# ---- two results: task outcome and release ------------------------------------------


def test_speech_leak_fails_release_not_outcome_under_gate_policy():
    d = decide_two([F("hygiene.stage_direction", gate="clean_speech")], "gate")
    assert d.outcome == "PASS" and d.release == "FAIL"
    assert any("hygiene.stage_direction" in r for r in d.reasons)


def test_speech_leak_is_a_warning_under_soft_policy():
    d = decide_two([F("hygiene.stage_direction", gate="clean_speech")], "soft")
    assert d.outcome == "PASS" and d.release == "PASS"
    assert d.reasons == [] and d.warnings and d.warnings[0].startswith("WARN clean_speech")


def test_task_failure_fails_both():
    for policy in ("gate", "soft"):
        d = decide_two([F("termination.talks_past_terminal", gate="proper_termination", severity="critical")],
                       policy)
        assert (d.outcome, d.release) == ("FAIL", "FAIL")


def test_soft_major_is_a_release_warning_only():
    d = decide_two([F("termination.contradictory_close", gate=None, severity="major")], "gate")
    assert (d.outcome, d.release) == ("PASS", "PASS")
    assert d.warnings == ["WARN soft: termination.contradictory_close (major, turn 1): problem of "
                          "termination.contradictory_close"]


def test_review_outcome_with_speech_fail_is_release_fail():
    d = decide_two([F(gate="proper_termination", uncertain=True), F(gate="clean_speech")], "gate")
    assert (d.outcome, d.release) == ("NEEDS_REVIEW", "FAIL")


def test_outcome_ignores_speech_findings():
    assert decide([F(gate="clean_speech")], TASK_GATES)[0] == "PASS"


def test_gate_status_not_applicable_replaces_pass_only():
    status = gate_status([], not_applicable={"correct_routing"})
    assert status["correct_routing"] == "n/a"
    status = gate_status([F(gate="correct_routing")], not_applicable={"correct_routing"})
    assert status["correct_routing"] == "fail"
