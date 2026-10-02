from callcheck.model import Finding
from callcheck.verdict import decide, gate_status


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
