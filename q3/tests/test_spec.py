import pytest

from callcheck.spec import reachable_terminals, successors, walk

INACTIVE_NO_COVERAGE = {
    "q1_intent": "yes",
    "q2_active": "inactive",
    "q3_coverage": "false",
    "q4_residency": "true",
}


def test_item_order(spec):
    assert list(spec.items)[:3] == ["q1_intent", "q2_active", "q3_coverage"]


def test_goto_resolved_to_item_id(spec):
    assert spec.items["q1_intent"].question.scenario("yes").goto_item == "q2_active"
    assert spec.items["q5_packet"].question.scenario("no_or_lost").goto_item == "_complete"


def test_successors(spec):
    assert successors(spec, "q1_intent") == {"q2_active", "decline"}
    assert successors(spec, "q2_active") == {"medi_cal_active", "q3_coverage"}
    assert "_complete" in successors(spec, "decline")


@pytest.mark.parametrize(
    "answers, terminal, items",
    [
        (
            {"q1_intent": "yes", "q2_active": "active"},
            "medi_cal_active",
            ["q1_intent", "q2_active"],
        ),
        (
            {**INACTIVE_NO_COVERAGE, "q5_packet": "no_or_lost"},
            "_complete",
            ["q1_intent", "q2_active", "q3_coverage", "q4_residency", "q5_packet"],
        ),
        (
            {**INACTIVE_NO_COVERAGE, "q5_packet": "still_has", "q5_choice": "in_person"},
            "escalated_chw_appointment",
            ["q1_intent", "q2_active", "q3_coverage", "q4_residency", "q5_packet", "q5_choice"],
        ),
        (
            {"q1_intent": "yes", "q2_active": "inactive", "q3_coverage": "true",
             "q3_plan": "Kaiser Permanente"},
            "other_coverage_no_action",
            ["q1_intent", "q2_active", "q3_coverage", "q3_plan"],
        ),
        (
            {"q1_intent": "not_interested", "decline": "privacy_concern"},
            "declined_privacy_callback",
            ["q1_intent", "decline"],
        ),
    ],
)
def test_walk_terminals(spec, answers, terminal, items):
    path = walk(spec, answers)
    assert path.terminal == terminal
    assert path.items == items
    assert path.stopped_at is None


def test_walk_q3_plan_any_plan_closes(spec):
    base = {"q1_intent": "yes", "q2_active": "inactive", "q3_coverage": "true"}
    for plan in ("Other", "Molina", "__close__"):
        assert walk(spec, {**base, "q3_plan": plan}).terminal == "other_coverage_no_action"


def test_walk_decline_unknown_answer_uses_default_route(spec):
    path = walk(spec, {"q1_intent": "not_interested", "decline": "something_unmapped"})
    assert path.terminal == "_complete"


def test_walk_missing_answer_stops(spec):
    path = walk(spec, {"q1_intent": "yes"})
    assert path.terminal is None
    assert path.stopped_at == "q2_active"
    assert path.items == ["q1_intent", "q2_active"]


def test_walk_unclear_stops(spec):
    path = walk(spec, {"q1_intent": "yes", "q2_active": "unclear"})
    assert path.terminal is None
    assert path.stopped_at == "q2_active"


def test_reachable_terminals(spec):
    assert reachable_terminals(spec) == set(spec.outcomes) | {"_complete"}
