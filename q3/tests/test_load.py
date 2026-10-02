import json

from callcheck.load import load_min_confidence
from callcheck.model import FlowSpec, Thread


def test_load_returns_spec_and_ten_threads(spec, threads):
    assert isinstance(spec, FlowSpec)
    assert len(threads) == 10
    assert all(isinstance(t, Thread) for t in threads)
    assert threads[0].thread_id == "thread_01"


def test_items_and_outcomes_parsed(spec):
    # The dataset defines 8 items (the card said 9; the data is authoritative).
    assert list(spec.items) == [
        "q1_intent",
        "q2_active",
        "q3_coverage",
        "q3_plan",
        "q4_residency",
        "q5_packet",
        "q5_choice",
        "decline",
    ]
    assert len(spec.outcomes) == 7
    assert spec.entry == "q1_intent"
    assert spec.outcomes["declined_privacy_callback"].flags == {
        "declined": True,
        "human_callback": True,
    }
    assert spec.outcomes["escalated_chw_appointment"].next_action == "handoff"


def test_marker_stripped_thread_01(threads):
    # thread_01: last caller message ends with the simulator marker.
    t = threads[0]
    last_caller = [m for m in t.messages if m.role == "caller"][-1]
    assert "[GOAL_ACHIEVED]" not in last_caller.text
    assert last_caller.text.endswith("thank you for checking for me!")
    assert last_caller.raw_text.endswith("[GOAL_ACHIEVED]")
    assert last_caller.had_marker is True
    assert t.simulator_goal_marker is True


def test_opener_is_turn_minus_one(threads):
    first = threads[0].messages[0]
    assert first.turn == -1
    assert first.role == "agent"


def test_unmarked_message_text_unchanged(threads):
    m = threads[0].messages[1]
    assert m.text == m.raw_text
    assert m.had_marker is False


def test_min_confidence_from_team_config():
    from conftest import DATA

    assert load_min_confidence(DATA) == 0.7


def test_min_confidence_fallback(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"team_config": {"configuration": {}}}))
    assert load_min_confidence(p, default=0.65) == 0.65
    p.write_text(json.dumps({}))
    assert load_min_confidence(p, default=0.65) == 0.65
