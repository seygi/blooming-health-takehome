import pytest

from callcheck.align import align, normalize


def by_id(threads, thread_id):
    return next(t for t in threads if t.thread_id == thread_id)


def test_normalize_unifies_quotes_dashes_and_punctuation():
    assert normalize("That\u2019s great news \u2014 it\u2019s Medi-Cal!") == "thats great news its medi cal"


@pytest.mark.parametrize(
    "thread_id, terminal, turn",
    [
        ("thread_01", "medi_cal_active", 1),  # partial outcome say, no "Take care."
        ("thread_03", "escalated_chw_out_of_county", 3),
        ("thread_04", "medi_cal_active", 2),
        ("thread_05", "escalated_chw_appointment", 5),
        ("thread_06", "escalated_chw_out_of_county", 1),  # decline path
        ("thread_02", "_complete", 4),  # q5_packet no_or_lost say
        ("thread_08", "_complete", 4),
        ("thread_07", None, None),
        ("thread_09", None, None),
        ("thread_10", None, None),
    ],
)
def test_actual_terminal(spec, threads, thread_id, terminal, turn):
    a = align(spec, by_id(threads, thread_id))
    assert a.actual_terminal == terminal
    assert a.terminal_turn == turn


def test_thread_09_scenario_say_is_not_an_outcome(spec, threads):
    a = align(spec, by_id(threads, "thread_09"))
    assert a.asked == [(-1, "q1_intent"), (0, "q2_active"), (1, "q3_coverage"), (2, "q3_plan")]
    last = a.turns[-1]
    assert last.turn == 2
    assert last.outcome_says == []
    assert last.scenario_says == [("q3_coverage", "true")]
    assert last.questions == ["q3_plan"]
    # both nodes score high on the raw text; the longer scenario say wins the span
    assert last.scores["o:other_coverage_no_action"] >= 0.8


def test_thread_04_off_script_reask_is_not_a_question_match(spec, threads):
    a = align(spec, by_id(threads, "thread_04"))
    assert [item for _, item in a.asked].count("q2_active") == 1
    assert (0, "q2_active") in a.asked
    turn1 = next(t for t in a.turns if t.turn == 1)
    assert turn1.questions == []
    assert "?" in turn1.residual


def test_thread_01_partial_outcome_and_residual(spec, threads):
    a = align(spec, by_id(threads, "thread_01"))
    turn1 = next(t for t in a.turns if t.turn == 1)
    assert turn1.outcome_says == ["medi_cal_active"]
    assert turn1.residual == ""
    turn0 = next(t for t in a.turns if t.turn == 0)
    assert turn0.questions == ["q2_active"]
    assert turn0.residual == "Please let me know."


def test_thread_05_full_path_asked_in_order(spec, threads):
    a = align(spec, by_id(threads, "thread_05"))
    assert [item for _, item in a.asked] == [
        "q1_intent",
        "q2_active",
        "q3_coverage",
        "q4_residency",
        "q5_packet",
        "q5_choice",
    ]
    last = a.turns[-1]
    assert last.outcome_says == ["escalated_chw_appointment"]
    assert last.residual == ""


def test_thread_02_residual_after_complete(spec, threads):
    a = align(spec, by_id(threads, "thread_02"))
    turn4 = next(t for t in a.turns if t.turn == 4)
    assert turn4.scenario_says == [("q5_packet", "no_or_lost")]
    assert "take care" in turn4.residual.lower()
    turn5 = next(t for t in a.turns if t.turn == 5)
    assert turn5.questions == [] and "?" in turn5.residual


def test_thread_06_decline_question_matched(spec, threads):
    a = align(spec, by_id(threads, "thread_06"))
    assert a.asked == [(-1, "q1_intent"), (0, "decline")]
