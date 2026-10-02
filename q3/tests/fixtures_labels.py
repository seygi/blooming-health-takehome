"""Hand labels for tests, NOT model output.

Written by reading each transcript in q3/data/gym_agent_conversations.json, following the judge's
own rules (caller lines only, every item labelled, first_available_turn is the earliest caller turn
that already held the information, a plain yes to q1 answers q1 only). Items left out are
not_discussed (FakeJudge fills them with confidence 1.0).

They let routing, verdict and report be tested end to end without a network or a judgment cache.
"""

from __future__ import annotations

from callcheck.judge import FakeJudge, ItemJudgment


def L(item: str, answer: str, conf: float, evidence: str, turn: int, first: int | None = None,
      value: str | None = None) -> ItemJudgment:
    return ItemJudgment(item, answer, conf, evidence, turn, turn if first is None else first, value)


def _labels(*items: ItemJudgment) -> dict[str, ItemJudgment]:
    return {i.item_id: i for i in items}


HAND_LABELS: dict[str, dict[str, ItemJudgment]] = {
    "thread_01": _labels(
        L("q1_intent", "yes", 0.95, "Yes, please.", 0),
        L("q2_active", "active", 0.95, "I definitely have active coverage right now", 1),
    ),
    "thread_02": _labels(
        L("q1_intent", "yes", 0.95, "Yes, please, I definitely need help with that.", 0),
        # "I don't think" is a hedge, but q2 `inactive` explicitly covers "unsure".
        L("q2_active", "inactive", 0.9, "No, I don't think it's active right now", 1),
        L("q3_coverage", "false", 0.95, "No, I don't have any other insurance at all.", 2),
        L("q4_residency", "true", 0.95, "Yeah, I still live here in San Diego.", 3),
        L("q5_packet", "no_or_lost", 0.95, "No, I never actually got it in the mail, so I don't have it.", 4,
          first=3),
    ),
    "thread_03": _labels(
        L("q1_intent", "yes", 0.95, "Yeah, I'd definitely like some help with that.", 0),
        L("q2_active", "inactive", 0.95, "No, it's definitely not active", 1, first=0),
        L("q3_coverage", "false", 0.95, "No, I don't have any other insurance at all.", 2, first=0),
        L("q4_residency", "false", 0.95, "I actually moved out of the county", 3, first=1),
    ),
    "thread_04": _labels(
        L("q1_intent", "yes", 0.95, "Yes, please", 0),
        L("q2_active", "active", 0.95, "Yes, I'm 100% sure it's active", 2, first=1),
    ),
    "thread_05": _labels(
        L("q1_intent", "yes", 0.95, "Yeah, I'd definitely like some help with that.", 0),
        L("q2_active", "inactive", 0.95, "No, it's not active right now.", 1),
        L("q3_coverage", "false", 0.95, "No, I don't have any other insurance at all.", 2, first=1),
        L("q4_residency", "true", 0.95, "Yeah, I still live here in San Diego.", 3),
        L("q5_packet", "still_has", 0.95, "Yeah, I got it, and I still have it", 4),
        L("q5_choice", "in_person", 0.95, "I'd definitely prefer to come in for an in-person appointment", 5),
    ),
    "thread_06": _labels(
        L("q1_intent", "not_interested", 0.95, "I'm not interested in renewing over the phone right now.", 0),
        L("q4_residency", "false", 0.9, "I actually moved out of San Diego County recently", 1),
        L("decline", "reveals_out_of_county", 0.95, "I actually moved out of San Diego County recently", 1,
          value="moved out of San Diego County recently"),
    ),
    "thread_07": _labels(
        L("q1_intent", "yes", 0.95, "Yes, I'd really appreciate your help with that.", 0),
        L("q2_active", "inactive", 0.95, "No, it's not active right now.", 1),
        L("q3_coverage", "false", 0.95, "No, I don't have any other insurance.", 2, first=1),
        L("q4_residency", "true", 0.95, "I'm still in San Diego County.", 3, first=2),
        L("q5_packet", "still_has", 0.95, "Yeah, I still have it right here in front of me.", 4, first=2),
        # Volunteered before q5_choice was ever asked; explicit about phone, so 0.85 not 0.95.
        L("q5_choice", "by_phone", 0.85, "Can we just go ahead and finish it over the phone now?", 4, first=3),
    ),
    "thread_08": _labels(
        L("q1_intent", "yes", 0.95, "Yes, I definitely want help with that.", 0),
        L("q2_active", "inactive", 0.95, "No, it's not active right now.", 1),
        L("q3_coverage", "false", 0.95, "No, I don't have any other insurance at all.", 2, first=1),
        L("q4_residency", "true", 0.95, "Yeah, I still live here in San Diego.", 3, first=0),
        L("q5_packet", "no_or_lost", 0.95, "No, like I said, I never got it in the mail.", 4, first=0),
    ),
    "thread_09": _labels(
        L("q1_intent", "yes", 0.95, "Yeah, I'd really like some help with that.", 0),
        L("q2_active", "inactive", 0.95, "No, it's definitely not active", 1, first=0),
        L("q3_coverage", "true", 0.95, "I'm covered by Kaiser Permanente.", 2, first=0),
        L("q3_plan", "__close__", 0.95, "I'm covered by Kaiser Permanente.", 2, first=0, value="Kaiser Permanente"),
    ),
    "thread_10": _labels(
        L("q1_intent", "yes", 0.95, "Yes, I'd definitely like your help with that.", 0),
        L("q2_active", "inactive", 0.9, "I'm not currently active", 0),
        L("q3_coverage", "false", 0.95, "I don't have any other insurance", 0),
        L("q4_residency", "true", 0.95, "I'm still in San Diego County.", 0),
        L("q5_packet", "still_has", 0.9, "I have the yellow packet right here", 0),
        L("q5_choice", "by_phone", 0.85, "let's do this over the phone now.", 0),
    ),
}

# Verdicts the harness must give under HAND_LABELS and today's checks, with the deciding reason.
EXPECTED_VERDICTS: dict[str, str] = {
    "thread_01": "PASS",  # routed active -> medi_cal_active, clean close
    "thread_02": "FAIL",  # talks past _complete, starts collecting PII
    "thread_03": "FAIL",  # leaked "(Please let me know your answer.)"
    "thread_04": "FAIL",  # leaked "This was the final message of this follow-up."
    "thread_05": "FAIL",  # leaked "(Waiting for your response.)"
    "thread_06": "FAIL",  # leaked "That was the final message of this follow-up."
    "thread_07": "NEEDS_REVIEW",  # simulator stopped before any terminal; nothing else wrong
    "thread_08": "FAIL",  # closes the call on a _complete handoff
    "thread_09": "FAIL",  # leaked "(Waiting for your answer.)"
    "thread_10": "FAIL",  # leaked "(Waiting for your response.)"
}

EXPECTED_TERMINALS: dict[str, str] = {
    "thread_01": "medi_cal_active",
    "thread_02": "_complete",
    "thread_03": "escalated_chw_out_of_county",
    "thread_04": "medi_cal_active",
    "thread_05": "escalated_chw_appointment",
    "thread_06": "escalated_chw_out_of_county",
    "thread_07": "_complete",
    "thread_08": "_complete",
    "thread_09": "other_coverage_no_action",
    "thread_10": "_complete",
}


def hand_judge() -> FakeJudge:
    return FakeJudge(HAND_LABELS, model="hand-labels")
