"""Test expectations under the hand labels. The labels themselves live in callcheck.hand_labels."""

from __future__ import annotations

from callcheck.hand_labels import HAND_LABELS, L, hand_judge  # noqa: F401  (re-export for tests)

# (task outcome, release under speech policy "gate") the harness must give under HAND_LABELS, with the reason.
EXPECTED_VERDICTS: dict[str, tuple[str, str]] = {
    "thread_01": ("PASS", "PASS"),  # routed active -> medi_cal_active, clean close
    "thread_02": ("FAIL", "FAIL"),  # talks past _complete, starts collecting PII
    "thread_03": ("PASS", "FAIL"),  # leaked "(Please let me know your answer.)"
    "thread_04": ("PASS", "FAIL"),  # leaked "This was the final message of this follow-up."
    "thread_05": ("PASS", "FAIL"),  # leaked "(Waiting for your response.)"
    "thread_06": ("PASS", "FAIL"),  # leaked "That was the final message of this follow-up."
    "thread_07": ("NEEDS_REVIEW", "NEEDS_REVIEW"),  # simulator stopped before any terminal
    "thread_08": ("PASS", "PASS"),  # closes on a _complete handoff: release warning only
    "thread_09": ("NEEDS_REVIEW", "FAIL"),  # simulator stopped; leaked "(Waiting for your answer.)"
    "thread_10": ("NEEDS_REVIEW", "FAIL"),  # simulator stopped; leaked "(Waiting for your response.)"
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
