"""Proper termination: a terminal is reached, nothing substantive follows it, handoffs do not close.

Outcome next_action (end_call / handoff) is not visible in transcript text,
so it is not checked here.
"""

from __future__ import annotations

import re

from callcheck.align import Alignment, alignment_of, normalize
from callcheck.checks import Context
from callcheck.model import COMPLETE, Finding, Thread

GATE = "proper_termination"

# Closing phrases that end the conversation (matched on normalized text).
CLOSING = re.compile(
    r"\b(good ?bye|bye|take care|thank you for your time|thanks for your time|"
    r"have a (?:wonderful|great|good|nice|lovely) (?:day|one|evening))\b"
)
# Sentences that are only politeness, harmless after a terminal.
PLEASANTRY = re.compile(
    r"\b(youre (?:very |so |most )?welcome|my pleasure|glad|thank you|thanks|no problem)\b"
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCES.split(text) if s.strip()]


def _substantive(residual: str) -> bool:
    for sentence in _sentences(residual):
        if "?" in sentence:
            return True
        norm = normalize(sentence)
        if norm and not (CLOSING.search(norm) or PLEASANTRY.search(norm)):
            return True
    return False


def _no_terminal(thread: Thread) -> Finding:
    callers = thread.caller_messages()
    last_caller = callers[-1] if callers else None
    last = thread.messages[-1] if thread.messages else None
    turn = last.turn if last else None
    if last_caller is not None and last_caller.had_marker:
        return Finding(
            check="termination.truncated_by_simulator",
            severity="major",
            gate=GATE,
            turn=last_caller.turn,
            evidence=f'last caller message: "{last_caller.raw_text}"',
            problem=(
                "No outcome say or _complete handoff was delivered: the simulator emitted "
                "[GOAL_ACHIEVED] and the gym stopped: episode incomplete, success cannot be confirmed."
            ),
            fix_hint=(
                "Simulator stop condition fires before the agent reaches a disposition; require the "
                "episode to run until the agent emits an outcome or handoff."
            ),
            owner="simulator",
            uncertain=True,
        )
    return Finding(
        check="termination.no_terminal",
        severity="critical",
        gate=GATE,
        turn=turn,
        evidence=f'last message: "{last.text}"' if last else "empty thread",
        problem="The call ended without any outcome say or _complete handoff.",
        fix_hint=(
            "Engine must drive every call to an outcome or handoff; check for dropped turns or a "
            "missing default route on the last item asked."
        ),
        owner="engine",
    )


def _talks_past(alignment: Alignment) -> list[Finding]:
    findings = []
    for ta in alignment.turns:
        if ta.turn <= alignment.terminal_turn or not _substantive(ta.residual):  # type: ignore[operator]
            continue
        findings.append(
            Finding(
                check="termination.talks_past_terminal",
                severity="critical",
                gate=GATE,
                turn=ta.turn,
                evidence=f'"{ta.residual}"',
                problem=(
                    f"The call reached {alignment.actual_terminal} at turn {alignment.terminal_turn} but "
                    "the agent kept talking with new questions or content."
                ),
                fix_hint=(
                    "Engine must hand off / end_call after terminal; agent improvised next steps and "
                    "started collecting PII outside the renewal agent."
                    if "?" in ta.residual
                    else "Engine must hand off / end_call after terminal; agent improvised next steps."
                ),
                owner="engine",
            )
        )
    return findings


def _contradictory_close(alignment: Alignment) -> list[Finding]:
    ta = next(t for t in alignment.turns if t.turn == alignment.terminal_turn)
    match = CLOSING.search(normalize(ta.residual))
    if match is None:
        return []
    return [
        Finding(
            check="termination.contradictory_close",
            severity="major",
            gate=GATE,
            turn=ta.turn,
            evidence=f'"{ta.text}"',
            problem=(
                "The flow reached _complete, which hands off to the next agent, but the same message "
                f"closes the call ('{match.group(0)}'). Assumption: _complete means transfer to the "
                "next team agent (team_config connects fhcsd_triage_screener to MC_216), not end of call."
            ),
            fix_hint=(
                "Prompt: on _complete do not close the call; say a transition line into the renewal "
                "flow instead of goodbye or take care."
            ),
            owner="prompt",
        )
    ]


def check(ctx: Context) -> list[Finding]:
    alignment = alignment_of(ctx)
    if alignment.actual_terminal is None:
        return [_no_terminal(ctx.thread)]
    findings = _talks_past(alignment)
    if alignment.actual_terminal == COMPLETE:
        findings += _contradictory_close(alignment)
    return findings
