"""Proper termination: a terminal is reached, nothing substantive follows it, handoffs do not close.

A closing that matches no outcome say is reported as terminal_unrecognized (NEEDS_REVIEW), not as a
confident no_terminal: the matcher may simply have missed a paraphrase.

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
# Whole phrases of politeness. A sentence is a pleasantry only when it is made of these phrases plus a
# short tail of filler words; "No problem, your coverage is now renewed." is not one.
_PLEASANTRY_PHRASE = re.compile(
    r"(?:youre|you are) (?:very |so |most )?welcome|my pleasure|no problem|not a problem|of course|"
    r"thank you(?: so much| very much)?|thanks(?: so much| a lot)?|"
    r"(?:im |i am |were |we are )?(?:so |really |very )?glad (?:i|we) (?:could|can|was able to|were able to)|"
    r"(?:im |i am |were |we are )?(?:so |really |very )?glad to hear(?: that| it)?|"
    r"have a (?:wonderful|great|good|nice|lovely|blessed) (?:day|one|evening|afternoon|weekend)|"
    r"good ?bye|bye|take care(?: of yourself)?|you too|all the best"
)
# Filler words allowed around the phrases, at most MAX_TAIL of them per sentence.
_TAIL_WORDS = frozenset(
    "and so too again then you to for with that this it all everything today sorted out get help "
    "helping assist calling very much really there okay ok alright well oh".split()
)
MAX_TAIL = 5
# Text that asks the caller for personal data.
_PII = re.compile(
    r"\b(?:name|date of birth|birth ?date|birthday|dob|address|social security|ssn|phone number|"
    r"email|medi ?cal (?:id|number)|case number|income)\b"
)
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENTENCES.split(text) if s.strip()]


def is_pleasantry(sentence: str) -> bool:
    """True when the whole sentence is politeness: pleasantry phrases plus at most MAX_TAIL filler words."""
    norm = normalize(sentence)
    pos, tail, phrases = 0, 0, 0
    while pos < len(norm):
        match = _PLEASANTRY_PHRASE.match(norm, pos)
        if match and (match.end() == len(norm) or norm[match.end()] == " "):
            phrases += 1
            pos = match.end() + 1
            continue
        end = norm.find(" ", pos)
        end = len(norm) if end == -1 else end
        if norm[pos:end] not in _TAIL_WORDS or tail >= MAX_TAIL:
            return False
        tail += 1
        pos = end + 1
    return phrases > 0


def _substantive(residual: str) -> bool:
    for sentence in _sentences(residual):
        if "?" in sentence:
            return True
        if normalize(sentence) and not is_pleasantry(sentence):
            return True
    return False


def _unrecognized_close(thread: Thread) -> Finding | None:
    """No terminal matched, but the agent's last message reads like a closing: a paraphrased outcome say the
    matcher missed, or an improvised close. Either way a human has to say which outcome was meant."""
    agents = thread.agent_messages()
    if not agents:
        return None
    last = agents[-1]
    if "?" in last.text:
        return None
    final = thread.messages[-1] is last
    previous = [m for m in thread.messages if m.role == "caller" and m.turn <= last.turn]
    caller_asked = bool(previous) and "?" in previous[-1].text
    if not (CLOSING.search(normalize(last.text)) or (final and not caller_asked)):
        return None
    return Finding(
        check="termination.terminal_unrecognized",
        severity="major",
        gate=GATE,
        turn=last.turn,
        evidence=f'last agent message: "{last.text}"',
        problem=(
            "The agent closed the call, but its closing text matches no outcome say or _complete scenario "
            "say, so the harness cannot tell which terminal was reached."
        ),
        fix_hint=(
            "agent closed with text that matches no outcome say; a human should confirm which outcome it "
            "intended; if paraphrase is intended, add it to the script or relax the matcher"
        ),
        owner="harness",
        uncertain=True,
    )


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
                    if _PII.search(normalize(ta.residual))
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
            gate=None,  # soft: gate 3 is talking past the terminal, and _complete semantics are assumed
            turn=ta.turn,
            evidence=f'"{ta.text}"',
            problem=(
                "The flow reached _complete, which hands off to the next agent, but the same message "
                f"closes the call ('{match.group(0)}'). Assumption: _complete means transfer to the "
                "next team agent (team_config connects fhcsd_triage_screener to MC_216), not end of call. "
                "The data shows the agent ran standalone, so this is reported as a quality warning, not a "
                "gate failure."
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
        unrecognized = _unrecognized_close(ctx.thread)
        return [unrecognized if unrecognized is not None else _no_terminal(ctx.thread)]
    findings = _talks_past(alignment)
    if alignment.actual_terminal == COMPLETE:
        findings += _contradictory_close(alignment)
    return findings
