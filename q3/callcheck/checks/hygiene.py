"""Leaked stage directions, system text, simulator markers and markdown in agent speech.

One finding per distinct pattern per thread, listing every turn it occurs in,
so an engineer sees "this template leaks on every turn" instead of six lines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from callcheck.checks import Context
from callcheck.model import Finding


@dataclass(frozen=True)
class _Pattern:
    check: str
    regexes: tuple[re.Pattern[str], ...]  # any of them is a hit; hits are merged into one finding
    problem: str
    fix_hint: str


# Words that narrate turn taking. A parenthetical at the end of a line containing one of them is a cue.
_TURN_TAKING = r"(?:waiting|wait|waits|let me know|go ahead|pause|pauses|response|respond|answer)"


_S2S_NOTE = (
    "GPT Realtime is speech to speech: there is no text stage to filter before audio, so the "
    "model would say this aloud. Fix it at the source and gate releases on this check. "
    "It may be text simulation scaffolding: verify it does not occur in voice mode."
)

PATTERNS = [
    _Pattern(
        check="hygiene.stage_direction",
        regexes=(
            # a whole line that is one parenthetical, e.g. "(Waiting for your response.)"
            re.compile(r"^[ \t]*\([^()\n]+\)[ \t]*$", re.MULTILINE),
            # a parenthetical closing a line that narrates turn taking: "...County? (Waiting for your answer.)"
            re.compile(r"\([^()\n]*\b" + _TURN_TAKING + r"\b[^()\n]*\)[ \t]*$", re.MULTILINE | re.I),
            # an asterisk action: "*waits for response*" (single asterisks; bold is the markdown check)
            re.compile(r"(?<!\*)\*[a-z][^*\n]*[^*\s]\*(?!\*)"),
            # a lowercase bracketed cue: "[pause]" (uppercase markers are the simulator_marker check)
            re.compile(r"\[[a-z][a-z ]*\]"),
        ),
        problem="Agent speech contains a stage direction (parenthetical, asterisk action or bracketed cue).",
        fix_hint=(
            "Find the template or engine step that appends '(Waiting for your response.)' style "
            "lines to agent turns and remove it; add an instruction never to narrate turn taking. " + _S2S_NOTE
        ),
    ),
    _Pattern(
        check="hygiene.system_text",
        regexes=(re.compile(r"\b(?:this|that) (?:was|is) the final message of this follow[ -]?up\b[.!]?", re.I),),
        problem="Agent speech contains system or meta text about the follow up itself.",
        fix_hint=(
            "Remove the 'final message of this follow-up' line from the prompt template or engine "
            "closing step; the outcome say is the whole closing. " + _S2S_NOTE
        ),
    ),
    _Pattern(
        check="hygiene.simulator_marker",
        regexes=(re.compile(r"\[[A-Z][A-Z_]{3,}\]"),),
        problem="Agent speech contains a bracketed simulator or control marker.",
        fix_hint=(
            "Keep simulator and control markers out of the agent context so the model cannot echo "
            "them. " + _S2S_NOTE
        ),
    ),
    _Pattern(
        check="hygiene.markdown",
        regexes=(
            re.compile(
                r"\*\*[^*\n]+\*\*|__[^_\n]+__|`[^`\n]+`|^[ \t]{0,3}#{1,6}[ \t]|^[ \t]*(?:[-*+]|\d+\.)[ \t]+\S",
                re.MULTILINE,
            ),
        ),
        problem="Agent speech contains markdown formatting.",
        fix_hint=(
            "Instruct the prompt to answer in plain spoken sentences, no formatting. " + _S2S_NOTE
        ),
    ),
]


def check(ctx: Context) -> list[Finding]:
    findings: list[Finding] = []
    for pattern in PATTERNS:
        turns: list[int] = []
        quotes: list[str] = []
        for message in ctx.thread.agent_messages():
            hits = [m.group(0).strip() for rx in pattern.regexes for m in rx.finditer(message.text)]
            if not hits:
                continue
            turns.append(message.turn)
            for hit in hits:
                if hit not in quotes:
                    quotes.append(hit)
        if not turns:
            continue
        quoted = "; ".join(f'"{q}"' for q in quotes)
        findings.append(
            Finding(
                check=pattern.check,
                severity="major",
                gate="clean_speech",
                turn=turns[0],
                evidence=f"{quoted} at turns {', '.join(str(t) for t in turns)}",
                problem=pattern.problem,
                fix_hint=pattern.fix_hint,
                owner="prompt",
            )
        )
    return findings
