"""Script adherence: off script questions (soft) and question order against the graph.

Question order needs to know what the caller said: a skipped item is fine when the caller volunteered
its answer. routing.compare_paths does that with the judge's labels, so with a judgment this module
leaves order alone. Without one it still reports an order break, as uncertain (NEEDS_REVIEW).
"""

from __future__ import annotations

import re

from callcheck.align import Alignment, alignment_of
from callcheck.checks import Context
from callcheck.model import Finding, FlowSpec
from callcheck.spec import successors

_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def _off_script(alignment: Alignment) -> list[Finding]:
    findings = []
    for index, ta in enumerate(alignment.turns):
        if index == 0:
            continue  # the opener
        if alignment.terminal_turn is not None and ta.turn > alignment.terminal_turn:
            continue  # reported by termination.talks_past_terminal
        questions = [s for s in _SENTENCES.split(ta.residual) if "?" in s]
        if not questions:
            continue
        findings.append(
            Finding(
                check="script.off_script_question",
                severity="minor",
                gate=None,
                turn=ta.turn,
                evidence=" ".join(f'"{q.strip()}"' for q in questions),
                problem="The agent asked a question that matches no scripted question.",
                fix_hint=(
                    "Prompt: use the scripted question text verbatim; for clarification re-ask the "
                    "scripted question or add an approved clarification line to the item."
                ),
                owner="prompt",
            )
        )
    return findings


def _out_of_order(spec: FlowSpec, alignment: Alignment) -> list[Finding]:
    findings = []
    prev: str | None = None
    for turn, item in alignment.asked:
        if prev is None:
            ok = item == spec.entry
            expected = spec.entry
        else:
            ok = item == prev or item in successors(spec, prev)
            expected = ", ".join(sorted(successors(spec, prev)))
        if not ok:
            findings.append(
                Finding(
                    check="script.out_of_order",
                    severity="major",
                    gate="correct_routing",
                    turn=turn,
                    evidence=f"asked {item} after {prev or 'call start'}; graph allows {expected}",
                    problem=(
                        "Skipped or out of order question: the asked item is not a graph successor. No "
                        "judgment is available to tell whether the caller volunteered the skipped answers."
                    ),
                    fix_hint=(
                        "Engine: advance only along engine_config scenario gotos; check how the item "
                        "pointer moved past the expected question."
                    ),
                    owner="engine",
                    uncertain=True,  # the caller may have volunteered the skipped answers
                )
            )
        prev = item
    return findings


def check(ctx: Context) -> list[Finding]:
    alignment = alignment_of(ctx)
    findings = _off_script(alignment)
    if ctx.judgment is None:
        findings += _out_of_order(ctx.spec, alignment)
    return findings
