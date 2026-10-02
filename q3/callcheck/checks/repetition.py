"""Soft: same item asked twice, or asked after the caller already volunteered the answer."""

from __future__ import annotations

from callcheck.align import alignment_of
from callcheck.checks import Context
from callcheck.model import Finding


def _asked_twice(asked: list[tuple[int, str]]) -> list[Finding]:
    turns_by_item: dict[str, list[int]] = {}
    for turn, item in asked:
        turns_by_item.setdefault(item, []).append(turn)
    findings = []
    for item, turns in turns_by_item.items():
        if len(turns) < 2:
            continue
        findings.append(
            Finding(
                check="repetition.asked_twice",
                severity="minor",
                gate=None,
                turn=turns[1],
                evidence=f"{item} asked at turns {', '.join(str(t) for t in turns)}",
                problem=f"The scripted question for {item} was asked more than once.",
                fix_hint=(
                    "Prompt: when the caller's reply is ambiguous, ask a short targeted clarification "
                    "instead of repeating the full scripted question."
                ),
                owner="prompt",
            )
        )
    return findings


def _volunteered(asked: list[tuple[int, str]], judgment: object) -> list[Finding]:
    items = getattr(judgment, "items", None)
    if not hasattr(items, "get"):
        return []
    first_ask: dict[str, int] = {}
    for turn, item in asked:
        first_ask.setdefault(item, turn)
    findings = []
    for item, asked_turn in first_ask.items():
        available = getattr(items.get(item), "first_available_turn", None)
        if not isinstance(available, int) or isinstance(available, bool):
            continue
        # Caller turn t is spoken before agent turn t, so info available at
        # caller turn <= the asking agent turn was already on the table.
        if available > asked_turn:
            continue
        # Quote from the turn where the info first appeared. The judge's `evidence` quote may be
        # from a later turn, so it is never used here: no quote beats a quote from the wrong turn.
        quote = getattr(items.get(item), "first_available_quote", "") or ""
        evidence = f"{item} asked at turn {asked_turn}; caller gave it at turn {available}"
        if quote:
            evidence += f': "{quote}"'
        findings.append(
            Finding(
                check="repetition.asked_after_volunteered",
                severity="minor",
                gate=None,
                turn=asked_turn,
                evidence=evidence,
                problem=f"The agent asked {item} although the caller had already volunteered the answer.",
                fix_hint=(
                    "Prompt: before asking a scripted question, check whether earlier caller turns already "
                    "answer it; if so, confirm briefly ('You mentioned X, is that right?') or skip ahead."
                ),
                owner="prompt",
            )
        )
    return findings


def check(ctx: Context) -> list[Finding]:
    asked = alignment_of(ctx).asked
    findings = _asked_twice(asked)
    if ctx.judgment is not None:
        findings += _volunteered(asked, ctx.judgment)
    return findings
