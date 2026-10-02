"""Missing scripted sentences in a say the agent did deliver (soft).

align matches a say by its anchor and treats later sentences as optional, so a closing like thread_01's
"That's great news ... right now." still counts as the medi_cal_active outcome without "Take care.".
This check reports what was dropped: one minor finding per thread listing every missing sentence.
"""

from __future__ import annotations

from callcheck.align import alignment_of, contains, node_sentences
from callcheck.checks import Context
from callcheck.model import Finding


def check(ctx: Context) -> list[Finding]:
    alignment = alignment_of(ctx)
    missing: list[tuple[int, str, str]] = []  # (turn, node label, sentence)
    for ta in alignment.turns:
        says: list[tuple[str, str]] = [(f"outcome {o}", ctx.spec.outcomes[o].say) for o in ta.outcome_says]
        for item_id, scen_id in ta.scenario_says:
            scenario = ctx.spec.items[item_id].question.scenario(scen_id)
            if scenario is not None and scenario.say:
                says.append((f"scenario {item_id}:{scen_id}", scenario.say))
        for label, say in says:
            missing += [(ta.turn, label, s) for s in node_sentences(say) if not contains(s, ta.text)]
    if not missing:
        return []
    evidence = "; ".join(f'turn {turn} {label} missing "{sentence}"' for turn, label, sentence in missing)
    return [
        Finding(
            check="say_coverage.partial_say",
            severity="minor",
            gate=None,
            turn=missing[0][0],
            evidence=evidence,
            problem="The agent delivered a scripted say but dropped part of it.",
            fix_hint="Prompt: deliver scripted say lines in full, verbatim, including the closing sentence.",
            owner="prompt",
        )
    ]
