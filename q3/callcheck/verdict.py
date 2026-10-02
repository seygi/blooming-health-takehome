"""Findings to two verdicts per thread: task outcome and release.

Task outcome uses the task gates only (correct_routing, decisive_answers, proper_termination): did the
call do its job? Release adds clean_speech under a speech policy: with "gate" (default for voice) a
speech leak blocks release; with "soft" it becomes a release warning and release equals the outcome.

Per verdict:
FAIL: at least one confident hard failure (a counted gate, severity critical or major, not uncertain).
NEEDS_REVIEW: no confident failure, but at least one uncertain finding on a counted gate.
PASS: neither. Soft findings (no gate, or minor) never change a verdict; soft majors become warnings.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from callcheck.model import Finding, Verdict

TASK_GATES = ("correct_routing", "decisive_answers", "proper_termination")
RELEASE_GATE = "clean_speech"
GATES = TASK_GATES + (RELEASE_GATE,)
GateStatus = str  # "pass" | "fail" | "uncertain" | "n/a"
SpeechPolicy = Literal["gate", "soft"]
SPEECH_POLICIES = ("gate", "soft")

_HARD = ("critical", "major")
_RANK = {"critical": 0, "major": 1, "minor": 2}


def is_hard_failure(f: Finding) -> bool:
    return f.gate is not None and f.severity in _HARD and not f.uncertain


def is_uncertain(f: Finding) -> bool:
    return f.gate is not None and f.uncertain


def severity_rank(f: Finding) -> int:
    return _RANK.get(f.severity, 3)


def _reason(f: Finding, tag: str) -> str:
    turn = f"turn {f.turn}" if f.turn is not None else "no turn"
    where = f.gate if f.gate else "soft"
    return f"{tag} {where}: {f.check} ({f.severity}, {turn}): {f.problem}"


def decide(findings: list[Finding], gates: Iterable[str] = GATES) -> tuple[Verdict, list[str]]:
    counted = set(gates)
    scoped = [f for f in findings if f.gate in counted]
    hard = sorted((f for f in scoped if is_hard_failure(f)), key=severity_rank)
    unsure = sorted((f for f in scoped if is_uncertain(f)), key=severity_rank)
    if hard:
        return "FAIL", [_reason(f, "FAIL") for f in hard] + [_reason(f, "REVIEW") for f in unsure]
    if unsure:
        return "NEEDS_REVIEW", [_reason(f, "REVIEW") for f in unsure]
    return "PASS", []


@dataclass
class Decision:
    outcome: Verdict
    release: Verdict
    reasons: list[str] = field(default_factory=list)  # every reason that counts toward the release verdict
    warnings: list[str] = field(default_factory=list)  # release warnings: soft majors, speech under "soft"


def decide_two(findings: list[Finding], speech_policy: SpeechPolicy = "gate") -> Decision:
    outcome, outcome_reasons = decide(findings, TASK_GATES)
    speech = [f for f in findings if f.gate == RELEASE_GATE]
    warnings = [_reason(f, "WARN") for f in findings if f.gate is None and f.severity in _HARD]
    if speech_policy == "gate":
        release, reasons = decide(findings, GATES)
    else:
        release, reasons = outcome, outcome_reasons
        warnings = [_reason(f, "WARN") for f in speech if f.severity in _HARD] + warnings
    return Decision(outcome=outcome, release=release, reasons=reasons, warnings=warnings)


def gate_status(
    findings: list[Finding],
    unassessed: set[str] | None = None,
    not_applicable: set[str] | None = None,
) -> dict[str, GateStatus]:
    """Per gate: fail on a confident hard failure, uncertain on an uncertain finding or when the gate could
    not be assessed (e.g. decisive_answers without a judgment), n/a instead of pass when the gate has no
    meaning for this call (correct_routing when no terminal was reached), else pass."""
    status: dict[str, GateStatus] = {}
    for gate in GATES:
        on_gate = [f for f in findings if f.gate == gate]
        if any(is_hard_failure(f) for f in on_gate):
            status[gate] = "fail"
        elif any(f.uncertain for f in on_gate) or (unassessed and gate in unassessed):
            status[gate] = "uncertain"
        elif not_applicable and gate in not_applicable:
            status[gate] = "n/a"
        else:
            status[gate] = "pass"
    return status
