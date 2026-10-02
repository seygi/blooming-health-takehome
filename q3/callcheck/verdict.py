"""Findings to a verdict.

FAIL: at least one confident hard failure (a gate, severity critical or major, not uncertain).
NEEDS_REVIEW: no confident failure, but at least one uncertain finding on a gate.
PASS: neither. Soft findings (no gate, or minor) never change the verdict.
"""

from __future__ import annotations

from callcheck.model import Finding, Verdict

GATES = ("correct_routing", "decisive_answers", "proper_termination", "clean_speech")
GateStatus = str  # "pass" | "fail" | "uncertain"

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
    return f"{tag} {f.gate}: {f.check} ({f.severity}, {turn}): {f.problem}"


def decide(findings: list[Finding]) -> tuple[Verdict, list[str]]:
    hard = sorted((f for f in findings if is_hard_failure(f)), key=severity_rank)
    unsure = sorted((f for f in findings if is_uncertain(f)), key=severity_rank)
    if hard:
        return "FAIL", [_reason(f, "FAIL") for f in hard] + [_reason(f, "REVIEW") for f in unsure]
    if unsure:
        return "NEEDS_REVIEW", [_reason(f, "REVIEW") for f in unsure]
    return "PASS", []


def gate_status(findings: list[Finding], unassessed: set[str] | None = None) -> dict[str, GateStatus]:
    """Per gate: fail on a confident hard failure, uncertain on an uncertain finding or when the gate could
    not be assessed at all (e.g. decisive_answers without a judgment), else pass."""
    status: dict[str, GateStatus] = {}
    for gate in GATES:
        on_gate = [f for f in findings if f.gate == gate]
        if any(is_hard_failure(f) for f in on_gate):
            status[gate] = "fail"
        elif any(f.uncertain for f in on_gate) or (unassessed and gate in unassessed):
            status[gate] = "uncertain"
        else:
            status[gate] = "pass"
    return status
