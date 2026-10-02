"""Evaluate threads and render the report (plain ASCII text, 120 columns, and JSON)."""

from __future__ import annotations

import json
import textwrap
import unicodedata
from dataclasses import asdict
from pathlib import Path

from callcheck.align import align
from callcheck.checks import REGISTRY, Context
from callcheck.judge import Judge
from callcheck.model import Finding, FlowSpec, Thread, ThreadReport
from callcheck.routing import MIN_CONFIDENCE, check_routing
from callcheck.verdict import GATES, decide, gate_status, severity_rank

WIDTH = 118
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "out" / "report.json"

_GATE_HEAD = {"correct_routing": "route", "decisive_answers": "answer", "proper_termination": "term",
              "clean_speech": "speech"}
_GATE_CELL = {"pass": "pass", "fail": "FAIL", "uncertain": "review"}
_ASCII = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
          "\u2026": "...", "\u00a0": " "}


def evaluate(
    spec: FlowSpec,
    thread: Thread,
    judge: Judge | None,
    captured: dict[str, str] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> ThreadReport:
    alignment = align(spec, thread)
    judgment = judge.judge(spec, thread) if judge is not None else None
    ctx = Context(spec=spec, thread=thread, alignment=alignment, judgment=judgment)
    findings: list[Finding] = []
    for check in REGISTRY:
        findings += check(ctx)
    routing_findings, result = check_routing(spec, thread, alignment, judgment, captured, min_confidence)
    findings += routing_findings
    findings.sort(key=lambda f: (severity_rank(f), f.turn if f.turn is not None else 99, f.check))
    verdict, reasons = decide(findings)
    unassessed = {"decisive_answers"} if judgment is None else set()
    return ThreadReport(
        thread_id=thread.thread_id,
        verdict=verdict,
        findings=findings,
        reasons=reasons,
        expected_terminal=result.expected_terminal,
        actual_terminal=result.actual_terminal,
        simulator_goal_marker=thread.simulator_goal_marker,
        gates=gate_status(findings, unassessed),
        expected_path=result.expected_path,
        judge_source=judgment.source if judgment is not None else "none",
    )


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def ascii_text(text: str) -> str:
    for src, dst in _ASCII.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "replace").decode("ascii")


def _wrap(text: str, indent: str, first: str | None = None) -> list[str]:
    text = " ".join(ascii_text(text).split())
    return textwrap.wrap(text, WIDTH, initial_indent=first if first is not None else indent,
                         subsequent_indent=indent) or [first or indent]


def _terminal_cell(r: ThreadReport) -> str:
    expected = r.expected_terminal or "?"
    actual = r.actual_terminal or "none"
    if r.expected_terminal and r.expected_terminal == r.actual_terminal:
        return f"{expected} (match)"
    return f"{expected} -> {actual}"


def summary_table(reports: list[ThreadReport]) -> list[str]:
    tw = max([len("expected -> actual terminal")] + [len(_terminal_cell(r)) for r in reports])
    tid = max([len("thread")] + [len(r.thread_id) for r in reports])
    head = (f"{'thread':<{tid}}  {'verdict':<12}  {'expected -> actual terminal':<{tw}}  "
            + " ".join(f"{_GATE_HEAD[g]:<6}" for g in GATES) + "  sim marker")
    lines = [head, "-" * len(head)]
    for r in reports:
        gates = " ".join(f"{_GATE_CELL.get(r.gates.get(g, ''), '?'):<6}" for g in GATES)
        marker = "yes" if r.simulator_goal_marker else "no"
        lines.append(f"{r.thread_id:<{tid}}  {r.verdict:<12}  {_terminal_cell(r):<{tw}}  {gates}  {marker}")
    lines.append("gates: route = correct routing, answer = decisive answers, term = proper termination, "
                 "speech = clean speech")
    lines.append("cells: pass, FAIL (confident), review (harness not confident). "
                 "sim marker = simulator [GOAL_ACHIEVED], not ground truth")
    return lines


def fix_groups(reports: list[ThreadReport]) -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for r in reports:
        for f in r.findings:
            g = groups.setdefault((f.check, f.fix_hint), {
                "check": f.check, "fix_hint": f.fix_hint, "severity": f.severity, "owner": f.owner,
                "gate": f.gate, "threads": [], "uncertain": True, "example": None,
            })
            if severity_rank(f) < {"critical": 0, "major": 1, "minor": 2}[g["severity"]]:
                g["severity"] = f.severity
            if r.thread_id not in g["threads"]:
                g["threads"].append(r.thread_id)
            g["uncertain"] = g["uncertain"] and f.uncertain
            if g["example"] is None:
                g["example"] = {"thread": r.thread_id, "turn": f.turn, "evidence": f.evidence}
    rank = {"critical": 0, "major": 1, "minor": 2}
    return sorted(groups.values(), key=lambda g: (-len(g["threads"]), rank[g["severity"]], g["check"]))


def _fix_lines(n: int, g: dict) -> list[str]:
    tags = f"{g['severity']}, owner {g['owner']}" + (", uncertain" if g["uncertain"] else "")
    lines = _wrap(f"{len(g['threads'])} thread(s): {', '.join(g['threads'])}", "      ",
                  f"{n:>2}. {g['check']}  [{tags}]  ")
    lines += _wrap(g["fix_hint"], "      ", "    fix: ")
    ex = g["example"]
    turn = f" turn {ex['turn']}" if ex["turn"] is not None else ""
    lines += _wrap(ex["evidence"], "      ", f"    e.g. {ex['thread']}{turn}: ")
    return lines


def fix_list(reports: list[ThreadReport]) -> list[str]:
    """Hard gate fixes first (they decide verdicts), then soft ones; each ranked by threads affected,
    then severity."""
    groups = fix_groups(reports)
    hard = [g for g in groups if g["gate"]]
    soft = [g for g in groups if not g["gate"]]
    lines = ["FIX LIST (same check and fix grouped across threads; ranked by threads affected, then severity)"]
    if not groups:
        return lines + ["  nothing to fix"]
    lines.append("Hard gates (these decide FAIL / NEEDS_REVIEW):")
    lines += [ln for n, g in enumerate(hard, 1) for ln in _fix_lines(n, g)] or ["  none"]
    lines.append("Soft (quality only, never change the verdict):")
    lines += [ln for n, g in enumerate(soft, len(hard) + 1) for ln in _fix_lines(n, g)] or ["  none"]
    return lines


def thread_detail(r: ThreadReport) -> list[str]:
    if not r.expected_path:
        path = "unknown (no judgment)"
    else:
        end = r.expected_terminal or f"stopped, no usable answer at {r.expected_path[-1]}"
        path = " > ".join(r.expected_path + [end])
    lines = [
        f"=== {r.thread_id}  {r.verdict}  (judge: {r.judge_source})",
        f"  expected path: {path}",
        f"  actual terminal: {r.actual_terminal or 'none'}",
    ]
    if r.reasons:
        lines.append("  reasons:")
        for reason in r.reasons:
            lines += _wrap(reason, "      ", "    - ")
    if not r.findings:
        lines.append("  findings: none")
        return lines
    lines.append("  findings:")
    for f in r.findings:
        turn = f"turn {f.turn}" if f.turn is not None else "no turn"
        gate = f"gate {f.gate}" if f.gate else "soft"
        flag = "  UNCERTAIN" if f.uncertain else ""
        lines.append(f"    [{f.severity}] {f.check}  {turn}  {gate}  owner {f.owner}{flag}")
        lines += _wrap(f.evidence, "        ", "      evidence: ")
        lines += _wrap(f.problem, "        ", "      problem: ")
        lines += _wrap(f.fix_hint, "        ", "      fix: ")
    return lines


def simulator_agreement(reports: list[ThreadReport]) -> dict:
    agree, disagree, review = [], [], []
    for r in reports:
        if r.verdict == "NEEDS_REVIEW":
            review.append(r.thread_id)
        elif (r.verdict == "PASS") == r.simulator_goal_marker:
            agree.append(r.thread_id)
        else:
            disagree.append(r.thread_id)
    return {"agree": agree, "disagree": disagree, "not_comparable": review}


def counts(reports: list[ThreadReport]) -> dict[str, int]:
    out = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}
    for r in reports:
        out[r.verdict] += 1
    return out


def footer(reports: list[ThreadReport]) -> list[str]:
    c = counts(reports)
    a = simulator_agreement(reports)
    by_id = {r.thread_id: r for r in reports}
    lines = [
        f"TOTAL {len(reports)} threads: PASS {c['PASS']}  FAIL {c['FAIL']}  NEEDS_REVIEW {c['NEEDS_REVIEW']}",
        "Simulator [GOAL_ACHIEVED] marker vs harness verdict (comparison only, not ground truth):",
        f"  agree {len(a['agree'])}, disagree {len(a['disagree'])}, not comparable (NEEDS_REVIEW) "
        f"{len(a['not_comparable'])}",
    ]
    if a["disagree"]:
        parts = [f"{t} (marker {'yes' if by_id[t].simulator_goal_marker else 'no'}, harness {by_id[t].verdict})"
                 for t in a["disagree"]]
        lines += _wrap(", ".join(parts), "    ", "  disagreements: ")
    return lines


def render_text(reports: list[ThreadReport]) -> str:
    sources = sorted({r.judge_source for r in reports})
    lines = [f"callcheck: {len(reports)} thread(s), judge source: {', '.join(sources) or 'none'}", ""]
    lines += summary_table(reports) + [""] + fix_list(reports) + [""]
    for r in reports:
        lines += thread_detail(r) + [""]
    lines += footer(reports)
    return "\n".join(ascii_text(line) for line in lines) + "\n"


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(reports: list[ThreadReport]) -> dict:
    return {
        "counts": counts(reports),
        "simulator_agreement": {"note": "comparison only, not ground truth", **simulator_agreement(reports)},
        "fix_list": fix_groups(reports),
        "threads": [asdict(r) for r in reports],
    }


def write_json(reports: list[ThreadReport], path: Path | str = DEFAULT_JSON) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_json(reports), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def exit_code(reports: list[ThreadReport]) -> int:
    verdicts = {r.verdict for r in reports}
    if "FAIL" in verdicts:
        return 1
    if "NEEDS_REVIEW" in verdicts:
        return 2
    return 0
