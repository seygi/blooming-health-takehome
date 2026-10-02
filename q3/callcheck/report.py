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
from callcheck.verdict import GATES, TASK_GATES, SpeechPolicy, decide_two, gate_status, severity_rank

WIDTH = 118
DEFAULT_JSON = Path(__file__).resolve().parents[1] / "out" / "report.json"

_GATE_HEAD = {"correct_routing": "route", "decisive_answers": "answer", "proper_termination": "term",
              "clean_speech": "speech"}
_GATE_CELL = {"pass": "pass", "fail": "FAIL", "uncertain": "review", "n/a": "n/a"}
NO_TERMINAL_REASON = "agent never reached a terminal"
_ASCII = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
          "\u2026": "...", "\u00a0": " "}


def evaluate(
    spec: FlowSpec,
    thread: Thread,
    judge: Judge | None,
    captured: dict[str, str] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
    speech_policy: SpeechPolicy = "gate",
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
    decision = decide_two(findings, speech_policy)
    unassessed = {"decisive_answers"} if judgment is None else set()
    not_applicable = {"correct_routing"} if result.actual_terminal is None else set()
    gates = gate_status(findings, unassessed, not_applicable)
    reasons = list(decision.reasons)
    if gates["correct_routing"] == "n/a":
        reasons.append(f"n/a correct_routing: {NO_TERMINAL_REASON}")
    return ThreadReport(
        thread_id=thread.thread_id,
        outcome_verdict=decision.outcome,
        release_verdict=decision.release,
        findings=findings,
        reasons=reasons,
        release_warnings=decision.warnings,
        speech_policy=speech_policy,
        expected_terminal=result.expected_terminal,
        actual_terminal=result.actual_terminal,
        simulator_goal_marker=thread.simulator_goal_marker,
        gates=gates,
        expected_path=result.expected_path,
        judge_source=judgment.source if judgment is not None else "none",
        judge_detail=judge_detail(judgment),
    )


def judge_detail(judgment) -> str:
    """Model, transport and prompt version of a model judgment ("" for hand and fake labels)."""
    if judgment is None or judgment.source not in ("cache", "live"):
        return ""
    via = f" via {judgment.transport}" if judgment.transport else ""
    return f"{judgment.model}{via}, prompt {judgment.prompt_version}"


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


def _release_cell(r: ThreadReport) -> str:
    return r.release_verdict + ("*" if r.release_warnings and r.release_verdict == "PASS" else "")


def summary_table(reports: list[ThreadReport]) -> list[str]:
    tw = max([len("expected -> actual terminal")] + [len(_terminal_cell(r)) for r in reports])
    tid = max([len("thread")] + [len(r.thread_id) for r in reports])
    task = " ".join(f"{_GATE_HEAD[g]:<6}" for g in TASK_GATES)
    head = (f"{'thread':<{tid}}  {'outcome':<12}  {'release':<12}  {'expected -> actual terminal':<{tw}}  "
            f"{task} | {_GATE_HEAD['clean_speech']:<6}  sim marker")
    lines = [head, "-" * len(head)]
    for r in reports:
        cells = [f"{_GATE_CELL.get(r.gates.get(g, ''), '?'):<6}" for g in GATES]
        marker = "yes" if r.simulator_goal_marker else "no"
        lines.append(f"{r.thread_id:<{tid}}  {r.outcome_verdict:<12}  {_release_cell(r):<12}  "
                     f"{_terminal_cell(r):<{tw}}  {' '.join(cells[:3])} | {cells[3]}  {marker}")
    lines.append("outcome = task gates only (route, answer, term). release = outcome plus clean speech under the "
                 "speech policy.")
    lines.append("gates: route = correct routing, answer = decisive answers, term = proper termination, "
                 "speech = clean speech")
    lines.append("cells: pass, FAIL (confident), review (harness not confident), n/a (no terminal reached, "
                 "routing has no meaning).")
    lines.append("PASS* = release PASS with warnings. sim marker = simulator [GOAL_ACHIEVED], not ground truth")
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
    lines.append("Hard gates (these decide FAIL / NEEDS_REVIEW; clean_speech counts for release only):")
    lines += [ln for n, g in enumerate(hard, 1) for ln in _fix_lines(n, g)] or ["  none"]
    lines.append("Soft (quality only, never change a verdict; majors show as release warnings):")
    lines += [ln for n, g in enumerate(soft, len(hard) + 1) for ln in _fix_lines(n, g)] or ["  none"]
    return lines


def thread_detail(r: ThreadReport) -> list[str]:
    if not r.expected_path:
        path = "unknown (no judgment)"
    else:
        end = r.expected_terminal or f"stopped, no usable answer at {r.expected_path[-1]}"
        path = " > ".join(r.expected_path + [end])
    lines = [
        f"=== {r.thread_id}  outcome {r.outcome_verdict}  release {r.release_verdict}  (judge: {r.judge_source})",
        f"  expected path: {path}",
        f"  actual terminal: {r.actual_terminal or 'none'}",
    ]
    if r.reasons:
        lines.append("  reasons:")
        for reason in r.reasons:
            lines += _wrap(reason, "      ", "    - ")
    if r.release_warnings:
        lines.append("  release warnings:")
        for warning in r.release_warnings:
            lines += _wrap(warning, "      ", "    - ")
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
    for r in reports:  # the simulator judges the task, so compare with the task outcome
        if r.outcome_verdict == "NEEDS_REVIEW":
            review.append(r.thread_id)
        elif (r.outcome_verdict == "PASS") == r.simulator_goal_marker:
            agree.append(r.thread_id)
        else:
            disagree.append(r.thread_id)
    return {"agree": agree, "disagree": disagree, "not_comparable": review}


def counts(reports: list[ThreadReport], kind: str = "release") -> dict[str, int]:
    out = {"PASS": 0, "FAIL": 0, "NEEDS_REVIEW": 0}
    for r in reports:
        out[r.outcome_verdict if kind == "outcome" else r.release_verdict] += 1
    return out


def _fmt_counts(c: dict[str, int]) -> str:
    return f"PASS {c['PASS']}, FAIL {c['FAIL']}, NEEDS_REVIEW {c['NEEDS_REVIEW']}"


def top_blocker(reports: list[ThreadReport]) -> dict | None:
    """The fix group behind the most release FAILs (then release NEEDS_REVIEWs), counting only findings
    that count for release under each thread's speech policy."""
    by_id = {r.thread_id: r for r in reports}
    best: tuple[int, int] | None = None
    chosen: dict | None = None
    for g in fix_groups(reports):
        if g["gate"] is None:
            continue
        hit = [t for t in g["threads"] if by_id[t].release_verdict != "PASS"
               and (g["gate"] != "clean_speech" or by_id[t].speech_policy == "gate")]
        if not hit:
            continue
        fails = 0 if g["uncertain"] else sum(1 for t in hit if by_id[t].release_verdict == "FAIL")
        key = (fails, len(hit))
        if best is None or key > best:
            best, chosen = key, {**g, "blocked": hit}
    return chosen


def headline(reports: list[ThreadReport]) -> str:
    line = (f"Task outcome: {_fmt_counts(counts(reports, 'outcome'))}. "
            f"Release: {_fmt_counts(counts(reports, 'release'))}.")
    b = top_blocker(reports)
    if b is not None:
        fix = "one template fix" if b["check"].startswith("hygiene.") else f"one fix, owner {b['owner']}"
        line += f" Top blocker: {b['check']} ({len(b['blocked'])} threads, {fix})."
    return line


def footer(reports: list[ThreadReport]) -> list[str]:
    a = simulator_agreement(reports)
    by_id = {r.thread_id: r for r in reports}
    lines = [
        f"TOTAL {len(reports)} threads."]
    lines += _wrap(headline(reports), "  ", "  ")
    lines += [
        "Simulator [GOAL_ACHIEVED] marker vs harness task outcome (comparison only, not ground truth):",
        f"  agree {len(a['agree'])}, disagree {len(a['disagree'])}, not comparable (NEEDS_REVIEW) "
        f"{len(a['not_comparable'])}",
    ]
    if a["disagree"]:
        parts = [f"{t} (marker {'yes' if by_id[t].simulator_goal_marker else 'no'}, "
                 f"harness {by_id[t].outcome_verdict})"
                 for t in a["disagree"]]
        lines += _wrap(", ".join(parts), "    ", "  disagreements: ")
    return lines


_SOURCE_NOTE = {"hand": "hand (the author's own labels, NOT model output)"}


def render_text(reports: list[ThreadReport], policy_note: str | None = None) -> str:
    sources = sorted({(r.judge_source, r.judge_detail) for r in reports})
    policies = sorted({r.speech_policy for r in reports})
    shown = ", ".join(_SOURCE_NOTE.get(s, s) + (f" ({d})" if d else "") for s, d in sources) or "none"
    lines = [
        f"callcheck: {len(reports)} thread(s), judge source: {shown}",
        f"speech policy: {policy_note or ', '.join(policies) or 'gate'}",
        "",
    ]
    lines += _wrap(headline(reports), "  ", "")
    lines += [""] + summary_table(reports) + [""] + fix_list(reports) + [""]
    for r in reports:
        lines += thread_detail(r) + [""]
    lines += footer(reports)
    return "\n".join(ascii_text(line) for line in lines) + "\n"


def _ratio(n: int, d: int) -> str:
    return f"{n}/{d}" + (f" ({100 * n / d:.0f}%)" if d else "")


def render_agreement(rows: list, item_ids: list[str], judge_desc: str) -> str:
    """Agreement table: model judge answers vs the author's hand labels (neither is ground truth)."""
    lines = _wrap(f"callcheck agreement: {judge_desc} vs hand labels (the author's own reading, not ground "
                  "truth)", "  ", "") + [""]
    judged = [r for r in rows if r.judged]
    if not judged:
        lines += _wrap("No model judgments available: no cache entry for any thread in q3/cache/judgments.json "
                       "and no usable live backend.", "  ", "")
        lines += _wrap("Create them with ANTHROPIC_API_KEY set (uv run callcheck --live) or through the Claude "
                       "Code CLI (CALLCHECK_BACKEND=claude-cli uv run callcheck --live)", "  ", "")
        return "\n".join(ascii_text(ln) for ln in lines) + "\n"
    lines.append(f"{'thread':<10}  {'all items':<12}  {'on path':<12}  {'terminal':<9}  disagreements")
    lines.append("-" * 96)
    for r in rows:
        if not r.judged:
            lines.append(f"{r.thread_id:<10}  no cache entry")
            continue
        term = "same" if r.terminal_hand == r.terminal_judge else "DIFF"
        lead = (f"{r.thread_id:<10}  {r.matched}/{r.total:<10}  {r.path_matched}/{r.path_total:<10}  "
                f"{term:<9}  ")
        lines += _wrap("; ".join(r.disagreements) or "none", " " * len(lead), lead)
    lines.append("")
    lines.append(f"{'item':<14}  agree (exact answer match, judged threads)")
    for item_id in item_ids:
        agree = sum(1 for r in judged if not any(d.startswith(item_id + ":") for d in r.disagreements))
        lines.append(f"{item_id:<14}  {_ratio(agree, len(judged))}")
    lines.append("")
    m, t = sum(r.matched for r in judged), sum(r.total for r in judged)
    pm, pt = sum(r.path_matched for r in judged), sum(r.path_total for r in judged)
    same = sum(1 for r in judged if r.terminal_hand == r.terminal_judge)
    lines += _wrap(f"TOTAL {len(judged)} judged thread(s), {len(rows) - len(judged)} without a judgment. "
                   f"All items {_ratio(m, t)}. On the expected path {_ratio(pm, pt)}. "
                   f"Expected terminal {_ratio(same, len(judged))}.", "  ", "")
    lines.append("on path = items on the expected path under the hand labels; those decide routing.")
    return "\n".join(ascii_text(ln) for ln in lines) + "\n"


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def to_json(reports: list[ThreadReport]) -> dict:
    return {
        "speech_policy": sorted({r.speech_policy for r in reports}),
        "counts": {"outcome": counts(reports, "outcome"), "release": counts(reports, "release")},
        "headline": headline(reports),
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
    """Follows the release verdict."""
    verdicts = {r.release_verdict for r in reports}
    if "FAIL" in verdicts:
        return 1
    if "NEEDS_REVIEW" in verdicts:
        return 2
    return 0
