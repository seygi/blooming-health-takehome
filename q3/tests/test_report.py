"""End to end: evaluate, verdicts, text report, JSON and CLI exit codes. No network."""

import json

import pytest
from fixtures_labels import EXPECTED_VERDICTS, HAND_LABELS, L, hand_judge

from callcheck import cli
from callcheck.judge import FakeJudge
from callcheck.report import evaluate, exit_code, render_text, to_json


class NoJudge:
    def judge(self, spec, thread):
        return None


def by_id(threads, tid):
    return next(t for t in threads if t.thread_id == tid)


@pytest.fixture(scope="module")
def hand_reports(spec, threads):
    judge = hand_judge()
    return {t.thread_id: evaluate(spec, t, judge) for t in threads}


@pytest.fixture(scope="module")
def nojudge_reports(spec, threads):
    return {t.thread_id: evaluate(spec, t, NoJudge()) for t in threads}


# ---- verdicts -------------------------------------------------------------------------


@pytest.mark.parametrize("tid", sorted(EXPECTED_VERDICTS))
def test_verdict_under_hand_labels(hand_reports, tid):
    r = hand_reports[tid]
    assert r.verdict == EXPECTED_VERDICTS[tid], r.reasons
    assert r.judge_source == "fake"


def test_thread_01_passes_all_gates(hand_reports):
    r = hand_reports["thread_01"]
    assert set(r.gates.values()) == {"pass"}
    assert r.expected_terminal == r.actual_terminal == "medi_cal_active"


def test_thread_07_review_reason_is_simulator_truncation(hand_reports):
    r = hand_reports["thread_07"]
    assert r.gates["proper_termination"] == "uncertain"
    assert r.gates["correct_routing"] == "pass"
    assert any("truncated_by_simulator" in reason for reason in r.reasons)


def test_volunteered_answers_are_soft_only(hand_reports):
    r = hand_reports["thread_10"]
    soft = [f for f in r.findings if f.check == "repetition.asked_after_volunteered"]
    assert soft and all(f.gate is None for f in soft)


def test_no_judge_thread_02_still_fails(nojudge_reports):
    r = nojudge_reports["thread_02"]
    assert r.verdict == "FAIL"
    assert r.judge_source == "none"
    assert r.gates["correct_routing"] == "uncertain"
    assert r.gates["decisive_answers"] == "uncertain"
    assert r.gates["proper_termination"] == "fail"


@pytest.mark.parametrize("tid", ["thread_01", "thread_07"])
def test_no_judge_without_hard_failure_needs_review(nojudge_reports, tid):
    r = nojudge_reports[tid]
    assert r.verdict == "NEEDS_REVIEW"
    assert any("routing.judge_unavailable" in reason for reason in r.reasons)


def test_no_judge_never_passes(nojudge_reports):
    assert all(r.verdict != "PASS" for r in nojudge_reports.values())


def test_wrong_terminal_fails_end_to_end(spec, threads):
    labels = dict(HAND_LABELS["thread_01"])
    labels["q2_active"] = L("q2_active", "inactive", 0.95, "it's not active", 1)
    r = evaluate(spec, by_id(threads, "thread_01"), FakeJudge({"thread_01": labels}))
    assert r.verdict == "FAIL"
    assert r.gates["correct_routing"] == "fail"


def test_captured_mismatch_end_to_end(spec, threads):
    r = evaluate(spec, by_id(threads, "thread_09"), hand_judge(), captured={"q3_plan": "Aetna"})
    assert r.gates["decisive_answers"] == "fail"


# ---- text ------------------------------------------------------------------------------


def test_text_report_shape(hand_reports):
    text = render_text(list(hand_reports.values()))
    assert text.isascii()
    assert max(len(line) for line in text.splitlines()) <= 120
    assert "FIX LIST" in text
    assert "not ground truth" in text
    assert "TOTAL 10 threads: PASS 1  FAIL 8  NEEDS_REVIEW 1" in text
    assert "\u2014" not in text


def test_fix_list_ranks_by_threads_affected(hand_reports):
    text = render_text(list(hand_reports.values()))
    fix = text.split("FIX LIST", 1)[1].split("===", 1)[0]
    first = [ln for ln in fix.splitlines() if ln.strip().startswith("1.")][0]
    assert "hygiene.stage_direction" in first and "4 thread(s)" in first


def test_json_shape(hand_reports):
    data = to_json(list(hand_reports.values()))
    assert data["counts"] == {"PASS": 1, "FAIL": 8, "NEEDS_REVIEW": 1}
    t = next(t for t in data["threads"] if t["thread_id"] == "thread_01")
    assert t["gates"]["correct_routing"] == "pass" and t["expected_path"] == ["q1_intent", "q2_active"]
    t7 = next(t for t in data["threads"] if t["thread_id"] == "thread_07")
    assert any(f["uncertain"] for f in t7["findings"])
    json.dumps(data)


# ---- exit codes and CLI ----------------------------------------------------------------


def _report(verdict):
    from callcheck.model import ThreadReport

    return ThreadReport(thread_id="t", verdict=verdict)


def test_exit_codes():
    assert exit_code([_report("PASS")]) == 0
    assert exit_code([_report("PASS"), _report("NEEDS_REVIEW")]) == 2
    assert exit_code([_report("NEEDS_REVIEW"), _report("FAIL")]) == 1


@pytest.fixture
def no_cache(monkeypatch, tmp_path):
    """CLI with no API key and an empty cache: the judge is unavailable for every thread."""
    from callcheck.judge import CachedJudge

    monkeypatch.setattr(
        cli, "default_judge", lambda live=False: CachedJudge(None, model="m", path=tmp_path / "missing.json")
    )


def test_cli_fail_exit_code(no_cache, capsys):
    assert cli.main(["--thread", "thread_02"]) == 1
    out = capsys.readouterr().out
    assert "thread_02" in out and "FAIL" in out


def test_cli_needs_review_exit_code(no_cache, capsys):
    assert cli.main(["--thread", "thread_01"]) == 2
    assert "NEEDS_REVIEW" in capsys.readouterr().out


def test_cli_pass_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "default_judge", lambda live=False: hand_judge())
    assert cli.main(["--thread", "thread_01"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_unknown_thread(no_cache, capsys):
    assert cli.main(["--thread", "nope"]) == 3


def test_cli_json_written(no_cache, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(["--json", "--out", str(out)])
    assert code == 1
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("report.json")
    data = json.loads(out.read_text())
    assert len(data["threads"]) == 10
    assert data["counts"]["FAIL"] >= 1


def test_cli_captured_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "default_judge", lambda live=False: hand_judge())
    cap = tmp_path / "captured.json"
    cap.write_text(json.dumps({"thread_01": {"q2_active": "Not active or unsure"}}))
    assert cli.main(["--thread", "thread_01", "--captured", str(cap)]) == 1
    out = capsys.readouterr().out
    assert "routing.captured_mismatch" in out
    assert "agent recorded Not active or unsure, caller said active at turn 1" in out


def test_cli_bad_captured_file(tmp_path):
    cap = tmp_path / "captured.json"
    cap.write_text("[1, 2]")
    assert cli.main(["--captured", str(cap)]) == 3
