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
    assert (r.outcome_verdict, r.release_verdict) == EXPECTED_VERDICTS[tid], r.reasons
    assert r.verdict == r.release_verdict
    assert r.judge_source == "hand"


def test_thread_08_outcome_pass_with_release_warning(hand_reports):
    r = hand_reports["thread_08"]
    assert r.outcome_verdict == "PASS" and r.release_verdict == "PASS"
    assert any("termination.contradictory_close" in w for w in r.release_warnings)


@pytest.mark.parametrize("tid", ["thread_03", "thread_05", "thread_10"])
def test_soft_speech_policy_releases_on_outcome(spec, threads, tid):
    r = evaluate(spec, by_id(threads, tid), hand_judge(), speech_policy="soft")
    assert r.release_verdict == r.outcome_verdict == EXPECTED_VERDICTS[tid][0]
    assert any(w.startswith("WARN clean_speech") for w in r.release_warnings)
    assert r.gates["clean_speech"] == "fail"  # the gate cell still shows the leak


@pytest.mark.parametrize("tid", ["thread_07", "thread_09", "thread_10"])
def test_routing_is_not_applicable_without_terminal(hand_reports, tid):
    r = hand_reports[tid]
    assert r.actual_terminal is None
    assert r.gates["correct_routing"] == "n/a"
    assert "n/a correct_routing: agent never reached a terminal" in r.reasons


def test_thread_01_passes_all_gates(hand_reports):
    r = hand_reports["thread_01"]
    assert set(r.gates.values()) == {"pass"}
    assert r.expected_terminal == r.actual_terminal == "medi_cal_active"


def test_thread_07_review_reason_is_simulator_truncation(hand_reports):
    r = hand_reports["thread_07"]
    assert r.gates["proper_termination"] == "uncertain"
    assert r.gates["correct_routing"] == "n/a"
    assert any("truncated_by_simulator" in reason for reason in r.reasons)


def test_volunteered_answers_are_soft_only(hand_reports):
    r = hand_reports["thread_10"]
    soft = [f for f in r.findings if f.check == "repetition.asked_after_volunteered"]
    assert soft and all(f.gate is None for f in soft)


def test_no_judge_thread_02_still_fails(nojudge_reports):
    r = nojudge_reports["thread_02"]
    assert r.outcome_verdict == r.release_verdict == "FAIL"
    assert r.judge_source == "none"
    assert r.gates["correct_routing"] == "uncertain"
    assert r.gates["decisive_answers"] == "uncertain"
    assert r.gates["proper_termination"] == "fail"


@pytest.mark.parametrize("tid", ["thread_01", "thread_07"])
def test_no_judge_without_hard_failure_needs_review(nojudge_reports, tid):
    r = nojudge_reports[tid]
    assert r.outcome_verdict == r.release_verdict == "NEEDS_REVIEW"
    assert any("routing.judge_unavailable" in reason for reason in r.reasons)


def test_no_judge_never_passes(nojudge_reports):
    assert all(r.outcome_verdict != "PASS" and r.release_verdict != "PASS" for r in nojudge_reports.values())


def test_wrong_terminal_fails_end_to_end(spec, threads):
    labels = dict(HAND_LABELS["thread_01"])
    labels["q2_active"] = L("q2_active", "inactive", 0.95, "it's not active", 1)
    r = evaluate(spec, by_id(threads, "thread_01"), FakeJudge({"thread_01": labels}))
    assert r.outcome_verdict == "FAIL"
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
    assert "Task outcome: PASS 6, FAIL 1, NEEDS_REVIEW 3. Release: PASS 2, FAIL 7, NEEDS_REVIEW 1." in text
    assert "Top blocker: hygiene.stage_direction (4 threads, one template fix)." in " ".join(text.split())
    assert "speech policy: gate" in text
    head = next(ln for ln in text.splitlines() if ln.startswith("thread "))
    assert "outcome" in head and "release" in head
    assert "PASS*" in next(ln for ln in text.splitlines() if ln.startswith("thread_08"))
    assert "\u2014" not in text


def test_fix_list_ranks_by_threads_affected(hand_reports):
    text = render_text(list(hand_reports.values()))
    fix = text.split("FIX LIST", 1)[1].split("===", 1)[0]
    first = [ln for ln in fix.splitlines() if ln.strip().startswith("1.")][0]
    assert "hygiene.stage_direction" in first and "4 thread(s)" in first


def test_json_shape(hand_reports):
    data = to_json(list(hand_reports.values()))
    assert data["counts"] == {"outcome": {"PASS": 6, "FAIL": 1, "NEEDS_REVIEW": 3},
                              "release": {"PASS": 2, "FAIL": 7, "NEEDS_REVIEW": 1}}
    assert data["speech_policy"] == ["gate"]
    t = next(t for t in data["threads"] if t["thread_id"] == "thread_01")
    assert t["outcome_verdict"] == t["release_verdict"] == "PASS"
    assert t["gates"]["correct_routing"] == "pass" and t["expected_path"] == ["q1_intent", "q2_active"]
    t7 = next(t for t in data["threads"] if t["thread_id"] == "thread_07")
    assert any(f["uncertain"] for f in t7["findings"])
    json.dumps(data)


# ---- exit codes and CLI ----------------------------------------------------------------


def _report(verdict):
    from callcheck.model import ThreadReport

    return ThreadReport(thread_id="t", outcome_verdict="PASS", release_verdict=verdict)


def test_exit_codes_follow_release():
    assert exit_code([_report("PASS")]) == 0
    assert exit_code([_report("PASS"), _report("NEEDS_REVIEW")]) == 2
    assert exit_code([_report("NEEDS_REVIEW"), _report("FAIL")]) == 1


@pytest.fixture
def no_cache(monkeypatch, tmp_path):
    """CLI replaying an empty cache: the judge is unavailable for every thread."""
    from callcheck.judge import CachedJudge

    monkeypatch.setattr(cli, "cached_judge", lambda: CachedJudge(None, model="m", path=tmp_path / "missing.json"))


def test_cli_fail_exit_code(no_cache, capsys):
    assert cli.main(["--cached", "--thread", "thread_02"]) == 1
    out = capsys.readouterr().out
    assert "thread_02" in out and "FAIL" in out


def test_cli_needs_review_exit_code(no_cache, capsys):
    assert cli.main(["--cached", "--thread", "thread_01"]) == 2
    assert "NEEDS_REVIEW" in capsys.readouterr().out


def test_cli_pass_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cached_judge", hand_judge)
    assert cli.main(["--cached", "--thread", "thread_01"]) == 0
    assert "PASS" in capsys.readouterr().out


def test_cli_unknown_thread(no_cache, capsys):
    assert cli.main(["--cached", "--thread", "nope"]) == 3


def test_cli_json_written(no_cache, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(["--cached", "--json", "--out", str(out)])
    assert code == 1
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("report.json")
    data = json.loads(out.read_text())
    assert len(data["threads"]) == 10
    assert data["counts"]["release"]["FAIL"] >= 1


def test_cli_captured_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "cached_judge", hand_judge)
    cap = tmp_path / "captured.json"
    cap.write_text(json.dumps({"thread_01": {"q2_active": "Not active or unsure"}}))
    assert cli.main(["--cached", "--thread", "thread_01", "--captured", str(cap)]) == 1
    out = capsys.readouterr().out
    assert "routing.captured_mismatch" in out
    assert "agent recorded Not active or unsure, caller said active at turn 1" in out


def test_cli_bad_captured_file(tmp_path):
    cap = tmp_path / "captured.json"
    cap.write_text("[1, 2]")
    assert cli.main(["--captured", str(cap)]) == 3


def test_cli_speech_policy_default_voice(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cached_judge", hand_judge)
    assert cli.main(["--cached", "--thread", "thread_05"]) == 1  # voice: the leak blocks release
    out = capsys.readouterr().out
    assert "speech policy: gate (default for agent_config.transport_mode = voice)" in out


def test_cli_speech_policy_soft(monkeypatch, capsys):
    monkeypatch.setattr(cli, "cached_judge", hand_judge)
    assert cli.main(["--cached", "--thread", "thread_05", "--speech-policy", "soft"]) == 0
    assert "speech policy: soft (set by --speech-policy)" in capsys.readouterr().out


def test_speech_policy_default_text_mode():
    assert cli.speech_policy(None, "text")[0] == "soft"
    assert cli.speech_policy(None, None)[0] == "soft"
    assert cli.speech_policy(None, "voice")[0] == "gate"
