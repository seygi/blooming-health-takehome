from callcheck import cli, hand_labels
from callcheck.hand_labels import HAND_LABELS, L, agreement, hand_judge
from callcheck.judge import CachedJudge, FakeJudge
from callcheck.report import render_agreement


class NoJudge:
    def judge(self, spec, thread):
        return None


def test_docstring_says_not_model_output():
    assert "NOT model output" in hand_labels.__doc__


def test_every_thread_labelled(threads):
    assert {t.thread_id for t in threads} == set(HAND_LABELS)


def test_hand_judge_source_is_hand(spec, threads):
    j = hand_judge().judge(spec, threads[0])
    assert j is not None and j.source == "hand" and j.model == "hand-labels"
    assert set(j.items) == set(spec.items)


# ---- agreement -------------------------------------------------------------------------


def _judge_with_one_flip():
    presets = {tid: dict(items) for tid, items in HAND_LABELS.items()}
    # thread_01: the model reads q2_active as inactive (on the path, flips the terminal)
    presets["thread_01"]["q2_active"] = L("q2_active", "inactive", 0.8, "I definitely have active coverage", 1)
    # thread_06: the model misses q4_residency (off the decline path)
    del presets["thread_06"]["q4_residency"]
    return FakeJudge(presets)


def test_agreement_counts(spec, threads):
    rows = {r.thread_id: r for r in agreement(spec, threads, _judge_with_one_flip())}
    r1 = rows["thread_01"]
    assert r1.judged and r1.total == len(spec.items) and r1.matched == r1.total - 1
    assert (r1.path_matched, r1.path_total) == (1, 2)
    assert r1.terminal_hand == "medi_cal_active" and r1.terminal_judge != "medi_cal_active"
    assert r1.disagreements == ["q2_active: judge inactive, hand active"]
    r6 = rows["thread_06"]
    assert r6.path_matched == r6.path_total  # q4_residency is not on the decline path
    assert r6.matched == r6.total - 1
    r5 = rows["thread_05"]
    assert r5.matched == r5.total and r5.disagreements == []


def test_agreement_table(spec, threads):
    text = render_agreement(agreement(spec, threads, _judge_with_one_flip()), list(spec.items), "fake judge")
    assert "not ground truth" in text
    line = next(ln for ln in text.splitlines() if ln.startswith("thread_01"))
    assert "DIFF" in line and "q2_active: judge inactive, hand active" in line
    assert "q2_active       9/10 (90%)" in text
    assert "Expected terminal 9/10 (90%)" in " ".join(text.split())
    assert max(len(ln) for ln in text.splitlines()) <= 120


def test_agreement_without_judgments_says_so(spec, threads):
    text = render_agreement(agreement(spec, threads, NoJudge()), list(spec.items), "model judge")
    assert "No model judgments available" in text and "--live" in text


# ---- CLI -------------------------------------------------------------------------------


def test_cli_labels_hand_header(capsys):
    code = cli.main(["--labels", "hand"])
    out = capsys.readouterr().out
    assert "judge source: hand (the author's own labels, NOT model output)" in out
    assert "medi_cal_active (match)" in out
    assert code == 1  # release FAIL on leaked stage directions


def test_cli_agreement_no_cache(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        cli, "default_judge", lambda live=False: CachedJudge(None, model="m", path=tmp_path / "missing.json")
    )
    assert cli.main(["--agreement"]) == 0
    out = capsys.readouterr().out
    assert "model judge (cache, model m)" in out and "No model judgments available" in out


def test_cli_agreement_rejects_hand_labels():
    assert cli.main(["--agreement", "--labels", "hand"]) == 3
