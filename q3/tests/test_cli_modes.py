"""CLI judge modes: live by default (key required), --cached replay, claude-cli backend. No network."""

import pytest
from fixtures_labels import hand_judge

from callcheck import cli, judge as J
from callcheck.judge import JudgeAuthError

THREAD = "thread_01"  # real thread from q3/data/gym_agent_conversations.json


class NoJudgment:
    """Stands in for a live judge whose call failed for this thread only (refusal, timeout)."""

    model = "fake"

    def judge(self, spec, thread):
        return None


class RejectedKey:
    model = "fake"

    def judge(self, spec, thread):
        raise JudgeAuthError("credentials rejected (AuthenticationError)")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CALLCHECK_BACKEND", raising=False)


@pytest.fixture
def forbid_judging(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no judge may be built or run")

    monkeypatch.setattr(cli, "default_judge", boom)
    monkeypatch.setattr(cli, "evaluate", boom)


def test_missing_key_exits_3_with_help_and_runs_nothing(forbid_judging, capsys):
    assert cli.main(["--thread", THREAD]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    for needle in ("ANTHROPIC_API_KEY", "export ANTHROPIC_API_KEY", "--cached", "--labels hand",
                   "CALLCHECK_BACKEND=claude-cli"):
        assert needle in captured.err


def test_missing_key_also_blocks_agreement(forbid_judging, capsys):
    assert cli.main(["--agreement"]) == 3
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_default_run_is_live_and_refreshes_the_cache(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    seen = {}

    def fake_default(live=False):
        seen["live"] = live
        return hand_judge()

    monkeypatch.setattr(cli, "default_judge", fake_default)
    assert cli.main(["--thread", THREAD]) == 0
    assert seen["live"] is True


def test_live_flag_is_an_accepted_noop(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "default_judge", lambda live=False: hand_judge())
    assert cli.main(["--thread", THREAD, "--live"]) == 0


def test_rejected_key_mid_run_exits_3_not_needs_review(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-bad")
    monkeypatch.setattr(cli, "default_judge", lambda live=False: RejectedKey())
    assert cli.main([]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "rejected" in captured.err and "--cached" in captured.err


def test_transient_judge_failure_stays_needs_review(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(cli, "default_judge", lambda live=False: NoJudgment())
    assert cli.main(["--thread", THREAD]) == 2
    assert "NEEDS_REVIEW" in capsys.readouterr().out


def test_cached_replays_committed_run_without_key(capsys):
    code = cli.main(["--cached", "--thread", THREAD])
    out = capsys.readouterr().out
    assert code in (0, 1, 2)
    assert (f"judge source: cache (replay of committed run: claude-sonnet-5-5 via claude-cli, "
            f"prompt {J.PROMPT_VERSION})") in out.splitlines()[0]


def test_cached_agreement_works_without_key(capsys):
    assert cli.main(["--cached", "--agreement"]) == 0
    assert "model judge (cache" in capsys.readouterr().out


def test_cached_missing_entry_is_needs_review(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "cached_judge", lambda: J.CachedJudge(None, model="m", path=tmp_path / "none.json"))
    assert cli.main(["--cached", "--thread", THREAD]) == 2
    assert "NEEDS_REVIEW" in capsys.readouterr().out


def test_cached_and_live_are_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--cached", "--live"])
    assert e.value.code == 2


def test_claude_cli_backend_needs_no_key(monkeypatch):
    monkeypatch.setenv("CALLCHECK_BACKEND", "claude-cli")
    monkeypatch.setattr(J.shutil, "which", lambda name: "/usr/bin/claude")
    seen = {}

    def fake_default(live=False):
        seen["live"] = live
        return hand_judge()

    monkeypatch.setattr(cli, "default_judge", fake_default)
    assert cli.main(["--thread", THREAD]) == 0
    assert seen["live"] is True


def test_claude_cli_backend_without_binary_exits_3(forbid_judging, monkeypatch, capsys):
    monkeypatch.setenv("CALLCHECK_BACKEND", "claude-cli")
    monkeypatch.setattr(J.shutil, "which", lambda name: None)
    assert cli.main([]) == 3
    assert "claude binary is not on PATH" in capsys.readouterr().err


def test_hand_labels_need_no_key(monkeypatch, capsys):
    monkeypatch.setattr(cli, "default_judge", lambda live=False: pytest.fail("no model judge for hand labels"))
    assert cli.main(["--labels", "hand", "--thread", THREAD]) == 0
    assert "judge source: hand" in capsys.readouterr().out
