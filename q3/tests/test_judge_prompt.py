"""Prompt text, version and the `python -m callcheck.judge` entry point. No network."""

import subprocess
import sys
from pathlib import Path

from callcheck import judge as J
from callcheck.judge import CachedJudge, FakeJudge, ItemJudgment

ROOT = Path(__file__).resolve().parents[2]


def test_prompt_version_bumped():
    assert J.PROMPT_VERSION == "v2"


def test_prompt_explains_turn_ids_and_order():
    p = J.SYSTEM_PROMPT
    assert "[A-open]" in p and "[C<t>]" in p and "[A<t>]" in p
    assert "Caller turn C<t> is spoken before agent turn A<t>" in p
    assert "never an A id" in p


def test_prompt_rule_1_agent_lines_are_context_only():
    p = J.SYSTEM_PROMPT
    assert "Use AGENT lines only to know which question a CALLER line answers." in p
    assert "Facts come only from CALLER words." in p
    assert "An agent summary or assumption is never evidence" in p
    assert "it sounds like" in p
    # the old wording forbade reading agent lines at all, which contradicts using them for context
    assert "Judge CALLER lines only" not in p


def test_prompt_rule_2_map_by_meaning():
    p = J.SYSTEM_PROMPT
    assert "Map by meaning. Implied answers are allowed" in p
    assert "literally" not in p
    assert "0.7 to 0.89 when the meaning is clear but indirect or implied" in p


def test_prompt_corrections_and_opener_caveat():
    p = J.SYSTEM_PROMPT
    assert "the final answer is the corrected one" in p
    assert "first_available_turn is the turn where the caller first stated the corrected answer" in p
    assert 'already says "over the phone"' in p
    assert "is not a q5_choice answer" in p
    assert "volunteered" in p and "first_available_quote" in p
    assert "Worked example" in p


def test_prompt_follows_writing_rules():
    # no em dashes, no spaced hyphen used as a sentence connector
    assert "\u2014" not in J.SYSTEM_PROMPT
    assert " - " not in J.SYSTEM_PROMPT


def test_module_entry_help_runs():
    r = subprocess.run([sys.executable, "-m", "callcheck.judge", "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--refresh" in r.stdout


def test_main_offline_reports_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(J, "default_judge", lambda live=False: CachedJudge(None, model="m", path=tmp_path / "c.json"))
    assert J._main(["thread_01"]) == 1
    assert "thread_01: no judgment" in capsys.readouterr().out


def test_main_refresh_passes_live_and_prints(monkeypatch, capsys):
    seen = {}
    preset = {"q5_choice": ItemJudgment("q5_choice", "by_phone", 0.8, "finish it over the phone now", 4, 3,
                                        None, "finish that yellow packet over the phone")}

    def fake_default(live=False):
        seen["live"] = live
        return FakeJudge({"thread_07": preset})

    monkeypatch.setattr(J, "default_judge", fake_default)
    assert J._main(["--refresh", "thread_07"]) == 0
    out = capsys.readouterr().out
    assert seen["live"] is True
    assert "thread_07 [fake" in out and "v2" in out
    assert "first quote: 'finish that yellow packet over the phone'" in out
