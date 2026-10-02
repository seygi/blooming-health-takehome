"""ClaudeCliJudge tests. No network and no real subprocess: subprocess.run is replaced by a fake runner."""

import json
import subprocess

import pytest

from callcheck import judge as J
from callcheck.judge import NOT_DISCUSSED, CachedJudge, ClaudeCliJudge, Judgment, build_tool, cache_key
from callcheck.report import evaluate, render_text


def _thread(threads, tid):
    return next(t for t in threads if t.thread_id == tid)


def _structured(spec):
    items = {iid: {"evidence": "", "answer": NOT_DISCUSSED, "confidence": 0.9, "evidence_turn": None,
                   "first_available_turn": None, "first_available_quote": "", "value": None} for iid in spec.items}
    items["q1_intent"] = {"evidence": "Yes, please.", "answer": "yes", "confidence": 0.97, "evidence_turn": "C0",
                          "first_available_turn": "C0", "first_available_quote": "Yes, please.", "value": None}
    return {"items": items}


def _stdout(structured, is_error=False, model="claude-sonnet-5-5"):
    return json.dumps({"type": "result", "is_error": is_error, "structured_output": structured,
                       "modelUsage": {model: {"inputTokens": 1, "outputTokens": 1}}})


class _Runner:
    """Stands in for subprocess.run. Each response is a stdout string, a CompletedProcess or an exception."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        r = self.responses.pop(0)
        if isinstance(r, BaseException):
            raise r
        if isinstance(r, subprocess.CompletedProcess):
            return r
        return subprocess.CompletedProcess(cmd, 0, stdout=r, stderr="")


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_cli_success_path(spec, threads):
    t = _thread(threads, "thread_01")
    runner = _Runner([_stdout(_structured(spec))])
    j = ClaudeCliJudge("claude-sonnet-5-5", runner=runner).judge(spec, t)
    assert isinstance(j, Judgment)
    assert j.source == "live" and j.transport == "claude-cli"
    assert j.model == "claude-sonnet-5-5" and j.prompt_version == J.PROMPT_VERSION
    assert j.items["q1_intent"].answer == "yes" and j.items["q1_intent"].evidence_turn == 0
    assert j.items["q2_active"].answer == NOT_DISCUSSED
    cmd, kwargs = runner.calls[0]
    assert cmd[0] == "claude" and "-p" in cmd
    assert _flag(cmd, "--model") == "claude-sonnet-5-5"
    assert _flag(cmd, "--output-format") == "json"
    assert _flag(cmd, "--system-prompt") == J.SYSTEM_PROMPT
    assert json.loads(_flag(cmd, "--json-schema")) == build_tool(spec, t)["input_schema"]
    assert _flag(cmd, "--tools") == ""
    assert kwargs["input"] == J.build_user_message(spec, t)
    assert kwargs["timeout"] == 180
    assert kwargs.get("shell") in (None, False)


def test_cli_child_env_has_no_api_key(spec, threads, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("CALLCHECK_SENTINEL", "kept")
    runner = _Runner([_stdout(_structured(spec))])
    ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01"))
    env = runner.calls[0][1]["env"]
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CALLCHECK_SENTINEL"] == "kept"


def test_cli_is_error_retries_once_then_none(spec, threads):
    runner = _Runner([_stdout(None, is_error=True), _stdout(None, is_error=True)])
    assert ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01")) is None
    assert len(runner.calls) == 2


def test_cli_error_then_success_uses_retry(spec, threads):
    runner = _Runner([_stdout(None, is_error=True), _stdout(_structured(spec))])
    j = ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01"))
    assert j is not None and len(runner.calls) == 2


def test_cli_invalid_json_returns_none(spec, threads):
    runner = _Runner(["not json at all", "{\"truncated\": "])
    assert ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01")) is None


def test_cli_missing_structured_output_returns_none(spec, threads):
    runner = _Runner([_stdout(None), _stdout("a string")])
    assert ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01")) is None


def test_cli_nonzero_exit_returns_none(spec, threads):
    bad = subprocess.CompletedProcess(["claude"], 1, stdout="", stderr="boom")
    runner = _Runner([bad, bad])
    assert ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01")) is None


def test_cli_timeout_returns_none(spec, threads):
    runner = _Runner([subprocess.TimeoutExpired("claude", 180), subprocess.TimeoutExpired("claude", 180)])
    assert ClaudeCliJudge("m", runner=runner).judge(spec, _thread(threads, "thread_01")) is None
    assert len(runner.calls) == 2


def test_cli_binary_missing_returns_none_and_disables(spec, threads):
    runner = _Runner([FileNotFoundError("claude")])
    judge = ClaudeCliJudge("m", runner=runner)
    assert judge.judge(spec, _thread(threads, "thread_01")) is None
    assert judge.available is False
    assert judge.judge(spec, _thread(threads, "thread_02")) is None
    assert len(runner.calls) == 1


def test_cli_validation_is_shared_with_api_judge(spec, threads):
    # An agent turn id and an answer outside the enum go through the same parse_tool_input.
    raw = _structured(spec)
    raw["items"]["q2_active"] = {**raw["items"]["q2_active"], "answer": "maybe"}
    raw["items"]["q1_intent"] = {**raw["items"]["q1_intent"], "evidence_turn": "A0"}
    j = ClaudeCliJudge("m", runner=_Runner([_stdout(raw)])).judge(spec, _thread(threads, "thread_01"))
    assert j.items["q2_active"].answer == J.UNCLEAR
    assert any("agent turn id" in n for n in j.notes)


def test_cache_key_identical_between_backends(spec, threads, tmp_path, monkeypatch):
    t = _thread(threads, "thread_01")
    monkeypatch.setenv("CALLCHECK_BACKEND", "api")
    k_api = cache_key("claude-sonnet-5-5", spec, t)
    monkeypatch.setenv("CALLCHECK_BACKEND", "claude-cli")
    k_cli = cache_key("claude-sonnet-5-5", spec, t)
    assert k_api == k_cli
    # An entry written through the CLI is a hit for a cache reader with no live backend at all.
    path = tmp_path / "judgments.json"
    CachedJudge(ClaudeCliJudge("claude-sonnet-5-5", runner=_Runner([_stdout(_structured(spec))])),
                model="claude-sonnet-5-5", path=path).judge(spec, t)
    entry = next(iter(json.loads(path.read_text())["entries"].values()))
    assert entry["transport"] == "claude-cli"
    assert set(json.loads(path.read_text())["entries"]) == {k_api}
    j = CachedJudge(None, model="claude-sonnet-5-5", path=path).judge(spec, t)
    assert j.source == "cache" and j.transport == "claude-cli"


def test_old_cache_entry_without_transport_reads_as_none(spec, threads, tmp_path):
    t = _thread(threads, "thread_01")
    path = tmp_path / "judgments.json"
    CachedJudge(ClaudeCliJudge("m", runner=_Runner([_stdout(_structured(spec))])), model="m", path=path).judge(spec, t)
    data = json.loads(path.read_text())
    for e in data["entries"].values():
        del e["transport"]
    path.write_text(json.dumps(data))
    j = CachedJudge(None, model="m", path=path).judge(spec, t)
    assert j.source == "cache" and j.transport is None


def test_default_judge_backend_selection(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CALLCHECK_BACKEND", "claude-cli")
    monkeypatch.setattr(J.shutil, "which", lambda name: "/usr/bin/claude")
    j = J.default_judge()
    assert isinstance(j, CachedJudge) and isinstance(j.inner, ClaudeCliJudge)
    monkeypatch.setenv("CALLCHECK_BACKEND", "api")
    assert J.default_judge().inner is None  # api without a key stays offline
    monkeypatch.delenv("CALLCHECK_BACKEND")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert isinstance(J.default_judge().inner, J.ClaudeJudge)


def test_default_judge_cli_backend_without_binary_is_offline(monkeypatch):
    monkeypatch.setenv("CALLCHECK_BACKEND", "claude-cli")
    monkeypatch.setattr(J.shutil, "which", lambda name: None)
    assert J.default_judge().inner is None


def test_unknown_backend_falls_back_to_api_with_warning(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CALLCHECK_BACKEND", "carrier-pigeon")
    assert J.default_judge().inner is None
    assert "CALLCHECK_BACKEND" in capsys.readouterr().err


def test_report_header_names_model_transport_and_prompt(spec, threads, tmp_path):
    t = _thread(threads, "thread_01")
    path = tmp_path / "judgments.json"
    CachedJudge(ClaudeCliJudge("claude-sonnet-5-5", runner=_Runner([_stdout(_structured(spec))])),
                model="claude-sonnet-5-5", path=path).judge(spec, t)
    r = evaluate(spec, t, CachedJudge(None, model="claude-sonnet-5-5", path=path))
    out = render_text([r])
    assert (f"judge source: cache (claude-sonnet-5-5 via claude-cli, prompt {J.PROMPT_VERSION})"
            in out.splitlines()[0])
