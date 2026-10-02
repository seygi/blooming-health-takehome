"""The committed judge cache must cover every thread, so the harness runs offline."""

import json

from callcheck import judge as J
from callcheck.judge import CACHE_PATH, CachedJudge, Judgment, answer_enum


class _Boom:
    def judge(self, spec, thread):  # pragma: no cover
        raise AssertionError("inner judge called")


def test_committed_cache_covers_all_threads(spec, threads):
    assert CACHE_PATH.exists(), "q3/cache/judgments.json must be committed"
    entries = json.loads(CACHE_PATH.read_text())["entries"]
    models = {e["model_requested"] for e in entries.values()}
    assert models == {J.DEFAULT_MODEL}
    assert len(threads) == 10
    assert {e["transport"] for e in entries.values()} <= {"api", "claude-cli"}
    cj = CachedJudge(_Boom(), model=models.pop())
    for t in threads:
        j = cj.judge(spec, t)
        assert isinstance(j, Judgment), t.thread_id
        assert j.source == "cache" and j.prompt_version == J.PROMPT_VERSION
        assert set(j.items) == set(spec.items)
        for iid, ij in j.items.items():
            assert ij.answer in answer_enum(spec.items[iid])
