# Plan (3h cap)

## Wave 0 (sequential): foundation
- [ ] W0 scaffold: pyproject (uv, `callcheck` script, anthropic, pytest), model.py, load.py, spec.py (graph + walk),
      checks/__init__.py registry + Context, cli skeleton printing each thread's scripted path. Tests for load/spec/walk.

## Wave 1 (parallel, worktrees)
- [ ] Q1 writer: q1/ANSWER.md
- [ ] Q2 writer: q2/ANSWER.md
- [ ] F1 align.py + checks hygiene, repetition (asked twice), termination, script
- [ ] F2 judge.py: prompt, tool schema, cache, Judge protocol, fake judge for tests, populate cache live for 10 threads

## Wave 2
- [ ] F3 routing.py + verdict.py + report.py + repetition "volunteered" using judge + cli wiring; JSON out
- [ ] prod-reviewer pass, fix critical only

## Wave 3: ship
- [ ] docs-writer: README.md + q3/README.md
- [ ] coordinator: final review of Q1/Q2, push public repo, draft reply email for Seygi (not sent)
