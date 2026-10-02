# Plan (3h cap)

## Wave 0 (sequential): foundation
- [x] W0 scaffold: pyproject (uv, `callcheck` script, anthropic, pytest), model.py, load.py, spec.py (graph + walk),
      checks/__init__.py registry + Context, cli skeleton printing each thread's scripted path. Tests for load/spec/walk.

## Wave 1 (parallel, worktrees)
- [x] Q1 writer: q1/ANSWER.md
- [x] Q2 writer: q2/ANSWER.md
- [x] F1 align.py + checks hygiene, repetition (asked twice), termination, script
- [x] F2 judge.py: prompt, tool schema, cache, Judge protocol, fake judge for tests, populate cache live for 10 threads

## Wave 2
- [x] F3 routing.py + verdict.py + report.py + repetition "volunteered" using judge + cli wiring; JSON out
- [x] prod-reviewer pass, fix critical only

## Wave 3: ship
- [x] docs-writer: README.md + q3/README.md
- [x] coordinator: final review of Q1/Q2, push repo (private first), draft reply email (not sent)

## Added during the run
- [x] R1 fixes from the production review (judge prompt v2, outcome vs release verdicts, termination and hygiene hardening, hand labels and agreement)
- [x] J1 Claude Code CLI judge transport (API key was invalid), real cache for 10 threads, agreement 80/80 against hand labels
