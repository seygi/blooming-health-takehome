# Decisions log (one line each)

- Interviewer-side file found in candidate folder: quarantined outside the repo, not used. Disclosed in README.
- Brief says Q3 has "a single call plus captured field values"; the data has 10 threads and no captured-values record. Harness evaluates all 10 and derives captured answers from the transcript; an optional captured map is cross-checked when present.
- Simulator `[GOAL_ACHIEVED]` marker treated as the simulator's opinion, not ground truth; shown as comparison column.
- Python + uv because Blooming's stack is Python/Go.
- engine_config has 8 items; scenario ids wrapped in double underscores (q3_plan __close__) are catch-alls; answers matching no scenario follow default_route.
