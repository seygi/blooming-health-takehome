# Decisions log (one line each)

- Interviewer-side file found in candidate folder: quarantined outside the repo, not used. Disclosed in README.
- Brief says Q3 has "a single call plus captured field values"; the data has 10 threads and no captured-values record. Harness evaluates all 10 and derives captured answers from the transcript; an optional captured map is cross-checked when present.
- Simulator `[GOAL_ACHIEVED]` marker treated as the simulator's opinion, not ground truth; shown as comparison column.
- Python + uv because Blooming's stack is Python/Go.
- engine_config has 8 items; scenario ids wrapped in double underscores (q3_plan __close__) are catch-alls; answers matching no scenario follow default_route.
- Correction to own first draft: hygiene fix hint said 'strip in TTS text filter', but GPT Realtime is speech to speech, there is no text stage before audio. Fix belongs in the prompt/engine source plus eval gating.
- Judge on claude-sonnet-5-5: forced tool_choice and non default temperature return 400 on this model, so the call uses tool_choice auto + strict tool schema + one retry if no tool call; determinism comes from the committed cache, not temperature.
- Judge labels all 8 items for every thread (path independent); decline is not_discussed unless the caller declined; a plain yes to q1 is not evidence for later items.
- Judge API errors (bad key, model not found, network) return None (harness reports NEEDS_REVIEW), never crash; bad credentials disable live calls for the rest of the run.
- Server side refusal fallbacks not enabled: on Sonnet 5.5 they only retry cyber and frontier_llm declines, irrelevant for this domain; a refusal returns None.
