# Blooming Health take-home

Seygi Kutani, Senior AI Engineer (Conversational AI, Voice) take-home for Blooming Health. Two written answers (Q1, Q2) and a runnable evaluation harness for the triage voice agent transcripts (Q3, `callcheck`). Every assumption is listed below; where time ran out, the next step is named.

## Map

| Question | Where | Thesis in one line |
|---|---|---|
| Q1 Capacity under a burst | [q1/ANSWER.md](q1/ANSWER.md) | Choppy audio is a missed 20 ms frame deadline, not a throughput limit: pre-warm from the forecast, admit calls on measured frame latency, and let the dialer be the throttle. |
| Q2 Keeping a 40 minute agent on course | [q2/ANSWER.md](q2/ANSWER.md) | Move the form out of the model: a deterministic controller owns the ledger and cursor, re-renders a small instruction block via `session.update`, and the first thing to break is capture fidelity at the tool boundary. Includes a speech to speech vs cascaded tradeoff from agents I've shipped. |
| Q3 Evaluation harness | [q3/](q3/) ([q3/README.md](q3/README.md)) | `callcheck` decides per call whether the agent did its job (task outcome) and whether it is fit to ship (release), deterministic first, a model only for reading free caller speech. |

## Q3 quick start

Needs Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run callcheck                          # model judge from the committed cache q3/cache/judgments.json, no key needed
uv run callcheck --labels hand            # the author's own hand labels instead of the model (no key, no cache)
uv run callcheck --live                   # refresh judgments from the API: needs ANTHROPIC_API_KEY
CALLCHECK_BACKEND=claude-cli uv run callcheck --live   # same refresh through the Claude Code CLI login
uv run callcheck --json                   # write q3/out/report.json (or --out FILE), print only the path
uv run callcheck --thread thread_06       # one thread
uv run callcheck --agreement              # model judge vs hand labels, per item and per thread
uv run callcheck --speech-policy soft     # speech leaks become release warnings instead of blockers
uv run pytest -q                          # 300 passed, no network
```

Model: `claude-sonnet-5-5` by default, override with `CALLCHECK_MODEL`. The key is read from `ANTHROPIC_API_KEY` only and never written anywhere. `CALLCHECK_BACKEND=claude-cli` sends the same prompt and schema through the Claude Code CLI (`claude -p`) instead; the cache key does not depend on the backend. With no cache entry and no working backend, the harness does not crash: it prints a warning and every affected thread becomes NEEDS_REVIEW.

Exit codes follow the release verdict: `0` all PASS, `1` any FAIL, `2` NEEDS_REVIEW and no FAIL, `3` usage or input error.

## Q3 in six lines

1. Two verdicts per call. **Task outcome**: correct routing (terminal matches what the `engine_config` graph prescribes for what the caller said), decisive answers captured, proper termination. **Release**: outcome plus clean member facing speech (no spoken stage directions), gated by default because `transport_mode` is voice.
2. Code decides everything that can be decided from the script and the transcript: alignment to script nodes, graph walk, termination, speech hygiene, repetition, verdicts.
3. The model does one job: map each free form caller answer to a scenario id with a quote, a confidence and the first turn the information appeared.
4. Uncertainty is its own verdict: low confidence, unclear answers, missing judgments, unrecognized closings and simulator truncation give NEEDS_REVIEW, never PASS.
5. Every finding carries evidence, turn, problem, fix hint and owner (prompt, engine, simulator, harness); the fix list groups the same fix across threads.
6. The simulator's `[GOAL_ACHIEVED]` marker is shown as a comparison column only.

## Sample output

The run below uses the model judge from the committed cache, so it needs no key.

```
$ uv run callcheck
callcheck: 10 thread(s), judge source: cache (claude-sonnet-5-5 via claude-cli, prompt v2)
speech policy: gate (default for agent_config.transport_mode = voice)

Task outcome: PASS 6, FAIL 1, NEEDS_REVIEW 3. Release: PASS 2, FAIL 7, NEEDS_REVIEW 1. Top blocker:
  hygiene.stage_direction (4 threads, one template fix).

thread     outcome       release       expected -> actual terminal          route  answer term   | speech  sim marker
---------------------------------------------------------------------------------------------------------------------
thread_01  PASS          PASS          medi_cal_active (match)              pass   pass   pass   | pass    yes
thread_02  FAIL          FAIL          _complete (match)                    pass   pass   FAIL   | pass    no
thread_03  PASS          FAIL          escalated_chw_out_of_county (match)  pass   pass   pass   | FAIL    yes
thread_04  PASS          FAIL          medi_cal_active (match)              pass   pass   pass   | FAIL    no
thread_05  PASS          FAIL          escalated_chw_appointment (match)    pass   pass   pass   | FAIL    yes
thread_06  PASS          FAIL          escalated_chw_out_of_county (match)  pass   pass   pass   | FAIL    no
thread_07  NEEDS_REVIEW  NEEDS_REVIEW  _complete -> none                    n/a    pass   review | pass    yes
thread_08  PASS          PASS*         _complete (match)                    pass   pass   pass   | pass    no
thread_09  NEEDS_REVIEW  FAIL          other_coverage_no_action -> none     n/a    pass   review | FAIL    yes
thread_10  NEEDS_REVIEW  FAIL          _complete -> none                    n/a    pass   review | FAIL    yes
outcome = task gates only (route, answer, term). release = outcome plus clean speech under the speech policy.
gates: route = correct routing, answer = decisive answers, term = proper termination, speech = clean speech
cells: pass, FAIL (confident), review (harness not confident), n/a (no terminal reached, routing has no meaning).
PASS* = release PASS with warnings. sim marker = simulator [GOAL_ACHIEVED], not ground truth
```

Judgments in q3/cache/judgments.json were generated with claude-sonnet-5-5 through the Claude Code CLI (`CALLCHECK_BACKEND=claude-cli`); `--live` with `ANTHROPIC_API_KEY` uses the API directly with the same prompt and schema.

Agreement with the author's hand labels (`uv run callcheck --agreement`): 80/80 item answers (100%), 42/42 on the expected path, 10/10 expected terminals, and `first_available_turn` matches on all 80 items. These are one person's labels on 10 short calls, so this shows the judge reads them the same way, not that it is calibrated. `--labels hand` gives the same verdicts and gate cells as above.

The full output continues with a ranked fix list, per thread detail and the simulator comparison (see [q3/README.md](q3/README.md#output-anatomy)).

## Assumptions

- The brief describes one call plus the field values the agent captured. The data in `q3/data/gym_agent_conversations.json` has 10 threads and no captured values record. The harness evaluates all 10 and derives answers from the transcript; an optional `--captured FILE` (`{thread_id: {item_id: value}}`) is cross checked when supplied.
- The transcripts come from a different, shorter agent than the Q2 Medicaid form: an 8 item triage screener (`fhcsd_triage_screener`). Success is defined against its `engine_config`.
- `_complete` means handoff to the next team agent (`team_config` connects `fhcsd_triage_screener` to `MC_216`), not end of call. Because the run exercised the agent standalone, a goodbye on `_complete` is a soft warning, not a gate failure.
- The simulator's `[GOAL_ACHIEVED]` marker is the simulator's opinion, not ground truth. When it ends an episode before any terminal, the result is NEEDS_REVIEW (owner simulator), not FAIL.
- The run is a text simulation (`source.mode`) of a voice agent (`transport_mode: voice`, `gpt-realtime-mini`). Stage directions in the text would be spoken aloud in voice, so they gate release by default; the fix hint says to confirm they also occur in voice mode.
- Speech to speech has no text stage before audio, so leaks are fixed at the prompt or engine source and caught by evals, not filtered.
- Scenario ids wrapped in double underscores (q3_plan `__close__`) are capture only placeholders: the plan name never changes the route, so a missing plan is a minor soft finding.
- Skipping a question is fine when the caller already gave a confident answer before the agent moved on.
- The confidence threshold is `team_config` `quality_gates.min_confidence_score` (0.7 in the data).
- The campaign is named "All Paths (random explorer, 20 eps)" but the file holds 10 threads; the harness evaluates what is there.

## Time and what I would do next

Built within the 3 hour budget, directing AI agents (details below). Next, in order:

1. Calibrate the judge on more labeled calls, and measure inter rater agreement between two human labelers before trusting judge agreement numbers (`--agreement` currently compares against one author's labels).
2. Audio level checks for voice: was a stage direction actually spoken, latency, talk over, silence.
3. Cross check captured field values once the real captured record is available (the code path exists behind `--captured`).
4. Run on the full 20 episode campaign, and fix the simulator stop condition so episodes run to a terminal.
5. A CI gate: run `callcheck` on every prompt or engine change and block on release FAIL.

## How I used AI tools

A coordinator Claude Code session planned the work as cards and dispatched subagents per card, each in its own git worktree: implementers for the loader, graph, alignment, checks, judge, routing and report; writers for Q1, Q2 and these docs; and a production reviewer pass. I reviewed every diff, ran the tests in the worktree and merged with `--no-ff`; the history is left unsquashed so it can be read. Decisions are logged one line each in [docs/decisions-log.md](docs/decisions-log.md).

Where the tools were wrong and got corrected:

- The first hygiene fix hint said to strip stage directions in a TTS text filter. GPT Realtime is speech to speech, there is no text stage before audio; the hint now points at the prompt or engine source plus eval gating.
- The first verdict design made 8 of 10 threads FAIL and hid that routing was correct on every decidable thread. A review pass led to the split into a task outcome verdict and a release verdict.
- A correct closing that the agent paraphrased was a confident FAIL (no terminal). It is now NEEDS_REVIEW (`terminal_unrecognized`): the matcher may have missed a paraphrase.
- The judge cache key ignored the prompt text, so a prompt edit could silently reuse stale judgments. The key now hashes the system prompt, tool schema and rendered user message.
- The judge prompt had a turn numbering ambiguity. Caller and agent lines now have distinct ids (`C3`, `A3`), and the schema only accepts caller ids as evidence.
- `claude-sonnet-5-5` rejects forced `tool_choice` and non default temperature with a 400. The judge uses a strict tool schema with one retry, and determinism comes from the committed cache, not from temperature.
- Q1 and Q2 were first drafted from general knowledge, then rewritten around measurements from production voice agents I've shipped (generalized, rounded). In review I removed two sentences a writer agent had embellished beyond what was measured, and corrected one figure (unfulfilled agent commitments are counted per action, not per call).
- A piped test command once hid failures (the pipe's exit code masked pytest's), and a red commit got through. That commit was rebuilt green.

## Disclosure

The shared Drive folder also contained a file whose header says it is interviewer side and should not be shared with the candidate (it holds rewards and goal labels). I noticed it from the header, moved it out of the working directory without using it, and nothing in this repo is derived from it. Let me know if you want me to delete it.
