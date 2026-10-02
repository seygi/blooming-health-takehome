# callcheck: did this triage call succeed?

Input: `q3/data/gym_agent_conversations.json`, 10 threads of the `fhcsd_triage_screener` agent plus its `agent_config` (the `engine_config` flow: 8 items, scenarios with `goto` and `disposition`, outcomes with `say`) and `team_config`. Run instructions are in the [root README](../README.md#q3-quick-start).

## Success definition

A triage call has one job: find out what the caller's situation is and send them to the right place, saying only what a member should hear. So each thread gets two verdicts.

**Task outcome** (did the call do its job): all three task gates pass.
**Release** (is this agent fit to put on a phone line): task outcome plus clean speech. Under `--speech-policy gate` (the default when `agent_config.transport_mode` is voice) a speech leak blocks release; under `soft` it is a release warning.

Why two: with one verdict, the speech leaks made most threads FAIL and hid that routing was right on every thread where it could be decided. An engineer needs to know both "the flow logic works" and "do not ship this prompt".

| Gate | Hard or soft | What it checks | Why |
|---|---|---|---|
| `correct_routing` | hard (task) | The terminal the agent reached (outcome disposition or `_complete` handoff) equals the terminal the `engine_config` graph gives for what the caller actually said, and no item on that path was skipped without the caller having answered it. | Routing is the product: a wrong terminal sends a member to the wrong team or closes them out. |
| `decisive_answers` | hard (task) | Every branch deciding answer on the expected path is clear in the caller's own words, at or above `min_confidence_score`. With `--captured`, recorded values match what the caller said. | A route built on an ambiguous answer is a guess; the agent should have clarified. |
| `proper_termination` | hard (task) | The call reaches a terminal and the agent does not keep talking past it (new questions, improvised next steps, PII collection). | Talking past the terminal means the engine did not end or hand off; the agent improvises outside its scope. |
| `clean_speech` | hard (release only) | No stage directions, system text, simulator markers or markdown in agent speech. | In speech to speech every character is spoken aloud. |
| soft checks | soft | Re-asking an answered or volunteered question, off script questions, partial scripted `say`, goodbye on a `_complete` handoff, plan name not captured. | They degrade quality. They never flip a verdict on their own; soft majors show as release warnings. |

Verdict per scope: **FAIL** when at least one confident hard failure (counted gate, severity critical or major, not uncertain). **NEEDS_REVIEW** when no confident failure but at least one uncertain finding on a counted gate. **PASS** otherwise.

## Deterministic vs model

The model maps free caller speech to scenario meanings and finds when information first appeared. Everything else is code.

| Code (deterministic) | Model (`judge.py`, one call per thread) |
|---|---|
| Parse the `engine_config` into a graph and walk it (`spec.py`) | For each of the 8 items: which scenario the caller's words match, or `unclear`, or `not_discussed` |
| Align each agent message to scripted questions, scenario says and outcome says, with leftover text (`align.py`) | Confidence 0..1, a verbatim caller quote and its turn |
| Which terminal the agent actually reached | `first_available_turn`: the earliest caller turn that already held the information (volunteered answers) |
| Expected terminal (graph walk on the judge's labels), path comparison, skip detection (`routing.py`) | Normalized value for capture items (plan name, decline reason) |
| Hygiene, termination, repetition, script and say coverage checks (`checks/`) | |
| Verdicts, gate table, fix list (`verdict.py`, `report.py`) | |

Why the line sits there: verdicts must be reproducible and auditable. If a regex or a graph walk says FAIL, anyone can rerun it and point at the line that caused it. Mapping "I just used it for an appointment last week" to "active coverage", or noticing that "I have the yellow packet right here, let's do it over the phone" answers two items nobody asked yet, needs language understanding; code would be brittle there.

Guardrails on the model part:

- The judge comes from a different model family than the agent under test (agent: `gpt-realtime-mini`, judge: Claude). Same practice as the LLM judged benchmarks I built for production voice agents: keep the judge outside the family under test to avoid self preference bias.
- The judge never decides the verdict. It only labels what the caller said; code turns that into PASS, FAIL or NEEDS_REVIEW. A judge score that can drift with a prompt tweak never overrides a deterministic failure.
- The judge reads caller lines as evidence. Agent lines only tell it which question a caller line answers, so a wrong agent cannot grade itself right.
- It labels all 8 items on every thread, independent of the path the agent took, so a skipped branch deciding question is still caught.
- Strict tool schema; turn fields are an enum of this thread's caller ids (`C<n>`), so an agent turn cannot be cited as caller evidence. Output is validated in code (unknown items dropped, bad turns nulled, quotes checked against their turn, confidence clamped).
- Results are cached in `q3/cache/judgments.json`, keyed by sha256 of model, prompt version, system prompt, tool schema and rendered user message. Any prompt edit misses the cache. `claude-sonnet-5-5` rejects forced `tool_choice` and non default temperature, so determinism comes from the cache. The committed entries were generated with `claude-sonnet-5-5` through the Claude Code CLI (`CALLCHECK_BACKEND=claude-cli`, same prompt and schema as the API path); the transport is stored per entry and shown in the report header, never in the key.
- The default run is live and needs `ANTHROPIC_API_KEY` (or `CALLCHECK_BACKEND=claude-cli`, which uses a local Claude Code subscription and no key). A missing or rejected key exits 3 with a message listing `--cached`, `--labels hand` and `claude-cli`; it never degrades every thread to NEEDS_REVIEW. Per thread failures (refusal, timeout, network) give NEEDS_REVIEW for that thread only, and it never crashes.
- `--cached` replays the committed `q3/cache/judgments.json` with no key and no network. During development I ran the judge through my Claude subscription using the Claude Code CLI (`CALLCHECK_BACKEND=claude-cli`); the committed cache is that run. `--live` is kept as a no-op alias.
- `--labels hand` swaps in the author's own labels (`callcheck/hand_labels.py`), and `--agreement` (live, or with `--cached`) reports how often the model matches them, per item and per thread.
- Agreement on this data (prompt v2): 80/80 item answers (100%), 42/42 on the expected path, 10/10 expected terminals, and `first_available_turn` matches on all 80 items. One labeler, 10 short calls: consistent, not yet calibrated.

## Uncertainty handling

The harness never returns PASS when it is unsure. These produce NEEDS_REVIEW (an uncertain finding on a counted gate):

- A branch deciding answer is `unclear` or below the confidence threshold (`routing.low_confidence_answer`). The threshold is read from `team_config.configuration.rules_of_engagement.quality_gates.min_confidence_score` (0.7 in the data), with 0.7 as fallback.
- A wrong terminal or path divergence when any branch deciding answer on the expected path is below the threshold: the finding is kept but marked uncertain.
- No judgment for the thread: no cache entry under `--cached`, or a failed model call for that thread (`routing.judge_unavailable`). `decisive_answers` then shows `review`, never `pass`.
- The agent closed with text that matches no outcome say (`termination.terminal_unrecognized`): the matcher may have missed a paraphrase, so a human confirms which outcome was meant.
- The simulator emitted `[GOAL_ACHIEVED]` and the episode stopped before any terminal (`termination.truncated_by_simulator`, owner simulator).
- Without a judgment, a question asked out of graph order (`script.out_of_order`), since the caller may have volunteered the skipped answer.

When no terminal is reached, `correct_routing` shows `n/a` rather than `pass`.

## Check catalog

Example threads are from `uv run callcheck --cached` (model judge from the committed cache); `--labels hand` fires exactly the same checks on the same threads. "none in this data" means the check exists and is tested but did not fire on these 10 threads; "none with these labels" means it depends on the judge labels and did not fire with either the model or the hand labels.

| Check id | Gate | Severity | Owner | What it catches | Example |
|---|---|---|---|---|---|
| `routing.wrong_terminal` | correct_routing | critical | prompt | Agent delivered a different outcome than the graph gives for the caller's answers | none with these labels |
| `routing.path_divergence` | correct_routing | major | engine or prompt | Skipped an item on the expected path, or asked an item off the expected path | none with these labels |
| `routing.terminal_without_answer` | correct_routing | critical | engine | Disposition delivered although the decisive answer was never given | none with these labels |
| `routing.judge_unavailable` | correct_routing | major, uncertain | harness | No judgment, so routing cannot be confirmed | every thread when `--cached` runs with an empty cache |
| `script.out_of_order` | correct_routing | major, uncertain | engine | Asked item is not a graph successor (only runs without a judgment) | none in this data |
| `routing.low_confidence_answer` | decisive_answers | major, uncertain | harness | Branch deciding answer unclear or below threshold | none with these labels |
| `routing.captured_mismatch` | decisive_answers | critical | prompt | Recorded value differs from what the caller said (needs `--captured`) | no captured record in the data |
| `termination.talks_past_terminal` | proper_termination | critical | engine | New questions or content after the terminal; flags PII collection | thread_02 (asks for name, address, date of birth after `_complete`) |
| `termination.no_terminal` | proper_termination | critical | engine | Call ended with no outcome say or handoff | none in this data |
| `termination.truncated_by_simulator` | proper_termination | major, uncertain | simulator | Simulator stopped the episode before any terminal | thread_07, thread_09, thread_10 |
| `termination.terminal_unrecognized` | proper_termination | major, uncertain | harness | Closing text matches no outcome say | none in this data |
| `hygiene.stage_direction` | clean_speech | major | prompt | "(Waiting for your response.)", asterisk actions, bracketed cues | thread_03, thread_05, thread_09, thread_10 |
| `hygiene.system_text` | clean_speech | major | prompt | "This was the final message of this follow-up." | thread_04, thread_06 |
| `hygiene.simulator_marker` | clean_speech | major | prompt | Bracketed control markers echoed by the agent | none in this data |
| `hygiene.markdown` | clean_speech | major | prompt | Bold, code, headings, list markers in speech | none in this data |
| `termination.contradictory_close` | soft | major | prompt | Goodbye on a `_complete` handoff | thread_02, thread_08 |
| `repetition.asked_after_volunteered` | soft | minor | prompt | Asked a question the caller had already answered (uses the judge's `first_available_turn`) | 7 threads, e.g. thread_07 |
| `repetition.asked_twice` | soft | minor | prompt | Same scripted question asked twice | none in this data |
| `script.off_script_question` | soft | minor | prompt | Question that matches no scripted question | thread_04 |
| `say_coverage.partial_say` | soft | minor | prompt | Scripted say delivered with sentences dropped | thread_01 (dropped "Take care.") |
| `routing.plan_not_captured` | soft | minor | prompt | q3_plan value not obtained (capture only, route unaffected) | none with these labels |

Every finding has `check, severity, gate, turn, evidence, problem, fix_hint, owner, uncertain`, so it says what is wrong, where (turn and quote) and who fixes it.

## Output anatomy

1. **Header and headline**: judge source (cache, live, hand or none), speech policy and why, then counts for both verdicts and the top blocker.
2. **Summary table**: per thread, both verdicts, expected vs actual terminal, the four gate cells (`pass`, `FAIL`, `review`, `n/a`) and the simulator marker.
3. **Fix list**: the same check and fix grouped across threads. Hard gate fixes come first, then soft, each ranked by threads affected and severity, with one example quote. A soft re-ask seen in 7 threads does not bury a leak that fails 4.
4. **Per thread detail**: expected path, actual terminal, the reasons behind each verdict, release warnings, and every finding with evidence, problem and fix.
5. **Simulator comparison**: `[GOAL_ACHIEVED]` vs the harness task outcome (agree, disagree, not comparable). With the cached model judge (and with hand labels): agree 4, disagree 3 (thread_04, thread_06, thread_08: marker no, harness PASS), not comparable 3.

`--json` writes the same content to `q3/out/report.json` with keys `speech_policy, counts, headline, simulator_agreement, fix_list, threads`.

## Module layout

```
q3/callcheck/
  model.py        dataclasses: Thread, Message, FlowSpec, Item, Question, Scenario, Outcome, Finding, ThreadReport
  load.py         JSON to FlowSpec + Thread list; strips [GOAL_ACHIEVED] into a comparison flag; reads threshold, transport
  spec.py         engine_config as a graph: resolve gotos, successors, walk(answers) to a terminal
  align.py        deterministic agent message to script node alignment, actual terminal, residual text
  checks/         deterministic checks, one file each: hygiene, repetition, termination, script, say_coverage
  judge.py        model judge: prompt, strict tool schema, validation, cache, Judge protocol, FakeJudge for tests
  hand_labels.py  author's labels for tests, --labels hand and --agreement
  routing.py      expected path from judge labels vs agent route; answer confidence; captured cross check
  verdict.py      findings to task outcome and release verdicts, gate status
  report.py       evaluation pipeline, text report (ASCII, 120 columns), JSON
  cli.py          argument parsing, speech policy, exit codes
q3/tests/         pytest, no network (judge is faked or hand labelled)
q3/cache/         committed judge cache
```

## Limits and next steps

- The judge is checked against one person's labels. Next: more labeled calls, a second labeler and inter rater agreement, then calibrate the confidence threshold on that.
- Text only. Voice needs audio checks: whether a leak was actually spoken, latency, talk over, silence.
- `next_action` (end_call vs handoff) is not visible in the transcript, so termination cannot confirm it.
- Captured values are only cross checked when supplied with `--captured`; the data has none.
- Alignment uses token similarity. A paraphrased closing goes to NEEDS_REVIEW (`terminal_unrecognized`) and a paraphrased question is reported as off script, rather than being matched.
- Next: run on the full 20 episode campaign, fix the simulator stop condition, and run `callcheck` as a CI gate on every prompt or engine change.
