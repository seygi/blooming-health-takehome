# Q2: Keeping a 40 Minute Agent on Course

## The problem, framed

Drift at minute 25 is what you get when the model is the state store. The position in a 130 question form, what has been answered and what is pending all exist only implicitly, spread across 25 minutes of audio, under instructions written at minute 0. The model is doing two jobs: holding a conversation, and being a database with a cursor. It is good at the first and unreliable at the second, more so as context grows.

There is also a hard ceiling. Back of envelope: audio input runs around 10 tokens per second and output audio more. Say 15 minutes of caller speech and 15 of agent speech in a 35 minute call: roughly 30k to 40k tokens before tool traffic. If the realtime model's window is in the 32k range (I would confirm the limit for the exact model version), the server is truncating the oldest items by minute 30. Those are exactly the opening instructions and the early answers. "Losing the thread" is partly attention dilution and partly the thread literally being dropped.

The fix: **move the form out of the model.** A deterministic controller in the Pipecat pipeline owns the form. The model renders one small step at a time and reports what it heard through tools.

The gym data already points here. `engine_config` is a state machine (`goto` and `disposition` edges) handed to the model as prose. Even in five turn calls the agent re-asks volunteered answers (thread_07: county and yellow packet, both asked again) and leaks stage directions like "(Waiting for your response.)". If that happens at minute 2, minute 35 with 130 questions is not a tuning problem.

## Speech to speech or cascaded

This choice decides how much of the rest is enforceable. The voice agents I've shipped run cascaded on Pipecat: Deepgram class streaming STT, a GPT 4.1 class LLM, ElevenLabs class TTS, local Silero VAD plus a smart turn model. I evaluated OpenAI speech to speech for the same kind of work (scripted, tool heavy flows) and judged it weaker. That was my own evaluation, not a formal benchmark, so treat it as informed reasoning.

| Dimension | Favors |
|---|---|
| Latency, prosody, barge in naturalness | Speech to speech |
| Instruction following on long scripts | Cascaded |
| Tool call reliability | Cascaded |
| Guards on text before it is spoken | Cascaded (S2S has no text before audio) |
| Verbatim record of what was said | Cascaded |
| Voice choice, swappable components | Cascaded |

The core difference: in a cascaded pipeline every model turn can be a strict structured output `{say, collected, next}`, where `next` is an enum of legal routes. An invalid transition cannot be emitted at all. In speech to speech the same checks become post hoc.

My recommendation respects your stack. If latency and naturalness matter for your members (plausible for a 35 minute call with older or stressed callers), keep GPT Realtime for the conversation, but run a parallel transcription stream feeding the controller. Then evaluate a cascaded path for the data capture heavy sections (income, household, identifiers). Decide with an A/B on **capture accuracy at minute 30+**, because that is the metric that matters for a benefits application. Everything below works with either path, and is stronger with a text stage.

## What lives where

| Outside the model (controller, source of truth) | Inside the model's context (small) |
|---|---|
| Form schema: 130 questions, branching graph, validation rules | Small always on kernel: persona and rules, under ~300 tokens |
| Answer ledger: field, value, confidence, source turn, confirmed flag | Current section goal, one sentence |
| Cursor: section, question, set of open fields | Next 1 to 3 questions, exact wording, accepted values |
| Retry and no progress counters, elapsed time | Only prior answers this section depends on |
| Transcript of every model turn | What comes after this step |

In a voice platform I built, we moved from sending a roughly 50 section script every turn to a kernel plus only the current block. First speakable text dropped from about 1 s to about 640 ms p50, and routing was correct in every replayed case. Smaller context was faster and more accurate.

The model captures answers with `record_answer(field, value, verbatim)` or `record_answers([...])` when the caller volunteers several. Validation happens in code; a bad date comes back as "invalid: needs month and year, ask again". Three rules from production:

* **Sealed fields.** High stakes fields (SSN, date of birth, income, household size) get a spoken read back and are sealed after `confirm_field`. A later STT mishearing of digits cannot overwrite them; only an explicit `update_answer` after the caller corrects can.
* **Provenance.** A value like a callback number must come from the caller's own words, never from call metadata.
* **Out of order fills.** A volunteered answer ahead of the cursor is recorded and the question becomes a quick confirmation. That is exactly the thread_07 failure.

## What triggers a change

Every trigger leads to the same action: the controller re-renders from ledger and cursor.

| Trigger | Controller action |
|---|---|
| `record_answer` accepted | Advance cursor, render next questions |
| Validation failure | Render failure reason and rephrased prompt; bump retry |
| Section boundary | Read back summary, then new section goal |
| Off cursor turn or confirmed field re-asked | One shot correction: "Field X is captured. Ask Y next." |
| 3 turns with no ledger progress | Narrower step; second strike offers callback or handoff |
| Caller side branch | Allow briefly, then resume at cursor |
| Every ~5 min | Refresh even if nothing changed |

**Append, do not rewrite.** In production, appending state updates kept about 90% of prompt tokens cached; rewriting the system message dropped cache hits to zero. So the render keeps a stable prefix and changes a small tail. I would measure whether Realtime `session.update` has the same cache behavior before relying on frequent full rewrites.

**Corrections are one shot, never permanent context.** In production, a tool loop breaker wrote its nudge text into the context, where it stayed forever and later stopped a caller from hanging up. The fix was to send the next completion without tools and write nothing to context. Same for language switch instructions: inject for one completion, right before the caller's last message. In Realtime terms, corrections go in per response instructions on `response.create`, not `session.update`.

**One lock.** Tool results, timers and transcripts race. One async lock wraps every entry point that mutates call state, and each render carries a sequence number so checks judge a response against the version that was live.

## Bounding the audio context

You cannot instruct your way past what the context keeps showing the model. In a voice platform I built, a repetition loop (the model kept re-asking) was replayed from failed calls: better guidance alone fixed 0 of 4; collapsing the repeated rows in context plus guidance fixed 4 of 4.

With audio native history you cannot collapse anything, which is the strongest argument for either the controller plus `conversation.item.delete` / `truncate`, or a cascaded path. If deletion works on the exact model version, delete answered items once the ledger confirms them, keeping the last few turns, so context stays roughly flat at minute 35. If deletion confuses the model, fall back to refresh plus tool results. The ledger makes old audio redundant either way.

## Recovering what the model is no longer carrying

The model never remembers, it asks the ledger.

* `get_answer(field)` and `get_section_summary(section)` for "like I told you earlier".
* Each schema question declares its dependencies and the renderer injects them. Section 7 gets "Household: Maria (self), Jose (spouse), Ana (daughter, 9)" without digging through 20 minutes of audio.
* Corrections go through `update_answer`; dependent fields are marked stale for re-confirmation.

Section read backs catch capture errors while the caller is on the line and re-anchor the model.

## Guarding every turn

In the cascaded agents I've shipped, guards run on the text before it is spoken: a completion claim gate, a commitment guard, a digit verbalizer, and a normalizer that once stopped a raw data structure from being read aloud to a caller. Verbatim transcripts flag scripted versus generated lines. On Realtime these checks run on the output transcript against the cursor (expected question asked, no sealed field re-asked, no stage directions), feed the next update, and never block audio. That is the honest cost of speech to speech: you detect, you do not prevent.

## What breaks first

The controller makes skipping and re-asking structurally hard. What breaks first is the model's interface into the controller: **capture fidelity at the tool boundary**. The model says "got it" and never calls `record_answer`, records a volunteered answer under the wrong field, or a correction never becomes `update_answer`. The conversation sounds perfect while the ledger is wrong, and wrong values go into a real benefits application.

This is not hypothetical. When I added a commitment guard in production, it caught about half a dozen promised actions that were never performed across roughly 50 live calls. On short calls. I would expect the rate to grow with call length, and I would measure that first.

Detection:

* **Ledger versus transcript reconciliation.** An async extractor proposes field values from each caller turn. Disagreements (spoken but not recorded, recorded but not heard) trigger a live confirmation and get logged.
* **Claim versus tool log.** Any "I've saved that" or "I'll send you" without a matching tool call in the same turn is flagged.
* **Drift metrics by minute.** Re-asks of sealed fields, turns per captured field, ledger completeness against elapsed time. Alert when minute 30 to 40 diverges from minute 0 to 10.

Second in line: audio token cost and latency growth if item deletion is unusable; watch input tokens and time to first audio by minute.

## Testing

Simulated 40 minute callers (LLM personas, like the gym) seeded to volunteer several answers at once, correct an answer at minute 28, or go silent. Replay every failed production call as a regression case; that is how the repetition and loop breaker fixes above were proven. Release gate: late call capture accuracy within a small margin of early call.

## What I would do next with more time

Measure the real context limit, truncation and `session.update` cache behavior on the production model. A/B item deletion against refresh only. Run the Realtime versus cascaded capture accuracy A/B on the data heavy sections. Measure how often callers volunteer out of order. Then decide between Pipecat Flows and a custom controller.
