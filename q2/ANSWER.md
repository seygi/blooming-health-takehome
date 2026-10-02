# Q2: Keeping a 40 Minute Agent on Course

## The problem, framed

Drift at minute 25 is what you get when the model is the state store. The position in a 130 question form, what has been answered and what is pending all exist only implicitly, spread across 25 minutes of audio, under instructions written at minute 0. The model is doing two jobs: holding a conversation, and being a database with a cursor. It is good at the first and unreliable at the second, more so as context grows.

There is also a hard ceiling. Some back of envelope math: audio input runs around 10 tokens per second and output audio more. Say 15 minutes of caller speech and 15 minutes of agent speech in a 35 minute call. That comes to roughly 30k to 40k tokens before tool traffic. If the realtime model's window is in the 32k range (I would confirm the current limit for the exact model version), then by minute 30 the server is already truncating the oldest items. Those items are exactly the opening instructions and the early answers. So "losing the thread" is partly attention dilution and partly the thread literally being dropped.

The fix: **move the form out of the model.** A deterministic controller in the Pipecat pipeline owns the form. The model renders one small step at a time and reports what it heard through tools. We cannot rewrite what has been said, but we can stop depending on it.

We already have evidence for this shape. The gym data's `engine_config` encodes the screener as items, questions and scenarios, with `goto` and `disposition` edges: a graph. It is a state machine that today gets handed to the model as prose. Even in those five turn calls, the agent re-asks volunteered answers (in thread_07 the caller says up front they live in San Diego County and have the yellow packet, and is asked both again) and leaks stage directions like "(Waiting for your response.)". If that happens at minute 2 with 5 questions, minute 35 with 130 is not a tuning problem.

## What lives where

| Outside the model (controller, source of truth) | Inside the model's context (small, rewritten often) |
|---|---|
| Form schema: 130 questions, branching graph, validation rules (types, enums, date ranges) | Persona and rules: short, stable, under ~300 tokens |
| Answer ledger: field, value, confidence, source turn id, confirmed flag, timestamp | Current section goal, one sentence |
| Cursor: section, question, plus the set of open fields | The next 1 to 3 questions with exact wording, accepted values, and what counts as a valid answer |
| Retry counts, no progress counters, elapsed time | Only the prior answers this section depends on (e.g. the household members from section 2 when section 7 asks about each person's income) |
| Transcript of every model turn, for checks | What to do after this step ("after this, confirm the address block") |

The model captures answers by calling `record_answer(field, value, verbatim)`. It can also call `record_answers([...])` when the caller volunteers several at once. Validation happens in code. A bad date or an out of enum value comes back as a tool result like "invalid: needs month and year, ask again", never a silent accept. High stakes fields (SSN, date of birth, income amounts, household size) get a spoken read back, and they are only marked `confirmed` after a `confirm_field` call that follows the caller's yes.

The controller accepts out of order fills. When a volunteered answer lands for a field ahead of the cursor, the ledger records it, and that question is either skipped or turned into a quick confirmation ("You mentioned you're still in San Diego County, is that right?"). This is the exact failure in thread_07.

## What triggers a change

Every trigger leads to the same action. The controller re-renders the instruction block from the ledger and cursor and sends a `session.update`.

| Trigger | Controller action |
|---|---|
| `record_answer` accepted | Advance cursor, re-render with next questions |
| Validation failure | Re-render with the failure reason and a rephrased prompt; bump retry count |
| Section boundary | Re-render with read back summary of the section, then the new section goal |
| Off cursor turn (model asked something not in the current step, or a confirmed field) | Re-render with an explicit correction: "Field X is captured. Ask Y next." |
| N turns (say 3) with no ledger progress | Re-render with a narrower step; after a second strike, offer a callback or human handoff |
| Caller changes topic or interrupts | Allow a short side branch, then re-render to resume at cursor |
| Time thresholds (every ~5 min) | Re-render the full block even if nothing changed, as a refresh |
| Long silence | Re-render with a check in prompt |

Sequencing matters. Updates apply between responses, never mid stream, and each render carries a sequence number so the turn checker judges every response against the version that was live. For corrections I would use per response instructions on `response.create` where supported, so the fix hits exactly the next utterance.

## Bounding the audio context

We cannot edit the past, but we may be able to drop it. The Realtime API has `conversation.item.delete` and `conversation.item.truncate`, and newer versions expose a server side truncation policy; I would verify behavior on the exact model version first. If deletion is solid, delete answered question and answer items once the ledger has them confirmed, keeping the last few turns for continuity, so context stays roughly flat at minute 35. If deletion confuses the model, skip it and rely on instruction refresh and tool results. The ledger makes old audio redundant either way.

## Recovering what the model is no longer carrying

The rule is that the model never remembers, it asks the ledger.

* `get_answer(field)` and `get_section_summary(section)` are tools the model can call when the caller says "like I told you earlier".
* The controller does the dependency work proactively. Each question in the schema declares the fields it depends on, and the renderer injects their current values. Section 7 gets "Household: Maria (self), Jose (spouse), Ana (daughter, 9)" without the model having to dig through 20 minutes of audio.
* Corrections ("actually, it's three kids, not two") go through `update_answer`. The controller marks dependent fields stale and schedules them for re-confirmation.

Section end read backs do double duty: they catch capture errors while the caller is on the line, and they re-anchor the model on fresh, correct state. Spoken progress ("we're about halfway") is good UX for a 35 minute call and another anchor.

## Guarding every turn

Every model utterance (from the output transcript) goes through cheap deterministic checks against the cursor: expected question asked, no confirmed field re-asked, no required field passed without a `record_answer`, no parenthetical stage directions. With speech to speech there is no text to filter before audio, so leaks are fixed in the instructions and caught by evals. A small async classifier handles fuzzy cases ("is this a paraphrase of Q14?"). It feeds the next instruction update and never blocks audio.

## What breaks first

The controller makes skipping and re-asking structurally hard. What breaks first is the model's interface into the controller, meaning **capture fidelity at the tool boundary**. The model says "got it" but never calls `record_answer`. Or it records a volunteered answer under the wrong field. Or a correction never becomes an `update_answer`. That failure is silent: the conversation sounds perfect while the ledger is wrong, and wrong ledger values go into a real benefits application. It also gets worse with call length, just like the original drift.

Detection:

* **Ledger versus transcript reconciliation.** An async extractor reads each caller turn and proposes field values. Disagreements with the ledger (value spoken but not recorded, recorded value not heard) raise a flag live, so the controller can schedule a confirmation, and get logged per call.
* **Per call drift metrics by minute.** Expected versus asked question, re-asks of confirmed fields, turns per captured field, ledger completeness against elapsed time. Alert when minute 30 to 40 numbers diverge from minute 0 to 10.
* **Offline eval.** The same checks run as a harness over recorded and simulated calls (the Q3 tool is the seed of this).

Second in line: cost and latency growth from audio tokens if item deletion turns out to be unusable. Watch input tokens per response and time to first audio by minute.

## Testing

Run simulated 40 minute callers (LLM caller personas, like the gym) with seeded behaviors: volunteering three answers at once, correcting an earlier answer at minute 28, going silent, rambling. Score every run by minute on the metrics above, with a release gate that late call accuracy is within a small margin of early call accuracy.

## What I would do next with more time

Measure the real context limit and truncation behavior on the production model. A/B item deletion against refresh only. Measure how often callers volunteer out of order, which sets how much the multi field capture path matters. Then decide between Pipecat Flows and a custom controller.
