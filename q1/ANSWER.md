# Q1: Capacity Under a Burst

## The short version

- This is a real time deadline problem, not a throughput problem. Choppy audio means a 20 ms frame or an end of turn decision missed its deadline. Average CPU will look healthy while that happens.
- The dial window (30 min) is shorter than the call (40 min), so peak concurrency equals the number of answered calls.
- The burst is scheduled. Provision before the window from a forecast; do not let an autoscaler discover it. In production voice agents I've shipped, scale up speed broke before steady state load did.
- The dialer is the cheapest throttle. Admission control protects calls in flight by delaying new ones.

## 1. Reframe: quality fails before availability

Every call produces a frame every 20 ms that must be decoded, denoised, run through VAD and turn detection, and sent to the model, while model audio must reach Twilio on time. If a step is late, the caller hears a gap or a slow reply. Nothing errors, so availability stays green.

What I would suspect, in order:

- **CPU starvation per call, which surfaces as turn taking.** VAD and the turn detection model run locally on the agent's CPU. On a voice platform I built, a fractional vCPU per session was too small under real call load: end of turn detection slowed, and replies came late. Moving to about 1.5 vCPU per session fixed it. "Audio quality" complaints are often turn taking, not packet loss.
- **Many calls in one Python process.** Pipecat runs on asyncio; DSP, base64 and JSON compete for one GIL. A process with 30 calls can need 2 cores and have 1 while the node shows 30%.
- **CFS throttling.** Burn a container's quota in 70 ms of a 100 ms period and it pauses for 30 ms: lost frames on every call it hosts.
- **Event loop lag, GC pauses, network jitter to OpenAI**, which sound identical to the caller.

There are two packing models. The platform I built ran one session per agent process: no GIL contention, a crash or slow call hurts one caller, and CPU per call is easy to reason about. The cost is memory per process and a cold start per call, which is exactly what bites in a burst. Packing many calls per process is denser and warms once, but one bad call degrades its neighbours. Either works if the pool is warm before minute 0; I would trust isolation first.

## 2. The shape of the load

Arrival: 10,000 / 30 min = 333 per minute, about 5.6 dials per second. The last call is placed at minute 30 and the first ends around minute 40, so peak concurrency is the answered count, held from minute 30 to about 40, draining until about minute 70 plus the duration tail.

The brief gives no answer rate, so I carry two cases:

| Case | Answered (live) | Peak concurrent calls |
|---|---|---|
| Design ceiling (as stated) | 100% | 10,000 |
| Working assumption | 40% | 4,000 |

Ringing legs add about 5.6 × 25 s ≈ 140 at any moment, small unless a session starts at dial instead of at answer. Inside a 30 minute window, shorter calls do not lower the peak; they shorten the time spent at it.

## 3. Numbers I need, and what each one changes

| Number | What it changes |
|---|---|
| Peak concurrency from history, computed correctly | Baseline for everything. A peak query I relied on once over counted by about 2x because it counted overlapping windows. Use a sweep line over call start and end events (+1, -1, running max). |
| CPU per session under real load (p50, p99), including VAD and turn detection | vCPU per call. Size so p99 end of turn latency holds, not to average utilization. |
| Session ready time (process spawn, model load, first greeting) and safe launch rate | How far ahead to pre warm and how fast the dialer may ramp. Likely the first real limit. |
| Answer rate, voicemail rate, AMD time and accuracy by hour | Turns 10,000 dials into a real peak, and decides whether the session starts at dial or at answer. |
| Duration distribution, not the mean | Drain time and the scale in grace period. |
| OpenAI Realtime org limits and latency under load | Probably the true ceiling, and it sits outside GCP. |
| Twilio outbound CPS, number pool, spam labeling | 5.6 CPS may exceed defaults; concentrated dialing lowers answer rate. |
| Control plane capacity: config fetch, CRM, webhooks, DB pool | Every call hits these at start; a burst hits them all at once. |
| GCP regional quotas (vCPU, IPs, node pool size) | Raise weeks before, not on the day. |

On ceilings: in production I set the operating ceiling at about half the configured max and kept the rest as burst headroom. The configured max is where things fail, not where you plan to run.

## 4. Assumptions and math

Assume 1.5 vCPU per session, the figure that fixed turn detection for me under process isolation. A packed pipeline with DSP in native code could be far cheaper, which is the first thing I would measure.

- 4,000 calls × 1.5 ≈ 6,000 vCPU at peak; 10,000 calls ≈ 15,000 vCPU.
- With half the configured max as the operating ceiling, configured capacity is roughly double that.
- If a packed design reaches 0.25 vCPU per call, the 4,000 case drops to about 1,000 vCPU.

The spread is the point: answer rate and CPU per session dominate the bill by an order of magnitude, and both need measurement before any provisioning decision.

## 5. Infrastructure levers

The strongest evidence I have is about ramp. In a capacity test on a voice platform I built, across roughly 1,200 sessions, every failure was a call that connected but never got the agent's greeting during cold scale up. None were rate limits. Launching about one session per second collapsed around 25 concurrent with roughly 40% failing. Pacing to one session every 4 seconds held 50 concurrent under 1% failure. Once the pool was warm, reply latency was flat from 5 to 50 concurrent, about 1.6 s including endpointing. Cold start was the problem, not load, and 5.6 dials per second is far above what cold scale up tolerated there.

- **Pre warm from the forecast.** Bring peak capacity up before minute 0, with processes ready to take a call. Reactive autoscaling arrives after callers heard silence.
- **Admission control on readiness and headroom.** Dispatch only to workers that are warm and below threshold (p99 frame time, loop lag). When nothing is ready, the dialer waits. That test is why I built a "burst admission" test mode: launch at the target rate and verify every call gets its greeting.
- **Guaranteed CPU.** Whole cores, the static CPU manager on GKE, a dedicated compute optimized pool. No shared cores, no Spot (preemption kills live calls).
- **Drain, do not kill.** Grace period above the duration tail; keep the cluster autoscaler off busy nodes.
- **Retry discipline.** An overall deadline, capped retries on 5xx only, jitter (section 7).

## 6. Scheduling and product levers

My first question is why 30 minutes. If it is campaign configuration rather than a real constraint, spreading over 2 hours drops the peak to 10,000 / 120 × 40 ≈ 3,300 even at 100% answer, and about 1,300 at 40%.

- **Pacing dialer, closed loop.** Dial rate follows ready capacity and live quality signals, ramping like the paced test above rather than flat out.
- **Priority ordering.** If the dialer slows, patients nearest their renewal deadline go first.
- **Voicemail handling.** Carrier AMD alone is fast but imperfect. What I've shipped is hybrid: carrier AMD, then an LLM classifier on the first caller turn as confirmation. This keeps machine answered calls from holding a full session.
- **Endpointing tuned per turn, not on averages.** On a platform I built, per turn timelines (caller silence to first agent audio, split into spans) exposed two hidden fixed waits; one hit about 10% of turns and added over a second. Averages hid both. I also rejected a tighter VAD stop that saved about 10 ms because it caused more mid sentence cut ins per call. Under burst, protect turn taking first.

## 7. What breaks first, and how I would see it

My order:

1. **Cold scale up.** Calls connect and get no greeting. This matches what I saw.
2. **Retry amplification on the control plane.** In a load test, a slowdown in a platform API became an outage because every session retried: roughly 3x the calls, the database connection pool exhausted, almost all runtime requests failing with p95 near 50 s. A burst plus naive retries is a self inflicted DDoS on config fetch, CRM and webhooks. Fix: deadlines, capped retries, jitter, circuit breakers.
3. **CPU per session too thin**, seen first as slow end of turn, then as per process saturation.
4. **OpenAI and Twilio limits**, as latency, refused sessions, or a quietly lower answer rate.

Availability metrics miss the worst failures. I've seen answering machine detection silently dead for a while because an exception was swallowed; a call where inbound audio went to exact digital silence for about 50 s while the carrier reported it completed; and a misclassified provider error producing about 14 s of dead air. Uptime dashboards stayed green through all three.

Detection, with an SLO on audio and turns rather than uptime:

- Time from answer to first agent audio, per call. A missing greeting is the burst failure signature.
- Per call audio level and dead air duration in both directions; alert on sustained silence.
- Per turn latency spans (end of speech, model first token, first audio out) at p99.
- Event loop lag, cgroup throttled time, CPU per session.
- Retry rate and DB pool saturation per dependency.
- AMD outcome rates; a flat line means it is broken, not that nobody has voicemail.
- Synthetic canary calls throughout the burst measuring gaps and greeting time.

## 8. What I would do next with more time

Run burst admission tests at increasing launch rates with replayed caller audio to find the safe ramp and the warm pool size. Measure CPU per session for process isolation versus packed workers. Run an OpenAI session concurrency test, the ceiling I control least. Then rehearse a reduced scale burst with the closed loop dialer and canaries on.
