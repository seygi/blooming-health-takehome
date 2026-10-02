# Q1: Capacity Under a Burst

## The short version

- This is a real time deadline problem, not a throughput problem. Choppy audio means a 20 ms frame missed its deadline somewhere. Average CPU utilization will look healthy while that happens.
- Because the dial window (30 min) is shorter than the call (40 min), peak concurrency equals the number of answered calls. Nobody hangs up before the last call is placed.
- The burst is scheduled. Capacity should be provisioned before the window from a forecast, not discovered by an autoscaler during it.
- The cheapest throttle is the dialer. Admission control should protect calls already in flight by refusing or delaying new ones.

## 1. Reframe: quality fails before availability because of deadlines

Every call produces a frame every 20 ms that must be decoded, denoised, run through VAD and sent to the model, while model audio must reach Twilio on time. If any step is late, the caller hears a gap. Nothing errors, so availability metrics stay green.

Likely causes, in the order I would suspect them:

- **One Python process, many calls.** Pipecat runs on asyncio. DSP inside the event loop, plus base64 and JSON for audio over WebSockets, all compete for one GIL. A process with 30 calls at 7% of a core each needs 2 cores and has 1. The node shows 30% CPU; every call in that process stutters.
- **CFS throttling.** A container CPU limit is enforced per 100 ms period. Burn the quota in 70 ms and the container is paused for 30 ms, which is one or two lost frames on every call it hosts.
- **Event loop lag, GC pauses, noisy neighbours**, which hit every call in a process or node at once.
- **Network jitter** on the WebSocket to OpenAI. This one is outside our CPU entirely and looks identical from the caller's side.

So the signal is tail latency per frame, per process. Utilization lies.

## 2. The shape of the load

Arrival rate: 10,000 calls / 30 min = 333 per minute, about 5.6 dials per second.

Little's law gives steady state concurrency of 333 × 40 = 13,300, but we never reach steady state. The last call is placed at minute 30 and the first ends around minute 40, so peak concurrency is simply the answered count. It holds from minute 30 to about 40, then drains until about minute 70 (longer, given the duration tail).

The brief gives no answer rate, so I carry two cases:

| Case | Answered (live) | Peak concurrent long calls |
|---|---|---|
| Design ceiling (as stated) | 100% | 10,000 |
| Working assumption | 40% | 4,000 |

Unanswered calls are not free: 5.6/s × 25 s of ringing is about 140 legs at any moment, plus voicemail drops. Small, unless a pipeline and model session start at dial instead of at answer.

One useful consequence: inside a 30 minute window, shortening calls does not lower the peak. It shortens how long you sit at it.

## 3. Numbers I need, and what each one changes

| Number | What I would do with it |
|---|---|
| CPU ms per 20 ms frame per call (p50, p99), noise suppression + VAD + codec + serialization | Calls per vCPU. Pack to a headroom target (60 to 70%) chosen so p99 frame time stays under budget, not to average utilization. |
| Where that CPU runs (in the event loop or in native code that releases the GIL) | Calls per process, which is likely the real limit before calls per node. |
| Answer rate, voicemail rate, AMD time, by hour and population | Turns 10,000 dials into a real peak. Also tells me whether to start the model session at dial or at answer. |
| Call duration distribution, not the average | The tail sets drain time and the grace period for scale in. |
| OpenAI Realtime limits for the org: concurrent sessions, rate limits, latency under load | Probably the true ceiling, and it is outside GCP. If the org cannot hold 4,000 to 10,000 sessions, no amount of GKE capacity matters. |
| Twilio outbound CPS for the account and numbers, number pool size, spam labeling | 5.6 CPS sustained may exceed default limits. Concentrated dialing from a few numbers gets labeled as spam, which lowers the answer rate. |
| Bandwidth and memory per call | Per node NIC and memory ceilings. G.711 at 8 kHz is 64 kbps each way before base64 overhead; 24 kHz PCM16 to the model is several times that. |
| Instance and pod ready time (node boot, image pull, model load) | How far ahead of the window to pre-warm. |
| GCP regional quotas (vCPU, IPs, node pool size) | A hard stop that should be raised weeks before, not discovered on the day. |

## 4. Assumptions to get moving

Assume 1.5 ms of CPU per 20 ms frame per call, all in. That is 7.5% of a vCPU, so about 13 calls per vCPU in theory and about 8 at a safe headroom.

- 10,000 calls / 8 ≈ 1,250 vCPU at peak.
- 4,000 calls / 8 ≈ 500 vCPU.

If base64, resampling and JSON double the per frame cost (plausible in Python), so does the fleet. That is why the first measurement is per frame CPU under realistic load, not a DSP benchmark.

## 5. Infrastructure levers

- **Pre-warm from the forecast.** The window is known, so bring up peak capacity before minute 0. CPU based autoscaling reacts after frames are already late, and at the design ceiling the ramp adds about 40 vCPU of demand per minute.
- **Admission control on measured headroom.** Each worker publishes its p99 frame time and event loop lag. The dispatcher only places a new call on a worker below its threshold, and the threshold sits below the load test knee. Refuse new calls before existing ones degrade, never after.
- **Bound calls per process.** One process per core with a fixed call cap, or move noise suppression and VAD into native code (or a Go/Rust sidecar) that does not hold the GIL.
- **No CPU limit throttling.** Guaranteed QoS with whole cores and the static CPU manager on GKE, on a dedicated node pool with compute optimized machines. No shared core types, no Spot (preemption kills live calls).
- **Drain, do not kill.** With 40 minute calls, scale in means stop admitting, then wait. Set `terminationGracePeriodSeconds` above the duration tail and keep the cluster autoscaler from evicting busy nodes.
- **Spread across zones**, in a region close to Twilio's media edge and OpenAI.
- **Cut work per frame.** If the model takes G.711 directly and noise suppression works at 8 kHz, skip resampling.

## 6. This is partly a scheduling and product problem

My first question is why 30 minutes. If it is campaign configuration rather than a real constraint (patient availability, staff for warm transfers), spreading the same 10,000 dials over 2 hours drops the peak to 10,000 / 120 × 40 ≈ 3,300 even at a 100% answer rate, and about 1,300 at 40%.

Other levers on that side:

- **A pacing dialer driven by live capacity.** Closed loop: when fleet frame deadline misses or loop lag rise, the dialer slows. Dial in waves weighted by expected answer rate per time slot.
- **Priority ordering**: if the dialer slows, patients closest to their renewal deadline go first.
- **Graceful degradation per call:** measure line noise in the first seconds and skip noise suppression on clean lines; under pressure, fall back to the model's server side VAD instead of local VAD.

## 7. What breaks first, and how I would see it

My guess at the order:

1. Per process saturation (GIL, event loop) giving choppy audio while node CPU looks fine.
2. OpenAI session or rate limits, showing up as latency first and then refused sessions.
3. Twilio CPS and spam labeling, which quietly throttle the dial rate and lower answer rate.

Detection, with an SLO on audio quality rather than uptime (for example, 99.9% of frames processed within budget per call):

- Per frame processing time histogram, per process.
- Event loop lag, GC pause time, cgroup `nr_throttled` and throttled time.
- Late or dropped frames toward Twilio, jitter buffer underruns, inter arrival jitter of model audio on the WebSocket.
- Conversation level signals: callers saying "hello?", talk over, repeated questions.
- Synthetic canary calls placed throughout the burst that measure gaps in the received audio.

Alert on these leading indicators so the dialer slows before anyone hears a gap.

## 8. What I would do next with more time

Load test: replay recorded caller audio into workers at rising concurrency per process and per node, find the knee where p99 frame time crosses budget, and set the admission limit below it. Separately, run a coordinated OpenAI session concurrency test, the ceiling I control least. Then rehearse a reduced scale burst with the pacing dialer in closed loop.
