# What the first runs found

A ramp from 1 to 32 cascaded agents, 60 seconds a step, against a local router with the
`mock/*` providers. Reports in `out/before` and `out/after`.

## Replies were never the problem; opening a session was

Reply latency was flat at every level: p50 1360-1400 ms, p95 1400-1900 ms, and every turn
answered bar one. `POST /v1/agents/sessions` was not. Its p95 grew with concurrency:

| agents | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| before | 2164ms | 2207ms | 2659ms | 4290ms | 7088ms | 13421ms |
| after | 2069ms | 9713ms | 2030ms | 1950ms | 2150ms | 2117ms |

Server-side the 32 durations laddered evenly from 2.1 s to 12.3 s, about 330 ms apart:
one unit of work the creates were taking in turn. `conversation.Service.open` held the
service-wide `mu` across three Stream Chat round trips (`CreateMissingUsers`,
`GetOrCreateChannel`, then the history read), so every conversation opened anywhere queued
behind every other, and anything else wanting that lock waited behind a stranger's
round trip, up to the function's 20 s budget.

One lock was guarding two unrelated things: the map of live conversations, and the
building of one. A conversation is now opened behind a gate of its own, taken by its key,
so two opens of one conversation are still taken in turn while opens of different ones
never meet. `mu` covers the map and nothing slow.

At 32 agents that is 13.4 s to 2.1 s at p95, and session create no longer grows with
concurrency. The 9.7 s at the 2-agent step is one create of 63 (see below).

What is left of the 2.1 s, measured on the single-agent step where nothing can be
queueing: 490 ms to open the conversation (Stream Chat users, channel, history) and the
database, then 1571 ms of SFU join (`signaling_ms` 951, `subscribe_ms` 210). A create
cannot be faster than the join, because it waits for the agent to be on the call.

Three other places hold a lock across I/O. None is the same bug:
`session/plugin_tools.go` is scoped to one session and the lock is what stops two MCP
sessions being opened for one plugin; `api/popularity.go` holds one across a single
indexed query, which coalesces concurrent callers, though a request for one modality does
wait on another's refresh; `pluginevents.Reconcile` is the real repeat — process-wide
across store reads and `subscribe`, where its own comment says it wants to serialise *one
config*. It is a background reconciler, so no request waits on it.

## sync.Pool: measured, not warranted

The per-frame and per-chunk paths each allocate exactly one buffer, and the router is
nowhere near allocation-bound:

```
BenchmarkFromBytes        348 ns/op     640 B/op   1 allocs/op
BenchmarkBytes            313 ns/op     640 B/op   1 allocs/op
BenchmarkResampleDown     886 ns/op     640 B/op   1 allocs/op
BenchmarkLevel           1737 ns/op    1024 B/op   1 allocs/op
BenchmarkAudioMessage    6051 ns/op   19456 B/op   2 allocs/op
```

Under 32 concurrent conversations (`GODEBUG=gctrace=1`) the heap sat at 105-210 MB, a
cycle ran every 4 seconds, and the collector used 0% of CPU. Pooling the audio buffers
would remove allocations that cost nothing and would need a retention contract across the
`agent.Edge` interface that the loopback and test edges do not keep. Worth revisiting only
when a run is CPU-bound with GC in the profile.

## Still open, in the order worth taking them

1. **One create in 63 stalled 9.7 s** while a second one beside it took 2.1 s. It was the
   first step after the previous one's sessions were torn down, and the stall was before
   the agent started, so it is neither the lock nor the SFU join (`join_ms` p95 1078 ms
   throughout). Worth a goroutine dump taken during a step change.
2. **A Stream client per session.** 65 `client.initialized` for 63 sessions: every
   `streamedge` edge builds its own through `rtc.NewRTCClient`, minting a token and
   building the Chat and Video halves each time. They share the transport, so this is CPU
   and garbage rather than connections, but one client per identity would do.
3. ~~**`MaxIdleConnsPerHost` is 2.**~~ Fixed in the loop below, for both the Stream API
   client and the default transport the WebRTC SDK signals through.
4. **Turn stages are dominated by waiting, not work.** Per turn: cadence 350 ms, decision
   251 ms, model to first text 250 ms, text to TTS 179 ms, TTS to first audio 306 ms,
   transcript to audio 1340 ms. The mock providers answer in 250 ms by construction, so
   what is left to shave is the 350 ms cadence window and the 179 ms between the model's
   first text and the voice taking it.
5. **A bench shape to know about.** Half the turns report `decision_ms` near 8.4 s: the
   caller here talks every 2 seconds whether or not the agent is still speaking, and those
   turns queue until it finishes. That is the caller being rude, not the router being slow.
   A shorter mock answer would make the number mean something.

Once the trace collector lands, redo this from spans: these numbers come from the bench's
client-side latencies and the router's own `voice turn timing` and
`call connection timing` lines.

# Ten iterations on the benchmark cluster

Each iteration: profile the router under load (pprof on the pods' loopback), fix the top
cost, `rocky fastdeploy` to `accelerate-benchmark`, then ramp 4 to 1024 agents at 60 s a
step. The router is pinned to two pods on two n4a-standard-4 nodes (4 vCPU, $0.154/h each,
on demand in us-east1); iterations 0-2 ran on one node. From iteration 4 the callers run in
the benchmark cluster itself (pod `perfbench`, 16 cores, joins at 20/s): from the laptop
its uplink queued 1.7 s of audio under load and the caller-side numbers measured that.

| iter | change | held | cores at 256 | reply p95 at 512 | $/voice-min (node cost ÷ held) |
|---|---|---|---|---|---|
| 0 | baseline (1 node, laptop callers) | 128 | - | - | 0.0000201 |
| 1 | Opus encoded at the TTS rate, not resampled to 48 kHz | 64* | - | - | 0.0000401 |
| 2 | Chat users trusted for 10 min, no QueryUsers per session | 256 | 3.17 | 2360 ms (285 joined) | 0.0000100 |
| 3 | bounded Postgres pool (was 2 idle, redialling TLS) | 64* | - | - | 0.0000802 |
| 4 | one GetOrCreateChannel per session start (was 3) | 256 | 4.04 | 3740 ms | 0.0000201 |
| 5 | speaker wakes its writer once under the bound, not every frame | 256 | 4.03 | 3560 ms | 0.0000201 |
| 6 | Opus low-delay CELT at complexity 4 | 256 | 3.54 | 3820 ms | 0.0000201 |
| 7 | Stream API client keeps 256 idle connections a host | 256 | 3.61 | 4040 ms | 0.0000201 |
| 8 | default transport (SDK SFU signalling) keeps 256 idle | 256 | 2.67 | 4020 ms | 0.0000201 |
| 9 | ndots 1 on the router pods (chart) | **512** | 3.46 | 2980 ms | **0.0000100** |
| 10 | TLS session resumption | 256 | 3.04 | 3480 ms | 0.0000201 |
| 10b | conversations let go after release; tick copies only to send | 256 | 3.34 | 3240 ms | 0.0000201 |
| 11 | no TURN relays, stats decoded once, Opus DTX, turns and call events in Redis | 512, 256† | 2.55, 3.03 | 2500, 2440 ms | 0.0000100-0.0000201 |
| 12 | Cloud NAT 32768 ports a VM, 2048 connections a Stream host | 512 | 3.26 | 2767 ms | 0.0000100 |
| 13 | Chat limits raised to 20k/min; a refused live update waits out Retry-After, not every tick | 512‡ | 2.76 | 1180 ms | 0.0000100 |

\* Iterations 1 and 3 broke on Stream rate limits (a draining old router, then Chat
GetOrCreateChannel 429s), not on the router; the next iteration removed each cause.

† Iteration 11 is two clean runs. The second broke at 512 on joins (43 of 512, 8%), all
timeouts dialling Stream's coordinator socket or posting to its API, with reply p95 at 2.44 s
and the router at 5 cores. Against iterations 9-10b, router CPU per joined agent fell 14% at
256 and 11% at 512, and reply p95 at 512 fell 20-25%. The agent's track now reaches the
caller about 260 ms later at every step (p50 700 ms, was 430-450 ms). It is a fixed cost,
not load; replies are not slower.

The join failures at 512 were Cloud NAT, not Stream's coordinator. Stream allows this app
1,000 `SendMessage` and 1,000 `UpdateMessage` a minute server-side, and the chat log sends
at least two messages a turn, so at 256 agents most sends get a 429. Stream answers many of
those 429s with `Connection: close`. Every one costs a new connection to the same edge
address, about 440 a second per pod, which filled the node's 8192 NAT ports (13,000 to 22,000
sockets in TIME_WAIT). Cloud NAT then dropped every new dial to that address: the
coordinator join, the SFU signalling and the chat posts alike. `max_ports_per_vm` is now
32768 and the router keeps 2048 connections a Stream host, which moved the wall. Below
roughly 1,000 sends a minute, about 80 concurrent agents on one Stream app, nothing of this
shows; above it the transcripts in Chat are mostly lost.

‡ Iteration 13 held 512 with room: reply p95 1.18 s (was 2.77 s), session p95 2.6 s (was
5.7 s), and at 1024 96% of turns were answered (was 37%). 258 of 1024 still never joined,
almost all dialling the coordinator socket (`dial tcp 8.233.33.37:443: i/o timeout`), with
no NAT drops and at most 2,300 NAT ports in use a node, so the wall is no longer NAT. The
routers were at 6.8 of their 8 cores. The only Chat 429s left came at teardown: 1024
conversations storing their last lines in the same 4 s lost 360 of them.

A run whose callers die mid-step leaves its sessions running: a voice session does not end
when the human leaves the call. One aborted run held 346 sessions (260 mCPU, 10,000
goroutines) on a pod with nobody on them, until the pod was restarted.

"Held" is the last step where every agent joined, under 3 s reply p95, and nearly every
turn was answered. Reply p50 sat at 1.04-1.08 s throughout; the mock models answer in
250 ms, so most of that is the cadence and the voice's own pacing.

## What 512 is

From iteration 9 the router reaches 512 at 5.9-6.3 cores of the 7.84 two nodes allocate,
beside the gateway and the system pods: the nodes are full, and reply p95 sits on the 3 s
line (2.98, 3.48, 3.24 s across three runs). 512 is the ceiling for two nodes; 256 holds
with p95 1.14 s at about 3 cores. Hosting alone is $0.0000100 a voice minute at 512
($0.0006 a voice hour), $0.0000201 at 256. Counting only the CPU the router used, it is
$0.0000074 at 512.

Per agent the router went from 18.8 millicores (iteration 0, 128 agents) to 11.6 at 512.
Run to run, the same step moves by about 10%, so compare at the same step across several
iterations rather than one pair.

## Where the CPU goes at 512 (iteration 10b)

| what | share |
|---|---|
| Opus encode (the agent's voice) | 23% |
| Opus decode (the caller's audio) | 11% |
| GC | 8.7% |
| TLS handshakes (new joins) | 3.7% (was 17% in iteration 7) |
| conversation tick loop | 1.6% (was 5.5% in iteration 10) |

## Still open

1. **Opus encode and decode are a third of the CPU.** gopus at complexity 4 is close to
   the floor in pure Go; the next step is not encoding silence (DTX) or a cgo libopus.
2. **2,500 TURN refresh timers at 512 agents.** Every agent gathers TURN relay candidates
   and refreshes them; for a router beside the SFU they are never used. A WebRTC SDK option
   to skip TURN would remove them.
3. **The SDK's `decodeRtcStats` unmarshals each report twice** (about 8% of CPU in
   iteration 6's profile).
4. **The bench's join path is the next wall past 512**: Stream connect 429s and
   SFU requests waiting for a connection at 1024, before the router's CPU is.
