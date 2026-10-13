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
behind every other. `mu` now guards the map it protects and is let go of before Stream is
asked anything, and an open that loses the race is told the conversation is already open,
which is what it would have been told had the two gone one after the other.

At 32 agents that is 13.4 s to 2.1 s at p95, and session create no longer grows with
concurrency. The 9.7 s at the 2-agent step is one create of 63 (see below).

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
3. **`MaxIdleConnsPerHost` is 2.** `streamapp.NewClients` clones
   `http.DefaultTransport`, so concurrent Stream calls beyond two keep re-handshaking TLS
   instead of reusing a connection. Nothing in these runs is pinned on it yet, but it is
   one field.
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
