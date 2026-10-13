---
name: performance_benchmark
description: Run the voice-agent load benchmark in performance_benchmark/ to find how many cascaded agents one router holds, and read or compare its reports. Use when asked how many concurrent agents we support, to load-test or stress the router, to measure a performance change, or to explain a perfbench run.
---

# The load benchmark

`performance_benchmark/` answers one question: how many cascaded voice agents does one
router hold at once, and what gives way first. Every agent in a run is a real
conversation — the Go SDK opens a session, the router joins the call and runs the whole
loop, and a caller in the benchmark joins the same call over WebRTC and talks to it. The
three models are the router's own `mock/*` providers, so a run reaches no vendor and bills
nothing. **What a run measures is the router**: its audio path, turn taking, sockets and
database. It says nothing about model quality; that is Voicebench (`voicebench` skill).

## 1. What a run needs

**Keys.** `STREAM_API_KEY` and `STREAM_API_SECRET`, from the repo-root `.env`. Nothing else:
there is no provider key in a run, by design.

**A router with the mocks on.** The mock providers are built only when
`ROUTER_MOCK_PROVIDERS=true`, so a default router has no `mock/mock-stt` to route to and
every session fails with an unknown model. Check before a run, not after:

```bash
curl -sS "$ROUTER/api/v1/models?capability=stt" | grep -c mock/mock-stt
```

Staging carries `router.mockProviders: true` in `shards/us-east1-acceleration/accelerate-staging.yaml`
in the private infra repo. Locally, `ROUTER_MOCK_PROVIDERS=true` in the root `.env`.

**Native audio.** The callers speak Opus over WebRTC: `brew install pkg-config opus opusfile libsoxr`,
and every command needs `CGO_ENABLED=1` and `-tags webrtc`. Without the tag it builds and
then refuses to place a call.

**`kubectl` on the router's cluster**, for `-namespace` to report pod CPU and restarts. Without
it the run still works and the report just has no pod column.

## 2. Run one

From `performance_benchmark/`. Staging is the default target and the one to use: say so to the
user first, since a run saturates it for its duration.

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/perfbench run \
  -target staging -label before -start 1 -factor 2 -max 64 -hold 90s -out out/before
```

The ramp doubles the agents each step, holds each step for `-hold` of real conversation, and
stops at the first step that broke. Wall time is about `-hold` per step plus joining, so the
default ramp to 64 is around ten minutes. Start with `-max 8 -hold 30s` to check the setup
works before asking for a long one.

| What you want | Change |
|---|---|
| Check the setup end to end | `-max 1 -hold 20s` |
| A number to quote | `-max 128 -hold 120s`, twice, and take the lower |
| A local router | `-target http://localhost:8080 -namespace ""` |
| Finer steps near the ceiling | `-start <last held> -factor 1` is refused; use `-start`/`-max` around it with `-factor 2` |

A step breaks when more than 5% of conversations never get the agent onto the call, more than
10% of turns go unanswered, a reply takes over three seconds at p95, or a router pod restarts.
The step before it is the answer, printed as `Held N agents`.

## 3. Read it

A run writes `out/<run>/report.json` and prints the same thing as Markdown: a row per step
with sessions created, agents that joined, turns answered, and p50/p95/p99 for session
creation, time to the agent's track, reply latency and mid-reply audio gaps, plus the router
pods' CPU and restarts.

The four timings say where a ceiling is:

- **SessionMs** growing: the API and its database, not the media path.
- **TrackMs** growing: the router is slow to get an agent onto a call — dispatch, the SFU join.
- **ReplyMs** growing while the stage spans stay flat: queueing, not a model.
- **GapMs** appearing: the router is producing audio slower than real time. This is the one
  that users hear first.

**The caller can be the bottleneck.** Each conversation costs this machine an Opus encode and
decode. When a step reports `host CPU saturated`, that ceiling is the laptop's, not the
router's: the numbers above it are meaningless. Rerun that step from a bigger machine, or say
in the report that the ceiling was the harness.

## 4. Did a change help?

Two runs, same target, same ramp, same machine, back to back, and the router rebuilt and
redeployed between them (`deploy` skill). Then:

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/perfbench compare out/before out/after
```

It prints the held ceiling before and after and, for every step both runs reached, reply p50
and p95 with the percentage moved. One run is noisy: a move under about 10% on one pair of
runs is not a result, and staging has other traffic. Say how many runs a claim rests on.

For a change meant to cut allocations, a `go test -bench` micro-benchmark on the hot function
is the evidence that it allocates less; this benchmark is the evidence that it mattered.

`performance_benchmark/findings.md` is what the runs so far found, what was fixed and what is
still slow. Read it before chasing a ceiling, and add to it after a run worth remembering.

## 5. Per-turn stages

The router emits a span per turn (`agent.turn`) with a child per stage — `turn.stt`,
`turn.cadence`, `turn.decision`, `turn.llm`, `turn.tts`, `turn.playout` — so a slow reply can
be attributed without guessing. They go nowhere unless the router has
`OTEL_EXPORTER_OTLP_ENDPOINT` set. With the mocks the model stages are fixed costs
(mock STT settles a turn 600ms after speech stops, mock LLM answers in 250ms, mock TTS starts
audio in 150ms), so anything above that in `turn.decision` or `turn.playout` is the router's.

## 6. Report back

```
Ran: <target>, <start>→<max> agents x<factor>, hold <d>, commit <sha>
Held: <n> agents. Broke at <n>: <why>
At <held>: reply p50 <ms> p95 <ms>, track p95 <ms>, session p95 <ms>, gaps p95 <ms>
Router: <cpu>m across <n> pods, <restarts> restarts
Harness: host CPU <busy> of <cores> cores
Next: <the one thing to fix>
```

Never quote a ceiling from a run whose host was saturated, and never quote one from a router
without the mocks: that run measured a vendor.
