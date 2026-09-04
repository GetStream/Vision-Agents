# Voicebench

Voicebench evaluates real-time voice agents on restaurant, healthcare, and telecom calls. It plays a scripted caller over Stream WebRTC, records both sides at 16 kHz, and checks speech, tools, and the final state of a seeded world.

The same scenarios evaluate the Python [Vision Agents](https://visionagents.ai) reference agents, the Go router in [`acceleration/`](../acceleration/) (raw session API or the SDK `stream.Accelerated` bundle), a LiveKit agent dispatch, or another agent that joins the call. Design notes for STT, TTS, and tracking acceleration over time live in [RESEARCH.md](RESEARCH.md).

## Setup

Live evaluations require Go 1.26, CGO, and the native libraries listed in [`acceleration/`](../acceleration/README.md):

```bash
# macOS
brew install pkg-config opus opusfile libsoxr

# Debian/Ubuntu
sudo apt-get install -y pkg-config libopus-dev libopusfile-dev libsoxr-dev
```

Put credentials in `benchmark/.env` or the repository root:

```bash
STREAM_API_KEY=...
STREAM_API_SECRET=...
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
LIVEKIT_AGENT_NAME=...
GOOGLE_API_KEY=...
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
STREAM_ACCELERATION_URL=...   # for --target accelerated
```

ElevenLabs creates caller audio with no tone fallback. Deepgram Nova-3 transcribes both recordings, and the pinned `gpt-4.1-mini-2025-04-14` grades policy and say-do consistency. Missing caller TTS, agent STT, or judge output makes the trial invalid and fails the run; `--skip-stt` and `--skip-judge` intentionally produce invalid trials. Caller-leg STT is diagnostic only.

## Run

From `benchmark/`, evaluate Vision Agents:

```bash
cd agents && uv sync && cd ..
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target python --spawn --k 3
```

Evaluate the shipped acceleration bundle (`stream.Accelerated` in Python, function calling still in Python). The router must already be running, or pass `--bin` to spawn it:

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target accelerated --spawn --k 3
```

Evaluate acceleration through its public session API. The router is unmodified: Voicebench creates a session per trial, answers `tool_call` frames against the world server, and closes the session.

```bash
cd ../acceleration
CGO_ENABLED=1 go build -o /tmp/accel-router ./cmd/router
cd ../benchmark
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target acceleration --spawn --bin /tmp/accel-router --k 3
```

Evaluate LiveKit with the reference worker in [`agents-livekit/`](agents-livekit/). It registers as `voicebench`, builds its tools from the dispatch metadata, and answers with the same OpenAI Realtime model (`gpt-realtime-2`) and voice (`marin`) as the Vision Agents reference agents, so a Stream-vs-LiveKit gap reflects the framework and transport rather than the model. `agents-livekit/uv.lock` records the stack that ran.

```bash
cd agents-livekit && uv sync && cd ..
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target livekit --spawn --k 3
```

`--spawn` runs the worker locally; it dials out to LiveKit, so the loopback world server stays reachable. `--livekit-agent your-agent` dispatches your own worker instead. That worker must read `world_url`, the vertical `pack`, contract `instructions`, and tool schemas from the dispatch metadata and call the world server itself, and it needs a `--world-url` it can reach — Voicebench refuses to run a remote worker against a loopback world server. A worker that ignores the metadata will call no tools, so Voicebench flags the trial in the report's Warnings section.

`--target acceleration --target-url http://127.0.0.1:8080` targets a router that is already running. `--target python --target-url http://127.0.0.1:8000` targets a Python Vision Agents server that is already running. `--target accelerated --target-url http://127.0.0.1:8000` targets a Python server whose LLM is `stream.Accelerated`. `--target livekit --target-url wss://... --livekit-agent my-agent` runs the caller in a LiveKit room and creates a LiveKit agent dispatch with benchmark metadata. For targets Voicebench did not spawn, pass `--target-model` and `--target-voice` so the manifest identifies their runtime configuration. Set `--network-profile` (or `VOICEBENCH_NETWORK_PROFILE`) to a stable runner-region and connection label shared by comparable runs. For a quick smoke test, add `--scenario restaurant.golden --k 1`. `--frozen` runs only the scenario ids in [`scenarios/frozen.txt`](scenarios/frozen.txt).

Results go to `out/<run_id>/`: `report.md`, schema-v3 `summary.json` with a `kind` of `agent`, `stt`, or `tts`, a reproducibility manifest, recordings, timestamped transcripts, judge verdicts, tool logs, world state, and per-call metrics. Compare two runs with:

```bash
go run ./cmd/voicebench compare --baseline out/old out/new --mde-v2v-ms 50
```

Score transcripts (raw and normalized WER) or clip health without a live call:

```bash
go run ./cmd/voicebench stt --manifest clips.jsonl
go run ./cmd/voicebench tts --wav out/run/agent.wav
```

Regenerate a report with:

```bash
go run ./cmd/voicebench report --dir out/<run_id>
```

## Test

```bash
# Voicebench
go test ./...
go test -tags webrtc ./...

# acceleration
(cd ../acceleration && go test ./...)

# Vision Agents
(cd .. && uv run --no-sync pytest -m "not integration")
```

## Judge calibration

The pinned judge must pass the maintained labeled set before a baseline is publishable:

```bash
go run ./cmd/voicebench calibrate \
  --out out/judge-calibration.json
```

A maintainer must review the labels in [`calibration/judge.json`](calibration/judge.json) and set `reviewed_by`. Calibration requires at least 90% agreement across the labeled policy, say-do, and coherence decisions and no missed critical policy violation. Exact-case agreement is also reported. The output records the judge model, fixture hash, verdicts, and disagreements.

## Methodology

1. **Seed:** A YAML scenario creates known inventory, patient records, or subscriber state.
2. **Call:** A synthesized caller follows a timed script. Noise tests add kitchen, street, or competing speech at 10 dB SNR.
3. **Act:** Every implementation receives the same prompt, tools, data, and caller audio. Each trial uses a new call and empty history.
4. **Observe:** Voicebench records both legs, tool calls, and final world state. Timing comes from speech energy in the recordings.
5. **Grade:** Deterministic checks cover state, expected tools and arguments, tool order, and entities. An LLM judge checks policy, coherence, and claims against successful tools. The scripted caller text is the canonical caller transcript for judging; caller STT remains a diagnostic artifact.
6. **Repeat:** Each scenario runs `k` times, three by default. Reliability is calculated per scenario: `pass@k` means any requested trial passed and `pass^k` means every requested trial passed. Evaluator failures are invalid, make the scenario incomplete, and never count as agent failures.

A trial passes only when every hard gate passes. Latency is reported separately, so speed cannot hide an incorrect result.

Not every scripted turn yields a latency sample. A barge-in turn has no reply gap by definition, a turn the caller played while the agent was still speaking has no meaningful one, and a turn the agent never answered has none at all. Those turns are counted and named in `dropped_turns` in each call's `metrics.json` and totalled per pack in the report, so a P50 cannot quietly rest on one measurement. Percentiles are pooled over every measured turn in the pack — not a median of per-call medians — and every reported P50 carries its sample count.

## Metrics and Voicebench targets

There is no single industry-standard score across these verticals. Voicebench targets are fixed acceptance thresholds for this suite, defined in [`timing.go`](internal/score/timing.go) and [`board.go`](internal/report/board.go). They are not universal industry standards, compliance certification, or claims of state of the art.

| Metric | Measurement | Voicebench target | Gate |
| --- | --- | --- | --- |
| Task success | Required final world state | Every assertion passes | Yes |
| Tool order | Scenario before/after constraints | Every constraint passes | Yes |
| Entity fidelity | Required values in tool arguments and speech | No missing or changed value | Yes |
| Policy and say-do | Policy breaks or unsupported claims | Zero failures | Yes |
| Tool filler | Filler begins before a delayed tool returns | Heard without blocking | Yes |
| Barge-in | Interruption to agent silence | ≤ 800 ms | Yes |
| Selectivity | Ignore coughs and side talk; accept real interruptions | Hold on non-directed speech | Yes |
| Reply gap | Caller end to agent onset, excluding tool turns | P50 300–700 ms | No |
| Voice-to-voice | Caller end to agent onset, every measurable turn | P50 300–700 ms; P95 and sample count reported | No |
| Stability | Non-tool gap over 2× that call's P50 | Zero spikes | No |
| False cutoffs | Agent starts while caller is speaking | Zero | No |
| Reliability | Repeated runs | `pass^k`; default target 3/3 | Aggregate |

### Restaurant

| What matters | What is evaluated | Target |
| --- | --- | --- |
| Task accuracy | Availability, booking or order, and final state | All state and entity gates pass |
| Inventory integrity | No invented table, overbooking, or unavailable item | Zero policy failures |
| Safety details | Allergen retained through changes and read back | Exact value in tools and speech |
| Dense details | Name, time, party, seating, phone, and pickup | Zero entity failures |

See the [restaurant contract](agents/contracts/restaurant.prompt).

### Healthcare

| What matters | What is evaluated | Target |
| --- | --- | --- |
| Identity and privacy | Verification before protected information | No disclosure before verification |
| Patient isolation | Similar records remain separate | Zero cross-patient disclosure |
| Workflow accuracy | Appointment and insurance changes | Exact final state and tool values |
| Safety policy | Refuse controlled substances, avoid diagnosis, escalate | Zero policy failures |

See the [healthcare contract](agents/contracts/healthcare.prompt). Its data-minimization scenarios follow the US HHS [HIPAA minimum necessary principle](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html); this is not a compliance certification.

### Telecom

| What matters | What is evaluated | Target |
| --- | --- | --- |
| Account security | PIN, last four, and address before changes | Exact values before protected tools |
| Repair flow | Outage check, reboot, ticket, then dispatch | Reboot before dispatch |
| Confirmation | Announce actions only after tool success | Zero say-do failures |
| Commercial policy | No ineligible credit or coerced plan change | Zero policy failures |
| Handoff | Three-line transfer summary after repair | Summary exists and is concise |

See the [telecom contract](agents/contracts/telecom.prompt).

Each vertical includes task completion, two-minute coherence, 10 dB noise, competing-talker selectivity, delayed-tool filler, interruption, entity-dense, and adversarial scenarios. The three verticals remain separate score columns; they are never combined into one score.

## Scenario world

Voicebench starts a local world server for every run. Each scenario seeds that server with restaurant, healthcare, or telecom state. Tool calls mutate the world, and final scoring checks both the recorded conversation and the final world state. Python Vision Agents call the world server directly through `VOICEBENCH_WORLD_URL`; acceleration emits websocket `tool_call` events, and Voicebench forwards those calls to the same world server before returning `tool_result` frames.

## Evaluate another agent

HTTP agents (including the Python reference agents) implement the selected contract's tools against `POST $VOICEBENCH_WORLD_URL/v1/session/tools/{name}`, join the Stream call, then run:

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target python --target-url http://127.0.0.1:8000 --system your-agent --k 1
```

`--spawn` sets `VOICEBENCH_WORLD_URL` on a supported local child. Session-API agents declare the same tools on `POST /v1/agents/sessions`; Voicebench forwards `tool_call` frames to the world server. Use `--target acceleration --target-url ...` for an already-running router, or `--call-id` for an agent already in the call.

`summary.json` schema version 2 is the leaderboard ingest contract. It contains per-scenario reliability and a manifest fingerprinting the source, scenarios, contracts, transport, models, caller voice, and runtime.

## Comparing runs

A LiveKit column is only comparable when the worker actually received the contract: check the report for zero tool calls and Warnings before reading its score. Trials that produce no verdict are invalid, make scenario reliability incomplete, and fail the run. A comparable run should use matching manifest values: methodology version, scenario and contract hashes, `k`, target and transport, target model and voice, caller configuration, region/network conditions, and evaluator configuration. `summary.json` records these fields without credentials. Voicebench scores are directly comparable to other Voicebench runs under the same setup; they are not directly comparable to EVA, τ²-bench, eot-bench, or other benchmark scores.

## Public benchmark basis

Voicebench combines ideas from public work; it does not import their datasets or reproduce any one benchmark:

- [ServiceNow EVA](https://github.com/ServiceNow/eva) — repeated trials and pass@k / pass^k
- [τ²-bench](https://github.com/sierra-research/tau2-bench) — scripted users, stateful worlds, and outcome-based success
- [VAmoS reference agents](https://github.com/veris-ai/riley-agent) — live calls and descriptive latency
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) — interruption and overlap handling
- [LiveKit eot-bench](https://github.com/livekit/eot-bench) — end-of-turn behavior and response timing
- [aiewf-eval](https://github.com/kwindla/aiewf-eval) — recording-based voice-to-voice latency
- [Open voice-agent benchmark discussion](https://www.reddit.com/r/voiceagents/comments/1vjdwxj/independent_open_source_benchmark_of_voice_agent/) — independent open-benchmark motivation
