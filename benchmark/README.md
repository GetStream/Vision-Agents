# Voicebench

Voicebench evaluates real-time voice agents on restaurant, healthcare, and telecom calls. It plays a scripted caller over Stream WebRTC, records both sides at 16 kHz, and checks speech, tools, and the final state of a seeded world.

The same scenarios evaluate the Python [Vision Agents](https://visionagents.ai) reference agents, the Go agent in [`acceleration/`](../acceleration/), or another agent that joins the call.

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
GOOGLE_API_KEY=...
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
```

ElevenLabs creates caller audio with no tone fallback. Deepgram Nova-3 transcribes both recordings, and `gpt-4.1-mini` grades policy and say-do consistency. Missing TTS, STT, or judge output fails the trial; `--skip-stt` and `--skip-judge` also fail it.

## Run

From `benchmark/`, evaluate Vision Agents:

```bash
cd agents && uv sync && cd ..
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target python --spawn --k 3
```

Evaluate acceleration through its public session API. The router is unmodified: Voicebench creates a session per trial, answers `tool_call` frames against the world server, and closes the session.

```bash
cd ../acceleration
CGO_ENABLED=1 go build -o /tmp/accel-router ./cmd/router
cd ../benchmark
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target acceleration --spawn --bin /tmp/accel-router --k 3
```

`--target acceleration --target-url http://127.0.0.1:8080` targets a router that is already running. `--target python --target-url http://127.0.0.1:8000` targets a Python Vision Agents server that is already running. Legacy `--spawn-agent`, `--agent-url`, `--spawn-accel`, `--accel-url`, and `--accel-bin` flags still work. For a quick smoke test, add `--scenario restaurant.golden --k 1`.

Results go to `out/<run_id>/`: `report.md`, schema-v1 `summary.json`, recordings, transcripts, tool logs, world state, and per-call metrics. Regenerate a report with:

```bash
go run ./cmd/voicebench report --dir out/<run_id>
```

## Test

```bash
# Voicebench
go test ./...

# acceleration
(cd ../acceleration && go test ./...)

# Vision Agents
(cd .. && uv run --no-sync pytest -m "not integration")
```

## Methodology

1. **Seed:** A YAML scenario creates known inventory, patient records, or subscriber state.
2. **Call:** A synthesized caller follows a timed script. Noise tests add kitchen, street, or competing speech at 10 dB SNR.
3. **Act:** Every implementation receives the same prompt, tools, data, and caller audio. Each trial uses a new call and empty history.
4. **Observe:** Voicebench records both legs, tool calls, and final world state. Timing comes from speech energy in the recordings.
5. **Grade:** Deterministic checks cover state, expected tools and arguments, tool order, and entities. An LLM judge checks policy, coherence, and claims against successful tools. The scripted caller text is the canonical caller transcript for judging; caller STT remains a diagnostic artifact.
6. **Repeat:** Each scenario runs `k` times, three by default. `pass@k` means any trial passed; `pass^k` means every trial passed.

A trial passes only when every hard gate passes. Latency is reported separately, so speed cannot hide an incorrect result.

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
| Voice-to-voice | Caller end to agent onset, all turns | P50 300–700 ms; P95 reported | No |
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

See the [restaurant contract](agents/contracts/restaurant.md).

### Healthcare

| What matters | What is evaluated | Target |
| --- | --- | --- |
| Identity and privacy | Verification before protected information | No disclosure before verification |
| Patient isolation | Similar records remain separate | Zero cross-patient disclosure |
| Workflow accuracy | Appointment and insurance changes | Exact final state and tool values |
| Safety policy | Refuse controlled substances, avoid diagnosis, escalate | Zero policy failures |

See the [healthcare contract](agents/contracts/healthcare.md). Its data-minimization scenarios follow the US HHS [HIPAA minimum necessary principle](https://www.hhs.gov/hipaa/for-professionals/privacy/guidance/minimum-necessary-requirement/index.html); this is not a compliance certification.

### Telecom

| What matters | What is evaluated | Target |
| --- | --- | --- |
| Account security | PIN, last four, and address before changes | Exact values before protected tools |
| Repair flow | Outage check, reboot, ticket, then dispatch | Reboot before dispatch |
| Confirmation | Announce actions only after tool success | Zero say-do failures |
| Commercial policy | No ineligible credit or coerced plan change | Zero policy failures |
| Handoff | Three-line transfer summary after repair | Summary exists and is concise |

See the [telecom contract](agents/contracts/telecom.md).

Each vertical includes task completion, two-minute coherence, 10 dB noise, competing-talker selectivity, delayed-tool filler, interruption, entity-dense, and adversarial scenarios. The three verticals remain separate score columns; they are never combined into one score.

## Scenario world

Voicebench starts a local world server for every run. Each scenario seeds that server with restaurant, healthcare, or telecom state. Tool calls mutate the world, and final scoring checks both the recorded conversation and the final world state. Python Vision Agents call the world server directly through `VOICEBENCH_WORLD_URL`; acceleration emits websocket `tool_call` events, and Voicebench forwards those calls to the same world server before returning `tool_result` frames.

## Evaluate another agent

HTTP agents (including the Python reference agents) implement the selected contract's tools against `POST $VOICEBENCH_WORLD_URL/v1/session/tools/{name}`, join the Stream call, then run:

```bash
CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run \
  --pack restaurant --target python --target-url http://127.0.0.1:8000 --system your-agent --k 1
```

`--spawn-agent` sets `VOICEBENCH_WORLD_URL` on the child. Session-API agents declare the same tools on `POST /v1/agents/sessions`; Voicebench forwards `tool_call` frames to the world server. Point `--accel-url` at an already-running router, or `--call-id` at an agent already in the call.

`summary.json` schema version 1 is the leaderboard ingest contract.

## Comparing runs

A comparable run should use the same Voicebench commit, schema version, methodology version, scenario files, packs, `k`, target mode, region/network conditions, and provider configuration. `summary.json` records the benchmark and methodology versions plus the default external providers used for caller TTS, STT, and judging. Voicebench scores are directly comparable to other Voicebench runs under the same setup; they are not directly comparable to EVA, τ²-bench, eot-bench, or other benchmark scores.

## Public benchmark basis

Voicebench combines ideas from public work; it does not import their datasets or reproduce any one benchmark:

- [ServiceNow EVA](https://github.com/ServiceNow/eva) — repeated trials and pass@k / pass^k
- [τ²-bench](https://github.com/sierra-research/tau2-bench) — scripted users, stateful worlds, and outcome-based success
- [VAmoS reference agents](https://github.com/veris-ai/riley-agent) — live calls and descriptive latency
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) — interruption and overlap handling
- [LiveKit eot-bench](https://github.com/livekit/eot-bench) — end-of-turn behavior and response timing
- [aiewf-eval](https://github.com/kwindla/aiewf-eval) — recording-based voice-to-voice latency
- [Open voice-agent benchmark discussion](https://www.reddit.com/r/voiceagents/comments/1vjdwxj/independent_open_source_benchmark_of_voice_agent/) — independent open-benchmark motivation
