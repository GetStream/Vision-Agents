# Voicebench

Go harness for scoring a voice agent on healthcare, restaurant, and telecom calls.

The harness plays a scripted caller over PSTN (Telnyx) or Stream WebRTC, records both legs, and grades the call against a seeded mock world. Any system that accepts the media and points its tools at the world server can be scored.

PSTN and WebRTC are separate columns. Do not mix them in one `--system` score — V2V is not comparable across transports.

## Layout

- `cmd/voicebench` — `synth`, `run`, `report`
- `internal/` — telephony, streamrtc, caller, world, scoring, report
- `scenarios/{restaurant,healthcare,telecom}/` — YAML packs (`noise:` / `snr_db:` mix kitchen, street, or conversation noise generated in `internal/audio`)
- `agents/` — reference Vision Agents (`python -m voicebench_agents <pack>`) that call the world server

## Metrics

Hard AND-gates (any miss fails the trial):

- World end-state assertions
- Tool order (when the YAML lists `tool_order`)
- Say-do (judge: speech vs tool log)
- No policy break (LLM judge, sees transcript + tool log)
- Entity fidelity in tool args and in agent speech
- Filler speech during delayed tools (`one moment` / `checking`, and agent onset before the tool returns)
- Barge-in stop under 800 ms when the script talks over the agent
- Hold through a mid-speech cough or side talker; do not treat a post-turn cough as a new request
- STT, judge, and caller TTS must succeed (`--skip-stt` / `--skip-judge` fail the trial; missing ElevenLabs does not fall back to tones)

Reported, not gated: voice-to-voice P50/P95 ([VAmoS](#prior-art)-style descriptive latency), non-tool P50, false-cutoff proxy, reply-gap vs 300–700 ms on **non-tool** turns, latency spikes vs that call's non-tool P50.

Each scenario is tried `k` times (default 3). The report shows [EVA](#prior-art)-style **pass@k** and **pass^k** per call type per vertical.

## Run (WebRTC, spawned agent)

From `benchmark/`. Needs `-tags webrtc`, `CGO_ENABLED=1`, and the same native libs as acceleration (`pkg-config`, libopus, libopusfile, libsoxr).

```bash
cd agents && uv sync && cd ..
go run -tags webrtc ./cmd/voicebench run --pack restaurant --transport webrtc --spawn-agent --k 3
```

`--spawn-agent` starts `python -m voicebench_agents <pack>` on `--agent-port` (default 8000), waits for `GET /ready`, then POSTs `/calls/{id}/sessions`. Requires a `.env` with `STREAM_API_KEY`, `STREAM_API_SECRET`, `GOOGLE_API_KEY`, `ELEVENLABS_API_KEY`, `DEEPGRAM_API_KEY`, and `OPENAI_API_KEY`. `--skip-stt` / `--skip-judge` skip those calls and **fail** the trial.

Open `out/<run_id>/report.md`. Healthcare and telecom: `--pack healthcare` / `telecom`.

## Run (PSTN)

1. Start ngrok in front of the agent (`NGROK_URL=...`).
2. Start the reference agent (restaurant shown):

```bash
cd agents
uv sync
WORLD_URL=http://127.0.0.1:8090 uv run python -m voicebench_agents restaurant --transport telnyx --setup-telnyx --port 8000
```

3. In another shell, synthesize caller audio once, then dial:

```bash
cd benchmark
go run ./cmd/voicebench synth --pack restaurant
go run ./cmd/voicebench run --pack restaurant --number +1AGENT --from +1HARNESS --k 3
```

The harness binds the world server on `127.0.0.1:8090` and a media websocket on `127.0.0.1:8091`. Telnyx reaches the websocket through `--stream-host` / `NGROK_URL`.

## Run (WebRTC, external agent)

The harness joins the Stream call as `voicebench-caller` and resamples to 8 kHz so scoring matches PSTN. Use `--agent-url` when the agent is already running:

```bash
cd agents
uv sync
WORLD_URL=http://127.0.0.1:8090 uv run python -m voicebench_agents restaurant --transport webrtc
```

```bash
cd benchmark
go run -tags webrtc ./cmd/voicebench run --transport webrtc --pack restaurant \
    --agent-url http://127.0.0.1:8000 --system vision-agents-webrtc --k 3
```

**Agent already in the call** (acceleration `agent -call X`, or any Stream participant):

```bash
go run -tags webrtc ./cmd/voicebench run --transport webrtc --pack restaurant \
    --call-id X --system acceleration-webrtc --k 1
```

Empty `--call-id` generates `vb-{scenario}-{trial}-{hex}` per trial.

## Env

| Variable | Used by |
| --- | --- |
| `TELNYX_API_KEY` | Harness dial + agent answer |
| `TELNYX_CONNECTION_ID` or `TELNYX_CALL_CONTROL_APP_ID` | Outbound Call Control |
| `TELNYX_PHONE_NUMBER` | Harness caller ID (`--from`) |
| `TELNYX_PHONE_NUMBER_ID` | Agent inbound routing |
| `NGROK_URL` | Public host for media / webhooks |
| `ELEVENLABS_API_KEY` | `synth` and live caller speech (required; no tone fallback) |
| `DEEPGRAM_API_KEY` | STT of both legs (`--skip-stt` fails the trial) |
| `OPENAI_API_KEY` | Policy / say-do judge (`--skip-judge` fails the trial) |
| `STREAM_API_KEY` / `STREAM_API_SECRET` / `GOOGLE_API_KEY` | Reference agents and `--transport webrtc` |
| `STREAM_USER_TOKEN` | Optional; used instead of `STREAM_API_SECRET` when joining WebRTC |
| `WORLD_URL` | Agent tools → harness world server. `--spawn-agent` sets this on the child. |

## External systems / leaderboard

Submit a phone number **or** a Stream call, plus a world-server client. The job, tools, and seeded data are the same for every implementation:

- [Restaurant contract](agents/contracts/restaurant.md)
- [Healthcare contract](agents/contracts/healthcare.md)
- [Telecom contract](agents/contracts/telecom.md)

1. Implement the vertical tools against `POST $WORLD_URL/v1/session/tools/{name}` as listed in the contract.
2. Answer `--number` (PSTN) or join the Stream call (WebRTC).
3. `voicebench run --system your-name --pack restaurant --number +1...`
   or `go run -tags webrtc ./cmd/voicebench run --system your-name-webrtc --transport webrtc --pack restaurant --call-id ...`

`out/<run_id>/summary.json` is schema version 1 (`system`, `k`, per-pack `pass_at_k` / `pass_hat_k`, V2V P50, false-cutoff rate) and is the ingest format for a later leaderboard.

Healthcare, restaurant, and telecom are three columns, not one score.

## Prior art

Voicebench composes ideas from existing voice and agent evals. It is not a drop-in clone of any of them.

- [ServiceNow EVA](https://github.com/ServiceNow/eva) — pass@k / pass^k; a trial passes only when every hard gate holds.
- [tau2-bench](https://github.com/sierra-research/tau2-bench) — seeded stateful world plus a scripted user; success is end-state, not transcript similarity.
- [VAmoS / riley-agent](https://github.com/veris-ai/riley-agent) — latency is reported, not gated; live-call protocol; the judge sees transcript and tools. Same job, tools, and data across implementations.
- [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) — barge-in stop time and selectivity (cough / side conversation).
- [eot-bench](https://github.com/livekit/eot-bench) — reply-gap versus the 300–700 ms human band.
- [aiewf-eval](https://github.com/kwindla/aiewf-eval) — recording-based voice-to-voice latency.
- [r/voiceagents benchmark thread](https://www.reddit.com/r/voiceagents/comments/1vjdwxj/independent_open_source_benchmark_of_voice_agent/) — independent open-benchmark motivation.
