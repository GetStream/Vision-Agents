---
name: voicebench
description: Run the Voicebench voice-agent benchmark locally and read its results. Use when asked to run, smoke-test, load-test or debug the benchmark, to benchmark the accelerated stack, STT or TTS, or to explain why a Voicebench run failed.
---

# Voicebench, locally

Voicebench (`benchmark/`) places scripted calls against our agent over Stream WebRTC and grades
them: world state, tool calls, entities, a judge, turn-taking and reply time. Read
`benchmark/README.md` for the method; this skill is how to run it on a laptop without the
failures every first run hits.

Work from `benchmark/` on a checkout of `accelerate`. Runs cost money (ElevenLabs, Deepgram,
OpenAI, Gemma): start small and say how much you are about to run.

## 1. Check the setup first

Every one of these has failed a run. Check them before the first call, and never print a key.

**Which .env.** voicebench reads `benchmark/.env` and, only when that file does not exist, the
repository's `.env`. A key added to the root `.env` is ignored while `benchmark/.env` exists.

**Keys the default pipeline needs** (Flux STT, Gemma 4 on Baseten, ElevenLabs v4 Turbo, GPT-6.1
Sol): `STREAM_API_KEY`, `STREAM_API_SECRET`, `ELEVENLABS_API_KEY` (caller lines and the agent's
voice), `DEEPGRAM_API_KEY` (Flux, and the transcript the judge reads), `OPENAI_API_KEY` (judge
and subagent), `GEMMA_BASE_URL` and `BASETEN_API_KEY`. `INWORLD_API_KEY` for the TTS bench.

Probe the ones that run out or go cold, from the folder whose `.env` voicebench will read:

```bash
EL=$(grep -E '^ELEVENLABS_API_KEY=' .env | cut -d= -f2- | tr -d "\"'\r\n "); curl -sS -o /dev/null -w "ElevenLabs HTTP %{http_code}\n" -H "xi-api-key: $EL" -H 'Content-Type: application/json' -d '{"text":"Hi.","model_id":"eleven_flash_v2_5"}' "https://api.elevenlabs.io/v1/text-to-speech/VR6AewLTigWG4xSOukaG?output_format=pcm_16000"
```

`401 quota_exceeded` is the key's character limit, whatever the model: someone with workspace
access has to raise it. Already-cached caller lines (`benchmark/cache/tts/`) need no quota, the
agent's own voice always does.

```bash
GU=$(grep -E '^GEMMA_BASE_URL=' .env | cut -d= -f2- | tr -d "\"'\r\n "); BK=$(grep -E '^BASETEN_API_KEY=' .env | cut -d= -f2- | tr -d "\"'\r\n "); curl -sS -m 90 -w "\nGemma HTTP %{http_code} %{time_total}s\n" "$GU/chat/completions" -H "Authorization: Bearer $BK" -H 'Content-Type: application/json' -d '{"model":"google/gemma-4-26B-A4B-it","max_tokens":1,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Hi"}]}' | tail -c 200
```

`deactivated` or `not ready` means the Baseten deployment is down or waking; a first answer can
take minutes. Wait until it answers in about a second before placing calls: calls made while it
wakes go unanswered and score as failures with no tools.

**Services.** The router needs Postgres and Redis: `docker compose up -d --wait postgres redis`
from the repo root (Postgres on 55432, Redis on 56379, which the router's local profile expects).
If the router stops at startup with a migration error ("column … already exists"), the local
database has migrations from another branch: point the run at a fresh database rather than
touching that one, for example `ROUTER_POSTGRES_DSN=postgres://postgres:postgres@localhost:55432/voicebench_packs?sslmode=disable`
after `createdb` on the same server.

**Ports.** The router listens on 8080 and world servers on 8090 and up. `scripts/packs.sh` puts
agents on 8000 and up and `scripts/load.sh` on 8001 and up (`VOICEBENCH_AGENT_PORT_BASE`); a
local stream-api often holds 8000, so stop it or use `load.sh`'s base. A run killed half-way can
leave a Python agent on its port: free it before the next run.

## 2. Run the smallest thing that answers the question

Native audio libraries (`brew install pkg-config opus opusfile libsoxr`) and CGO are needed for
anything that places calls.

| Question | Command | Time |
|---|---|---|
| Does one scenario work? | `CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run --pack restaurant --scenario restaurant.golden --k 1 --target accelerated --spawn --bin <router> --target-url http://127.0.0.1:8001` | ~2 min |
| Quick check of every pack | `scripts/packs.sh` (short set, packs side by side, one router built from the checkout) | ~8 min |
| The trend-line set | `VOICEBENCH_SET=frozen scripts/packs.sh` | ~35 min |
| Harder callers and rooms | `VOICEBENCH_SET=extended scripts/packs.sh` | ~15 min |
| Under load | `VOICEBENCH_CONCURRENCY="1 3" VOICEBENCH_PACK=restaurant scripts/load.sh` | per level, one set |
| STT and TTS alone | `scripts/components.sh` (Flux on the scenarios' caller lines; ElevenLabs and Inworld on their agent lines) | ~15 min |

`--k 3` repeats each scenario; one call is too few to call a change. Override the pipeline with
`VOICEBENCH_STT`, `VOICEBENCH_MODEL`, `VOICEBENCH_TTS` (the subagent is `thinking_llm` in
`agents/accelerated/<pack>/agent.yaml`). Build the router fresh for every run so you never test
an old one; the scripts do.

On CI instead, with the repo's keys: `gh workflow run voicebench.yml --ref accelerate -f set=short -f packs=restaurant`
(`-f bench=components` for STT/TTS). The run's progress lines show each call as it finishes.

## 3. Read the results

Each run writes `out/<run>/`: `summary.json`, `report.md`, and a folder per call with the audio,
transcripts, `heard.json` (what the agent's speech-to-text acted on), `tools.json`,
`timeline.json` (router stages per turn), `judge.json` and `metrics.json`.

Render the report people read, the same one the nightly posts:

```bash
go run ./cmd/voicebench digest --title "Voicebench" --out out/digest out/<run-dir> [more run dirs]
```

`out/digest/voicebench.html` gives passed out of total and a 0-100 score per pack and scenario
type, every failure with its cause, the router's median per stage, and under each call a "What
happened" with the caller's script beside what the agent heard, the agent's turns, the tool
calls and the judge's notes. Read the causes before blaming the model:

- **infra**: no verdict. A key, quota, a cold Gemma, a service: fix the setup, never the agent.
- **heard wrong**: the caller said a value the agent's speech-to-text never heard.
- **did wrong**: heard it, acted wrongly or not at all (a booking claimed but never made is this).
- **said wrong**: a policy or say-do break.
- **turn-taking**: talked over the caller, did not stop, no filler while a tool ran.

`go run ./cmd/voicebench compare --baseline <old> <new>` sets runs side by side with intervals;
a gap inside them is noise. `voicebench noise` over five repeat runs measures how big a change
has to be before it is real.

## 4. Report back

Say what ran (pack, set, k, pipeline, commit), pass out of total, the main causes, and reply
time P50 with its sample count. Separate setup failures from agent behaviour, and quote the
call's own evidence (a tool call, a heard line, a judge note) for each claim.
