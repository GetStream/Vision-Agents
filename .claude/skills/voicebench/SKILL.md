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
Run the checks below from the repository root.

**Every variable is there.** These are what the default pipeline (Flux STT, Gemma 4 on
Baseten, ElevenLabs v4 Turbo, GPT-6.1 Sol) and the benches use. The check prints names only:

```bash
f=.env; [ -f benchmark/.env ] && f=benchmark/.env; echo "voicebench reads $f"; for k in STREAM_API_KEY STREAM_API_SECRET ELEVENLABS_API_KEY DEEPGRAM_API_KEY OPENAI_API_KEY GEMMA_BASE_URL BASETEN_API_KEY INWORLD_API_KEY GOOGLE_API_KEY; do v=$(grep -E "^$k=" "$f" | tail -1 | cut -d= -f2- | tr -d "\"' \r"); [ -n "$v" ] && echo "ok       $k" || echo "MISSING  $k"; done; grep -qE '^GEMMA_BASE_URL=https://' "$f" || echo "BAD      GEMMA_BASE_URL is not an https URL"
```

**If any is missing, fetch the environment** with rocky, which writes every key the team keeps
in Secret Manager to the file it is given:

```bash
cp "$f" "$f.bak" 2>/dev/null; rocky agents local secrets create_env -f "$f"
```

It **overwrites** the file and keeps no comments, so restore any local-only lines from the
backup afterwards (a `ROUTER_POSTGRES_DSN` pointing at your own database, the Slack bot
variables): `diff "$f.bak" "$f"` shows what was lost. Write to `$f`, the file voicebench reads: a
fresh root `.env` does nothing while `benchmark/.env` exists. Then run the check again. If rocky
fails, it is the user's gcloud login or access, not something to work around: tell them.

**Every key works.** A key can be present and dead, and fetching again does not fix that: the
team's file can carry the same exhausted key. Probe each one; none of these prints a key, and
only the ElevenLabs probe spends anything (three characters):

```bash
key() { grep -E "^$1=" "$f" | tail -1 | cut -d= -f2- | tr -d "\"' \r"; }
curl -sS -o /dev/null -w "Deepgram   HTTP %{http_code}\n" -H "Authorization: Token $(key DEEPGRAM_API_KEY)" https://api.deepgram.com/v1/projects
curl -sS -o /dev/null -w "OpenAI     HTTP %{http_code}\n" -H "Authorization: Bearer $(key OPENAI_API_KEY)" https://api.openai.com/v1/models
curl -sS -o /dev/null -w "ElevenLabs HTTP %{http_code}\n" -H "xi-api-key: $(key ELEVENLABS_API_KEY)" -H 'Content-Type: application/json' -d '{"text":"Hi.","model_id":"eleven_flash_v2_5"}' "https://api.elevenlabs.io/v1/text-to-speech/VR6AewLTigWG4xSOukaG?output_format=pcm_16000"
curl -sS -m 90 -o /dev/null -w "Gemma      HTTP %{http_code} in %{time_total}s\n" "$(key GEMMA_BASE_URL)/chat/completions" -H "Authorization: Bearer $(key BASETEN_API_KEY)" -H 'Content-Type: application/json' -d '{"model":"google/gemma-4-26B-A4B-it","max_tokens":1,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Hi"}]}'
```

All four should be `200`. What the others mean:

- **401 from Deepgram or OpenAI**: the key is wrong or revoked. Fetch again with rocky; if the
  fetched key fails too, the team's secret is stale and someone with access has to replace it.
- **401 from ElevenLabs**: rerun the probe without `-o /dev/null`. `quota_exceeded` is the
  key's character limit, whatever the model: someone with workspace access has to raise it.
  Cached caller lines (`benchmark/cache/tts/`) need no quota, the agent's own voice always does.
- **400 from Gemma**, or a slow first answer: the Baseten deployment is deactivated or waking.
  Wait until it answers in about a second before placing calls: calls made while it wakes go
  unanswered and score as failures with no tools.

Stop and tell the user about any key that stays broken; do not start calls with one.

**Services.** The router needs Postgres and Redis: `docker compose up -d --wait postgres redis`
from the repo root (Postgres on 55432, Redis on 56379, which the router's local profile expects).
If the router stops at startup with a migration error ("column … already exists"), the local
database has migrations from another branch: point the run at a fresh database rather than
touching that one, for example `ROUTER_POSTGRES_DSN=postgres://postgres:postgres@localhost:55432/voicebench_packs?sslmode=disable`
after `createdb` on the same server.

**Leftovers.** A run killed half-way leaves its Python agents, router and world servers behind,
holding their ports and sometimes still in a call. Look before every run and stop what you find
(only processes a bench started: a local stream-api on 8000 is the user's):

```bash
lsof -nP -iTCP -sTCP:LISTEN | grep -E ':(800[0-9]|8080|809[0-9]) '; pgrep -fl 'voicebench|simple_voice_ai|agents/accelerated' || true
```

The router listens on 8080 and world servers on 8090 and up. `scripts/packs.sh` puts agents on
8000 and up, `scripts/digest.sh` and `scripts/load.sh` on 8001 and up (`VOICEBENCH_AGENT_PORT_BASE`).

**Gemma is awake.** The probe above tells you whether Gemma answers; before placing calls, wait
until it answers fast, the way the CI job does. A deployment scaled to zero takes minutes (`key`
and `f` come from the checks above):

```bash
for i in $(seq 1 40); do t=$(curl -sS -m 60 -o /dev/null -w '%{http_code} %{time_total}' "$(key GEMMA_BASE_URL)/chat/completions" -H "Authorization: Bearer $(key BASETEN_API_KEY)" -H 'Content-Type: application/json' -d '{"model":"google/gemma-4-26B-A4B-it","max_tokens":1,"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"Hi"}]}'); echo "Gemma $t"; case "$t" in 200\ 0.*|200\ 1.*) break;; esac; sleep 15; done
```

**Caller lines are cached.** Caller audio is ElevenLabs speech cached in `benchmark/cache/tts/`
by voice and text; a line missing from the cache is synthesized during the call and spends
quota then. Fill it up front, so a quota failure stops you here instead of half-way through:
`go run ./cmd/voicebench synth --pack <pack>` (no `--pack` for all). The agent's own voice is
never cached: every call spends ElevenLabs characters on it.

## 2. Say what it will cost, then run the smallest thing that answers the question

**Estimate first.** Count the calls: scenarios in the set for the chosen packs, times `k`, times
the number of arms. A call takes about four minutes for coherence, a minute and a half for a
monologue and under a minute for the rest, and is cut at its `max_duration_s` (180 or 240). The
agent's voice spends a few hundred ElevenLabs characters a call; the judge, Deepgram and Gemma
spend per call too.

```bash
set=short; packs="restaurant healthcare telecom"; k=1; for p in $packs; do echo "$p $(grep -c "^$p\." scenarios/$set.txt)"; done; echo "x k=$k"
```

Tell the user the call count and rough wall time before running, and ask before anything over
about fifteen minutes or a frozen set at `k` above 1. A single scenario needs no asking.

Native audio libraries (`brew install pkg-config opus opusfile libsoxr`) and CGO are needed for
anything that places calls.

| Question | Command | Time |
|---|---|---|
| Does one scenario work? | `CGO_ENABLED=1 go run -tags webrtc ./cmd/voicebench run --pack restaurant --scenario restaurant.golden --k 1 --target accelerated --spawn --bin <router> --target-url http://127.0.0.1:8001` | ~2 min |
| Quick check of every pack | `VOICEBENCH_SET=short VOICEBENCH_LIVEKIT_ARMS= VOICEBENCH_DIGEST_POST=0 scripts/digest.sh` | ~10 min |
| The trend-line set | `VOICEBENCH_LIVEKIT_ARMS= VOICEBENCH_DIGEST_POST=0 scripts/digest.sh` | ~15 min |
| Harder callers and rooms | `VOICEBENCH_SET=extended VOICEBENCH_LIVEKIT_ARMS= VOICEBENCH_DIGEST_POST=0 scripts/digest.sh` | ~10 min |
| Under load | `VOICEBENCH_CONCURRENCY="1 3" VOICEBENCH_PACK=restaurant scripts/load.sh` | per level, one set |
| STT and TTS alone | `scripts/components.sh` (Flux on the scenarios' caller lines; ElevenLabs and Inworld on their agent lines) | ~15 min |

Prefer `scripts/digest.sh` over `scripts/packs.sh` for anything bigger than one scenario: both
run the packs side by side against one router built from the checkout, but `digest.sh` stops a
pack after `VOICEBENCH_PACK_TIMEOUT` seconds (25 minutes a trial) and kills everything it
started, and renders the digest into the run directory. An empty `VOICEBENCH_LIVEKIT_ARMS` runs
our stack alone; `VOICEBENCH_DIGEST_POST=0` keeps it off Slack.

`VOICEBENCH_K=3` repeats each scenario; one call is too few to call a change. Override the
pipeline with `VOICEBENCH_STT`, `VOICEBENCH_MODEL`, `VOICEBENCH_TTS` (the subagent is
`thinking_llm` in `agents/accelerated/<pack>/agent.yaml`). Build the router fresh for every run
so you never test an old one; the scripts do.

**Watch it, and stop a stuck call.** A run prints one progress line per call event,
`voicebench: [3/8] restaurant.selectivity: started` then its verdict. Run it in the background
and follow those lines. A call takes at most four minutes, so no new line for six means it is
stuck: there is a known deadlock in the WebRTC receive path when the agent's track is
resubscribed, and a stuck call ignores its deadline and Ctrl-C. Kill the whole tree, not just
the parent: the agents run under `uv`, two levels down.

```bash
kill_tree() { for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill -TERM "$1" 2>/dev/null; }
kill_tree <pid>
```

Then check Leftovers again (`kill -KILL` what survived) and count that call as **infra**, not as
an agent failure. `digest.sh` does this itself at its timeout.

## 3. Read the results

Each run writes `out/<run>/`: `summary.json`, `report.md`, and a folder per call with the audio,
`transcript.json`, `heard.json` (what the agent's speech-to-text acted on), `tools.json`,
`timeline.json` (router stages per turn), `judge.json` and `metrics.json`.

Render the report people read, the same one the nightly posts (`digest.sh` already did):

```bash
go run ./cmd/voicebench digest --title "Voicebench" --out out/digest out/<run-dir> [more run dirs]
```

`out/digest/voicebench.html` gives passed out of total and a 0-100 score per pack and scenario
type, every failure with its cause, the router's median per stage, and under each call a "What
happened" with the caller's script beside what the agent heard, the agent's turns, the tool
calls and the judge's notes. `digest.sh` also streams every caller line of the frozen or short set alone through the
agent's speech-to-text (`out/<run>/stt`) and the report shows its WER in a Speech-to-text
section: what the model hears, apart from turn-taking. `VOICEBENCH_DIGEST_STT=0` skips it;
pass an `stt` run directory to `digest` to add one by hand. Read the causes before blaming the model:

- **infra**: no verdict. A key, quota, a cold Gemma, a service: fix the setup, never the agent.
- **heard wrong**: the caller said a value the agent's speech-to-text never heard.
- **did wrong**: heard it, acted wrongly or not at all (a booking claimed but never made is this).
- **said wrong**: a policy or say-do break.
- **turn-taking**: talked over the caller, did not stop, no filler while a tool ran.

**Known signatures.** These have each been seen; match a failure against them before
investigating from scratch:

| What the report shows | What it usually is | Where to look |
|---|---|---|
| Every call invalid, `quota_exceeded` or `INVALID_AUTH` in the error | A key (ElevenLabs quota, a stale Deepgram or Stream key) | Setup, section 1. Not the agent |
| No reply, `0 tools`, on the first calls of a run | Gemma was waking | Wait for Gemma, rerun those calls |
| Router exits at start, "column … already exists" | Local database migrated by another branch | A fresh `ROUTER_POSTGRES_DSN` |
| Router cannot reach Redis | Redis not up on 56379 | `docker compose up -d --wait redis` |
| `create_reservation not called`, the agent's last turn is a question | The agent asked to confirm and the caller hung up | `transcript.json` and `tools.json`; is the scenario's go-ahead line there? |
| The agent says "all set", no write tool in `tools.json` | A booking claimed but never made | Model or prompt: a say-do break, even if the judge passed it |
| `false_cutoff` on most calls, a large `decision` stage in `timeline.json` | End-of-turn detection cut into the caller's pauses | The router's turn-taking, not the prompt |
| Filler gate fails, yet the agent spoke while the tool ran | The harness did not recognise the filler phrase (`fillerPhrases` in `internal/score/timing.go`) | Harness: report it, do not change the agent |
| `missed <value>` under "caller said vs heard" | Speech-to-text | `heard.json`; keyterms or the STT choice |
| `search … API_KEY` warnings in the agent log | No web-search provider configured | Harmless for these packs |

## 4. Look at one call

When a cause is not obvious, open the call's folder and rerun only that scenario
(`--scenario <id> --k 3`, it costs a few minutes):

1. `result.json` and `metrics.json`: which gates failed and by how much.
2. The caller's script (`scenarios/<pack>/<name>.yaml`) beside `heard.json`: did the agent hear
   what was said?
3. `transcript.json` and `tools.json`: what the agent said, and whether its words match its tool
   calls (name, arguments, order).
4. `timeline.json`: the router's stages per turn (`decision`, model to first text, TTS), to tell
   a slow model from a slow turn decision.
5. `judge.json`: the judge's notes; it is gpt-4.1-mini and can miss a say-do break.
6. The audio (`mixed.wav` holds both sides): listen when a turn-taking gate fails, to hear
   whether the agent really talked over the caller.

## 5. Did a change help?

One run says little. To answer "is the new version better":

1. Fix everything but the change: same set, packs, `k` (3 at least), pipeline and network
   profile, back to back on the same machine.
2. Run the base commit, then the changed one, each with a fresh router build.
3. Compare: `go run ./cmd/voicebench compare --baseline out/<base> out/<new>`, with
   `--mde baselines/<target>/noise-<packs>.json` when a noise floor exists.
4. A difference inside the intervals, or below the noise floor, is no difference: say so, do not
   round it up. Measure the floor with five repeat runs and `voicebench noise <dirs>`.

A different STT, model or TTS is a different series, not a regression or a win against the
old one: label the runs (`--system`) and compare them as alternatives.

## 6. Add or change a scenario

A scenario is a YAML file in `scenarios/<pack>/`: the persona, its `turns` (a turn's `segments`
make a long turn with pauses, `voice` changes the speaker, `aside` is someone else in the room,
`check_in` holds the caller silent so the agent should check in), `hold_floor` for a monologue,
the seeded world, `end_state`, `expected_tools`, `entities`, `policy`, and `agent_replies`, a
reference reply that must pass the scenario's own gates. If the agent is expected to act without
asking, give the caller a go-ahead line ("yes, go ahead and book it"), or the agent's
confirmation question ends the call with nothing booked.

New scenarios go into `scenarios/extended.txt`. `frozen.txt` is the trend line: changing it or
a scenario in it changes the scenario hash and needs a methodology bump in `README.md`. After
adding one, run `voicebench synth --pack <pack>` and then the scenario alone at `--k 3`.

## 7. CI and Slack

The workflow `.github/workflows/voicebench.yml` runs the frozen set nightly. By hand, with the
repo's keys (the file is not on `main` yet, so name the branch):

```bash
gh workflow run voicebench.yml --ref accelerate -f set=short -f packs=restaurant
```

Inputs: `bench` (`agents` or `components`), `set` (`frozen`, `short`, `extended`), `packs`
(`all` or one). Follow it with `gh run watch <id>` and the progress lines in the log
(`gh run view <id> --log | grep 'voicebench: \['`); fetch the results with
`gh run download <id> -n voicebench-nightly` (or `voicebench-components`) and render the digest
locally as in section 3. A failed step named "Wake the Gemma deployment" is Gemma, not the
agent.

In Slack, `/agents-bench run [branch] set=frozen|short packs=<pack>` dispatches the same workflow, and
`/agents-bench help` lists the options. Ask the user before posting to Slack (`--slack`, or
`digest.sh` without `VOICEBENCH_DIGEST_POST=0`) and never set or change repository secrets:
those are theirs to do.

## 8. Report back

Always in this shape, so reports compare:

```
Ran: <set> set, <packs>, k=<k>, <stt> / <model> / <tts>, commit <sha>, <local|CI run url>
Result: <passed>/<valid> passed, score <0-100>; <n> invalid (<why>)
Per pack: restaurant <p>/<v>, healthcare <p>/<v>, telecom <p>/<v>
Causes: <n> heard wrong, <n> did wrong, <n> said wrong, <n> turn-taking
Reply time: P50 <ms> over <n> turns (tool turns P50 <ms>)
Failures:
- <scenario>: <cause>. Evidence: "<a tool call, a heard line or a judge note, quoted>"
Next: <the one thing to fix or rerun>
```

Keep setup failures out of the score and say what they were. Never claim a change helped
without the comparison in section 5.
