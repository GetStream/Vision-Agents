# Realtime STT example

The same clip through four configs, in Go.

`routers/` holds them all. None names a model at the call site — `Realtime` is handed nil
options, so what answers a session is whatever the named config resolves to, and the example
prints which model that turned out to be.

| Config                         | Asks for                                            |
| ------------------------------ | --------------------------------------------------- |
| `stt-realtime-fast`            | A fast answer, four models deep                     |
| `stt-realtime-accurate`        | To be told who spoke, and a knob per vendor         |
| `stt-realtime-fast-private`    | The same speed, with nothing kept or trained on     |
| `stt-realtime-accurate-private`| The same policy, ordered for accuracy               |

## The two configs

`stt-realtime-fast` is transcription for a live call where answering quickly is the point:

```yaml
name: stt-realtime-fast
stt:
  providers:
    - deepgram/flux-general-en
    - cartesia/ink-2
    - inworld/inworld-stt-1
    - together-nemotron/nvidia/nemotron-3-asr-streaming-0.6b
  diarize: false
```

`providers` is ordered and list order is try order, each entry expanded where it stands, so
that is a four-deep fallback chain rather than a set. Flux first because Deepgram's turn
detector is the one the other three are measured against; Ink 2 next; then Inworld; then
Nemotron on Together, the cheapest of the four by a factor of six.

`stt-realtime-accurate` is the same shape, ordered by accuracy instead — Muse Voice,
Scribe v2 Realtime, Ink 2, Nemotron — with `diarize: true`.

That one line is the whole difference, and it does more than turn a feature on.

## What `diarize` does to the chain

`diarize: false` asks not to be told who spoke, which rules nothing out. Only a term that
is *on* narrows the candidates, so the fast config keeps all four and fails over down the
list when one is unhealthy or refuses.

`diarize: true` is a term, and `serving()` keeps only the models whose `supports:` includes
it. Of the accurate config's four, only Muse declares `diarize` — ElevenLabs say plainly
that Scribe v2 Realtime does not diarize and that batch Scribe is where that lives, and
Cartesia and NVIDIA make no claim to it either. So that config resolves to Muse with
nothing behind it.

That is the design rather than a mistake in the config. A request that asked to be told who
spoke is answered by something that can, or refused outright, never served quietly by
something that cannot — which is what an optional label would mean in practice. Set the
line to `false` and the same four models are a four-deep chain again.

## One knob per vendor, in `overwrites`

Where a turn ended is the question all four of the accurate config's models answer
differently, and none of them answers it in words the others would understand. So it is not
a shared term. It is a block per vendor, keyed by provider name:

```yaml
  overwrites:
    muse:
      mode: DIARIZATION              # a mode, not a number
    elevenlabs:
      vad_silence_threshold_secs: 0.4 # how long a silence counts as a stop
      min_silence_duration_ms: 200
    cartesia:
      turn_end_threshold: 0.7        # how sure the model has to be the turn ended
      turn_end_timeout_ms: 600
    together-nemotron:
      turn_grace_ms: 400             # the router's own wait, not the vendor's
```

Four spellings of one question. Muse takes it as a mode, because `DIARIZATION` and
`PUSH_TO_TALK` are the only vocabulary it has for a boundary — and `PUSH_TO_TALK` is the one
nothing but this block can ask for, since it hands the boundary back to the caller's client.
Scribe takes it as what counts as a stop. Ink 2 judges it with a model, so its numbers are
confidences rather than durations. Nemotron says nothing about turns at all — its protocol
has no boundary in it — so that last one is the router's own wait for the transcript to stop
changing, and it is why `silence_ms` would be a lie there.

Only the block belonging to whichever model serves the session is read, and the vendor named
parses its own. A field it does not have is an error rather than a setting that was accepted
and never sent, so a misspelling refuses the session instead of quietly changing nothing:

```
routing: overwrites for muse-voice-transcribe-1.0: json: unknown field "moed"
```

## Nothing kept, nothing trained on

The two `-private` configs add a `data_policy`, which is a requirement rather than a
description — a model that has not said no is not asked:

```yaml
  data_policy:
    allow_training: false
    retention: none
```

That is a much shorter list of models than it sounds like. `allow_training: false` needs a
model that has said outright it does not train on what it is sent, and publishing nothing
counts as not having said no. `retention: none` needs one that keeps nothing, not one that
keeps it briefly. Of every live transcriber the router knows, three answer both: the two
Deepgram Flux models and our own Parakeet.

So none of the six models the first two configs name survives. Cartesia and ElevenLabs train
unless the account has opted out, Muse and Inworld declare unknown, and Nemotron on Together
says it does not train but states no retention window — and an unstated window is not none.

Parakeet is the strongest answer to both questions, since the audio never leaves our own
deployment at all, and on this clip it is also the most accurate of the three — the only one
that both hears every word and writes the time down as *7:30* rather than as *seven thirty*.
So the accurate config leads with it and the fast one does not, because what it costs is the
first call: the
deployment scales to zero and a cold one takes minutes to answer, during which the session
hears silence rather than an error, so the fallback behind it never gets its turn. **If
`stt-realtime-accurate-private` prints nothing, run it again** — the second run finds a warm
GPU.

`stt-realtime-accurate-private` is the interesting one, because the two requirements collide:
nothing that will promise to keep nothing can also name the voice on a live call. Of the
models that declare `diarize`, Muse answers unknown on both questions and Grok keeps audio
for 30 days. Asking for both would resolve to nothing and the request would be refused —
correctly, but a config that can never serve is not worth shipping. So that config gives up
the label rather than the policy, and says so in a comment. Both together can be had on the
recording path, where `deepgram/nova-3` diarizes and promises both halves; a live call is
where the choice has to be made.

## The audio

`saturday_seven_thirty.wav`, eight seconds of somebody booking a table, delivered by
`stream.RecordedCall` — 100 ms at a time paced to real time the way a call arrives, then two
seconds of silence. The silence is not padding: a streaming model decides a turn is over by
hearing the caller stop, and a recording that ends the instant the speech does never gives
it that, so without it the last thing said is settled by the hangup rather than by the
speaker finishing.

```
$ go run .
deepgram/flux-general-en: Hi. I'd like to book a table for four this Saturday at seven thirty patio if you have it, a high chair, and one of us has a peanut allergy.

$ go run . -use-case stt-realtime-accurate
muse/muse-voice-transcribe-1.0: speaker A: Hi. I'd like to book a table for four this Saturday at 7.30 patio if you have it. A high chair and one of us has a peanut allergy.
```

## Run it against a local router

Everything is read from the repository-root `.env`, so both of the steps below are run from
a checkout with one.

**1. Postgres and Redis.** A stored router config is a row, so the router needs a database
before it will accept one. From the repository root:

```bash
docker compose up -d postgres redis
```

That is Postgres on `:55432` and Redis on `:56379`. The standalone `va-pg` and `va-redis`
containers use the same ports, so stop those first if they are running.

**2. The router.** It does not read `.env` itself, which is the step that is easy to miss:

```bash
cd acceleration
set -a && . ../.env && set +a
go run ./cmd/router
```

It is up at `msg=listening address=:8080`. `running without authentication` in that output is
expected locally: the router trusts the customer header rather than a credential.

The `.env` needs `ROUTER_POSTGRES_DSN`, and the key for whichever model answers: that is
`DEEPGRAM_API_KEY` for both fast configs as written, `META_API_KEY` for the accurate one,
and `PARAKEET_WS_URL` with `BASETEN_API_KEY` for `stt-realtime-accurate-private`, which
tries our own deployment first. Provider credentials live with the router, not with this
example. See [acceleration/README.md](../../../acceleration/README.md) for the rest of them.

**3. This example**, in another terminal:

```bash
cd examples/routers/stt_realtime_example
go run .                                          # stt-realtime-fast
go run . -use-case stt-realtime-accurate
go run . -use-case stt-realtime-fast-private
go run . -use-case stt-realtime-accurate-private
```

Each run takes about 13 seconds: eight of clip paced to real time, two of trailing silence,
three of settling. It needs two more variables, which say where the router is and who the
cost rows are billed to:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

Both configs in `routers/` are stored on every run, by name, so editing a file there edits
the config rather than storing another copy.

## Run it against staging

`-staging` transcribes on the hosted router at `accelerate.gcp.stream-io-api.com`, which
needs neither of the two steps above — no database and no local process, since the configs
are stored there.

It is reached through a proxy that authenticates a Stream app rather than trusting a
customer header, so it wants a key and secret instead of a customer id:

```
STREAM_API_KEY=...
STREAM_API_SECRET=...
```

Then, from this directory:

```bash
go run . -staging
go run . -staging -use-case stt-realtime-accurate
```

Half a credential is refused rather than quietly downgraded to an unauthenticated request,
so a missing secret reports `STREAM_ACCELERATION_AUTHENTICATE needs STREAM_API_KEY and
STREAM_API_SECRET` before anything is sent.

One caveat while this change is new: a router refuses a config naming a model it does not
have, and both configs here name Ink 2, Inworld and Scribe. Until staging runs a build with
those three, `-staging` reports which name it did not recognise rather than transcribing:

```
stream: "elevenlabs/scribe_v2_realtime" is not a provider, a provider/model or a
capability shortcut this deployment offers
```
