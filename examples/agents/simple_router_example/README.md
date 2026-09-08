# Simple router example

Every other example points a `Router` at a config somebody else wrote. This one writes
them. `routers/` holds one YAML file per config, `sync_routers` stores the directory, and
what is inside is the interesting half: who to try and in what order, what may happen to
the audio afterwards, and the per-provider settings the shared vocabulary has no word for.

Nothing here names a model to transcribe with. The router picks, and the config is what it
picks by.

## What the two configs ask for

`clinic.yaml` cares where the audio goes more than which model is best at it:

```yaml
stt:
  providers: [deepgram, parakeet]
  data_policy:
    allow_training: false
    retention: none
  overwrites:
    deepgram:
      eot_threshold: 0.6
```

`providers` is an ordered list, and list order is try order — Deepgram first because Flux
is the better turn detector, then our own Parakeet. `data_policy` is a requirement rather
than a description: a provider who has not said they train on nothing and keep nothing is
not asked, which rules out five of the seven vendors the router knows. `overwrites` is for
settings only one vendor has, so there is nothing to standardise them against.

`podcast.yaml` is a recording config: nobody is waiting for it, two people are talking,
and it is going in front of an audience.

```yaml
stt:
  providers: [deepgram]
  diarize: true
  words: true
  format: true
  profanity_filter: true
```

Asking a live session for diarization and word timings would be refused, because Flux
cannot express them. Asking for a recording routes to Nova-3, which can. That is the
difference between naming a capability and naming a model.

## What it prints

```
stored from routers/:
  clinic: {'providers': ['deepgram', 'parakeet'], 'data_policy': {...}, 'overwrites': {...}}
  podcast: {'providers': ['deepgram'], 'diarize': True, 'format': True, ...}

transcribed:
  served by deepgram/nova-3
  Hi. I'd like to book a table for four this Saturday at 07:30 patio if you have it. ...
  29 words timed, speakers: ['speaker_0']

reconfigured:
  clinic now tries ['parakeet', 'deepgram/flux-general-en'] in that order

refused:
  smart + diarize: options: smart mode cannot diarize, since it rewrites what was said
  a misspelt vendor: "deepgramm" is not a provider, a provider/model or a capability
  shortcut this deployment offers
```

The last two are the point of storing a config rather than passing keywords at a call
site. `smart` rewrites what was said, so a word cannot also be timed or attributed to a
speaker, and a vendor nothing routes to would have failed every request made under the
config. Both come back while the config is being written, from an exception with a
message, rather than at three in the morning from a call that would not start.

`configure_stt` writes the same block from code, and leaves the other three modalities as
they were stored — saying how something is heard is not a statement about how it speaks.

## Prerequisites

A running acceleration router: see [acceleration/README.md](../../../acceleration/README.md).

- `ROUTER_POSTGRES_DSN`, since a stored config is a row.
- `DEEPGRAM_API_KEY`, which is who the one transcription here routes to. Provider
  credentials live with the router, not with this example.

## Run

```bash
cd examples/agents/simple_router_example
uv sync
uv run simple_router_example.py
```

Running it twice edits the two configs rather than storing another copy of each, so it is
safe to keep running while editing the YAML.

Needs a `.env` with:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```

## Not shown here

`mode: smart` is Gemini's disfluency removal and grammar cleanup, and `mode: verbatim` is
every um and false start kept. Both are options on a live session rather than a recording,
so neither is in these two configs; `verbatim` is declared by Gemini, xAI and Deepgram
Flux, and `smart` by Gemini alone.
