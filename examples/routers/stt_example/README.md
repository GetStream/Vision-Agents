# STT example

Transcribing through a config this example writes itself, rather than one somebody else
wrote. `routers/` holds a folder per config, each with a `router.yaml`, and naming one is
what stores it. What is inside is the interesting half: who to try and in what order, and
what may happen to the audio afterwards.

Nothing here names a model to transcribe with — the router picks, and the config is what
it picks by.

## The config

`routers/clinic/router.yaml` cares where the audio goes more than which model is best at it:

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

The block says nothing about live or recorded. The same one serves a streaming session and
a whole recording, and the router picks the streaming or the batch half of a vendor from
which one is asking.

## The audio

`saturday_seven_thirty.wav`, eight seconds of somebody booking a table, because an example
cannot talk. It is streamed a chunk at a time the way a call arrives, then two seconds of
silence — not padding: a streaming model decides a turn is over by hearing the caller stop,
and a clip that ends the instant the speech does never gives it that.

```
flux-general-en: Hi. I'd like to book a table for four this Saturday at seven thirty patio if you have it, a high chair, and one of us has a peanut allergy.
```

## Run

A running acceleration router first: see [acceleration/README.md](../../../acceleration/README.md).
It needs `ROUTER_POSTGRES_DSN`, since a stored config is a row, and `DEEPGRAM_API_KEY`.
Provider credentials live with the router, not with this example.

```bash
cd examples/routers/stt_example
uv sync
uv run stt_example.py
```

Nothing here calls `sync_routers`. `Router("clinic")` finds `routers/clinic/router.yaml`
and stores it on the first session; `.router_sync` records the md5, so a run that edits
nothing sends nothing, and one that edits the YAML edits the config rather than storing
another copy.

Needs a `.env` with:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```
