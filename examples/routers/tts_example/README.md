# TTS example

Speaking through a config this example writes itself, rather than one somebody else wrote.
`routers/switchboard/router.yaml` says what the voice has to be instead of which model it
is, naming the config stores it, and the socket streams PCM back as the provider makes it.

## The config

```yaml
name: switchboard

tags:
  team: support

tts:
  target: en-low-latency
```

`en-low-latency` is a capability, not a model: English, and quick to first audio. Four
voices answer to it today — Cartesia Sonic, Inworld's flash model, ElevenLabs Flash and
the same S2-Pro weights on our own deployment — ranked by which have been up and quick
lately, so the config outlives any one of them being down.

That is the whole block, and what is missing from it is the interesting part:

- **A voice id belongs to one provider**, so naming one in a config that does not name a
  provider is a contradiction. Left out, whichever answers uses its own default.
- **`speed`, `emotion` and `stability` are terms**, and no live voice declares one. A
  provider that cannot express what was asked refuses rather than speaking flatly and
  hoping nobody minds, so a live request naming any of them has no candidates at all.

## The two paths

Those terms are not gone, they are on the other path. Nobody is listening to an audiobook
while it is being made, which is what lets a recording be a file with a codec rather than
PCM on a socket, and what lets it be routed to a slower model that takes direction:

```python
speech = await router.tts.recording(
    "Chapter one, in full.", target="en-recorded", format="mp3_44100_128", speed=1.05
)
```

## Run

A running acceleration router first: see [acceleration/README.md](../../../acceleration/README.md).
It needs `ROUTER_POSTGRES_DSN`, since a stored config is a row, and a key for whichever
voice answers. Provider credentials live with the router, not with this example.

```bash
cd examples/routers/tts_example
uv sync
uv run tts_example.py
```

Both lines are spoken over one socket, and the numbers are the reason it stays open:

```
Your table for four is booked for Saturday at seven thirty.
  first audio after 1627ms

You are on the patio, with a high chair and a note about the peanut allergy.
  first audio after 93ms

spoken.wav: 7.4s at 24000Hz
```

The first line pays for reaching the provider, the second is what has already been paid
for. `spoken.wav` is both of them, written next to this file because an example has no
speaker.

Nothing here calls `sync_routers`. `Router("switchboard")` finds
`routers/switchboard/router.yaml` and stores it on the first session; `.router_sync` records
the md5, so a run that edits nothing sends nothing.

Needs a `.env` with:

```
STREAM_ACCELERATION_URL=http://localhost:8080
STREAM_ACCELERATION_CUSTOMER_ID=examples
```
