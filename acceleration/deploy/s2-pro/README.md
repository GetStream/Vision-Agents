# Streaming S2 Pro on Baseten

Fish Audio S2 Pro served over a WebSocket, so it can back the self-hosted tier of the TTS
router with audio that starts playing before the sentence is finished.

## Licence

S2 Pro is released under the **Fish Audio Research License**: free for research and
non-commercial use, commercial use needs a separate licence from Fish Audio. Settle that
before this becomes production infrastructure. The hosted Fish API (`internal/tts/fish`)
is the commercially licensed way to reach the same model.

## Deploy

```bash
truss push --promote
```

`truss push` alone creates a published deployment but leaves the `production` environment
pointing at the previous one, so `--promote` matters. To promote a deployment after the
fact:

```bash
curl -X POST -H "Authorization: Api-Key $BASETEN_API_KEY" \
  https://api.baseten.co/v1/models/$MODEL_ID/deployments/$DEPLOYMENT_ID/promote
```

Point `S2PRO_WS_URL` at
`wss://model-$MODEL_ID.api.baseten.co/environments/production/websocket`.

`fishaudio/s2-pro` is public and ungated, so no `HF_TOKEN` is needed to fetch the weights.

The engine and its serving config are pinned to the same release: `requirements` installs
`sglang-omni==0.1.1` and `build_commands` clones tag `0.1.1` for
`examples/configs/s2pro_tts.yaml`. Bump them together. S2 Pro also decodes through the
Descript codec, which the base engine leaves out, so `descript-audiotools` and
`descript-audio-codec` are installed alongside it.

## GPU sizing

Fish report a real-time factor of 0.195 and a time-to-first-audio near 100 ms on an H200.
`config.yaml` asks for an H100, which is the floor rather than an optimisation: the 4B Slow
AR plus 400M Fast AR does not leave useful headroom on an L4, and the inference engine
pins the FlashAttention 3 backend, which needs Hopper. Measure time-to-first-audio with
`cmd/say` after deploying and move to an H200 if it disappoints.

## Wire protocol

Deliberately the mirror image of the Parakeet deployment next door.

1. Client sends a JSON text frame: `{"sample_rate": 44100, "encoding": "linear16"}`.
   The server replies `{"type": "ready", "sample_rate": 44100}`, or
   `{"type": "error", ...}` and closes.
2. Client sends `{"type": "text", "id": "s1", "text": "..."}` one or more times, then
   `{"type": "flush", "id": "s1"}` to say it.
3. Server streams binary frames of little-endian mono PCM16, then
   `{"type": "final", "id": "s1", "audio_duration_ms": ..., "processing_time_ms": ...}`.
4. `{"type": "cancel", "id": "s1"}` stops generation for barge-in and is answered with a
   final carrying `"cancelled": true`. `{"type": "close"}` ends the session.

Several ids may be in flight at once; every frame carries the id it belongs to.

44.1 kHz is not a choice: S2 Pro reconstructs through the Descript codec, which runs at
that rate. The engine reports the rate it used on each stream and the server compares it
against the rate the handshake promised, failing the utterance on a mismatch rather than
letting the client play the audio back at the wrong speed.

The model is served by `sglang-omni`'s OpenAI-compatible endpoint, started once per replica
and reached over localhost. Wrapping it rather than reimplementing it keeps the engine's
continuous batching, paged KV cache and prefix caching.

## Test it

The Go provider's integration test covers this deployment:

```bash
cd ../../ && go test -tags integration ./internal/tts/s2pro/
```
