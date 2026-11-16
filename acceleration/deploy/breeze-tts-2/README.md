# Streaming Breeze TTS 2 on Baseten

Breeze TTS 2 served over a WebSocket, so it can back the expressive tier of the TTS router
with audio that starts playing before the sentence is finished.

It is here rather than alongside the hosted providers because it is the only voice we can
run that both acts a direction and takes one in plain English. ElevenLabs' v3 socket acts
bracketed tags; this model also takes a sentence describing how to say the line, and can
invent a voice from a description with no reference audio at all.

## Licence

Breeze TTS 2's **source code is Apache 2.0, but the weights are not**: model weights,
derivative models and self-hosted outputs are under the BreezeBlue Research and
Non-Commercial License. Commercial use needs written authorisation from RESONIA, INC.
(contact@breeze.blue).

Settle that before this becomes production infrastructure. The hosted Breeze Blue API at
`api.breeze.blue` is the commercially licensed way to reach the same model, and it exposes
a realtime WebSocket of its own; if the licence cannot be had, that is the port to write
rather than this deployment. This is the same position `deploy/s2-pro` is in.

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

Point `BREEZE_WS_URL` at
`wss://model-$MODEL_ID.api.baseten.co/environments/production/websocket`.

`BreezeBlue/Breeze-TTS-2` is public, so no `HF_TOKEN` is needed to fetch the weights. The
inference code is not on PyPI, so `build_commands` clones `breezeblue-ai/breeze-tts` and
installs its requirements. There is no release tag to pin to yet, which is worth revisiting
once there is: an unpinned clone means a rebuild can pick up a different engine.

## GPU sizing

Breeze report a real-time factor of 0.32 and a time to first audio under 40 ms on a warmed
H100 with the fast path. `config.yaml` asks for an H100 for that reason: the latency is the
whole point of choosing this model, and it is the number that goes away first on a smaller
card.

Memory is not the constraint. Eager inference needs about 7.7 GiB and `--fast-all` about
14.4 GiB, so the weights would fit on an L4 — but an L4 would not hold the latency, and the
fast path's 24 GB floor rules it out anyway.

`ENGINE_FAST` trades cold start for latency by building CUDA graphs and compiling the
decode stages at startup. That is why `ENGINE_STARTUP_S` is 900 rather than the 600 the
deployment next door allows.

## Concurrency

The reference streaming API serves **one request at a time**, so `predict_concurrency` is
1. This matters more than it looks: two callers on one replica means the second waits for
the first's whole utterance inside the engine, where this server cannot see the queue and
cannot cancel fairly. Scale with replicas.

It also undermines the cost estimate in `internal/routing/router.yaml`, which assumes the
GPU stays busy. A replica serving one conversation is idle between turns, so the real cost
per character is some multiple of the figure there. Measure it before believing it.

## Wire protocol

The same one the S2 Pro deployment next door speaks, so the two Go providers differ only in
what they put in a flush.

1. Client sends a JSON text frame: `{"sample_rate": 24000, "encoding": "linear16"}`.
   The server replies `{"type": "ready", "sample_rate": 24000}`, or
   `{"type": "error", ...}` and closes.
2. Client sends `{"type": "text", "id": "s1", "text": "..."}` one or more times, then
   `{"type": "flush", "id": "s1"}` to say it. The flush may carry `instruction`,
   `reference_audio` and `reference_text`.
3. Server streams binary frames of little-endian mono PCM16, then
   `{"type": "final", "id": "s1", "audio_duration_ms": ..., "processing_time_ms": ...}`.
4. `{"type": "cancel", "id": "s1"}` stops generation for barge-in and is answered with a
   final carrying `"cancelled": true`. `{"type": "close"}` ends the session.

Several ids may be in flight from the client's point of view, but only one is generating.

24 kHz is the rate the model produces, and the handshake rejects a session that asks for
anything else rather than letting the client play the audio back at the wrong speed.

## Directions are translated here

Breeze writes an English vocal event in parentheses — `(laugh)`, `(sigh)`,
`(clears throat)` — and a Chinese one in square brackets: `[笑]`, `[叹气]`.

The agent writes every direction in square brackets, whichever provider is listening,
because parentheses are ordinary punctuation that a reply may well use for something else.
So `[sigh]` becomes `(sigh)` on the way into the engine. The rewrite is deliberately
ASCII-only, which leaves a Chinese event in the brackets it already wanted.

It happens here rather than in the Go provider because a direction can be split across two
text deltas, and this is the point where the whole utterance is back in one piece.

## Test it

The Go provider's integration test covers this deployment:

```bash
cd ../../ && go test -tags integration ./internal/tts/breeze/
```
