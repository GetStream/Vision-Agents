# Streaming Parakeet on Baseten

NVIDIA Parakeet TDT 0.6B V3 served over a WebSocket, so it can back the realtime tier of
the STT router instead of only batch transcription.

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

Point `PARAKEET_WS_URL` at
`wss://model-$MODEL_ID.api.baseten.co/environments/production/websocket`.

## Wire protocol

1. Client sends a JSON text frame: `{"sample_rate": 16000, "encoding": "linear16"}`.
   The server replies `{"type": "ready"}`, or `{"type": "error", ...}` and closes.
2. Client streams binary frames of little-endian mono PCM16.
3. Server emits `{"type": "start_of_turn"}` when speech begins, then
   `{"type": "partial", "text": ...}` roughly every `WINDOW_MS`, then
   `{"type": "final", "text": ...}` once `SILENCE_MS` of trailing silence ends the turn.
   Partials and finals carry `audio_duration_ms` and `processing_time_ms`.
4. `{"type": "end_audio"}` flushes buffered audio; the server answers with a final (if any
   speech is pending) followed by `{"status": "finished"}`.

Partials are replacements, not deltas: Parakeet TDT is a chunk model, so the server
re-decodes the whole utterance each window and sends the full hypothesis. A partial is
skipped while the previous one is still decoding, which keeps long utterances from
queueing up behind their own inference.

Tuning knobs live in `config.yaml` under `environment_variables`.

## Test it

The Go provider's integration test covers this deployment:

```bash
cd ../../ && go test -tags integration ./internal/stt/parakeet/
```
