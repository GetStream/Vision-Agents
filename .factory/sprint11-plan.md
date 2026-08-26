# Sprint 11 implementation plan

Three independent workstreams from `sprint11.md`, against the Go `acceleration` module.
Nothing in one blocks another. Suggested order is smallest first: keyterms, then
expressiveness, then custom voices.

---

## Findings that shape the plan

**Audio tags need a different endpoint, not a different flag.** Our ElevenLabs provider
uses the multi-stream Text-to-Speech WebSocket with `eleven_flash_v2_5`. ElevenLabs does
not serve v3 on that endpoint at all, and multi-context WebSockets are explicitly
unsupported for v3. The tag-capable realtime model is `eleven_v3_conversational` on the
Text to Dialogue WebSocket, `/v1/text-to-dialogue/stream-input`. That is roughly 280 ms
time-to-first-byte against the ~75 ms we get today, and each open connection reserves a
dialogue session from a pool separate from the standard concurrency limit.

**Text to Dialogue has no contexts.** The existing provider maps `tts.Request.ID` onto
server-side context ids, which is what makes per-utterance attribution and barge-in clean.
Text to Dialogue has a single stream with `new_turn` boundaries and no context ids, so
audio must be attributed by flush order. `internal/tts/s2pro` already does exactly this,
so the new provider should be modelled on s2pro rather than on the existing elevenlabs one.

**Deepgram keyterms are already half-built.** `deepgram.Options.Keyterms` flows into the
SDK's `keyterm` query param. Nothing ever sets it, because `routing.Spec` carries only
model, language hints and voice.

**No object storage exists yet.** None of the three Go modules depends on an S3 or GCS
client, so custom voices needs one added.

**A raw voice id breaks router failover.** `AgentConfig.Voice` is a provider-specific id,
but the TTS router fails over between providers. Custom voices should be stored per
provider and resolved once the candidate is known.

---

## Workstream 1 — TTS expressiveness

Two halves: telling the model it may be expressive, and having a voice that can perform it.

### 1a. Provider-contributed instructions

Add a capability method to the `tts.TTS` contract in
`acceleration/internal/tts/tts.go`, alongside the existing `Provider()`, `Model()` and
`Streaming()`:

```go
// Prompt is provider-specific guidance for the model whose text this voice speaks, or
// empty when the voice needs none.
Prompt() string
```

- Implement on all providers. ElevenLabs returns the audio-tag guidance only for models
  that honour tags, and empty for `eleven_flash_v2_5`. Cartesia, Fish and s2pro return
  empty for now.
- The guidance text is prose, not an enum: it names the documented emotion, delivery,
  human-sound, pacing and character tags from the sprint notes as examples, and says any
  bracketed direction is allowed. Keep it short — it is prepended to every turn.
- Forward through `ttsrouter.Session`, same as `Streaming()`.
- Capture it into an agent field at join time rather than reading `a.tts` inside
  `instructions()`, which already runs under the agent lock.
- Append in `Agent.instructions()` after the recalled memory and the configured prompt.

### 1b. Keeping tags out of the transcript

`say()` writes the same string to `a.spoken` (which becomes conversation history) and to
the `ResponseDelta` event that text clients and transcripts render. Tags must reach the
voice but not either of those.

- Strip bracketed tags from the text used for `ResponseDelta` and `a.spoken`; pass the raw
  text to the chunker and on to `Synthesize`.
- Strip on the synthesis path too whenever the routed voice does not perform tags, so a
  model that emits them anyway does not have them read aloud.
- Watch the sentence chunker: a tag adjacent to terminal punctuation must not produce an
  empty or tag-only sentence.

### 1c. Text to Dialogue provider for `eleven_v3_conversational`

New implementation inside `internal/tts/elevenlabs`, selected by model rather than by a
new provider name — the registry factory already receives `spec.Model`, and keeping
`ProviderName = "elevenlabs"` keeps the stats and pricing rows correct. Add the model to
the `tts:` section of `internal/routing/router.yaml` with its own price and a
`high-quality` tier so the low-latency aliases keep routing to Flash.

Protocol, from the ElevenLabs docs:

- Connect to `wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input?model_id=…&output_format=…`,
  API key in the `xi-api-key` header.
- First frame registers voices: `{"voices": ["<voice id>"]}`. `eleven_v3_conversational`
  permits exactly one.
- Text frames are `{"inputs": [{"text": …, "voice_id": …, "new_turn": …}]}`. Set
  `new_turn: true` at a turn boundary so prosody resets.
- Buffering is a fixed server threshold, about 40 characters and 8 words — there is no
  `chunk_length_schedule`. `{"flush": true}` forces generation without closing.
- `{"close_socket": true}` flushes, drains, sends `is_final: true`, then closes.
- Response fields are snake_case (`audio`, `is_final`, `error`), unlike the camelCase
  `contextId`/`isFinal` of the existing provider.
- The connection dies after 20 s of client silence; `{"keep_alive": true}` resets it, so
  this provider needs a keepalive goroutine. Deepgram's ping loop is the local precedent.

Design points to settle while building:

- **Verify `output_format=pcm_24000` is accepted.** The docs example uses
  `mp3_44100_128`. Everything downstream (`audio.PcmData`, the Opus speaker) expects PCM,
  so if only MP3 is offered this needs a decode step and the latency case gets worse.
- **Barge-in.** There is no context to close. Options are to close and reopen the socket,
  which costs a dialogue session and reconnect latency, or to drop audio locally and let
  the server finish generating, which we still pay for. The agent already gates playback
  on `speaking(turnOf(…))`, so local dropping is correct for the listener but wasteful.
  Recommend closing and reopening, with the reconnect hidden behind the same idle-reconnect
  pattern Cartesia already uses.
- **Attribution.** Map utterances to flush order the way s2pro does.
- **Concurrency.** A held-open dialogue session per call is a different capacity model
  from the current one. Worth a note in `acceleration/README.md`.

### Tests

- `Prompt()` returns guidance for a tag-capable model and empty for `eleven_flash_v2_5`.
- Agent-level: the guidance reaches `llm.Request.Instructions`, and a tagged delta reaches
  `Synthesize` while the matching `ResponseDelta` has no tags.
- Dialogue provider against a `httptest` WebSocket, mirroring the existing
  `elevenlabs_test.go` and `s2pro_test.go` shapes: handshake registers one voice, deltas
  become `inputs`, flush and close behaviour, keepalive, audio attribution by order,
  interrupt.
- Integration test behind `//go:build integration` needing `ELEVENLABS_API_KEY`.

### Files

`internal/tts/tts.go`, the four provider packages, `internal/ttsrouter/session.go`,
`internal/ttsrouter/registry.go`, `internal/routing/router.yaml`,
`internal/agent/agent.go`, plus tests.

---

## Workstream 2 — Custom voices

Mirrors the agent-config resource file for file. Audio is uploaded to our API and stored
in S3 or GCS.

### Storage

Add `gocloud.dev/blob` with the `s3blob` and `gcsblob` drivers, so one `ROUTER_BLOB_URL`
env var (`s3://bucket?region=…` or `gs://bucket`) covers both without a second code path.
Credentials come from the ambient provider chain, matching how every other secret in this
service is already an env var. Without the var set, the voices endpoints report "no blob
storage configured" the way the API already reports "no database configured".

### Data model

Two tables, because the same custom voice can be prepared with more than one provider and
the router fails over between them.

- `voices` — `id`, `customer_id`, `name`, `description`, `created_at`, `updated_at`,
  `deleted_at`. Unique on `(customer_id, name)` where `deleted_at is null`, matching
  `agent_configs`.
- `voice_samples` — `id`, `voice_id`, `object_key`, `content_type`, `bytes`, `created_at`.
- `voice_bindings` — `id`, `voice_id`, `provider`, `external_id`, `state`
  (`pending`/`ready`/`failed`), `error`, `created_at`, `updated_at`. Unique on
  `(voice_id, provider)`.

Goose migration under `acceleration/migrations/`, bun models in `internal/store/models.go`,
CRUD in a new `internal/store/voices.go` following `internal/store/agents.go` — `newID()`,
soft delete, full-replacement updates.

### HTTP API

Add to `api/openapi.yaml`, regenerate with
`go tool oapi-codegen -config api/oapi-codegen.yaml api/openapi.yaml`, implement in a new
`internal/api/voices.go` against the generated `StrictServerInterface`.

- `GET /v1/voices`, `POST /v1/voices`, `GET|PUT|DELETE /v1/voices/{id}` — plain CRUD.
- `POST /v1/voices/{id}/samples` — multipart upload, streamed to blob storage, returns the
  sample record. Enforce a size cap and an audio content-type allowlist.
- `POST /v1/voices/{id}/prepare` — body names the TTS provider. Reads the samples back from
  blob storage, calls that provider's clone API, and writes the resulting binding.
- `DELETE /v1/voices/{id}` also deletes the remote voices for every ready binding.

`prepare` is synchronous to start with, since cloning is seconds not minutes, and the
`state` column leaves room to make it a background job later without an API change.

### Provider preparation

Cloning is a control-plane HTTP call with no session attached, so it does not belong on the
streaming `tts.TTS` interface. New package `internal/tts/voices` with a small contract and
its own registry keyed by provider name, mirroring `ttsrouter.Registry`:

```go
// Cloner prepares a voice with a provider and returns the id its sessions use.
type Cloner interface {
	Prepare(ctx context.Context, request Request) (string, error)
	Delete(ctx context.Context, externalID string) error
}
```

- ElevenLabs — instant voice clone via `POST /v1/voices/add`, multipart samples.
- Cartesia — `POST /voices/clone`.
- Fish — create a model, keep the returned `reference_id`.
- s2pro — **not implemented.** The deployment takes `ReferenceAudio` and `ReferenceText`
  per session rather than registering a voice, so a binding would have to carry the audio
  itself down through `routing.Spec` into the factory. Instead s2pro registers no cloner:
  no binding exists, and the router skips it for a call that asks for a custom voice. The
  plumbing is worth doing only if s2pro becomes a voice somebody actually wants cloned.

### Resolving a custom voice at session start

`routing.Router.startCandidate` is the only place that knows both the requested voice and
the chosen provider:

```go
spec := Spec{
	Model:         candidate.Config.Model,
	LanguageHints: request.LanguageHints,
	Voice:         request.Voice,
	Logger:        r.logger,
}
```

Add an optional resolver hook to `routing.Options` — `func(ctx, provider, voice string)
(string, error)` — and have `ttsrouter` supply one backed by its existing `*store.Store`.
When the voice value names a stored voice it is swapped for that provider's binding;
otherwise it passes through unchanged, so today's raw provider ids keep working. A voice
with no binding for the chosen provider fails that candidate, which makes the router move
to the next one rather than speak in the wrong voice.

### Tests

- Store CRUD round-trip, and cascade behaviour on delete, in the `//go:build integration`
  suite alongside the agent-config tests.
- API: upload then prepare then read back, against a `httptest` provider.
- Resolver: a stored voice resolves per provider, an unknown value passes through, a
  missing binding fails the candidate rather than the request.

### Files

New migration, `internal/store/models.go`, `internal/store/voices.go`, `api/openapi.yaml`,
regenerated `internal/api/generated.go`, `internal/api/voices.go`, new
`internal/tts/voices/` package with four implementations, `internal/routing/router.go`,
`internal/routing/registry.go`, `internal/ttsrouter/router.go`, `internal/blob` wrapper,
`cmd/router/main.go` for the new env var, `acceleration/README.md`.

---

## Workstream 3 — STT keyterms

Mostly plumbing, because the Deepgram half already exists.

1. Migration: `ALTER TABLE agent_configs ADD COLUMN keyterms JSONB NOT NULL DEFAULT '[]'`.
2. `store.AgentConfig.Keyterms []string`, and add it to `normalizeConfig` so the column is
   never SQL `NULL`.
3. `api/openapi.yaml`: `keyterms` on `AgentConfig` and `AgentConfigRequest`. Regenerate,
   then map it in `storedConfig` and `agentConfigOf` in `internal/api/configs.go`.
   Remember `PUT` is a full replace, so omitting it clears the list.
4. `session.Spec.Keyterms`, populated in `FromConfig`, overridable per session in
   `specOf`.
5. `agent.Options.Keyterms`, passed into `sttrouter.Request` where the agent opens a
   listener per participant.
6. `sttrouter.Request.Keyterms` into `routing.Spec.Keyterms`. The `Spec` doc comment
   already says fields that mean nothing to a modality are ignored by its factories, which
   is exactly how `Voice` behaves, so this is consistent.
7. `sttrouter` registry passes them to `deepgram.Options.Keyterms`. Parakeet ignores them.
8. Validate in the config handler: trim, drop empties, cap the count. Deepgram limits how
   many keyterms a request may carry; confirm the current limit and reject above it with a
   400 rather than letting the upstream fail at connect time.

Out of scope: Deepgram's mid-session `Configure()` can update keyterms on a live socket,
which would let a config edit apply without a reconnect. Not needed now.

### Tests

- Registry passes keyterms to the Deepgram provider, mirroring the existing "registry
  passes voice and language to the provider" case.
- Config round-trip through the API in the integration suite.
- Keyterm validation rejects an over-long list.

### Files

New migration, `internal/store/models.go`, `internal/store/agents.go`, `api/openapi.yaml`,
regenerated `internal/api/generated.go`, `internal/api/configs.go`,
`internal/session/spec.go`, `internal/api/sessions.go`, `internal/agent/agent.go`,
`internal/sttrouter/router.go`, `internal/sttrouter/registry.go`,
`internal/routing/registry.go`, `internal/routing/router.go`, plus tests.

---

## Open questions

1. Does the Text to Dialogue WebSocket accept `output_format=pcm_24000`? If it is MP3
   only, the provider needs a decoder and the latency argument gets weaker.
2. Barge-in on Text to Dialogue: reconnect per interrupt, or drop locally and pay for the
   generated audio?
3. Which bucket and which cloud for voice samples, and is there a retention policy for the
   uploaded audio?
4. Should `prepare` stay synchronous, or is a background job wanted from the start?
