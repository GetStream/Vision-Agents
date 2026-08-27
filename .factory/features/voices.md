# Voices of your own

[Sprint 11](../sprint11.md), "Custom voices".

## Asked for

CRUD for storing custom voices in the Go backend, and an endpoint that prepares a voice for
a given text-to-speech provider, since cloning is different at every one of them.

## What exists

[internal/tts/voices](../../acceleration/internal/tts/voices) is the service and
[store/voices.go](../../acceleration/internal/store/voices.go) is where it is kept. A voice
is three things:

| Part      | What it holds                                                              |
| --------- | -------------------------------------------------------------------------- |
| `Voice`   | A name of the customer's choosing, and a description                        |
| `VoiceSample` | A recording. The audio is in the bucket at `ROUTER_BLOB_URL`; Postgres keeps the metadata |
| `VoiceBinding` | What one provider calls this voice, and whether it is `pending`, `ready` or `failed` |

```
POST   /v1/agents/voices              create
POST   /v1/agents/voices/{id}/samples upload a recording
POST   /v1/agents/voices/{id}/prepare teach it to the providers
GET    /v1/agents/voices              read back, with samples and binding state
```

## A voice is a name, not a provider id

An agent config names a voice of the customer's own, and the resolver turns that into the
id the selected provider knows it by. That indirection is the point: a config naming
`elevenlabs`' id directly could not fail over, because the next voice in the ranking has
never heard of it. A provider with no ready binding reports `ErrVoiceNotPrepared` and
routing moves on, so an unprepared provider is a candidate that loses rather than a call
that fails. A name that is not one of the customer's voices passes through untouched, which
is what keeps provider-native ids working.

## Cloning is per provider, and two cannot

ElevenLabs takes every sample, Cartesia takes only the first, Fish takes the samples with
their transcripts. `s2pro` and `breeze` have no clone API at all: S2 Pro is given reference
audio per session instead, which is a different mechanism and not one a binding can hold.

Prepare is synchronous, one provider at a time, bounded at two minutes each. It is slow and
rare — nobody clones a voice mid-call — and a caller who has to poll to find out whether a
recording was accepted is worse served than one who waits.

## Not done

- No endpoint deletes a single sample; a voice is deleted whole.
- Adding a sample does not invalidate a `ready` binding. Prepare again to pick it up.
- A deployment with Postgres but no `ROUTER_BLOB_URL` can speak voices that were already
  prepared and cannot make new ones.
