---
name: router-tts
description: What the TTS router can be asked for, what each vendor calls it, and what it refuses to fake. Read before adding a TTS option or a voice provider.
---

# Routing text to speech

The per-modality half of [router-interface](../router-interface/SKILL.md). The vocabulary is
[`options.TTS`](../../../acceleration/internal/options/options.go); who can express what is
declared per model in [`router.yaml`](../../../acceleration/internal/routing/router.yaml).

Two paths again. A live voice is a socket at `/v1/tts/stream`, which streams PCM back a chunk
at a time and is judged on time-to-first-byte. A recording is a job at `/v1/tts/recordings`,
which returns one file — nobody is listening to an audiobook while it is being made, and that
is exactly what lets a codec and a bitrate be chosen instead of raw PCM.

## The three that carry the live path, and what each calls the same thing

`eleven_v3_conversational`, Sonic 3.6 and Realtime TTS-2 are the models the router leans on.
`eleven_flash_v2_5` is in the fourth column because it is where most of this vocabulary came
from, and it is the only one of the four that still has all of it.

| Option | ElevenLabs v3 conversational | Cartesia Sonic 3.6 | Inworld Realtime TTS-2 | ElevenLabs flash v2.5 |
| --- | --- | --- | --- | --- |
| endpoint | `/v1/text-to-dialogue/stream-input` | `/tts/websocket` | `voice:streamBidirectional` | `.../multi-stream-input` |
| `voice` | `voices: [id]`, exactly one per socket | `voice.id` | `create.voiceId` | in the path |
| `languages` | `language_code` query, ISO 639-1 | `language` base code **or** `locale` (`en-GB`, 3.6+), never both, plus `accent` | `language` BCP-47, inferred from the text when absent | `language_code` query |
| `speed` | — | `generation_config.speed`, 0.6–1.5 | `audioConfig.speakingRate`, 0.5–1.5, above 0.8 recommended | `voice_settings.speed`, 0.7–1.2 |
| `volume` | — | `generation_config.volume`, 0.5–2.0 | — | — |
| `emotion`, `style` | audio tags in the text | `generation_config.emotion`, closed enum, English only, beta | `[bracket]` prose, or an `instruction` field on HTTP | — |
| `stability` | `voice_settings.stability`, first message only, 0–1 | — | `deliveryMode`: `STABLE`, `BALANCED`, `CREATIVE` | `voice_settings.stability`, continuous |
| `similarity` | — | — | — | `similarity_boost` |
| elevenlabs overwrite `voice_settings.style` | — (refused) | — | — | `voice_settings.style`, 0–1 |
| `format` | `output_format=pcm_<rate>` | `output_format.{container,encoding,sample_rate}`, `raw` only on the socket | `audioConfig.{audioEncoding,sampleRateHertz,bitRate}` | `output_format` |
| `pronunciations` | `pronunciation_dictionary_locators`, first message only | `pronunciation_dict_id` | — | same as v3 |
| `chunk_schedule` | — | `max_buffer_delay_ms`, 0–5000, default 3000 | `maxBufferDelayMs` and `bufferCharThreshold` (≤1000) | `generation_config.chunk_length_schedule` |

What the table is saying:

- **The best model has the fewest knobs.** `eleven_v3_conversational` takes one setting,
  `stability`, and takes it only on the first message. No speed, no similarity, no chunk
  schedule. Everything else it can do is said in the text, which is why it is the one that
  answers `Performs()` and ships [`AudioTagPrompt`](../../../acceleration/internal/tts/elevenlabs).
- **Speed ranges do not overlap, and one provider has none.** 0.6–1.5, 0.5–1.5, 0.7–1.2, and
  nothing at all. `speed` is not one number every voice accepts, and a provider asked for a
  speed outside its own range refuses.
- **`stability` is a dial at ElevenLabs and three modes at Inworld.** ElevenLabs documents
  0 to 1, and a live 0.3 spoke on v3 conversational, v4 and v4 Turbo (2026-10-15); a
  third-party claim that v3 takes only 0.0 / 0.5 / 1.0 is in no ElevenLabs doc. Inworld has
  `STABLE`, `BALANCED`, `CREATIVE` (its `delivery_mode` overwrite). `similarity` survives
  only on the v2 models.
- **"Sound urgent" has three incompatible shapes.** A closed enum of 58 words, English only
  (Cartesia). Free English prose that persists until `[reset]`, like a stage direction
  (Inworld). Tags the model acts as it reads them (ElevenLabs v3). Prose does not reduce to
  `angry`, and `angry` throws away everything a director would actually say. A single
  `emotion` string is expressible by Cartesia and nobody else; the other two want the
  direction in the text.
- **`volume` is Cartesia's alone**, and Inworld's nearest equivalent is `[very quiet]` — prose
  again, not a number.
- **`pronunciations` is not a per-request map anywhere.** ElevenLabs and Cartesia both want a
  dictionary uploaded first and referenced by id; Inworld has nothing. `map[string]string` on
  the options is the wrong shape, and the honest fix is the voice-catalogue treatment: prepare
  it once, reference it per session.
- **`chunk_schedule` is characters for one vendor and milliseconds for two.** It is really
  "how long to buffer before speaking". v3 conversational decides for itself — it waits for
  roughly 40 characters and 8 words — and gives you `flush` as the only lever, which is what
  `Final` already sends.
- **Format barely matters live.** Every socket here is PCM. It is the recorded path that needs
  it: `mp3_44100_128` for a file somebody downloads, `ulaw_8000` for telephony.

## Model ids drift, and two of these have

- Sonic 3.6 went generally available as `sonic-3.6`. `sonic-preview` — which
  [`cartesia.go`](../../../acceleration/internal/tts/cartesia/cartesia.go) still pins and its
  own comment predicted — now means the beta channel that tracks whatever is next, explicitly
  not for production.
- The router ships `inworld-tts-2-flash`, five times faster to first byte than `inworld-tts-2`
  and the cheapest of the four. It also ignores steering entirely: bracket instructions and
  the `instruction` field are dropped, though non-verbals like `[laugh]` still work. Steering is
  `inworld-tts-2` only, so declaring `emotion` on the flash entry would be a lie. `temperature`
  is ignored across the TTS-2 family; `deliveryMode` replaced it.
- `eleven_v3_conversational` registers exactly one voice per socket. `eleven_v3` allows ten,
  which is what makes it the scripted-dialogue model rather than the conversational one.

## What the router refuses to fake

A voice asked to sound urgent that speaks flatly is worse than one that says it cannot: the
caller cannot tell the difference from the audio, and the whole point of routing is that they
did not have to know which vendor answered. So options become `Terms()`, and only models whose
`supports:` lists every term asked for are candidates.

Today the batch entry `elevenlabs/eleven_v3` declares `[stability, similarity, format]`, which is the recorded path; it declares no `speed`, so a recording that asks for one is refused. Of the live entries, `eleven_flash_v2_5` and `eleven_multilingual_v2`
declare `[speed, stability, similarity]`, sent as `voice_settings` when each context opens (stability 0.5,
similarity 0.8 when not asked); a speed outside 0.7–1.2 or a stability or similarity outside
0–1 is refused when the provider is built. The dialogue models (v3 conversational, v4, v4
Turbo) declare `[stability]` and have no speed, so a request naming one routes past them, and
`NewDialogue` refuses a speed or a similarity rather than drop it. A refused value, a speed
included, is the caller's mistake (`routing.ErrRequest`): no stat is recorded, so it cannot
mark the model unavailable for everyone. Every other live entry declares nothing, so it
serves requests that ask only for a target, a voice and a language. Cartesia
(`generation_config.speed`, 0.6–1.5) and Inworld (`audioConfig.speakingRate`, 0.5–1.5) can
express a speed but do not send one yet; wire it in their package, refuse outside their range,
then add `speed` to their `supports:`. Adding to that list means sending the field in that
provider's package under [`internal/tts`](../../../acceleration/internal/tts) first.

An agent's voice session passes no voice options: `startVoice` builds its request with a
target and a voice, and no voice options (`internal/agent/pipeline.go`), and the agent config has no `speed`
or other voice setting since it was taken out on purpose ("a voice speaks at its own pace").
A speed, stability, similarity or overwrite reaches a voice only through a request to
`/v1/tts/stream` or `/v1/tts/recordings`.

Batch v3 has no `speed` because it ignores one. Live on 2026-10-16 a speed of 0.7, 1.0, 1.2 and
1.5 all got a 200 from `eleven_v3`, in times that did not follow the speed, where Flash answered
1.5 with a 400. None of the ElevenLabs pages opened (best-practices, models,
text-to-speech/convert) says so either way, so re-probe before adding it back.

Voice ids are not a term. They are resolved by the voice catalogue in
[`internal/tts/voices`](../../../acceleration/internal/tts/voices), so a name that no provider
has is a 404 rather than a routing failure. `custom:` in front of one says it is the
customer's own and nothing else, so a name that is not theirs is refused rather than passed to
the provider's library.

A voice that is not the customer's own is checked when the voice opens (`Resolver.ResolveVoice`,
`Catalogue.Check`), for every caller: the socket, a session's agent and a recording.

- A voice in the provider's cached listing (`libraryTTL`, an hour) goes through with no
  round trip. Any other voice is looked up, and one that exists is remembered for an hour.
- Only the provider's own "no such voice" refuses. ElevenLabs answers that as
  `detail.code: voice_not_found`, with a 404 or a 400, so the code decides and not the status.
  The refusal is `routing.ErrRequest` and names the closest listed voice by edit distance over
  id and name. When none takes the voice the caller gets the refusal, not "every candidate failed".
- A refused voice is not handed to a vendor that cannot look voices up (Cartesia, Inworld,
  any vendor with no library): it would start and fail on the first line (`tts-quality` did,
  on Inworld and Cartesia, before AI-1395's second PR). `Select` asks the resolver once,
  before the first candidate (`routing.VoiceVetter`, `Resolver.Unvouched`,
  `Catalogue.Unvouched`), so the answer does not depend on which vendor health ranks first.
  Such a vendor still takes the voice when its own cached listing holds that id: a listed
  name does not count, since Cartesia and Inworld speak a voice by id. A listing that cannot
  be read holds nothing.
- That first ask is made only for a voice shaped like an ElevenLabs id: 20 letters and digits
  (`Finder.Shaped`; ElevenLabs does not document the format, every id seen live on 2026-10-16
  has it). A Cartesia UUID, an Inworld name such as `Ashley`, a mistyped name such as
  `Rachell`, `custom:` and no voice are not asked about and route as #989 left them: a name
  ElevenLabs refuses on a shortcut that also reaches Cartesia or Inworld still starts there
  and fails on the first line. Staging's Inworld key cannot list voices (403, 2026-10-16), so
  leaving Inworld out for a name its listing does not vouch for would refuse every Inworld voice.
- ElevenLabs answers an id that cannot be one of its voices, such as a Cartesia UUID, with
  400 `detail.status: invalid_uid` (live 2026-10-16). `Exists` returns `errNotAVoiceID`, and
  `Check` lets the voice through without a warning and without turning the check off.
- Nothing is left out when no candidate can look the voice up, when the lookup failed, for a
  customer's own voice, for `custom:`, or for a candidate whose overwrites name a `voice_id`.
- A lookup that fails another way (a bad key, a 5xx, past `lookupTimeout`, 3s) lets the voice
  through as before. An unreadable listing still refuses, without a suggestion.
- ElevenLabs looks twice, `GET /v1/voices/{id}` and then `GET /v1/shared-voices?search={id}`,
  matched on the exact id. The first alone says `voice_not_found` for a Voice Library voice
  that speaks (ElevenLabs adds it to the account the first time it does). Do not probe a
  Voice Library voice with a synthesis: that adds it to the account.
- `/v1/voices?show_legacy=false` leaves legacy voices out of the listing, so
  `21m00Tcm4TlvDq8ikWAM` is looked up, found and spoken. "Not in the listing" is not a
  refusal.
- A "no such voice" answer is remembered for `rememberFor` (30s), and so is a failed listing
  or lookup per provider, so one `Start` that tries five ElevenLabs models asks once and the
  refusal is worded once; a failing or hanging vendor costs one lookup per window, not one per open.
- A candidate whose `overwrites.<vendor>.voice_id` is set is not checked: the overwrite is the
  voice spoken (`routing.VoiceOverwritten`, `ttsrouter/registry.go` `voiceOr`), so the
  request's `voice` is never asked of the provider.
- Not checked: Cartesia and Inworld voices on a target with no ElevenLabs candidate (their listers have no `Exists`), a router with no database (no resolver) and the standalone `cmd/agent` binary.

## Three things that narrow candidates without being terms

- `providers` is a priority list tried in the order written, and this router's half of it is
  the streaming one: a vendor named live gets its socket model, not its file-returning one.
- `data_policy` is a requirement. Every entry in `router.yaml` declares what the vendor does
  with the text, `Config.Validate()` refuses one that does not, and an undeclared handling
  satisfies no policy. Only the two entries on our own Baseten deployment currently promise no
  training and no retention, so `allow_training: false` routes there until an account
  elsewhere is opted out and its entry is edited.
- `overwrites` is per vendor, decoded into a typed struct with `DisallowUnknownFields`, so a
  misspelt key or a wrong type fails the request with `routing.ErrOverwrite`, naming the
  provider and the field. `Select` then tries only that vendor's other models (OpenAI's
  realtime and Live models read different structs out of one block) and never another
  vendor, which is handed its own block and would drop the overwrite in silence. What a voice can be
  steered by this way: `elevenlabs: {voice_id, voice_settings: {style}}`,
  `inworld: {voice_id, delivery_mode}`. The recording path reads the same `elevenlabs` block.
  ElevenLabs' `style` is a number from 0 to 1, how far to exaggerate the speaker's own style,
  and not the shared `style`, which is a word to sound like. Only the models before v3 take
  it (Flash v2.5, Multilingual v2): the v3 and v4 models, the batch `eleven_v3` included,
  refuse it as an overwrite error, so `tts-quality` refuses it and `en-low-latency` is
  spoken by Flash.
- A `voice_id` in any vendor's block is the voice spoken: `startCandidate` hands it to the
  factory as the spec's voice, so Cartesia, whose factory reads no block, speaks it too. A
  voice asked only that way, with no top-level `voice`, narrows the candidates to the vendors
  that named one before anything is tried (`Router.voiced`): any other vendor would be handed
  no voice and speak its default, which nobody can hear is wrong (ElevenLabs' default is
  "Spuds Oxley - Grandpa", `elevenlabs.DefaultVoiceID`). None of them a candidate, or none
  with a factory (a recording asked in a Cartesia voice), is refused as `routing.ErrRequest`.
  A top-level voice goes to every candidate, so nothing is left out for it.
- The voice that spoke, the provider's default included, is `voice` on the socket's `started`
  frame and on `Speech` (`tts.Voiced`). A provider that cannot say leaves it out rather than
  echo the voice asked for.

## Only ElevenLabs records

- A recording is built by `ttsrouter.DefaultRecorderRegistry` (`ttsrouter/recordings.go`), which
  has ElevenLabs alone. Every other vendor in `router.yaml` (Cartesia, Inworld, Breeze, Speechify,
  Gemini, Fish, s2pro) only streams. Any ElevenLabs model records, its live ones included: on
  staging on 2026-10-19 an inline recording on `elevenlabs/eleven_v4` returned audio, and one on
  `cartesia/sonic-preview`, `inworld/inworld-tts-2` or `breeze/breeze-tts-2` failed with
  `no factory registered`.
- `Recordings.Check` runs before a recording is accepted (`recordSpeech` in `api/recordings.go`).
  When no candidate's vendor has a recorder, such as a pinned `cartesia/sonic-preview` or a
  shortcut or priority list made only of vendors that stream, the request is a 400 that names
  them and the models that record. One recorder among the candidates is enough: `tts-quality`
  is served by ElevenLabs, because `Select` passes over a candidate with no factory without
  counting it against the model. A voice asked only in a streaming vendor's overwrites is still
  refused once routed (`Router.voiced`), as a failed job.
- `GET /v1/tts/routes` and `/v1/tts/routes/{target}` say per candidate whether a recording can
  use it (`can_record`, from the recorder registry). The list and its health stay the live
  router's. STT and STS leave the field out.
- A new recorder is one `Register` in `DefaultRecorderRegistry`; the check and `can_record`
  follow from it.

## Adding an option

1. A field on `options.TTS`, with `Merge` and a `Term` plus a line in `Terms()` if it is
   optional behaviour.
2. The same field on `TtsOptions`, on its Go struct in
   `internal/api`. Then run
   `go run ./cmd/openapi` and regenerate all three clients.
3. Read it in each provider that can express it, refuse it where the vendor has a range and the
   request is outside it, and declare it in `supports:`.
4. A test for the refusal, not only for the happy path.
