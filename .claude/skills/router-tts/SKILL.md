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
| `stability` | `voice_settings.stability`, first message only, **only 0.0 / 0.5 / 1.0** | — | `deliveryMode`: `STABLE`, `BALANCED`, `CREATIVE` | `voice_settings.stability`, continuous |
| `similarity` | — | — | — | `similarity_boost` |
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
- **`stability` has stopped being a dial.** ElevenLabs v3 accepts only 0.0, 0.5 and 1.0 —
  creative, natural, robust — and values between them are not supported. Inworld says the same
  thing in words: `STABLE`, `BALANCED`, `CREATIVE`. Two of the three model it as three named
  modes and the third does not model it at all, so `Stability *float64` is the wrong shape for
  every provider that has the feature. `similarity` survives only on the v2 models.
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

Today the batch entry `elevenlabs/eleven_v3` declares `[speed, stability, format]`, which is
the recorded path; the live entries declare nothing, so they serve requests that ask only for a
target, a voice and a language. Adding to that list means sending the field in that provider's
package under [`internal/tts`](../../../acceleration/internal/tts) first.

That `speed` is the one declaration the vendor docs do not clearly back. `voice_settings.speed`
is documented at 0.7–1.2 for text to speech generally, but v3's documented voice setting is
`stability` alone, and its dialogue socket accepts nothing else. Confirm against a live
recording before trusting it, and drop the term if v3 ignores it.

Voice ids are not a term. They are resolved by the voice catalogue in
[`internal/tts/voices`](../../../acceleration/internal/tts/voices), so a name that no provider
has is a 404 rather than a routing failure.

## Adding an option

1. A field on `options.TTS`, with `Merge` and a `Term` plus a line in `Terms()` if it is
   optional behaviour.
2. The same field on `TtsOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all three clients.
3. Read it in each provider that can express it, refuse it where the vendor has a range and the
   request is outside it, and declare it in `supports:`.
4. A test for the refusal, not only for the happy path.
