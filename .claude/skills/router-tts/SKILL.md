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

## The top five, and what each calls the same thing

| Option | ElevenLabs | Cartesia Sonic | Inworld | Deepgram Aura | OpenAI |
| --- | --- | --- | --- | --- | --- |
| `voice` | `voice_id` | `voice.id` | `voiceId` | part of `model` | `voice` |
| `speed` | `voice_settings.speed` (0.7–1.2) | `speed` (0.6–1.5) | `audioConfig.speakingRate` | — | `speed` |
| `volume` | — | `volume` | `audioConfig.volumeGainDb` | — | — |
| `emotion`, `style` | `voice_settings.style`, v3 audio tags | `__experimental_controls.emotion` | markup in the text | — | `instructions` as prose |
| `stability`, `similarity` | `stability`, `similarity_boost` | — | `temperature` | — | — |
| `format` | `output_format` (`mp3_44100_128`, `pcm_16000`, `ulaw_8000`) | `output_format.{container,encoding,sample_rate}` | `audioConfig.{audioEncoding,sampleRateHertz}` | `encoding`, `container`, `sample_rate` | `response_format` |
| `pronunciations` | `pronunciation_dictionary_locators` | — | — | — | — |
| `chunk_schedule` | `generation_config.chunk_length_schedule` | `max_buffer_delay_ms` | — | — | — |
| `languages` | `language_code` | `language` | `language` | model per language | — |

What the table is saying:

- **Speed ranges do not overlap.** ElevenLabs stops at 1.2 and Cartesia goes to 1.5, so `speed`
  is not one number every voice accepts. A provider asked for a speed outside its own range
  refuses.
- **"Sound urgent" has three different shapes.** A settings field, an experimental control, and
  markup inside the text. `emotion` and `style` are one term because a voice either takes
  direction or it does not.
- **`stability` and `similarity` are ElevenLabs' vocabulary.** They are one term for the same
  reason: they describe cloned-voice behaviour that other vendors do not model at all.
- **Format is the one option a recording really needs.** `pcm_16000` for a live socket,
  `mp3_44100_128` for a file somebody downloads, `ulaw_8000` for telephony.

## What the router refuses to fake

A voice asked to sound urgent that speaks flatly is worse than one that says it cannot: the
caller cannot tell the difference from the audio, and the whole point of routing is that they
did not have to know which vendor answered. So options become `Terms()`, and only models whose
`supports:` lists every term asked for are candidates.

Today the batch entry `elevenlabs/eleven_v3` declares `[speed, stability, format]`, which is the
recorded path; the live entries declare nothing, so they serve requests that ask only for a
target, a voice and a language. Adding to that list means sending the field in that provider's
package under [`internal/tts`](../../../acceleration/internal/tts) first.

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
