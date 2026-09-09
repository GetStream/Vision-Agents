---
name: router-stt
description: What the STT router can be asked for, what each vendor calls it, and what it refuses to fake. Read before adding an STT option or an STT provider.
---

# Routing speech to text

The per-modality half of [router-interface](../router-interface/SKILL.md). The vocabulary is
[`options.STT`](../../../acceleration/internal/options/options.go); who can express what is
declared per model in [`router.yaml`](../../../acceleration/internal/routing/router.yaml).

Two paths, two sets of models. Live transcription is a socket at `/v1/stt/stream` and routes
to the models marked `realtime: true`. A recording is a job at `/v1/stt/recordings` and routes
to the ones marked `realtime: false` — the batch endpoints, which are both cheaper per hour and
more accurate than the same vendor's streaming model, because they have the whole file in front
of them. `en-recorded` and `multilingual-recorded` are the aliases for that half, and they
`require_recorded`, so a recording is never streamed at a live model by accident.

## The top five, and what each calls the same thing

| Option | Deepgram | AssemblyAI | ElevenLabs Scribe | Speechmatics | Cartesia Ink |
| --- | --- | --- | --- | --- | --- |
| `languages` | `language` | `language_code` | `language_code` | `transcription_config.language` | `language` |
| `detect_language` | `detect_language` | `language_detection` | detected by default | `language_identification` | — |
| `interim` | `interim_results` | partials by default | live only | `enable_partials` | partials by default |
| `endpointing`, `silence_ms` | `endpointing`, `utterance_end_ms` | `end_of_turn_confidence_threshold` | — | `max_delay` | `max_silence_duration_secs` |
| `diarize` | `diarize` | `speaker_labels` | `diarize` | `diarization: speaker` | — |
| `max_speakers` | — | `speakers_expected` (caps at 10) | `num_speakers` (up to 32) | `speaker_diarization_config.max_speakers` | — |
| `keyterms` | `keyterm` (batch), `keyterms` (Flux) | `keyterms_prompt`, `word_boost` | — | `additional_vocab` | — |
| `format` | `smart_format`, `punctuate` | `format_text`, `punctuate` | on by default | `punctuation_overrides` | — |
| `redact` | `redact` | `redact_pii` + policies | — | `transcript_filtering_config` | — |
| `words` | `words` in the response | `words` in the response | `timestamps_granularity` | `words` in the response | — |
| `summary`, `entities` | `summarize`, `detect_entities` | `summarization`, `entity_detection` | — | `summarization` | — |
| `channels` | `multichannel` | `multichannel` | — | `diarization: channel` | — |

Three things this table is really saying:

- **Everyone diarizes differently and only some can be told to stop.** `max_speakers` is its own
  term, separate from `diarize`, because a provider that finds however many speakers it thought
  it heard is not serving a request that said "two people".
- **Endpointing is either a pause or a judgement.** Deepgram and Speechmatics take milliseconds
  of silence; AssemblyAI's streaming model decides a turn is over from the words. `endpointing`
  names which, and `silence_ms` only means anything for the first kind.
- **Vocabulary is not universally supported.** Deepgram, AssemblyAI and Speechmatics take term
  lists; Scribe and Ink do not. A call full of drug names has to be routed, not hoped for.

## What the router refuses to fake

A request's options become `Terms()`, and routing only considers models whose `supports:` lists
every term asked for. A term nothing can serve is a 400 naming it, not a transcript that quietly
lacks it. That is the whole design: being told is better than being answered wrongly.

So `supports:` is a promise about the client code, not about the vendor's docs. Today
`deepgram/flux-general-*` declares `[keyterms, endpointing]`, the Gemini, Grok and Muse live
models declare `[keyterms]`, and `deepgram/nova-3` — the recorded path — declares the batch
list: language detection, diarization, words, formatting, redaction, summary, entities, keyterms
and channels. Everything else declares nothing, which means it serves the requests that ask for
nothing extra.

Adding a term to a model means sending it in that provider's package under
[`internal/stt`](../../../acceleration/internal/stt) *first*, then declaring it. Declaring it to
make routing succeed is the one thing this design exists to prevent.

`max_speakers` is where refusing earns its keep: AssemblyAI caps at 10 and Scribe at 32, so a
request for 16 speakers is a different set of candidates than a request for 8.

## Adding an option

1. A field on `options.STT`, with `Merge` and, if it is optional behaviour, a `Term` and a line
   in `Terms()`. Pointers, so "say nothing" and "turn this off" stay different.
2. The same field on `SttOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all three clients.
3. Read it in each provider that can express it, and declare it in `supports:`.
4. A test that a provider which cannot express it refuses rather than drops it.

Subtitles are the exception worth knowing: `output: srt` is not a term, because rendering
lines out of words and timings is something the router does itself once it has the timings.
