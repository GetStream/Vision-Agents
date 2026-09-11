---
name: router-sts
description: What the speech-to-speech router can be asked for, what each vendor calls it, and what it refuses to fake. Read before adding an STS option or a speech-to-speech provider.
---

# Routing speech to speech

The per-modality half of [router-interface](../router-interface/SKILL.md). The vocabulary is
[`options.STS`](../../../acceleration/internal/options/options.go); who can express what is
declared per model in [`router.yaml`](../../../acceleration/internal/routing/router.yaml);
the contract a provider implements is
[`sts.STS`](../../../acceleration/internal/sts/sts.go).

One path. A conversation is a socket at `/v1/sts/stream`, and there is no batch half: every
entry is `realtime: true`. The model on the other end is the voice activity detector, the
turn detector, the transcriber and the voice all at once, so nothing in front of it may act
as one of them. The caller's audio goes up at 16 kHz mono; the model's comes back at
whatever it speaks at, 24 kHz for all of these, under a header that says which reply each
chunk belongs to.

## The four, and what each calls the same thing

| Option | OpenAI `gpt-realtime-2` | xAI `grok-voice-think-fast-2.0` | Alibaba `qwen3.5-omni-plus-realtime` | Google `gemini-3.1-flash-live-preview` |
| --- | --- | --- | --- | --- |
| protocol | Realtime events over a socket | the same events; xAI documents it as OpenAI-compatible | the same events, older spellings kept | `BidiGenerateContent`, the Live API |
| audio in / out | 24 kHz / 24 kHz | 24 kHz / 24 kHz | 16 kHz / 24 kHz | 16 kHz / 24 kHz |
| `voice` | `audio.output.voice` (marin, cedar, …); locked after the first reply | `voice` (ara, rex, sal, eve, leo) | `voice` (Cherry, …) | `speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName` (Kore, Leda, …) |
| `instructions` | `instructions`, changeable any time | `instructions`, changeable | `instructions`, set once before the first audio | `systemInstruction`, set once at setup |
| `turn_detection: server_vad` | `audio.input.turn_detection.type: server_vad` | `turn_detection.type: server_vad` | `turn_detection.type: server_vad` | `realtimeInputConfig.automaticActivityDetection`, always on |
| `turn_detection: semantic` | `semantic_vad`, with `eagerness` | — | — | — |
| `turn_detection: none` | `turn_detection: null` plus manual commits | the same | `turn_detection: null` | `automaticActivityDetection.disabled` plus `activityStart`/`activityEnd` |
| `silence_ms`, `prefix_padding_ms` | `silence_duration_ms`, `prefix_padding_ms` | the same | the same | `silenceDurationMs`, `prefixPaddingMs` |
| threshold | `threshold`, a float | `threshold` | `threshold` | `startOfSpeechSensitivity` / `endOfSpeechSensitivity`, named levels |
| `interrupt_response` | `interrupt_response` | `interrupt_response` | — | the model always stops when it hears the caller |
| `input_transcript` | `audio.input.transcription: {model}`, once, after the stop | `input_audio_transcription: {}`; restates the turn as it grows, settled by the stop | always on, `gummy-realtime-v1` | `inputAudioTranscription: {}`, in pieces |
| `output_transcript` | always sent with audio | always sent | always sent | `outputAudioTranscription: {}` |
| `tools` | `tools: [{type: function, …}]`, changeable | the same | — | `tools: [{functionDeclarations}]`, set once; synchronous only |
| `text` | `conversation.item.create` + `response.create` | the same | — | `clientContent` with `turnComplete: true` |
| `images` | `conversation.item.create` with `input_image` | — | `input_image_buffer.append`, after audio, ≤ 2 fps | `realtimeInput.video`, ≤ 1 fps |
| interrupt from our side | `response.cancel` + `conversation.item.truncate` at the ms heard | the same | `response.cancel` | nothing: the model is not told, the rest of the reply is dropped here |
| usage | `response.done.usage`, audio and text tokens apart | not reported | `response.done.usage` | `usageMetadata`, by modality |
| session limit | 60 min | — | — | 15 min; the connection is cut every ~10 and resumed with a handle |

What the table is saying:

- **Three of the four speak one wire.** xAI and Alibaba send OpenAI's event names, so
  [`internal/sts/openairealtime`](../../../acceleration/internal/sts/openairealtime) is one
  package registered three times, differing in a `Vendor` value that says how the session
  frame is spelled and what rate to feed. Gemini is its own package, and shares no code with
  the transcriber on the same socket on purpose: the transcriber must never reconnect, and
  a conversation must.
- **Only one of them reads the words.** `semantic_turns` is a term because a caller who
  asked for it and got a silence timer cannot hear the difference. Nobody here declares
  `manual_turns`: the contract has no method for a manual commit yet, and declaring a term
  the code cannot send is the one thing terms exist to prevent.
- **The threshold has two shapes.** A float at three vendors and two named levels at the
  fourth, so it is not in the shared vocabulary. It goes in `overwrites`, spelled the way
  the vendor spells it.
- **What is set once cannot be set again.** Gemini and Qwen take instructions and tools at
  setup. `SetInstructions` and `SetTools` return `ErrInstructionsFixed` and `ErrToolsFixed`
  there rather than pretending, and `Capabilities.InstructionsMidSession` says which is
  which before a session is opened. Tools that a model takes only at setup travel on the
  request rather than in the option block, which is why `stsrouter.Request` and the `start`
  frame both carry them.
- **`interrupted` means one thing.** Every vendor reports the caller talking over the reply
  its own way; the Python plugins turned some of them into two events and had to special-case
  the duplicate. Here `SpeechStarted` says only that the model heard someone, and
  `ResponseComplete{Interrupted: true}` is the one signal a consumer drops its buffer on.
  The router session then drops any audio of that generation still arriving, because the
  model learns of a barge-in a round trip after the caller.
- **Transcripts arrive three ways.** OpenAI writes the turn down once, after the caller
  stops. xAI writes it down about once a second while they talk, each report restating the
  turn from its beginning, so those are `replacement` transcripts and the one after the stop
  is the `final`, the way the Grok transcriber already works. Gemini sends pieces that add
  up, settled when the model begins to speak, since its API has no stop event at all.
- **Temperature is not here.** Only Nova Sonic, which is not yet a provider, takes it on a
  live session, and a knob the others ignore in silence is worse than one that is absent.

## Model ids drift

`gemini-3.1-flash-live-preview` replaced `gemini-2.5-flash-native-audio-preview-12-2025`;
the older id still answers but is not what Google points at. `gpt-realtime-2` is the
family; dated snapshots share its capabilities. Check the Artificial Analysis
speech-to-speech board before adding a model: quality and time-to-first-audio move
between releases, and the `sts-fast` pin follows them.

## What the router refuses to fake

Options become `Terms()`, and only models whose `supports:` lists every term asked for are
candidates. A term nothing can serve is a 400 naming it, not a session that quietly lacks
it. `images` is not a term: it is an input modality, gated by `input_modalities: [image]`
the way `vlm` gates a text model, and `sts-vision` is the alias that requires it.

`supports:` is a promise about the Go code, checked twice. `stsrouter.New` walks the config
at boot against each provider package's `CapabilitiesFor` and refuses an entry that
promises what the package cannot send; the routing core's `Validate` hook asks the same of
the provider actually built, for a build that registered something the tables do not know.

Voices are not resolved against the voice catalogue. None of these models takes a cloned
voice, so a `custom:` name is refused up front rather than looked up at every candidate.

## Billing

One row per reply, settled by `ResponseComplete`. `AudioMs` is the caller's audio the
session forwarded since the last reply, which is the unit a transcriber bills by and the
one number every vendor lets us count ourselves. Tokens are what the vendor reported.
`Price` has one token bucket and these models bill audio in, audio out and text at three
rates, so the yaml applies the audio rates to every token and says so: an estimate, the
`perplexity/sonar` treatment. The split into audio and text tokens arrives on the event
already, for the day `Price` grows the fields.

## Adding an option

1. A field on `options.STS`, with `Merge`, `Validate` and, if it is optional behaviour, a
   `Term` and a line in `Terms()`. Pointers, so "say nothing" and "turn this off" stay
   different.
2. The same field on `StsOptions` in
   [`openapi.yaml`](../../../acceleration/api/openapi.yaml), then regenerate all five sides.
3. A field on `sts.Capabilities` and a case in `Expresses`, so a config can only declare it
   for a model whose package reports it.
4. Read it in each provider that can express it, refuse it where the vendor has no such
   thing, and declare it in `supports:`.
5. A test that a provider which cannot express it refuses rather than drops it.

## Adding a provider

One package under [`internal/sts`](../../../acceleration/internal/sts), implementing
`sts.STS`, with a `CapabilitiesFor(model)` table the router can read before a session
exists. Three test files: a no-network `handleMessage` test fed the vendor's own frames, a
fake-socket test asserting the setup frame and the interrupt on the wire, and an
`integration_test.go` embedding
[`stssuite.Suite`](../../../acceleration/internal/sts/stssuite), which speaks `mia.mp3` at
the model and holds it to an accurate transcript, audio at the declared rate, a clean
barge-in, a typed turn, a tool round and an honest answer about mid-session instructions.
Register it in [`stsrouter/registry.go`](../../../acceleration/internal/stsrouter/registry.go)
and add its `capabilitiesFor` case, then declare it in `router.yaml` with a `data_policy`,
which every model that hears the caller must have.
