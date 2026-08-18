# Speech to text

[Sprint 1](../sprint1.md), steps 1, 2 and 5.

## Asked for

Run [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) on
Baseten. Write a Go abstraction covering both Parakeet and Deepgram Flux, working out from
the Python codebase which parts of speech-to-text actually need standardising. Then connect
to a call, feed its audio through the router, and print the transcripts.

## What exists

[internal/stt](../../acceleration/internal/stt) is the contract, and it is short on purpose:
`Start`, `ProcessAudio`, `Events`, `Close`, plus `Provider` and `Model`.
Audio is always signed 16-bit PCM at 16 kHz, the only rate both providers accept.

| Provider                                                            | Where it runs                        |
| ------------------------------------------------------------------- | ------------------------------------ |
| [deepgram](../../acceleration/internal/stt/deepgram)                | Hosted Flux, English and multilingual |
| [parakeet](../../acceleration/internal/stt/parakeet)                | Our own Baseten deployment, 25 languages |

[deploy/parakeet](../../acceleration/deploy/parakeet) is the Truss behind the second one: a
streaming WebSocket wrapper around the NeMo model, deployed and reachable at
`PARAKEET_WS_URL`. It is self-hosted on an L4, so its price in config is the GPU divided by
measured throughput rather than a per-hour-of-audio rate, which is what makes it the cheapest
candidate at $0.079 an audio hour against Deepgram's $0.276.

[cmd/transcribe](../../acceleration/cmd/transcribe) is step 5: it joins a LiveKit room,
decodes Opus to 16 kHz mono and prints what it hears.

## What had to be standardised

Three things came out of reading the Python side, and they are the parts a caller cannot
work around:

- **Transcripts have modes.** `delta` appends, `replacement` supersedes the in-progress
  text, `final` is settled. Both providers send replacements, so a consumer that assumed
  appends would print every revision.
- **Cadence belongs to the conversation.** Providers emit transcript revisions and may mark
  one final, but `TurnStarted`, `TurnEnded`, and `TurnDetection()` are not in the shared
  contract. The agent decides from stable words instead.
- **Audio is one type everywhere.** `audio.PcmData`, so nothing in the pipeline converts
  between representations to hand a chunk to the next thing.

Everything else stayed on the concrete type, reachable through `Client()`.

## Cadence

Deepgram start/resume events and Parakeet's server-side start marker remain provider details.
Only transcript replacements and finals cross the contract. The
[voice agent](voice-agent.md) debounces those revisions per participant, so the same
conversation behavior works with either provider.

## Targets

The four capability shortcuts resolve here as everywhere. Speech-to-text also keeps its
original sprint-1 names as synonyms (`en-realtime-best`, `multi-language-realtime-best`,
`en-slow-best`, `multi-language-slow-best`) so callers written against them keep working.

`en-high-accuracy` deliberately sets no tier requirement: accuracy is worth waiting for, so
a model too slow for a live call is still a candidate.

## Not done

Nothing outstanding.
