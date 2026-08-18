# Text to speech

[Sprint 2](../sprint2.md), steps 1, 2 and 5.

## Asked for

Spin up [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) on Baseten. Write a Go
abstraction covering Fish and ElevenLabs, and also Qwen-Audio-3.0-TTS Flash and Plus. Expose
a CLI where you type text and hear it.

## What exists

[internal/tts](../../acceleration/internal/tts) is the contract: `Start`, `Synthesize`,
`Interrupt`, `Events`, `Close`, plus `Provider`, `Model` and `Streaming`.

| Provider                                                        | Where it runs                              |
| --------------------------------------------------------------- | ------------------------------------------ |
| [elevenlabs](../../acceleration/internal/tts/elevenlabs)        | Hosted: Flash for latency, Multilingual v2 for quality |
| [fish](../../acceleration/internal/tts/fish)                    | Fish's hosted S2 Pro                        |
| [s2pro](../../acceleration/internal/tts/s2pro)                  | The same weights on our own Baseten deployment |

[cmd/say](../../acceleration/cmd/say) is step 5: `-text` says one line, or type lines and
hear each. It prints which provider served the utterance, the wait for first audio, how much
speech came back and what it cost.

## Two things the contract has to carry

- **A synthesis is a stream of requests, not one request.** A streaming provider is fed a
  sentence in pieces that share an `ID`, the last with `Final` set, so one spoken turn stays
  one billed synthesis. A provider that cannot take deltas reports `Streaming() == false` and
  the caller buffers a sentence into a single final request instead. This is what lets the
  agent speak sentence by sentence without paying per sentence.
- **An audio chunk does not say whether it is the last.** A streaming voice only learns that
  after the fact, and buffering a chunk to find out would cost exactly the latency the design
  exists for. `SynthesisComplete` ends an utterance instead.

`Interrupt` is on the interface rather than left to the concrete type because barge-in is not
optional for a phone call: a voice that keeps talking over the caller is broken, whichever
provider it is.

## Deployment state

[deploy/s2-pro](../../acceleration/deploy/s2-pro) is written and validated but not pushed.
Two questions are worth settling first: S2 Pro is under the Fish Audio Research License, and
the Truss wants an H100. Until then `S2PRO_WS_URL` is unset, the provider fails to build, and
routing moves to the next candidate, so a shortcut still resolves. The hosted `fish` provider
serves the same model in the meantime, at $15 per million characters against the
deployment's estimated $9.

The Sprint 6 full-stack test selects concrete `s2pro/s2-pro` when `S2PRO_WS_URL` is present,
otherwise concrete `fish/s2-pro`; it never silently routes to another voice.

## Not done

**Qwen-Audio-3.0-TTS Flash and Plus.** Not implemented. The shape of the work is known —
another provider package and a config entry, with no change to the contract — and the two
tiers map onto `low-latency` and `high-quality` the way ElevenLabs' two models already do.
