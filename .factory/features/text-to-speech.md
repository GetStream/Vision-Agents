# Text to speech

[Sprint 2](../sprint2.md), steps 1, 2 and 5, with expressiveness and
[voices of your own](voices.md) added in [sprint 11](../sprint11.md).

## Asked for

Spin up [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) on Baseten. Write a Go
abstraction covering Fish and ElevenLabs, and also Qwen-Audio-3.0-TTS Flash and Plus. Expose
a CLI where you type text and hear it.

## What exists

[internal/tts](../../acceleration/internal/tts) is the contract: `Start`, `Synthesize`,
`Interrupt`, `Events`, `Close`, plus `Provider`, `Model` and `Streaming`.

| Provider                                                        | Where it runs                              |
| --------------------------------------------------------------- | ------------------------------------------ |
| [cartesia](../../acceleration/internal/tts/cartesia)            | Hosted Sonic                                |
| [elevenlabs](../../acceleration/internal/tts/elevenlabs)        | Hosted: Flash for latency, Multilingual v2 and v3 for quality |
| [fish](../../acceleration/internal/tts/fish)                    | Fish's hosted S2 Pro                        |
| [s2pro](../../acceleration/internal/tts/s2pro)                  | The same weights on our own Baseten deployment |
| [breeze](../../acceleration/internal/tts/breeze)                | Breeze TTS 2 on our own deployment, English and Chinese |

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

## Expressiveness is a prompt, not a field

ElevenLabs v3 performs `[laughs]` and `[whispers]` inline; Breeze performs a closed set of
four; the rest say the brackets or ignore them. So a provider declares `Performs()` and
carries its own `Prompt()`, which the agent appends to its instructions when that voice is
selected. Telling the model what this voice can do is the only thing that generalises —
there is no shared tag vocabulary to standardise, and the ElevenLabs set is open-ended by
design.

Tags are stripped from the transcript and the events either way, so what is stored is what
was said rather than how. A provider that does not perform them is sent the stripped text
too, which is what stops a voice reading stage directions aloud.

## Deployment state

[deploy/s2-pro](../../acceleration/deploy/s2-pro) is written and validated but not pushed.
Two questions are worth settling first: S2 Pro is under the Fish Audio Research License, and
the Truss wants an H100. Until then `S2PRO_WS_URL` is unset, the provider fails to build, and
routing moves to the next candidate, so a shortcut still resolves. The hosted `fish` provider
serves the same model in the meantime, at $15 per million characters against the
deployment's estimated $9.

The Sprint 6 full-stack test selects concrete `s2pro/s2-pro` when `S2PRO_WS_URL` is present,
otherwise concrete `fish/s2-pro`; it never silently routes to another voice.

Breeze is in the same position under the BreezeBlue Research and Non-Commercial License, and
is not deployed either.

## Not done

**Qwen-Audio-3.0-TTS Flash and Plus.** Not implemented. The shape of the work is known —
another provider package and a config entry, with no change to the contract — and the two
tiers map onto `low-latency` and `high-quality` the way ElevenLabs' two models already do.
