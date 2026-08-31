---
name: tts
description: Build and test a new streaming TTS provider in acceleration/internal/tts.
---

# New TTS provider (Go)

## Build

- One package per provider: `acceleration/internal/tts/<name>/<name>.go`, implementing `tts.TTS` (`Start`, `Synthesize`, `Interrupt`, `Events`, `Close`, `Provider`, `Model`, `Streaming`, `Performs`, `Prompt`) plus `SampleRate() int`, which the suite needs.
- One utterance is one `Request.ID`: several requests stream text into it, the one with `Final` closes it. Let `tts.NewSynthesis(id)` do the bookkeeping — `AddText`, `Chunk`, `Complete` — so characters, audio duration and time-to-first-byte are counted the same way everywhere.
- Emit through `tts.NewEmitter(64)`. Every utterance must settle exactly once: `Interrupt`, a dead connection and `Close` all owe a `SynthesisComplete`, interrupted, or the stats lose work.
- Options struct with an `APIKey` that falls back to the provider's env var. Validate in `New` and return a descriptive error rather than failing at connect time.
- Register it in [ttsrouter/registry.go](../../../acceleration/internal/ttsrouter/registry.go) so routing config can name it.

## Test

Three files, in this order of value:

1. `<name>_test.go` — no network. Feed `handleMessage` (or its equivalent) server frames and assert the events that come out.
2. `socket_test.go` — a fake WebSocket or HTTP server. Asserts what goes on the wire (setup frame, flush on final, close).
3. `integration_test.go` — `//go:build integration`, embeds `ttssuite.Suite`:

```go
suite.Run(t, &XIntegrationSuite{Suite: ttssuite.Suite{
    New:                func() ttssuite.Provider { p, err := New(Options{}); require.NoError(t, err); return p },
    Requires:           []string{"X_API_KEY"},
    Interruptible:      true,   // set unless the provider generates the whole utterance regardless
    MaxTimeToFirstByte: 2_000,  // ms; omit when latency is not what this provider is kept for
    Timeout:            time.Minute, // raise for a deployment that scales to zero
}})
```

That inherits real speech (format, chunking, billed characters, duration, TTFB), deltas becoming one utterance, and barge-in. Only add provider-specific tests on top, built on `s.Started`, `s.Say` and `s.Hangup`; `.env` is loaded for you. A test needing its own options builds the provider itself and passes it to `s.Start`.

## Run

```bash
cd acceleration
go test ./internal/tts/...                                              # unit + socket
go test -tags integration -run TestXIntegrationSuite ./internal/tts/x    # live
```
