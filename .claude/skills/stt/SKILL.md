---
name: stt
description: Build and test a new streaming STT provider in acceleration/internal/stt.
---

# New STT provider (Go)

## Build

- One package per provider: `acceleration/internal/stt/<name>/<name>.go`, implementing `stt.STT` (`Start`, `ProcessAudio`, `Events`, `Close`, `Provider`, `Model`).
- Emit `stt.ModeReplacement` for hypotheses and `stt.ModeFinal` for the settled turn. Never `ModeDelta` unless the server really sends deltas.
- Options struct with an `APIKey` that falls back to the provider's env var. Validate the options in `New` and return a descriptive error rather than failing at connect time.
- Register it in [sttrouter/registry.go](../../../acceleration/internal/sttrouter/registry.go) so routing config can name it.

## Test

Three files, in this order of value:

1. `<name>_test.go` — no network. Feed `handleMessage` (or its equivalent) server frames and assert the events that come out.
2. `socket_test.go` — a fake WebSocket server. Asserts what goes on the wire (setup frame, keyterms, flush on close).
3. `integration_test.go` — `//go:build integration`, embeds `sttsuite.Suite`:

```go
suite.Run(t, &XIntegrationSuite{Suite: sttsuite.Suite{
    New:      func() stt.STT { p, err := New(Options{}); require.NoError(t, err); return p },
    Requires: []string{"X_API_KEY"},
    ChunkMs:  80,             // the pace the server prefers; 100 by default
    SettlesOnClose: true,     // set if ending the audio stream flushes the tail
}})
```

That inherits accuracy (>=90% of the fixture's words), settle time, interim arrival, transcript identity and the tail-of-the-call test. Only add provider-specific tests on top; `.env` is loaded for you. Tune `MaxSettle`, `MinAccuracy` or `SessionTimeout` only when the provider genuinely cannot meet the default, and say why in a comment.

## Run

```bash
cd acceleration
go test ./internal/stt/...                                              # unit + socket
go test -tags integration -run TestXIntegrationSuite ./internal/stt/x    # live
```
