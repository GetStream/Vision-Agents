# internal/agent

Builds each reply's `harness.Turn` and keeps the conversation. Keep the reply's prompt cacheable (rules in `internal/harness/AGENTS.md`):

- `prompted()` is the same for the whole call. The hour, a second speaker and anything else that moves on go in `Turn.Tell` (`tellLocked`), never in `prompted()`.
- `a.history` is append-only: keep what `Remember` returns ahead of the reply. Every caller's words go through `harness.Defang`.

Guarded by `TestEachRequestOpensWithTheOneBeforeItAcrossADeliveredAnswer` (`appendonly_test.go`).
