# internal/agent

Builds each reply's `harness.Turn` and keeps the conversation. Keep the reply's prompt cacheable (rules in `internal/harness/AGENTS.md`):

- `prompted()` is the same for the whole call. The hour, a second speaker and anything else that moves on go in `Turn.Tell` (`tellLocked`), never in `prompted()`.
- `a.history` is append-only: keep what `Remember` returns ahead of the reply. Every caller's words go through `harness.Defang`.

Guarded by `TestEachRequestOpensWithTheOneBeforeItAcrossADeliveredAnswer` (`appendonly_test.go`).

`Startup` (`startup.go`) times a voice session's start-up in `router_agent_startup_seconds` and `router_agent_startup_failures_total`. Only a session on a call the session manager's `EdgeFactory` opens has one: in the router that is a Stream call (`cmd/router/sessions.go` `edgeFor`). `session.newStartup` gives none to a text session or to a session on an edge its caller hands in as `Spec.Edge`: an audio socket (`internal/api/socketws.go`) or a simulation's loopback (`internal/simulation/audio.go`). Those edges never report a connection coming up, so their stages would not be a call's. Guarded by `TestASessionOnAnEdgeItsCallerHandsInHasNoStartUp` (`internal/session/startup_test.go`).
