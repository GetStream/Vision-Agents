---
name: go
description: How Go is written in this repo - errors, logging, spans, context, goroutines, comments. Read before writing or changing Go code.
paths:
  - "**/*.go"
---

# Go

These rules are the choices this repo makes. General Go style is not repeated here.

## Rules

- Check the `go` directive of the nearest `go.mod` before you use a newer feature. The 8 modules pin different versions: `acceleration` is on 1.27, `tui` on 1.26, `sdks/go` on 1.25.
- Wrap an error with `%w`. Start the message with the package name and say what was being done, with no "failed to": `oauth2cc: token: %w`. Return the error as is when it already says enough.
- Handle an error once. Never log an error and also return it.
- Log with `log/slog`. Put the data in key-value fields, not in the message: `logger.Warn("could not store an MCP event subscription", "subscription", sub.ID, "error", err)`.
- An HTTP handler returns its error and does not log it. `withRequestLog` (`internal/api/server.go`) logs the error with its stack, and a 5xx at error level (see "HTTP API" in AGENTS.md).
- Name a span `package.verb`, in lowercase: `appconfig.read`, `quota.allow`, `llm.provider.create`.
- Pass `ctx context.Context` as the first parameter. Do not keep a context in a struct that lives longer than the operation, such as a service or a client.
- Start no goroutine in `init()` or in a constructor. Start it in `Start(ctx)` or `Run(ctx)`, as `chatlog.Log.Start` does.
- Return a nil slice for "none". A JSON response is the exception: it sends `[]`, not `null`.
- Defensive code (a retry, a fallback, a lock, a second check) gets a `// Why:` comment that names the concrete threat: concurrent writers, a network failure the caller retries, malformed input. If you cannot name one, do not add it (see "Code style" in AGENTS.md).

## Comments

- A comment says why, once, in the fewest words. Do not narrate what the code does.
- No ticket, PR or plan references in code: they belong in the commit and the PR.
- A link to a public, stable source (an RFC, an engine's documented limits) is fine. The comment still states the fact.

## Other skills

- Tests: the `go-testing` skill (`.claude/skills/go-testing/SKILL.md`).
- Timeouts on outbound calls and streams: the `security` skill (`.claude/skills/security/go.md`, "Abuse limits").
- N+1 queries and batch sizes: the `loop_performance_bench` skill (`.claude/skills/loop_performance_bench/SKILL.md`).
