---
name: go-testing
description: How Go tests are written in acceleration/ and sdks/go. Read before adding a test or a test suite.
---

# Go testing

## Rules

- Group tests in a testify suite (`suite.Suite`), one per resource or behaviour: `SessionCreateSuite`, `SessionListSuite`.
- Never mock. A provider is a stub registered in its router's registry (`scriptedLLM`, `quietSTT`, `silentEdge` in `internal/api/sessions_test.go`); everything else is real.
- Test behaviour through the outside: HTTP in, status and body out. Assert on outputs and state, never on which method was called.
- Name a test as the sentence it proves: `TestAnIdAnotherSessionHasIsRefused`.
- Set up in `SetupSuite`/`SetupTest`, clean up with `s.T().Cleanup`. No setup helpers called from each test.

## Integration tests

- Anything that needs Postgres or Redis starts with `//go:build integration` and skips when `ROUTER_POSTGRES_DSN` (or `ROUTER_REDIS_ADDR`) is unset.
- Run them with:

```bash
cd acceleration
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable' \
  go test -tags integration -run TestSessionListSuite ./internal/api
```

- Embed `RouterSuite` (`internal/api/router_suite_test.go`). It runs the router against Postgres with real API key auth and hands each test three clients: `anonymous`, `client` (the end user alice) and `backend` (server-side). `s.user("bob")` is another end user.
- Every test is a fresh app, so rows from other tests and earlier runs are never listed. Do not share ids across tests.
- Use `s.createSession`, which closes the session when the test ends.

## Waiting

Rows are written off the request path, so wait for them with `s.Require().Eventually`. Inside the condition use plain Go (`slices.Contains`), never `s.Contains`: an assertion there records a failure on every poll that has not caught up yet.
