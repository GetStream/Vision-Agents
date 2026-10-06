# Controller testing

HTTP operations in `internal/api`, tested through the router: HTTP in, status and body out.

## Rules

- Put a test next to the controller it tests: `session_create.go` is tested in `session_create_test.go`.
- One suite per controller: `SessionCreateSuite`, `SessionListSuite`.
- A provider is a stub registered in its router's registry. They all live in `stubs_test.go` (`scriptedLLM`, `quietSTT`, `recordingTTS`, `silentEdge`) and are wired in `RouterSuite.routers`.
- Every test in `internal/api` is an integration test on `RouterSuite`. There are no in-memory server tests: a controller asserted against a router with no database is a controller asserted against a deployment nobody runs.
- Use fresh UUIDs for everything a test makes, and never clean up. Unique ids are what keep tests apart, not deleting rows.

## The router suite

Integration suites in `internal/api` embed `RouterSuite` (`router_suite_test.go`). It runs the whole router against Postgres and Redis with the same API key auth as production: keys are rows, and their secrets are sealed. A suite's `SetupTest` picks the app its clients call:

```go
func (s *SessionCreateSuite) SetupTest() {
	s.useFixture("standard")
}
```

Each test then has:

- One client per sort of caller: `s.unauthenticatedClient` (no credentials), `s.anonymousClient` (the app's key, going by a name nothing proves), `s.guestClient`, `s.client` (a signed-in end user) and `s.serverClient` (the app's own backend).
- `s.serverClient.actingFor(owner)`: the backend naming the user it acts for, with `X-Stream-User-Id`.
- `s.utils.uuid()`: a fresh UUIDv7.
- `s.data.createUser()`, `createGuest()`, `createAnonymous()`: a client for a new caller of that sort.
- `s.data.createApp()`: a new organization and app with a key, for `s.useApp`.
- `s.data.createAppAdmitting(settings)`: an app that turns a level of end user away, for a test about a caller it refuses.
- `s.data.createAgentConfig()`: an agent of the suite's app, for the endpoints that need one to name.
- `s.data.backendOfAnotherApp()`: the server of an app this test has nothing to do with.
- `s.store` and `s.live`: the real Postgres and Redis behind the router, for seeding a row no endpoint writes.

Clients call the API by name and fail the test unless it succeeds: `s.client.createSession(textSession(nil))`, `getSession`, `updateSession`, `stopSession`, `deleteSession`, `querySessions`. Sockets go through `c.opens(path)`, which sends the credentials in the query string as a browser does and in the headers as a backend does; `c.watch(path)` returns the status when the handshake is refused. To assert on a status, use `do`:

```go
func (s *SessionCreateSuite) TestAnIdThatIsNotAUUIDIsRefused() {
	id := "my-session"

	s.Equal(http.StatusBadRequest, s.client.do(http.MethodPost, "/v1/agents/sessions", textSession(&id), nil))
}
```

Add a helper to the client or to `s.data` when a second test needs it, not to each test.

## Security posture

Every endpoint gets a test saying who may call it, and every resource one saying whose it is. The shortcuts are in `security_test.go`:

```go
func (s *SessionCreateSuite) TestAnyoneHoldingTheAppsKeyMayCreateASession() {
	s.assertPosture(anyAppCaller, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/sessions", textSession(nil), nil)
	})
}

func (s *SessionCreateSuite) TestABackendCreatesASessionForTheUserItNames() {
	owner := s.data.createUser()

	s.assertOwnedBy(s.serverClient.actingFor(owner).createSession(textSession(nil)).Id, owner)
}
```

- `assertPosture(admits, call)` makes the call as every sort of caller. The admitted ones must get a 2xx; the rest are refused, unauthenticated with a 401 and everyone else with a 403. The postures are `anyAppCaller` and `serverOnly`; write `posture{guest, user}` for anything else.
- `assertOwnedBy(id, owner)`: the owner reaches the session, and neither another user nor an anonymous caller claiming the owner's name does.
- `assertOwnedByNobody(id)`: the backend reaches it and no end user does.
- `assertHiddenFromOtherApps(call)`: makes the call as another app's backend, which must be answered 404. Anything else, including a 403, tells a stranger the id is real.
- A caller who does not own a session is told it does not exist (404), never that it is forbidden, so the answer does not confirm the id is real.

`posture_test.go` covers the whole surface from the spec: every operation is called by every sort of caller and checked against `x-client-accessible`, so a new endpoint is refused correctly without its own posture test. What a suite still writes is the posture of its own endpoint, because that is the line somebody will move by accident.

## Fixtures

A fixture is data that many tests read, loaded once: `s.useFixture("standard")` loads it for the first test that asks, and every later test in the run gets the same one. `s.requireFixture(name)` returns it when a test needs its data. They live in `fixtureLoaders` in `fixtures_test.go`.

- `standard` is an organization, an app with an API key, and a user of the app. Start here.
- Tests never change a fixture's data. Making rows of their own in its app is fine; renaming the app or deleting its key is not, because every other test shares it.
- Tests sharing a fixture's app see each other's rows. Find your own by a UUID you made, such as a project: `s.client.querySessions(inProject(project))`.
- A test that needs the app empty, like one listing everything in it, calls `s.useApp(s.data.createApp())` instead (see `SessionListSuite`).
- Add a fixture when several tests need the same read-only data. Build it from `s.data`, so it is made the way tests make things.

## Integration tests

- They skip when `ROUTER_POSTGRES_DSN` or `ROUTER_REDIS_ADDR` is unset. Both are required: the quota, the page queue and the live client are real.
- Run them with:

```bash
cd acceleration
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router?sslmode=disable' \
  ROUTER_REDIS_ADDR='localhost:56379' \
  go test -tags integration -run TestSessionCreateSuite ./internal/api
```

- A DSN that does not name a `_test` database is redirected to one, so they run against `model_router_test` whatever the DSN names. Nothing here is ever written into the database a local router is serving. Look in `model_router_test` when checking what a test wrote.

## Running suites in parallel

Integration suites run beside each other. Start every one with `runSuite` rather than `suite.Run`:

```go
func TestSessionCreateSuite(t *testing.T) {
	runSuite(t, new(SessionCreateSuite))
}
```

- Suites run in parallel; the tests inside one run in order, because they share the suite's clients. Ten suites at once is what the setup is sized for: `go test -tags integration -parallel 10 ./internal/api`.
- Each suite has its own router, sessions and connection pool, capped at `suiteConnections` (5), so ten stay inside the hundred connections Postgres allows.
- Migrations run once per run (`migrated`), not once per suite: they set goose's globals and attach triggers.
- Fixtures load once behind a lock, so a suite asking for one another suite is loading waits for it.
- Each suite has its own Redis database, handed out in order, so one suite's keys are not another's.
- Nothing may be shared but Postgres and fixtures. No package-level state a test writes, and no fixed ids: another suite is making rows in the same app at the same time.
- A stub is shared by every test in its suite and keeps what it was asked for. Name your own prompt, state or instructions with a UUID and read back that one, rather than resetting the stub between tests. A stub that has to answer something particular is a provider of its own in `routers`, built fresh per session: `echo` answers with the session's instructions, `tooling` reaches for the caller's tool, `slow` is a while in the writing.
- Run with `-race` when you touch the harness. Every race report must come from the code under test, never from the suite.

## Cost

A session on a call is the expensive thing in these suites: an agent holding a line open is given a moment to finish speaking when it closes, so a suite that opens ten of them spends seconds waiting on them. Open one only for what only a call does — speaking, hanging up, a turn left waiting mid-call. Everything else takes an incognito conversation in writing, which records nothing and closes at once.

## Waiting

Rows are written off the request path, so wait for them with `s.Require().Eventually`. Inside the condition use plain Go (`slices.Contains`), never `s.Contains`: an assertion there records a failure on every poll that has not caught up yet.
