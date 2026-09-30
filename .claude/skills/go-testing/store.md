# DB model and state testing

The models and queries in `internal/store`, tested against a real Postgres: rows in, rows and aggregates out.

## Rules

- Put a test next to the file it tests: `sessions.go` is tested in `sessions_test.go`.
- Every integration test is a method on the one `StoreSuite` (`store_test.go`), so the schema is migrated once and every file shares its helpers.
- Assert on what a query returns, never on the SQL it ran. Reach for `s.store.DB()` only to set up a state the API cannot make, such as backdating `updated_at`.
- Pin time to `s.base`, the start of a fixed hour, so hourly and daily buckets fall where the test expects.

## The store suite

`SetupSuite` opens a database of the suite's own (`testenv.Database`), because dropping the schema and truncating tables is not something to do to a database another package's suite is reading while `go test` runs them side by side. It refuses any database whose name does not end in `_test`, drops the schema and runs the embedded migrations, so the migrations are what the tests exercise. `SetupTest` truncates the tables, so each test starts empty and fixed ids such as `"app"` and `"one"` are safe here, unlike in controller tests.

A table a test writes to must be in that `TRUNCATE` list, or rows leak from one test into the next.

## Helpers

A helper makes one row, defaulting the fields a test does not care about, and takes a function to change the ones it does:

```go
func (s *StoreSuite) TestSavingASessionTwiceIsOneConversation() {
	s.opened("one", "app", s.base, func(session *AgentSession) { session.Title = "first" })
	s.opened("one", "app", s.base, func(session *AgentSession) { session.Title = "renamed" })

	found, err := s.store.StoredSession(s.ctx, "app", "one")
	s.Require().NoError(err)
	s.Equal("renamed", found.Title)
}
```

`opened` (sessions, in `sessions_test.go`) and `record` (requests, in `store_test.go`) are the ones to copy.

## Scoping

Every query that takes a customer gets a test that another customer's rows are invisible to it, both fetched by id and listed (`TestASessionIsOnlyItsOwnCustomers`).

## Without Postgres

Logic decided before any SQL runs, such as validation or which query a call picks, goes in `<file>_unit_test.go` with no build tag and plain `require`: see `sessions_unit_test.go`.

## Run

```bash
cd acceleration
go test ./internal/store                    # unit
ROUTER_POSTGRES_DSN='postgres://postgres:postgres@localhost:55432/model_router_test?sslmode=disable' \
  go test -tags integration -run TestStoreSuite ./internal/store
```
