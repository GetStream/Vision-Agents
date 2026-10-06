# internal/connectors/resolver

The `core.Resolver` the router uses: the one door to a connection's access credential. `Resolve` reads the connection, hands out a cached credential while the row still allows it, and otherwise gets one through the connection's scheme under the credential store's lock, with the checkpoint before any refresh. It moves the connection's status by what the scheme answered. `Invalidate` is the caller's way to say a provider refused the credential. It replaces the prototype's `connectors.ResolveCredentials` (`internal/connectors/runtime.go:26-153` on `codex/connector-support` at `cf62af0d`), which refreshed on the tool call's context. `cmd/router/resolver.go` builds it over `pgsealed`. Request wrapping (T13) and rate limiting (T28) are not here.

## Flow

```
Resolve(ref, req)
  row         store.ConnectorConnection, every call, no lock      deleted -> store.ErrNoConnectorConnection
  fast path   connected, cache[ref].revision == row.revision,     -> cached credential
              younger than maxAge, ExpiresAt after req.Deadline
  slow path   on a goroutine; the caller's ctx ending returns ctx's error at once
    load      definition at the row's definition_revision (outside the lock)
    lock      CredentialStore.Update(ctx, ...)                    ctx bounds only the wait
    status    not connected -> ErrNotConnected
    Retrieve  scheme.Retrieve(detached, ..., opts)                context.WithoutCancel + retrieveTimeout
      ValidUntil  req.Deadline: the scheme renews a credential that expires before it
      Checkpoint  called by the scheme right before a refresh:
                  commit needs_reauthorization + lostRefresh, lock still held
    outcome   nil              -> connected, last_error "", expiry, new credentials (revision + 1)
              invalid_grant    -> needs_reauthorization, rejectedGrant       ErrNotConnected
              scope_required   -> needs_reauthorization, missingScope        ErrNotConnected
              uncertain        -> needs_reauthorization, lostRefresh         ErrNotConnected
              transient, other -> connected, temporarilyUnavailable          ErrTemporarilyUnavailable
              a credential beside the error -> handed out for this call, never cached
    CAS       the credential store's final commit at the revision it loaded
    cache     only after a nil Retrieve, keyed by ref, at the revision the store left in state
    return    the credential, its Revision set to that revision

Invalidate(ref, rejected, why)    why is invalid_grant or scope_required, else refused
  drop the cache entry
  rejected had expired                     -> nothing more
  under the lock, connected and the stored revision still rejected.Revision
                                           -> needs_reauthorization; otherwise nothing more
```

## Cache

- **Key**: the connection, holding the revision of the stored credentials the credential came from. A new revision (a refresh on any router, a reconnect) is a miss.
- **The row is read on every call.** A delete, a disconnect or an `Invalidate` on any router reaches every other router's next `Resolve`, with no window. A soft delete (`store.DeleteConnectorConnection`, #746) sets `deleted_at` and `disconnected` and moves no revision, so a revision check alone would miss it. A maximum age alone would leave a window of that age on every router. An `Invalidate` from the delete handler would reach only the router that served the delete. Example: Alice deletes her connection through router A while her session runs on router B, which cached her token a second earlier. B's next `Resolve` reads no live row and fails, so the deleted connection's token never leaves B again.
- **A status other than connected is never served from the cache.** It goes to the lock, which is where a refresh in flight on another router (status `needs_reauthorization` by its checkpoint) ends: the waiter then sees the final state.
- **maxAge (30 s) bounds what the row does not show**: a credential the scheme would now renew, and stored credentials under an old key that `pgsealed` rewraps on its next use.
- **A credential that expires before `req.Deadline` is not served**: the call goes to the lock, and the scheme renews it for the deadline (`RetrieveOptions.ValidUntil`). The renewed one is cached, so the next call inside the window takes no lock. One exception: a credential this router renewed within `maxAge` is served while it has not expired, even when it expires before the deadline. It is as long as the provider issues them, so renewing again gives one no longer, and asking for that on every call would be a refresh per call. Then a call does start with a credential that dies during it; nothing can give it a longer one. Check: `go test -tags integration -run 'TestResolverSuite/(TestACachedCredentialThatExpiresBefore|TestACredentialThatExpiresBefore|TestACredentialRenewedForADeadline)' ./internal/connectors/resolver`.
- Entries past `maxAge` are dropped at most once per `maxAge`, when a new one is put.

## Rules

- **Retrieve runs detached from the caller.** A caller whose ctx ends during a refresh gets ctx's error at once. The refresh finishes, is committed, and the connection stays `connected` with the rotated credentials, never `needs_reauthorization`. Check: `go test -tags integration -run TestResolverSuite/TestCancellingTheCaller ./internal/connectors/resolver`.
- **The checkpoint is committed before the refresh token leaves**, and a lost answer is never followed by the same token. Check: `go test -tags integration -run 'TestResolverSuite/(TestNeedsReauthorizationIsCommitted|TestALostRefresh)' ./internal/connectors/resolver`, and the scheme side: `go test -run 'TestOAuth2CodeSuite/(TestARefreshCheckpoints|TestARefreshWhoseCheckpoint|TestAccessCredentialThatIsNotDueNever)' ./internal/connectors/schemes/oauth2code`.
- **One refresh across routers.** Check: `go test -tags integration -run TestResolverSuite/TestRoutersResolving ./internal/connectors/resolver`.
- **A deleted connection does not resolve, even from the cache.** Check: `go test -tags integration -run 'TestResolverSuite/(TestADeleted|TestAnotherCustomers)' ./internal/connectors/resolver`.
- **The cache is keyed by revision.** Check: `go test -tags integration -run TestResolverSuite/TestAReconnect ./internal/connectors/resolver`.
- **`Invalidate` writes the status, not only the cache, and only for the grant behind the refused credential.** RFC 6750 §3.1 answers `invalid_token` for a token «expired, revoked, malformed, or invalid for other reasons». A refused credential that had expired, or whose revision the stored credentials have moved past (another router renewed them), leaves the status alone. Example: router B resolves at revision 4; router A renews to 5; the provider refuses B's old token; B's `Invalidate` drops its cache entry and the connection stays `connected`. Check: `go test -tags integration -run TestResolverSuite/TestInvalidate ./internal/connectors/resolver`.
- **The revision is the credential store's.** The resolver reads the committed one from `state` after `Update` (`core.CredentialStore`); it never numbers revisions itself.
- **Outcomes map to statuses as in the flow.** Check: `go test -tags integration -run 'TestResolverSuite/(TestARejected|TestAProvider|TestAPending)' ./internal/connectors/resolver`.
- **Never parse a scheme's error text.** The outcome is read with `errors.As` on `*core.OutcomeError` (`core/AGENTS.md`). Check: `grep -n 'Error()' internal/connectors/resolver/resolver.go` prints nothing.
- **Secrets never print.** Nothing here logs, and the errors it returns wrap the scheme's `*core.OutcomeError`, which carries no secret (`core/contracttest` checks every scheme). Tests read a token only through `fixture.token` and compare two with `==`, not `Equal`, so a failure prints no token. Check: `grep -n 'slog\|log\.' internal/connectors/resolver/resolver.go` prints nothing.
- **Every hardcoded value says where it comes from**, beside it: `retrieveTimeout`, `maxAge`, the `last_error` texts. Status strings are the store's constants.

## Tests

From `acceleration/`, with Postgres on `:55432`:

```bash
go test -tags integration -count=1 -race ./internal/connectors/resolver
go test -tags integration -run x -bench . ./internal/connectors/resolver     # p50 and p95, spike 4
```

`ResolverSuite` drops and migrates a database of its own (`testenv.Database(dsn, "resolver")`). Each router is a pool of its own (`fixture.router`), so a race reaches Postgres. The provider is `fakeprovider` and the scheme is the real `oauth2code`; `fixture.interposed` runs a hook when a refresh reaches the transport, which is how a test reads the row mid-refresh or cancels the caller. `fixture.hold` keeps a connection's lock on another router, which is how a test tells the fast path (answers) from the slow one (waits). Benchmarks print p50 and p95 and assert no target (spike 4). No mocks (`.claude/skills/go-testing/SKILL.md`).
