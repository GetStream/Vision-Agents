# internal/connectors/backends/pgsealed

The `core.Backend` that keeps a grant in Postgres: the `connector_connections` row, with `core.Material` sealed by `auth.Sealer` (AES-GCM, KEK keyring) into `material_sealed`. The SQL is the store's (`internal/store/grants.go`: `WithLockedConnectorConnection`, `SaveConnectorConnectionAtRevision`); this package turns the row into a `core.Grant` and back, and owns sealing, the revision and the rewrap. Registering it in `core.Registry.Backends` is T12's (the resolver), not this package's.

## Flow

```
WithLocked(ref, fn)
  1. lock     pg_try_advisory_lock(839, key(customer, id)) on a pool connection, kept if it
              got the lock; otherwise the connection goes back, pause 5ms doubling to 100ms, retry
              until ctx is done
  2. load     the live row, on that connection
  3. open     material_sealed under material_kek_version, AAD = customer, id, revision
              key version not in the keyring -> return auth.ErrKeyVersionUnavailable, write nothing
              does not authenticate -> grant is needs_reauthorization with empty Material, committed now
  4. fn       fn(grant, checkpoint)
  5. checkpoint (any number of times, inside fn)
              seal if Material changed (revision + 1) or the key is old (same revision)
              UPDATE ... WHERE id AND customer AND revision = <last committed> AND deleted_at IS NULL
              detached from ctx (2s), commits at once: the lock is session-level, so it stays held
  6. final    changed -> one more CAS like 5 with fn's grant
              unchanged but material under an older key -> CAS like 5 with the grant as stored
  7. unlock   pg_advisory_unlock, then the connection goes back to the pool
```

Nothing waits inside Postgres at 1, so a caller that gives up just stops trying: no wait stays queued on the server and no connection is held while it pauses. A blocking `pg_advisory_lock` cannot promise that, because pgdriver bounds every read by its `ReadTimeout` (10s by default, `newDefaultConfig` in `uptrace/bun/driver/pgdriver` v1.2.18) or the ctx deadline, drops the socket when it passes, and the server "will detect the loss of the connection only at the next interaction with the socket" ([client_connection_check_interval](https://www.postgresql.org/docs/current/runtime-config-connection.html#GUC-CLIENT-CONNECTION-CHECK-INTERVAL)). The cost of trying is that waiters are not served in order. Writes at 5 and 6 run detached from ctx for the same driver reason: a passed deadline is a passed socket deadline, which would drop tokens a provider already rotated.

## Why session-level here and xact-level in connectors.go

```
connectors.go (T6, definition revisions)      grants.go (T8, this backend)
BEGIN                                          pg_try_advisory_lock        <- held
  pg_advisory_xact_lock    <- held             UPDATE  status = needs_reauthorization   (checkpoint, autocommit)
  SELECT latest, INSERT next                   POST refresh_token to the provider        <- side effect
COMMIT                     <- released         UPDATE  material at revision + 1           (final CAS)
                                               pg_advisory_unlock          <- released
```

T6 does all its work in one transaction, so the lock can end with it. Here the checkpoint has to be visible to every other router before the refresh token leaves, which means it commits while the lock is still needed: a transaction-level lock would be released by that commit. "Once acquired at session level, an advisory lock is held until explicitly released or the session ends", and session-level requests "do not honor transaction semantics" ([explicit-locking, Advisory Locks](https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS)). Because it outlives transactions, it is held on one dedicated `db.Conn` and released in a defer; an unlock that fails closes the connection, which ends the session and the lock.

The key is the two-integer form `(839, first 4 bytes of SHA-256(customer NUL id))`. Postgres keeps the one-bigint and two-integer key spaces apart ("these two key spaces do not overlap", [functions-admin, Advisory Lock Functions](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS)), and every other advisory lock in the repository takes one bigint, so no grant lock can be a T6 or agent-log lock.

The second key is 32 bits because the two-integer form is what buys that separation, and its first half is the namespace. So two connections can share a key, which costs contention, never correctness: the lock only orders callbacks, and each callback writes its own row under its own compare-and-swap. With keys uniform over 2^32 values, a lock taken while k others are held shares a key with one of them with probability about (k-1)/2^32, 2.3e-7 for k = 1000; a table of a million connections has about 116 colliding pairs (n(n-1)/2 / 2^32), and a pair waits only when both are locked at once (`grantLockKey`).

## Rules

- **The backend owns the revision.** New Material (any byte of `Scheme`, `Version`, `Payload` differs) is sealed for revision + 1; anything else keeps the revision. fn changing `Grant.Revision` gets `errRevisionIsTheBackends`. Check: `go test -tags integration -run 'TestPGSealedSuite/(TestNewMaterial|TestTheRevisionIsNot|TestAStatusChange)' ./internal/connectors/backends/pgsealed`.
- **Every write is a compare-and-swap on the revision last committed.** A stale write, or one onto a deleted row, is `store.ErrConnectorConnectionChanged`. Check: `go test -tags integration -run 'TestStoreSuite/(TestASaveAtAStale|TestASaveOntoADeleted|TestAWriteThatSkipped|TestALockedCallbackCannotMove)' ./internal/store`.
- **`SaveConnectorConnectionAtRevision` is new material at the next revision, nothing else.** A checkpoint can commit a status without moving the revision, so a same-revision save from an older snapshot would put back the grant the checkpoint retired. Check: `go test -tags integration -run 'TestStoreSuite/(TestASaveIsNewMaterial|TestASaveCannotPutBack)' ./internal/store`.
- **A commit lands even after the caller's deadline**, within 2s of its own. Check: `go test -tags integration -run TestStoreSuite/TestACommitAfter ./internal/store`.
- **One locked callback per connection at a time, across routers.** Check: `go test -tags integration -run TestStoreSuite/TestLockedCallbacks ./internal/store` and `go test -tags integration -run TestPGSealedSuite/TestConcurrent ./internal/connectors/backends/pgsealed`.
- **Checkpoint before a side effect that cannot be taken back.** A refresh whose outcome is never learned leaves `needs_reauthorization` and the refresh token is never sent again. Check: `go test -tags integration -run TestPGSealedSuite/TestRefreshOutcomeSurvives ./internal/connectors/backends/pgsealed`.
- **A waiter waits as long as its ctx allows, and one that gives up leaves no lock, no queued wait and no borrowed connection.** `TestAWaiterOutlastsTheDriversReadTimeout` takes 12s by design. Check: `go test -tags integration -run TestStoreSuite/TestAWaiter ./internal/store`.
- **Material is sealed for one tenant, connection and revision.** The AAD is `accelerate:connector-material:v1:<len>:<customer>:<len>:<id>:<revision>`; a blob from another row, customer or revision does not open and the grant becomes `needs_reauthorization`, with the blob left for a reconnect to replace. Check: `go test -tags integration -run 'TestPGSealedSuite/TestABlobFrom' ./internal/connectors/backends/pgsealed` and `go test -run TestSealSuite ./internal/connectors/backends/pgsealed`.
- **A key version the keyring lacks is the keyring's fault, not the grant's.** `WithLocked` returns `auth.ErrKeyVersionUnavailable`, runs no fn and writes nothing, so putting the key back restores the connection. Check: `go test -tags integration -run TestPGSealedSuite/TestAKeyVersionMissing ./internal/connectors/backends/pgsealed`.
- **Material under an older key is rewrapped on the next use that returns no error**, at the same revision, and a use that returned changed false has its rewrap written from what was stored, never from fn's edits. Check: `go test -tags integration -run 'TestPGSealedSuite/(TestMaterialSealedUnder|TestAFailedUse|TestARewrapKeeps)' ./internal/connectors/backends/pgsealed`.
- **fn does not lock the same connection again** (`WithLocked`, `SaveConnectorConnectionAtRevision`): that is a second pool connection waiting on the first.
- **Secrets never print.** Material goes into an error or a log only as `core.Material`, whose `String`, `GoString` and `LogValue` redact the payload; tests compare blobs with `bytes.Equal`, not `Equal`, so a failure prints no ciphertext.
- **Every hardcoded value says where it comes from**, beside it.

## Tests

From `acceleration/`, with Postgres on `:55432`:

```bash
go test ./internal/connectors/backends/...                                   # unit: seal, open, AAD
go test -tags integration -count=1 -race ./internal/store ./internal/connectors/backends/...
```

`PGSealedSuite` drops and migrates a database of its own (`testenv.Database(dsn, "pgsealed")`), as `StoreSuite` does. A concurrent caller is a router with a pool of its own (`router()`), or the race never reaches Postgres. The token endpoint and the resolver in the tests are written there, not taken from `fakeprovider` or a scheme: the backend is under test, not refresh.
