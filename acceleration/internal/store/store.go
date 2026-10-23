// Package store persists request rows in Postgres and aggregates them into hourly and
// daily rollups. Uptime, latency percentiles, billable usage and cost all come from the
// same request rows, so there is no separate health-probe pipeline to keep in sync.
//
// Rows from every modality share one table, distinguished by a modality column, because
// every question worth asking about usage is asked the same way whatever the modality is.
package store

import (
	"cmp"
	"context"
	"crypto/rand"
	"database/sql"
	"database/sql/driver"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/pressly/goose/v3"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
	"github.com/uptrace/bun/extra/bunotel"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/migrations"
)

// maxConns is how many Postgres connections one router holds at most.
const maxConns = 40

// Store reads and writes router statistics.
type Store struct {
	// db is the database every shard shares, and shard this one's, which is db for a
	// deployment of one shard. peers are the other shards' (see shards.go).
	db    *bun.DB
	shard *bun.DB
	peers []*bun.DB
	// shardNames names each of shards(), as Options.Shards does, for what is kept per shard
	// in the shared database.
	shardNames []string
	LogDrops   atomic.Int64
	// shapes is what each table an export carries looks like, read from the catalogue
	// once rather than kept in a list here that a migration could leave behind.
	shapes tableShapes
	// pins says how Stream app pins cross between deployments, once the router has said.
	pins pinsHolder
	// deliveriesPruned is when hook deliveries were last forgotten, in Unix nanoseconds.
	deliveriesPruned atomic.Int64
}

// Open connects to Postgres using a pgdriver DSN, for example
// postgres://user:pass@localhost:5432/router?sslmode=disable.
func Open(dsn string) (*Store, error) {
	return OpenShards(Options{DSN: dsn})
}

// connect opens a pool of at most conns connections.
func connect(dsn string, conns int) *bun.DB {
	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(dsn)))
	// database/sql keeps two idle connections by default, so every burst of writes past two
	// dials Postgres and shakes hands over TLS again, and under load those handshakes are
	// what times out. The pool is kept instead, and bounded so that two pods, each running
	// overlapping generations, stay well inside the server's connection limit.
	sqldb.SetMaxOpenConns(conns)
	sqldb.SetMaxIdleConns(conns)
	sqldb.SetConnMaxIdleTime(5 * time.Minute)
	db := bun.NewDB(sqldb, pgdialect.New())
	// A span per query, so a request's trace says which reads it waited on. Queries are
	// recorded unformatted: the arguments of these are customer ids and api keys.
	db.AddQueryHook(bunotel.NewQueryHook(bunotel.WithDBName("router")))
	return db
}

// DB exposes the shared database's bun handle so callers can run queries this store does
// not wrap.
func (s *Store) DB() *bun.DB { return s.db }

// ShardDB is this shard's bun handle, where its calls, sessions and requests are. It is
// DB when the deployment has one database.
func (s *Store) ShardDB() *bun.DB { return s.shard }

// Close releases the connection pools.
func (s *Store) Close() error {
	return errors.Join(s.closeShards(), s.db.Close())
}

// Ping verifies the shared database and this shard's are usable. Another shard being down
// is not this router failing: it only leaves that shard's calls off a page.
func (s *Store) Ping(ctx context.Context) error {
	if err := s.db.PingContext(ctx); err != nil {
		return stack.Wrap(err)
	}
	return stack.Wrap(s.shard.PingContext(ctx))
}

// Migrate applies every pending migration, to the shared database and to this shard's.
func (s *Store) Migrate(ctx context.Context) error {
	if err := s.migrate(ctx, s.db); err != nil {
		return err
	}
	return s.migrateShard(ctx)
}

func (s *Store) migrate(ctx context.Context, db *bun.DB) error {
	// Every shard's routers start against the shared database at once in a rollout, and
	// two goose runs on one database collide, so one migrates while the others wait.
	conn, err := db.DB.Conn(ctx)
	if err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	locked := false
	defer func() {
		if !locked {
			_ = conn.Close()
			return
		}
		// Released by hand: closing hands the connection back to the pool, session and lock
		// intact. A failed unlock ends the session instead, which releases it.
		if _, err := conn.ExecContext(context.WithoutCancel(ctx), "SELECT pg_advisory_unlock(hashtextextended('router.migrate', 0))"); err != nil {
			_ = conn.Raw(func(any) error { return driver.ErrBadConn })
		}
		_ = conn.Close()
	}()
	if err := lockMigrations(ctx, conn); err != nil {
		return err
	}
	locked = true

	goose.SetBaseFS(migrations.FS)
	if err := goose.SetDialect("postgres"); err != nil {
		return fmt.Errorf("store: set dialect: %w", err)
	}
	if err := goose.UpContext(ctx, db.DB, "."); err != nil {
		return fmt.Errorf("store: migrate: %w", err)
	}
	if err := ensureUsageIndexes(ctx, conn); err != nil {
		return err
	}
	// Attached here rather than in a migration of their own, so that a table added later
	// is covered by the deployment reaching it rather than by somebody remembering.
	return s.watchDataChanges(ctx, db)
}

// migrateLockFirst and migrateLockMost are the pause between two tries at the migrate lock:
// the first, doubled after each miss up to the most, which bounds how long a router idles
// after the one migrating is done.
const (
	migrateLockFirst = 20 * time.Millisecond
	migrateLockMost  = 500 * time.Millisecond
)

// lockMigrations takes the migrate lock on conn, trying it until it is free. A router waiting
// in a blocking pg_advisory_lock holds a snapshot all the while, and CREATE INDEX CONCURRENTLY
// in the migration the holder runs waits for every older snapshot to end, so the two would
// wait on each other forever. Between two tries the waiter holds none.
func lockMigrations(ctx context.Context, conn *sql.Conn) error {
	pause := migrateLockFirst
	for {
		var locked bool
		if err := conn.QueryRowContext(ctx, "SELECT pg_try_advisory_lock(hashtextextended('router.migrate', 0))").Scan(&locked); err != nil {
			// The lock may have been granted with the answer lost; dropping the connection
			// ends the session, which releases it.
			_ = conn.Raw(func(any) error { return driver.ErrBadConn })
			return fmt.Errorf("store: migrate: lock: %w", err)
		}
		if locked {
			return nil
		}
		timer := time.NewTimer(pause)
		select {
		case <-ctx.Done():
			timer.Stop()
			return fmt.Errorf("store: migrate: wait for lock: %w", context.Cause(ctx))
		case <-timer.C:
		}
		pause = min(2*pause, migrateLockMost)
	}
}

// usageIndexes are built CONCURRENTLY, so building them stops no write. A build that fails
// leaves an invalid index behind, which the migration's IF NOT EXISTS then skips once goose
// has recorded it, and one dropped by hand is never built again by the migration. Writing
// usage needs requests_usage_id_idx: ON CONFLICT (usage_id) fails without it.
var usageIndexes = []struct{ name, create string }{
	{"requests_usage_id_idx", "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS requests_usage_id_idx ON requests (usage_id)"},
	{"requests_to_ship_idx", "CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_to_ship_idx ON requests (started_at) WHERE to_ship"},
	// What a subject spent, summed with no usage service (20261112120000_limits.sql).
	{"requests_key_spend_idx", "CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_key_spend_idx ON requests (customer_id, api_key, started_at) " +
		"INCLUDE (cost_micros) WHERE billable"},
	{"requests_user_spend_idx", "CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_user_spend_idx ON requests (customer_id, user_id, started_at) " +
		"INCLUDE (cost_micros) WHERE billable AND user_id <> ''"},
	{"requests_category_spend_idx", "CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_category_spend_idx ON requests (customer_id, started_at) " +
		"INCLUDE (cost_micros, usage_categories) WHERE billable AND usage_categories <> '{}'"},
	// The rows naming a grant, which a reconcile counts again (20261112121000_requests_limit_grants.sql).
	{"requests_limit_grants_started_idx", "CREATE INDEX CONCURRENTLY IF NOT EXISTS requests_limit_grants_started_idx ON requests (started_at, id) " +
		"WHERE limit_grants <> '{}'"},
}

// ensureUsageIndexes builds each usage index that is missing, dropping one a failed build left
// invalid first. A build that fails again fails Migrate, so the router does not start and fail
// every write; its next start tries again.
func ensureUsageIndexes(ctx context.Context, conn *sql.Conn) error {
	for _, index := range usageIndexes {
		valid, found, err := indexState(ctx, conn, index.name)
		if err != nil {
			return err
		}
		if found && valid {
			continue
		}
		if found {
			if _, err := conn.ExecContext(ctx, "DROP INDEX CONCURRENTLY IF EXISTS "+index.name); err != nil {
				return fmt.Errorf("store: migrate: drop invalid %s: %w", index.name, err)
			}
		}
		if _, err := conn.ExecContext(ctx, index.create); err != nil {
			return fmt.Errorf("store: migrate: build %s: %w", index.name, err)
		}
		if valid, found, err = indexState(ctx, conn, index.name); err != nil {
			return err
		}
		if !found || !valid {
			return fmt.Errorf("store: migrate: %s did not build", index.name)
		}
	}
	return nil
}

// indexState reports whether the index named is on the search path, and if it is, whether
// it is valid.
func indexState(ctx context.Context, conn *sql.Conn, name string) (valid, found bool, err error) {
	var state sql.NullBool
	err = conn.QueryRowContext(ctx, "SELECT (SELECT indisvalid FROM pg_index WHERE indexrelid = to_regclass($1))", name).Scan(&state)
	if err != nil {
		return false, false, fmt.Errorf("store: migrate: check %s: %w", name, err)
	}
	return state.Bool, state.Valid, nil
}

// MigrationDrift compares the migrations this binary carries with the ones the database applied,
// and writes nothing: unlike goose, which creates its version table when it is missing, it only
// reads. pending are the carried versions not applied, oldest first; a database goose never
// touched has every version pending. unknown are the applied versions this binary does not
// carry, oldest first: a newer build migrated the database. Version 0 is goose's own first row,
// not a migration. An applied version is the newest goose_db_version row of that version with
// is_applied set, as goose reads its own table (Provider.ListMigrations, v3).
func (s *Store) MigrationDrift(ctx context.Context) (pending, unknown []int64, err error) {
	carried, err := fs.Glob(migrations.FS, "*.sql")
	if err != nil {
		return nil, nil, stack.Wrap(err)
	}
	var exists bool
	if err := s.db.QueryRowContext(ctx, "SELECT to_regclass('goose_db_version') IS NOT NULL").Scan(&exists); err != nil {
		return nil, nil, stack.Wrap(fmt.Errorf("store: migration drift: %w", err))
	}
	applied := map[int64]bool{}
	if exists {
		rows, err := s.db.QueryContext(ctx, "SELECT version_id, is_applied FROM goose_db_version ORDER BY id DESC")
		if err != nil {
			return nil, nil, stack.Wrap(fmt.Errorf("store: migration drift: %w", err))
		}
		defer rows.Close()
		seen := map[int64]bool{}
		for rows.Next() {
			var version int64
			var isApplied bool
			if err := rows.Scan(&version, &isApplied); err != nil {
				return nil, nil, stack.Wrap(fmt.Errorf("store: migration drift: %w", err))
			}
			if !seen[version] {
				seen[version], applied[version] = true, isApplied
			}
		}
		if err := rows.Err(); err != nil {
			return nil, nil, stack.Wrap(fmt.Errorf("store: migration drift: %w", err))
		}
	}
	carries := map[int64]bool{}
	for _, name := range carried {
		prefix, _, _ := strings.Cut(name, "_")
		version, err := strconv.ParseInt(prefix, 10, 64)
		if err != nil {
			return nil, nil, stack.Wrap(fmt.Errorf("store: migration %s has no version: %w", name, err))
		}
		carries[version] = true
		if !applied[version] {
			pending = append(pending, version)
		}
	}
	for _, version := range slices.Sorted(maps.Keys(applied)) {
		if applied[version] && version != 0 && !carries[version] {
			unknown = append(unknown, version)
		}
	}
	return pending, unknown, nil
}

// ErrInvalidRequest is a request that can never be written as it is, because it lacks
// something every row names.
var ErrInvalidRequest = errors.New("store: invalid request")

// ErrRequestLogsMissing is RecordRequests writing its rows and failing to write the agent log
// rows of some of them: the usage is recorded, and a timeline has a gap.
var ErrRequestLogsMissing = errors.New("store: request log rows missing")

// RecordRequest stores one request, giving it a usage id when it has none. Latency is
// optional because a request that failed before reaching the provider has none.
func (s *Store) RecordRequest(ctx context.Context, request *Request) error {
	if request.UsageID == uuid.Nil {
		request.UsageID = uuid.Must(uuid.NewV7())
	}
	requests := []Request{*request}
	_, err := s.RecordRequests(ctx, requests)
	*request = requests[0]
	return err
}

// RecordRequests writes rows that are not written yet, in one transaction in this shard's
// database, and returns the usage ids it wrote. A row whose usage id is already there is
// skipped, so writing a batch again after a failure whose commit did land writes nothing twice.
// A row RecordRequests wrote has its ID set. With one database each row it wrote counts toward
// the grants it names in the same transaction (AddGrantUses); with shards that is the caller's
// to do, apart, since the grants are in the shared database.
func (s *Store) RecordRequests(ctx context.Context, requests []Request) ([]uuid.UUID, error) {
	for i := range requests {
		if err := prepareRequest(&requests[i]); err != nil {
			return nil, err
		}
	}
	if len(requests) == 0 {
		return nil, nil
	}
	// The agent log trigger takes a lock per customer for each row it logs, so two routers
	// writing batches that name the same customers in another order would deadlock. Written in
	// customer order, every batch takes them in one order.
	sorted := slices.Clone(requests)
	slices.SortStableFunc(sorted, func(a, b Request) int { return strings.Compare(a.CustomerID, b.CustomerID) })

	var written []writtenRequest
	err := s.shard.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if s.Sharded() {
			if _, err := tx.ExecContext(ctx, "SELECT set_config('router.request_logged', 'on', true)"); err != nil {
				return err
			}
		}
		// Scanned apart from the models: a row skipped by the conflict returns nothing, so the
		// rows that come back do not line up with the ones sent.
		_, err := tx.NewInsert().Model(&sorted).
			On("CONFLICT (usage_id) DO NOTHING").
			Returning("id, usage_id").
			Exec(ctx, &written)
		if err != nil || s.Sharded() {
			return err
		}
		// One database: the rows count toward their grants in the transaction that writes them.
		_, err = addGrantUses(ctx, tx, GrantUsesOf(sorted, usageIDsOf(written)))
		return err
	})
	if err != nil {
		if IDTaken(err) {
			if advanced := s.advanceRequestIDs(ctx); advanced != nil {
				err = errors.Join(err, advanced)
			}
		}
		return nil, stack.Wrap(fmt.Errorf("store: record requests: %w", err))
	}

	ids := usageIDsOf(written)
	byUsage := make(map[uuid.UUID]int64, len(written))
	for _, row := range written {
		byUsage[row.UsageID] = row.ID
	}
	var missing error
	var logged []*Request
	for i := range requests {
		id, ok := byUsage[requests[i].UsageID]
		if !ok {
			continue
		}
		// Only the first of two rows sent with one usage id was written.
		delete(byUsage, requests[i].UsageID)
		requests[i].ID = id
		if s.Sharded() && requests[i].AgentID != "" {
			logged = append(logged, &requests[i])
		}
	}
	if err := s.logShardRequests(ctx, logged); err != nil {
		missing = errors.Join(missing, fmt.Errorf("%w: %w", ErrRequestLogsMissing, err))
	}
	if missing != nil {
		return ids, stack.Wrap(missing)
	}
	return ids, nil
}

func usageIDsOf(written []writtenRequest) []uuid.UUID {
	ids := make([]uuid.UUID, len(written))
	for i, row := range written {
		ids[i] = row.UsageID
	}
	return ids
}

// advanceRequestIDs moves the requests id sequence past the highest id written: a row carried
// in with its own id, as a data move from before usage ids did, leaves the sequence behind it,
// and every id it would still hand out is taken. It never sets the sequence below where the
// statement read it, but under a race it can still move it back: a writer drawing ids between
// that read and the setval may draw past the highest id read, and the ids after it are then
// handed out again. That costs whoever draws one another try, one of the writer's immediate
// retries, never a row.
func (s *Store) advanceRequestIDs(ctx context.Context) error {
	_, err := s.shard.ExecContext(ctx, `
SELECT setval(pg_get_serial_sequence('requests', 'id'), written)
FROM (SELECT max(id) AS written FROM requests) AS highest
WHERE written > COALESCE(pg_sequence_last_value(pg_get_serial_sequence('requests', 'id')::regclass), 0)`)
	if err != nil {
		return fmt.Errorf("store: advance request ids: %w", err)
	}
	return nil
}

// writtenRequest is what RecordRequests reads back of a row it wrote.
type writtenRequest struct {
	ID      int64     `bun:"id"`
	UsageID uuid.UUID `bun:"usage_id"`
}

// prepareRequest checks a request before it is written and fills what defaults.
func prepareRequest(request *Request) error {
	request.ErrorMessage = SafeLogText(request.ErrorMessage)
	if request.CustomerID == "" {
		return stack.Wrap(fmt.Errorf("%w: customer id is required", ErrInvalidRequest))
	}
	if request.Modality == "" {
		return stack.Wrap(fmt.Errorf("%w: modality is required", ErrInvalidRequest))
	}
	if request.UsageID == uuid.Nil {
		return stack.Wrap(fmt.Errorf("%w: usage id is required", ErrInvalidRequest))
	}
	if request.StartedAt.IsZero() {
		request.StartedAt = time.Now().UTC()
	}
	return nil
}

// requestsPrimaryKey is named for the table's first name: 20260815090000_multi_modality.sql
// renamed the table and not its key.
const requestsPrimaryKey = "stt_requests_pkey"

// Unwritable reports whether writing the same rows again can only fail the same way: a row
// is invalid, or Postgres refused its data, a constraint or its size. Anything else, a
// timeout, a lost connection, a lock, may pass on a later try. So may an id already taken
// (IDTaken): the next try draws new ids.
func Unwritable(err error) bool {
	if errors.Is(err, ErrInvalidRequest) {
		return true
	}
	var refused pgdriver.Error
	if !errors.As(err, &refused) {
		return false
	}
	if IDTaken(err) {
		return false
	}
	code := refused.Field('C')
	return strings.HasPrefix(code, "22") || strings.HasPrefix(code, "23") || strings.HasPrefix(code, "54")
}

// IDTaken reports whether RecordRequests failed on a request id another row already has,
// which a row carried in with its own id leaves behind. RecordRequests has moved the sequence
// past it by then, so trying again at once writes the rows.
func IDTaken(err error) bool {
	var refused pgdriver.Error
	return errors.As(err, &refused) && refused.Field('C') == "23505" && refused.Field('n') == requestsPrimaryKey
}

// logShardRequests writes written requests' log rows, which a trigger writes beside a request
// in one database, to the shared one with the rest of the logs: the calls they happened in read
// in one query, and the rows written in one statement under every customer's agent log lock,
// taken in customer order as every batch takes them. The two writes are not one transaction:
// a log row missing is a gap in a timeline, where holding the shared database's lock for the
// shard's insert would slow every request of the customer.
func (s *Store) logShardRequests(ctx context.Context, requests []*Request) error {
	if len(requests) == 0 {
		return nil
	}
	calls, err := s.callsOf(ctx, requests)
	if err != nil {
		return err
	}

	entries := make([]AgentLog, len(requests))
	customers := make([]string, 0, len(requests))
	for i, request := range requests {
		outcome, severity := "completed", "info"
		switch {
		case request.ErrorCode == ErrorCancelled:
			// A caller closing its own request is routine, a superseded flow decision or a
			// barge-in, and no failure, as the rollups count it.
			outcome = "cancelled"
		case !request.Success:
			outcome, severity = "failed", "error"
		}
		entries[i] = AgentLog{
			CustomerID: request.CustomerID, ConfigID: calls[i].ConfigID, AgentID: request.AgentID, SessionID: calls[i].ID,
			Source: "system", Severity: severity, EventType: "provider_request",
			Message:    request.Modality + " request " + outcome,
			OccurredAt: request.StartedAt, IngestedAt: time.Now().UTC(),
			Details: map[string]any{
				"provider": request.Provider, "model": request.Model, "request_id": strconv.FormatInt(request.ID, 10),
				"duration_ms": request.LatencyMs, "input_tokens": request.InputTokens, "output_tokens": request.OutputTokens,
				"cost_micros": request.CostMicros, "error_code": nullIfEmpty(request.ErrorCode),
				"error_message": nullIfEmpty(request.ErrorMessage),
			},
		}
		customers = append(customers, request.CustomerID)
	}
	slices.Sort(customers)
	customers = slices.Compact(customers)
	return s.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		// unnest reads the array in order, so the locks are taken in it.
		if _, err := tx.ExecContext(ctx,
			"SELECT pg_advisory_xact_lock(hashtextextended(customer, 731)) FROM unnest(?::text[]) AS customer",
			pgdialect.Array(customers)); err != nil {
			return err
		}
		_, err := tx.NewInsert().Model(&entries).Exec(ctx)
		return err
	})
}

// callsOf is the call each request happened in, in their order, as the agent log trigger
// finds it: the agent's latest call of the customer started by the request, the one it names
// when it names one. A request in no call has an empty one.
func (s *Store) callsOf(ctx context.Context, requests []*Request) ([]Call, error) {
	customers := make([]string, len(requests))
	agents := make([]string, len(requests))
	callIDs := make([]string, len(requests))
	started := make([]string, len(requests))
	for i, request := range requests {
		customers[i], agents[i], callIDs[i] = request.CustomerID, request.AgentID, request.CallID
		started[i] = request.StartedAt.UTC().Format(time.RFC3339Nano)
	}
	var found []struct {
		Ordinal  int    `bun:"ordinal"`
		ConfigID string `bun:"config_id"`
		ID       string `bun:"id"`
	}
	err := s.shard.NewRaw(`
SELECT r.ordinal, c.config_id, c.id
FROM unnest(?::text[], ?::text[], ?::text[], ?::timestamptz[]) WITH ORDINALITY
    AS r (customer_id, agent_id, call_id, started_at, ordinal)
CROSS JOIN LATERAL (
    SELECT config_id, id FROM calls
    WHERE calls.customer_id = r.customer_id AND calls.agent_id = r.agent_id
        AND calls.started_at <= r.started_at AND (r.call_id = '' OR calls.call_id = r.call_id)
    ORDER BY calls.started_at DESC
    LIMIT 1) AS c`,
		pgdialect.Array(customers), pgdialect.Array(agents), pgdialect.Array(callIDs), pgdialect.Array(started),
	).Scan(ctx, &found)
	if err != nil {
		return nil, err
	}
	calls := make([]Call, len(requests))
	for _, call := range found {
		calls[call.Ordinal-1] = Call{ConfigID: call.ConfigID, ID: call.ID}
	}
	return calls, nil
}

func nullIfEmpty(text string) *string {
	if text == "" {
		return nil
	}
	return &text
}

// ErrorCancelled is the error code of a request its caller gave up on, whether before the
// provider answered or by closing the stream it had opened. It is not a success, but it is no
// failure either, so the rollups count it as a request and for what it cost and leave it out
// of errors, uptime and latency.
const ErrorCancelled = "cancelled"

// notCancelled is the SQL condition that a request was not cancelled, on the request row the
// prefix names.
func notCancelled(prefix string) string {
	return prefix + "error_code IS DISTINCT FROM '" + ErrorCancelled + "'"
}

// Rollup aggregates the requests in [from, to) into the rollup tables for the given
// granularity, both the provider breakdown and the cost-tag breakdown, and returns how
// many buckets were written across the two. It is idempotent: re-running it recomputes
// the buckets it touches, so a missed run is fixed by running it again over the same
// window.
func (s *Store) Rollup(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	if !granularity.Valid() {
		return 0, stack.Wrap(fmt.Errorf("store: unknown granularity %q", granularity))
	}
	if !to.After(from) {
		return 0, stack.Wrap(fmt.Errorf("store: rollup window must be non-empty, got %s to %s", from, to))
	}

	providers, err := s.rollupProviders(ctx, granularity, from, to)
	if err != nil {
		return 0, err
	}
	tags, err := s.rollupTags(ctx, granularity, from, to)
	if err != nil {
		return 0, err
	}
	return providers + tags, nil
}

func (s *Store) rollupProviders(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	query := fmt.Sprintf(`
INSERT INTO %[1]s (
    modality, customer_id, provider, model, bucket,
    audio_ms_total, characters_total,
    input_tokens_total, cached_input_tokens_total, output_tokens_total,
    images_total,
    cost_micros_total,
    request_count, error_count,
    latency_p50_ms, latency_p95_ms
)
SELECT
    modality,
    customer_id,
    provider,
    model,
    date_trunc('%[2]s', started_at) AS bucket,
    COALESCE(SUM(audio_ms), 0),
    COALESCE(SUM(characters), 0),
    COALESCE(SUM(input_tokens), 0),
    COALESCE(SUM(cached_input_tokens), 0),
    COALESCE(SUM(output_tokens), 0),
    COALESCE(SUM(images), 0),
    COALESCE(SUM(cost_micros), 0),
    COUNT(*),
    COUNT(*) FILTER (WHERE NOT success AND %[3]s),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE %[3]s),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE %[3]s)
FROM requests
WHERE started_at >= ? AND started_at < ?
GROUP BY modality, customer_id, provider, model, bucket
ON CONFLICT (modality, customer_id, provider, model, bucket) DO UPDATE SET
    audio_ms_total = EXCLUDED.audio_ms_total,
    characters_total = EXCLUDED.characters_total,
    input_tokens_total = EXCLUDED.input_tokens_total,
    cached_input_tokens_total = EXCLUDED.cached_input_tokens_total,
    output_tokens_total = EXCLUDED.output_tokens_total,
    images_total = EXCLUDED.images_total,
    cost_micros_total = EXCLUDED.cost_micros_total,
    request_count = EXCLUDED.request_count,
    error_count = EXCLUDED.error_count,
    latency_p50_ms = EXCLUDED.latency_p50_ms,
    latency_p95_ms = EXCLUDED.latency_p95_ms`,
		granularity.table(), granularity.truncateUnit(), notCancelled(""))

	// Each shard rolls up the requests it recorded, into its own rollups.
	result, err := s.shard.ExecContext(ctx, query, from, to)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: rollup %s: %w", granularity, err))
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: rollup %s: %w", granularity, err))
	}
	return affected, nil
}

// rollupTags aggregates the same window by cost tag. Each request is unrolled into one
// row per label it carries, so a request tagged with a project and an environment counts
// towards both breakdowns.
func (s *Store) rollupTags(ctx context.Context, granularity Granularity, from, to time.Time) (int64, error) {
	query := fmt.Sprintf(`
INSERT INTO %[1]s (
    modality, customer_id, tag_key, tag_value, bucket,
    audio_ms_total, characters_total,
    input_tokens_total, cached_input_tokens_total, output_tokens_total,
    images_total,
    cost_micros_total,
    request_count, error_count,
    latency_p50_ms, latency_p95_ms
)
SELECT
    r.modality,
    r.customer_id,
    tag.key,
    tag.value,
    date_trunc('%[2]s', r.started_at) AS bucket,
    COALESCE(SUM(r.audio_ms), 0),
    COALESCE(SUM(r.characters), 0),
    COALESCE(SUM(r.input_tokens), 0),
    COALESCE(SUM(r.cached_input_tokens), 0),
    COALESCE(SUM(r.output_tokens), 0),
    COALESCE(SUM(r.images), 0),
    COALESCE(SUM(r.cost_micros), 0),
    COUNT(*),
    COUNT(*) FILTER (WHERE NOT r.success AND %[3]s),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.latency_ms) FILTER (WHERE %[3]s),
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY r.latency_ms) FILTER (WHERE %[3]s)
FROM requests AS r
CROSS JOIN LATERAL jsonb_each_text(r.tags) AS tag(key, value)
WHERE r.started_at >= ? AND r.started_at < ?
GROUP BY r.modality, r.customer_id, tag.key, tag.value, bucket
ON CONFLICT (modality, customer_id, tag_key, tag_value, bucket) DO UPDATE SET
    audio_ms_total = EXCLUDED.audio_ms_total,
    characters_total = EXCLUDED.characters_total,
    input_tokens_total = EXCLUDED.input_tokens_total,
    cached_input_tokens_total = EXCLUDED.cached_input_tokens_total,
    output_tokens_total = EXCLUDED.output_tokens_total,
    images_total = EXCLUDED.images_total,
    cost_micros_total = EXCLUDED.cost_micros_total,
    request_count = EXCLUDED.request_count,
    error_count = EXCLUDED.error_count,
    latency_p50_ms = EXCLUDED.latency_p50_ms,
    latency_p95_ms = EXCLUDED.latency_p95_ms`,
		granularity.tagTable(), granularity.truncateUnit(), notCancelled("r."))

	result, err := s.shard.ExecContext(ctx, query, from, to)
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: rollup %s tags: %w", granularity, err))
	}

	affected, err := result.RowsAffected()
	if err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: rollup %s tags: %w", granularity, err))
	}
	return affected, nil
}

// CustomerStats returns the buckets for one customer and modality in [from, to), oldest
// first.
//
// Reads recorded requests directly so recent activity is visible before rollups run.
func (s *Store) CustomerStats(
	ctx context.Context,
	modality, customerID string,
	granularity Granularity,
	from, to time.Time,
	tags map[string]string,
) ([]Bucket, error) {
	if !granularity.Valid() {
		return nil, stack.Wrap(fmt.Errorf("store: unknown granularity %q", granularity))
	}
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	if modality == "" {
		return nil, stack.Wrap(errors.New("store: modality is required"))
	}

	if tags == nil {
		tags = map[string]string{}
	}
	return s.taggedStats(ctx, modality, customerID, granularity, from, to, tags)
}

// taggedStats aggregates raw request rows carrying every one of the given labels.
func (s *Store) taggedStats(
	ctx context.Context,
	modality, customerID string,
	granularity Granularity,
	from, to time.Time,
	tags map[string]string,
) ([]Bucket, error) {
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer stats: encode tag filter: %w", err))
	}

	query := fmt.Sprintf(`
SELECT
    modality,
    customer_id,
    provider,
    model,
    date_trunc('%[1]s', started_at) AS bucket,
    COALESCE(SUM(audio_ms), 0) AS audio_ms_total,
    COALESCE(SUM(characters), 0) AS characters_total,
    COALESCE(SUM(input_tokens), 0) AS input_tokens_total,
    COALESCE(SUM(cached_input_tokens), 0) AS cached_input_tokens_total,
    COALESCE(SUM(output_tokens), 0) AS output_tokens_total,
    COALESCE(SUM(images), 0) AS images_total,
    COALESCE(SUM(cost_micros), 0) AS cost_micros_total,
    COUNT(*) AS request_count,
    COUNT(*) FILTER (WHERE NOT success AND %[2]s) AS error_count,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE %[2]s) AS latency_p50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) FILTER (WHERE %[2]s) AS latency_p95_ms,
    (COUNT(*) - COUNT(*) FILTER (WHERE NOT success AND %[2]s))::double precision
        / NULLIF(COUNT(*), 0) AS uptime
FROM requests
WHERE modality = ? AND customer_id = ?
  AND started_at >= ? AND started_at < ?
  AND tags @> ?::jsonb
GROUP BY modality, customer_id, provider, model, bucket
ORDER BY bucket ASC, provider ASC, model ASC`, granularity.truncateUnit(), notCancelled(""))

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]Bucket, error) {
		var buckets []Bucket
		err := db.NewRaw(query, modality, customerID, from, to, string(filter)).Scan(ctx, &buckets)
		return buckets, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer stats by tag: %w", err))
	}
	return mergeBuckets(lists), nil
}

// CustomerTagStats returns what each value of one tag key cost, oldest bucket first. It
// is the "what drives our spend" query: group by project, or by environment, or by
// whichever label the customer bills on.
func (s *Store) CustomerTagStats(
	ctx context.Context,
	modality, customerID, tagKey string,
	granularity Granularity,
	from, to time.Time,
) ([]TagBucket, error) {
	if !granularity.Valid() {
		return nil, stack.Wrap(fmt.Errorf("store: unknown granularity %q", granularity))
	}
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	if modality == "" {
		return nil, stack.Wrap(errors.New("store: modality is required"))
	}
	if tagKey == "" {
		return nil, stack.Wrap(errors.New("store: tag key is required"))
	}

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]TagBucket, error) {
		var buckets []TagBucket
		err := db.NewSelect().
			Table(granularity.tagTable()).
			Where("modality = ?", modality).
			Where("customer_id = ?", customerID).
			Where("tag_key = ?", tagKey).
			Where("bucket >= ?", from).
			Where("bucket < ?", to).
			Order("bucket ASC", "tag_value ASC").
			Scan(ctx, &buckets)
		return buckets, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer tag stats: %w", err))
	}
	return mergeTagBuckets(lists), nil
}

// spendGroupByModality is the group_by that means "where the money went" rather than
// "what it was spent on", and the only one that is not a cost label key.
const spendGroupByModality = "modality"

// spendOther is the value every group outside the biggest few is summed into.
const spendOther = "other"

// CustomerSpend returns what one customer spent per bucket and group, oldest bucket first.
// It is the whole bill rather than one modality's share of it, which is what a spend trend
// is read as.
//
// groupBy is either "modality" or a cost label key. Only the limit biggest values over the
// whole window keep a group of their own: a label such as customer_id has as many values as
// the customer has customers, and a chart of all of them says nothing. The rest are summed
// into "other", and requests carrying no such label into the empty value, so the rows still
// add up to the total.
//
// Reads the request rows rather than the rollups, so today's spend is there before a
// rollup has run.
func (s *Store) CustomerSpend(
	ctx context.Context,
	customerID, groupBy string,
	granularity Granularity,
	from, to time.Time,
	limit int,
	tags map[string]string,
	who UsageFilter,
) ([]SpendBucket, error) {
	if !granularity.Valid() {
		return nil, stack.Wrap(fmt.Errorf("store: unknown granularity %q", granularity))
	}
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	if limit < 1 {
		return nil, stack.Wrap(errors.New("store: limit must be at least 1"))
	}
	if tags == nil {
		tags = map[string]string{}
	}
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer spend: encode tag filter: %w", err))
	}

	valueExpr := "modality"
	args := []any{}
	if groupBy != spendGroupByModality {
		valueExpr = "COALESCE(tags->>?, '')"
		args = append(args, groupBy)
	}
	args = append(args, customerID, from, to, string(filter))
	narrowed, narrowing := who.where()
	args = append(args, narrowing...)

	// Each shard's groups unfolded, and the biggest picked over all of them: a group too small
	// to keep in one shard may be the biggest once the others are added.
	query := fmt.Sprintf(`
SELECT
    date_trunc('%s', started_at) AS bucket,
    %s AS value,
    COALESCE(SUM(cost_micros), 0) AS cost_micros_total,
    %s AS request_count
FROM requests
WHERE customer_id = ?
  AND started_at >= ? AND started_at < ?
  AND tags @> ?::jsonb%s
GROUP BY 1, 2`, granularity.truncateUnit(), valueExpr, requestCount, narrowed)

	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]spendRow, error) {
		var rows []spendRow
		err := db.NewRaw(query, args...).Scan(ctx, &rows)
		return rows, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer spend: %w", err))
	}
	return foldSpend(lists, limit), nil
}

// UsageFilter narrows usage to who asked for it: the key or origin it was started with, the end
// user it was for, and categories every row carries, a team among them as team:<name>. The zero
// filter narrows nothing.
type UsageFilter struct {
	APIKey     string
	UserID     string
	Categories []string
}

// where is the filter as conditions on requests, each opening with AND, and their arguments.
func (f UsageFilter) where() (string, []any) {
	var conditions strings.Builder
	var args []any
	if f.APIKey != "" {
		conditions.WriteString("\n  AND api_key = ?")
		args = append(args, f.APIKey)
	}
	if f.UserID != "" {
		conditions.WriteString("\n  AND user_id = ?")
		args = append(args, f.UserID)
	}
	if len(f.Categories) > 0 {
		conditions.WriteString("\n  AND usage_categories @> ?")
		args = append(args, pgdialect.Array(f.Categories))
	}
	return conditions.String(), args
}

// tagValueRow is what one value of one cost label key was spent on.
type tagValueRow struct {
	Key             string `bun:"key"`
	Value           string `bun:"value"`
	CostMicrosTotal int64  `bun:"cost_micros_total"`
	RequestCount    int64  `bun:"request_count"`
}

// topTagValues is how many values of a key come back with it. Enough to see what drives the
// key's spend, few enough that a key naming an end customer does not return a database.
const topTagValues = 10

// CustomerTagKeys returns which cost label keys one customer's spend carries, biggest spend
// first, each with its ten largest values.
//
// Cost labels are the customer's own, so nothing here knows in advance whether spend is
// broken down by product, by environment or by the end customer it was incurred for. What
// tells them apart is how many values a key was used with and how much of the traffic
// carries it, which is what this reports.
func (s *Store) CustomerTagKeys(
	ctx context.Context,
	customerID string,
	from, to time.Time,
	tags map[string]string,
) ([]TagKeySummary, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}
	if tags == nil {
		tags = map[string]string{}
	}
	filter, err := json.Marshal(tags)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer tag keys: encode tag filter: %w", err))
	}

	// Coverage is measured against every request in the window, labelled or not, because a
	// key on half the traffic breaks down half the bill and reading it as the whole of it
	// is the mistake this number exists to prevent. Runtime is spend and not requests.
	type shardTags struct {
		rows, total int64
		values      []tagValueRow
	}
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (shardTags, error) {
		var read shardTags
		err := db.NewSelect().
			Table("requests").
			ColumnExpr("COUNT(*)").
			ColumnExpr(requestCount).
			Where("customer_id = ?", customerID).
			Where("started_at >= ?", from).
			Where("started_at < ?", to).
			Where("tags @> ?::jsonb", string(filter)).
			Scan(ctx, &read.rows, &read.total)
		if err != nil || read.rows == 0 {
			return read, err
		}
		err = db.NewRaw(`
SELECT tag.key AS key, tag.value AS value,
       SUM(r.cost_micros) AS cost_micros_total, `+requestCount+` AS request_count
FROM requests AS r
CROSS JOIN LATERAL jsonb_each_text(r.tags) AS tag(key, value)
WHERE r.customer_id = ?
  AND r.started_at >= ? AND r.started_at < ?
  AND r.tags @> ?::jsonb
GROUP BY tag.key, tag.value`, customerID, from, to, string(filter)).Scan(ctx, &read.values)
		return read, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer tag keys: %w", err))
	}

	var rows, total int64
	type valueKey struct{ key, value string }
	var order []valueKey
	values := map[valueKey]*tagValueRow{}
	for _, answer := range answers {
		rows += answer.rows
		total += answer.total
		for _, row := range answer.values {
			k := valueKey{row.Key, row.Value}
			if values[k] == nil {
				values[k] = &tagValueRow{Key: row.Key, Value: row.Value}
				order = append(order, k)
			}
			values[k].CostMicrosTotal += row.CostMicrosTotal
			values[k].RequestCount += row.RequestCount
		}
	}
	if rows == 0 {
		return nil, nil
	}

	byKey := map[string]*TagKeySummary{}
	ranked := map[string][]tagValueRow{}
	var keys []string
	for _, k := range order {
		row := *values[k]
		summary := byKey[row.Key]
		if summary == nil {
			summary = &TagKeySummary{Key: row.Key}
			byKey[row.Key] = summary
			keys = append(keys, row.Key)
		}
		summary.ValueCount++
		summary.CostMicrosTotal += row.CostMicrosTotal
		summary.RequestCount += row.RequestCount
		ranked[row.Key] = append(ranked[row.Key], row)
	}
	slices.SortFunc(keys, func(a, b string) int {
		return cmp.Or(cmp.Compare(byKey[b].CostMicrosTotal, byKey[a].CostMicrosTotal),
			cmp.Compare(byKey[b].RequestCount, byKey[a].RequestCount), strings.Compare(a, b))
	})

	var summaries []TagKeySummary
	for _, name := range keys {
		key := byKey[name]
		if total > 0 {
			key.Coverage = float64(key.RequestCount) / float64(total)
		}
		rows := ranked[name]
		slices.SortFunc(rows, func(a, b tagValueRow) int {
			return cmp.Or(cmp.Compare(b.CostMicrosTotal, a.CostMicrosTotal),
				cmp.Compare(b.RequestCount, a.RequestCount), strings.Compare(a.Value, b.Value))
		})
		for _, row := range rows[:min(topTagValues, len(rows))] {
			// A deployment with no prices configured records every row at zero, so a share of
			// spend would be a division by nothing. Requests are what is left to rank by.
			var share float64
			switch {
			case key.CostMicrosTotal > 0:
				share = float64(row.CostMicrosTotal) / float64(key.CostMicrosTotal)
			case key.RequestCount > 0:
				share = float64(row.RequestCount) / float64(key.RequestCount)
			}
			key.TopValues = append(key.TopValues, TagValueSummary{
				Value:           row.Value,
				CostMicrosTotal: row.CostMicrosTotal,
				RequestCount:    row.RequestCount,
				Share:           share,
			})
		}
		summaries = append(summaries, *key)
	}
	return summaries, nil
}

// CustomerActivity returns how much one customer's agents were used per bucket, oldest
// first, and by how many distinct people.
//
// A bucket with nothing in it is left out rather than returned as zeroes, the same way the
// stats paths leave out a bucket nobody used.
func (s *Store) CustomerActivity(
	ctx context.Context,
	customerID string,
	granularity ActivityGranularity,
	from, to time.Time,
) ([]ActivityBucket, error) {
	if !granularity.Valid() {
		return nil, stack.Wrap(fmt.Errorf("store: unknown activity granularity %q", granularity))
	}
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	unit := granularity.truncateUnit()
	// Who was seen is read apart from the counts: a person is one person however many shards
	// they talked to in a bucket, and a guest who signed up is the account they became, which
	// the shared database knows rather than the shard.
	people := fmt.Sprintf(`
SELECT DISTINCT bucket, user_id FROM (
    SELECT date_trunc('%[1]s', s.created_at) AS bucket, s.user_id, s.caller_kind
    FROM agent_sessions AS s
    WHERE s.customer_id = ? AND s.created_at >= ? AND s.created_at < ?
    UNION ALL
    SELECT date_trunc('%[1]s', a.created_at) AS bucket, s.user_id, s.caller_kind
    FROM agent_responses AS a
    JOIN agent_sessions AS s ON s.id = a.session_id
    WHERE a.customer_id = ? AND a.created_at >= ? AND a.created_at < ?
) AS seen
WHERE user_id <> '' AND caller_kind <> 'anonymous'`, unit)
	query := fmt.Sprintf(`
WITH session_counts AS (
    SELECT date_trunc('%[1]s', created_at) AS bucket, COUNT(*) AS n
    FROM agent_sessions
    WHERE customer_id = ? AND created_at >= ? AND created_at < ?
    GROUP BY bucket
),
message_counts AS (
    SELECT date_trunc('%[1]s', created_at) AS bucket, COUNT(*) AS n
    FROM agent_responses
    WHERE customer_id = ? AND created_at >= ? AND created_at < ?
    GROUP BY bucket
),
call_counts AS (
    SELECT
        date_trunc('%[1]s', started_at) AS bucket,
        COUNT(*) AS n,
        SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, now()) - started_at)))::double precision / 60
            AS voice_minutes,
        COALESCE(SUM(EXTRACT(EPOCH FROM (COALESCE(ended_at, now()) - started_at)))
            FILTER (WHERE from_number IS NOT NULL OR to_number IS NOT NULL), 0)::double precision / 60
            AS phone_minutes
    FROM calls
    WHERE customer_id = ? AND started_at >= ? AND started_at < ?
    GROUP BY bucket
),
buckets AS (
    SELECT bucket FROM session_counts
    UNION SELECT bucket FROM message_counts
    UNION SELECT bucket FROM call_counts
)
SELECT
    b.bucket AS bucket,
    COALESCE(s.n, 0) AS sessions,
    COALESCE(m.n, 0) AS messages,
    COALESCE(c.n, 0) AS calls,
    COALESCE(c.voice_minutes, 0) AS voice_minutes,
    COALESCE(c.phone_minutes, 0) AS phone_minutes
FROM buckets AS b
LEFT JOIN session_counts AS s ON s.bucket = b.bucket
LEFT JOIN message_counts AS m ON m.bucket = b.bucket
LEFT JOIN call_counts AS c ON c.bucket = b.bucket`, unit)

	type seenUser struct {
		Bucket time.Time `bun:"bucket"`
		UserID string    `bun:"user_id"`
	}
	type shardActivity struct {
		counts []ActivityBucket
		seen   []seenUser
	}
	answers, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) (shardActivity, error) {
		var read shardActivity
		if err := db.NewRaw(query,
			customerID, from, to,
			customerID, from, to,
			customerID, from, to,
		).Scan(ctx, &read.counts); err != nil {
			return read, err
		}
		err := db.NewRaw(people, customerID, from, to, customerID, from, to).Scan(ctx, &read.seen)
		return read, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer activity: %w", err))
	}

	var seen []seenUser
	ids := map[string]bool{}
	for _, answer := range answers {
		for _, user := range answer.seen {
			seen = append(seen, user)
			ids[user.UserID] = true
		}
	}
	claimed := map[string]string{}
	if len(ids) > 0 {
		var guests []User
		err := s.db.NewSelect().Model(&guests).Column("id", "claimed_by").
			Where("customer_id = ?", customerID).
			Where("kind = ?", UserKindGuest).
			Where("claimed_by IS NOT NULL AND claimed_by <> ''").
			Where("id IN (?)", bun.In(slices.Collect(maps.Keys(ids)))).
			Scan(ctx)
		if err != nil {
			return nil, stack.Wrap(fmt.Errorf("store: customer activity: claimed guests: %w", err))
		}
		for _, guest := range guests {
			claimed[guest.ID] = guest.ClaimedBy
		}
	}

	byBucket := map[int64]*ActivityBucket{}
	at := func(bucket time.Time) *ActivityBucket {
		key := bucket.Unix()
		if byBucket[key] == nil {
			byBucket[key] = &ActivityBucket{Bucket: bucket}
		}
		return byBucket[key]
	}
	for _, answer := range answers {
		for _, counted := range answer.counts {
			b := at(counted.Bucket)
			b.Sessions += counted.Sessions
			b.Messages += counted.Messages
			b.Calls += counted.Calls
			b.VoiceMinutes += counted.VoiceMinutes
			b.PhoneMinutes += counted.PhoneMinutes
		}
	}
	type person struct {
		bucket int64
		id     string
	}
	counted := map[person]bool{}
	for _, user := range seen {
		id := cmp.Or(claimed[user.UserID], user.UserID)
		if key := (person{user.Bucket.Unix(), id}); !counted[key] {
			counted[key] = true
			at(user.Bucket).ActiveUsers++
		}
	}

	var buckets []ActivityBucket
	for _, key := range slices.Sorted(maps.Keys(byBucket)) {
		buckets = append(buckets, *byBucket[key])
	}
	return buckets, nil
}

// CustomerConcurrencyPeak returns the most sessions one customer had open at the same time
// in a window. A session still open counts up to now, and one that closed at the instant
// another opened did not overlap it.
func (s *Store) CustomerConcurrencyPeak(ctx context.Context, customerID string, from, to time.Time) (ConcurrencyPeak, error) {
	if customerID == "" {
		return ConcurrencyPeak{}, stack.Wrap(errors.New("store: customer id is required"))
	}

	const spans = `
SELECT GREATEST(created_at, ?0) AS opened, LEAST(COALESCE(closed_at, now()), ?1) AS closed
FROM agent_sessions
WHERE customer_id = ?2 AND created_at < ?1 AND (closed_at IS NULL OR closed_at > ?0)`
	if len(s.peers) > 0 {
		return s.peakAcrossShards(ctx, spans, customerID, from, to)
	}

	var peaks []ConcurrencyPeak
	err := s.shard.NewRaw(`
WITH spans AS (`+spans+`
),
edges AS (
    SELECT opened AS at, 1 AS delta FROM spans WHERE closed > opened
    UNION ALL
    SELECT closed AS at, -1 AS delta FROM spans WHERE closed > opened
),
running AS (
    SELECT at, SUM(delta) OVER (ORDER BY at, delta ROWS UNBOUNDED PRECEDING) AS sessions
    FROM edges
)
SELECT sessions, at FROM running ORDER BY sessions DESC, at ASC LIMIT 1`,
		from, to, customerID,
	).Scan(ctx, &peaks)
	if err != nil {
		return ConcurrencyPeak{}, stack.Wrap(fmt.Errorf("store: concurrency peak: %w", err))
	}
	if len(peaks) == 0 || peaks[0].Sessions <= 0 {
		return ConcurrencyPeak{}, nil
	}
	return peaks[0], nil
}

// peakAcrossShards is CustomerConcurrencyPeak over sessions open in several shards at once,
// which no one shard's query can see: it reads every shard's spans and walks their edges here,
// in the order the query does.
func (s *Store) peakAcrossShards(ctx context.Context, spans, customerID string, from, to time.Time) (ConcurrencyPeak, error) {
	type span struct {
		Opened time.Time `bun:"opened"`
		Closed time.Time `bun:"closed"`
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]span, error) {
		var read []span
		err := db.NewRaw(spans, from, to, customerID).Scan(ctx, &read)
		return read, err
	})
	if err != nil {
		return ConcurrencyPeak{}, stack.Wrap(fmt.Errorf("store: concurrency peak: %w", err))
	}
	type edge struct {
		at    time.Time
		delta int64
	}
	var edges []edge
	for _, list := range lists {
		for _, open := range list {
			if open.Closed.After(open.Opened) {
				edges = append(edges, edge{open.Opened, 1}, edge{open.Closed, -1})
			}
		}
	}
	slices.SortFunc(edges, func(a, b edge) int {
		return cmp.Or(a.at.Compare(b.at), cmp.Compare(a.delta, b.delta))
	})
	var peak ConcurrencyPeak
	var running int64
	for _, e := range edges {
		running += e.delta
		if running > peak.Sessions {
			at := e.at
			peak = ConcurrencyPeak{Sessions: running, At: &at}
		}
	}
	return peak, nil
}

// CustomerToolUsage returns how every tool one customer's agents ran in a window was used,
// most called first, and the same figures over all of them. A web search the agent made
// has no result item, so it is not a tool call here.
func (s *Store) CustomerToolUsage(ctx context.Context, customerID string, from, to time.Time) (ToolUsage, []ToolUsage, error) {
	if customerID == "" {
		return ToolUsage{}, nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	type usageRow struct {
		Tool      string   `bun:"tool"`
		Overall   bool     `bun:"overall"`
		Calls     int64    `bun:"calls"`
		Failed    int64    `bun:"failed"`
		AvgTokens float64  `bun:"avg_tokens"`
		AvgMs     *float64 `bun:"avg_ms"`
		P99Ms     *float64 `bun:"p99_ms"`
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]usageRow, error) {
		var rows []usageRow
		err := db.NewRaw(`
WITH ran AS (
    SELECT
        r.tool_name AS tool,
        r.payload->>'error' IS NOT NULL AS failed,
        (octet_length(r.tool_name) + octet_length(COALESCE(r.payload->>'arguments', ''))
            + octet_length(r.text) + ?0 - 1) / ?0 AS tokens,
        EXTRACT(EPOCH FROM (r.at - started.at)) * 1000 AS ms
    FROM agent_responses AS a
    JOIN agent_response_items AS r ON r.response_id = a.id AND r.kind = ?1
    LEFT JOIN LATERAL (
        SELECT c.at FROM agent_response_items AS c
        WHERE c.response_id = r.response_id AND c.kind = ?2 AND c.tool_name = r.tool_name
            AND c.ordinal < r.ordinal
            AND COALESCE(c.payload->>'call_id', '') = COALESCE(r.payload->>'call_id', '')
        ORDER BY c.ordinal DESC LIMIT 1
    ) AS started ON true
    WHERE a.customer_id = ?3 AND a.created_at < ?5 AND r.at >= ?4 AND r.at < ?5
)
SELECT
    COALESCE(tool, '') AS tool,
    GROUPING(tool) = 1 AS overall,
    COUNT(*) AS calls,
    COUNT(*) FILTER (WHERE failed) AS failed,
    COALESCE(AVG(tokens), 0)::double precision AS avg_tokens,
    AVG(ms)::double precision AS avg_ms,
    percentile_cont(0.99) WITHIN GROUP (ORDER BY ms) AS p99_ms
FROM ran
GROUP BY GROUPING SETS ((tool), ())
ORDER BY overall DESC, calls DESC, tool ASC`,
			charsPerToken, ItemToolResult, ItemToolCall, customerID, from, to,
		).Scan(ctx, &rows)
		return rows, err
	})
	if err != nil {
		return ToolUsage{}, nil, stack.Wrap(fmt.Errorf("store: tool usage: %w", err))
	}

	// Averages and the 99th percentile of several shards are each weighted by the calls behind
	// them (see shard_merge.go); with one shard they are its own.
	type usageKey struct {
		tool    string
		overall bool
	}
	var order []usageKey
	sums := map[usageKey]*ToolUsage{}
	tokens, ms, p99s, calls := map[usageKey][]*float64{}, map[usageKey][]*float64{}, map[usageKey][]*float64{}, map[usageKey][]int64{}
	for _, rows := range lists {
		for _, row := range rows {
			k := usageKey{row.Tool, row.Overall}
			if sums[k] == nil {
				sums[k] = &ToolUsage{Tool: row.Tool}
				order = append(order, k)
			}
			sums[k].Calls += row.Calls
			sums[k].Failed += row.Failed
			avgTokens := row.AvgTokens
			tokens[k] = append(tokens[k], &avgTokens)
			ms[k] = append(ms[k], row.AvgMs)
			p99s[k] = append(p99s[k], row.P99Ms)
			calls[k] = append(calls[k], row.Calls)
		}
	}
	var overall ToolUsage
	tools := make([]ToolUsage, 0, len(order))
	for _, k := range order {
		usage := *sums[k]
		if avg := weighted(tokens[k], calls[k]); avg != nil {
			usage.AvgTokens = *avg
		}
		usage.AvgMs, usage.P99Ms = weighted(ms[k], calls[k]), weighted(p99s[k], calls[k])
		if len(lists) == 1 {
			usage.AvgMs, usage.P99Ms = ms[k][0], p99s[k][0]
		}
		if k.overall {
			overall = usage
			continue
		}
		tools = append(tools, usage)
	}
	slices.SortStableFunc(tools, func(a, b ToolUsage) int {
		return cmp.Or(cmp.Compare(b.Calls, a.Calls), strings.Compare(a.Tool, b.Tool))
	})
	return overall, tools, nil
}

// charsPerToken is the rule of thumb llm.Compose estimates tool use with.
const charsPerToken = 4

// ModelRequests returns how many requests each "provider/model" served for a modality
// since a time, across every customer. It is what makes a model popular, so it counts
// calls rather than spend, and is read from the raw requests so it needs no rollup.
func (s *Store) ModelRequests(ctx context.Context, modality string, since time.Time) (map[string]int64, error) {
	if modality == "" {
		return nil, stack.Wrap(errors.New("store: modality is required"))
	}

	type modelRow struct {
		Provider string `bun:"provider"`
		Model    string `bun:"model"`
		Requests int64  `bun:"requests"`
	}
	lists, err := everyShard(ctx, s, func(ctx context.Context, db bun.IDB) ([]modelRow, error) {
		var rows []modelRow
		err := db.NewSelect().
			Table("requests").
			Column("provider", "model").
			ColumnExpr("COUNT(*) AS requests").
			Where("modality = ?", modality).
			Where("started_at >= ?", since).
			Group("provider", "model").
			Scan(ctx, &rows)
		return rows, err
	})
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: model requests: %w", err))
	}

	counts := map[string]int64{}
	for _, rows := range lists {
		for _, row := range rows {
			counts[row.Provider+"/"+row.Model] += row.Requests
		}
	}
	return counts, nil
}

// RecordNumber stores a number a customer now holds.
func (s *Store) RecordNumber(ctx context.Context, number *PhoneNumber) error {
	if !IsE164(number.E164) {
		return stack.Wrap(fmt.Errorf("%w: %q", ErrNotE164, number.E164))
	}
	if number.Vendor == "" {
		return stack.Wrap(errors.New("store: vendor is required"))
	}
	if number.CustomerID == "" {
		return stack.Wrap(errors.New("store: customer id is required"))
	}
	if number.PurchasedAt.IsZero() {
		number.PurchasedAt = time.Now().UTC()
	}
	// A vendor that says nothing about what a number carries is not the same as a null
	// column, and the array is not nullable.
	if number.Capabilities == nil {
		number.Capabilities = []string{}
	}

	if _, err := s.db.NewInsert().Model(number).Exec(ctx); err != nil {
		return stack.Wrap(fmt.Errorf("store: record number: %w", err))
	}
	return nil
}

// ReleaseNumber marks a number as given back. The row stays, because what it cost while
// it was held is still part of that month's bill.
func (s *Store) ReleaseNumber(ctx context.Context, customerID, e164 string, at time.Time) error {
	if customerID == "" || e164 == "" {
		return stack.Wrap(errors.New("store: a customer and a number are required"))
	}
	if at.IsZero() {
		at = time.Now().UTC()
	}

	result, err := s.db.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("released_at = ?", at).
		Set("attached_at = NULL").
		Set("sip_trunk_id = NULL").
		Set("bridge_uri = NULL").
		Set("bridge_username = NULL").
		Set("bridge_password_sealed = NULL").
		Set("bridge_password_kek_version = NULL").
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: release number: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: release number: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("store: %s is not a number %s holds", e164, customerID))
	}
	return nil
}

// AttachNumber records which SIP trunk calls to a number arrive on and which Stream call
// they land in.
//
// The call is recorded as well as the trunk because an inbound call arrives over a webhook
// that names the call, so without it there is nothing to attribute the call to.
func (s *Store) AttachNumber(ctx context.Context, customerID, e164 string, attached NumberAttachment) error {
	if customerID == "" || e164 == "" {
		return stack.Wrap(errors.New("store: a customer and a number are required"))
	}
	if attached.TrunkID == "" {
		return stack.Wrap(errors.New("store: a trunk id is required"))
	}

	result, err := s.db.NewUpdate().Model((*PhoneNumber)(nil)).
		Set("stream_trunk_id = ?", attached.TrunkID).
		Set("stream_route_id = ?", nullable(attached.RouteID)).
		Set("stream_app_pk = ?", nullablePin(attached.StreamAppPK)).
		Set("stream_call_id = ?", attached.CallID).
		Set("stream_call_type = ?", attached.CallType).
		Set("bridge_uri = ?", nullable(attached.BridgeURI)).
		Set("bridge_username = ?", nullable(attached.BridgeUsername)).
		Set("bridge_password_sealed = ?", nullableBytes(attached.BridgePasswordSealed)).
		Set("bridge_password_kek_version = ?", nullableVersion(attached.BridgePasswordKEKVersion)).
		Set("attached_at = now()").
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Exec(ctx)
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: attach number: %w", err))
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: attach number: %w", err))
	}
	if affected == 0 {
		return stack.Wrap(fmt.Errorf("store: %s is not a number %s holds", e164, customerID))
	}
	return nil
}

// NumberAttachment is what attaching a number made in Stream, and where its calls go.
type NumberAttachment struct {
	TrunkID string
	RouteID string
	// StreamAppPK is the app the trunk and route were made in, zero for the deployment's.
	StreamAppPK int64
	CallType    string
	CallID      string
	// BridgeURI and BridgeUsername are the trunk's address, kept whenever a number is attached.
	// The sealed password is empty for a service with no key to seal it under.
	BridgeURI                string
	BridgeUsername           string
	BridgePasswordSealed     []byte
	BridgePasswordKEKVersion int
}

// AttachedBridge is the Stream trunk an attached number's calls go to, with its password
// still sealed.
type AttachedBridge struct {
	CustomerID     string
	URI            string
	Username       string
	PasswordSealed []byte
	KEKVersion     int
}

// ErrNoStoredBridge is AttachedBridge's error for a number that is attached but whose trunk
// credentials were not kept: it was attached before they were, or arrived by a data move,
// which leaves the password behind. Attaching it again keeps them.
var ErrNoStoredBridge = errors.New("store: the number's trunk credentials were not kept")

// AttachedBridge returns the trunk calls to a number of the vendor go to. It matches
// ErrNoNumber when nobody holds the number at that vendor or it is not attached, and
// ErrNoStoredBridge when it was attached without its credentials kept.
func (s *Store) AttachedBridge(ctx context.Context, vendor, e164 string) (AttachedBridge, error) {
	if vendor == "" || e164 == "" {
		return AttachedBridge{}, stack.Wrap(errors.New("store: a vendor and a number are required"))
	}

	var number PhoneNumber
	err := s.db.NewSelect().Model(&number).
		Where("vendor = ?", vendor).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Where("stream_trunk_id IS NOT NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return AttachedBridge{}, stack.Wrap(numberNotHeld{fmt.Sprintf("store: %s is not an attached %s number", e164, vendor)})
	}
	if err != nil {
		return AttachedBridge{}, stack.Wrap(fmt.Errorf("store: attached bridge: %w", err))
	}
	if number.BridgeURI == "" || len(number.BridgePasswordSealed) == 0 {
		return AttachedBridge{}, stack.Wrap(fmt.Errorf("%w: %s", ErrNoStoredBridge, e164))
	}
	return AttachedBridge{
		CustomerID:     number.CustomerID,
		URI:            number.BridgeURI,
		Username:       number.BridgeUsername,
		PasswordSealed: number.BridgePasswordSealed,
		KEKVersion:     number.BridgePasswordKEKVersion,
	}, nil
}

func nullableBytes(value []byte) any {
	if len(value) == 0 {
		return nil
	}
	return value
}

func nullableVersion(version int) any {
	if version == 0 {
		return nil
	}
	return version
}

// nullablePin stores the deployment's own app as NULL, as every pin written before apps
// had identities reads.
func nullablePin(app int64) any {
	if app == 0 {
		return nil
	}
	return app
}

// CustomerNumbers returns the numbers a customer holds, newest first. Released numbers
// are left out unless asked for, since what is normally wanted is what can be called.
func (s *Store) CustomerNumbers(ctx context.Context, customerID string, includeReleased bool) ([]PhoneNumber, error) {
	if customerID == "" {
		return nil, stack.Wrap(errors.New("store: customer id is required"))
	}

	query := s.db.NewSelect().Model((*PhoneNumber)(nil)).
		Where("customer_id = ?", customerID).
		Order("purchased_at DESC")
	if !includeReleased {
		query = query.Where("released_at IS NULL")
	}

	var numbers []PhoneNumber
	if err := query.Scan(ctx, &numbers); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: customer numbers: %w", err))
	}
	return numbers, nil
}

// Number returns one number a customer holds.
func (s *Store) Number(ctx context.Context, customerID, e164 string) (PhoneNumber, error) {
	if customerID == "" || e164 == "" {
		return PhoneNumber{}, stack.Wrap(errors.New("store: a customer and a number are required"))
	}
	// Nobody holds a number written another way, and the vendor may read it as one somebody
	// else holds.
	if !IsE164(e164) {
		return PhoneNumber{}, stack.Wrap(numberNotHeld{fmt.Sprintf("store: %s is not a number %s holds", e164, customerID)})
	}

	var number PhoneNumber
	err := s.db.NewSelect().Model(&number).
		Where("customer_id = ?", customerID).
		Where("e164 = ?", e164).
		Where("released_at IS NULL").
		Limit(1).
		Scan(ctx)
	if errors.Is(err, sql.ErrNoRows) {
		return PhoneNumber{}, stack.Wrap(numberNotHeld{fmt.Sprintf("store: %s is not a number %s holds", e164, customerID)})
	}
	if err != nil {
		return PhoneNumber{}, stack.Wrap(fmt.Errorf("store: number: %w", err))
	}
	return number, nil
}

// newID is the handle a caller holds a row by. It is random rather than sequential because
// it is the only thing standing between two customers who both guessed at an id.
func newID() string {
	raw := make([]byte, 16)
	// rand.Read on crypto/rand never returns an error, which is why the result is not
	// checked: the alternative would be a row that could not be created.
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}

// NewID is newID for callers that have to name something before the row holding it exists,
// as an uploaded file must be given an object key before it can be recorded.
func NewID() string { return newID() }
