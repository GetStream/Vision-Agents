// Package ledger is the usage service's store: the events producers ship, once per usage id,
// and the running total of every subject in every window, upserted in the transaction that
// records the events and mirrored to Redis after it commits.
package ledger

import (
	"bytes"
	"context"
	"database/sql"
	"embed"
	"fmt"
	"io/fs"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/pressly/goose/v3"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

//go:embed migrations/*.sql
var migrationFiles embed.FS

const (
	// windowChunk bounds one upsert statement; a batch's windows are sent in chunks of it,
	// in key order, so the locks are still taken in one order.
	windowChunk = 1000
	// pruneBatch is how many ended windows one delete removes.
	pruneBatch = 10000
	// Retention is how long after it ends a window is kept. The longest window is a month,
	// so no limit ever reads one that is gone.
	Retention = 40 * 24 * time.Hour
)

// Ledger records usage events and keeps the windows.
type Ledger struct {
	db     *bun.DB
	mirror *Mirror
	now    func() time.Time
}

// Options configure a Ledger.
type Options struct {
	// DSN is the service's own database.
	DSN string
	// Mirror writes the windows to Redis; nil keeps them in Postgres alone.
	Mirror *Mirror
	// Now is the service's clock; nil is time.Now.
	Now func() time.Time
}

// Open connects to the service's database.
func Open(options Options) (*Ledger, error) {
	if options.DSN == "" {
		return nil, fmt.Errorf("ledger: a database is required")
	}
	if options.Now == nil {
		options.Now = time.Now
	}
	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(options.DSN)))
	return &Ledger{db: bun.NewDB(sqldb, pgdialect.New()), mirror: options.Mirror, now: options.Now}, nil
}

// DB is the database, for what the ledger does not wrap.
func (l *Ledger) DB() *bun.DB { return l.db }

// Close releases the database.
func (l *Ledger) Close() error { return l.db.Close() }

// Ping checks the database answers.
func (l *Ledger) Ping(ctx context.Context) error {
	if err := l.db.PingContext(ctx); err != nil {
		return fmt.Errorf("ledger: ping: %w", err)
	}
	return nil
}

// Migrate applies every pending migration.
func (l *Ledger) Migrate(ctx context.Context) error {
	// Every replica starts at once in a rollout, and two goose runs on one database collide,
	// so one migrates while the others wait.
	conn, err := l.db.DB.Conn(ctx)
	if err != nil {
		return fmt.Errorf("ledger: migrate: %w", err)
	}
	defer conn.Close() //nolint:errcheck // the connection goes back to the pool either way
	if _, err := conn.ExecContext(ctx, "SELECT pg_advisory_lock(hashtextextended('usage.migrate', 0))"); err != nil {
		return fmt.Errorf("ledger: migrate: lock: %w", err)
	}
	// Released by hand: closing hands the connection back to the pool, session and lock intact.
	defer conn.ExecContext(context.WithoutCancel(ctx), "SELECT pg_advisory_unlock(hashtextextended('usage.migrate', 0))") //nolint:errcheck // a failed unlock ends with the session

	files, err := fs.Sub(migrationFiles, "migrations")
	if err != nil {
		return fmt.Errorf("ledger: migrate: %w", err)
	}
	// A provider of its own rather than goose's globals, which the router's store sets in
	// the same process when both run in one test binary.
	provider, err := goose.NewProvider(goose.DialectPostgres, l.db.DB, files)
	if err != nil {
		return fmt.Errorf("ledger: migrate: %w", err)
	}
	if _, err := provider.Up(ctx); err != nil {
		return fmt.Errorf("ledger: migrate: %w", err)
	}
	return nil
}

// Result is what one Ingest did.
type Result struct {
	// Accepted are the events this call recorded, as stored.
	Accepted []usage.Event
	// Duplicates counts events already recorded, by an earlier call or earlier in the batch.
	Duplicates int
	// Rejected are the events that failed the contract's check, and are not recorded, each
	// with its index in the events given.
	Rejected []usage.Rejection
	// ReceivedAt is when the accepted events were recorded.
	ReceivedAt time.Time
	// Windows are the totals of every window the accepted events touched, after them.
	Windows []UsageWindow
	// Repaired are the totals of the windows that events recorded by an earlier call count
	// for, read after the commit to write them to the mirror again; none without a mirror.
	Repaired []UsageWindow
	// Mirrored is whether the mirror took every total in Windows and Repaired.
	Mirrored bool
}

// Ack is the result as the producer is answered.
func (r Result) Ack() usage.Ack {
	rejected := r.Rejected
	if rejected == nil {
		rejected = []usage.Rejection{}
	}
	return usage.Ack{Accepted: len(r.Accepted), Duplicates: r.Duplicates, Rejected: rejected, Mirrored: r.Mirrored}
}

// written is what the event insert returns for each row it wrote.
type written struct {
	UsageID    uuid.UUID `bun:"usage_id"`
	ReceivedAt time.Time `bun:"received_at"`
}

// Ingest records the events that pass the contract's check and are not recorded yet, and adds
// exactly those to their windows, in one transaction. Then it writes the windows' totals to
// the mirror, before returning, so an event acked is already in Redis unless Redis failed.
// The windows of an event recorded by an earlier call are written too: that call may have
// committed and then died before its own mirror write, and the producer is sending it again.
func (l *Ledger) Ingest(ctx context.Context, events []usage.Event) (Result, error) {
	var result Result
	now := l.now()
	seen := make(map[uuid.UUID]bool, len(events))
	rows := make([]UsageEvent, 0, len(events))
	for i, event := range events {
		if rejection := event.Check(now); rejection != nil {
			rejection.Index = i
			result.Rejected = append(result.Rejected, *rejection)
			continue
		}
		if seen[event.UsageID] {
			result.Duplicates++
			continue
		}
		seen[event.UsageID] = true
		rows = append(rows, eventRow(event.Normal()))
	}
	if len(rows) == 0 {
		result.Mirrored = true
		return result, nil
	}
	// Two batches holding the same ids, sent at once by two replicas of a producer, would
	// each wait on the other's insert if they took them in different orders.
	slices.SortFunc(rows, func(a, b UsageEvent) int { return bytes.Compare(a.UsageID[:], b.UsageID[:]) })

	var recorded []usage.Event
	err := l.db.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		var inserted []written
		if err := tx.NewInsert().Model(&rows).On("CONFLICT (usage_id) DO NOTHING").
			Returning("usage_id, received_at").Scan(ctx, &inserted); err != nil {
			return fmt.Errorf("ledger: insert events: %w", err)
		}
		fresh := make(map[uuid.UUID]bool, len(inserted))
		for _, row := range inserted {
			fresh[row.UsageID] = true
		}
		if len(inserted) > 0 {
			result.ReceivedAt = inserted[0].ReceivedAt
		}
		for _, row := range rows {
			if fresh[row.UsageID] {
				result.Accepted = append(result.Accepted, row.Event())
			} else {
				recorded = append(recorded, row.Event())
			}
		}
		totals, err := upsertWindows(ctx, tx, windowsOf(result.Accepted))
		if err != nil {
			return err
		}
		result.Windows = totals
		return nil
	})
	if err != nil {
		return Result{Rejected: result.Rejected}, err
	}
	result.Duplicates += len(rows) - len(result.Accepted)
	WindowsUpserted.Add(float64(len(result.Windows)))
	if l.mirror == nil {
		return result, nil
	}
	// Read after the commit, so other batches of the organization do not wait on this one's
	// window locks meanwhile: a committed total read later is only larger, and the mirror
	// keeps the larger.
	repaired := true
	if len(recorded) > 0 {
		if result.Repaired, err = storedWindows(ctx, l.db, windowsOf(recorded)); err != nil {
			repaired = false
			l.mirror.logger.Error("could not read the windows of a batch's duplicates to mirror them again", "error", err)
		}
	}
	result.Mirrored = l.mirror.Write(ctx, append(slices.Clone(result.Windows), result.Repaired...)) && repaired
	return result, nil
}

// windowsOf is what the events add to each window they count for: every subject, every
// interval, sorted by key.
func windowsOf(events []usage.Event) []UsageWindow {
	sums := map[windowKey]*UsageWindow{}
	for _, event := range events {
		for _, ref := range usage.Subjects(event) {
			for _, interval := range usage.Intervals {
				window := windowOf(ref, interval, event.OccurredAt)
				sum, ok := sums[window.key()]
				if !ok {
					sum = &window
					sums[window.key()] = sum
				}
				sum.Events++
				if event.Billable {
					sum.SpentMicros += event.CostMicros
				}
			}
		}
	}
	windows := make([]UsageWindow, 0, len(sums))
	for _, sum := range sums {
		windows = append(windows, *sum)
	}
	slices.SortFunc(windows, func(a, b UsageWindow) int { return compareKeys(a.key(), b.key()) })
	return windows
}

// upsertWindows adds windows to the table and returns each one's total after it.
func upsertWindows(ctx context.Context, tx bun.Tx, windows []UsageWindow) ([]UsageWindow, error) {
	totals := make([]UsageWindow, 0, len(windows))
	for start := 0; start < len(windows); start += windowChunk {
		chunk := windows[start:min(start+windowChunk, len(windows))]
		var after []UsageWindow
		err := tx.NewInsert().Model(&chunk).
			On("CONFLICT (organization_id, subject, app_id, api_key, user_id, category, interval, window_start) DO UPDATE").
			Set("spent_micros = uw.spent_micros + EXCLUDED.spent_micros").
			Set("events = uw.events + EXCLUDED.events").
			Set("updated_at = now()").
			Returning("*").Scan(ctx, &after)
		if err != nil {
			return nil, fmt.Errorf("ledger: upsert windows: %w", err)
		}
		totals = append(totals, after...)
	}
	return totals, nil
}

// Current reads each subject's current windows, at the ledger's now, a window nobody has
// spent in answered at zero, and writes them to the mirror.
func (l *Ledger) Current(ctx context.Context, refs []usage.SubjectRef) ([]UsageWindow, error) {
	now := l.now()
	wanted := make([]UsageWindow, 0, len(refs)*len(usage.Intervals))
	for _, ref := range refs {
		for _, interval := range usage.Intervals {
			wanted = append(wanted, windowOf(ref, interval, now))
		}
	}
	if len(wanted) == 0 {
		return nil, nil
	}
	found, err := storedWindows(ctx, l.db, wanted)
	if err != nil {
		return nil, err
	}
	stored := make(map[windowKey]UsageWindow, len(found))
	for _, window := range found {
		stored[window.key()] = window
	}
	for i, window := range wanted {
		if row, ok := stored[window.key()]; ok {
			wanted[i] = row
		}
	}
	l.mirror.Write(ctx, wanted)
	return wanted, nil
}

// storedWindows reads the rows of the windows wanted that the table has, by primary key, in
// chunks of windowChunk. The keys are joined as arrays: an 8-column tuple IN list of a
// thousand is either a scan of the organization's rows or a long plan of a bitmap OR, where
// the join is one primary key lookup a key.
func storedWindows(ctx context.Context, db bun.IDB, wanted []UsageWindow) ([]UsageWindow, error) {
	var found []UsageWindow
	for start := 0; start < len(wanted); start += windowChunk {
		chunk := wanted[start:min(start+windowChunk, len(wanted))]
		var organizations, subjects, apps, keys, users, categories, intervals []string
		starts := make([]time.Time, 0, len(chunk))
		for _, window := range chunk {
			organizations = append(organizations, window.OrganizationID)
			subjects = append(subjects, string(window.Subject))
			apps = append(apps, window.AppID)
			keys = append(keys, window.APIKey)
			users = append(users, window.UserID)
			categories = append(categories, window.Category)
			intervals = append(intervals, string(window.Interval))
			starts = append(starts, window.WindowStart)
		}
		var rows []UsageWindow
		err := db.NewSelect().Model(&rows).
			Join(`JOIN unnest(?::text[], ?::text[], ?::text[], ?::text[], ?::text[], ?::text[], ?::text[], ?::timestamptz[])
				AS k(organization_id, subject, app_id, api_key, user_id, category, interval, window_start)
				ON uw.organization_id = k.organization_id AND uw.subject = k.subject AND uw.app_id = k.app_id
				AND uw.api_key = k.api_key AND uw.user_id = k.user_id AND uw.category = k.category
				AND uw.interval = k.interval AND uw.window_start = k.window_start`,
				pgdialect.Array(organizations), pgdialect.Array(subjects), pgdialect.Array(apps), pgdialect.Array(keys),
				pgdialect.Array(users), pgdialect.Array(categories), pgdialect.Array(intervals), pgdialect.Array(starts)).
			Scan(ctx)
		if err != nil {
			return nil, fmt.Errorf("ledger: read windows: %w", err)
		}
		found = append(found, rows...)
	}
	return found, nil
}

// PruneWindows deletes the windows that ended before cutoff, in batches, and returns how many
// it deleted. The events stay.
func (l *Ledger) PruneWindows(ctx context.Context, cutoff time.Time) (int64, error) {
	var deleted int64
	for {
		result, err := l.db.NewRaw(`DELETE FROM usage_windows WHERE ctid IN (
			SELECT ctid FROM usage_windows WHERE window_end < ? LIMIT ?)`, cutoff, pruneBatch).Exec(ctx)
		if err != nil {
			return deleted, fmt.Errorf("ledger: prune windows: %w", err)
		}
		count, err := result.RowsAffected()
		if err != nil {
			return deleted, fmt.Errorf("ledger: prune windows: %w", err)
		}
		deleted += count
		if count < pruneBatch {
			return deleted, nil
		}
	}
}
