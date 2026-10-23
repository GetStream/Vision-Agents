package store

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"slices"
	"time"

	"github.com/google/uuid"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// reshipBatch is the most rows one statement of Reship queues, so a long span does not hold
// every row's lock in one transaction.
const reshipBatch = 10_000

// shipLockTimeout bounds giving the ship lock back as the router stops.
const shipLockTimeout = 5 * time.Second

// RequestsToShip is up to limit of this shard's rows still to reach the usage service that
// started more than settle ago, oldest first, after the row after when it is not nil: the
// rows started since are the writer's to ship.
func (s *Store) RequestsToShip(ctx context.Context, settle time.Duration, after *Request, limit int) ([]Request, error) {
	var rows []Request
	query := s.shard.NewSelect().Model(&rows).
		Where("to_ship").
		Where("started_at < now() - make_interval(secs => ?)", settle.Seconds()).
		OrderExpr("started_at, id").
		Limit(limit)
	if after != nil {
		query = query.Where("(started_at, id) > (?, ?)", after.StartedAt, after.ID)
	}
	if err := query.Scan(ctx); err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: requests to ship: %w", err))
	}
	return rows, nil
}

// Shipped is which of the usage ids this shard holds a row for that the usage service has, as
// whichever router's sweeper shipped it recorded.
func (s *Store) Shipped(ctx context.Context, usageIDs []uuid.UUID) ([]uuid.UUID, error) {
	if len(usageIDs) == 0 {
		return nil, nil
	}
	var shipped []uuid.UUID
	err := s.shard.NewSelect().Model((*Request)(nil)).Column("usage_id").
		Where("usage_id IN (?)", bun.In(usageIDs)).
		Where("shipped_at IS NOT NULL").
		Scan(ctx, &shipped)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("store: shipped: %w", err))
	}
	return shipped, nil
}

// RecordShipment marks what the usage service answered for rows it was sent: shipped are the
// usage ids it has, and rejected the ones it refused, each with why. Neither is sent again
// until somebody reships it.
func (s *Store) RecordShipment(ctx context.Context, shipped []uuid.UUID, rejected map[uuid.UUID]string) error {
	if len(shipped) == 0 && len(rejected) == 0 {
		return nil
	}
	ids := make([]string, 0, len(rejected))
	reasons := make([]string, 0, len(rejected))
	for id, reason := range rejected {
		ids, reasons = append(ids, id.String()), append(reasons, reason)
	}
	err := s.shard.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := boundShipping(ctx, tx); err != nil {
			return err
		}
		if len(shipped) > 0 {
			if _, err := tx.NewUpdate().Model((*Request)(nil)).
				Set("to_ship = false").Set("shipped_at = now()").
				Where("usage_id IN (?)", bun.In(shipped)).
				Exec(ctx); err != nil {
				return err
			}
		}
		if len(rejected) > 0 {
			if _, err := tx.ExecContext(ctx, `
UPDATE requests AS r SET to_ship = false, ship_error = v.reason
FROM unnest(?::uuid[], ?::text[]) AS v(usage_id, reason)
WHERE r.usage_id = v.usage_id`, pgdialect.Array(ids), pgdialect.Array(reasons)); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: record shipment: %w", err))
	}
	return nil
}

// SetRequestsOrganization writes the organization on rows written without one.
func (s *Store) SetRequestsOrganization(ctx context.Context, ids []uuid.UUID, organizationID string) error {
	if len(ids) == 0 {
		return nil
	}
	err := s.shard.RunInTx(ctx, nil, func(ctx context.Context, tx bun.Tx) error {
		if err := boundShipping(ctx, tx); err != nil {
			return err
		}
		_, err := tx.NewUpdate().Model((*Request)(nil)).
			Set("organization_id = ?", organizationID).
			Where("usage_id IN (?)", bun.In(ids)).
			Where("organization_id = ''").
			Exec(ctx)
		return err
	})
	if err != nil {
		return stack.Wrap(fmt.Errorf("store: set requests organization: %w", err))
	}
	return nil
}

// shippingLockWait bounds how long a shipping write waits on another session's row locks, and
// shippingIdle how long its transaction may sit idle before Postgres ends the session. A
// router whose network went silent mid-transaction keeps its locks only that long, rather than
// until the kernel's keepalive gives up on the pool's connection, two hours on.
const (
	shippingLockWait = 5 * time.Second
	shippingIdle     = 15 * time.Second
)

// boundShipping sets shippingLockWait and shippingIdle for the rest of tx.
func boundShipping(ctx context.Context, tx bun.Tx) error {
	_, err := tx.ExecContext(ctx, "SELECT set_config('lock_timeout', ?, true), set_config('idle_in_transaction_session_timeout', ?, true)",
		fmt.Sprintf("%dms", shippingLockWait.Milliseconds()), fmt.Sprintf("%dms", shippingIdle.Milliseconds()))
	return err
}

// Unshipped is how many of this shard's rows are still to reach the usage service, and how
// long ago the oldest of them started: zero when there is none.
func (s *Store) Unshipped(ctx context.Context) (int64, time.Duration, error) {
	var rows int64
	var oldest float64
	err := s.shard.QueryRowContext(ctx,
		"SELECT count(*), COALESCE(EXTRACT(EPOCH FROM now() - min(started_at)), 0)::float8 FROM requests WHERE to_ship",
	).Scan(&rows, &oldest)
	if err != nil {
		return 0, 0, stack.Wrap(fmt.Errorf("store: unshipped: %w", err))
	}
	return rows, time.Duration(max(oldest, 0) * float64(time.Second)), nil
}

// Reship queues this shard's rows started since then to be sent to the usage service again:
// the rows it has (shipped_at set), or with rejected the rows it refused (ship_error set),
// clearing why. A row it never had nor refused is left alone: one written before the router
// shipped, or carried in by a data move, which the deployment that recorded it bills. So is a
// row from before usage ids, which the service could not record once, one already queued,
// and one shipped since Reship started, so a sweeper shipping meanwhile cannot keep it going.
// It walks the rows by id, so each is looked at once. It returns how many it queued.
func (s *Store) Reship(ctx context.Context, since time.Time, rejected bool) (int64, error) {
	var started time.Time
	var last sql.NullInt64
	if err := s.shard.QueryRowContext(ctx, "SELECT now(), (SELECT min(id) - 1 FROM requests WHERE started_at >= ?)", since).
		Scan(&started, &last); err != nil {
		return 0, stack.Wrap(fmt.Errorf("store: reship: %w", err))
	}
	if !last.Valid {
		return 0, nil
	}
	filter, args := "shipped_at IS NOT NULL AND shipped_at < ?", []any{started}
	if rejected {
		filter, args = "ship_error <> ''", nil
	}
	var queued int64
	for {
		var ids []int64
		err := s.shard.NewRaw(`
WITH batch AS (
    SELECT id FROM requests
    WHERE id > ? AND started_at >= ? AND usage_id IS NOT NULL AND NOT to_ship AND `+filter+`
    ORDER BY id LIMIT ?
)
UPDATE requests AS r SET to_ship = true, ship_error = '' FROM batch WHERE r.id = batch.id
RETURNING r.id`, append(append([]any{last.Int64, since}, args...), reshipBatch)...).Scan(ctx, &ids)
		if err != nil {
			return queued, stack.Wrap(fmt.Errorf("store: reship: %w", err))
		}
		queued += int64(len(ids))
		if len(ids) < reshipBatch {
			return queued, nil
		}
		last.Int64 = slices.Max(ids)
	}
}

// ShipLock is the session advisory lock that makes one router of this shard the one that
// sweeps its unshipped rows, held on a connection kept for it: a router that dies ends its
// session, and the lock with it. One goroutine uses it.
type ShipLock struct {
	db   *bun.DB
	conn *sql.Conn
	held bool
}

// ShipLock is this shard's ship lock, not taken yet.
func (s *Store) ShipLock() *ShipLock {
	return &ShipLock{db: s.shard}
}

// shipLockKeepalives has Postgres probe the holder's connection, so a holder whose network
// went silent loses its session, and the lock, within about a minute rather than at the
// kernel's default of two hours. Ignored on a Unix socket.
const shipLockKeepalives = "SELECT set_config('tcp_keepalives_idle', '30', false), " +
	"set_config('tcp_keepalives_interval', '10', false), set_config('tcp_keepalives_count', '3', false)"

// Hold takes the lock when it is free and reports whether this router holds it. Once taken it
// is kept, and each call checks the session holding it is still there. A router that does not
// get it gives the connection back to the pool until the next try. Bound ctx: a connection that
// stops answering holds the call until it ends.
func (l *ShipLock) Hold(ctx context.Context) (bool, error) {
	if l.conn == nil {
		conn, err := l.db.DB.Conn(ctx)
		if err != nil {
			return false, stack.Wrap(fmt.Errorf("store: ship lock: %w", err))
		}
		l.conn = conn
	}
	if l.held {
		if _, err := l.conn.ExecContext(ctx, "SELECT 1"); err != nil {
			l.drop()
			return false, stack.Wrap(fmt.Errorf("store: ship lock: %w", err))
		}
		return true, nil
	}
	if err := l.conn.QueryRowContext(ctx, "SELECT pg_try_advisory_lock(hashtextextended('router.usage.ship', 0))").Scan(&l.held); err != nil {
		// The lock may have been granted with the answer lost; ending the session releases it.
		l.drop()
		return false, stack.Wrap(fmt.Errorf("store: ship lock: %w", err))
	}
	if !l.held {
		_ = l.conn.Close()
		l.conn = nil
		return false, nil
	}
	if _, err := l.conn.ExecContext(ctx, shipLockKeepalives); err != nil {
		l.drop()
		return false, stack.Wrap(fmt.Errorf("store: ship lock: keepalives: %w", err))
	}
	return true, nil
}

// Release gives the lock back, if it is held, and the connection with it.
func (l *ShipLock) Release() {
	if l.conn == nil {
		return
	}
	if l.held {
		ctx, cancel := context.WithTimeout(context.Background(), shipLockTimeout)
		defer cancel()
		if _, err := l.conn.ExecContext(ctx, "SELECT pg_advisory_unlock(hashtextextended('router.usage.ship', 0))"); err != nil {
			l.drop()
			return
		}
	}
	_ = l.conn.Close()
	l.conn, l.held = nil, false
}

// drop ends the session rather than handing it back to the pool with the lock in it.
func (l *ShipLock) drop() {
	_ = l.conn.Raw(func(any) error { return driver.ErrBadConn })
	_ = l.conn.Close()
	l.conn, l.held = nil, false
}
