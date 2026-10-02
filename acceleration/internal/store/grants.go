package store

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"database/sql/driver"
	"encoding/binary"
	"errors"
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// ErrConnectorConnectionChanged says a grant write found the connection no longer at the
// revision it expected, or deleted: someone else saved first, or the connection is gone.
// The writer reloads under the lock rather than overwrite what it did not see.
var ErrConnectorConnectionChanged = errors.New("store: connector connection changed while saving")

// grantLockNamespace is the first key of the two-integer advisory lock a connection's grant
// is held under; the second is grantLockKey. Postgres keeps the one-bigint and the
// two-integer key spaces apart ("note that these two key spaces do not overlap",
// https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS),
// and every other advisory lock in this repository takes one bigint
// (hashtextextended(..., 731) in logs.go and the agent_logs migrations,
// hashtextextended(..., definitionLockSeed) in connectors.go), so no key here can be one of
// theirs. 839 is this lock's issue, AI-839, as 833 is definitionLockSeed's; the value only
// has to differ from any other two-integer lock, of which there is none.
const grantLockNamespace int32 = 839

// grantReleaseTimeout bounds the unlock after fn returns, which runs on a context of its own
// because the caller's may already be done. It is the prototype's value
// (withConnectorConnectionLock in internal/store/connectors.go on codex/connector-support at
// cf62af0d); unverified, not measured. An unlock that misses it closes the connection
// instead, and the session's end releases the lock.
const grantReleaseTimeout = 2 * time.Second

// grantCancelTimeout bounds the pg_cancel_backend that ends a wait the caller gave up on.
// Unverified, not measured: it is one round trip on a pool connection, and if it misses, the
// abandoned wait still ends once the holder unlocks, when its connection is closed.
const grantCancelTimeout = 2 * time.Second

// grantColumns are what a grant write sets: the grant itself (core.Grant: status, revision,
// material, expiry, last error) and what a consent captures beside it (account, scopes,
// metadata). The prototype's list (SaveConnectorConnectionAtRevision at cf62af0d) with its
// credential columns renamed to material ones and metadata added, less cached_tools and
// tools_*: those have a writer of their own, and a grant write would put back a stale copy.
var grantColumns = []string{
	"status", "revision", "material_sealed", "material_kek_version", "expires_at", "last_error",
	"account_id", "granted_scopes", "metadata", "updated_at",
}

// SaveConnectorConnectionAtRevision writes the connection's grant columns only if the stored
// row is live and still at expectedRevision, under the same lock WithLockedConnectorConnection
// holds, so it never lands in the middle of a refresh. The material must already be sealed
// for connection.Revision, which is expectedRevision or later: a write never moves the
// revision back, since that would make a blob sealed for an earlier revision open again.
func (s *Store) SaveConnectorConnectionAtRevision(ctx context.Context, connection *ConnectorConnection, expectedRevision int) error {
	if connection.CustomerID == "" || connection.ID == "" {
		return errors.New("store: a customer and a connection id are required")
	}
	if expectedRevision < 1 {
		return errors.New("store: the expected revision is 1 or more")
	}
	return s.withGrantLock(ctx, connection.CustomerID, connection.ID, func(conn bun.Conn) error {
		return s.saveAtRevision(ctx, conn, connection, expectedRevision)
	})
}

// WithLockedConnectorConnection loads one live connection under the session-level advisory
// lock on its grant and runs fn. fn may call checkpoint to commit the connection as it is
// now, before a side effect it cannot take back, and keeps the lock while it does; returning
// changed commits the final state the same way. Each commit is a compare-and-swap on the
// revision the previous one left. fn must not lock the same connection again: that is a
// second pool connection waiting on the first.
func (s *Store) WithLockedConnectorConnection(
	ctx context.Context,
	customerID, id string,
	fn func(connection *ConnectorConnection, checkpoint func() error) (changed bool, err error),
) error {
	if customerID == "" || id == "" {
		return errors.New("store: a customer and a connection id are required")
	}
	return s.withGrantLock(ctx, customerID, id, func(conn bun.Conn) error {
		var connection ConnectorConnection
		err := s.db.NewSelect().Conn(conn).Model(&connection).
			Where("customer_id = ?", customerID).
			Where("id = ?", id).
			Where("deleted_at IS NULL").
			Scan(ctx)
		if errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("%w: %s", ErrNoConnectorConnection, id)
		}
		if err != nil {
			return fmt.Errorf("store: load locked connector connection: %w", err)
		}
		expected := connection.Revision
		checkpoint := func() error {
			if err := s.saveAtRevision(ctx, conn, &connection, expected); err != nil {
				return err
			}
			expected = connection.Revision
			return nil
		}
		changed, err := fn(&connection, checkpoint)
		if err != nil || !changed {
			return err
		}
		return checkpoint()
	})
}

// saveAtRevision is the compare-and-swap every grant write is: it matches only the live row
// still at expected, so of two writers that read the same revision one wins and the other
// gets ErrConnectorConnectionChanged.
func (s *Store) saveAtRevision(ctx context.Context, conn bun.IConn, connection *ConnectorConnection, expected int) error {
	if connection.Revision < expected {
		return fmt.Errorf("store: revision %d is behind the stored %d", connection.Revision, expected)
	}
	connection.UpdatedAt = time.Now().UTC().Truncate(time.Microsecond)
	if connection.MaterialSealed == nil {
		connection.MaterialSealed = []byte{}
	}
	if connection.Metadata == nil {
		connection.Metadata = map[string]string{}
	}
	if connection.GrantedScopes == nil {
		connection.GrantedScopes = []string{}
	}
	result, err := s.db.NewUpdate().Conn(conn).Model(connection).
		Column(grantColumns...).
		Where("id = ?", connection.ID).
		Where("customer_id = ?", connection.CustomerID).
		Where("revision = ?", expected).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save connector grant: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: save connector grant: %w", err)
	}
	if affected == 0 {
		return ErrConnectorConnectionChanged
	}
	return nil
}

// withGrantLock runs fn on one pool connection that holds the session-level advisory lock on
// the connection's grant, and gives the lock and the connection back when fn returns.
//
// Session-level, not pg_advisory_xact_lock as connectors.go uses: checkpoint commits while
// the lock must stay held, and a transaction-level lock goes with the commit. A session lock
// is "held until explicitly released or the session ends" and "not released on transaction
// commit or rollback" (https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS),
// which is also why it lives on one dedicated connection: a lock taken on a pooled connection
// and not released would stay with whoever borrows it next.
func (s *Store) withGrantLock(ctx context.Context, customerID, id string, fn func(bun.Conn) error) error {
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("store: grant lock connection: %w", err)
	}
	var pid int
	if err := conn.QueryRowContext(ctx, "SELECT pg_backend_pid()").Scan(&pid); err != nil {
		discard(conn)
		return fmt.Errorf("store: grant lock connection: %w", err)
	}
	key := grantLockKey(customerID, id)
	if err := s.acquireGrantLock(ctx, conn, pid, key); err != nil {
		return err
	}
	defer releaseGrantLock(conn, key)
	return fn(conn)
}

// acquireGrantLock waits for the lock until it is granted or ctx is done. The wait runs
// without ctx, because the driver turns a deadline into a socket read deadline
// (Conn.deadline in github.com/uptrace/bun/driver/pgdriver v1.2.18) and ignores a plain
// cancel: dropping the socket would leave the server still queued for the lock, since it
// "will detect the loss of the connection only at the next interaction with the socket"
// (client_connection_check_interval defaults to 0,
// https://www.postgresql.org/docs/current/runtime-config-connection.html#GUC-CLIENT-CONNECTION-CHECK-INTERVAL).
// So a caller that gives up returns at once, and pg_cancel_backend ("cancels the current
// query of the session", allowed for a backend of the same role,
// https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADMIN-SIGNAL-TABLE)
// ends the wait on the server. Whether the cancel or the grant came first, the connection is
// then closed, which ends the session and with it any lock it got.
func (s *Store) acquireGrantLock(ctx context.Context, conn bun.Conn, pid int, key int32) error {
	acquired := make(chan error, 1)
	go func() {
		_, err := conn.ExecContext(context.WithoutCancel(ctx),
			"SELECT pg_advisory_lock(?::integer, ?::integer)", grantLockNamespace, key)
		acquired <- err
	}()
	select {
	case err := <-acquired:
		if err != nil {
			discard(conn)
			return fmt.Errorf("store: acquire grant lock: %w", err)
		}
		return nil
	case <-ctx.Done():
		go func() {
			cancelCtx, cancel := context.WithTimeout(context.Background(), grantCancelTimeout)
			defer cancel()
			_, _ = s.db.ExecContext(cancelCtx, "SELECT pg_cancel_backend(?)", pid)
			<-acquired
			discard(conn)
		}()
		return fmt.Errorf("store: wait for grant lock: %w", context.Cause(ctx))
	}
}

// releaseGrantLock unlocks and hands the connection back to the pool. pg_advisory_unlock
// returns false when the lock was not held
// (https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS);
// that, or an unlock that fails, closes the connection rather than return a session in a
// state nobody knows.
func releaseGrantLock(conn bun.Conn, key int32) {
	ctx, cancel := context.WithTimeout(context.Background(), grantReleaseTimeout)
	defer cancel()
	var released bool
	err := conn.QueryRowContext(ctx, "SELECT pg_advisory_unlock(?::integer, ?::integer)", grantLockNamespace, key).Scan(&released)
	if err != nil || !released {
		discard(conn)
		return
	}
	_ = conn.Close()
}

// discard closes the connection instead of returning it to the pool: database/sql drops a
// connection whose Raw callback returns driver.ErrBadConn, and Postgres ends the session.
func discard(conn bun.Conn) {
	_ = conn.Raw(func(any) error { return driver.ErrBadConn })
	_ = conn.Close()
}

// grantLockKey is the second lock key: the first four bytes of SHA-256 over customer and
// connection id, joined by NUL, which no Postgres text value contains ("the character with
// code zero (sometimes called NUL) cannot be stored",
// https://www.postgresql.org/docs/current/datatype-character.html), so no two pairs join to
// the same input. It is computed here rather than by hashtext, so it does not depend on
// the server's hash function. Two connections that share a key only take turns; the lock
// guards no row the other one writes.
func grantLockKey(customerID, id string) int32 {
	sum := sha256.Sum256([]byte(customerID + "\x00" + id))
	return int32(binary.BigEndian.Uint32(sum[:4]))
}
