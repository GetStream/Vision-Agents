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

// ErrConnectorConnectionChanged says a credentials write found the connection no longer at the
// revision it expected, or deleted: someone else saved first, or the connection is gone.
// The writer reloads under the lock rather than overwrite what it did not see.
var ErrConnectorConnectionChanged = errors.New("store: connector connection changed while saving")

// credentialLockNamespace is the first key of the two-integer advisory lock a connection's
// credential state is held under; the second is credentialLockKey. Postgres keeps the one-bigint and the
// two-integer key spaces apart ("note that these two key spaces do not overlap",
// https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS),
// and every other advisory lock in this repository takes one bigint
// (hashtextextended(..., 731) in logs.go and the agent_logs migrations,
// hashtextextended(..., definitionLockSeed) in connectors.go), so no key here can be one of
// theirs. 839 is this lock's issue, AI-839, as 833 is definitionLockSeed's; the value only
// has to differ from any other two-integer lock, of which there is none.
const credentialLockNamespace int32 = 839

// credentialDetachedTimeout bounds the statements that run on a context of their own because the
// caller's may already be done: a credentials write, so tokens a provider already rotated are not
// dropped by the caller's deadline, and the unlock. It is the prototype's unlock timeout
// (withConnectorConnectionLock in internal/store/connectors.go on codex/connector-support at
// cf62af0d); unverified, not measured. Each is one statement by primary key or lock key. An
// unlock that misses it closes the connection instead, and the session's end releases the
// lock.
const credentialDetachedTimeout = 2 * time.Second

// credentialRetryFirst and credentialRetryMost are the pause between two pg_try_advisory_lock attempts:
// the first pause, doubled after each miss up to the most. A miss means another router holds
// the lock, usually across a provider round trip, and the most bounds how long a waiter idles
// after the holder unlocks. Unverified, not measured.
const (
	credentialRetryFirst = 5 * time.Millisecond
	credentialRetryMost  = 100 * time.Millisecond
)

// credentialColumns are what a credentials write sets: the credential state itself
// (core.CredentialState: status, revision, stored credentials, expiry, last error) and what a
// consent captures beside it (account, scopes,
// metadata). The prototype's list (SaveConnectorConnectionAtRevision at cf62af0d) with its
// credential columns renamed to credentials ones and metadata added, less cached_tools and
// tools_*: those have a writer of their own, and a credentials write would put back a stale copy.
var credentialColumns = []string{
	"status", "revision", "credentials_sealed", "credentials_kek_version", "expires_at", "last_error",
	"account_id", "granted_scopes", "metadata", "updated_at",
}

// SaveConnectorConnectionAtRevision saves new stored credentials, already sealed for
// expectedRevision + 1, onto a connection that is live and still at expectedRevision, under
// the same lock WithLockedConnectorConnection holds, so it never lands in the middle of a
// refresh. It always advances the revision by one. A locked callback can commit a status
// without advancing it (needs_reauthorization before a refresh), so a save at the same
// revision from a snapshot read before that would pass the compare-and-swap and put back the
// credentials the callback retired; a save of new credentials replaces them instead.
func (s *Store) SaveConnectorConnectionAtRevision(ctx context.Context, connection *ConnectorConnection, expectedRevision int) error {
	if connection.CustomerID == "" || connection.ID == "" {
		return errors.New("store: a customer and a connection id are required")
	}
	if expectedRevision < 1 {
		return errors.New("store: the expected revision is 1 or more")
	}
	if connection.Revision != expectedRevision+1 {
		return fmt.Errorf("store: a save is new credentials at revision %d, not %d", expectedRevision+1, connection.Revision)
	}
	return s.withCredentialLock(ctx, connection.CustomerID, connection.ID, func(conn bun.Conn) error {
		return s.saveAtRevision(ctx, conn, connection, expectedRevision)
	})
}

// WithLockedConnectorConnection loads one live connection under the session-level advisory
// lock on its credential state and runs fn. fn may call checkpoint to commit the connection as it is
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
	return s.withCredentialLock(ctx, customerID, id, func(conn bun.Conn) error {
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

// saveAtRevision is the compare-and-swap every credentials write is: it matches only the live row
// still at expected, so of two writers that read the same revision one wins and the other
// gets ErrConnectorConnectionChanged. A write never moves the revision back, since that would
// make a blob sealed for an earlier revision open again. It runs detached from ctx, within
// credentialDetachedTimeout: the driver turns a passed deadline into a passed socket deadline
// (Conn.deadline in github.com/uptrace/bun/driver/pgdriver v1.2.18), which would drop a
// rotation the provider already made.
func (s *Store) saveAtRevision(ctx context.Context, conn bun.IConn, connection *ConnectorConnection, expected int) error {
	if connection.Revision < expected {
		return fmt.Errorf("store: revision %d is behind the stored %d", connection.Revision, expected)
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), credentialDetachedTimeout)
	defer cancel()
	connection.UpdatedAt = time.Now().UTC().Truncate(time.Microsecond)
	if connection.CredentialsSealed == nil {
		connection.CredentialsSealed = []byte{}
	}
	if connection.Metadata == nil {
		connection.Metadata = map[string]string{}
	}
	if connection.GrantedScopes == nil {
		connection.GrantedScopes = []string{}
	}
	// A save that leaves the connection anything but connected frees its provider unit
	// (SetConnectorConnectionProviderUnit says why); one that keeps it connected, such as a
	// refresh, keeps the unit the row holds. The row's, not the caller's copy, which may
	// predate the unit.
	if connection.Status != ConnectionConnected {
		connection.ProviderUnitID = ""
	}
	result, err := s.db.NewUpdate().Conn(conn).Model(connection).
		Column(credentialColumns...).
		Set("provider_unit_id = CASE WHEN ? = ? THEN cc.provider_unit_id END", connection.Status, ConnectionConnected).
		Where("id = ?", connection.ID).
		Where("customer_id = ?", connection.CustomerID).
		Where("revision = ?", expected).
		Where("deleted_at IS NULL").
		Exec(ctx)
	if err != nil {
		return fmt.Errorf("store: save connector credentials: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("store: save connector credentials: %w", err)
	}
	if affected == 0 {
		return ErrConnectorConnectionChanged
	}
	return nil
}

// withCredentialLock runs fn on one pool connection that holds the session-level advisory lock on
// the connection's credential state, and gives the lock and the connection back when fn returns.
//
// Session-level, not pg_advisory_xact_lock as connectors.go uses: checkpoint commits while
// the lock must stay held, and a transaction-level lock goes with the commit. "Once acquired
// at session level, an advisory lock is held until explicitly released or the session ends",
// and session-level requests "do not honor transaction semantics"
// (https://www.postgresql.org/docs/current/explicit-locking.html#ADVISORY-LOCKS). That is also
// why it lives on one dedicated connection: a lock taken on a pooled connection and not
// released would stay with whoever borrows it next.
func (s *Store) withCredentialLock(ctx context.Context, customerID, id string, fn func(bun.Conn) error) error {
	key := credentialLockKey(customerID, id)
	conn, err := s.acquireCredentialLock(ctx, credentialLockNamespace, key)
	if err != nil {
		return err
	}
	defer releaseCredentialLock(conn, credentialLockNamespace, key)
	return fn(conn)
}

// acquireCredentialLock tries the lock with pg_try_advisory_lock, which "will either obtain the
// lock immediately and return true, or return false without waiting"
// (https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS),
// and pauses between tries until it gets it or ctx is done. namespace is the first key of the
// two (credentialLockNamespace, providerAppLockNamespace). Nothing waits in Postgres, so a
// caller that gives up leaves no wait queued on the server, and a waiter holds no connection
// while it pauses. A blocking pg_advisory_lock could not promise that: the driver bounds
// every read by its ReadTimeout, 10s unless set (newDefaultConfig in
// github.com/uptrace/bun/driver/pgdriver v1.2.18, which store.Open does not change), or by
// ctx's deadline, and drops the socket when it passes, while the server "will detect the loss
// of the connection only at the next interaction with the socket"
// (https://www.postgresql.org/docs/current/runtime-config-connection.html#GUC-CLIENT-CONNECTION-CHECK-INTERVAL),
// so the wait would stay queued. The cost is that waiters are not served in order.
func (s *Store) acquireCredentialLock(ctx context.Context, namespace, key int32) (bun.Conn, error) {
	pause := credentialRetryFirst
	for {
		conn, err := s.db.Conn(ctx)
		if err != nil {
			return bun.Conn{}, fmt.Errorf("store: credential lock connection: %w", err)
		}
		var locked bool
		err = conn.QueryRowContext(ctx, "SELECT pg_try_advisory_lock(?::integer, ?::integer)", namespace, key).Scan(&locked)
		if err != nil {
			// The lock may have been granted with the answer lost; closing the connection
			// ends the session, which releases it.
			discard(conn)
			if ctx.Err() != nil {
				return bun.Conn{}, fmt.Errorf("store: wait for credential lock: %w", context.Cause(ctx))
			}
			return bun.Conn{}, fmt.Errorf("store: acquire credential lock: %w", err)
		}
		if locked {
			return conn, nil
		}
		_ = conn.Close()
		timer := time.NewTimer(pause)
		select {
		case <-ctx.Done():
			timer.Stop()
			return bun.Conn{}, fmt.Errorf("store: wait for credential lock: %w", context.Cause(ctx))
		case <-timer.C:
		}
		pause = min(2*pause, credentialRetryMost)
	}
}

// releaseCredentialLock unlocks and hands the connection back to the pool. pg_advisory_unlock
// returns false when the lock was not held
// (https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADVISORY-LOCKS);
// that, or an unlock that fails, closes the connection rather than return a session in a
// state nobody knows.
func releaseCredentialLock(conn bun.Conn, namespace, key int32) {
	ctx, cancel := context.WithTimeout(context.Background(), credentialDetachedTimeout)
	defer cancel()
	var released bool
	err := conn.QueryRowContext(ctx, "SELECT pg_advisory_unlock(?::integer, ?::integer)", namespace, key).Scan(&released)
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

// credentialLockKey is the second lock key: the first four bytes of SHA-256 over customer and
// connection id, joined by NUL, which no Postgres text value contains ("the character with
// code zero (sometimes called NUL) cannot be stored",
// https://www.postgresql.org/docs/current/datatype-character.html), so no two pairs join to
// the same input. It is computed here rather than by hashtext, so it does not depend on
// the server's hash function.
//
// 32 bits, because the two-integer form is what keeps credential locks apart from every bigint
// lock (credentialLockNamespace), and its other half is the namespace. So two connections can
// share a key. That costs contention, never correctness: the lock only orders callbacks, and
// each callback writes its own row under its own compare-and-swap, so a collision makes two
// connections take turns. With keys spread uniformly over 2^32 values, a lock taken while k
// others are held shares a key with one of them with probability about (k-1)/2^32, 2.3e-7
// for k = 1000; a table of a million connections has about 116 colliding pairs
// (n(n-1)/2 / 2^32), and a pair waits only when both are locked at once.
func credentialLockKey(customerID, id string) int32 {
	sum := sha256.Sum256([]byte(customerID + "\x00" + id))
	return int32(binary.BigEndian.Uint32(sum[:4]))
}
