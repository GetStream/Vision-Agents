//go:build integration

package store

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// router is another router's store: a pool of its own on the suite's database, connected
// before the test starts, so a concurrent caller contends in Postgres, not in one pool.
func (s *StoreSuite) router() *Store {
	router, err := Open(s.dsn)
	s.Require().NoError(err)
	s.T().Cleanup(func() { router.Close() })
	s.Require().NoError(router.Ping(s.ctx))
	return router
}

// grantLocks counts the advisory locks in this database under the grant namespace, held
// (granted) or waited for, as pg_locks shows a two-integer key: classid the first key and
// objsubid 2 (https://www.postgresql.org/docs/current/view-pg-locks.html).
func (s *StoreSuite) grantLocks(granted bool) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `
SELECT count(*) FROM pg_locks
WHERE locktype = 'advisory' AND classid = ? AND objsubid = 2 AND granted = ?
  AND database = (SELECT oid FROM pg_database WHERE datname = current_database())`,
		grantLockNamespace, granted).Scan(&count))
	return count
}

// holding locks the connection's grant from another router until the returned release is
// called, and returns once the lock is held.
func (s *StoreSuite) holding(connection ConnectorConnection) (release func()) {
	held, done, finished := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	holder := s.router()
	go func() {
		finished <- holder.WithLockedConnectorConnection(s.ctx, connection.CustomerID, connection.ID,
			func(*ConnectorConnection, func() error) (bool, error) {
				close(held)
				<-done
				return false, nil
			})
	}()
	<-held
	return func() {
		close(done)
		s.Require().NoError(<-finished)
	}
}

func (s *StoreSuite) TestASaveAtTheStoredRevisionAdvancesIt() {
	connection := s.connection("acme-app", nil)
	expires := time.Now().UTC().Add(time.Hour).Truncate(time.Microsecond)

	connection.Revision = 2
	connection.Status = ConnectionConnected
	connection.MaterialSealed = []byte("sealed for revision 2")
	connection.MaterialKEKVersion = 1
	connection.ExpiresAt = &expires
	connection.AccountID = "acct-1"
	connection.GrantedScopes = []string{"read"}
	connection.Metadata = map[string]string{"instance": "https://eu.acme.example"}
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1))

	stored, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal(2, stored.Revision)
	s.Equal(ConnectionConnected, stored.Status)
	s.Equal([]byte("sealed for revision 2"), stored.MaterialSealed)
	s.Equal(1, stored.MaterialKEKVersion)
	s.Require().NotNil(stored.ExpiresAt)
	s.True(expires.Equal(*stored.ExpiresAt))
	s.Equal("acct-1", stored.AccountID)
	s.Equal([]string{"read"}, stored.GrantedScopes)
	s.Equal(map[string]string{"instance": "https://eu.acme.example"}, stored.Metadata)
}

func (s *StoreSuite) TestASaveAtAStaleRevisionIsRefused() {
	connection := s.connection("acme-app", nil)
	stale := connection

	connection.Revision = 2
	connection.LastError = "the first writer"
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1))

	stale.Revision = 2
	stale.LastError = "a writer that read revision 1 too"
	err := s.store.SaveConnectorConnectionAtRevision(s.ctx, &stale, 1)
	s.ErrorIs(err, ErrConnectorConnectionChanged)

	stored, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal("the first writer", stored.LastError)
}

func (s *StoreSuite) TestASaveThatMovesTheRevisionBackIsRefused() {
	connection := s.connection("acme-app", nil)
	connection.Revision = 3
	s.Require().NoError(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1))

	connection.Revision = 2
	s.ErrorContains(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 3), "behind")
}

func (s *StoreSuite) TestASaveOntoADeletedConnectionIsRefused() {
	connection := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	connection.Revision = 2
	connection.MaterialSealed = []byte("sealed after the delete")
	s.ErrorIs(s.store.SaveConnectorConnectionAtRevision(s.ctx, &connection, 1), ErrConnectorConnectionChanged)
}

func (s *StoreSuite) TestASaveIsOnlyItsOwnCustomers() {
	theirs := s.connection("other-app", nil)
	theirs.CustomerID = "acme-app"
	theirs.Revision = 2

	s.ErrorIs(s.store.SaveConnectorConnectionAtRevision(s.ctx, &theirs, 1), ErrConnectorConnectionChanged)
	_, err := s.store.ConnectorConnection(s.ctx, "acme-app", theirs.ID)
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestLockingAnotherCustomersConnectionFindsNone() {
	theirs := s.connection("other-app", nil)

	err := s.store.WithLockedConnectorConnection(s.ctx, "acme-app", theirs.ID,
		func(*ConnectorConnection, func() error) (bool, error) {
			s.Fail("the callback runs only for the customer's own connection")
			return false, nil
		})
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestLockingADeletedConnectionFindsNone() {
	connection := s.connection("acme-app", nil)
	s.Require().NoError(s.store.DeleteConnectorConnection(s.ctx, "acme-app", connection.ID))

	err := s.store.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
		func(*ConnectorConnection, func() error) (bool, error) { return true, nil })
	s.ErrorIs(err, ErrNoConnectorConnection)
}

func (s *StoreSuite) TestLockedCallbacksOnOneConnectionNeverOverlapAcrossRouters() {
	connection := s.connection("acme-app", nil)

	const routers = 8
	stores := make([]*Store, routers)
	for i := range stores {
		stores[i] = s.router()
	}
	errs := make([]error, routers)
	start := make(chan struct{})
	var active atomic.Int32
	var overlapped atomic.Bool
	var wg sync.WaitGroup
	for i, router := range stores {
		wg.Go(func() {
			<-start
			errs[i] = router.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
				func(locked *ConnectorConnection, _ func() error) (bool, error) {
					if active.Add(1) != 1 {
						overlapped.Store(true)
					}
					time.Sleep(10 * time.Millisecond)
					locked.Revision++
					locked.LastError = fmt.Sprintf("revision %d", locked.Revision)
					active.Add(-1)
					return true, nil
				})
		})
	}
	close(start)
	wg.Wait()

	for _, err := range errs {
		s.NoError(err)
	}
	s.False(overlapped.Load(), "two routers ran the callback for one connection at once")
	stored, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal(1+routers, stored.Revision, "every router saw the revision the one before it committed")
	s.Zero(s.grantLocks(true), "every lock is released")
}

func (s *StoreSuite) TestACheckpointIsCommittedWhileTheLockIsStillHeld() {
	connection := s.connection("acme-app", nil)
	other := s.router()

	err := s.store.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
		func(locked *ConnectorConnection, checkpoint func() error) (bool, error) {
			locked.Status = ConnectionNeedsReauthorization
			locked.LastError = "refresh in flight"
			s.Require().NoError(checkpoint())

			seen, err := other.ConnectorConnection(s.ctx, "acme-app", connection.ID)
			s.Require().NoError(err)
			s.Equal(ConnectionNeedsReauthorization, seen.Status, "another router reads the checkpoint at once")
			s.Equal(1, s.grantLocks(true), "and the lock is still held")

			locked.Revision++
			locked.Status = ConnectionConnected
			locked.LastError = ""
			return true, nil
		})
	s.Require().NoError(err)

	stored, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal(2, stored.Revision, "the final commit follows the checkpoint's revision")
	s.Equal(ConnectionConnected, stored.Status)
}

func (s *StoreSuite) TestACallbackThatFailsCommitsNothingMore() {
	connection := s.connection("acme-app", nil)
	failed := errors.New("the callback failed")

	err := s.store.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
		func(locked *ConnectorConnection, _ func() error) (bool, error) {
			locked.Revision++
			return true, failed
		})
	s.ErrorIs(err, failed)

	stored, err := s.store.ConnectorConnection(s.ctx, "acme-app", connection.ID)
	s.Require().NoError(err)
	s.Equal(1, stored.Revision)
}

func (s *StoreSuite) TestAWriteThatSkippedTheLockCannotBeOverwrittenByTheLockedCallback() {
	connection := s.connection("acme-app", nil)

	err := s.store.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
		func(locked *ConnectorConnection, _ func() error) (bool, error) {
			// A writer that does not take the lock, which no store method is: only the
			// revision condition stands between it and the callback.
			_, err := s.store.DB().ExecContext(s.ctx,
				"UPDATE connector_connections SET revision = 2 WHERE id = ?", connection.ID)
			s.Require().NoError(err)
			locked.Revision++
			return true, nil
		})
	s.ErrorIs(err, ErrConnectorConnectionChanged)
}

func (s *StoreSuite) TestAnotherConnectionsLockDoesNotWait() {
	held := s.connection("acme-app", nil)
	free := s.connection("acme-app", nil)
	release := s.holding(held)
	defer release()

	ctx, cancel := context.WithTimeout(s.ctx, 2*time.Second)
	defer cancel()
	err := s.store.WithLockedConnectorConnection(ctx, "acme-app", free.ID,
		func(*ConnectorConnection, func() error) (bool, error) { return false, nil })
	s.NoError(err)
}

func (s *StoreSuite) TestAWaiterPastItsDeadlineReturnsAndLeavesNoLockOrConnection() {
	connection := s.connection("acme-app", nil)
	release := s.holding(connection)
	waiter := s.router()

	ctx, cancel := context.WithTimeout(s.ctx, 200*time.Millisecond)
	defer cancel()
	began := time.Now()
	err := waiter.WithLockedConnectorConnection(ctx, "acme-app", connection.ID,
		func(*ConnectorConnection, func() error) (bool, error) {
			s.Fail("a waiter that gave up never runs the callback")
			return false, nil
		})

	s.ErrorIs(err, context.DeadlineExceeded)
	s.Less(time.Since(began), time.Second, "the waiter returns at its deadline, not when the lock frees")
	s.assertTheWaitEnded(waiter)
	release()
	s.assertNothingIsHeld(waiter)
}

func (s *StoreSuite) TestAWaiterThatIsCanceledReturnsAndLeavesNoLockOrConnection() {
	connection := s.connection("acme-app", nil)
	release := s.holding(connection)
	waiter := s.router()

	// No deadline: a plain cancel, which the driver does not see while a query runs.
	ctx, cancel := context.WithCancel(s.ctx)
	time.AfterFunc(200*time.Millisecond, cancel)
	err := waiter.WithLockedConnectorConnection(ctx, "acme-app", connection.ID,
		func(*ConnectorConnection, func() error) (bool, error) {
			s.Fail("a waiter that gave up never runs the callback")
			return false, nil
		})

	s.ErrorIs(err, context.Canceled)
	s.assertTheWaitEnded(waiter)
	release()
	s.assertNothingIsHeld(waiter)

	err = waiter.WithLockedConnectorConnection(s.ctx, "acme-app", connection.ID,
		func(*ConnectorConnection, func() error) (bool, error) { return false, nil })
	s.NoError(err, "the waiter's router locks the connection again once it is free")
}

// assertTheWaitEnded checks that a waiter that gave up is no longer queued for the lock in
// Postgres and gave its connection up, while the holder still holds it.
func (s *StoreSuite) assertTheWaitEnded(waiter *Store) {
	s.Eventually(func() bool { return s.grantLocks(false) == 0 }, 2*time.Second, 10*time.Millisecond,
		"the server stops waiting for the lock on behalf of a caller that gave up")
	s.Equal(1, s.grantLocks(true), "the holder keeps the lock")
	s.Eventually(func() bool { return waiter.DB().Stats().InUse == 0 }, 2*time.Second, 10*time.Millisecond,
		"the waiter's connection goes back")
}

// assertNothingIsHeld checks that once the holder is done no grant lock is left, so an
// abandoned wait did not end up holding the lock it was canceled out of.
func (s *StoreSuite) assertNothingIsHeld(waiter *Store) {
	s.Eventually(func() bool { return s.grantLocks(true) == 0 && s.grantLocks(false) == 0 },
		2*time.Second, 10*time.Millisecond, "no grant lock is left behind")
	s.Zero(waiter.DB().Stats().InUse)
}
