//go:build integration

package store

import (
	"time"
)

// waitForLockWaiters waits until n statements of this database wait on a lock.
func (s *StoreSuite) waitForLockWaiters(n int, what string) {
	s.Require().Eventually(func() bool {
		var waiting int
		s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
			"SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND wait_event_type = 'Lock'").Scan(&waiting))
		return waiting >= n
	}, 5*time.Second, 10*time.Millisecond, what)
}
