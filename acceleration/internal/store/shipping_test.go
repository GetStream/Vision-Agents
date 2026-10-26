//go:build integration

package store

import (
	"time"

	"github.com/google/uuid"
)

func (s *StoreSuite) TestMarkingARowAnotherSessionHoldsFailsWithinTheLockWait() {
	request := &Request{UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "lock-" + uuid.NewString(),
		Provider: "openai", Model: "gpt", StartedAt: time.Now().UTC(), Success: true, ToShip: true}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
	// A router that went silent with the row locked.
	tx, err := s.store.ShardDB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	defer func() { _ = tx.Rollback() }()
	_, err = tx.ExecContext(s.ctx, "SELECT 1 FROM requests WHERE usage_id = ? FOR UPDATE", request.UsageID)
	s.Require().NoError(err)

	started := time.Now()
	err = s.store.RecordShipment(s.ctx, []uuid.UUID{request.UsageID}, nil)

	s.Require().ErrorContains(err, "lock timeout")
	s.Less(time.Since(started), shippingLockWait+3*time.Second)
	s.GreaterOrEqual(time.Since(started), shippingLockWait-time.Second)
	err = s.store.SetRequestsOrganization(s.ctx, []uuid.UUID{request.UsageID}, "org")
	s.ErrorContains(err, "lock timeout", "the sweeper's other write waits no longer")
}

func (s *StoreSuite) TestPostgresProbesTheConnectionHoldingTheShipLock() {
	lock := s.store.ShipLock()
	defer lock.Release()

	held, err := lock.Hold(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)

	var idle, interval, count string
	s.Require().NoError(lock.conn.QueryRowContext(s.ctx,
		"SELECT current_setting('tcp_keepalives_idle'), current_setting('tcp_keepalives_interval'), current_setting('tcp_keepalives_count')",
	).Scan(&idle, &interval, &count))
	s.Equal([]string{"30", "10", "3"}, []string{idle, interval, count})
}

func (s *StoreSuite) TestARouterWithoutTheShipLockKeepsNoConnectionForIt() {
	holder, other := s.store.ShipLock(), s.store.ShipLock()
	defer holder.Release()
	defer other.Release()
	held, err := holder.Hold(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)

	held, err = other.Hold(s.ctx)

	s.Require().NoError(err)
	s.False(held)
	s.Nil(other.conn, "given back to the pool until the next try")
}
