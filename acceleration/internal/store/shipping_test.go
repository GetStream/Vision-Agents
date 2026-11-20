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

// shippable records a row of organization's started at at, still to be shipped.
func (s *StoreSuite) shippable(organization string, at time.Time) uuid.UUID {
	request := &Request{UsageID: uuid.Must(uuid.NewV7()), Modality: "llm", CustomerID: "acme", OrganizationID: organization,
		Provider: "openai", Model: "gpt", StartedAt: at, Success: true, ToShip: true}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
	return request.UsageID
}

// shipping is the row's to_ship and ship_error.
func (s *StoreSuite) shipping(usageID uuid.UUID) (bool, string) {
	var request Request
	s.Require().NoError(s.store.ShardDB().NewSelect().Model(&request).Where("usage_id = ?", usageID).Scan(s.ctx))
	return request.ToShip, request.ShipError
}

func (s *StoreSuite) TestUnshippedCountsTheRowsWithAnOrganizationStillToShip() {
	now := time.Now().UTC()
	s.shippable("org", now.Add(-2*time.Hour))
	s.shippable("org", now.Add(-time.Hour))
	s.shippable("", now.Add(-3*time.Hour))
	shipped := s.shippable("org", now.Add(-4*time.Hour))
	s.Require().NoError(s.store.RecordShipment(s.ctx, []uuid.UUID{shipped}, nil))

	rows, oldest, err := s.store.Unshipped(s.ctx)

	s.Require().NoError(err)
	s.EqualValues(2, rows, "neither the shipped row nor the one with no organization")
	s.InDelta((2 * time.Hour).Seconds(), oldest.Seconds(), 60)
}

func (s *StoreSuite) TestReshipQueuesTheShippedRowsStartedSinceAgain() {
	since := time.Now().UTC().Add(-time.Hour)
	recent := s.shippable("org", since.Add(time.Minute))
	older := s.shippable("org", since.Add(-time.Minute))
	refused := s.shippable("org", since.Add(2*time.Minute))
	waiting := s.shippable("org", since.Add(3*time.Minute))
	s.Require().NoError(s.store.RecordShipment(s.ctx, []uuid.UUID{recent, older}, map[uuid.UUID]string{refused: "unknown price"}))

	queued, err := s.store.Reship(s.ctx, since, false)

	s.Require().NoError(err)
	s.EqualValues(1, queued)
	toShip, _ := s.shipping(recent)
	s.True(toShip)
	toShip, _ = s.shipping(older)
	s.False(toShip, "started before since")
	toShip, reason := s.shipping(refused)
	s.False(toShip, "the service refused it, it never had it")
	s.Equal("unknown price", reason)
	toShip, _ = s.shipping(waiting)
	s.True(toShip, "still queued, as it was")
}

func (s *StoreSuite) TestReshipOfRejectedRowsQueuesTheRefusedOnesAndClearsWhy() {
	since := time.Now().UTC().Add(-time.Hour)
	shipped := s.shippable("org", since.Add(time.Minute))
	refused := s.shippable("org", since.Add(2*time.Minute))
	s.Require().NoError(s.store.RecordShipment(s.ctx, []uuid.UUID{shipped}, map[uuid.UUID]string{refused: "unknown price"}))

	queued, err := s.store.Reship(s.ctx, since, true)

	s.Require().NoError(err)
	s.EqualValues(1, queued)
	toShip, reason := s.shipping(refused)
	s.True(toShip)
	s.Empty(reason)
	toShip, _ = s.shipping(shipped)
	s.False(toShip, "the service has it")
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

// waiting writes a row the sweeper has not shipped, of app, with no organization unless change
// gives it one.
func (s *StoreSuite) waiting(app string, startedAt time.Time, change func(*Request)) {
	request := &Request{UsageID: uuid.Must(uuid.NewV7()), Modality: "knowledge", CustomerID: app, APIKey: "job:crawler",
		Provider: "turbopuffer", Model: "write", StartedAt: startedAt, Success: true, CostMicros: 10, ToShip: true}
	if change != nil {
		change(request)
	}
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

func (s *StoreSuite) TestUnownedUsageIsTheWaitingRowsOfAppsWithNoOrganization() {
	s.waiting("app", s.base.Add(time.Minute), nil)
	s.waiting("app", s.base, func(r *Request) { r.CostMicros = 5 })
	s.waiting("app", s.base.Add(2*time.Minute), func(r *Request) {
		r.APIKey, r.Modality, r.Provider, r.Model, r.CostMicros = "key-1", "llm", "openai", "gpt", 7
	})
	s.waiting("app", s.base.Add(-time.Hour), func(r *Request) { r.ToShip = false })
	s.waiting("app", s.base.Add(-time.Hour), func(r *Request) { r.OrganizationID = "org" })
	s.waiting("joined", s.base, nil)
	s.Require().NoError(s.store.JoinOrganization(s.ctx, "joined", "org"))

	unowned, err := s.store.UnownedUsage(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(unowned, 1, "an app that has joined one ships at the next sweep")
	s.Equal("app", unowned[0].App)
	s.EqualValues(3, unowned[0].Rows)
	s.EqualValues(22, unowned[0].CostMicros)
	s.True(s.base.Equal(unowned[0].OldestStartedAt), "%v", unowned[0].OldestStartedAt)
	s.Equal([]UnownedGroup{
		{APIKey: "job:crawler", Modality: "knowledge", Provider: "turbopuffer", Model: "write", Rows: 2, CostMicros: 15},
		{APIKey: "key-1", Modality: "llm", Provider: "openai", Model: "gpt", Rows: 1, CostMicros: 7},
	}, unowned[0].Groups)
}

// The groups are ordered by rows, so the oldest row can be in any of them.
func (s *StoreSuite) TestUnownedUsageStartsAtTheOldestRowOfAnyGroup() {
	s.waiting("app", s.base, nil)
	s.waiting("app", s.base, nil)
	s.waiting("app", s.base.Add(-time.Minute), func(r *Request) { r.APIKey = "key-1" })

	unowned, err := s.store.UnownedUsage(s.ctx)

	s.Require().NoError(err)
	s.Require().Len(unowned, 1)
	s.Equal("job:crawler", unowned[0].Groups[0].APIKey, "most rows first")
	s.True(s.base.Add(-time.Minute).Equal(unowned[0].OldestStartedAt), "%v", unowned[0].OldestStartedAt)
}
