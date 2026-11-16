//go:build integration

package store

import (
	"time"

	"github.com/google/uuid"
)

// orgSpend writes one request to store's shard, started at at, for organization and costing cost.
func (s *StoreSuite) orgSpend(store *Store, organization string, at time.Time, cost int64, change ...func(*Request)) {
	request := &Request{
		Modality: "llm", CustomerID: "app-" + organization, OrganizationID: organization, Provider: "openai", Model: "m",
		StartedAt: at, CostMicros: cost, Billable: true, Success: true,
	}
	for _, apply := range change {
		apply(request)
	}
	s.Require().NoError(store.RecordRequest(s.ctx, request))
}

func (s *StoreSuite) TestDailySpendIsEachOrganizationsBillableSpendPerUTCDay() {
	day := time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC)
	s.orgSpend(s.store, "org-a", day, 300)
	s.orgSpend(s.store, "org-a", day.Add(23*time.Hour+59*time.Minute), 200)
	s.orgSpend(s.store, "org-a", day.Add(24*time.Hour), 7)
	s.orgSpend(s.store, "org-a", day.Add(2*time.Hour), 1000, func(r *Request) { r.Billable = false })
	s.orgSpend(s.store, "org-a", day.Add(-time.Second), 50)
	s.orgSpend(s.store, "org-b", day.Add(3*time.Hour), 40)
	s.orgSpend(s.store, "", day.Add(3*time.Hour), 9000, func(r *Request) { r.CustomerID = "app-unowned" })
	s.orgSpend(s.store, "org-a", day.Add(48*time.Hour), 11)

	spent, err := s.store.DailySpend(s.ctx, day, day.AddDate(0, 0, 2))

	s.Require().NoError(err)
	s.Equal([]DaySpend{
		{OrganizationID: "org-a", Day: day, CostMicros: 500},
		{OrganizationID: "org-a", Day: day.AddDate(0, 0, 1), CostMicros: 7},
		{OrganizationID: "org-b", Day: day, CostMicros: 40},
	}, spent, "free work, a row with no organization and rows outside the range are left out; the first and "+
		"last minutes of a day are that day's")
}

// A day is the UTC day whatever time zone the database session is in: without AT TIME ZONE 'UTC'
// a session in Los Angeles files a request made at 03:00 UTC under the day before.
func (s *StoreSuite) TestDailySpendIsUTCWhateverTheSessionTimeZone() {
	day := time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC)
	la, err := Open(s.dsn + "&timezone=America/Los_Angeles")
	s.Require().NoError(err)
	defer la.Close()
	var zone string
	s.Require().NoError(la.DB().QueryRowContext(s.ctx, "SHOW timezone").Scan(&zone))
	s.Require().Equal("America/Los_Angeles", zone, "the session is not in UTC")
	s.orgSpend(la, "org-tz", day.Add(3*time.Hour), 300)

	spent, err := la.DailySpend(s.ctx, day, day.AddDate(0, 0, 1))

	s.Require().NoError(err)
	s.Equal([]DaySpend{{OrganizationID: "org-tz", Day: day, CostMicros: 300}}, spent)
}

func (s *ShardedStoreSuite) TestDailySpendSumsEveryShard() {
	day := time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC)
	s.orgSpend(s.store, "org-a", day.Add(time.Hour), 300)
	s.orgSpend(s.other, "org-a", day.Add(2*time.Hour), 200)

	spent, err := s.store.DailySpend(s.ctx, day, day.AddDate(0, 0, 1))

	s.Require().NoError(err)
	s.Equal([]DaySpend{{OrganizationID: "org-a", Day: day, CostMicros: 500}}, spent)
}

func (s *StoreSuite) costAnomaly(organization string, day time.Time) CostAnomaly {
	return CostAnomaly{OrganizationID: organization, Day: day, SpentMicros: 500, BaselineMicros: 100}
}

func (s *StoreSuite) TestACostAnomalyIsWrittenOncePerOrganizationAndDay() {
	day := time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC)
	organization := "org-" + uuid.NewString()

	written, err := s.store.NoteCostAnomaly(s.ctx, s.costAnomaly(organization, day))
	s.Require().NoError(err)
	s.True(written)
	written, err = s.store.NoteCostAnomaly(s.ctx, s.costAnomaly(organization, day))
	s.Require().NoError(err)
	s.False(written, "the same day again")
	written, err = s.store.NoteCostAnomaly(s.ctx, s.costAnomaly(organization, day.AddDate(0, 0, 1)))
	s.Require().NoError(err)
	s.True(written, "the next day is its own")
	written, err = s.store.NoteCostAnomaly(s.ctx, s.costAnomaly("org-"+uuid.NewString(), day))
	s.Require().NoError(err)
	s.True(written, "another organization the same day is its own")

	_, err = s.store.NoteCostAnomaly(s.ctx, s.costAnomaly("", day))
	s.ErrorContains(err, "needs an organization")
}

func (s *StoreSuite) TestAnUnpostedCostAnomalyIsClaimedOnceItsClaimIsStale() {
	day := time.Date(2026, 3, 10, 0, 0, 0, 0, time.UTC)
	organization := "org-" + uuid.NewString()
	_, err := s.store.NoteCostAnomaly(s.ctx, s.costAnomaly(organization, day))
	s.Require().NoError(err)
	backdate := func() {
		_, err := s.store.DB().NewUpdate().Model((*CostAnomaly)(nil)).Set("claimed_at = now() - interval '2 hours'").
			Where("organization_id = ?", organization).Exec(s.ctx)
		s.Require().NoError(err)
	}
	claimedOrganizations := func() []string {
		claimed, err := s.store.ClaimUnpostedCostAnomalies(s.ctx, time.Hour, 100)
		s.Require().NoError(err)
		var names []string
		for _, anomaly := range claimed {
			names = append(names, anomaly.OrganizationID)
			if anomaly.OrganizationID == organization {
				s.Equal(day, anomaly.Day.UTC())
				s.EqualValues(500, anomaly.SpentMicros)
				s.EqualValues(100, anomaly.BaselineMicros)
			}
		}
		return names
	}

	s.NotContains(claimedOrganizations(), organization, "its writer's claim stands")
	backdate()
	s.Contains(claimedOrganizations(), organization)
	s.NotContains(claimedOrganizations(), organization, "claimed again just now")

	_, err = s.store.NoteCostAnomaly(s.ctx, s.costAnomaly(organization, day.AddDate(0, 0, -1)))
	s.Require().NoError(err)
	s.Require().NoError(s.store.MarkCostAnomalyPosted(s.ctx, organization, day))
	backdate()
	claimed, err := s.store.ClaimUnpostedCostAnomalies(s.ctx, time.Hour, 100)
	s.Require().NoError(err)
	s.Require().Len(claimed, 1, "posted, and the day before is not")
	s.Equal(day.AddDate(0, 0, -1), claimed[0].Day.UTC())
}

func (s *StoreSuite) TestOneRouterHoldsTheCostAnomalyLock() {
	holder, other, ship := s.store.CostAnomalyLock(), s.store.CostAnomalyLock(), s.store.ShipLock()
	defer holder.Release()
	defer other.Release()
	defer ship.Release()
	held, err := ship.Hold(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)

	held, err = holder.Hold(s.ctx)
	s.Require().NoError(err)
	s.True(held, "the ship lock is another lock")
	held, err = other.Hold(s.ctx)
	s.Require().NoError(err)
	s.False(held)

	holder.Release()
	held, err = other.Hold(s.ctx)
	s.Require().NoError(err)
	s.True(held, "given back")
}

func (s *ShardedStoreSuite) TestOneRouterOfEveryShardHoldsTheCostAnomalyLock() {
	here, there := s.store.CostAnomalyLock(), s.other.CostAnomalyLock()
	defer here.Release()
	defer there.Release()

	held, err := here.Hold(s.ctx)
	s.Require().NoError(err)
	s.True(held)
	held, err = there.Hold(s.ctx)
	s.Require().NoError(err)
	s.False(held, "a router of another shard checks the same organizations")
}
