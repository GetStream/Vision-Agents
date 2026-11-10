//go:build integration

package store

import (
	"database/sql"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"
)

// activityRow is one customer_activity row as it reads.
type activityRow struct {
	organizationID string
	requests       int64
	refused        int64
	code           sql.NullString
	refusedAt      sql.NullTime
	firstAt        time.Time
	lastAt         time.Time
}

// activityOf reads the row of app, area and client on the day of s.base.
func (s *StoreSuite) activityOf(appID, area, client string) activityRow {
	var row activityRow
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx, `
SELECT organization_id, requests, refused, last_refusal_code, last_refusal_at, first_at, last_at
FROM customer_activity WHERE app_id = ? AND area = ? AND client = ? AND day = ?::date`,
		appID, area, client, s.base.Format(time.DateOnly)).
		Scan(&row.organizationID, &row.requests, &row.refused, &row.code, &row.refusedAt, &row.firstAt, &row.lastAt))
	return row
}

// activityRows is how many rows app has.
func (s *StoreSuite) activityRows(appID string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM customer_activity WHERE app_id = ?", appID).Scan(&count))
	return count
}

// served is a delta of requests answered at at, on s.base's day, for app's agent setup from the CLI.
func (s *StoreSuite) served(appID string, requests int64, at time.Time) ActivityDelta {
	return ActivityDelta{
		OrganizationID: "org", AppID: appID, Area: "agent_setup", Client: AuditSourceCLI,
		Day: s.base, Requests: requests, FirstAt: at, LastAt: at,
	}
}

// refusedWith is served with one of the requests refused at at with code.
func (s *StoreSuite) refusedWith(appID, code string, at time.Time) ActivityDelta {
	delta := s.served(appID, 1, at)
	delta.Refused, delta.LastRefusalCode, delta.LastRefusalAt = 1, code, at
	return delta
}

// Defect: an upsert that sets the count it was given (requests = EXCLUDED.requests) rather than
// adding it, so each minute's write replaces the day's count.
// The organization is the latest write's.
func (s *StoreSuite) TestTwoWritesForOneRowAreSummed() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.refusedWith("app", "forbidden", s.base)}))
	later := s.served("app", 2, s.base.Add(time.Minute))
	later.OrganizationID = "moved"
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{later}))

	row := s.activityOf("app", "agent_setup", AuditSourceCLI)
	s.Equal(int64(3), row.requests)
	s.Equal(int64(1), row.refused)
	s.Equal("moved", row.organizationID)
}

// Defect: an empty refusal code stored as an empty string rather than NULL, so a row never refused reads as
// refused with a code of nothing.
func (s *StoreSuite) TestARowNeverRefusedHoldsNoRefusalCode() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.served("never-refused", 1, s.base)}))

	row := s.activityOf("never-refused", "agent_setup", AuditSourceCLI)
	s.False(row.code.Valid, "code %q", row.code.String)
}

// Defect: deltas for one row written as two rows of one statement, which Postgres refuses ("ON
// CONFLICT DO UPDATE command cannot affect row a second time"), losing the whole write.
func (s *StoreSuite) TestDeltasForOneRowInOneWriteAreSummed() {
	deltas := []ActivityDelta{
		s.served("app", 1, s.base.Add(time.Minute)), s.refusedWith("app", "forbidden", s.base.Add(2*time.Minute)),
		s.refusedWith("app", "earlier", s.base), s.served("app", 1, s.base.Add(time.Minute)),
	}
	deltas[3].OrganizationID = "moved"
	s.Require().NoError(s.store.RecordActivity(s.ctx, deltas))

	row := s.activityOf("app", "agent_setup", AuditSourceCLI)
	s.Equal(int64(4), row.requests)
	s.Equal(int64(2), row.refused)
	s.Equal("forbidden", row.code.String, "the latest refusal is the last one, whatever order it came in")
	s.True(row.firstAt.Equal(s.base), row.firstAt)
	s.True(row.lastAt.Equal(s.base.Add(2*time.Minute)), row.lastAt)
	s.Equal("moved", row.organizationID, "the organization is the last delta's")
}

// Defect: a refusal written later but made earlier, by a replica whose minute ended after
// another's, replacing the latest one; or a write with no refusal clearing it.
func (s *StoreSuite) TestOnlyALaterRefusalReplacesTheLastOne() {
	// A row that has known no refusal takes the first one.
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.served("app", 1, s.base)}))
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.refusedWith("app", "second", s.base.Add(2*time.Minute))}))
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.refusedWith("app", "first", s.base.Add(time.Minute))}))
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.served("app", 1, s.base.Add(3*time.Minute))}))

	row := s.activityOf("app", "agent_setup", AuditSourceCLI)
	s.Equal("second", row.code.String)
	s.True(row.refusedAt.Time.Equal(s.base.Add(2*time.Minute)), row.refusedAt.Time)

	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.refusedWith("app", "third", s.base.Add(4*time.Minute))}))
	row = s.activityOf("app", "agent_setup", AuditSourceCLI)
	s.Equal("third", row.code.String)
	s.True(row.refusedAt.Time.Equal(s.base.Add(4*time.Minute)), row.refusedAt.Time)
}

// Defect: the first and last times taken from the latest write, so a replica writing an earlier
// minute late moves the first request forward or the last one back.
func (s *StoreSuite) TestTheFirstAndLastTimesOnlyWiden() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.served("app", 1, s.base.Add(time.Hour))}))
	early, late := s.served("app", 1, s.base), s.served("app", 1, s.base.Add(30*time.Minute))
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{early}))
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{late}))

	row := s.activityOf("app", "agent_setup", AuditSourceCLI)
	s.True(row.firstAt.Equal(s.base), row.firstAt)
	s.True(row.lastAt.Equal(s.base.Add(time.Hour)), row.lastAt)
	s.False(row.refusedAt.Valid, "nothing was refused")
}

// Defect: a key missing one of app, area, client and day, which sums them into one row.
func (s *StoreSuite) TestEachAppAreaClientAndDayIsARowOfItsOwn() {
	delta := s.served("app", 1, s.base)
	otherArea, otherClient, otherDay, otherApp := delta, delta, delta, delta
	otherArea.Area = "data"
	otherClient.Client = AuditSourceDashboard
	otherDay.Day = s.base.AddDate(0, 0, 1)
	otherApp.AppID = "another"

	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{delta, otherArea, otherClient, otherDay, otherApp}))

	s.Equal(4, s.activityRows("app"))
	s.Equal(1, s.activityRows("another"))
	s.Equal(int64(1), s.activityOf("app", "agent_setup", AuditSourceCLI).requests)
}

// Defect: rows past the first statement's thousand dropped, or one written twice, at the
// boundary between two statements.
func (s *StoreSuite) TestAWriteOfMoreRowsThanOneStatementTakesWritesEveryOne() {
	deltas := make([]ActivityDelta, 0, activityBatch+1)
	for i := range activityBatch + 1 {
		deltas = append(deltas, s.served(fmt.Sprintf("app-%d", i), 1, s.base))
	}

	s.Require().NoError(s.store.RecordActivity(s.ctx, deltas))

	var rows, requests int64
	s.Require().NoError(s.store.DB().QueryRowContext(s.ctx,
		"SELECT count(*), sum(requests) FROM customer_activity").Scan(&rows, &requests))
	s.Equal(int64(activityBatch+1), rows)
	s.Equal(int64(activityBatch+1), requests)
}

// Defect: the statements of one write committed one by one, so a write that failed in its
// second statement leaves the first's rows, which the recorder then writes again.
func (s *StoreSuite) TestAWriteThatFailsWritesNothing() {
	// A row whose count one more request takes past what a BIGINT holds, which Postgres
	// refuses. It sorts after every other row of the write, into its second statement.
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{s.served("app-9999", math.MaxInt64, s.base)}))
	deltas := make([]ActivityDelta, 0, activityBatch+1)
	for i := range activityBatch {
		deltas = append(deltas, s.served(fmt.Sprintf("app-%04d", i), 1, s.base))
	}
	deltas = append(deltas, s.served("app-9999", 1, s.base))

	s.Require().Error(s.store.RecordActivity(s.ctx, deltas))

	s.Equal(0, s.activityRows("app-0000"))
}

// Defect: two replicas writing the same rows in different orders, each holding a row the other
// waits for, which Postgres ends by failing one of them ("deadlock detected").
func (s *StoreSuite) TestTwoWritesOfTheSameRowsAtOnceBothSucceed() {
	forward := make([]ActivityDelta, 0, 200)
	for i := range 200 {
		forward = append(forward, s.served(fmt.Sprintf("app-%03d", i), 1, s.base))
	}
	backward := slices.Clone(forward)
	slices.Reverse(backward)

	const rounds = 5
	failures := make(chan error, 2*rounds)
	var wait sync.WaitGroup
	for range rounds {
		for _, deltas := range [][]ActivityDelta{forward, backward} {
			wait.Go(func() { failures <- s.store.RecordActivity(s.ctx, deltas) })
		}
	}
	wait.Wait()
	close(failures)

	for err := range failures {
		s.NoError(err)
	}
	s.Equal(int64(2*rounds), s.activityOf("app-000", "agent_setup", AuditSourceCLI).requests)
}

// day is a delta of requests in area from client for app on the day of at, refused of them.
func (s *StoreSuite) day(appID, area, client string, at time.Time, requests, refused int64) ActivityDelta {
	day := at.UTC().Truncate(24 * time.Hour)
	delta := ActivityDelta{
		OrganizationID: "org", AppID: appID, Area: area, Client: client,
		Day: day, Requests: requests, Refused: refused, FirstAt: at, LastAt: at,
	}
	if refused > 0 {
		delta.LastRefusalCode, delta.LastRefusalAt = "refused-"+at.Format(time.DateOnly), at
	}
	return delta
}

// Defect: an area read for one app of the organization, or for every app (a lost IN), or one
// row per day rather than one per area and client.
func (s *StoreSuite) TestActivityAreasSumTheAppsAskedForOverDaysAndLeaveOtherAppsOut() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{
		s.day("one", "agent_setup", AuditSourceCLI, s.base.Add(time.Minute), 2, 0),
		s.day("one", "agent_setup", AuditSourceCLI, s.base.Add(25*time.Hour), 3, 0),
		s.day("two", "agent_setup", AuditSourceCLI, s.base, 1, 0),
		s.day("two", "conversations", AuditSourceSDK, s.base.Add(time.Hour), 4, 0),
		s.day("stranger", "agent_setup", AuditSourceCLI, s.base, 100, 0),
		s.day("stranger", "workers", AuditSourceAPI, s.base, 100, 0),
	}))

	areas, err := s.store.ActivityAreas(s.ctx, []string{"one", "two"}, s.base)
	s.Require().NoError(err)

	s.Require().Len(areas, 2, "%+v", areas)
	s.Equal("agent_setup", areas[0].Area)
	s.Equal(AuditSourceCLI, areas[0].Client)
	s.Equal(int64(6), areas[0].Requests)
	s.True(s.base.Equal(areas[0].FirstAt), "the first of either app, %s", areas[0].FirstAt)
	s.True(s.base.Add(25*time.Hour).Equal(areas[0].LastAt), "the last of any day, %s", areas[0].LastAt)
	s.Equal("conversations", areas[1].Area)
	s.Equal(AuditSourceSDK, areas[1].Client)
	s.Equal(int64(4), areas[1].Requests)
}

// Defect: the days before since read too, or the day since falls on left out because its row
// began before since.
func (s *StoreSuite) TestActivityAreasReadTheWholeDaySinceFallsOnAndNothingBefore() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{
		s.day("one", "agent_setup", AuditSourceCLI, s.base.Add(-24*time.Hour), 7, 0),
		s.day("one", "agent_setup", AuditSourceCLI, s.base, 2, 0),
	}))

	areas, err := s.store.ActivityAreas(s.ctx, []string{"one"}, s.base.Add(3*time.Hour))
	s.Require().NoError(err)

	s.Require().Len(areas, 1)
	s.Equal(int64(2), areas[0].Requests)
}

// Defect: the code of the earliest refusal kept, or of a later day with none (its NULL sorting
// first in a descending order).
func (s *StoreSuite) TestTheCodeOfAnAreaIsItsLatestRefusals() {
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{
		s.day("one", "agent_setup", AuditSourceCLI, s.base, 1, 1),
		s.day("one", "agent_setup", AuditSourceCLI, s.base.Add(24*time.Hour), 2, 1),
		s.day("one", "agent_setup", AuditSourceCLI, s.base.Add(48*time.Hour), 5, 0),
		s.day("one", "conversations", AuditSourceCLI, s.base, 3, 0),
	}))

	areas, err := s.store.ActivityAreas(s.ctx, []string{"one"}, s.base)
	s.Require().NoError(err)

	s.Require().Len(areas, 2)
	s.Equal(int64(8), areas[0].Requests)
	s.Equal(int64(2), areas[0].Refused)
	s.Equal("refused-"+s.base.Add(24*time.Hour).Format(time.DateOnly), areas[0].LastRefusalCode)
	s.Empty(areas[1].LastRefusalCode)
}

// Defect: a day counted twice for two areas, a day of refusals only counted, or a day of another
// area, another app or before since counted.
func (s *StoreSuite) TestActivityDaysCountEachDayAnAreaNamedAnsweredOnce() {
	day := func(n int) time.Time { return s.base.Add(time.Duration(n) * 24 * time.Hour) }
	s.Require().NoError(s.store.RecordActivity(s.ctx, []ActivityDelta{
		s.day("one", "conversations", AuditSourceSDK, day(0), 1, 0),
		s.day("one", "model_api", AuditSourceSDK, day(0), 1, 0),
		s.day("two", "model_api", AuditSourceAPI, day(1), 3, 1),
		s.day("one", "conversations", AuditSourceSDK, day(2), 2, 2),
		s.day("one", "agent_setup", AuditSourceCLI, day(3), 1, 0),
		s.day("stranger", "conversations", AuditSourceSDK, day(4), 1, 0),
		s.day("one", "conversations", AuditSourceSDK, day(-1), 1, 0),
	}))

	days, err := s.store.ActivityDays(s.ctx, []string{"one", "two"}, s.base, []string{"conversations", "model_api"})
	s.Require().NoError(err)
	s.Equal(2, days)
}

// usedBy is a request of app's of modality at at, its quantities set by change.
func (s *StoreSuite) usedBy(appID, modality string, at time.Time, change func(*Request)) {
	request := &Request{Modality: modality, CustomerID: appID, Provider: "stub", Model: "stub", StartedAt: at, Success: true}
	change(request)
	s.Require().NoError(s.store.RecordRequest(s.ctx, request))
}

// Defect: a meter summed for one app of the organization or for every app, a quantity left
// out or summed from another column, a request before since counted, or the sliced time of
// runtime and sandboxes counted as requests.
func (s *StoreSuite) TestMeterUsesSumTheAppsAskedForByModality() {
	took := func(ms float64) *float64 { return &ms }
	s.usedBy("one", "llm", s.base, func(r *Request) { r.InputTokens, r.OutputTokens, r.CostMicros = 10, 20, 5 })
	s.usedBy("two", "llm", s.base.Add(time.Hour), func(r *Request) { r.InputTokens, r.OutputTokens, r.CostMicros = 1, 2, 1 })
	s.usedBy("one", "tts", s.base, func(r *Request) { r.AudioMs, r.Characters, r.CostMicros = 1500, 42, 3 })
	s.usedBy("two", "sandbox", s.base, func(r *Request) { r.DurationMs, r.CostMicros = took(60_000), 7 })
	s.usedBy("one", "runtime", s.base, func(r *Request) { r.DurationMs, r.CostMicros = took(30_000.4), 2 })
	s.usedBy("stranger", "llm", s.base, func(r *Request) { r.InputTokens, r.CostMicros = 1000, 1000 })
	s.usedBy("one", "llm", s.base.Add(-time.Second), func(r *Request) { r.InputTokens, r.CostMicros = 1000, 1000 })

	uses, err := s.store.MeterUses(s.ctx, []string{"one", "two"}, s.base)
	s.Require().NoError(err)

	s.Equal([]MeterUse{
		{Modality: "llm", Requests: 2, InputTokens: 11, OutputTokens: 22, CostMicros: 6},
		{Modality: "runtime", DurationMs: 30_000, CostMicros: 2},
		{Modality: "sandbox", DurationMs: 60_000, CostMicros: 7},
		{Modality: "tts", Requests: 1, AudioMs: 1500, Characters: 42, CostMicros: 3},
	}, uses)
}

// Defect: the requests of another shard left out, or a shard's answer replacing another's.
func (s *ShardedStoreSuite) TestMeterUsesSumEveryShard() {
	took := 1000.0
	s.usedBy("one", "llm", s.base, func(r *Request) {
		r.InputTokens, r.OutputTokens, r.AudioMs, r.Characters, r.DurationMs, r.CostMicros = 10, 20, 30, 40, &took, 5
	})
	request := &Request{Modality: "llm", CustomerID: "two", Provider: "stub", Model: "stub", StartedAt: s.base,
		InputTokens: 1, OutputTokens: 2, AudioMs: 3, Characters: 4, DurationMs: &took, CostMicros: 1, Success: true}
	s.Require().NoError(s.other.RecordRequest(s.ctx, request))

	uses, err := s.store.MeterUses(s.ctx, []string{"one", "two"}, s.base)
	s.Require().NoError(err)
	s.Equal([]MeterUse{{Modality: "llm", Requests: 2, InputTokens: 11, OutputTokens: 22, AudioMs: 33, Characters: 44,
		DurationMs: 2000, CostMicros: 6}}, uses)
}
