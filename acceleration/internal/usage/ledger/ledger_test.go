//go:build integration

package ledger_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// LedgerSuite records events into a database of its own, emptied before each test.
type LedgerSuite struct {
	suite.Suite
	ctx    context.Context
	dsn    string
	ledger *ledger.Ledger
	// now is the ledger's clock, which a test may move.
	mu  sync.Mutex
	now time.Time
}

func TestLedgerSuite(t *testing.T) {
	suite.Run(t, new(LedgerSuite))
}

func (s *LedgerSuite) SetupSuite() {
	s.ctx = context.Background()
	s.dsn = usagetest.Database(s.T(), "usage_ledger")
	opened, err := ledger.Open(ledger.Options{DSN: s.dsn, Now: s.clock})
	s.Require().NoError(err)
	s.ledger = opened
}

func (s *LedgerSuite) TearDownSuite() {
	if s.ledger != nil {
		s.Require().NoError(s.ledger.Close())
	}
}

func (s *LedgerSuite) SetupTest() {
	s.setNow(time.Now().UTC())
	usagetest.Empty(s.T(), s.ledger)
}

func (s *LedgerSuite) clock() time.Time {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.now
}

func (s *LedgerSuite) setNow(now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.now = now
}

// stored is every recorded event, oldest first.
func (s *LedgerSuite) stored() []ledger.UsageEvent {
	var rows []ledger.UsageEvent
	s.Require().NoError(s.ledger.DB().NewSelect().Model(&rows).Order("occurred_at", "usage_id").Scan(s.ctx))
	return rows
}

// windows is every window of organization, in key order.
func (s *LedgerSuite) windows(organization string) []ledger.UsageWindow {
	var rows []ledger.UsageWindow
	s.Require().NoError(s.ledger.DB().NewSelect().Model(&rows).Where("organization_id = ?", organization).
		Order("subject", "app_id", "api_key", "user_id", "category", "interval", "window_start").Scan(s.ctx))
	return rows
}

// window is the one window of a subject in an interval, which the test expects there to be.
func (s *LedgerSuite) window(ref usage.SubjectRef, interval usage.Interval) ledger.UsageWindow {
	var found []ledger.UsageWindow
	for _, window := range s.windows(ref.OrganizationID) {
		if window.Ref() == ref && window.Interval == interval {
			found = append(found, window)
		}
	}
	s.Require().Len(found, 1, "%+v %s", ref, interval)
	return found[0]
}

func (s *LedgerSuite) ingest(events ...usage.Event) ledger.Result {
	result, err := s.ledger.Ingest(s.ctx, events)
	s.Require().NoError(err)
	return result
}

func (s *LedgerSuite) TestAnEventIsRecordedOnce() {
	event := usagetest.Event("o1", func(e *usage.Event) {
		e.UserID, e.UsageCategories = "u1", []string{"team:finance", "project:x"}
		e.AgentID, e.CallID, e.SessionID, e.Purpose, e.PriceVersion = "agent", "call", "session", "reply", "20261110_seed"
		e.Attributes, e.Tags = map[string]string{"country": "US"}, map[string]string{"env": "prod"}
	})

	first := s.ingest(event)
	second := s.ingest(event)

	s.Len(first.Accepted, 1)
	s.Equal(usage.Ack{Accepted: 1, Duplicates: 0, Rejected: []usage.Rejection{}}, first.Ack())
	s.Equal(usage.Ack{Accepted: 0, Duplicates: 1, Rejected: []usage.Rejection{}}, second.Ack())
	rows := s.stored()
	s.Require().Len(rows, 1)
	stored := rows[0].Event()
	stored.OccurredAt = stored.OccurredAt.UTC()
	event.UsageCategories = []string{"project:x", "team:finance"}
	s.Equal(event, stored)
	s.WithinDuration(time.Now(), rows[0].ReceivedAt, time.Minute)
}

func (s *LedgerSuite) TestABatchSentTwiceIsRecordedOnce() {
	batch := []usage.Event{usagetest.Event("o1"), usagetest.Event("o1"), usagetest.Event("o1")}

	first := s.ingest(batch...)
	second := s.ingest(batch...)

	s.Equal(3, first.Ack().Accepted)
	s.Equal(usage.Ack{Accepted: 0, Duplicates: 3, Rejected: []usage.Rejection{}}, second.Ack())
	s.Empty(second.Windows, "a batch that recorded nothing adds to no window")
	s.Len(s.stored(), 3)
	daily := s.window(usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectOrganization}, usage.Daily)
	s.Equal(int64(3), daily.Events)
	s.Equal(int64(3000), daily.SpentMicros)
}

func (s *LedgerSuite) TestAUsageIDTwiceInOneBatchIsRecordedOnce() {
	event := usagetest.Event("o1")

	result := s.ingest(event, event)

	s.Equal(usage.Ack{Accepted: 1, Duplicates: 1, Rejected: []usage.Rejection{}}, result.Ack())
	s.Equal(int64(1), s.window(usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectOrganization}, usage.Monthly).Events)
}

func (s *LedgerSuite) TestOnlyBillableCostIsSummed() {
	custom := usagetest.Event("o1", func(e *usage.Event) { e.Billable, e.CostMicros, e.Model = false, 5000, "custom/mine" })

	s.ingest(custom, usagetest.Event("o1"))

	s.Len(s.stored(), 2)
	for _, window := range s.windows("o1") {
		s.Equal(int64(2), window.Events, "%+v", window)
		s.Equal(int64(1000), window.SpentMicros, "%+v", window)
	}
}

func (s *LedgerSuite) TestAnEventCountsForEverySubjectInEveryInterval() {
	event := usagetest.Event("o1", func(e *usage.Event) {
		e.AppID, e.APIKey, e.UserID, e.UsageCategories, e.CostMicros = "a1", "k1", "u1", []string{"a", "b"}, 1234
	})

	result := s.ingest(event)

	windows := s.windows("o1")
	s.Len(windows, 32)
	s.Len(result.Windows, 32)
	subjects := map[usage.SubjectRef]int{}
	for _, window := range windows {
		subjects[window.Ref()]++
		s.Equal(int64(1234), window.SpentMicros)
		s.Equal(int64(1), window.Events)
		start, end := usage.Window(window.Interval, event.OccurredAt)
		s.Equal(start, window.WindowStart.UTC())
		s.Equal(end, window.WindowEnd.UTC())
	}
	s.Equal(map[usage.SubjectRef]int{
		{OrganizationID: "o1", Subject: usage.SubjectOrganization}:                            4,
		{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"}:                        4,
		{OrganizationID: "o1", Subject: usage.SubjectAPIKey, AppID: "a1", APIKey: "k1"}:       4,
		{OrganizationID: "o1", Subject: usage.SubjectUser, AppID: "a1", UserID: "u1"}:         4,
		{OrganizationID: "o1", Subject: usage.SubjectCategory, Category: "a"}:                 4,
		{OrganizationID: "o1", Subject: usage.SubjectCategory, Category: "b"}:                 4,
		{OrganizationID: "o1", Subject: usage.SubjectAppCategory, AppID: "a1", Category: "a"}: 4,
		{OrganizationID: "o1", Subject: usage.SubjectAppCategory, AppID: "a1", Category: "b"}: 4,
	}, subjects)
}

func (s *LedgerSuite) TestAnEventWithNoUserOrCategoriesCountsForTwelveWindows() {
	s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.APIKey = "hook:chat" }))

	s.Len(s.windows("o1"), 12)
}

func (s *LedgerSuite) TestAnEventIsCountedInTheWindowItHappened() {
	// The last second of a Sunday that ends a month and a bi-weekly window, received in the
	// first second of the next day.
	happened := time.Date(2026, 5, 31, 23, 59, 59, 0, time.UTC)
	s.setNow(time.Date(2026, 6, 1, 0, 0, 1, 0, time.UTC))

	s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.OccurredAt = happened }))

	organization := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectOrganization}
	for interval, start := range map[usage.Interval]time.Time{
		usage.Daily:    time.Date(2026, 5, 31, 0, 0, 0, 0, time.UTC),
		usage.Weekly:   time.Date(2026, 5, 25, 0, 0, 0, 0, time.UTC),
		usage.Biweekly: time.Date(2026, 5, 18, 0, 0, 0, 0, time.UTC),
		usage.Monthly:  time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC),
	} {
		window := s.window(organization, interval)
		s.Equal(start, window.WindowStart.UTC(), interval)
		s.Equal(time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC), window.WindowEnd.UTC(), interval)
	}
}

func (s *LedgerSuite) TestOldWindowsArePruned() {
	s.setNow(time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC))
	s.ingest(
		usagetest.Event("o1", func(e *usage.Event) { e.OccurredAt = time.Date(2026, 8, 1, 9, 0, 0, 0, time.UTC) }),
		usagetest.Event("o1", func(e *usage.Event) { e.OccurredAt = time.Date(2026, 10, 22, 9, 0, 0, 0, time.UTC) }),
	)
	s.Len(s.windows("o1"), 24)

	deleted, err := s.ledger.PruneWindows(s.ctx, s.clock().Add(-ledger.Retention))

	s.Require().NoError(err)
	s.Equal(int64(12), deleted)
	for _, window := range s.windows("o1") {
		s.True(window.WindowEnd.After(time.Date(2026, 10, 1, 0, 0, 0, 0, time.UTC)), "%+v", window)
	}
	s.Len(s.windows("o1"), 12)
	s.Len(s.stored(), 2, "the events are kept")
}

func (s *LedgerSuite) TestARejectedEventIsNotRecordedAndTheRestAre() {
	keyless := usagetest.Event("o1", func(e *usage.Event) { e.APIKey = "" })

	result := s.ingest(usagetest.Event("o1"), keyless, usagetest.Event("o1"))

	s.Equal(2, result.Ack().Accepted)
	s.Equal([]usage.Rejection{{Index: 1, UsageID: keyless.UsageID, Reason: usage.ReasonAPIKey,
		Message: "api_key is required: the key the work was started with, or its origin"}}, result.Rejected)
	s.Len(s.stored(), 2)
}

func (s *LedgerSuite) TestCostsThatWouldOverflowTheSumAreRejected() {
	huge := func(e *usage.Event) { e.CostMicros = 4_600_000_000_000_000_000 }
	first, second := usagetest.Event("o1", huge), usagetest.Event("o1", huge)

	result := s.ingest(first, usagetest.Event("o1"), second)

	s.Equal(1, result.Ack().Accepted)
	s.Require().Len(result.Rejected, 2)
	s.Equal([]int{0, 2}, []int{result.Rejected[0].Index, result.Rejected[1].Index})
	s.Equal(usage.ReasonCostTooLarge, result.Rejected[0].Reason)
	for _, window := range s.windows("o1") {
		s.Equal(int64(1000), window.SpentMicros, "%+v", window)
	}
}

func (s *LedgerSuite) TestAWindowsTotalCannotBeNegative() {
	_, err := s.ledger.DB().ExecContext(s.ctx, `INSERT INTO usage_windows
		(organization_id, subject, interval, window_start, window_end, spent_micros)
		VALUES ('o1', 'organization', 'daily', now(), now(), -1)`)
	s.Require().ErrorContains(err, "usage_windows_spent_micros_check")

	_, err = s.ledger.DB().ExecContext(s.ctx, `INSERT INTO usage_windows
		(organization_id, subject, interval, window_start, window_end, events)
		VALUES ('o1', 'organization', 'daily', now(), now(), -1)`)
	s.Require().ErrorContains(err, "usage_windows_events_check")
}

func (s *LedgerSuite) TestWithoutAMirrorADuplicatesWindowsAreNotRead() {
	event := usagetest.Event("o1")
	s.ingest(event)

	again := s.ingest(event)

	s.Equal(1, again.Duplicates)
	s.Empty(again.Repaired)
	s.False(again.Mirrored)
}

func (s *LedgerSuite) TestABatchOfOnlyRejectedEventsTouchesNothing() {
	result := s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.UsageID = uuid.Nil }))

	s.Equal(usage.ReasonUsageID, result.Rejected[0].Reason)
	s.Empty(s.stored())
	s.Empty(s.windows("o1"))
}

func (s *LedgerSuite) TestTwoBatchesRacingOnOneOrganizationBothLand() {
	other, err := ledger.Open(ledger.Options{DSN: s.dsn})
	s.Require().NoError(err)
	defer other.Close() //nolint:errcheck // a pool the test is done with

	batch := func(first, second string) []usage.Event {
		events := make([]usage.Event, 0, 200)
		for i := range 200 {
			organization := first
			if i >= 100 {
				organization = second
			}
			events = append(events, usagetest.Event(organization, func(e *usage.Event) {
				e.AppID, e.UserID, e.UsageCategories = "a", "u", []string{"c1", "c2"}
			}))
		}
		return events
	}
	for range 5 {
		forwards, backwards := batch("o1", "o2"), batch("o2", "o1")
		var wg sync.WaitGroup
		errs := make([]error, 2)
		wg.Go(func() { _, errs[0] = s.ledger.Ingest(s.ctx, forwards) })
		wg.Go(func() { _, errs[1] = other.Ingest(s.ctx, backwards) })
		wg.Wait()
		s.Require().NoError(errs[0])
		s.Require().NoError(errs[1])
	}

	s.Len(s.stored(), 2000)
	for _, organization := range []string{"o1", "o2"} {
		daily := s.window(usage.SubjectRef{OrganizationID: organization, Subject: usage.SubjectAppCategory, AppID: "a", Category: "c2"}, usage.Daily)
		s.Equal(int64(1000), daily.Events)
		s.Equal(int64(1000*1000), daily.SpentMicros)
	}
}

func (s *LedgerSuite) TestCurrentAnswersEveryIntervalOfEverySubjectAskedAbout() {
	s.ingest(usagetest.Event("o1", func(e *usage.Event) { e.AppID = "a1" }))
	spent := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a1"}
	never := usage.SubjectRef{OrganizationID: "o1", Subject: usage.SubjectApp, AppID: "a2"}

	windows, err := s.ledger.Current(s.ctx, []usage.SubjectRef{spent, never})

	s.Require().NoError(err)
	s.Require().Len(windows, 8)
	for i, window := range windows {
		ref, interval := spent, usage.Intervals[i%4]
		if i >= 4 {
			ref = never
		}
		start, end := usage.Window(interval, s.clock())
		s.Equal(ref, window.Ref())
		s.Equal(interval, window.Interval)
		s.Equal(start, window.WindowStart.UTC())
		s.Equal(end, window.WindowEnd.UTC())
		if ref == spent {
			s.Equal(int64(1000), window.SpentMicros)
			s.Equal(int64(1), window.Events)
		} else {
			s.Zero(window.SpentMicros)
			s.Zero(window.Events)
		}
	}
}

func (s *LedgerSuite) TestMigratingAgainChangesNothing() {
	s.Require().NoError(s.ledger.Migrate(s.ctx))
	var version int64
	s.Require().NoError(s.ledger.DB().QueryRowContext(s.ctx, "SELECT max(version_id) FROM goose_db_version").Scan(&version))
	s.Equal(int64(3), version)
}
