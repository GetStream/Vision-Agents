package routing

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

type UsageAlertsSuite struct {
	suite.Suite
}

func TestUsageAlertsSuite(t *testing.T) {
	suite.Run(t, new(UsageAlertsSuite))
}

// named is the router's alert condition named, for a loop sending one event of it a minute.
func (s *UsageAlertsSuite) named(name string) alerts.Condition {
	for _, condition := range Alerts(time.Minute) {
		if condition.Name == name {
			return condition
		}
	}
	s.FailNow("no router alert is named " + name)
	return alerts.Condition{}
}

func (s *UsageAlertsSuite) TestTheRouterHasItsEightConditions() {
	levels := map[string]alerts.Level{}
	for _, condition := range Alerts(time.Minute) {
		levels[condition.Name] = condition.Level
	}

	s.Equal(map[string]alerts.Level{
		"lost": alerts.Page, "rejected": alerts.Page, "write_failing": alerts.Page, "ship_stale": alerts.Page,
		"holds_flush_failing": alerts.Page, "unowned": alerts.Warn, "pending_slow": alerts.Warn, "unattributed": alerts.Warn,
	}, levels)
}

// The Slack message of an alert is its Sentry title: without a summary, or an item's
// description, it names only the alert.
func (s *UsageAlertsSuite) TestEveryConditionSaysWhatItMeans() {
	for _, condition := range Alerts(time.Minute) {
		if condition.Items != nil {
			s.NotEmpty(strings.TrimSpace(condition.Describe("app")), condition.Name)
			continue
		}
		s.NotEmpty(strings.TrimSpace(condition.Summary), condition.Name)
	}
}

// counted makes the apps the last count of a sweeper found waiting for an organization what
// the alerts read.
func (s *UsageAlertsSuite) counted(apps ...store.UnownedApp) {
	sweeper := &Sweeper{}
	sweeper.unowned.Store(&apps)
	sweepers.Store(sweeper, struct{}{})
	s.T().Cleanup(func() { sweepers.Delete(sweeper) })
}

func (s *UsageAlertsSuite) TestUnownedNamesEachAppWhoseRowsWaitedFifteenMinutes() {
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)
	written := []store.UnownedGroup{
		{APIKey: "job:crawler", Modality: "knowledge", Provider: "turbopuffer", Model: "write", Rows: 10, CostMicros: 300},
		{APIKey: "key-1", Modality: "llm", Provider: "openai", Model: "gpt", Rows: 2, CostMicros: 100},
	}
	s.counted(
		store.UnownedApp{App: "waiting", Rows: 12, CostMicros: 400, OldestStartedAt: now.Add(-26 * time.Hour), Groups: written},
		store.UnownedApp{App: "new", Rows: 1, CostMicros: 3, OldestStartedAt: now.Add(-14 * time.Minute)},
	)

	held := s.named("unowned").Items(now)

	s.Equal(map[string]map[string]any{"waiting": {
		"waiting_rows": int64(12), "cost_micros": int64(400), "oldest_started_at": "2026-11-18T10:00:00Z",
		"oldest_waiting": "26h0m0s", "written_by": written,
	}}, held, "an app is warned about once its oldest row has waited 15 minutes")
}

func (s *UsageAlertsSuite) TestUnownedWarnsFromTheFifteenthMinuteOn() {
	now := time.Date(2026, 11, 19, 12, 0, 0, 0, time.UTC)
	s.counted(
		store.UnownedApp{App: "fifteen", Rows: 1, OldestStartedAt: now.Add(-15 * time.Minute)},
		store.UnownedApp{App: "almost", Rows: 1, OldestStartedAt: now.Add(-15*time.Minute + time.Second)},
	)

	held := s.named("unowned").Items(now)

	s.Contains(held, "fifteen")
	s.NotContains(held, "almost")
}

// An app with many keys or models keeps a small event: the groups with most rows, and a count
// of the rest.
func (s *UsageAlertsSuite) TestUnownedSendsTheTenGroupsWithMostRowsAndSumsTheRest() {
	now := time.Now()
	var groups []store.UnownedGroup
	for i := range 13 {
		groups = append(groups, store.UnownedGroup{APIKey: fmt.Sprintf("key-%d", i), Modality: "llm", Provider: "openai",
			Model: "gpt", Rows: int64(20 - i), CostMicros: 5})
	}
	s.counted(store.UnownedApp{App: "busy", Rows: 221, CostMicros: 65, OldestStartedAt: now.Add(-time.Hour), Groups: groups})

	held := s.named("unowned").Items(now)["busy"]

	s.Equal(groups[:10], held["written_by"])
	s.Equal(3, held["other_groups"])
	s.Equal(int64(10+9+8), held["other_rows"])
}

// A sweeper registers before its first count, which may fail: the alerts loop must not stop.
func (s *UsageAlertsSuite) TestUnownedReadsNothingFromASweeperNotYetCounted() {
	sweeper := &Sweeper{}
	sweepers.Store(sweeper, struct{}{})
	s.T().Cleanup(func() { sweepers.Delete(sweeper) })

	s.Empty(s.named("unowned").Items(time.Now()))
}

// What a person who opens the alert checks first: the app's membership, then what wrote its rows.
func (s *UsageAlertsSuite) TestUnownedSaysWhyTheRowsWaitAndWhatToCheck() {
	described := s.named("unowned").Describe("app")
	s.Contains(described, "joined no organization")
	s.Contains(described, "not billed")
	s.Contains(described, "app_organizations")
	s.Contains(described, "api_key")
}

// A row lost is written back by replay and a refused one is queued again by reship: the summary
// must name the command, not say the work is gone or ships by itself.
func (s *UsageAlertsSuite) TestLostAndRejectedNameTheirWayBack() {
	s.Contains(s.named("lost").Summary, "router usage replay")
	s.Contains(s.named("rejected").Summary, "router usage reship --rejected")
}

// The two causes of a row waiting: Postgres slow, or writes failing and retried. The wait is
// the threshold the condition checks.
func (s *UsageAlertsSuite) TestPendingSlowNamesBothCausesAndItsThreshold() {
	summary := s.named("pending_slow").Summary
	s.Contains(summary, fmt.Sprintf("%d seconds", pendingSlowSeconds))
	s.Contains(summary, "Postgres is slow")
	s.Contains(summary, "writes are failing")
}

func (s *UsageAlertsSuite) TestHoldsFlushFailingPagesOncePastTenSeconds() {
	var unflushed atomic.Int64
	ReportUnflushedHolds(func() float64 { return float64(unflushed.Load()) })
	defer ReportUnflushedHolds(func() float64 { return 0 })
	failing := s.named("holds_flush_failing")
	start := time.Now()

	unflushed.Store(9)
	holds, _ := failing.Check(start)
	s.False(holds, "admission is still answered from memory")
	unflushed.Store(11)
	holds, extra := failing.Check(start.Add(15 * time.Second))
	s.True(holds, "billable work is being refused")
	s.Equal(float64(11), extra["unflushed_oldest_seconds"])
	unflushed.Store(0)
	holds, _ = failing.Check(start.Add(30 * time.Second))
	s.False(holds, "a flush landed")
}

func (s *UsageAlertsSuite) TestAnUnattributedRowWarnsOnce() {
	unattributed := s.named("unattributed")
	start := time.Now()
	usageUnattributed.Inc()

	holds, _ := unattributed.Check(start)
	s.Require().True(holds, "sent")
	for _, tick := range []time.Duration{15 * time.Second, time.Minute, 5 * time.Minute} {
		holds, _ = unattributed.Check(start.Add(tick))
		s.False(holds, tick)
	}
}

func (s *UsageAlertsSuite) TestARowLostJustAfterTheLostAlertStillHoldsAtTheFirstTickTheLimitAllows() {
	lost := s.named("lost")
	start := time.Now()
	usageLost.WithLabelValues(lostQueueFull).Inc()
	holds, _ := lost.Check(start)
	s.Require().True(holds, "sent")

	usageLost.WithLabelValues(lostQueueFull).Inc() // 5 s after the alert
	for _, tick := range []time.Duration{15*time.Second - 2*time.Microsecond, 30 * time.Second, 45 * time.Second, time.Minute - time.Microsecond} {
		lost.Check(start.Add(tick))
	}
	holds, _ = lost.Check(start.Add(75*time.Second - time.Microsecond))

	s.True(holds, "the first tick a minute after the alert still sees the second loss")
}

func (s *UsageAlertsSuite) TestOneLostRowHoldsTheLostAlertOnce() {
	lost := s.named("lost")
	start := time.Now()
	usageLost.WithLabelValues(lostQueueFull).Inc()

	holds, _ := lost.Check(start)
	s.Require().True(holds, "sent")
	for _, tick := range []time.Duration{15 * time.Second, 30 * time.Second, 45 * time.Second, time.Minute, 75 * time.Second, 5 * time.Minute} {
		holds, _ = lost.Check(start.Add(tick))
		s.False(holds, tick)
	}
}

func (s *UsageAlertsSuite) TestShipStalePagesOnlyOnceTheOldestRowHasStayedOldForTwoMinutes() {
	sweeper := &Sweeper{}
	sweepers.Store(sweeper, struct{}{})
	defer sweepers.Delete(sweeper)
	stale := s.named("ship_stale")
	start := time.Now()

	sweeper.oldest.Store(int64(14 * time.Minute))
	holds, _ := stale.Check(start)
	s.False(holds)
	sweeper.oldest.Store(int64(16 * time.Minute))
	holds, _ = stale.Check(start.Add(15 * time.Second))
	s.False(holds, "a long call written a moment ago is old at once, and shipped within seconds")
	sweeper.oldest.Store(int64(3 * time.Minute))
	holds, _ = stale.Check(start.Add(30 * time.Second))
	s.False(holds)
	sweeper.oldest.Store(int64(16 * time.Minute))
	stale.Check(start.Add(45 * time.Second))
	holds, extra := stale.Check(start.Add(3 * time.Minute))
	s.True(holds, "still past fifteen minutes two minutes later")
	s.Equal(float64(16*60), extra["oldest_unshipped_started_seconds_ago"])
}

func (s *UsageAlertsSuite) TestASourceNamesTheShardAndTheHost() {
	s.Equal("router/eu-1/router-7f9c", Source("eu-1", "router-7f9c"))
	s.Equal("router/unsharded/router-7f9c", Source("", "router-7f9c"))
	s.Equal("router/eu-1/unknown", Source("eu-1", ""))
}

func (s *UsageAlertsSuite) TestAJobSourceNamesTheShardAndTheHostUnderJob() {
	s.Equal("job/eu-1/usage-storage-29412", JobSource("eu-1", "usage-storage-29412"))
	s.Equal("job/unsharded/usage-storage-29412", JobSource("", "usage-storage-29412"))
	s.Equal("job/eu-1/unknown", JobSource("eu-1", ""))
}
