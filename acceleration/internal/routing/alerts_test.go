package routing

import (
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
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

func (s *UsageAlertsSuite) TestTheRouterHasItsSixConditions() {
	levels := map[string]alerts.Level{}
	for _, condition := range Alerts(time.Minute) {
		levels[condition.Name] = condition.Level
	}

	s.Equal(map[string]alerts.Level{
		"lost": alerts.Page, "rejected": alerts.Page, "write_failing": alerts.Page, "ship_stale": alerts.Page,
		"unowned": alerts.Warn, "pending_slow": alerts.Warn,
	}, levels)
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
}
