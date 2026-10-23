package alerts

import (
	"bytes"
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/getsentry/sentry-go"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/suite"
)

// AlertsSuite sends alerts through a real Sentry client with no DSN, whose BeforeSend keeps
// each event and drops it.
type AlertsSuite struct {
	suite.Suite
	hub  *sentry.Hub
	logs *bytes.Buffer

	mu     sync.Mutex
	events []*sentry.Event
}

func TestAlertsSuite(t *testing.T) {
	suite.Run(t, new(AlertsSuite))
}

func (s *AlertsSuite) SetupTest() {
	s.events = nil
	client, err := sentry.NewClient(sentry.ClientOptions{
		BeforeSend: func(event *sentry.Event, _ *sentry.EventHint) *sentry.Event {
			s.mu.Lock()
			defer s.mu.Unlock()
			s.events = append(s.events, event)
			return nil
		},
	})
	s.Require().NoError(err)
	s.hub = sentry.NewHub(client, sentry.NewScope())
	s.logs = &bytes.Buffer{}
}

func (s *AlertsSuite) sent() []*sentry.Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]*sentry.Event(nil), s.events...)
}

func (s *AlertsSuite) loop(every, limit time.Duration, conditions ...Condition) *Loop {
	return New(Options{
		Hub: s.hub, Logger: slog.New(slog.NewTextHandler(s.logs, nil)), Source: "usage/test",
		Tags: map[string]string{"shard": "here"}, Every: every, Limit: limit,
	}, conditions...)
}

func (s *AlertsSuite) run(loop *Loop, d time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	loop.Run(ctx)
}

func (s *AlertsSuite) TestAConditionIsSentAtMostOncePerInterval() {
	started := time.Now()
	held := Condition{Name: "lost", Level: Page, Check: func(now time.Time) (bool, map[string]any) {
		return now.Sub(started) < 900*time.Millisecond, map[string]any{"rows": 3}
	}}

	s.run(s.loop(50*time.Millisecond, time.Second, held), 1200*time.Millisecond)

	events := s.sent()
	s.Require().Len(events, 1)
	s.Equal(sentry.LevelFatal, events[0].Level)
	s.Equal([]string{"usage_alert:lost"}, events[0].Fingerprint)
	s.Equal(map[string]string{"usage_alert": "lost", "source": "usage/test", "shard": "here"}, events[0].Tags)
	s.Equal(sentry.Context{"rows": 3}, events[0].Contexts["usage_alert"])
}

func (s *AlertsSuite) TestAConditionHeldPastTheLimitIsSentAgain() {
	held := Condition{Name: "rejected", Level: Warn, Check: func(time.Time) (bool, map[string]any) { return true, nil }}

	s.run(s.loop(10*time.Millisecond, 150*time.Millisecond, held), 400*time.Millisecond)

	s.Len(s.sent(), 3)
	s.Equal(sentry.LevelError, s.sent()[0].Level)
}

func (s *AlertsSuite) TestWithNoDSNAnAlertIsLogged() {
	held := Condition{Name: "ingest_failing", Level: Page, Check: func(time.Time) (bool, map[string]any) {
		return true, map[string]any{"increase": 4}
	}}

	s.run(s.loop(time.Hour, time.Minute, held), 50*time.Millisecond)

	s.Contains(s.logs.String(), "level=ERROR msg=\"usage alert\" alert=ingest_failing level=page source=usage/test")
	s.Contains(s.logs.String(), "increase:4")
}

func (s *AlertsSuite) TestAConditionThatDoesNotHoldSendsNothing() {
	quiet := Condition{Name: "lost", Level: Page, Check: func(time.Time) (bool, map[string]any) { return false, nil }}

	s.run(s.loop(10*time.Millisecond, time.Millisecond, quiet), 100*time.Millisecond)

	s.Empty(s.sent())
	s.Empty(s.logs.String())
}

func (s *AlertsSuite) TestIncreasedHoldsOnlyWhenTheCounterRose() {
	counter := prometheus.NewCounterVec(prometheus.CounterOpts{Name: "test_rejected_total"}, []string{"reason"})
	check := Increased(func() float64 { return Total(counter) })
	now := time.Now()

	holds, _ := check(now)
	s.False(holds)
	counter.WithLabelValues("missing_api_key").Inc()
	counter.WithLabelValues("missing_tier").Inc()
	holds, extra := check(now)
	s.True(holds)
	s.Equal(map[string]any{"value": 2.0, "increase": 2.0}, extra)
	holds, _ = check(now)
	s.False(holds)
}

func (s *AlertsSuite) TestIncreasingForNeedsBothHalvesOfTheSpan() {
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "test_errors_total"})
	check := IncreasingFor(func() float64 { return Total(counter) }, 2*time.Minute)
	start := time.Now()

	counter.Inc()
	holds, _ := check(start)
	s.False(holds, "one error is not a failure that keeps happening")
	holds, _ = check(start.Add(time.Minute))
	s.False(holds, "nothing in the second minute")
	counter.Inc()
	holds, _ = check(start.Add(2 * time.Minute))
	s.False(holds, "nothing in the minute before")
	counter.Inc()
	holds, _ = check(start.Add(3 * time.Minute))
	s.True(holds)
}

func (s *AlertsSuite) TestShareAboveReadsThePercentileOverTheSpan() {
	lag := prometheus.NewHistogram(prometheus.HistogramOpts{Name: "test_lag_seconds", Buckets: []float64{60, 600}})
	check := ShareAbove(lag, 600, 0.01, 10*time.Minute)
	start := time.Now()

	for range 99 {
		lag.Observe(1)
	}
	lag.Observe(700)
	holds, _ := check(start)
	s.False(holds, "one in a hundred is the 99th percentile, not above it")
	lag.Observe(700)
	holds, extra := check(start.Add(time.Minute))
	s.True(holds)
	s.Equal(2.0, extra["above"])

	for range 1000 {
		lag.Observe(1)
	}
	holds, _ = check(start.Add(12 * time.Minute))
	s.False(holds, "the slow ones are older than the span")
}
