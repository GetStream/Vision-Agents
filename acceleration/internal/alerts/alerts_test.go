package alerts

import (
	"bytes"
	"context"
	"log/slog"
	"maps"
	"slices"
	"sync"
	"sync/atomic"
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

func (s *AlertsSuite) TestAConditionThatHoldsAsTheLoopStopsIsSent() {
	ctx, cancel := context.WithCancel(context.Background())
	var holds atomic.Bool
	late := Condition{Name: "lost", Level: Page, Check: func(time.Time) (bool, map[string]any) { return holds.Load(), nil }}
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.loop(time.Hour, time.Minute, late).Run(ctx)
	}()
	time.Sleep(20 * time.Millisecond)

	holds.Store(true)
	cancel()
	<-done

	s.Len(s.sent(), 1, "what the process's last moments did is checked once more")
}

func (s *AlertsSuite) TestIncreasedSinceAlertStillHoldsAnIncreaseTheLimitHeldBack() {
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "test_lost_total"})
	check := IncreasedSinceAlert(func() float64 { return Total(counter) }, time.Minute)
	start := time.Now()

	holds, _ := check(start)
	s.False(holds)
	counter.Inc()
	holds, _ = check(start.Add(time.Second))
	s.True(holds, "sent")
	counter.Inc()
	holds, _ = check(start.Add(10 * time.Second))
	s.False(holds, "held back by the limit")
	holds, extra := check(start.Add(61 * time.Second))
	s.True(holds, "the second increase is sent once the limit passes")
	s.Equal(1.0, extra["increase"])
	holds, _ = check(start.Add(2 * time.Minute))
	s.False(holds, "and not again")
}

// lostLoop is a loop over one lost condition of counter, checking every 15 s and sending at
// most once a minute, on a clock the test moves.
func (s *AlertsSuite) lostLoop(counter prometheus.Counter, now *time.Time) *Loop {
	limit := time.Minute
	return New(Options{
		Hub: s.hub, Logger: slog.New(slog.NewTextHandler(s.logs, nil)), Every: 15 * time.Second, Limit: limit,
		Now: func() time.Time { return *now },
	}, Condition{Name: "lost", Level: Page, Check: IncreasedSinceAlert(func() float64 { return Total(counter) }, limit)})
}

// ticks runs the loop's checks at each offset from start, as its ticker would.
func (s *AlertsSuite) ticks(loop *Loop, now *time.Time, start time.Time, offsets ...time.Duration) {
	for _, offset := range offsets {
		*now = start.Add(offset)
		loop.check()
	}
}

func (s *AlertsSuite) TestAnIncreaseJustAfterAnAlertIsSentAtTheFirstTickTheLimitAllows() {
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "test_lost_total"})
	start := time.Now()
	now := start
	loop := s.lostLoop(counter, &now)

	counter.Inc()
	s.ticks(loop, &now, start, 0)
	s.Require().Len(s.sent(), 1)
	counter.Inc() // 5 s after the alert
	// A ticker drifts by microseconds: the 15 s tick lands just before the 75 s one looks back
	// to, and the 60 s tick just under the limit, so the 75 s tick is the first that may send.
	s.ticks(loop, &now, start, 15*time.Second-2*time.Microsecond, 30*time.Second, 45*time.Second,
		time.Minute-time.Microsecond, 75*time.Second-time.Microsecond)

	s.Len(s.sent(), 2, "the second loss is sent once the limit has passed")
}

func (s *AlertsSuite) TestOneLossAlertsExactlyOnce() {
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "test_lost_total"})
	start := time.Now()
	now := start
	loop := s.lostLoop(counter, &now)

	// Lost just before a check, which a look-back of limit and interval counts again a minute on.
	counter.Inc()
	s.ticks(loop, &now, start, 0, 15*time.Second, 30*time.Second, 45*time.Second, time.Minute,
		75*time.Second, 90*time.Second, 5*time.Minute)

	s.Len(s.sent(), 1)
}

func (s *AlertsSuite) TestUnchangedForHoldsOnlyWhenTheReadingStoodStill() {
	counter := prometheus.NewCounter(prometheus.CounterOpts{Name: "test_written_total"})
	check := UnchangedFor(func() float64 { return Total(counter) }, time.Minute)
	start := time.Now()

	holds, _ := check(start)
	s.True(holds, "nothing was ever written")
	counter.Inc()
	holds, _ = check(start.Add(30 * time.Second))
	s.False(holds)
	holds, _ = check(start.Add(2 * time.Minute))
	s.True(holds, "nothing written in the last minute")
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

func (s *AlertsSuite) TestEachItemOfAConditionIsAnAlertOfItsOwn() {
	missing := prometheus.NewCounterVec(prometheus.CounterOpts{Name: "test_missing_total"}, []string{"modality", "model"})
	missing.WithLabelValues("llm", "a").Inc()
	missing.WithLabelValues("llm", "b").Inc()
	condition := Condition{Name: "missing", Level: Warn,
		Items: IncreasedPer(missing, time.Hour, []string{"modality", "model"}, nil)}

	s.run(s.loop(time.Hour, time.Hour, condition), 50*time.Millisecond)

	events := s.sent()
	s.Require().Len(events, 2)
	s.Equal([]string{"usage_missing:llm/a"}, events[0].Fingerprint)
	s.Equal([]string{"usage_missing:llm/b"}, events[1].Fingerprint)
	s.Equal("llm/a", events[0].Tags["item"])
	s.Equal("missing", events[0].Tags["usage_alert"])
	s.Equal(sentry.LevelError, events[0].Level, "a warning does not page")
	s.Contains(s.logs.String(), "item=llm/a")
}

func (s *AlertsSuite) TestAnItemIsSentAtMostOncePerLimitAndThenAgain() {
	missing := prometheus.NewCounterVec(prometheus.CounterOpts{Name: "test_missing_total"}, []string{"model"})
	check := IncreasedPer(missing, time.Hour, []string{"model"}, nil)
	start := time.Now()

	s.Empty(check(start), "nothing has gone missing")
	missing.WithLabelValues("a").Inc()
	s.Equal(map[string]map[string]any{"a": {"value": 1.0, "increase": 1.0}}, check(start.Add(time.Second)))
	missing.WithLabelValues("a").Inc()
	missing.WithLabelValues("b").Inc()
	s.Equal([]string{"b"}, slices.Collect(maps.Keys(check(start.Add(time.Minute)))),
		"a is held back by the limit, and b is not")
	s.Equal(map[string]map[string]any{"a": {"value": 2.0, "increase": 1.0}}, check(start.Add(2*time.Hour)),
		"a rise the limit held back is sent once it has passed")
	s.Empty(check(start.Add(3 * time.Hour)))
}

func (s *AlertsSuite) TestAnItemConditionCountsOnlyTheSeriesItIsAbout() {
	estimated := prometheus.NewCounterVec(prometheus.CounterOpts{Name: "test_estimated_total"}, []string{"model", "reason"})
	estimated.WithLabelValues("a", "cancelled").Add(3)
	estimated.WithLabelValues("b", "missing").Inc()
	check := IncreasedPer(estimated, time.Hour, []string{"model"}, map[string]string{"reason": "missing"})

	s.Equal([]string{"b"}, slices.Collect(maps.Keys(check(time.Now()))))
}
