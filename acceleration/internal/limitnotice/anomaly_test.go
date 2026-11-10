//go:build integration

package limitnotice

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// settleFor is how long a check a test waits for may take.
const settleFor = 10 * time.Second

// today is when every check runs unless a test moves it, pinned so no test runs across midnight.
var today = time.Date(2026, 3, 10, 15, 0, 0, 0, time.UTC)

// AnomalySuite checks spend in a database of its own, emptied before each test, since the check
// reads every organization's requests and another package's suite writes some. Slack is an HTTP
// server in the test process that keeps what it is sent.
type AnomalySuite struct {
	suite.Suite
	ctx   context.Context
	db    *store.Store
	slack *slackServer
	// clock is the time the checks run at, and reads how many times they have asked it: each
	// check asks once as it starts, so two more reads mean a whole check ran in between.
	clock atomic.Pointer[time.Time]
	reads atomic.Int64
}

func TestAnomalySuite(t *testing.T) {
	suite.Run(t, new(AnomalySuite))
}

func (s *AnomalySuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "anomaly"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *AnomalySuite) SetupTest() {
	_, err := s.db.DB().ExecContext(s.ctx, "TRUNCATE requests, cost_anomaly_notices")
	s.Require().NoError(err)
	s.slack = newSlackServer()
	s.T().Cleanup(s.slack.Close)
	at := today
	s.clock.Store(&at)
	s.reads.Store(0)
}

// slackServer answers chat.postMessage as Slack does, failing while fail is set.
type slackServer struct {
	*httptest.Server
	mu   sync.Mutex
	sent []string
	fail atomic.Bool
}

func newSlackServer() *slackServer {
	server := &slackServer{}
	server.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Text string `json:"text"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if server.fail.Load() {
			_, _ = w.Write([]byte(`{"ok":false,"error":"channel_not_found"}`))
			return
		}
		server.mu.Lock()
		server.sent = append(server.sent, body.Text)
		server.mu.Unlock()
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	return server
}

func (s *slackServer) posts() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.sent...)
}

// anomalies is a check over the suite's store posting to its Slack every few milliseconds, with
// every threshold at its default, closed when the test ends.
func (s *AnomalySuite) anomalies() *Anomalies {
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = s.slack.URL
	anomalies := NewAnomalies(AnomalyOptions{
		Store: s.db, Poster: slack, Environment: "prod", Logger: slog.New(slog.DiscardHandler),
		Every: 20 * time.Millisecond,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	s.T().Cleanup(anomalies.Close)
	return anomalies
}

// checked waits until the checks running have read the clock past reads more than once, so
// at least one whole check ran after them.
func (s *AnomalySuite) checked(reads int64) {
	s.Require().Eventually(func() bool { return s.reads.Load() >= reads+2 }, settleFor, time.Millisecond)
}

// organization is an organization id nothing else in the run uses.
func organization() string {
	return "org-" + rand.Text()
}

// spent writes organization's spend on the day daysAgo days before today, at noon.
func (s *AnomalySuite) spent(organization string, daysAgo int, cost int64) {
	day := time.Date(today.Year(), today.Month(), today.Day(), 12, 0, 0, 0, time.UTC).AddDate(0, 0, -daysAgo)
	s.Require().NoError(s.db.RecordRequest(s.ctx, &store.Request{
		Modality: "llm", CustomerID: "app-" + organization, OrganizationID: organization, Provider: "openai", Model: "m",
		StartedAt: day, CostMicros: cost, Billable: true, Success: true,
	}))
}

// steady gives organization the same spend on each of the baseline's days.
func (s *AnomalySuite) steady(organization string, cost int64) {
	for daysAgo := 1; daysAgo <= anomalyDays; daysAgo++ {
		s.spent(organization, daysAgo, cost)
	}
}

// noticed is the organizations whose anomaly was posted, by the messages naming them.
func (s *AnomalySuite) noticed(organizations ...string) []string {
	var found []string
	for _, post := range s.slack.posts() {
		for _, organization := range organizations {
			if strings.Contains(post, "`"+organization+"`") {
				found = append(found, organization)
			}
		}
	}
	return found
}

func (s *AnomalySuite) TestOnlyTheOrganizationAtTenTimesItsBaselineIsNoticedNotSteadyGrowth() {
	spike, growing := organization(), organization()
	s.steady(spike, 4_000_000)
	s.spent(spike, 0, 40_000_000)
	// 20% more every day, so today is 2.07x the median day, four days ago.
	cost := 10_000_000.0
	for daysAgo := anomalyDays; daysAgo >= 0; daysAgo-- {
		s.spent(growing, daysAgo, int64(cost))
		cost *= 1.2
	}

	s.anomalies().Start()
	s.checked(0)

	s.Equal([]string{spike}, s.noticed(spike, growing))
	s.Len(s.slack.posts(), 1)
	s.Equal(":chart_with_upwards_trend: Organization `"+spike+"` spent $40.00 on 2026-03-10 (UTC), "+
		"10.0x its median day of the 7 before ($4.00).", s.slack.posts()[0])
}

func (s *AnomalySuite) TestFiveTimesTheBaselineIsNoticedAndJustUnderIsNot() {
	at, under := organization(), organization()
	s.steady(at, 5_000_000)
	s.spent(at, 0, 25_000_000)
	s.steady(under, 5_000_000)
	s.spent(under, 0, 24_999_999)

	s.anomalies().Start()
	s.checked(0)

	s.Equal([]string{at}, s.noticed(at, under))
}

func (s *AnomalySuite) TestASecondCheckTheSameDayPostsNothing() {
	spike := organization()
	s.steady(spike, 4_000_000)
	s.spent(spike, 0, 40_000_000)
	s.spent(spike, 0, 40_000_000)

	first := s.anomalies()
	first.Start()
	s.checked(0)
	s.checked(s.reads.Load())
	s.Len(s.noticed(spike), 1, "a check every tick, one post")
	first.Close()

	s.anomalies().Start()
	s.checked(s.reads.Load())
	s.Len(s.noticed(spike), 1, "nor does a router started later the same day")

	next := today.AddDate(0, 0, 1)
	s.clock.Store(&next)
	s.spent(spike, -1, 200_000_000)
	s.Require().Eventually(func() bool { return len(s.noticed(spike)) == 2 }, settleFor, time.Millisecond,
		"the next day is a day of its own")
}

func (s *AnomalySuite) TestATinyOrganizationUnderTheMinimumSpendStaysQuiet() {
	tiny, least := organization(), organization()
	s.steady(tiny, 10_000)
	s.spent(tiny, 0, anomalyMinSpendMicros-1)
	s.steady(least, 10_000)
	s.spent(least, 0, anomalyMinSpendMicros)

	s.anomalies().Start()
	s.checked(0)

	s.Equal([]string{least}, s.noticed(tiny, least), "far past its baseline, and still spending too little to tell")
	s.EqualValues(20_000_000, anomalyMinSpendMicros)
}

func (s *AnomalySuite) TestAnOrganizationWithoutAWeekOfHistoryStaysQuiet() {
	brandNew, threeDays, fourDays := organization(), organization(), organization()
	s.spent(brandNew, 0, 100_000_000)
	for daysAgo := 1; daysAgo <= 3; daysAgo++ {
		s.spent(threeDays, daysAgo, 4_000_000)
	}
	s.spent(threeDays, 0, 100_000_000)
	s.spent(threeDays, anomalyDays+1, 4_000_000)
	for daysAgo := 1; daysAgo <= 4; daysAgo++ {
		s.spent(fourDays, daysAgo, 4_000_000)
	}
	s.spent(fourDays, 0, 100_000_000)

	s.anomalies().Start()
	s.checked(0)

	s.Equal([]string{fourDays}, s.noticed(brandNew, threeDays, fourDays),
		"three days of seven leave the median at nothing; four make one, and a day before the week counts for nothing")
}

func (s *AnomalySuite) TestTheBaselineIsReadAgainOnANewDay() {
	// It spends from three days ago: too little history today, enough tomorrow.
	organization := organization()
	for daysAgo := 0; daysAgo <= 3; daysAgo++ {
		s.spent(organization, daysAgo, 4_000_000)
	}

	s.anomalies().Start()
	s.checked(0)
	s.Empty(s.noticed(organization))

	next := today.AddDate(0, 0, 1)
	s.clock.Store(&next)
	s.spent(organization, -1, 40_000_000)
	s.Require().Eventually(func() bool { return len(s.noticed(organization)) == 1 }, settleFor, time.Millisecond)
}

func (s *AnomalySuite) TestOnlyTheRouterHoldingTheLockChecks() {
	spike := organization()
	s.steady(spike, 4_000_000)
	s.spent(spike, 0, 40_000_000)
	elsewhere := s.db.CostAnomalyLock()
	held, err := elsewhere.Hold(s.ctx)
	s.Require().NoError(err)
	s.Require().True(held)

	s.anomalies().Start()
	s.checked(0)
	s.Empty(s.noticed(spike), "another router holds the lock")

	elsewhere.Release()
	s.Require().Eventually(func() bool { return len(s.noticed(spike)) == 1 }, settleFor, time.Millisecond)
}

func (s *AnomalySuite) TestAPostThatFailedIsPostedByALaterCheck() {
	spike := organization()
	s.steady(spike, 4_000_000)
	s.spent(spike, 0, 40_000_000)
	s.slack.fail.Store(true)

	s.anomalies().Start()
	s.checked(0)
	s.Empty(s.noticed(spike))

	s.slack.fail.Store(false)
	s.checked(s.reads.Load())
	s.Empty(s.noticed(spike), "its writer's claim stands for a while")
	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE cost_anomaly_notices SET claimed_at = now() - interval '1 hour'")
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return len(s.noticed(spike)) == 1 }, settleFor, time.Millisecond)
	s.checked(s.reads.Load())
	s.Len(s.noticed(spike), 1, "posted once it went through")
}

func (s *AnomalySuite) TestTheCheckRunsHourlyUnlessToldOtherwise() {
	s.Equal(time.Hour, NewAnomalies(AnomalyOptions{Store: s.db, Poster: NewSlack("xoxb-test", "C123")}).every)
}

func (s *AnomalySuite) TestCloseStopsTheCheckAndGivesTheLockBack() {
	NewAnomalies(AnomalyOptions{Store: s.db, Poster: NewSlack("xoxb-test", "C123")}).Close()

	anomalies := s.anomalies()
	anomalies.Start()
	s.checked(0)

	closed := make(chan struct{})
	go func() {
		anomalies.Close()
		close(closed)
	}()
	select {
	case <-closed:
	case <-time.After(settleFor):
		s.FailNow("Close did not return")
	}
	other := s.db.CostAnomalyLock()
	defer other.Release()
	held, err := other.Hold(s.ctx)
	s.Require().NoError(err)
	s.True(held, "the lock went with the check")
}

// A holder whose check fails gives the lock back, so that a router that can check takes it:
// kept, the lock would leave every router without a check while a shard stays down.
func (s *AnomalySuite) TestACheckThatFailsGivesTheLockBack() {
	own := testenv.Database(os.Getenv("ROUTER_POSTGRES_DSN"), "anomaly")
	broken, err := store.OpenShards(store.Options{DSN: own, Shards: map[string]string{
		"here": own, "down": "postgres://postgres:postgres@127.0.0.1:1/none?sslmode=disable&dial_timeout=1s",
	}})
	s.Require().NoError(err)
	defer broken.Close()
	slack := NewSlack("xoxb-test", "C123")
	slack.baseURL = s.slack.URL
	failing := NewAnomalies(AnomalyOptions{
		Store: broken, Poster: slack, Logger: slog.New(slog.DiscardHandler), Every: 20 * time.Millisecond,
		Now: func() time.Time {
			s.reads.Add(1)
			return *s.clock.Load()
		},
	})
	failing.Start()
	defer failing.Close()
	s.checked(0)

	other := s.db.CostAnomalyLock()
	defer other.Release()

	s.Require().Eventually(func() bool {
		held, err := other.Hold(s.ctx)
		return err == nil && held
	}, settleFor, 10*time.Millisecond, "the holder whose check failed let the lock go")
}

// The day checked is the UTC day, not the clock's own: at UTC+10 it is already the next day.
func (s *AnomalySuite) TestTheDayIsUTCWhateverTheClocksZone() {
	spike := organization()
	s.steady(spike, 4_000_000)
	s.spent(spike, 0, 40_000_000)
	// 2026-03-10 15:00 UTC is 2026-03-11 01:00 at UTC+10.
	at := today.In(time.FixedZone("UTC+10", 10*3600))
	s.clock.Store(&at)

	s.anomalies().Start()
	s.checked(0)

	s.Equal([]string{spike}, s.noticed(spike))
}
