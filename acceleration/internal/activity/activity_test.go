//go:build integration

package activity

import (
	"context"
	"crypto/rand"
	"database/sql"
	"errors"
	"log/slog"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// settleFor is how long a write a test is waiting for may take to land.
const settleFor = 5 * time.Second

// day is when every request a test counts arrived, pinned so no test runs across midnight.
var day = time.Date(2026, 3, 1, 10, 0, 0, 0, time.UTC)

// ActivitySuite runs recorders over a real Postgres. Every test counts under apps of its own,
// so they never clean up.
type ActivitySuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store
}

func TestActivitySuite(t *testing.T) {
	suite.Run(t, new(ActivitySuite))
}

func (s *ActivitySuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	db, err := store.Open(dsn)
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

// recorder is a Recorder over the suite's store, closed when the test ends.
func (s *ActivitySuite) recorder(interval time.Duration, capacity int) *Recorder {
	recorder, err := New(Options{Store: s.db, Interval: interval, Capacity: capacity, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	s.T().Cleanup(recorder.Close)
	return recorder
}

// app is an app id nothing else in the run is using.
func (s *ActivitySuite) app() string {
	return "app-" + rand.Text()
}

// request is one request app made to set up an agent from the CLI, on day.
func request(app string) Request {
	return Request{OrganizationID: "org", AppID: app, Area: AreaAgentSetup, Client: store.AuditSourceCLI, At: day}
}

// row is app's agent setup row from the CLI, on day: its requests and refusals, the last
// refusal's code, and whether there is a row at all.
func (s *ActivitySuite) row(app string) (requests, refused int64, code string, found bool) {
	var last sql.NullString
	err := s.db.DB().QueryRowContext(s.ctx, `
SELECT requests, refused, last_refusal_code FROM customer_activity
WHERE app_id = ? AND area = ? AND client = ? AND day = ?::date`,
		app, AreaAgentSetup, store.AuditSourceCLI, day.Format(time.DateOnly)).Scan(&requests, &refused, &last)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, 0, "", false
	}
	s.Require().NoError(err)
	return requests, refused, last.String, true
}

// requests is how many requests app has on day, zero when it has no row.
func (s *ActivitySuite) requests(app string) int64 {
	requests, _, _, _ := s.row(app)
	return requests
}

// Defect: what was counted since the last write lost when the router stops.
func (s *ActivitySuite) TestCloseWritesWhatIsHeld() {
	app := s.app()
	recorder := s.recorder(time.Hour, 0)
	recorder.Start()
	recorder.Record(request(app))
	recorder.Record(request(app))

	recorder.Close()

	s.Equal(int64(2), s.requests(app))
}

// Defect: Close waiting for a writer that was never started, or writing nothing without one.
func (s *ActivitySuite) TestCloseOfARecorderNeverStartedWritesWhatIsHeld() {
	app := s.app()
	recorder := s.recorder(time.Hour, 0)
	recorder.Record(request(app))

	recorder.Close()

	s.Equal(int64(1), s.requests(app))
}

// Defect: a second Start running a second writer, which Close then cannot stop.
func (s *ActivitySuite) TestStartingTwiceRunsOneWriter() {
	app := s.app()
	recorder := s.recorder(time.Hour, 0)
	recorder.Start()
	recorder.Start()
	recorder.Record(request(app))

	recorder.Close()

	s.Equal(int64(1), s.requests(app))
}

// Defect: a writer that writes once and stops, or never writes before Close.
func (s *ActivitySuite) TestTheWriterWritesEveryInterval() {
	app := s.app()
	recorder := s.recorder(20*time.Millisecond, 0)
	recorder.Start()

	recorder.Record(request(app))
	s.Require().Eventually(func() bool { return s.requests(app) == 1 }, settleFor, 10*time.Millisecond)
	recorder.Record(request(app))
	s.Require().Eventually(func() bool { return s.requests(app) == 2 }, settleFor, 10*time.Millisecond,
		"a request after the first write was not written by a later one")
}

// Defect: a row under an empty app, which is everybody's unauthenticated requests counted as
// one customer's. The area is the test's own, so a row another run left is not this one.
func (s *ActivitySuite) TestARequestNamingNoAppIsNotCounted() {
	recorder := s.recorder(time.Hour, 0)
	nobody := request("")
	nobody.Area = "area-" + rand.Text()
	recorder.Record(nobody)

	recorder.Close()

	var rows int
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx,
		"SELECT count(*) FROM customer_activity WHERE area = ?", nobody.Area).Scan(&rows))
	s.Equal(0, rows)
}

// Defect: a refusal counted as a request only, or without what it was refused with.
func (s *ActivitySuite) TestARefusedRequestIsCountedWithItsCode() {
	app := s.app()
	recorder := s.recorder(time.Hour, 0)
	refusal := request(app)
	refusal.Refused, refusal.Code = true, "server_side_only"
	recorder.Record(request(app))
	recorder.Record(refusal)

	recorder.Close()

	requests, refused, code, _ := s.row(app)
	s.Equal(int64(2), requests)
	s.Equal(int64(1), refused)
	s.Equal("server_side_only", code)
}

// Defect: a recorder that holds a row for every key it is given, which a flood of apps grows
// without bound between two writes.
func (s *ActivitySuite) TestPastCapacityANewRowIsDroppedAndAHeldOneStillCounts() {
	held, dropped := s.app(), s.app()
	recorder := s.recorder(time.Hour, 1)
	recorder.Record(request(held))
	recorder.Record(request(dropped))
	recorder.Record(request(held))

	recorder.Close()

	s.Equal(int64(2), s.requests(held))
	_, _, _, found := s.row(dropped)
	s.False(found, "a row past capacity was kept")
}

// Defect: a recorder at capacity that waits out the interval, dropping everything new until
// then.
func (s *ActivitySuite) TestARecorderAtCapacityWritesAtOnce() {
	first, second := s.app(), s.app()
	recorder := s.recorder(time.Hour, 2)
	recorder.Start()

	recorder.Record(request(first))
	recorder.Record(request(second))

	s.Require().Eventually(func() bool { return s.requests(first) == 1 && s.requests(second) == 1 },
		settleFor, 10*time.Millisecond, "the write waited for the interval")
}

// Defect: the counts of a write the database refused thrown away rather than written by the
// next one.
func (s *ActivitySuite) TestAFailedWriteIsWrittenByTheNextOne() {
	stuck, other := s.app(), s.app()
	// One more request than a BIGINT holds is refused by Postgres, and the whole write with it.
	s.Require().NoError(s.db.RecordActivity(s.ctx, []store.ActivityDelta{{
		OrganizationID: "org", AppID: stuck, Area: AreaAgentSetup, Client: store.AuditSourceCLI,
		Day: day, Requests: math.MaxInt64, FirstAt: day, LastAt: day,
	}}))
	recorder := s.recorder(20*time.Millisecond, 0)
	recorder.Start()
	recorder.Record(request(stuck))
	recorder.Record(request(other))
	s.Never(func() bool { return s.requests(other) > 0 }, 200*time.Millisecond, 20*time.Millisecond,
		"a write holding the row that overflows succeeded")

	_, err := s.db.DB().ExecContext(s.ctx, "UPDATE customer_activity SET requests = 0 WHERE app_id = ?", stuck)
	s.Require().NoError(err)

	s.Require().Eventually(func() bool { return s.requests(other) == 1 && s.requests(stuck) == 1 },
		settleFor, 10*time.Millisecond, "what the failed write held was not written again")
}

// Defect: a request recorded while the writer swaps out what it holds, lost or counted twice.
// The callers record for a tenth of a second, over many writes. Run it with -race.
func (s *ActivitySuite) TestRequestsRecordedWhileItWritesAreEachCountedOnce() {
	app := s.app()
	recorder := s.recorder(time.Millisecond, 0)
	recorder.Start()
	const callers = 8
	var (
		wait     sync.WaitGroup
		recorded atomic.Int64
	)
	until := time.Now().Add(100 * time.Millisecond)
	for range callers {
		wait.Go(func() {
			for time.Now().Before(until) {
				recorder.Record(request(app))
				recorded.Add(1)
			}
		})
	}
	wait.Wait()

	recorder.Close()

	s.Equal(recorded.Load(), s.requests(app))
}
