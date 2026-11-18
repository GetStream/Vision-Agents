//go:build integration

package simulation

import (
	"context"
	"crypto/rand"
	"database/sql"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/activity"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// ActivitySuite covers how a run that ends is counted in the activity record.
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
	db, err := store.Open(testenv.Database(dsn, "simulation"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

// runsCounted is an app's evals row in the activity record.
type runsCounted struct {
	requests, refused int64
	code              string
}

// settled settles a run of an app of its own that ended in state, with a runner that counts in
// an activity record of its own, and returns what the record wrote for the app.
func (s *ActivitySuite) settled(state string) runsCounted {
	record, err := activity.New(activity.Options{Store: s.db, Interval: time.Hour, Logger: slog.New(slog.DiscardHandler)})
	s.Require().NoError(err)
	runner := &Runner{store: s.db, activity: record, logger: slog.New(slog.DiscardHandler)}
	run := store.SimulationRun{ID: "run-" + rand.Text(), CustomerID: "app-" + rand.Text(), State: state, Cases: 1}

	runner.settle(s.ctx, run)
	record.Close()

	var (
		row  runsCounted
		code sql.NullString
	)
	s.Require().NoError(s.db.DB().QueryRowContext(s.ctx, `
SELECT COALESCE(sum(requests), 0), COALESCE(sum(refused), 0), max(last_refusal_code)
FROM customer_activity WHERE app_id = ? AND area = ? AND client = ?`,
		run.CustomerID, activity.AreaEvals, activity.ClientRouter).Scan(&row.requests, &row.refused, &code))
	row.code = code.String
	return row
}

// Defect: a run the router could not get a ruling on counted as a run that worked, or left out.
func (s *ActivitySuite) TestARunThatErroredIsCountedRefused() {
	s.Equal(runsCounted{requests: 1, refused: 1, code: store.SimulationErrored}, s.settled(store.SimulationErrored))
}

// Defect: a run the judge ruled against counted as the router refusing it: the agent failed the
// check, and the router did what it was asked.
func (s *ActivitySuite) TestARunTheJudgeFailedIsCountedAndNotRefused() {
	s.Equal(runsCounted{requests: 1}, s.settled(store.SimulationFailed))
}

// Defect: a run stopped before it had every ruling counted refused.
func (s *ActivitySuite) TestARunThatWasCancelledIsCountedAndNotRefused() {
	s.Equal(runsCounted{requests: 1}, s.settled(store.SimulationCancelled))
}
