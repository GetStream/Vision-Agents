//go:build integration

package main

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/scheduling"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// DrainSuite is what a router holds its shutdown drain for, on a database of
// the suite's own.
type DrainSuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store
}

func TestDrainSuite(t *testing.T) {
	suite.Run(t, new(DrainSuite))
}

func (s *DrainSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}
	s.ctx = context.Background()
	settings := config.Defaults()
	settings.Postgres.DSN = testenv.Database(dsn, "drain")
	db, err := openStore(s.ctx, settings)
	s.Require().NoError(err)
	s.T().Cleanup(db.Close)
	s.db = db
}

func (s *DrainSuite) TestTheDrainWaitsForAScheduledDeliveryInFlight() {
	due := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	task := store.ScheduledTask{
		CustomerID: "customer-" + uuid.NewString(), ConfigID: "config-1", UserID: "user-1",
		CallerKind: store.UserKindAuthenticated, Title: "Stretch", Instructions: "Remind them to stretch.",
		StartLocal: due, Timezone: "UTC", NextRunAt: &due, State: store.ScheduledTaskActive, CreatedBy: "agent",
	}
	s.Require().NoError(s.db.CreateScheduledTask(s.ctx, &task, 25))
	s.T().Cleanup(func() {
		s.NoError(s.db.DeleteScheduledTask(context.Background(), task.CustomerID, task.ID))
	})
	release := make(chan struct{})
	scheduler := scheduling.NewScheduler(s.db, func(ctx context.Context, _ store.ClaimedScheduledTask) (string, string, error) {
		select {
		case <-release:
		case <-ctx.Done():
		}
		return "conversation", "", nil
	}, slog.New(slog.DiscardHandler))
	scheduler.Start(s.ctx)
	s.T().Cleanup(scheduler.Close)

	s.Require().Eventually(func() bool { return scheduler.Running() > 0 }, 10*time.Second, 10*time.Millisecond)
	s.Positive(running(nil, scheduler), "the drain counts the delivery in flight")
	s.Equal(scheduler.Running(), running(nil, scheduler))

	close(release)
	s.Require().Eventually(func() bool { return running(nil, scheduler) == 0 }, 10*time.Second, 10*time.Millisecond)
}
