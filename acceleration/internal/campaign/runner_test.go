//go:build integration

package campaign

import (
	"bytes"
	"context"
	"log/slog"
	"os"
	"regexp"
	"runtime/pprof"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type RunnerSuite struct {
	suite.Suite
	ctx context.Context
	db  *store.Store
}

func TestRunnerSuite(t *testing.T) {
	suite.Run(t, new(RunnerSuite))
}

func (s *RunnerSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN must be set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "campaignrunner"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.db = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

// runner is a Runner that lists a campaign as running, with no loop behind the entry: what is
// under test is what stopped does to the list and the row.
func (s *RunnerSuite) runner(campaign store.Campaign) *Runner {
	return &Runner{
		store:   s.db,
		logger:  slog.New(slog.DiscardHandler),
		running: map[string]context.CancelFunc{campaign.ID: func() {}},
	}
}

func (s *RunnerSuite) campaign(state string) store.Campaign {
	campaign := store.Campaign{CustomerID: "customer-" + s.T().Name(), Name: "campaign", ConfigID: "config", FromNumber: "+10000000000"}
	s.Require().NoError(s.db.CreateCampaign(s.ctx, &campaign))
	s.Require().NoError(s.db.SetCampaignState(s.ctx, campaign.CustomerID, campaign.ID, state))
	return campaign
}

// blockedOnTheList waits until a goroutine of this process stands in stopped, waiting for the
// runner's list lock.
func (s *RunnerSuite) blockedOnTheList() {
	waiting := regexp.MustCompile(`\[(sync\.Mutex\.Lock|semacquire)[^\]]*\]`)
	s.Require().Eventually(func() bool {
		var dump bytes.Buffer
		s.Require().NoError(pprof.Lookup("goroutine").WriteTo(&dump, 2))
		for _, stack := range strings.Split(dump.String(), "\n\n") {
			header, _, _ := strings.Cut(stack, "\n")
			if strings.Contains(stack, "(*Runner).stopped") && waiting.MatchString(header) {
				return true
			}
		}
		return false
	}, 10*time.Second, time.Millisecond, "stopped waits for the list")
}

func (s *RunnerSuite) TestAStartThatLandsWhileStoppedWaitsForTheListKeepsTheLoop() {
	campaign := s.campaign(store.Paused)
	runner := s.runner(campaign)
	runner.mu.Lock()

	left := make(chan bool, 1)
	go func() { left <- runner.stopped(s.ctx, campaign) }()
	s.blockedOnTheList()
	// A Start on this router, which holds the list while it writes the campaign running.
	s.Require().NoError(s.db.SetCampaignState(s.ctx, campaign.CustomerID, campaign.ID, store.Running))
	runner.mu.Unlock()

	select {
	case stopped := <-left:
		s.False(stopped, "the row says running, which is the last word")
	case <-time.After(10 * time.Second):
		s.FailNow("stopped did not return")
	}
	runner.mu.Lock()
	defer runner.mu.Unlock()
	s.Contains(runner.running, campaign.ID, "a loop that is told to go on stays listed, or nobody rings the campaign")
}

func (s *RunnerSuite) TestAPausedCampaignsLoopLeavesTheList() {
	campaign := s.campaign(store.Paused)
	runner := s.runner(campaign)

	s.True(runner.stopped(s.ctx, campaign))

	s.NotContains(runner.running, campaign.ID)
}
