//go:build integration

package agent

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// dsnEnvVar and redisEnvVar are where this looks for a Postgres to count turns into and a
// Redis to keep them in.
const (
	dsnEnvVar   = "ROUTER_POSTGRES_DSN"
	redisEnvVar = "ROUTER_REDIS_ADDR"
)

// writeGrace is how long the asynchronous turn writer is given to drain, since a
// conversation hands turns over rather than waiting on the database.
const writeGrace = 10 * time.Second

// TurnRecordingSuite runs a whole exchange through the in-process loopback edge and
// checks what it leaves behind, which is the part unit tests cannot reach.
type TurnRecordingSuite struct {
	AgentSuite
	pgStore    *store.Store
	liveClient *live.Client
	startedAt  time.Time
}

func TestTurnRecordingSuite(t *testing.T) {
	suite.Run(t, new(TurnRecordingSuite))
}

func (s *TurnRecordingSuite) SetupSuite() {
	dsn, redisAddr := os.Getenv(dsnEnvVar), os.Getenv(redisEnvVar)
	if dsn == "" || redisAddr == "" {
		s.T().Skipf("%s and %s must be set", dsnEnvVar, redisEnvVar)
	}

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.pgStore = pgStore
	s.Require().NoError(pgStore.Ping(context.Background()))
	s.Require().NoError(pgStore.Migrate(context.Background()))

	liveClient, err := live.New(live.Options{
		Address: redisAddr, Username: os.Getenv("ROUTER_REDIS_USERNAME"), Password: os.Getenv("ROUTER_REDIS_PASSWORD"),
	})
	s.Require().NoError(err)
	s.liveClient = liveClient
}

func (s *TurnRecordingSuite) SetupTest() {
	s.AgentSuite.SetupTest()
	// A fresh agent per test, so one run's turns are not another's.
	s.agentID = fmt.Sprintf("agent-test-%d", time.Now().UnixNano())
	s.records = s.pgStore
	s.live = s.liveClient
	s.startedAt = time.Now().Add(-time.Minute)
}

func (s *TurnRecordingSuite) TearDownSuite() {
	if s.liveClient != nil {
		s.liveClient.Close()
	}
	if s.pgStore != nil {
		s.Require().NoError(s.pgStore.Close())
	}
}

func (s *TurnRecordingSuite) TestAFinishedExchangeIsKeptAndCounted() {
	s.join(true)

	participant := stt.Participant{ID: "alice", UserID: "alice"}
	s.speak(participant)
	s.says(participant, "hello there")

	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the agent never answered")

	// Closing drains the writer, so what follows is not racing it.
	s.agent.Close()

	turns, err := s.liveClient.CallTurns(context.Background(), "acme", s.agentID, s.startedAt, nil)
	s.Require().NoError(err)
	s.Require().Len(turns, 1, "one exchange, one turn")
	s.Equal("acme", turns[0].CustomerID)
	s.Require().NotNil(turns[0].RoundtripMs)
	s.Positive(*turns[0].RoundtripMs, "the caller waited a measurable amount of time")
	s.Require().NotNil(turns[0].LLMTTFTMs)
	s.False(turns[0].Interrupted)

	stats, err := s.pgStore.CustomerTurnStats(
		context.Background(), "acme", s.agentID, store.Hourly, s.startedAt.Add(-time.Hour), time.Now().Add(time.Hour))
	s.Require().NoError(err)
	s.Require().Len(stats, 1)
	s.EqualValues(1, stats[0].TurnCount)
	s.NotNil(stats[0].RoundtripP50Ms, "the turn's wait is in its hour's percentiles")
}
