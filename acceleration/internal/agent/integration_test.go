//go:build integration

package agent

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// dsnEnvVar is where this looks for a Postgres to record turns into.
const dsnEnvVar = "ROUTER_POSTGRES_DSN"

// writeGrace is how long the asynchronous turn writer is given to drain, since a
// conversation hands turns over rather than waiting on the database.
const writeGrace = 10 * time.Second

// TurnRecordingSuite runs a whole exchange through the in-process loopback edge and
// checks the row it leaves behind, which is the part unit tests cannot reach.
type TurnRecordingSuite struct {
	AgentSuite
	pgStore *store.Store
}

func TestTurnRecordingSuite(t *testing.T) {
	suite.Run(t, new(TurnRecordingSuite))
}

func (s *TurnRecordingSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s not set", dsnEnvVar)
	}

	pgStore, err := store.Open(dsn)
	s.Require().NoError(err)
	s.pgStore = pgStore
	s.Require().NoError(pgStore.Ping(context.Background()))
	s.Require().NoError(pgStore.Migrate(context.Background()))
}

func (s *TurnRecordingSuite) SetupTest() {
	s.AgentSuite.SetupTest()
	// A fresh agent per test, so one run's turns are not another's.
	s.agentID = fmt.Sprintf("agent-test-%d", time.Now().UnixNano())
	s.records = s.pgStore
}

func (s *TurnRecordingSuite) TearDownSuite() {
	if s.pgStore != nil {
		s.Require().NoError(s.pgStore.Close())
	}
}

func (s *TurnRecordingSuite) TestAFinishedExchangeLeavesARowSayingWhatTheCallerWaitedFor() {
	s.join(true)

	participant := stt.Participant{ID: "alice", UserID: "alice"}
	s.speak(participant)
	s.says(participant, "hello there")

	s.eventually(func() bool { return len(s.voice.spoken()) > 0 }, "the agent never answered")

	// Closing drains the writer, so what follows is not racing it.
	s.agent.Close()

	turns := s.recorded(context.Background())
	s.Require().Len(turns, 1, "one exchange, one row")
	s.Equal("acme", turns[0].CustomerID)
	s.Require().NotNil(turns[0].RoundtripMs)
	s.Positive(*turns[0].RoundtripMs, "the caller waited a measurable amount of time")
	s.Require().NotNil(turns[0].LLMTTFTMs)
	s.False(turns[0].Interrupted)
}

func (s *TurnRecordingSuite) recorded(ctx context.Context) []store.Turn {
	deadline := time.Now().Add(writeGrace)
	for {
		var turns []store.Turn
		err := s.pgStore.DB().NewSelect().Model(&turns).
			Where("agent_id = ?", s.agentID).
			Scan(ctx)
		s.Require().NoError(err)

		if len(turns) > 0 || time.Now().After(deadline) {
			return turns
		}
		time.Sleep(500 * time.Millisecond)
	}
}
