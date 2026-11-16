//go:build integration

package agent

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// SpeculationBillingSuite runs the agent's speculative replies through a model router that
// records into Postgres, and reads back what each reply was billed as.
type SpeculationBillingSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
	// agents is the agent fixture, not embedded so its own tests are not run again here.
	agents *AgentSuite
}

func TestSpeculationBillingSuite(t *testing.T) {
	suite.Run(t, new(SpeculationBillingSuite))
}

func (s *SpeculationBillingSuite) SetupSuite() {
	dsn := os.Getenv(dsnEnvVar)
	if dsn == "" {
		s.T().Skipf("%s must be set", dsnEnvVar)
	}
	s.ctx = context.Background()
	opened, err := store.Open(testenv.Database(dsn, "agent_billing"))
	s.Require().NoError(err)
	s.Require().NoError(opened.Migrate(s.ctx))
	s.store = opened
	s.T().Cleanup(func() { s.Require().NoError(opened.Close()) })
}

func (s *SpeculationBillingSuite) SetupTest() {
	s.agents = new(AgentSuite)
	s.agents.SetT(s.T())
	s.agents.SetupTest()
	s.agents.agentID = "agent-" + uuid.NewString()
	s.agents.billing = s.store
	s.agents.speculates = true
}

// ruled has the participant say text, holding the flow controller's ruling until the reply
// has been asked for ahead of it, then lets the ruling, which is reply, through.
func (s *SpeculationBillingSuite) ruled(reply, text string) {
	agents := s.agents
	agents.join(true)
	agents.flow.reply = []string{reply}
	ruling := make(chan struct{})
	agents.flow.holdCreate = ruling
	participant := stt.Participant{ID: "alice"}
	agents.speak(participant)
	agents.says(participant, text)
	agents.eventually(func() bool { return len(agents.model.requests()) == 1 },
		"the reply should be asked for while the flow controller is still deciding")
	close(ruling)
}

// replies waits for the agent's reply rows, oldest first.
func (s *SpeculationBillingSuite) replies(want int) []store.Request {
	var rows []store.Request
	s.Require().Eventually(func() bool {
		rows = nil
		s.Require().NoError(s.store.DB().NewSelect().Model(&rows).
			Where("agent_id = ?", s.agents.agentID).
			Where("purpose = ?", "reply").
			Order("started_at").Scan(s.ctx))
		return len(rows) == want
	}, 5*time.Second, 10*time.Millisecond, "every reply asked for is recorded")
	return rows
}

func (s *SpeculationBillingSuite) TestADiscardedSpeculationIsTaggedAndBilled() {
	s.ruled(`{"disposition":"ignore","floor":"continue"}`, "mom where is my backpack")

	row := s.replies(1)[0]
	s.NotEmpty(row.Attributes[routing.AttributeSpeculation])
	s.Equal("false", row.Attributes[routing.AttributeKept], "nobody heard it")
	s.True(row.Billable, "what it generated before it was dropped is billed")
	s.Positive(row.CostMicros)
}

func (s *SpeculationBillingSuite) TestAPromotedSpeculationIsKept() {
	s.ruled(`{"disposition":"respond","floor":"stop"}`, "hello")
	s.agents.eventually(func() bool { return len(s.agents.edge.heard()) > 0 }, "the reply was never spoken")

	row := s.replies(1)[0]
	s.NotEmpty(row.Attributes[routing.AttributeSpeculation])
	s.Equal("true", row.Attributes[routing.AttributeKept])
	s.True(row.Billable)
}

func (s *SpeculationBillingSuite) TestAnAnswerAskedAgainSharesTheSpeculationItReplaced() {
	// A clarifying ruling cannot use the reply written without its note, so it is asked for
	// again: the turn shows both, one thrown away.
	s.ruled(`{"disposition":"clarify","floor":"continue"}`, "do it like last time")

	rows := s.replies(2)
	dropped, answered := rows[0], rows[1]
	s.NotEmpty(dropped.Attributes[routing.AttributeSpeculation])
	s.Equal(dropped.Attributes[routing.AttributeSpeculation], answered.Attributes[routing.AttributeSpeculation])
	s.Equal("false", dropped.Attributes[routing.AttributeKept])
	s.Equal("true", answered.Attributes[routing.AttributeKept])
	for _, row := range rows {
		s.True(row.Billable, "both are billed")
		s.Positive(row.CostMicros)
	}
}
