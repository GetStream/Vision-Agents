//go:build integration

package chatlog

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// writeGrace is how long the asynchronous writer is given to drain, since a conversation
// hands messages over rather than waiting for them.
const writeGrace = 15 * time.Second

type ChatlogIntegrationSuite struct {
	suite.Suite
	log     *Log
	agentID string
	ctx     context.Context
}

func TestChatlogIntegrationSuite(t *testing.T) {
	suite.Run(t, new(ChatlogIntegrationSuite))
}

func (s *ChatlogIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" || os.Getenv(apiSecretEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " and " + apiSecretEnvVar + " not set")
	}

	s.ctx = context.Background()
	// A fresh channel per run, so one run's transcript is not another's.
	s.agentID = fmt.Sprintf("chatlog-test-%d", time.Now().UnixNano())

	log, err := New(Options{
		AgentID: s.agentID,
		Agent:   User{ID: "agent-" + s.agentID, Name: "Test Agent"},
	})
	s.Require().NoError(err)
	s.log = log

	s.Require().NoError(log.Start(s.ctx))
	s.T().Cleanup(log.Close)
}

func (s *ChatlogIntegrationSuite) TestWhatWasSaidEndsUpInTheChannel() {
	participant := stt.Participant{ID: "session-1", UserID: "caller-1", Name: "Caller"}

	s.log.Record(agent.Heard{Participant: participant, Text: "what time do you open"})
	s.log.Record(agent.Responded{Text: "we open at nine"})

	messages := s.messages(2)

	said := make([]string, 0, len(messages))
	for _, message := range messages {
		said = append(said, message.Text)
	}
	s.Contains(said, "what time do you open")
	s.Contains(said, "we open at nine")
}

func (s *ChatlogIntegrationSuite) TestEachLineIsAttributedToWhoSaidIt() {
	participant := stt.Participant{ID: "session-2", UserID: "caller-2", Name: "Other Caller"}

	s.log.Record(agent.Heard{Participant: participant, Text: "is anyone there"})

	messages := s.messages(1)

	var authors []string
	for _, message := range messages {
		authors = append(authors, message.User.ID)
	}
	s.Contains(authors, "caller-2", "the caller's user id, not their per-call session id")
}

// messages waits for the writer to drain and returns what the channel holds.
func (s *ChatlogIntegrationSuite) messages(atLeast int) []getstream.MessageResponse {
	deadline := time.Now().Add(writeGrace)
	for {
		response, err := s.log.Chat().GetOrCreateChannel(s.ctx, "messaging", s.agentID,
			&getstream.GetOrCreateChannelRequest{})
		s.Require().NoError(err)

		if len(response.Data.Messages) >= atLeast {
			return response.Data.Messages
		}
		if time.Now().After(deadline) {
			s.Require().FailNowf("the transcript never arrived",
				"wanted at least %d messages, the channel has %d", atLeast, len(response.Data.Messages))
		}
		time.Sleep(time.Second)
	}
}
