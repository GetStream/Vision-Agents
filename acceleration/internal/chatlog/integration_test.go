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

	s.await(said("what time do you open"))
	s.await(said("we open at nine"))
}

func (s *ChatlogIntegrationSuite) TestEachLineIsAttributedToWhoSaidIt() {
	participant := stt.Participant{ID: "session-2", UserID: "caller-2", Name: "Other Caller"}

	s.log.Record(agent.Heard{Participant: participant, Text: "is anyone there"})

	stored := s.await(said("is anyone there"))
	s.Equal("caller-2", stored.User.ID, "the caller's user id, not their per-call session id")
}

func (s *ChatlogIntegrationSuite) TestAReplyIsVisibleBeforeItIsFinished() {
	s.log.Record(agent.ResponseDelta{TurnID: "turn-1", Text: "checking the diary "})

	writing := s.await(said("checking the diary "))
	s.Equal(true, writing.Custom[generatingField],
		"a client showing the reply needs to know more of it is coming")

	s.log.Record(agent.ResponseDelta{TurnID: "turn-1", Text: "now"})
	s.log.Record(agent.Responded{TurnID: "turn-1", Text: "checking the diary now"})

	finished := s.await(func(message getstream.MessageResponse) bool {
		return message.ID == writing.ID && message.Text == "checking the diary now"
	})
	s.Equal(false, finished.Custom[generatingField],
		"the pieces were ephemeral, so the whole reply has to be stored")
}

func (s *ChatlogIntegrationSuite) TestAnInterruptedReplyIsKeptAsFarAsItGot() {
	s.log.Record(agent.ResponseDelta{TurnID: "turn-2", Text: "let me check"})

	writing := s.await(said("let me check"))
	s.log.Record(agent.Interrupted{TurnID: "turn-2"})

	finished := s.await(func(message getstream.MessageResponse) bool {
		return message.ID == writing.ID && message.Custom[generatingField] == false
	})
	s.Equal("let me check", finished.Text, "the caller heard that much")
}

// said matches a message by what it says.
func said(text string) func(getstream.MessageResponse) bool {
	return func(message getstream.MessageResponse) bool { return message.Text == text }
}

// await waits for the writer to drain and returns the message that matches.
func (s *ChatlogIntegrationSuite) await(
	matches func(getstream.MessageResponse) bool,
) getstream.MessageResponse {
	limit := transcriptLimit
	// Without asking for the state the channel comes back without its messages.
	state := true
	deadline := time.Now().Add(writeGrace)

	for {
		response, err := s.log.Chat().GetOrCreateChannel(s.ctx, channelType, s.agentID,
			&getstream.GetOrCreateChannelRequest{
				State:    &state,
				Messages: &getstream.MessagePaginationParams{Limit: &limit},
			})
		s.Require().NoError(err)

		for _, message := range response.Data.Messages {
			if matches(message) {
				return message
			}
		}
		if time.Now().After(deadline) {
			s.Require().FailNowf("the transcript never arrived",
				"nothing in the %d messages the channel holds matched", len(response.Data.Messages))
		}
		time.Sleep(time.Second)
	}
}
