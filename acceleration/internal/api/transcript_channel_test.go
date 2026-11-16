package api

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// TranscriptChannelSuite is the agent channel a session's words are in, which a claimed guest
// is added to (addToGuestChannels) and a call's transcript is read from (transcriptReads).
type TranscriptChannelSuite struct {
	suite.Suite
}

func TestTranscriptChannelSuite(t *testing.T) {
	suite.Run(t, new(TranscriptChannelSuite))
}

// A conversation on a channel of a type an agent config answers in is the customer's own
// channel: no one is added to it, and no transcript is read from an agent channel of its id.
func (s *TranscriptChannelSuite) TestAConversationOnAChannelOfAnotherTypeNamesNoAgentChannel() {
	s.Equal("3f1c", transcriptChannel(store.AgentSession{AgentID: "front-desk", ConversationID: "agent:3f1c"}))
	s.Equal("front-desk", transcriptChannel(store.AgentSession{AgentID: "front-desk"}))
	s.Empty(transcriptChannel(store.AgentSession{AgentID: "config-1", ConversationID: "callouts:team-1"}))
}
