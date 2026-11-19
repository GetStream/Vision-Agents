//go:build integration

package phonic

import (
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/stssuite"
)

// PhonicIntegrationSuite inherits what every speech-to-speech provider owes a call.
type PhonicIntegrationSuite struct {
	stssuite.Suite
}

func TestPhonicIntegrationSuite(t *testing.T) {
	suite.Run(t, &PhonicIntegrationSuite{Suite: stssuite.Suite{
		New: func(ask stssuite.Ask) sts.STS {
			provider, err := New(Options{Instructions: ask.Instructions, Tools: ask.Tools})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}

// TestACallerWhoTalksOverAReplyCutsItOff is the barge-in Phonic detects itself, which it
// reports as the reply's end followed by interrupted_response. The reply has to be settled
// once, as cut off, rather than as finished.
func (s *PhonicIntegrationSuite) TestACallerWhoTalksOverAReplyCutsItOff() {
	provider := s.Started(stssuite.Ask{Instructions: "You are a storyteller. When asked, tell a long story of at least ten sentences."})
	defer s.Hangup(provider)

	s.Require().NoError(provider.SendText("Tell me a long story about a lighthouse keeper.", sts.Participant{ID: "test-user", UserID: "test-user"}))
	s.Collect(provider, func(event sts.Event) bool {
		_, spoke := event.(sts.AudioChunk)
		return spoke
	})
	s.Speak(provider)

	var settled []sts.ResponseComplete
	s.Collect(provider, func(event sts.Event) bool {
		complete, ok := event.(sts.ResponseComplete)
		if ok {
			settled = append(settled, complete)
		}
		return ok
	})
	s.Require().Len(settled, 1)
	s.True(settled[0].Interrupted, "the story the caller talked over should be settled as cut off")
	s.Equal(1, settled[0].Generation)
}
