//go:build integration

package openailive

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts/stssuite"
)

// OpenAILiveIntegrationSuite inherits what every speech-to-speech provider owes a call.
type OpenAILiveIntegrationSuite struct {
	stssuite.Suite
}

func TestOpenAILiveIntegrationSuite(t *testing.T) {
	suite.Run(t, &OpenAILiveIntegrationSuite{Suite: stssuite.Suite{
		New: func(ask stssuite.Ask) sts.STS {
			provider, err := New(Options{Instructions: ask.Instructions, Tools: ask.Tools})
			require.NoError(t, err)
			return provider
		},
		Requires: []string{apiKeyEnvVar},
	}})
}

// TestABackendCallLeavesOutTheOptionalArgumentsItDoesNotNeed is AI-993 against the backend
// that sent thread_ts:"" with strict left out: a plain post names the channel and the
// message, and nothing else.
func (s *OpenAILiveIntegrationSuite) TestABackendCallLeavesOutTheOptionalArgumentsItDoesNotNeed() {
	provider := s.Started(stssuite.Ask{
		Instructions: "Use the tools to do what the user asks.",
		Tools:        []llm.Tool{sendMessage},
	})
	defer s.Hangup(provider)

	s.Require().NoError(provider.SendText("Post 'hello from the e2e' to Slack channel C0123456789.", sts.Participant{ID: "test-user", UserID: "test-user"}))
	asked := s.Asked(provider)

	s.Require().NotEmpty(asked.ToolCalls)
	var arguments map[string]any
	s.Require().NoError(json.Unmarshal([]byte(asked.ToolCalls[0].Arguments), &arguments))
	s.Contains(arguments, "channel_id")
	s.Contains(arguments, "message")
	s.NotContains(arguments, "thread_ts")
	s.NotContains(arguments, "draft_id")
}
