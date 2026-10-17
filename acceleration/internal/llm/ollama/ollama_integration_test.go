//go:build integration

package ollama

import (
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmsuite"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type OllamaIntegrationSuite struct {
	llmsuite.Suite
}

func TestOllamaIntegrationSuite(t *testing.T) {
	suite.Run(t, &OllamaIntegrationSuite{hosttest.Live(t, Host, New, "deepseek-v4.1-flash:1017")})
}

func (s *OllamaIntegrationSuite) TestThinkingIsOffSoTheAnswerArrivesFirst() {
	// DeepSeek V4.1 Flash thinks by default on Ollama. With a small budget, thinking left
	// on spends all of it and the answer never appears.
	complete, events := s.Ask(llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Say hello in five words."}},
		MaxOutputTokens: 64,
	})

	s.NotEmpty(complete.OutputText)
	s.Zero(complete.Usage.OutputTokensDetails.ReasoningTokens, "reasoning_effort none really does disable thinking")
	for _, event := range events {
		_, thinking := event.(llm.ReasoningTextDelta)
		s.False(thinking, "a non-thinking request must not stream reasoning")
	}
}
