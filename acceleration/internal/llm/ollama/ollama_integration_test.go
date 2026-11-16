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
	// Ollama reports no reasoning tokens and streams thinking where openaicompat does not
	// look, so the budget is the only witness.
	complete, _ := s.Ask(llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Say hello in five words."}},
		MaxOutputTokens: 64,
	})

	s.Equal(llm.StatusCompleted, complete.Status, "the answer finished inside a budget thinking would have spent")
	s.NotEmpty(complete.OutputText)
	s.Less(complete.Usage.OutputTokens, int64(32), "five words with no thinking ahead of them")
}
