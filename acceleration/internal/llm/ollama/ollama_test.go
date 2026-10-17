package ollama

import (
	"context"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openweights/hosttest"
)

type OllamaSuite struct {
	hosttest.Unit
}

func TestOllamaSuite(t *testing.T) {
	suite.Run(t, &OllamaSuite{hosttest.Unit{
		Host:  Host,
		New:   New,
		Model: "deepseek-v4.1-flash:1017",
	}})
}

func (s *OllamaSuite) TestAModelEntryThatAsksToThinkIsRefused() {
	_, err := New(openweights.Options{APIKey: "k", Model: s.Model, Thinking: true})
	s.ErrorContains(err, "thinking is not supported")
}

func (s *OllamaSuite) TestAToolChoiceOllamaWouldIgnoreIsRefusedBeforeNetworkAccess() {
	provider, err := New(openweights.Options{APIKey: "k", Model: s.Model, BaseURL: "http://127.0.0.1:1"})
	s.Require().NoError(err)

	for _, choice := range []string{"required", "none", "get_weather"} {
		_, err := provider.Create(context.Background(), llm.ResponseParams{
			Input:      []llm.Message{{Role: llm.User, Content: "hi"}},
			ToolChoice: choice,
		})
		s.ErrorContains(err, "is not one of auto", choice)
	}
}

func (s *OllamaSuite) TestEveryRequestTurnsThinkingOff() {
	s.Equal(map[string]any{"reasoning_effort": "none"}, requestFields(llm.ResponseParams{}, ""))
}

func (s *OllamaSuite) TestAnOutputCapTravelsAsMaxTokensWhichIsTheOnlyNameOllamaReads() {
	fields := requestFields(llm.ResponseParams{MaxOutputTokens: 32}, "")
	s.Equal(32, fields["max_tokens"])
}
