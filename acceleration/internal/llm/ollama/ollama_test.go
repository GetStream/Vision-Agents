package ollama

import (
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

func (s *OllamaSuite) TestEveryRequestTurnsThinkingOff() {
	s.Equal(map[string]any{"reasoning_effort": "none"}, requestFields(llm.ResponseParams{}, ""))
}

func (s *OllamaSuite) TestAnOutputCapTravelsAsMaxTokensWhichIsTheOnlyNameOllamaReads() {
	fields := requestFields(llm.ResponseParams{MaxOutputTokens: 32}, "")
	s.Equal(32, fields["max_tokens"])
}
