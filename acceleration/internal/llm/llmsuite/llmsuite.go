//go:build integration

// Package llmsuite is what every language model provider is held to against its real API:
// it answers and says what the answer cost, it streams, it keeps the conversation it was
// handed, it calls a tool and answers from what the tool found, and it stops when the
// caller cuts in.
//
// A provider suite embeds Suite, says how to build its provider, and inherits those tests.
// What the model accepts beyond text, such as images or streamed reasoning, is read off its
// Capabilities, so a test only runs where the model claims to do the thing it checks.
// Anything only one provider does, such as Gemini's signed tool calls, stays in that
// provider's own file.
package llmsuite

import (
	"context"
	"os"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

const (
	// defaultMaxOutputTokens is enough for a one-word answer from a model that thinks
	// before it speaks, which a tighter budget spends entirely on thinking.
	defaultMaxOutputTokens = 512
	// defaultTimeout bounds one response. A model that has not answered by now has failed
	// as far as a conversation is concerned.
	defaultTimeout = 120 * time.Second
)

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds a provider configured for an ordinary call.
	New func() llm.LLM
	// Requires are the environment variables without which the provider cannot be
	// reached, and whose absence skips rather than fails.
	Requires []string

	// MaxOutputTokens is the budget for the short answers the tests ask for.
	MaxOutputTokens int
	// Timeout bounds one response.
	Timeout time.Duration

	// LLM is the provider every test in the suite talks to. It is built once and never
	// closed: each test abandons its own responses, and nothing it leaves open outlives
	// the run.
	LLM llm.LLM

	squares []llm.ContentPart
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if s.MaxOutputTokens == 0 {
		s.MaxOutputTokens = defaultMaxOutputTokens
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}
	s.LLM = s.New()

	pictures, err := squares()
	s.Require().NoError(err)
	s.squares = pictures
}

// Ask runs one request to the end on the suite's provider. A provider failure ends the
// stream, so a rejected request reports what went wrong rather than timing out.
func (s *Suite) Ask(params llm.ResponseParams) (llm.Response, []llm.Event) {
	return s.AskOn(s.LLM, params)
}

// AskOn runs one request on a provider the caller built, for a test of options the suite
// knows nothing about.
func (s *Suite) AskOn(provider llm.LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
	if params.MaxOutputTokens == 0 {
		params.MaxOutputTokens = s.MaxOutputTokens
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
	defer cancel()

	stream, err := provider.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	s.Require().NoError(stream.Err())
	return stream.Response(), events
}

// Reasons reports whether the suite's model thinks before it answers, which a small token
// budget can be spent on entirely.
func (s *Suite) Reasons() bool {
	capabilities := s.LLM.Capabilities()
	return capabilities.StreamsReasoning || capabilities.Effort(llm.ResponseParams{}) != ""
}
