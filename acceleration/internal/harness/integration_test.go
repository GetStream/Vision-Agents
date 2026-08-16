//go:build integration

package harness

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// answerWithin bounds how long a real subagent is given. It is far longer than a skill's
// own deadline, because what is being tested is that the answer arrives at all.
const answerWithin = 60 * time.Second

type HarnessIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	router *llmrouter.Router
}

func TestHarnessIntegrationSuite(t *testing.T) {
	suite.Run(t, new(HarnessIntegrationSuite))
}

func (s *HarnessIntegrationSuite) SetupSuite() {
	if os.Getenv("BASETEN_API_KEY") == "" {
		s.T().Skip("BASETEN_API_KEY not set")
	}
	s.ctx = context.Background()

	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	router, err := llmrouter.New(llmrouter.Options{
		Config:   config[routing.LLM],
		Registry: llmrouter.DefaultRegistry(),
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.router = router
	s.T().Cleanup(router.Close)
}

// session opens a real routed session against a target.
func (s *HarnessIntegrationSuite) session(target string) *llmrouter.Session {
	session, err := s.router.Start(s.ctx, llmrouter.Request{
		CustomerID: "harness-integration", Target: target,
	})
	s.Require().NoError(err)
	return session
}

func (s *HarnessIntegrationSuite) TestAQuestionIsHandedOverAndTheAnswerComesBack() {
	// End to end against real providers: the fast model writes a request for help, the
	// harness takes it out of what would be spoken, and the answer arrives on its own.
	voice := s.session("llm-fast")
	harness, err := New(Options{
		Model:    voice,
		Subagent: s.session("en-high-accuracy"),
		Skills: Skills{Skills: []Skill{{
			Name:         "think",
			Description:  "arithmetic and anything needing several steps",
			Instructions: "Answer in one short sentence, with the number in it.",
			Deadline:     answerWithin,
		}}},
		Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	defer harness.Close()

	drained := make(chan Event, 16)
	go func() {
		for event := range harness.Events() {
			drained <- event
		}
		close(drained)
	}()

	question := "A bill comes to 84 pounds 20 and I want to leave 15 percent. What is the tip?"
	s.Require().NoError(harness.Respond(Turn{
		ID: "turn-1",
		Instructions: "You are on the phone. You are bad at arithmetic and you know it, so " +
			"hand every calculation to your colleague rather than attempting it.",
		History: []llm.Message{{Role: llm.User, Content: question}},
	}))

	var spoken strings.Builder
	deadline := time.After(answerWithin)
	for settled := false; !settled; {
		select {
		case event := <-voice.Events():
			switch typed := event.(type) {
			case llm.TextDelta:
				spoken.WriteString(harness.Filter("turn-1", typed.Text))
			case llm.CompletionComplete:
				spoken.WriteString(harness.Flush())
				settled = true
			}
		case <-deadline:
			s.FailNow("the voice model never replied")
		}
	}

	s.NotContains(spoken.String(), "<ask", "a request for help is never spoken")

	// The answer arrives on its own, however long it takes, which is the point.
	answered := time.After(answerWithin)
	for {
		select {
		case event := <-drained:
			if settled, ok := event.(Settled); ok {
				s.Equal(Done, settled.State)
				s.Contains(settled.Text, "12", "15% of 84.20 is 12.63")
				s.Positive(settled.ElapsedMs)
				return
			}
		case <-answered:
			s.FailNow("the subagent never came back", "spoken: %q", spoken.String())
		}
	}
}
