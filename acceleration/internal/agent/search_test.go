package agent

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// stubSearch is a search provider with no web behind it: it answers with whatever the test
// wrote down, and records what it was asked.
type stubSearch struct {
	found search.Result
	err   error

	mu    sync.Mutex
	asked []search.Query
}

func (p *stubSearch) Search(_ context.Context, query search.Query) (search.Result, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.asked = append(p.asked, query)
	if p.err != nil {
		return search.Result{}, p.err
	}
	return p.found, nil
}

func (p *stubSearch) Start(context.Context) error { return nil }
func (p *stubSearch) Close() error                { return nil }
func (p *stubSearch) Provider() string            { return "stub" }
func (p *stubSearch) Model() string               { return "now" }

func (p *stubSearch) queries() []search.Query {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]search.Query(nil), p.asked...)
}

// searches gives the agent a search provider and the tools that reach it.
func (s *AgentSuite) searches(found search.Result) {
	s.finds = &stubSearch{found: found}
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
}

// searchRouter routes every search to the stub, so an agent under test goes through the
// same selection a deployed one does without needing a key for anything.
func (s *AgentSuite) searchRouter() *searchrouter.Router {
	registry := searchrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (search.Provider, error) {
		return s.finds, nil
	})

	router, err := searchrouter.New(searchrouter.Options{
		Config: routing.ModalityConfig{
			Providers: []routing.ProviderConfig{{
				Provider:  "stub",
				Model:     "now",
				Languages: []string{"en"},
				Realtime:  true,
				Tier:      routing.LowLatency,
			}},
		},
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *AgentSuite) TestASearchAnswersWhatIsTrueNow() {
	// The one thing neither the model nor the handbook can answer. Without this the agent
	// says it cannot check and the caller is sent to a website.
	s.searches(search.Result{
		Answer: "I-70 is clear through the Eisenhower Tunnel.",
		Documents: []search.Document{
			{Title: "COtrip", URL: "https://cotrip.org", Text: "No closures reported."},
		},
	})
	s.join(false)
	s.model.reply = []string{"Let me check that."}
	s.asksFor("search", `{"query":"traffic on I-70 in Colorado"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how is traffic on I-70")

	ran := s.awaitToolRan()
	s.Equal("search", ran.Tool)
	s.NoError(ran.Err)
	s.Contains(ran.Result, "I-70 is clear through the Eisenhower Tunnel.")
	s.Contains(ran.Result, "COtrip")

	asked := s.finds.queries()
	s.Require().Len(asked, 1)
	s.Equal("traffic on I-70 in Colorado", asked[0].Text)
}

func (s *AgentSuite) TestAQuestionTheWebDoesNotAnswerIsSaidInWords() {
	s.searches(search.Result{})
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("search", `{"query":"traffic on the road to atlantis"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how is traffic to atlantis")

	ran := s.awaitToolRan()
	s.NoError(ran.Err, "an unanswered question is not a broken tool")
	s.Contains(ran.Result, "could not find out")
}

func (s *AgentSuite) TestAWebSearchThatFailedIsToldToTheModelInWords() {
	s.searches(search.Result{})
	s.finds.err = errors.New("the search is rate limited")
	s.join(false)
	s.model.reply = []string{"One moment."}
	s.asksFor("search", `{"query":"traffic"}`)
	participant := stt.Participant{ID: "alice"}
	s.speak(participant)

	s.says(participant, "how is traffic")

	ran := s.awaitToolRan()
	s.Require().Error(ran.Err)
	s.Contains(ran.Result, "did not work")
}

func (s *AgentSuite) TestSearchIsNotOfferedWithoutAProvider() {
	// An agent that offers to check today's traffic and then cannot is worse than one that
	// never offered.
	tools, err := harness.DefaultTools()
	s.Require().NoError(err)
	s.tools = tools
	s.join(false)

	_, offered := s.agent.availableTools().Lookup("search")

	s.False(offered)
}
