package searchrouter

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
)

// stubSearch stands in for a real provider so routing can be driven without credentials.
type stubSearch struct {
	model string
	found search.Result
	err   error

	asked  []search.Query
	closed bool
}

func (s *stubSearch) Search(_ context.Context, query search.Query) (search.Result, error) {
	s.asked = append(s.asked, query)
	if s.err != nil {
		return search.Result{}, s.err
	}
	return s.found, nil
}

func (s *stubSearch) Start(context.Context) error { return nil }
func (s *stubSearch) Close() error                { s.closed = true; return nil }
func (s *stubSearch) Provider() string            { return "stub" }
func (s *stubSearch) Model() string               { return s.model }

type SearchRouterSuite struct {
	suite.Suite
	ctx context.Context
}

func TestSearchRouterSuite(t *testing.T) {
	suite.Run(t, new(SearchRouterSuite))
}

func (s *SearchRouterSuite) SetupTest() {
	s.ctx = context.Background()
}

// config is two providers at the same tier, which is what failover needs something to fall
// back to.
func (s *SearchRouterSuite) config() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "quick", Model: "fast", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
				Price: routing.Price{PerThousandRequests: 5},
			},
			{
				Provider: "spare", Model: "fast", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
			},
		},
		Aliases: map[string]routing.Alias{
			"en-low-latency": {
				Languages: []string{"en"}, RequireRealtime: true, Tier: routing.LowLatency,
			},
		},
	}
}

func (s *SearchRouterSuite) newRouter(factories map[string]routing.Factory[search.Provider]) *Router {
	registry := NewRegistry()
	for name, factory := range factories {
		registry.Register(name, factory)
	}

	router, err := New(Options{
		Config:   s.config(),
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *SearchRouterSuite) TestATargetResolvesAndSearches() {
	found := search.Result{
		Answer:    "I-70 is clear.",
		Documents: []search.Document{{Title: "COtrip", URL: "https://cotrip.org"}},
	}
	provider := &stubSearch{model: "fast", found: found}
	router := s.newRouter(map[string]routing.Factory[search.Provider]{
		"quick": func(routing.Spec) (search.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	answered, err := session.Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Equal("quick", session.Provider())
	s.Equal("fast", session.Model())
	s.Equal("I-70 is clear.", answered.Answer)
	s.Require().Len(provider.asked, 1)
	s.Equal("traffic on I-70", provider.asked[0].Text)
}

func (s *SearchRouterSuite) TestAProviderWithoutAKeyDropsToTheNextCandidate() {
	// A provider is built when the session opens, so a deployment holding a key for one
	// of three still searches rather than offering the tool and then failing on it.
	spare := &stubSearch{model: "fast"}
	router := s.newRouter(map[string]routing.Factory[search.Provider]{
		"quick": func(routing.Spec) (search.Provider, error) {
			return nil, errors.New("QUICK_API_KEY is required")
		},
		"spare": func(routing.Spec) (search.Provider, error) { return spare, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	s.Equal("spare", session.Provider())
}

func (s *SearchRouterSuite) TestNoProviderAtAllSaysWhatEachOneComplainedAbout() {
	router := s.newRouter(map[string]routing.Factory[search.Provider]{
		"quick": func(routing.Spec) (search.Provider, error) {
			return nil, errors.New("QUICK_API_KEY is required")
		},
		"spare": func(routing.Spec) (search.Provider, error) {
			return nil, errors.New("SPARE_API_KEY is required")
		},
	})

	_, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})

	s.Require().Error(err)
	s.ErrorContains(err, "QUICK_API_KEY")
	s.ErrorContains(err, "SPARE_API_KEY")
}

func (s *SearchRouterSuite) TestASearchThatFailedIsReportedRatherThanRetriedElsewhere() {
	// Failover is start-time only, as it is for a model: the caller is mid-sentence, and
	// a second provider's latency on top of the first one's failure is a longer silence
	// than saying it could not check.
	provider := &stubSearch{model: "fast", err: errors.New("rate limited")}
	router := s.newRouter(map[string]routing.Factory[search.Provider]{
		"quick": func(routing.Spec) (search.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	_, err = session.Search(s.ctx, search.Query{Text: "traffic"})

	s.ErrorContains(err, "rate limited")
	s.Equal("quick", session.Provider(), "the session stays on the provider it chose")
}

func (s *SearchRouterSuite) TestClosingASessionClosesTheProvider() {
	provider := &stubSearch{model: "fast"}
	router := s.newRouter(map[string]routing.Factory[search.Provider]{
		"quick": func(routing.Spec) (search.Provider, error) { return provider, nil },
	})

	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})
	s.Require().NoError(err)

	s.Require().NoError(session.Close())
	s.Require().NoError(session.Close())
	s.True(provider.closed)
}

func (s *SearchRouterSuite) TestTheDefaultRegistryHasEveryProviderTheConfigDeclares() {
	// A provider declared in router.yaml with no factory behind it is a candidate that
	// can never be chosen, and the router only says so once a call has already started.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	section, ok := config[routing.Search]
	s.Require().True(ok, "search is a routed modality")

	registry := DefaultRegistry()
	for _, provider := range section.Providers {
		s.Truef(registry.Has(provider.Provider),
			"%s is declared but nothing can build it", provider.Name())
	}
}
