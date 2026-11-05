package llmrouter

import (
	"crypto/rand"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing/failovertest"
)

// failingOver routes to down/m, whose responses fail with downErr, then up/m, whose fail
// with upErr; nil answers. Both are named for this test alone.
func (s *LLMRouterSuite) failingOver(downErr, upErr error) (*Session, string, string) {
	down, up := "down"+strings.ToLower(rand.Text()), "up"+strings.ToLower(rand.Text())
	registry := NewRegistry()
	registry.Register(down, func(routing.Spec) (Provider, error) {
		provider := newStubLLM()
		provider.createErr = downErr
		return Started[llm.LLM](provider, nil)
	})
	registry.Register(up, func(routing.Spec) (Provider, error) {
		provider := newStubLLM()
		provider.createErr = upErr
		return Started[llm.LLM](provider, nil)
	})
	router, err := New(Options{Registry: registry, Config: routing.ModalityConfig{Providers: []routing.ProviderConfig{
		{Provider: down, Model: "m", Languages: []string{"en"}},
		{Provider: up, Model: "m", Languages: []string{"en"}},
	}}})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Providers: []string{down + "/m", up + "/m"}})
	s.Require().NoError(err)
	s.T().Cleanup(func() { session.Close() })
	s.Require().Equal(down, session.Provider())
	return session, down, up
}

func (s *LLMRouterSuite) TestAResponseTheSessionsModelFailedIsAFailoverOnceAnotherAnswers() {
	session, down, up := s.failingOver(errors.New("down: 503"), nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Equal(map[string]float64{"llm m session vendor": 1}, failovertest.Of(s.T(), down))
	s.Empty(failovertest.Of(s.T(), up), "it answered")
}

func (s *LLMRouterSuite) TestAResponseTheSessionsModelAnsweredIsNoFailover() {
	session, down, _ := s.failingOver(nil, nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Empty(failovertest.Of(s.T(), down))
}

func (s *LLMRouterSuite) TestAResponseNoOtherModelAnsweredIsNoFailover() {
	session, down, up := s.failingOver(errors.New("down: 503"), errors.New("up: 503"))

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().ErrorContains(err, "down: 503")
	s.Empty(failovertest.Of(s.T(), down), "nothing took the response over")
	s.Empty(failovertest.Of(s.T(), up))
}
