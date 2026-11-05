package llmrouter

import (
	"crypto/rand"
	"errors"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing/failovertest"
)

// failingOver routes to one vendor per error in errs, in order, each with one model m whose
// responses fail with that error; nil answers. The vendors are named for this test alone.
func (s *LLMRouterSuite) failingOver(errs ...error) (*Session, []string) {
	registry := NewRegistry()
	var vendors, targets []string
	var providers []routing.ProviderConfig
	for i, createErr := range errs {
		vendor := "vendor" + string(rune('a'+i)) + strings.ToLower(rand.Text())
		registry.Register(vendor, func(routing.Spec) (Provider, error) {
			provider := newStubLLM()
			provider.createErr = createErr
			return Started[llm.LLM](provider, nil)
		})
		vendors = append(vendors, vendor)
		targets = append(targets, vendor+"/m")
		providers = append(providers, routing.ProviderConfig{Provider: vendor, Model: "m", Languages: []string{"en"}})
	}
	router, err := New(Options{Registry: registry, Config: routing.ModalityConfig{Providers: providers}})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	session, err := router.Start(s.ctx, Request{CustomerID: "acme", Providers: targets})
	s.Require().NoError(err)
	s.T().Cleanup(func() { session.Close() })
	s.Require().Equal(vendors[0], session.Provider())
	return session, vendors
}

func (s *LLMRouterSuite) TestAResponseTheSessionsModelFailedIsAFailoverOnceAnotherAnswers() {
	session, vendors := s.failingOver(errors.New("down: 503"), nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Equal(map[string]float64{"llm session vendor": 1}, failovertest.Of(s.T(), vendors[0]))
	s.Empty(failovertest.Of(s.T(), vendors[1]), "it answered")
}

func (s *LLMRouterSuite) TestEachFallbackThatFailedBeforeAnotherAnsweredIsAFailover() {
	session, vendors := s.failingOver(errors.New("a: 503"), errors.New("b: 503"), nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Equal(map[string]float64{"llm session vendor": 1}, failovertest.Of(s.T(), vendors[0]))
	s.Equal(map[string]float64{"llm session vendor": 1}, failovertest.Of(s.T(), vendors[1]), "the response moved off it too")
	s.Empty(failovertest.Of(s.T(), vendors[2]), "it answered")
}

func (s *LLMRouterSuite) TestASettingTheModelRefusedIsAFailoverUnderRequest() {
	session, vendors := s.failingOver(routing.Refused(errors.New("a: too long")), routing.Refused(errors.New("b: too long")), nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Equal(map[string]float64{"llm session request": 1}, failovertest.Of(s.T(), vendors[0]), "the session's own model")
	s.Equal(map[string]float64{"llm session request": 1}, failovertest.Of(s.T(), vendors[1]), "a fallback")
}

func (s *LLMRouterSuite) TestAResponseTheSessionsModelAnsweredIsNoFailover() {
	session, vendors := s.failingOver(nil, nil)

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().NoError(err)
	s.Empty(failovertest.Of(s.T(), vendors[0]))
}

func (s *LLMRouterSuite) TestAResponseNoOtherModelAnsweredIsNoFailover() {
	session, vendors := s.failingOver(errors.New("a: 503"), errors.New("b: 503"))

	_, err := session.Create(s.ctx, llm.ResponseParams{ID: "r1", Input: []llm.Message{{Role: llm.User, Content: "hello"}}})

	s.Require().ErrorContains(err, "a: 503")
	s.Empty(failovertest.Of(s.T(), vendors[0]), "nothing took the response over")
	s.Empty(failovertest.Of(s.T(), vendors[1]))
}
