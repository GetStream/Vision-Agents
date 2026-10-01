package routing

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
)

// admitting is a customer's policies, decided in advance.
type admitting Admission

func (a admitting) Admit(context.Context, string) (Admission, error) { return Admission(a), nil }

// newRouterAdmitting returns a router whose customers are held to the admission. Every
// provider starts unless failing names its model.
func (s *RoutingSuite) newRouterAdmitting(admission Admission, failing string) *Router[*stubProvider] {
	registry := NewRegistry[*stubProvider]()
	for _, provider := range []string{"quick", "lush", "batchy"} {
		registry.Register(provider, func(spec Spec) (*stubProvider, error) {
			if provider+"/"+spec.Model == failing {
				return &stubProvider{model: spec.Model, startErr: errors.New("upstream is down")}, nil
			}
			return &stubProvider{model: spec.Model}, nil
		})
	}

	router, err := New(Options[*stubProvider]{
		Modality: LLM,
		Config:   s.config(),
		Registry: registry,
		Gate:     admitting(admission),
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

func (s *RoutingSuite) TestAnAllowlistNarrowsShortcutsPriorityListsAndConcreteTargets() {
	router := s.newRouterAdmitting(Admission{Models: []string{"quick/multi"}}, "")

	for _, request := range []Request{
		{Target: "en-low-latency"},
		{Providers: []string{"lush/multi", "quick"}},
		{Target: "quick/multi"},
	} {
		request.CustomerID = "acme"
		_, config, err := router.Select(s.ctx, request)
		s.Require().NoError(err)
		s.Equal("quick/multi", config.Name())
	}
}

func (s *RoutingSuite) TestAModelOffTheAllowlistIsRefused() {
	router := s.newRouterAdmitting(Admission{Models: []string{"quick/multi"}}, "")

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})
	s.ErrorIs(err, ErrModelNotAllowed)
	s.ErrorContains(err, "quick/en")

	_, _, err = router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"lush", "quick/en"}})
	s.ErrorIs(err, ErrModelNotAllowed)
}

func (s *RoutingSuite) TestNoAllowlistRoutesToAnyModel() {
	router := s.newRouterAdmitting(Admission{}, "")

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en"})
	s.Require().NoError(err)

	s.Equal("quick/en", config.Name())
}

func (s *RoutingSuite) TestAnEmptyAllowlistAllowsNothing() {
	router := s.newRouterAdmitting(Admission{Models: []string{}}, "")

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "en-low-latency"})

	s.ErrorIs(err, ErrModelNotAllowed)
}

func (s *RoutingSuite) TestAnAllowlistDoesNotFailOverToAModelOffIt() {
	router := s.newRouterAdmitting(Admission{Models: []string{"quick/en"}}, "quick/en")

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{"quick/en", "lush/multi"}})

	s.ErrorContains(err, "quick/en: upstream is down")
	s.NotContains(err.Error(), "lush/multi:")
}

func (s *RoutingSuite) TestPolicyTagsMustFitBesideTheRequestsBeforeRouting() {
	router := s.newRouterAdmitting(Admission{Tags: Tags{"application": "athena"}}, "")
	tags := Tags{}
	for i := range tagLimit {
		tags[fmt.Sprintf("tag-%d", i)] = "value"
	}

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/en", Tags: tags})

	s.ErrorContains(err, "at most 16 tags")
}

func (s *RoutingSuite) TestPolicyTagsOverrideTheRequestsWithoutChangingThem() {
	tags := Tags{"application": "other", "feature": "chat"}

	labelled := Admission{Tags: Tags{"application": "athena"}}.Labelled(tags)

	s.Equal(Tags{"application": "athena", "feature": "chat"}, labelled)
	s.Equal("other", tags["application"], "the request's own tags are the caller's")
}
