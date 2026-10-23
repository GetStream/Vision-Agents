package routing

import (
	"errors"
	"log/slog"
	"os"
)

// errUnconfigured is what a vendor's factory says when the deployment never set its key.
var errUnconfigured = errors.New("quick: api key is required (set QUICK_API_KEY)")

// builds counts what a probing router's factory was asked to build, by "provider/model".
type builds struct {
	counted map[string]int
	last    map[string]*stubProvider
}

// newProbingRouter routes over the suite's config with a registry that probes, as the
// speech registries do, where every model of quick but en is one the deployment has no key
// for.
func (s *RoutingSuite) newProbingRouter() (*Router[*stubProvider], *Registry[*stubProvider], *builds) {
	built := &builds{counted: map[string]int{}, last: map[string]*stubProvider{}}
	registry := NewRegistry[*stubProvider]()
	for _, provider := range []string{"quick", "lush", "batchy"} {
		registry.Register(provider, func(spec Spec) (*stubProvider, error) {
			name := provider + "/" + spec.Model
			built.counted[name]++
			if name == "quick/multi" {
				return nil, errUnconfigured
			}
			built.last[name] = &stubProvider{provider: provider, model: spec.Model}
			return built.last[name], nil
		})
	}
	registry.Probe()

	router, err := New(Options[*stubProvider]{
		Modality: STT,
		Config:   s.config(),
		Registry: registry,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router, registry, built
}

// availability is each candidate's Available, by name.
func availability(candidates []Candidate) map[string]bool {
	available := map[string]bool{}
	for _, candidate := range candidates {
		available[candidate.Config.Name()] = candidate.Health.Available
	}
	return available
}

func (s *RoutingSuite) TestAModelTheDeploymentCannotBuildIsListedUnavailable() {
	router, _, _ := s.newProbingRouter()

	s.Equal(map[string]bool{
		"quick/en": true, "quick/multi": false, "lush/multi": true, "batchy/offline-en": true,
	}, availability(router.Providers(s.ctx)))
}

func (s *RoutingSuite) TestAShortcutStillResolvesToAModelTheDeploymentCannotBuildRankedLast() {
	router, _, _ := s.newProbingRouter()

	candidates, err := router.Resolve(s.ctx, "en-high-accuracy", nil)
	s.Require().NoError(err, "a shortcut keeps every model it names, so the routes listing it still answer")
	s.Equal([]string{"quick/en", "lush/multi", "batchy/offline-en", "quick/multi"}, names(candidates))
	s.False(candidates[3].Health.Available)
}

func (s *RoutingSuite) TestATargetTheDeploymentCannotBuildIsRefusedNamingWhatIsMissing() {
	router, _, built := s.newProbingRouter()
	stats := s.capturingStats(router)

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "quick/multi"})
	s.ErrorContains(err, `every candidate for "quick/multi" failed: quick/multi: quick: api key is required (set QUICK_API_KEY)`)
	s.Equal(1, built.counted["quick/multi"], "only the probe built it: the model was never tried, so nothing counts against it")
	s.Empty(stats.queue, "no stat marks the model down")
}

func (s *RoutingSuite) TestAShortcutPassesOverAModelTheDeploymentCannotBuild() {
	router, _, built := s.newProbingRouter()
	stats := s.capturingStats(router)

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Target: "multilingual-quality", Providers: []string{"quick/multi", "lush/multi"}})
	s.Require().NoError(err)
	s.Equal("lush/multi", config.Name())
	s.Equal(1, built.counted["quick/multi"], "only the probe built it")
	for len(stats.queue) > 0 {
		s.NotEqual("quick", (<-stats.queue).request.Provider, "no stat marks the passed-over model down")
	}
}

func (s *RoutingSuite) TestReadyBuildsAModelOnceAndClosesWhatItBuilt() {
	_, registry, built := s.newProbingRouter()

	s.Require().NoError(registry.Ready("quick", "en"))
	s.Require().NoError(registry.Ready("quick", "en"))
	s.Require().ErrorIs(registry.Ready("quick", "multi"), errUnconfigured)
	s.Require().ErrorIs(registry.Ready("quick", "multi"), errUnconfigured, "a refusal is kept as well")

	s.Equal(map[string]int{"quick/en": 1, "quick/multi": 1}, built.counted)
	s.True(built.last["quick/en"].closed, "the probe is never started, and must not be left open")
}

func (s *RoutingSuite) TestReadyAnswersNilForAProviderWithNoFactory() {
	_, registry, _ := s.newProbingRouter()

	s.NoError(registry.Ready("nobody", "m"), "Select already passes over a provider with no factory, and says so")
}

func (s *RoutingSuite) TestARegistryThatDoesNotProbeBuildsNothingToAnswer() {
	var built int
	registry := NewRegistry[*stubProvider]()
	registry.Register("quick", func(Spec) (*stubProvider, error) {
		built++
		return nil, errUnconfigured
	})

	s.NoError(registry.Ready("quick", "multi"))
	s.Zero(built, "a test's stubs may count their builds or share one provider")
}
