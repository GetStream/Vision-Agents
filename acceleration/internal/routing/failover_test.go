package routing

import (
	"context"
	"crypto/rand"
	"errors"
	"log/slog"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing/failovertest"
)

// failoversOf is every failover counted against provider, by "modality model stage reason".
func (s *RoutingSuite) failoversOf(provider string) map[string]float64 {
	return failovertest.Of(s.T(), provider)
}

// failoverVendors are two vendors no other test names, each with one model, m.
func failoverVendors() (string, string) {
	return "first" + strings.ToLower(rand.Text()), "second" + strings.ToLower(rand.Text())
}

// vendorFactory builds a vendor's candidates.
type vendorFactory func(vendor string, spec Spec) (*stubProvider, error)

// failoverRouter routes STT to first/m and second/m, built by build, on keys customers keep
// in keys (none when nil).
func (s *RoutingSuite) failoverRouter(first, second string, keys CustomerKeys, build vendorFactory) *Router[*stubProvider] {
	config := ModalityConfig{Providers: []ProviderConfig{
		{Provider: first, Model: "m", Languages: []string{"en"}, Realtime: true, DataPolicy: discard},
		{Provider: second, Model: "m", Languages: []string{"en"}, Realtime: true, DataPolicy: discard},
	}}
	registry := NewRegistry[*stubProvider]()
	for _, vendor := range []string{first, second} {
		registry.Register(vendor, func(spec Spec) (*stubProvider, error) { return build(vendor, spec) }, OverrideAPIKey)
	}
	router, err := New(Options[*stubProvider]{
		Modality: STT,
		Config:   config,
		Registry: registry,
		Keys:     keys,
		Logger:   slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// keptKeys are the keys customers keep, by customer, provider and region.
type keptKeys map[string]string

func (k keptKeys) CustomerKey(_ context.Context, customerID, provider, region string) (string, bool, error) {
	key, ok := k[customerID+"/"+provider+"/"+region]
	return key, ok, nil
}

// starting builds every candidate, and the ones in failing refuse to start with their error.
func starting(failing map[string]error) vendorFactory {
	return func(vendor string, spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model, startErr: failing[vendor]}, nil
	}
}

func (s *RoutingSuite) TestAFirstCandidateThatServesCountsNoFailover() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(nil))
	var asked []string

	_, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}}, failingUse(&asked, nil))

	s.Require().NoError(err)
	s.Empty(s.failoversOf(first))
	s.Empty(s.failoversOf(second))
}

func (s *RoutingSuite) TestACandidateThatFailedToStartIsAFailoverOnceTheNextIsStarted() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(map[string]error{first: errors.New("first: 503")}))

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}})

	s.Require().NoError(err)
	s.Equal(second+"/m", config.Name())
	s.Equal(map[string]float64{"stt m start vendor": 1}, s.failoversOf(first))
	s.Empty(s.failoversOf(second), "it served")
}

func (s *RoutingSuite) TestACandidateThatStartedAndFailedInUseIsAFailoverAtUse() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(nil))
	var asked []string

	config, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}},
		failingUse(&asked, map[string]error{first + "/m": errors.New("first: 500")}))

	s.Require().NoError(err)
	s.Equal(second+"/m", config.Name())
	s.Equal(map[string]float64{"stt m use vendor": 1}, s.failoversOf(first))
}

func (s *RoutingSuite) TestTheLastCandidateFailingIsNoFailover() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(nil))
	var asked []string

	_, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}},
		failingUse(&asked, map[string]error{first + "/m": errors.New("first: 500"), second + "/m": errors.New("second: 500")}))

	s.Require().Error(err)
	s.Equal(map[string]float64{"stt m use vendor": 1}, s.failoversOf(first))
	s.Empty(s.failoversOf(second), "nothing was moved to after it")
}

func (s *RoutingSuite) TestAFailedCandidateFollowedOnlyByOneThatIsNeverStartedIsNoFailover() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(map[string]error{first: errors.New("first: 503")}))
	// second has no factory, so it is passed over without being started.
	router.registry = NewRegistry[*stubProvider]()
	router.registry.Register(first, func(spec Spec) (*stubProvider, error) {
		return &stubProvider{model: spec.Model, startErr: errors.New("first: 503")}, nil
	})

	_, _, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}})

	s.Require().ErrorContains(err, "no factory registered")
	s.Empty(s.failoversOf(first))
}

func (s *RoutingSuite) TestASettingTheCandidateRefusedIsCountedUnderRequestAtTheStageItWasRefused() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, func(vendor string, spec Spec) (*stubProvider, error) {
		if vendor == first {
			return nil, Refused(errors.New("first: no such voice"))
		}
		return &stubProvider{model: spec.Model}, nil
	})
	var asked []string

	_, err := router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}},
		failingUse(&asked, map[string]error{second + "/m": Refused(errors.New("second: too long"))}))
	s.Require().ErrorIs(err, ErrRequest)
	s.Equal(map[string]float64{"stt m start request": 1}, s.failoversOf(first))
	s.Empty(s.failoversOf(second), "the last one asked")

	third, fourth := failoverVendors()
	router = s.failoverRouter(third, fourth, nil, starting(nil))
	_, err = router.Walk(s.ctx, Request{CustomerID: "acme", Providers: []string{third + "/m", fourth + "/m"}},
		failingUse(&asked, map[string]error{third + "/m": Refused(errors.New("third: too long"))}))
	s.Require().NoError(err)
	s.Equal(map[string]float64{"stt m use request": 1}, s.failoversOf(third))
}

func (s *RoutingSuite) TestACandidateOnTheCustomersOwnKeyIsCountedUnderCustomerKey() {
	first, second := failoverVendors()
	keys := keptKeys{"acme/" + first + "/" + pricing.DefaultRegion: "theirs"}
	router := s.failoverRouter(first, second, keys, func(_ string, spec Spec) (*stubProvider, error) {
		provider := &stubProvider{model: spec.Model}
		if spec.APIKey == "theirs" {
			provider.startErr = errors.New("first: 401 key revoked")
		}
		return provider, nil
	})

	_, config, err := router.Select(s.ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}})

	s.Require().NoError(err)
	s.Equal(first+"/m", config.Name())
	s.False(config.BYOK, "the deployment's instance took it")
	s.Equal(map[string]float64{"stt m start customer_key": 1}, s.failoversOf(first))
}

func (s *RoutingSuite) TestAWalkTheCallerLeftInUseCountsNoFailover() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(nil))
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()

	_, err := router.Walk(ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}},
		func(provider *stubProvider, config ProviderConfig) error {
			cancel()
			return context.Canceled
		})

	s.Require().ErrorIs(err, context.Canceled)
	s.Empty(s.failoversOf(first), "no next candidate was asked")
}

func (s *RoutingSuite) TestAStartThatFailedOnceTheCallerHadLeftIsCountedUnderCancelled() {
	first, second := failoverVendors()
	router := s.failoverRouter(first, second, nil, starting(map[string]error{first: context.Canceled}))
	ctx, cancel := context.WithCancel(s.ctx)
	cancel()

	_, _, err := router.Select(ctx, Request{CustomerID: "acme", Providers: []string{first + "/m", second + "/m"}})

	s.Require().NoError(err, "a stub starts whatever the context, as in TestASelectWhoseContextHasEndedStillStartsTheFirstCandidateAsBefore")
	s.Equal(map[string]float64{"stt m start cancelled": 1}, s.failoversOf(first))
}

func (s *RoutingSuite) TestAFailureIsCountedUnderTheFirstReasonThatExplainsIt() {
	own := ProviderConfig{Provider: "quick", Model: "en", BYOK: true}
	refused := Refused(errors.New("no such voice"))

	s.Equal(failoverRequest, failoverReason(own, refused, true), "a setting is refused the same way on any key and in any context")
	s.Equal(failoverCancelled, failoverReason(own, errors.New("dial: canceled"), true), "a caller who left says nothing of their key")
	s.Equal(failoverCustomerKey, failoverReason(own, errors.New("401"), false))
	s.Equal(failoverVendor, failoverReason(ProviderConfig{Provider: "quick", Model: "en"}, errors.New("503"), false))
}

func (s *RoutingSuite) TestAHandoverIsCountedAtSessionByWhatTheProviderSaidAndWhoseKeyItWasOn() {
	vendor, own := failoverVendors()

	CountHandover(TTS, ProviderConfig{Provider: vendor, Model: "m"}, errors.New("socket closed"))
	CountHandover(LLM, ProviderConfig{Provider: vendor, Model: "m"}, Refused(errors.New("too long")))
	CountHandover(STT, ProviderConfig{Provider: own, Model: "m", BYOK: true}, errors.New("401"))

	s.Equal(map[string]float64{"tts m session vendor": 1, "llm m session request": 1}, s.failoversOf(vendor))
	s.Equal(map[string]float64{"stt m session customer_key": 1}, s.failoversOf(own))
}

func (s *RoutingSuite) TestACustomersOwnModelIsCountedUnderCustomNotTheNameTheyChose() {
	chosen := strings.ToLower(rand.Text())
	before := s.failoversOf(CustomProvider)["llm custom session vendor"]

	CountHandover(LLM, ProviderConfig{Provider: CustomProvider, Model: chosen}, errors.New("500"))

	counted := s.failoversOf(CustomProvider)
	s.Equal(before+1, counted["llm custom session vendor"])
	s.NotContains(counted, "llm "+chosen+" session vendor")
}
