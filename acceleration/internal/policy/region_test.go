//go:build integration

package policy

import (
	"log/slog"
	"os"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// builtIn is the region each build of a classifier was asked for, in order.
type builtIn struct {
	mu      sync.Mutex
	regions []string
}

func (b *builtIn) last() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.regions[len(b.regions)-1]
}

// regionalRouter routes the stub classifier, declared in the us and in the eu and keeping
// nothing it is sent, held to gate, recording where each build was asked for in built.
func (s *PolicySuite) regionalRouter(gate routing.Gate, built *builtIn) *decisionrouter.Router {
	registry := decisionrouter.NewRegistry()
	registry.Register("stub", func(spec routing.Spec) (decisionmodel.Provider, error) {
		built.mu.Lock()
		defer built.mu.Unlock()
		built.regions = append(built.regions, spec.Region)
		return &classifier{}, nil
	}, routing.OverrideRegion)
	judge := routing.ProviderConfig{Provider: "stub", Model: "judge", Languages: []string{"en"}, Realtime: true,
		DataPolicy: options.DataHandling{TrainsOnData: options.ClaimNo, Retention: options.RetentionNone}}
	eu := judge
	eu.Region = pricing.RegionEU
	router, err := decisionrouter.New(decisionrouter.Options{
		Config:   routing.ModalityConfig{Providers: []routing.ProviderConfig{judge, eu}},
		Registry: registry,
		Gate:     gate,
		Logger:   slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError})),
	})
	s.Require().NoError(err)
	s.T().Cleanup(router.Close)
	return router
}

// started is where the router served work of the app done in region.
func (s *PolicySuite) started(router *decisionrouter.Router, built *builtIn, region string) string {
	session, err := router.Start(s.ctx, decisionrouter.Request{CustomerID: s.app, Region: region, Target: "stub/judge"})
	s.Require().NoError(err)
	s.Require().NoError(session.Close())
	return built.last()
}

func (s *PolicySuite) TestAPolicyReadThatFailsChangesNothingAboutWhereWorkIsDone() {
	closed, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	configs, err := appconfig.New(appconfig.Options{Store: closed})
	s.Require().NoError(err)
	built := &builtIn{}
	router := s.regionalRouter(s.newEnforcer(configs, slog.New(slog.DiscardHandler)), built)

	s.Equal(pricing.RegionEU, s.started(router, built, pricing.RegionEU), "an eu agent's work stays in the eu")
	s.Equal(pricing.RegionUS, s.started(router, built, ""), "work naming no region is in the us, as it always is")
}

func (s *PolicySuite) TestAPolicyWrittenWithARegionKeepsNoWorkInIt() {
	// As a policy saved before regions left policies, which a document read now ignores.
	_, err := s.store.DB().ExecContext(s.ctx, `INSERT INTO policies (scope, scope_id, document, updated_at) VALUES (?, ?, ?::jsonb, now())`,
		store.ScopeOrganization, s.org, `{"data_policy":{"region":"eu","retention":"30d"}}`)
	s.Require().NoError(err)
	built := &builtIn{}
	router := s.regionalRouter(s.enforcer, built)

	admitted, err := s.enforcer.Admit(s.ctx, ownerOf(s.app))

	s.Require().NoError(err)
	s.Empty(admitted.DataPolicy.Region)
	s.Equal("30d", string(admitted.DataPolicy.Retention), "the rest of its data policy still holds")
	s.Equal(pricing.RegionUS, s.started(router, built, ""))
}
