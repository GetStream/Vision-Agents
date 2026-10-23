//go:build integration

package policy

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/uptrace/bun"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// tierReads counts the reads of organization_tiers a store makes.
type tierReads struct{ reads atomic.Int64 }

func (*tierReads) BeforeQuery(ctx context.Context, _ *bun.QueryEvent) context.Context { return ctx }

func (r *tierReads) AfterQuery(_ context.Context, event *bun.QueryEvent) {
	if strings.HasPrefix(strings.TrimSpace(event.Query), "SELECT") && strings.Contains(event.Query, "organization_tiers") {
		r.reads.Add(1)
	}
}

// tiered is an Enforcer of its own, over a store of its own, on a clock the test moves.
type tiered struct {
	enforcer *Enforcer
	store    *store.Store
	reads    *tierReads
	clock    time.Time
}

func (s *PolicySuite) tiered() *tiered {
	db, err := store.Open(os.Getenv("ROUTER_POSTGRES_DSN"))
	s.Require().NoError(err)
	reads := &tierReads{}
	db.DB().AddQueryHook(reads)
	configs, err := appconfig.New(appconfig.Options{Store: db})
	s.Require().NoError(err)
	enforcer, err := New(configs, slog.New(slog.DiscardHandler))
	s.Require().NoError(err)
	t := &tiered{enforcer: enforcer, store: db, reads: reads, clock: time.Now()}
	enforcer.now = func() time.Time { return t.clock }
	s.T().Cleanup(func() { _ = db.Close() })
	return t
}

func (s *PolicySuite) recordedTier() string {
	tier, err := s.store.TierOf(s.ctx, s.org)
	s.Require().NoError(err)
	return tier
}

func (s *PolicySuite) TestAnOrganizationWithNoTierIsReadOncePerTTL() {
	t := s.tiered()

	for range 200 {
		admitted, err := t.enforcer.Attribute(s.ctx, s.app)
		s.Require().NoError(err)
		s.Require().Empty(admitted.OrganizationTier)
	}

	s.EqualValues(1, t.reads.reads.Load(), "the miss is remembered")
}

func (s *PolicySuite) TestARecordedTierIsReadAgainOnceTheTTLHasPassed() {
	t := s.tiered()
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, s.org, "tier-a", time.Now()))
	s.Require().Equal("tier-a", t.enforcer.tier(s.ctx, s.org))
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, s.org, "tier-b", time.Now().Add(time.Second)))

	s.Equal("tier-a", t.enforcer.tier(s.ctx, s.org), "believed for the TTL")
	t.clock = t.clock.Add(tierTTL)
	s.Equal("tier-b", t.enforcer.tier(s.ctx, s.org), "another router's tier, once the TTL has passed")
}

func (s *PolicySuite) TestARequestsTierIsWrittenAtMostOncePerTTL() {
	t := s.tiered()

	t.enforcer.Join(s.app, s.org, "tier-a")
	s.Eventually(func() bool { return s.recordedTier() == "tier-a" }, 5*time.Second, 20*time.Millisecond)
	t.enforcer.Join(s.app, s.org, "tier-b")
	s.Equal("tier-b", t.enforcer.tier(s.ctx, s.org), "the request's tier is the one rows carry at once")
	s.Never(func() bool { return s.recordedTier() == "tier-b" }, 300*time.Millisecond, 20*time.Millisecond,
		"a second write within the TTL")

	t.clock = t.clock.Add(tierTTL)
	t.enforcer.Join(s.app, s.org, "tier-b")

	s.Eventually(func() bool { return s.recordedTier() == "tier-b" }, 5*time.Second, 20*time.Millisecond)
}

func (s *PolicySuite) TestATierNamedWithinTheTTLOfTheLastWriteIsWrittenOnceTheOrganizationGoesQuiet() {
	t := s.tiered()
	t.enforcer.Join(s.app, s.org, "tier-a")
	s.Eventually(func() bool { return s.recordedTier() == "tier-a" }, 5*time.Second, 20*time.Millisecond)
	t.clock = t.clock.Add(time.Second)
	t.enforcer.Join(s.app, s.org, "tier-b")

	t.clock = t.clock.Add(tierTTL)

	s.Equal("tier-b", t.enforcer.tier(s.ctx, s.org), "the request's tier, not the older one recorded")
	s.Eventually(func() bool { return s.recordedTier() == "tier-b" }, 5*time.Second, 20*time.Millisecond)
	s.Equal("tier-b", t.enforcer.tier(s.ctx, s.org))
}

func (s *PolicySuite) TestAnUnwrittenTierIsNotWrittenOverANewerOne() {
	t := s.tiered()
	t.enforcer.Join(s.app, s.org, "tier-a")
	s.Eventually(func() bool { return s.recordedTier() == "tier-a" }, 5*time.Second, 20*time.Millisecond)
	t.clock = t.clock.Add(time.Second)
	t.enforcer.Join(s.app, s.org, "tier-b")
	s.Require().NoError(s.store.SetOrganizationTier(s.ctx, s.org, "tier-c", t.clock.Add(time.Second)))

	t.clock = t.clock.Add(tierTTL)
	t.enforcer.tier(s.ctx, s.org)

	s.Never(func() bool { return s.recordedTier() != "tier-c" }, 300*time.Millisecond, 20*time.Millisecond,
		"another router's newer tier")
}

func (s *PolicySuite) TestAFailedWriteIsWrittenAgainByTheNextRequest() {
	t := s.tiered()
	working := t.enforcer.config
	broken := s.tiered()
	s.Require().NoError(broken.store.Close())
	t.enforcer.config = broken.enforcer.config

	// seeTier rather than Join, so no membership write is reading the store being swapped.
	t.enforcer.seeTier(s.org, "tier-a")
	s.Eventually(func() bool {
		t.enforcer.tiersMu.Lock()
		defer t.enforcer.tiersMu.Unlock()
		return t.enforcer.tiers[s.org].written == ""
	}, 5*time.Second, 20*time.Millisecond, "the failed write is forgotten")
	t.enforcer.config = working
	t.enforcer.seeTier(s.org, "tier-a")

	s.Eventually(func() bool { return s.recordedTier() == "tier-a" }, 5*time.Second, 20*time.Millisecond,
		"written within the TTL, since the first write never landed")
}

func (s *PolicySuite) TestAFailedReadKeepsTheLastTier() {
	t := s.tiered()
	t.enforcer.Join(s.app, s.org, "tier-a")
	s.Eventually(func() bool { return s.recordedTier() == "tier-a" }, 5*time.Second, 20*time.Millisecond)
	s.Require().NoError(t.store.Close())
	t.clock = t.clock.Add(tierTTL)

	s.Equal("tier-a", t.enforcer.tier(s.ctx, s.org))
}
