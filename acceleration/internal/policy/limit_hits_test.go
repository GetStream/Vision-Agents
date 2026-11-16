//go:build integration

package policy

import (
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// limitHits keeps every hit Settings.LimitHit is told of.
type limitHits struct {
	mu   sync.Mutex
	hits []LimitHit
}

func (l *limitHits) take(hit LimitHit) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.hits = append(l.hits, hit)
}

func (l *limitHits) all() []LimitHit {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]LimitHit(nil), l.hits...)
}

func (s *PolicySuite) TestARefusalAtNoCreditIsAHit() {
	var hits limitHits
	enforcer := s.billed(func(settings *Settings) { settings.LimitHit = hits.take })
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	s.ErrorIs(s.admit(enforcer.Enforcer, s.app).Spend, ErrNoCredits)

	got := hits.all()
	s.Require().NotEmpty(got)
	s.Equal(LimitHit{
		OrganizationID: s.org, AppID: s.app, Tier: got[0].Tier, Kind: store.LimitNoCredit, Enforced: true,
		Message: got[0].Message,
	}, got[0])
	s.Contains(got[0].Message, "no credit")
}

func (s *PolicySuite) TestAWouldRefuseInShadowIsAHitThatWasNotEnforced() {
	var hits limitHits
	enforcer := s.billed(func(settings *Settings) {
		settings.Mode = config.LimitsShadow
		settings.LimitHit = hits.take
	})
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	s.NoError(s.admit(enforcer.Enforcer, s.app).Spend)

	got := hits.all()
	s.Require().NotEmpty(got)
	s.Equal(store.LimitNoCredit, got[0].Kind)
	s.False(got[0].Enforced, "shadow admitted it")
}

func (s *PolicySuite) TestASuspensionIsNoLimitHit() {
	var hits limitHits
	enforcer := s.billed(func(settings *Settings) { settings.LimitHit = hits.take })
	s.usage().Credit(s.T(), s.org, 1_000_000)
	s.usage().Suspend(s.T(), s.org, "a test")
	enforcer.Join(s.app, s.org, openTier)
	s.read(enforcer.Enforcer, s.org)

	_, err := enforcer.Admit(s.ctx, ownerOf(s.app))
	s.ErrorIs(err, ErrSuspended)

	s.Empty(hits.all(), "a suspension is the operator's own doing")
}

func (s *PolicySuite) TestLimitKinds() {
	for reason, kind := range map[string]string{
		reasonNoCredit:           store.LimitNoCredit,
		"day_spend_hard_limit":   store.LimitSpendHardLimit,
		"month_spend_hard_limit": store.LimitSpendHardLimit,
		"spend_hard_limit":       store.LimitSpendHardLimit,
		reasonLimit:              store.LimitSpendLimit,
		reasonUserLimit:          "",
		reasonSuspended:          "",
		reasonUnreadable:         "",
		reasonLimitsRead:         "",
	} {
		s.Equal(kind, limitKind(reason), reason)
	}
}
