//go:build integration

package policy

import (
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/appconfig"
	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/ledger"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage/usagetest"
)

// devTiers is the development tier file, which the local stack seeds too.
const devTiers = "../../tiers/dev.json"

// sandboxed is a usage service of the test's own seeded with the development tiers, and an
// enforcer reading it.
type sandboxed struct {
	*reading
	service *usagetest.Service
	tiers   map[string]usage.Tier
}

func (s *PolicySuite) sandboxed(change ...func(*Settings)) *sandboxed {
	service := usagetest.Start(s.T(), usagetest.Options{
		DSN: usagetest.Database(s.T(), "usage_policy_sandbox"), ManageKey: "manage-" + uuid.NewString(),
		Logger: slog.New(slog.DiscardHandler),
	})
	file, err := ledger.ReadTierFile(devTiers)
	s.Require().NoError(err)
	tiers := map[string]usage.Tier{}
	for _, tier := range file.Tiers {
		service.SetTier(s.T(), tier)
		tiers[tier.Name] = tier
	}
	return s.readingSandboxes(service, tiers, change...)
}

// readingSandboxes is another enforcer reading the same usage service, as another router.
func (s *PolicySuite) readingSandboxes(service *usagetest.Service, tiers map[string]usage.Tier, change ...func(*Settings)) *sandboxed {
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	client, err := usage.NewClient(usage.ClientOptions{URL: service.URL, Token: service.Token})
	s.Require().NoError(err)
	written := &logs{}
	logger := slog.New(slog.NewTextHandler(written, &slog.HandlerOptions{Level: slog.LevelInfo}))
	enforcer := s.newEnforcer(configs, logger, append([]func(*Settings){func(settings *Settings) {
		settings.Usage = client
	}}, change...)...)
	return &sandboxed{reading: &reading{Enforcer: enforcer, logs: written}, service: service, tiers: tiers}
}

// on puts the suite's organization on tier and waits for its standing to be read.
func (s *PolicySuite) on(enforcer *Enforcer, tier string) {
	enforcer.Join(s.app, s.org, tier)
	s.eventually(func() bool {
		enforcer.money.mu.Lock()
		defer enforcer.money.mu.Unlock()
		acct := enforcer.money.accounts[s.org]
		return acct != nil && !acct.read.IsZero() && acct.standing.OrganizationTier == tier
	}, 5*time.Second, 5*time.Millisecond, "the organization's standing is read")
}

// sandboxLimits are the tier's sandbox limits, which the test needs set.
func (s *PolicySuite) sandboxLimits(tier usage.Tier) usage.SandboxLimits {
	s.Require().NotNil(tier.Limits.Sandbox, "the %s tier in %s sets sandbox limits", tier.Name, devTiers)
	return *tier.Limits.Sandbox
}

// limited is a development tier whose sandbox limits set network and every size, with the
// network it allows.
func (s *PolicySuite) limited(tiers map[string]usage.Tier, network string) usage.Tier {
	for _, tier := range tiers {
		if l := tier.Limits.Sandbox; l != nil && l.Network != nil && *l.Network == network &&
			l.MaxCPU != nil && l.MaxMemoryGB != nil && l.MaxDiskGB != nil && l.MaxConcurrent != nil {
			return tier
		}
	}
	s.FailNow("no tier in " + devTiers + " limits sandboxes to network " + network)
	return usage.Tier{}
}

func (s *PolicySuite) sandboxAllowed(enforcer *Enforcer, ask SandboxAsk) error {
	return enforcer.SandboxAllowed(s.ctx, ownerOf(s.app), ask)
}

func (s *PolicySuite) TestATierWithNetworkNoneGetsNoNetwork() {
	b := s.sandboxed()
	tier := s.limited(b.tiers, usage.NetworkNone)
	s.on(b.Enforcer, tier.Name)

	for name, ask := range map[string]SandboxAsk{
		"the whole network":         {Config: sandbox.Config{Network: true}},
		"allowed domains":           {Config: sandbox.Config{Network: true, AllowedDomains: []string{"example.com"}}},
		"CIDR ranges":               {Config: sandbox.Config{NetworkAllowList: []string{"10.0.0.0/8"}}},
		"a connector's credentials": {Granted: true},
	} {
		err := s.sandboxAllowed(b.Enforcer, ask)
		s.Require().ErrorIs(err, ErrSandboxLimit, name)
		s.False(Transient(err), name)
		s.Contains(err.Error(), `the "`+tier.Name+`" tier allows sandboxes no network`, name)
	}
	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{}), "an isolated sandbox")
}

func (s *PolicySuite) TestASandboxAboveTheTierIsRefused() {
	b := s.sandboxed()
	tier := s.limited(b.tiers, usage.NetworkNone)
	limits := s.sandboxLimits(tier)
	s.on(b.Enforcer, tier.Name)

	for field, ask := range map[string]sandbox.Config{
		"cpu":       {CPU: *limits.MaxCPU + 1},
		"memory_gb": {MemoryGB: *limits.MaxMemoryGB + 1},
		"disk_gb":   {DiskGB: *limits.MaxDiskGB + 1},
	} {
		err := s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: ask})
		s.Require().ErrorIs(err, ErrSandboxLimit, field)
		s.Contains(err.Error(), "sandbox_options."+field, "the limit is named")
		s.Contains(err.Error(), `the "`+tier.Name+`" tier allows at most`)
	}
	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{
		CPU: *limits.MaxCPU, MemoryGB: *limits.MaxMemoryGB, DiskGB: *limits.MaxDiskGB,
	}}), "the tier's largest sandbox")
	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{}), "the provider's default size")
}

func (s *PolicySuite) TestATierWithAnAllowListAllowsOneAndNotTheWholeNetwork() {
	b := s.sandboxed()
	tier := s.limited(b.tiers, usage.NetworkAllowList)
	s.on(b.Enforcer, tier.Name)

	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{Network: true, AllowedDomains: []string{"example.com"}}}))
	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{Granted: true}))
	s.ErrorIs(s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{Network: true}}), ErrSandboxLimit)
}

func (s *PolicySuite) TestAnOrganizationsOwnSandboxLimitsAreLaidOverItsTiers() {
	b := s.sandboxed()
	tier := s.limited(b.tiers, usage.NetworkNone)
	limits := s.sandboxLimits(tier)
	open, cpu := usage.NetworkAny, *limits.MaxCPU+1
	b.service.SetLimits(s.T(), s.org, usage.Limits{Sandbox: &usage.SandboxLimits{Network: &open, MaxCPU: &cpu}})
	s.on(b.Enforcer, tier.Name)
	s.eventually(func() bool {
		return s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{Network: true}}) == nil
	}, 5*time.Second, 5*time.Millisecond, "the override is read")

	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{CPU: cpu}}))
	err := s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{CPU: cpu + 1}})
	s.Require().ErrorIs(err, ErrSandboxLimit)
	s.Contains(err.Error(), "the organization's own sandbox limits")
	err = s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{DiskGB: *limits.MaxDiskGB + 1}})
	s.Require().ErrorIs(err, ErrSandboxLimit)
	s.Contains(err.Error(), `the "`+tier.Name+`" tier`, "a field the override leaves is the tier's")
}

func (s *PolicySuite) TestATierWithNoSandboxLimitsBoundsNothing() {
	_, client := s.redis()
	b := s.sandboxed(sharingSlots(client))
	b.service.SetTier(s.T(), usage.Tier{Name: "unbounded", FailMode: usage.FailClosed})
	s.on(b.Enforcer, "unbounded")

	slots, err := b.Sandbox(s.ctx, ownerOf(s.app), SandboxAsk{Config: sandbox.Config{Network: true, CPU: 64}})

	s.NoError(err)
	s.Nil(slots, "nothing bounds how many run at once")
}

func (s *PolicySuite) TestShadowAllowsASandboxItWouldRefuseAndSaysSo() {
	b := s.sandboxed(func(settings *Settings) { settings.Mode = config.LimitsShadow })
	tier := s.limited(b.tiers, usage.NetworkNone)
	s.on(b.Enforcer, tier.Name)

	s.NoError(s.sandboxAllowed(b.Enforcer, SandboxAsk{Config: sandbox.Config{Network: true}}))
	s.Contains(b.logs.String(), "would refuse")
	s.Contains(b.logs.String(), "reason="+SandboxLimitCode)
}

func (s *PolicySuite) TestUnreadSandboxLimitsAreRefusedForATierThatFailsClosed() {
	unreachable, err := usage.NewClient(usage.ClientOptions{URL: "http://127.0.0.1:1", Token: "none"})
	s.Require().NoError(err)
	configs, err := appconfig.New(appconfig.Options{Store: s.store})
	s.Require().NoError(err)
	enforcer := s.newEnforcer(configs, nil, func(settings *Settings) {
		settings.Usage, settings.FirstRead = unreachable, 20*time.Millisecond
	})
	enforcer.Join(s.app, s.org, usage.DefaultTier)

	err = enforcer.SandboxAllowed(s.ctx, ownerOf(s.app), SandboxAsk{})

	s.Require().ErrorIs(err, ErrSandboxLimit, "no standing, and the default tier fails closed")
	s.True(Transient(err), "it passes once the limits can be read")
}

// sharingSlots shares the enforcer's sandbox slots, and holds, through client.
func sharingSlots(client rueidis.Client) func(*Settings) {
	return func(settings *Settings) { settings.Holds = client }
}

// slots are the organization's slots on the enforcer.
func (s *PolicySuite) slots(b *sandboxed) sandbox.Slots {
	slots, err := b.Sandbox(s.ctx, ownerOf(s.app), SandboxAsk{})
	s.Require().NoError(err)
	s.Require().NotNil(slots)
	return slots
}

func (s *PolicySuite) TestOneSandboxMoreThanTheTierRunsAtOnceIsRefusedOnEveryRouter() {
	_, client := s.redis()
	one := s.sandboxed(sharingSlots(client))
	other := s.readingSandboxes(one.service, one.tiers, sharingSlots(client))
	tier := s.limited(one.tiers, usage.NetworkNone)
	most := *s.sandboxLimits(tier).MaxConcurrent
	s.on(one.Enforcer, tier.Name)
	s.on(other.Enforcer, tier.Name)
	here, there := s.slots(one), s.slots(other)

	var releases []func()
	for range most {
		release, err := here.Take(s.ctx)
		s.Require().NoError(err)
		releases = append(releases, release)
	}
	_, err := there.Take(s.ctx)
	s.Require().ErrorIs(err, ErrSandboxLimit, "another router counts this one's")
	s.Contains(err.Error(), "at a time")

	releases[0]()
	releases[0]()
	release, err := there.Take(s.ctx)
	s.Require().NoError(err, "a slot given back, once, is free")
	release()
	for _, release := range releases[1:] {
		release()
	}
}

func (s *PolicySuite) TestADeadRoutersSandboxSlotLapses() {
	_, client := s.redis()
	b := s.sandboxed(sharingSlots(client))
	slots := &sandboxSlots{enforcer: b.Enforcer, organization: s.org, app: s.app, most: 1, lease: 300 * time.Millisecond,
		refusal: func(int) error { return ErrSandboxLimit }}
	held, _, err := slots.call(s.ctx, uuid.NewString(), "take", "1", "0")
	s.Require().NoError(err)
	s.Require().True(held, "taken by a router that then died, renewing nothing")

	_, err = slots.Take(s.ctx)
	s.Require().ErrorIs(err, ErrSandboxLimit)

	s.eventually(func() bool {
		release, err := slots.Take(s.ctx)
		if err != nil {
			return false
		}
		release()
		return true
	}, 5*time.Second, 50*time.Millisecond, "its lease lapses")
}

func (s *PolicySuite) TestARunningSandboxsSlotIsRenewed() {
	_, client := s.redis()
	b := s.sandboxed(sharingSlots(client))
	slots := &sandboxSlots{enforcer: b.Enforcer, organization: s.org, app: s.app, most: 1, lease: 300 * time.Millisecond,
		refusal: func(int) error { return ErrSandboxLimit }}
	release, err := slots.Take(s.ctx)
	s.Require().NoError(err)
	defer release()

	time.Sleep(time.Second)

	_, err = slots.Take(s.ctx)
	s.ErrorIs(err, ErrSandboxLimit, "held past its lease while its sandbox runs")
}

func (s *PolicySuite) TestASandboxRunsWhenTheSlotsCannotBeCounted() {
	proxy, client := s.redis()
	b := s.sandboxed(sharingSlots(client))
	tier := s.limited(b.tiers, usage.NetworkNone)
	s.on(b.Enforcer, tier.Name)
	slots := s.slots(b)
	proxy.stop()

	release, err := slots.Take(s.ctx)

	s.NoError(err, "Redis down fails open")
	release()
	s.Contains(b.logs.String(), "could not count the organization's running sandboxes")
}