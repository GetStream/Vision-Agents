package policy

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"strconv"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/config"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

// ErrSandboxLimit is a sandbox beyond what its organization's sandbox limits allow (its tier's,
// with the organization's own overrides laid over them): bigger, more open to the network, or
// one more than may run at once. The message names the limit.
var ErrSandboxLimit = errors.New("policy: the sandbox is beyond the organization's sandbox limits")

// SandboxLimitCode is the code a sandbox refusal is known by in logs and answers, and the
// reason the would-refuse counter gives it.
const SandboxLimitCode = "sandbox_limit"

const reasonSandbox = SandboxLimitCode

// The lease a running sandbox's slot is held on, renewed every third of it while the sandbox
// runs, so a router that dies gives its slots back within sandboxLease.
const (
	sandboxLease = 90 * time.Second
	// sandboxTimeout bounds one call to the shared Redis about slots.
	sandboxTimeout = 2 * time.Second
)

// sandboxPrefix starts the key of an organization's running sandboxes, in the organization's
// cluster slot: sandboxes:{<organization>}.
const sandboxPrefix = "sandboxes"

//go:embed sandboxes.lua
var sandboxesSource string

var sandboxesScript = rueidis.NewLuaScript(sandboxesSource)

// SandboxAsk is the sandbox a session, or an agent config, asks for.
type SandboxAsk struct {
	Config sandbox.Config
	// Granted is a sandbox whose code calls connectors with their credentials, which reaches
	// those connectors' hosts and so is network on an allow list.
	Granted bool
}

// SandboxNetwork is how far code in the sandbox asked for may reach: usage.NetworkNone,
// usage.NetworkAllowList for domains, ranges or connectors' hosts alone, or usage.NetworkAny.
func SandboxNetwork(ask SandboxAsk) string {
	c := ask.Config
	switch {
	case len(c.AllowedDomains) > 0 || len(c.NetworkAllowList) > 0:
		return usage.NetworkAllowList
	case c.Network:
		return usage.NetworkAny
	case ask.Granted:
		return usage.NetworkAllowList
	}
	return usage.NetworkNone
}

// networkRank orders the networks from closed to open; one not known is the most open.
func networkRank(network string) int {
	switch network {
	case usage.NetworkNone:
		return 0
	case usage.NetworkAllowList:
		return 1
	}
	return 2
}

// networkWords is how a message says what a network reaches.
func networkWords(network string) string {
	switch network {
	case usage.NetworkNone:
		return "no network"
	case usage.NetworkAllowList:
		return "an allow list of domains, ranges or connectors' hosts"
	}
	return "the whole network"
}

// SandboxAllowed refuses, with ErrSandboxLimit naming the limit, a sandbox bigger or more open
// to the network than the organization the owner's work bills to may have. A size the config
// leaves at zero is the provider's default, which every limit is taken to allow. It answers from
// the organization's standing in memory; one never read is waited on for at most FirstRead,
// then refused only if its tier fails closed, as a refusal that passes once it can be checked
// (Transient). An owner in no organization, a router with no usage service or limits off,
// and an organization whose limits set no sandbox, are allowed anything; in shadow a refusal is
// logged as would refuse and allowed.
func (e *Enforcer) SandboxAllowed(ctx context.Context, owner routing.Owner, ask SandboxAsk) error {
	_, err := e.sandboxLimits(ctx, owner, ask)
	return err
}

// Sandbox is SandboxAllowed, and the slots that bound how many of the organization's sandboxes
// run at once when its limits set max_concurrent: a slot is taken before a sandbox is made or
// started again, refused with ErrSandboxLimit once max_concurrent are running on any router.
// Slots are counted in the Redis holds are shared through (Settings.Holds); with none, nothing
// bounds how many run at once, and a Redis that cannot be reached gives every sandbox a slot.
func (e *Enforcer) Sandbox(ctx context.Context, owner routing.Owner, ask SandboxAsk) (sandbox.Slots, error) {
	found, err := e.sandboxLimits(ctx, owner, ask)
	if err != nil || found.limits == nil || found.limits.MaxConcurrent == nil || e.settings.Holds == nil {
		return nil, err
	}
	most := *found.limits.MaxConcurrent
	return &sandboxSlots{
		enforcer: e, organization: found.organization, app: owner.CustomerID, most: most, lease: sandboxLease,
		refusal: func(running int) error {
			return refusedWith(ErrSandboxLimit, "the organization already runs %d of its sandboxes, and %s allows at most %d at a time; "+
				"run the code again once one has stopped", running, found.whose(found.overridden(func(l usage.SandboxLimits) bool {
					return l.MaxConcurrent != nil
				})), most)
		},
	}, nil
}

// sandboxFound is what sandboxLimits found of the organization: its sandbox limits, nil when
// they bound nothing, and who set each.
type sandboxFound struct {
	organization string
	limits       *usage.SandboxLimits
	tier         string
	overrides    *usage.SandboxLimits
}

// overridden reports whether the organization's own overrides set what set reads.
func (f sandboxFound) overridden(set func(usage.SandboxLimits) bool) bool {
	return f.overrides != nil && set(*f.overrides)
}

// whose names who set a limit: the organization's own overrides, or its tier.
func (f sandboxFound) whose(overridden bool) string {
	if overridden {
		return "the organization's own sandbox limits"
	}
	return fmt.Sprintf("the %s tier", strconv.Quote(f.tier))
}

// sandboxLimits finds the owner's organization's sandbox limits and refuses ask if it is
// beyond them, as SandboxAllowed says.
func (e *Enforcer) sandboxLimits(ctx context.Context, owner routing.Owner, ask SandboxAsk) (sandboxFound, error) {
	if e == nil || owner.CustomerID == "" || e.money.off() || e.settings.Usage == nil {
		return sandboxFound{}, nil
	}
	organization := e.decide(ctx, owner.CustomerID).organization
	if organization == "" {
		if pending, ok := e.members.Load(owner.CustomerID); ok {
			organization, _ = pending.(string)
		}
	}
	if organization == "" {
		organization = owner.OrganizationID
	}
	if organization == "" {
		return sandboxFound{}, nil
	}
	found := sandboxFound{organization: organization}
	standing, read := e.money.standingOf(ctx, organization)
	if !read {
		m := &e.money
		m.mu.Lock()
		acct := m.accounts[organization]
		if acct == nil {
			acct = &account{}
		}
		closed := m.failsClosed(acct, organization)
		m.mu.Unlock()
		if !closed {
			return found, nil
		}
		return found, e.sandboxAnswer(organization, owner.CustomerID, uncheckedWith(ErrSandboxLimit,
			"the organization's sandbox limits could not be read, so its sandbox is refused until they can be; try again shortly"))
	}
	found.limits, found.overrides = standing.Limits.Sandbox, standing.Overrides.Sandbox
	found.tier = standing.OrganizationTier
	if found.tier == "" {
		found.tier = standing.Tier.Name
	}
	if found.limits == nil {
		return found, nil
	}
	return found, e.sandboxAnswer(organization, owner.CustomerID, found.beyond(ask))
}

// beyond is the refusal of a sandbox ask whose network or size the limits do not allow, nil
// when they allow it.
func (f sandboxFound) beyond(ask SandboxAsk) error {
	l := *f.limits
	if l.Network != nil {
		asked := SandboxNetwork(ask)
		if networkRank(asked) > networkRank(*l.Network) {
			return refusedWith(ErrSandboxLimit, "the sandbox may reach %s, and %s allows sandboxes %s: set sandbox_options.network "+
				"and its allow lists, and connector bindings with sandbox set, to no more than that", networkWords(asked),
				f.whose(f.overridden(func(l usage.SandboxLimits) bool { return l.Network != nil })), networkWords(*l.Network))
		}
	}
	for _, size := range []struct {
		field, unit string
		asked       int
		most        *int
		set         func(usage.SandboxLimits) bool
	}{
		{"cpu", "vCPUs", ask.Config.CPU, l.MaxCPU, func(l usage.SandboxLimits) bool { return l.MaxCPU != nil }},
		{"memory_gb", "GiB of memory", ask.Config.MemoryGB, l.MaxMemoryGB, func(l usage.SandboxLimits) bool { return l.MaxMemoryGB != nil }},
		{"disk_gb", "GiB of disk", ask.Config.DiskGB, l.MaxDiskGB, func(l usage.SandboxLimits) bool { return l.MaxDiskGB != nil }},
	} {
		if size.most != nil && size.asked > *size.most {
			return refusedWith(ErrSandboxLimit, "sandbox_options.%s is %d, and %s allows at most %d %s per sandbox",
				size.field, size.asked, f.whose(f.overridden(size.set)), *size.most, size.unit)
		}
	}
	return nil
}

// sandboxAnswer is what a sandbox refusal comes to: itself, or in shadow nothing, counted and
// logged as would refuse.
func (e *Enforcer) sandboxAnswer(organization, appID string, refusal error) error {
	if refusal == nil || e.settings.Mode != config.LimitsShadow {
		return refusal
	}
	wouldRefuse.WithLabelValues(reasonSandbox).Inc()
	if e.money.refusals.due(organization+"\x00"+reasonSandbox, logEvery) {
		e.logger.Info("would refuse", append(who(organization), "app", appID, "reason", reasonSandbox, "refusal", refusal)...)
	}
	return nil
}

// standingOf is the organization's standing in memory, waiting for at most FirstRead for its
// first read, and whether it has been read.
func (m *money) standingOf(ctx context.Context, organization string) (usage.Standing, bool) {
	m.want(organization)
	deadline := time.Now().Add(m.settings.FirstRead)
	for {
		m.mu.Lock()
		acct := m.accounts[organization]
		if acct != nil && !acct.read.IsZero() {
			standing := acct.standing
			m.mu.Unlock()
			return standing, true
		}
		m.mu.Unlock()
		if !time.Now().Before(deadline) || ctx.Err() != nil {
			return usage.Standing{}, false
		}
		select {
		case <-ctx.Done():
		case <-time.After(min(5*time.Millisecond, time.Until(deadline))):
		}
	}
}

// sandboxSlots are one organization's slots for running sandboxes, counted in the shared Redis.
type sandboxSlots struct {
	enforcer          *Enforcer
	organization, app string
	most              int
	lease             time.Duration
	// refusal says why a slot is refused while running others are held.
	refusal func(running int) error
}

func (s *sandboxSlots) key() string { return sandboxPrefix + ":{" + s.organization + "}" }

// call runs the script for the slot id, answering whether the slot is held and how many others
// are.
func (s *sandboxSlots) call(ctx context.Context, id, verb string, args ...string) (bool, int, error) {
	call, cancel := context.WithTimeout(ctx, sandboxTimeout)
	defer cancel()
	answered, err := sandboxesScript.Exec(call, s.enforcer.settings.Holds, []string{s.key()},
		append([]string{id, strconv.FormatInt(s.lease.Milliseconds(), 10), verb}, args...)).AsIntSlice()
	if err != nil {
		return false, 0, err
	}
	if len(answered) != 2 {
		return false, 0, fmt.Errorf("policy: %d numbers in an answer about sandbox slots, not 2", len(answered))
	}
	return answered[0] == 1, int(answered[1]), nil
}

// Take holds a slot for one more of the organization's sandboxes, unless most are running
// already. In shadow it is held whatever the count, which is logged as would refuse. A Redis
// that cannot be reached holds nothing and refuses nothing.
func (s *sandboxSlots) Take(ctx context.Context) (func(), error) {
	e := s.enforcer
	id := uuid.NewString()
	force := "0"
	if e.settings.Mode == config.LimitsShadow {
		force = "1"
	}
	held, running, err := s.call(ctx, id, "take", strconv.Itoa(s.most), force)
	if err != nil {
		if e.money.refusals.due(s.organization+"\x00sandbox_slots", logEvery) {
			e.logger.Error("could not count the organization's running sandboxes, so this one runs whatever its limit",
				append(who(s.organization), "error", err)...)
		}
		return func() {}, nil
	}
	if running >= s.most {
		if refusal := e.sandboxAnswer(s.organization, s.app, s.refusal(running)); refusal != nil {
			return nil, refusal
		}
	}
	if !held {
		return func() {}, nil
	}
	stop := make(chan struct{})
	done := make(chan struct{})
	go s.renew(id, stop, done)
	var released sync.Once
	return func() {
		released.Do(func() {
			close(stop)
			<-done
			if _, _, err := s.call(context.Background(), id, "release"); err != nil {
				e.logger.Warn("could not give back a sandbox's slot, which lapses on its own",
					append(who(s.organization), "lease", s.lease, "error", err)...)
			}
		})
	}, nil
}

// renew keeps the slot held until stop is closed.
func (s *sandboxSlots) renew(id string, stop <-chan struct{}, done chan<- struct{}) {
	defer close(done)
	ticker := time.NewTicker(s.lease / 3)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			if _, _, err := s.call(context.Background(), id, "renew"); err != nil &&
				s.enforcer.money.refusals.due(s.organization+"\x00sandbox_renew", logEvery) {
				s.enforcer.logger.Warn("could not renew a running sandbox's slot; it is tried again",
					append(who(s.organization), "error", err)...)
			}
		}
	}
}
