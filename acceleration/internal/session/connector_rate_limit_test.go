//go:build integration

package session

import (
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/redis/rueidis"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RateLimitSuite is what a session's connector call does after the provider answered 429
// with Retry-After: the calls on the same rate_limit.per key are refused here, without being
// sent, until it passes, on every router that shares the Redis.
type RateLimitSuite struct {
	connectorFixture
	address string
}

func TestRateLimitSuite(t *testing.T) {
	suite.Run(t, new(RateLimitSuite))
}

func (s *RateLimitSuite) SetupSuite() {
	s.address = os.Getenv("ROUTER_REDIS_ADDR")
	if s.address == "" {
		s.T().Skip("ROUTER_REDIS_ADDR is not set")
	}
	s.connectorFixture.SetupSuite()
}

// The text the model reads of a held call, the provider's Retry-After being 30.
const heldFor30 = "retry_after_seconds: 30."

// TestA429HoldsTheNextCallsOnTheSameKeyForItsRetryAfter: T28's acceptance, per app. The
// provider's 429 is answered as connector_rate_limited too, and the held calls never reach it,
// on either connection of the app.
func (s *RateLimitSuite) TestA429HoldsTheNextCallsOnTheSameKeyForItsRetryAfter() {
	s.limitPer(core.RateLimitPerApp)
	primary, secondary := s.connection("", "primary"), s.connection("", "secondary")
	d := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"), s.fixed("other", secondary, "whoami"))
	s.provider.limit("primary", "30")

	refused, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err, "not an error the model would retry")
	s.provider.lift()
	again, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	other, err := s.call(d, "other__whoami", "{}")
	s.Require().NoError(err)

	s.Contains(refused, "connector_rate_limited: the provider limits how often crm__whoami may be called")
	s.Contains(refused, heldFor30)
	s.Contains(again, "connector_rate_limited")
	s.Contains(again, heldFor30)
	s.Contains(other, "connector_rate_limited: the provider limits how often other__whoami may be called")
	s.Equal(1, s.provider.calls("primary"), "the held call was not sent")
	s.Zero(s.provider.calls("secondary"), "the app's other connection was held too")
	s.ElementsMatch([]string{store.InvocationExternalServer, store.InvocationDenied}, s.logged(primary, 2))
	s.Equal([]string{store.InvocationDenied}, s.logged(secondary, 1))
}

// TestAnotherKeyIsNotHeld: per user, the 429 of one account holds nothing of another.
func (s *RateLimitSuite) TestAnotherKeyIsNotHeld() {
	s.limitPer(core.RateLimitPerUser)
	primary, secondary := s.connection("", "primary"), s.connection("", "secondary")
	d := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"), s.fixed("other", secondary, "whoami"))
	s.provider.limit("primary", "30")

	refused, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	said, err := s.call(d, "other__whoami", "{}")
	s.Require().NoError(err)

	s.Contains(refused, "connector_rate_limited")
	s.Equal("secondary", said)
	s.Equal(1, s.provider.calls("secondary"))
}

// TestPerTenantTheConnectionsOfOneTenantShareTheHold: the tenant is the manifest's first
// identity part, here the account input, so two connections to one account share a key.
func (s *RateLimitSuite) TestPerTenantTheConnectionsOfOneTenantShareTheHold() {
	s.limitPer(core.RateLimitPerTenant)
	first, second, elsewhere := s.connection("", "primary"), s.connection("", "primary"), s.connection("", "secondary")
	d := s.attachWith(s.limitedManager(nil), s.fixed("crm", first, "whoami"), s.fixed("again", second, "whoami"),
		s.fixed("other", elsewhere, "whoami"))
	s.provider.limit("primary", "30")

	_, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.provider.lift()
	held, err := s.call(d, "again__whoami", "{}")
	s.Require().NoError(err)
	said, err := s.call(d, "other__whoami", "{}")
	s.Require().NoError(err)

	s.Contains(held, "connector_rate_limited")
	s.Equal(1, s.provider.calls("primary"))
	s.Equal("secondary", said)
}

// TestTheKeyIsFreeOnceTheRetryAfterPasses: after 30 s the call is sent again, by the model.
func (s *RateLimitSuite) TestTheKeyIsFreeOnceTheRetryAfterPasses() {
	s.limitPer(core.RateLimitPerApp)
	clock := &movingClock{now: time.Now()}
	primary := s.connection("", "primary")
	d := s.attachWith(s.limitedManager(clock.Now), s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "30")
	_, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.provider.lift()

	clock.Add(29 * time.Second)
	held, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)
	clock.Add(time.Second)
	said, err := s.call(d, "crm__whoami", "{}")
	s.Require().NoError(err)

	s.Contains(held, "connector_rate_limited")
	s.Contains(held, "retry_after_seconds: 1.")
	s.Equal("primary", said)
	s.Equal(2, s.provider.calls("primary"))
}

// TestTwoRoutersOnOneRedisHoldTheSameCalls: the hold is in Redis, so a session on another
// router, with its own clients, does not send the call either.
func (s *RateLimitSuite) TestTwoRoutersOnOneRedisHoldTheSameCalls() {
	s.limitPer(core.RateLimitPerApp)
	primary := s.connection("", "primary")
	first := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"))
	second := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "30")

	_, err := s.call(first, "crm__whoami", "{}")
	s.Require().NoError(err)
	s.provider.lift()
	held, err := s.call(second, "crm__whoami", "{}")
	s.Require().NoError(err)

	s.Contains(held, "connector_rate_limited")
	s.Contains(held, heldFor30)
	s.Equal(1, s.provider.calls("primary"))
}

// TestWithoutRedisNothingIsHeld: no limiter, as a router without Redis has, answers the 429
// as base does and sends the next call (Kanat, 2026-10-07, D7).
func (s *RateLimitSuite) TestWithoutRedisNothingIsHeld() {
	s.limitPer(core.RateLimitPerApp)
	primary := s.connection("", "primary")
	d := s.attachWith(s.manager, s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "30")

	_, refused := s.call(d, "crm__whoami", "{}")
	s.provider.lift()
	said, err := s.call(d, "crm__whoami", "{}")

	s.ErrorContains(refused, "Too Many Requests")
	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(2, s.provider.calls("primary"))
}

// TestAnAbsurdRetryAfterIsCappedInWhatTheModelReads: the model is told the wait the router
// holds for, not the provider's raw number.
func (s *RateLimitSuite) TestAnAbsurdRetryAfterIsCappedInWhatTheModelReads() {
	s.limitPer(core.RateLimitPerApp)
	primary := s.connection("", "primary")
	d := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "4294967295")

	refused, err := s.call(d, "crm__whoami", "{}")

	s.Require().NoError(err)
	s.Contains(refused, "retry_after_seconds: 3600.")
}

// TestA429WithoutRetryAfterHoldsNothing: the router never makes up a wait the provider did
// not ask for.
func (s *RateLimitSuite) TestA429WithoutRetryAfterHoldsNothing() {
	s.limitPer(core.RateLimitPerApp)
	primary := s.connection("", "primary")
	d := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "")

	_, refused := s.call(d, "crm__whoami", "{}")
	s.provider.lift()
	said, err := s.call(d, "crm__whoami", "{}")

	s.ErrorContains(refused, "Too Many Requests")
	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(2, s.provider.calls("primary"))
}

// TestAConnectorThatNamesNoRateLimitBehavesAsWithoutALimiter: the control. A manifest with
// no rate_limit, as every connector before this change could be, answers a 429 exactly as a
// router with no limiter does, and the next call is sent.
func (s *RateLimitSuite) TestAConnectorThatNamesNoRateLimitBehavesAsWithoutALimiter() {
	primary := s.connection("", "primary")
	limited := s.attachWith(s.limitedManager(nil), s.fixed("crm", primary, "whoami"))
	base := s.attachWith(s.manager, s.fixed("crm", primary, "whoami"))
	s.provider.limit("primary", "30")

	limitedParts, limitedErr := s.call(limited, "crm__whoami", "{}")
	baseParts, baseErr := s.call(base, "crm__whoami", "{}")
	s.provider.lift()
	said, err := s.call(limited, "crm__whoami", "{}")

	s.Require().Error(baseErr)
	s.Equal(baseErr.Error(), limitedErr.Error())
	s.Equal(baseParts, limitedParts)
	s.Require().NoError(err)
	s.Equal("primary", said)
	s.Equal(3, s.provider.calls("primary"))
}

// limitPer makes the next revision of the test's connector limited per scope, its accounts
// named by the account input, and has the test's connections pin it.
func (s *RateLimitSuite) limitPer(per core.RateLimitScope) {
	manifest, err := core.ParseManifest([]byte(fmt.Sprintf(`
id: %s
revision: 1
name: CRM
inputs:
  - name: account
    enum: [primary, secondary, moved]
endpoints:
  mcp: %s/{account}/mcp
schemes: [bearer]
identity: [account]
rate_limit:
  per: %s
sources:
  - kind: mcp
    endpoint: mcp
`, s.connectorID, s.provider.URL, per)))
	s.Require().NoError(err)
	definition, err := s.store.CreateConnectorDefinition(s.ctx, s.customerID, manifest)
	s.Require().NoError(err)
	s.revision = definition.Revision
}

// limitedManager is a router of its own whose limiter keeps holds in the shared Redis, with
// a client of its own, by now (nil is time.Now).
func (s *RateLimitSuite) limitedManager(now func() time.Time) *Manager {
	client, err := rueidis.NewClient(rueidis.ClientOption{InitAddress: []string{s.address}, DisableCache: true})
	s.Require().NoError(err)
	s.T().Cleanup(client.Close)
	manager := s.managerWith(fixtureRequestTimeout)
	manager.options.Connectors.Limiter = core.NewLimiter(client, now)
	return manager
}

// attachWith opens a session of manager with bindings, as attach does for the suite's.
func (s *RateLimitSuite) attachWith(manager *Manager, bindings ...store.ConnectorBinding) *dispatcher {
	spec := s.spec(s.config(bindings...), "", nil)
	d, _, unavailable, err := manager.attachConnectors(s.ctx, &spec)
	s.Require().NoError(err)
	s.Require().Empty(unavailable)
	s.T().Cleanup(d.Close)
	return d
}

// logged is the error types of the count rows connection's calls left, once all are written.
func (s *RateLimitSuite) logged(connection string, count int) []string {
	read := func() []store.ConnectorInvocation {
		rows, err := s.store.ConnectorInvocations(s.ctx, s.customerID, connection, 0, nil)
		s.Require().NoError(err)
		return rows
	}
	s.Require().Eventually(func() bool { return len(read()) >= count }, 5*time.Second, 20*time.Millisecond)
	s.Never(func() bool { return len(read()) > count }, 300*time.Millisecond, 50*time.Millisecond)
	var types []string
	for _, row := range read() {
		types = append(types, row.ErrorType)
	}
	return types
}

// movingClock is a clock a test moves.
type movingClock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *movingClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *movingClock) Add(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}
