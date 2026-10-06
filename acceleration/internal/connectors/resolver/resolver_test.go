//go:build integration

package resolver_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/resolver"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// routers is how many routers resolve one connection at once in the race test: enough to
// overlap, few enough to stay fast. contracttest's concurrency, a choice, not a measurement.
const routers = 8

// blocked is how long a test waits for a resolve it expects to wait on the lock: long enough
// that a cached answer would have come back, short enough to keep the suite fast. A choice.
const blocked = 300 * time.Millisecond

// What LastError says, as the resolver writes it for whoever reconnects.
const (
	lostRefresh            = "A credential refresh did not finish durably; reconnect the account"
	rejectedGrant          = "The provider rejected the grant; reconnect the account"
	temporarilyUnavailable = "The provider could not renew the credential just now; the next call tries again"
)

// ResolverSuite runs the resolver against Postgres and the fake provider, each router with
// a pool of its own.
type ResolverSuite struct {
	suite.Suite
	dsn string
	db  *store.Store
	f   *fixture
}

func TestResolverSuite(t *testing.T) {
	suite.Run(t, new(ResolverSuite))
}

func (s *ResolverSuite) SetupSuite() {
	s.dsn, s.db = database(s.T())
}

func (s *ResolverSuite) TearDownSuite() {
	if s.db != nil {
		s.Require().NoError(s.db.Close())
	}
}

func (s *ResolverSuite) SetupTest() {
	s.f = newFixture(s.T(), s.dsn, s.db)
}

func (s *ResolverSuite) TestAConnectedConnectionResolvesToACredentialTheProviderTakes() {
	ref := s.f.connected()

	credential, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
	s.False(credential.ExpiresAt.IsZero())
	s.True(credential.ExpiresAt.Equal(*s.f.stored(ref).ExpiresAt), "the first resolve writes the expiry the consent left unknown")
	s.Equal(0, s.f.srv.Refreshes())
}

func (s *ResolverSuite) TestACachedCredentialIsHandedOutWhileAnotherRouterHoldsTheLock() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	first, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.f.hold(ref)

	ctx, cancel := context.WithTimeout(s.f.ctx, blocked)
	defer cancel()
	again, err := r.Resolve(ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err, "the fast path takes no lock")
	s.True(s.f.token(first) == s.f.token(again), "the cached token")
}

func (s *ResolverSuite) TestACredentialOlderThanTheMaximumAgeIsRetrievedAgain() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	// 30 s is the resolver's maxAge; nothing else moves with it, since the token has an hour.
	s.f.clock.Add(31 * time.Second)
	release := s.f.hold(ref)

	ctx, cancel := context.WithTimeout(s.f.ctx, blocked)
	defer cancel()
	_, err = r.Resolve(ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, context.DeadlineExceeded, "past the maximum age the resolve waits for the lock")

	release()
	credential, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
}

func (s *ResolverSuite) TestACachedCredentialThatExpiresBeforeTheCallsDeadlineIsNotHandedOut() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	first, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.f.hold(ref)

	ctx, cancel := context.WithTimeout(s.f.ctx, blocked)
	defer cancel()
	_, err = r.Resolve(ctx, ref, core.CredentialRequest{Deadline: first.ExpiresAt.Add(time.Second)})
	s.ErrorIs(err, context.DeadlineExceeded, "a credential that dies during the call goes back to the scheme")
}

func (s *ResolverSuite) TestADeletedConnectionCannotBeResolvedFromTheCache() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(s.f.db.DeleteConnectorConnection(s.f.ctx, ref.CustomerID, ref.ConnectionID))
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, store.ErrNoConnectorConnection)
}

func (s *ResolverSuite) TestAnotherCustomersConnectionCannotBeResolved() {
	ref := s.f.connected()
	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, core.ConnectionRef{CustomerID: "another-app", ConnectionID: ref.ConnectionID}, core.CredentialRequest{})
	s.ErrorIs(err, store.ErrNoConnectorConnection)
}

func (s *ResolverSuite) TestAPendingConnectionIsNotConnected() {
	ref := s.f.pending()
	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
}

func (s *ResolverSuite) TestAReconnectIsHandedOutInsideTheCacheWindow() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	before, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.f.consent(ref)
	after, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.False(s.f.token(before) == s.f.token(after), "the reconnect's credentials are a new revision, which the cache does not have")
	s.True(s.f.works(after))
}

func (s *ResolverSuite) TestInvalidateOnAnotherRouterStopsTheCachedCredentialHere() {
	ref := s.f.connected()
	here := s.f.router(s.f.srv.Client())
	_, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Invalidate(s.f.ctx, ref, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	_, err = here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
}

func (s *ResolverSuite) TestInvalidateMovesTheConnectionToNeedsReauthorization() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(r.Invalidate(s.f.ctx, ref, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(rejectedGrant, stored.LastError)
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
}

func (s *ResolverSuite) TestInvalidateRefusesAnOutcomeThatEndsNoGrant() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())

	s.Error(r.Invalidate(s.f.ctx, ref, core.Outcome{Kind: core.OutcomeRateLimited}))
	s.Equal(store.ConnectionConnected, s.f.stored(ref).Status)
	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.NoError(err)
}

func (s *ResolverSuite) TestRoutersResolvingADueCredentialAtOnceRefreshItOnce() {
	ref := s.f.connected()
	before := s.f.stored(ref).Revision
	s.f.due()
	resolvers := make([]*resolver.Resolver, routers)
	for i := range resolvers {
		resolvers[i] = s.f.router(s.f.srv.Client())
	}

	var wg sync.WaitGroup
	credentials := make([]core.AccessCredential, routers)
	errs := make([]error, routers)
	for i := range routers {
		wg.Go(func() {
			credentials[i], errs[i] = resolvers[i].Resolve(s.f.ctx, ref, core.CredentialRequest{})
		})
	}
	wg.Wait()

	for i := range routers {
		s.Require().NoError(errs[i])
		s.True(s.f.works(credentials[i]), "every router got a token the provider takes")
	}
	s.Equal(1, s.f.srv.Refreshes(), "one refresh reached the provider; the others found it done")
	stored := s.f.stored(ref)
	s.Equal(before+1, stored.Revision)
	s.Equal(store.ConnectionConnected, stored.Status)
}

func (s *ResolverSuite) TestNeedsReauthorizationIsCommittedBeforeTheRefreshTokenLeaves() {
	ref := s.f.connected()
	var seen store.ConnectorConnection
	r := s.f.router(s.f.interposed(func() { seen = s.f.stored(ref) }))
	s.f.due()

	credential, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.Equal(store.ConnectionNeedsReauthorization, seen.Status, "a crash from here on leaves a connection nobody refreshes again")
	s.Equal(lostRefresh, seen.LastError)
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Empty(stored.LastError)
	s.True(s.f.works(credential))
}

func (s *ResolverSuite) TestALostRefreshAnswerLeavesNeedsReauthorizationAndTheTokenIsNeverSentAgain() {
	ref := s.f.connected()
	before := s.f.stored(ref)
	s.f.srv.Use(fakeprovider.LostResponse)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(lostRefresh, stored.LastError)
	s.Equal(before.Revision, stored.Revision, "no credentials were saved")

	_, err = s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
	s.Equal(1, s.f.srv.Refreshes(), "the next router does not send the spent refresh token")
}

func (s *ResolverSuite) TestCancellingTheCallerDuringARefreshLeavesTheConnectionConnectedAndRotated() {
	ref := s.f.connected()
	before := s.f.stored(ref).Revision
	ctx, cancel := context.WithCancel(s.f.ctx)
	returned := make(chan struct{})
	r := s.f.router(s.f.interposed(func() {
		cancel()
		<-returned
	}))
	s.f.due()

	_, err := r.Resolve(ctx, ref, core.CredentialRequest{})
	close(returned)
	s.ErrorIs(err, context.Canceled, "the caller does not wait for a refresh it gave up on")

	s.Eventually(func() bool { return s.f.stored(ref).Revision == before+1 }, 5*time.Second, 10*time.Millisecond,
		"the refresh finishes and is committed without the caller")
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Empty(stored.LastError)
	credential, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
	s.Equal(1, s.f.srv.Refreshes(), "the rotated token is the one stored, so nothing is refreshed again")
}

func (s *ResolverSuite) TestARejectedGrantMovesTheConnectionToNeedsReauthorization() {
	ref := s.f.connected()
	s.f.srv.Use(fakeprovider.InvalidGrant)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(rejectedGrant, stored.LastError)
}

func (s *ResolverSuite) TestAProviderThatIsDownBeforeExpiryHandsOutTheValidTokenAndKeepsTheConnection() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.f.srv.Use(fakeprovider.Unavailable)
	// Inside oauth2code's one-minute margin, not yet expired.
	s.f.clock.Add(fakeprovider.AccessTTL - 30*time.Second)

	credential, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Equal(temporarilyUnavailable, stored.LastError)
}

func (s *ResolverSuite) TestAProviderThatIsDownAfterExpiryIsTemporaryAndKeepsTheConnection() {
	ref := s.f.connected()
	s.f.srv.Use(fakeprovider.Unavailable)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrTemporarilyUnavailable)
	var outcome *core.OutcomeError
	s.Require().ErrorAs(err, &outcome)
	s.Equal(core.OutcomeTransient, outcome.Outcome.Kind)
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status)
	s.Equal(temporarilyUnavailable, stored.LastError)

	s.f.srv.Use()
	credential, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
	s.Empty(s.f.stored(ref).LastError, "a refresh that works clears the message")
}
