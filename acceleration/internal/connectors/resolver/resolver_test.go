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
	revoked                = "The provider says the grant was revoked; reconnect the account"
	uninstalled            = "The provider says its app was uninstalled from the account; reconnect the account"
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

func (s *ResolverSuite) TestACredentialThatExpiresBeforeTheCallsDeadlineIsRenewedForItAndThenCached() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	first, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	// Ten minutes into the token's hour, far outside oauth2code's one-minute margin, a call
	// that runs past the token's expiry.
	s.f.clock.Add(10 * time.Minute)
	deadline := first.ExpiresAt.Add(time.Minute)

	renewed, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{Deadline: deadline})
	s.Require().NoError(err)
	s.Equal(1, s.f.srv.Refreshes(), "renewed for the call, not for the margin")
	s.True(renewed.ExpiresAt.After(deadline), "the call never starts with a credential that dies during it")
	s.True(s.f.works(renewed))

	s.f.hold(ref)
	ctx, cancel := context.WithTimeout(s.f.ctx, blocked)
	defer cancel()
	again, err := r.Resolve(ctx, ref, core.CredentialRequest{Deadline: deadline})
	s.Require().NoError(err, "the next call inside the window takes no lock")
	s.True(s.f.token(renewed) == s.f.token(again), "the renewed token, from the cache")
	s.Equal(1, s.f.srv.Refreshes())
}

func (s *ResolverSuite) TestACredentialRenewedForADeadlineNoTokenReachesIsServedFromTheCache() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	s.f.clock.Add(10 * time.Minute)
	// Two hours: longer than any token the fake issues (fakeprovider.AccessTTL).
	request := core.CredentialRequest{Deadline: s.f.clock.Now().Add(2 * time.Hour)}

	renewed, err := r.Resolve(s.f.ctx, ref, request)
	s.Require().NoError(err)
	s.Equal(1, s.f.srv.Refreshes())

	s.f.hold(ref)
	ctx, cancel := context.WithTimeout(s.f.ctx, blocked)
	defer cancel()
	again, err := r.Resolve(ctx, ref, request)
	s.Require().NoError(err, "renewing again gives no longer token, so the call takes no lock")
	s.True(s.f.token(renewed) == s.f.token(again))
	s.Equal(1, s.f.srv.Refreshes(), "one refresh, not one per call")
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

// TestAConnectionOnABrokenRevisionGetsNoCredentialAndKeepsItsStatus: a later revision of a
// built-in marks the one a connected connection reads broken. Resolve refuses it as a
// connection that needs a reconnect, writes nothing, and once a consent moves it to the
// latest revision it resolves again.
func (s *ResolverSuite) TestAConnectionOnABrokenRevisionGetsNoCredentialAndKeepsItsStatus() {
	s.f.builtin(1, "")
	ref := s.f.connected()
	before := s.f.stored(ref)
	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err, "unmarked, it resolves")

	s.f.builtin(2, "broken_revisions:\n  - revisions: [1]\n    reason: reads the wrong path\n")
	_, err = s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.ErrorIs(err, resolver.ErrNotConnected)
	s.ErrorContains(err, "revision 1 of acme is broken (reads the wrong path)")
	after := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, after.Status)
	s.Equal(before.Revision, after.Revision)
	s.Equal(before.LastError, after.LastError)

	s.f.update(ref, func(state *core.CredentialState) { state.DefinitionRevision = 2 })
	credential, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err, "on the latest revision it resolves again")
	s.True(s.f.works(credential))
}

// TestAConnectionOnAnOutdatedRevisionKeepsWorking: a later revision that marks nothing broken
// leaves the connection on the revision it was made from resolving.
func (s *ResolverSuite) TestAConnectionOnAnOutdatedRevisionKeepsWorking() {
	s.f.builtin(1, "")
	ref := s.f.connected()
	s.f.builtin(2, "")

	credential, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().NoError(err)
	s.True(s.f.works(credential))
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
	rejected, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	_, err = here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
}

func (s *ResolverSuite) TestInvalidateMovesTheConnectionToNeedsReauthorization() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	rejected, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.Equal(s.f.stored(ref).Revision, rejected.Revision, "the revision the credential came from")

	s.Require().NoError(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(rejectedGrant, stored.LastError)
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
}

func (s *ResolverSuite) TestInvalidateOfACredentialAnotherRouterRenewedSinceKeepsTheConnection() {
	ref := s.f.connected()
	here := s.f.router(s.f.srv.Client())
	rejected, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	// Another router, a pool of its own, renews for a call that runs past rejected's expiry.
	there := s.f.router(s.f.srv.Client())
	renewed, err := there.Resolve(s.f.ctx, ref, core.CredentialRequest{Deadline: rejected.ExpiresAt.Add(time.Minute)})
	s.Require().NoError(err)
	s.Require().Equal(1, s.f.srv.Refreshes())
	s.Require().Equal(rejected.Revision+1, renewed.Revision)

	s.Require().NoError(here.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	stored := s.f.stored(ref)
	s.Equal(store.ConnectionConnected, stored.Status, "the provider refused an old token, not the grant")
	s.Empty(stored.LastError)
	credential, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.token(credential) == s.f.token(renewed), "the cache entry is gone, so the renewed token comes back")
}

func (s *ResolverSuite) TestInvalidateOfACredentialThatHadExpiredKeepsTheConnection() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	rejected, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.f.due()

	s.Require().NoError(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))
	s.Equal(store.ConnectionConnected, s.f.stored(ref).Status, "an expired token says nothing about the grant")
	credential, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.True(s.f.works(credential))
	s.Equal(1, s.f.srv.Refreshes())
}

func (s *ResolverSuite) TestInvalidateRefusesAnOutcomeThatEndsNoGrant() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())

	rejected, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Error(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeRateLimited}))
	s.Equal(store.ConnectionConnected, s.f.stored(ref).Status)
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.NoError(err)
}

// A revocation from another router reaches this one's next Resolve, which reads the row and
// fails without a request to the provider, though this router still holds the credential.
func (s *ResolverSuite) TestRevokeOnAnotherRouterFailsTheNextResolveHereFast() {
	ref := s.f.connected()
	here := s.f.router(s.f.srv.Client())
	_, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, time.Time{}))

	stored := s.f.stored(ref)
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(revoked, stored.LastError)
	_, err = here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.ErrorIs(err, resolver.ErrNotConnected)
	s.Zero(s.f.srv.Refreshes(), "nothing reached the provider")
}

// Invalidate keeps a connection whose stored credentials moved past the refused one; Revoke
// names no credential, so a renewal on another router does not save the grant.
func (s *ResolverSuite) TestRevokeEndsAGrantAnotherRouterJustRenewed() {
	ref := s.f.connected()
	s.f.due()
	renewed, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	s.Require().Equal(1, s.f.srv.Refreshes())

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalUninstalled, time.Time{}))

	stored := s.f.stored(ref)
	s.Equal(renewed.Revision, stored.Revision, "the stored credentials stay; only the status moves")
	s.Equal(store.ConnectionNeedsReauthorization, stored.Status)
	s.Equal(uninstalled, stored.LastError)
}

// Slack retries an event it got no 2xx for up to 5 minutes later
// (https://docs.slack.dev/apis/events-api/, «Retries»). A tokens_revoked that ended the old
// grant, retried after the account reconnected, is not about the new one.
func (s *ResolverSuite) TestRevokeLeavesAGrantConnectedAfterTheSignalConnected() {
	ref := s.f.connected()
	endedAt := time.Now().UTC().Add(-2 * time.Minute)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, endedAt))

	s.Equal(store.ConnectionConnected, s.f.stored(ref).Status)
}

func (s *ResolverSuite) TestRevokeEndsAGrantConnectedBeforeTheSignal() {
	ref := s.f.connected()
	endedAt := time.Now().UTC().Add(time.Minute)

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, endedAt))

	s.Equal(store.ConnectionNeedsReauthorization, s.f.stored(ref).Status)
}

func (s *ResolverSuite) TestRevokeLeavesAPendingConnectionPending() {
	ref := s.f.pending()

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, time.Time{}))

	s.Equal(store.ConnectionPending, s.f.stored(ref).Status)
}

func (s *ResolverSuite) TestRevokeRefusesAKindThatIsNoSignal() {
	ref := s.f.connected()

	s.Error(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalKind("paused"), time.Time{}))
	s.Equal(store.ConnectionConnected, s.f.stored(ref).Status)
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
