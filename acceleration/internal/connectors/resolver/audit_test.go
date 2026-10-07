//go:build integration

package resolver_test

import (
	"context"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// audited is ref's audit rows, newest first, once they hold count and a moment has passed in
// which a second row would have been written too.
func (s *ResolverSuite) audited(ref core.ConnectionRef, count int) []store.ConnectorAuditEvent {
	read := func() []store.ConnectorAuditEvent {
		rows, err := s.db.ConnectorAuditEvents(s.f.ctx, ref.CustomerID, store.AuditFilter{ConnectionID: ref.ConnectionID})
		s.Require().NoError(err)
		return rows
	}
	s.Require().Eventually(func() bool { return len(read()) >= count }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return len(read()) > count }, 200*time.Millisecond, 20*time.Millisecond)
	return read()
}

func (s *ResolverSuite) TestAuditOfARefreshIsOneRowAtTheNewRevisionWithItsCorrelation() {
	ref := s.f.connected()
	s.f.due()
	ctx := core.WithCorrelation(s.f.ctx, core.Correlation{RequestID: "request-1", SessionID: "session-1"})

	_, err := s.f.router(s.f.srv.Client()).Resolve(ctx, ref, core.CredentialRequest{})

	s.Require().NoError(err)
	rows := s.audited(ref, 1)
	s.Equal(store.AuditGrantRefreshed, rows[0].Action)
	s.Equal(s.f.stored(ref).Revision, rows[0].Revision)
	s.Equal("request-1", rows[0].RequestID)
	s.Equal("session-1", rows[0].SessionID)
	s.Equal("custom_acme", rows[0].ConnectorID)
	s.Equal(store.OwnerApp, rows[0].OwnerType)
}

func (s *ResolverSuite) TestAuditOfACredentialThatNeedsNoRenewalIsNothing() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())

	_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	_, err = r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Empty(s.audited(ref, 0))
}

func (s *ResolverSuite) TestAuditOfRoutersRefreshingAtOnceIsOneRow() {
	ref := s.f.connected()
	s.f.due()
	var wg sync.WaitGroup
	for range routers {
		r := s.f.router(s.f.srv.Client())
		wg.Go(func() {
			_, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
			s.NoError(err)
		})
	}
	wg.Wait()

	s.Require().Equal(1, s.f.srv.Refreshes())
	s.Len(s.audited(ref, 1), 1, "the routers that found the refresh done write nothing")
}

func (s *ResolverSuite) TestAuditOfARefreshTheCallerGaveUpOnIsStillWritten() {
	ref := s.f.connected()
	ctx, cancel := context.WithCancel(s.f.ctx)
	returned := make(chan struct{})
	r := s.f.router(s.f.interposed(func() {
		cancel()
		<-returned
	}))
	s.f.due()

	_, err := r.Resolve(ctx, ref, core.CredentialRequest{})
	close(returned)

	s.ErrorIs(err, context.Canceled)
	s.Equal(store.AuditGrantRefreshed, s.audited(ref, 1)[0].Action)
}

func (s *ResolverSuite) TestAuditOfARefreshTheProviderRejectsIsOneRevokedRow() {
	ref := s.f.connected()
	s.f.srv.Use(fakeprovider.InvalidGrant)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().Error(err)
	rows := s.audited(ref, 1)
	s.Equal(store.AuditGrantRevoked, rows[0].Action)
	s.Equal(string(core.OutcomeInvalidGrant), rows[0].Reason)
}

func (s *ResolverSuite) TestAuditOfAProviderThatIsDownIsNothing() {
	ref := s.f.connected()
	s.f.srv.Use(fakeprovider.Unavailable)
	s.f.due()

	_, err := s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{})

	s.Require().Error(err)
	s.Empty(s.audited(ref, 0), "the grant stands and nothing was renewed")
}

func (s *ResolverSuite) TestAuditOfInvalidateIsOneRevokedRow() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())
	rejected, err := r.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)

	s.Require().NoError(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeScopeRequired}))
	s.Require().NoError(r.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeScopeRequired}))

	rows := s.audited(ref, 1)
	s.Equal(store.AuditGrantRevoked, rows[0].Action)
	s.Equal(string(core.OutcomeScopeRequired), rows[0].Reason)
}

func (s *ResolverSuite) TestAuditOfInvalidatingACredentialAnotherRouterRenewedIsNothing() {
	ref := s.f.connected()
	here := s.f.router(s.f.srv.Client())
	rejected, err := here.Resolve(s.f.ctx, ref, core.CredentialRequest{})
	s.Require().NoError(err)
	_, err = s.f.router(s.f.srv.Client()).Resolve(s.f.ctx, ref, core.CredentialRequest{Deadline: rejected.ExpiresAt.Add(time.Minute)})
	s.Require().NoError(err)

	s.Require().NoError(here.Invalidate(s.f.ctx, ref, rejected, core.Outcome{Kind: core.OutcomeInvalidGrant}))

	rows := s.audited(ref, 1)
	s.Equal(store.AuditGrantRefreshed, rows[0].Action, "only the renewal; the grant stands")
}

func (s *ResolverSuite) TestAuditOfRevokeIsOneRevokedRowAndOfASecondNothing() {
	ref := s.f.connected()
	r := s.f.router(s.f.srv.Client())

	s.Require().NoError(r.Revoke(s.f.ctx, ref, core.SignalUninstalled, time.Time{}))
	s.Require().NoError(r.Revoke(s.f.ctx, ref, core.SignalUninstalled, time.Time{}))

	rows := s.audited(ref, 1)
	s.Equal(store.AuditGrantRevoked, rows[0].Action)
	s.Equal(string(core.SignalUninstalled), rows[0].Reason)
}

func (s *ResolverSuite) TestAuditOfRevokingAPendingConnectionIsNothing() {
	ref := s.f.pending()

	s.Require().NoError(s.f.router(s.f.srv.Client()).Revoke(s.f.ctx, ref, core.SignalRevoked, time.Time{}))

	s.Empty(s.audited(ref, 0))
}
