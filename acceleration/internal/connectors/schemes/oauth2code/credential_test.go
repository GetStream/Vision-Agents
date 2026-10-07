package oauth2code_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

func (s *OAuth2CodeSuite) TestAccessCredentialHandsOutTheTokenAsItIsUntilItIsInsideTheMargin() {
	srv := fakeprovider.New(s.T())
	scheme, stored := s.connected(srv, s.preregistered(srv), nil)

	credential, again, err := scheme.Retrieve(s.ctx, stored, s.preregistered(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
	s.Equal(stored, again, "nothing was renewed, so the stored credentials are the ones given")
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestAccessCredentialRefreshesInsideTheDefaultMinuteBeforeExpiry() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 59*time.Second)
	credential, rotated, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.NotEqual(s.accessToken(stored), s.accessToken(rotated))
	s.NotEqual(s.refreshToken(stored), s.refreshToken(rotated), "the fake rotated, so the new refresh token replaces the old")
	s.Equal(s.now.Add(fakeprovider.AccessTTL).Unix(), credential.ExpiresAt.Unix())
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestOutsideTheDefaultMinuteAccessCredentialDoesNotRefresh() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 61*time.Second)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestTheManifestMarginDecidesWhenARefreshIsDue() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Refresh.Margin = core.Duration(10 * time.Minute)
	scheme, stored := s.connected(srv, resolved, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 9*time.Minute)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestARefreshSendsNoScopeUnlessThePolicySays() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.accessCredentialDue(scheme, stored, resolved)
	_, sent := srv.RefreshScope()
	s.False(sent)
}

func (s *OAuth2CodeSuite) TestARefreshSendsTheGrantedScopesWhenThePolicySays() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Scopes.SendOnRefresh = true
	scheme, stored := s.connected(srv, resolved, nil)

	s.accessCredentialDue(scheme, stored, resolved)
	scope, sent := srv.RefreshScope()
	s.True(sent)
	s.Equal("files:read files:write", scope)
}

func (s *OAuth2CodeSuite) TestATokenThatExpiresBeforeValidUntilIsRefreshedOutsideTheMargin() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	before, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)

	// Ten minutes left, well outside the one-minute margin, and a call that needs more.
	s.now = before.ExpiresAt.Add(-10 * time.Minute)
	credential, rotated, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{ValidUntil: before.ExpiresAt})
	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.NotEqual(s.accessToken(stored), s.accessToken(rotated))
	s.True(credential.ExpiresAt.After(before.ExpiresAt), "the new token outlives the call")
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestATokenThatOutlivesValidUntilIsNotRefreshed() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	before, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)

	s.now = before.ExpiresAt.Add(-10 * time.Minute)
	_, again, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{ValidUntil: before.ExpiresAt.Add(-time.Second)})
	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
	s.Equal(stored, again)
}

func (s *OAuth2CodeSuite) TestARefreshCheckpointsOnceBeforeTheTokenLeaves() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	var refreshesAtCheckpoint []int
	opts := core.RetrieveOptions{Checkpoint: func() error {
		refreshesAtCheckpoint = append(refreshesAtCheckpoint, srv.Refreshes())
		return nil
	}}

	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, opts)
	s.Require().NoError(err)
	s.Equal([]int{0}, refreshesAtCheckpoint, "one checkpoint, before the refresh reached the provider")
	s.Equal(1, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestARefreshWhoseCheckpointFailsSendsNothing() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	failed := errors.New("the checkpoint did not commit")
	opts := core.RetrieveOptions{Checkpoint: func() error { return failed }}

	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, returned, err := scheme.Retrieve(s.ctx, stored, resolved, opts)
	s.ErrorIs(err, failed)
	s.Equal(0, srv.Refreshes(), "the refresh token never left")
	s.Equal(core.StoredCredentials{}, returned)
}

func (s *OAuth2CodeSuite) TestAccessCredentialThatIsNotDueNeverCheckpoints() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	checkpoints := 0
	opts := core.RetrieveOptions{Checkpoint: func() error {
		checkpoints++
		return nil
	}}

	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, opts)
	s.Require().NoError(err)
	s.Zero(checkpoints, "nothing that cannot be taken back was sent")
}

func (s *OAuth2CodeSuite) TestALostRefreshIsUncertainAndLeavesTheStoredCredentialsAlone() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	before := bytes.Clone(stored.Payload)
	srv.Use(fakeprovider.LostResponse)

	s.now = s.now.Add(fakeprovider.AccessTTL)
	credential, returned, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(1, srv.Refreshes(), "without a grace window the spent token is never sent again")
	s.Equal(before, []byte(stored.Payload), "the caller's stored credentials are untouched")
	s.Equal(core.StoredCredentials{}, returned, "no stored credentials to persist")
	s.Equal(core.AccessCredential{}, credential)
}

func (s *OAuth2CodeSuite) TestALostRefreshInsideTheGraceWindowIsRetriedOnceWithTheOldToken() {
	srv := fakeprovider.New(s.T())
	resolved := s.registered(srv)
	s.Require().Equal(core.Duration(30*time.Minute), resolved.Refresh.Grace, "the Linear manifest's grace")
	scheme, stored := s.connected(srv, resolved, nil)
	srv.Use(fakeprovider.LostResponseOnce, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	credential, rotated, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, srv.Refreshes(), "the lost one and one retry")
	s.NotEqual(s.refreshToken(stored), s.refreshToken(rotated))
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestTheGraceRetryIsMadeOnlyOnce() {
	srv := fakeprovider.New(s.T())
	resolved := s.registered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	srv.Use(fakeprovider.LostResponse, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(2, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestNoGraceRetryOnceTheWindowHasClosed() {
	srv := fakeprovider.New(s.T())
	resolved := s.registered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	srv.Use(fakeprovider.LostResponseOnce, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	// Every read of the clock now moves it a whole grace window, so the answer of the first
	// attempt arrives after the window it opened has closed.
	s.step = time.Duration(resolved.Refresh.Grace)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(1, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestANonRotatingRefreshKeepsTheRefreshToken() {
	srv := fakeprovider.New(s.T(), fakeprovider.NonRotatingRefresh)
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	renewed := s.accessCredentialDue(scheme, stored, resolved)
	s.Equal(s.refreshToken(stored), s.refreshToken(renewed))
	s.NotEqual(s.accessToken(stored), s.accessToken(renewed))
}

func (s *OAuth2CodeSuite) TestAnExpiredTokenWithNoRefreshTokenIsInvalidGrant() {
	srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken)
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - time.Second)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err, "inside the margin and still valid, it is handed out")

	s.now = s.now.Add(time.Second)
	_, _, err = scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeInvalidGrant, s.outcome(err).Kind)
	s.Equal(0, srv.Refreshes())
}

// TestARefusedAccessTokenIsRefreshedWhateverItsExpiry: the provider refused a token an hour
// from expiry (core.RetrieveOptions.Refused), so it is refreshed although it is not due.
func (s *OAuth2CodeSuite) TestARefusedAccessTokenIsRefreshedWhateverItsExpiry() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	_, renewed, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{Refused: true})

	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.True(s.accessToken(stored) != s.accessToken(renewed), "a new access token")
}

// TestARefusedAccessTokenWithNoRefreshTokenComesBackAsStored: nothing renews it, so the
// caller sees the same credential and tells the resolver the grant is gone.
func (s *OAuth2CodeSuite) TestARefusedAccessTokenWithNoRefreshTokenComesBackAsStored() {
	srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken)
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	_, returned, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{Refused: true})

	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
	s.True(bytes.Equal(stored.Payload, returned.Payload))
}

// TestAFailedRefreshOfARefusedTokenHandsBackNoToken: the refused token is not handed back
// beside the error, however long it has left, since the provider already refused it.
func (s *OAuth2CodeSuite) TestAFailedRefreshOfARefusedTokenHandsBackNoToken() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	srv.Use(fakeprovider.Unavailable)

	credential, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{Refused: true})

	s.Equal(core.OutcomeTransient, s.outcome(err).Kind)
	s.Empty(credential.Scheme, "no access credential beside the error")
}

// TestEachRefusedRefreshIsTheOutcomeItsAnswerMeans is the table of what the token endpoint
// can answer a refresh with and what AccessCredential makes of it. The caller's stored
// credentials never change and none come back.
func (s *OAuth2CodeSuite) TestEachRefusedRefreshIsTheOutcomeItsAnswerMeans() {
	for _, row := range []struct {
		personality fakeprovider.Personality
		want        core.Outcome
	}{
		{fakeprovider.InvalidGrant, core.Outcome{Kind: core.OutcomeInvalidGrant}},
		{fakeprovider.Unavailable, core.Outcome{Kind: core.OutcomeTransient}},
		{fakeprovider.ServerError, core.Outcome{Kind: core.OutcomeUncertain}},
		{fakeprovider.LostResponse, core.Outcome{Kind: core.OutcomeUncertain}},
		{fakeprovider.RateLimited, core.Outcome{Kind: core.OutcomeRateLimited, RetryAfter: fakeprovider.RetryAfter}},
	} {
		srv := fakeprovider.New(s.T())
		resolved := s.preregistered(srv)
		scheme, stored := s.connected(srv, resolved, nil)
		before := bytes.Clone(stored.Payload)
		srv.Use(row.personality)

		s.now = s.now.Add(fakeprovider.AccessTTL)
		_, returned, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
		s.Equal(row.want, s.outcome(err), row.personality)
		s.Equal(before, []byte(stored.Payload), row.personality)
		s.Equal(core.StoredCredentials{}, returned, row.personality)
	}
}

// TestARefreshThatFailsBeforeExpiryStillHandsOutTheValidToken is the table of refusals
// inside the margin: each comes back as its outcome, with the token that has not expired yet
// and no stored credentials.
func (s *OAuth2CodeSuite) TestARefreshThatFailsBeforeExpiryStillHandsOutTheValidToken() {
	for _, row := range []struct {
		personality fakeprovider.Personality
		want        core.OutcomeKind
	}{
		{fakeprovider.Unavailable, core.OutcomeTransient},
		{fakeprovider.RateLimited, core.OutcomeRateLimited},
		{fakeprovider.ServerError, core.OutcomeUncertain},
		{fakeprovider.LostResponse, core.OutcomeUncertain},
		{fakeprovider.InvalidGrant, core.OutcomeInvalidGrant},
	} {
		srv := fakeprovider.New(s.T())
		resolved := s.preregistered(srv)
		scheme, stored := s.connected(srv, resolved, nil)
		srv.Use(row.personality)

		s.now = s.now.Add(fakeprovider.AccessTTL - 30*time.Second)
		credential, returned, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
		s.Equal(row.want, s.outcome(err).Kind, row.personality)
		s.Equal(core.StoredCredentials{}, returned, row.personality)
		// Off again, since RateLimited answers resource calls with 429 too.
		srv.Use()
		s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential), row.personality)
	}
}

func (s *OAuth2CodeSuite) TestARefreshRefusedWithAnErrorCodeClassifyDoesNotNameIsTransient() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	resolved := s.preregistered(srv)
	resolved.Scopes.Separator = ","
	secret := srv.ClientSecret
	lookup := func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, source core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: srv.ClientID, Secret: secret}, source == core.ClientOperator, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)

	secret = "rotated-elsewhere"
	s.now = s.now.Add(fakeprovider.SlackAccessTTL)
	_, _, err = scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind, "200 ok:false invalid_client_id is a refusal, not a lost token")
	var refused *oauth2code.TokenError
	s.Require().ErrorAs(err, &refused)
	s.Equal("invalid_client_id", refused.Code)
}

func (s *OAuth2CodeSuite) TestARefusalWhoseBodyWasCutOffIsNotRetriedInTheGraceWindow() {
	srv := fakeprovider.New(s.T())
	resolved := s.registered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	srv.Use(fakeprovider.CutOffRefusal)

	s.now = s.now.Add(24 * time.Hour)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind, "the 400 arrived, so nothing was spent")
	s.Equal(1, srv.Refreshes(), "a refusal is not sent again")
}

func (s *OAuth2CodeSuite) TestARefreshLooksThePreregisteredClientSecretUpAgain() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	secret := srv.ClientSecret
	lookup := func(_ context.Context, ref core.ConnectionRef, _ core.ResolvedManifest, source core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		s.Equal(s.ref, ref, "the connection Complete ran for")
		return oauth2code.Client{ID: srv.ClientID, Secret: secret}, source == core.ClientOperator, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.NotContains(string(stored.Payload), srv.ClientSecret, "a preregistered secret is never sealed")

	secret = "rotated-elsewhere"
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err = scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind, "the fake refuses the new secret with invalid_client")
	var refused *oauth2code.TokenError
	s.Require().ErrorAs(err, &refused)
	s.Equal("invalid_client", refused.Code)
}

// DELETE /v1/agents/connectors/{id}/oauth-client (AI-846) removes the app's client record.
// The grant was issued to that client, so the refresh says the client is gone and needs a
// reconnect, sends nothing, and is InvalidGrant, which the resolver turns into
// needs_reauthorization.
func (s *OAuth2CodeSuite) TestARefreshAfterTheAppsClientWasRemovedSaysSoAndIsInvalidGrant() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Client.Registration = []core.ClientRegistrationMethod{core.ClientCustomer}
	removed := false
	lookup := func(context.Context, core.ConnectionRef, core.ResolvedManifest, core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, !removed, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)

	removed = true
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err = scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Equal(core.OutcomeInvalidGrant, s.outcome(err).Kind)
	var gone *oauth2code.ClientRemovedError
	s.Require().ErrorAs(err, &gone)
	s.Equal(core.ClientCustomer, gone.Registration)
	s.ErrorContains(err, "the app's OAuth client for this connector was removed; the connection needs a reconnect")
	s.NotContains(err.Error(), "changed during the consent")
	s.Equal(0, srv.Refreshes(), "nothing was sent")
}

// A client record that is there but names another client_id is still the other case: the
// app registered a new client while the consent was open.
func (s *OAuth2CodeSuite) TestARefreshWithAnotherClientInTheRecordSaysTheClientChanged() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Client.Registration = []core.ClientRegistrationMethod{core.ClientCustomer}
	id := srv.ClientID
	lookup := func(context.Context, core.ConnectionRef, core.ResolvedManifest, core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: id, Secret: srv.ClientSecret}, true, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)

	id = "another-client"
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err = scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.ErrorContains(err, "the customer client changed during the consent")
	var gone *oauth2code.ClientRemovedError
	s.False(errors.As(err, &gone))
}

func (s *OAuth2CodeSuite) TestARefreshTokenDyingBeforeTheNextRefreshIsWarnedAboutWithoutATokenInTheLog() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Refresh.RefreshTTL = core.Duration(30 * time.Minute)
	var log bytes.Buffer
	scheme, stored := s.connected(srv, resolved, slog.New(slog.NewJSONHandler(&log, nil)))

	renewed := s.accessCredentialDue(scheme, stored, resolved)
	s.Contains(log.String(), "the refresh token expires before the next refresh")
	s.Contains(log.String(), `"connection":"conn-1"`)
	for _, secret := range []string{s.accessToken(renewed), s.refreshToken(renewed), s.refreshToken(stored), srv.ClientSecret} {
		s.NotContains(log.String(), secret)
	}
}

func (s *OAuth2CodeSuite) TestARefreshTokenThatOutlivesTheNextRefreshIsNotWarnedAbout() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Refresh.RefreshTTL = core.Duration(90 * 24 * time.Hour)
	var log bytes.Buffer
	scheme, stored := s.connected(srv, resolved, slog.New(slog.NewJSONHandler(&log, nil)))

	s.accessCredentialDue(scheme, stored, resolved)
	s.Empty(log.String())
}

func (s *OAuth2CodeSuite) TestARefreshEndpointThatIsNotPublicIsRefusedBeforeTheTokenLeaves() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	resolved.Endpoints["refresh"] = "https://10.0.0.1/token"

	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().ErrorContains(err, "egress:")
	s.Equal(0, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestWrapSendsTheAccessTokenAsABearerHeaderOnACopyOfTheRequest() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)
	credential, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)

	client := &http.Client{Transport: scheme.Wrap(srv.Client().Transport, credential)}
	request := s.toolCall(srv)
	response, err := client.Do(request)
	s.Require().NoError(err)
	s.Require().NoError(response.Body.Close())
	s.Equal(http.StatusOK, response.StatusCode)
	s.Empty(request.Header.Get("Authorization"), "the caller's request is not modified")
	s.Equal(http.StatusUnauthorized, s.status(srv.Client().Do(s.toolCall(srv))), "without Wrap the fake refuses")
}

func (s *OAuth2CodeSuite) TestWrapWithoutAnAccessTokenSendsNothing() {
	srv := fakeprovider.New(s.T())
	scheme := s.scheme(srv.Client(), oauth2code.Config{})
	client := &http.Client{Transport: scheme.Wrap(srv.Client().Transport, core.AccessCredential{})}
	_, err := client.Do(s.toolCall(srv))
	s.Require().Error(err)
	s.Equal(0, srv.Hits(fakeprovider.PathMCP))
}

func (s *OAuth2CodeSuite) TestClassifyMakesARefusedGrantInvalidGrant() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
	}{
		{"RFC 6749 invalid_grant with 400", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.InvalidGrant)
			return s.refreshAnswer(srv, s.connectedToken(srv)["refresh_token"])
		}},
		{"Slack's invalid_refresh_token with 200", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes, fakeprovider.InvalidGrant)
			return s.refreshAnswer(srv, "a-refresh-token-nobody-issued")
		}},
		{"invalid_refresh_token with 400, the prototype's test shape", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusBadRequest, nil, `{"error":"invalid_refresh_token"}`)
		}},
		{"a 401 invalid_token from the resource", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken)
			access := s.connectedToken(srv)["access_token"]
			srv.Advance(fakeprovider.AccessTTL)
			return s.mcpAnswer(srv, access)
		}},
		{"a bare 401 from an MCP server, resource_metadata and no error", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.BareChallenge)
			access := s.connectedToken(srv)["access_token"]
			srv.Advance(fakeprovider.AccessTTL)
			return s.mcpAnswer(srv, access)
		}},
		{"a 401 with no challenge to a request that carried a bearer token", func() (*http.Response, []byte, error) {
			return s.bare401(http.Header{"Authorization": {"bearer any"}})
		}},
		{"a Bearer invalid_token after another scheme's padded token68", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusUnauthorized, http.Header{"Www-Authenticate": {`Newauth abc==, Bearer error="invalid_token"`}}, "")
		}},
		{"Slack's chat.postMessage invalid_auth with 200", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusOK, nil, `{"ok":false,"error":"invalid_auth"}`)
		}},
		{"Slack's token_revoked with 200", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusOK, nil, `{"ok":false,"error":"token_revoked"}`)
		}},
		{"Slack's account_inactive with 200", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusOK, nil, `{"ok":false,"error":"account_inactive"}`)
		}},
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestClassifyMakesAnAnswerThatMayHaveTakenEffectUncertain() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
	}{
		{"a connection closed after the rotation", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.LostResponse)
			return s.refreshAnswer(srv, s.connectedToken(srv)["refresh_token"])
		}},
		{"a 500 server_error after the rotation", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.ServerError)
			return s.refreshAnswer(srv, s.connectedToken(srv)["refresh_token"])
		}},
		{"Slack's internal_error with 200", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes, fakeprovider.ServerError)
			return s.refreshAnswer(srv, s.connectedToken(srv)["refresh_token"])
		}},
		{"Slack's fatal_error with 200", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusOK, nil, `{"ok":false,"error":"fatal_error"}`)
		}},
		{"a 502 from a gateway", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusBadGateway, nil, "")
		}},
		{"a 504 from a gateway", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusGatewayTimeout, nil, "")
		}},
		{"a 200 whose body was cut off", func() (*http.Response, []byte, error) {
			return s.cutOff(http.StatusOK, nil)
		}},
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeUncertain}, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestClassifyMakesAFailureThatChangedNothingTransient() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
		want   core.Outcome
	}{
		{"a 503", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.Unavailable)
			return s.refreshAnswer(srv, "any")
		}, core.Outcome{Kind: core.OutcomeTransient}},
		{"a 503 with Retry-After", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusServiceUnavailable, http.Header{"Retry-After": {"120"}}, "")
		}, core.Outcome{Kind: core.OutcomeTransient, RetryAfter: 2 * time.Minute}},
		{"a 503 whose body was cut off", func() (*http.Response, []byte, error) {
			return s.cutOff(http.StatusServiceUnavailable, nil)
		}, core.Outcome{Kind: core.OutcomeTransient}},
		{"a refused dial", func() (*http.Response, []byte, error) {
			closed := httptest.NewTLSServer(http.NotFoundHandler())
			closed.Close()
			response, err := closed.Client().Get(closed.URL)
			return response, nil, err
		}, core.Outcome{Kind: core.OutcomeTransient}},
		{"temporarily_unavailable", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusBadRequest, nil, `{"error":"temporarily_unavailable"}`)
		}, core.Outcome{Kind: core.OutcomeTransient}},
		{"invalid_client for a client that failed to authenticate", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T())
			return s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"refresh_token"}, "refresh_token": {"any"},
				"client_id": {srv.ClientID}, "client_secret": {"wrong"}})
		}, core.Outcome{Kind: core.OutcomeTransient}},
	} {
		s.Equal(row.want, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestClassifyMakesAScopeOrClaimsChallengeScopeRequired() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
		want   core.Outcome
	}{
		{"a 403 insufficient_scope", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.InsufficientScope)
			return s.mcpAnswer(srv, s.connectedToken(srv)["access_token"])
		}, core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"files:read", "files:write"}}},
		{"a 401 claims challenge", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.ClaimsChallenge)
			return s.mcpAnswer(srv, s.connectedToken(srv)["access_token"])
		}, core.Outcome{Kind: core.OutcomeScopeRequired, Claims: fakeprovider.ClaimsChallengeJSON}},
		{"a challenge after another scheme's, with escapes", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusForbidden, http.Header{"Www-Authenticate": {
				`Basic realm="a, b", Bearer realm="x\"y", error="insufficient_scope", scope="admin"`,
			}}, "")
		}, core.Outcome{Kind: core.OutcomeScopeRequired, Scopes: []string{"admin"}}},
	} {
		s.Equal(row.want, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestClassifyMakesA429RateLimitedWithItsRetryAfter() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
		want   time.Duration
	}{
		{"a resource request", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T())
			access := s.connectedToken(srv)["access_token"]
			srv.Use(fakeprovider.RateLimited)
			return s.mcpAnswer(srv, access)
		}, fakeprovider.RetryAfter},
		{"a refresh", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.RateLimited)
			return s.refreshAnswer(srv, "any")
		}, fakeprovider.RetryAfter},
		{"an HTTP-date", func() (*http.Response, []byte, error) {
			at := s.now.Add(2 * time.Minute).UTC().Format(http.TimeFormat)
			return s.synthetic(http.StatusTooManyRequests, http.Header{"Retry-After": {at}}, "")
		}, 2 * time.Minute},
		{"no Retry-After", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusTooManyRequests, nil, "")
		}, 0},
		{"a body that was cut off", func() (*http.Response, []byte, error) {
			return s.cutOff(http.StatusTooManyRequests, http.Header{"Retry-After": {"30"}})
		}, 30 * time.Second},
	} {
		got := s.classify(row.answer)
		s.Equal(core.OutcomeRateLimited, got.Kind, row.name)
		s.InDelta(row.want, got.RetryAfter, float64(time.Second), row.name)
	}
}

func (s *OAuth2CodeSuite) TestClassifyLeavesAnAnswerWithNothingForTheCoreOK() {
	for _, row := range []struct {
		name   string
		answer func() (*http.Response, []byte, error)
	}{
		{"a tool call that worked", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T())
			return s.mcpAnswer(srv, s.connectedToken(srv)["access_token"])
		}},
		{"a 404", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusNotFound, nil, "not found")
		}},
		{"a 404 with a resource's own error code", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusNotFound, nil, `{"error":"not_found"}`)
		}},
		{"a 200 ok:false with a resource's own error code, the Slack Web API shape", func() (*http.Response, []byte, error) {
			return s.synthetic(http.StatusOK, nil, `{"ok":false,"error":"channel_not_found"}`)
		}},
		{"a 400 whose body was cut off", func() (*http.Response, []byte, error) {
			return s.cutOff(http.StatusBadRequest, nil)
		}},
		{"a bare 401 to an MCP request that carried no token", func() (*http.Response, []byte, error) {
			srv := fakeprovider.New(s.T(), fakeprovider.BareChallenge)
			return s.read(srv.Client().Do(s.toolCall(srv)))
		}},
		// The token endpoint's request authenticates the client (RFC 6749 section 2.3.1), so
		// its bare 401 is the refusal redeem makes Transient, as before. The Basic credentials
		// are RFC 7617 section 2's example.
		{"a bare 401 from a token endpoint, to client_secret_basic", func() (*http.Response, []byte, error) {
			return s.bare401(http.Header{"Authorization": {"Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="}})
		}},
		{"a bare 401 from a token endpoint, to client_secret_post", func() (*http.Response, []byte, error) {
			return s.bare401(nil)
		}},
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeOK}, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestRevokeEndsTheGrantAtTheManifestsEndpoint() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Endpoints["revoke"] = srv.URL + fakeprovider.PathRevoke
	scheme, stored := s.connected(srv, resolved, nil)

	s.Require().NoError(scheme.Revoke(s.ctx, stored, resolved))
	s.Equal(1, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusUnauthorized, s.call(srv, s.accessToken(stored)), "revoking the refresh token ended the grant")
}

func (s *OAuth2CodeSuite) TestRevokeUsesTheEndpointTheServerAdvertised() {
	srv := fakeprovider.New(s.T())
	resolved := s.registered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.Require().NoError(scheme.Revoke(s.ctx, stored, resolved))
	s.Equal(1, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusUnauthorized, s.call(srv, s.accessToken(stored)))
}

func (s *OAuth2CodeSuite) TestRevokeWithNoEndpointSaysSoAndSendsNothing() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	scheme, stored := s.connected(srv, resolved, nil)

	s.ErrorIs(scheme.Revoke(s.ctx, stored, resolved), oauth2code.ErrNoRevocationEndpoint)
	s.Equal(0, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)))
}

func (s *OAuth2CodeSuite) TestRevokingAnAccessTokenTheProviderDoesNotRevokeSaysSo() {
	srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken, fakeprovider.AccessTokenNotRevocable)
	resolved := s.preregistered(srv)
	resolved.Endpoints["revoke"] = srv.URL + fakeprovider.PathRevoke
	scheme, stored := s.connected(srv, resolved, nil)

	err := scheme.Revoke(s.ctx, stored, resolved)
	s.ErrorIs(err, oauth2code.ErrTokenTypeNotRevocable)
	var failed *core.OutcomeError
	s.False(errors.As(err, &failed), "not a retryable outcome")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)), "nothing was revoked")
}

func (s *OAuth2CodeSuite) TestARefusedRevocationIsAnOutcomeError() {
	srv := fakeprovider.New(s.T())
	resolved := s.preregistered(srv)
	resolved.Endpoints["revoke"] = srv.URL + fakeprovider.PathRevoke
	secret := srv.ClientSecret
	lookup := func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, source core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: srv.ClientID, Secret: secret}, source == core.ClientOperator, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)

	secret = "rotated-elsewhere"
	err = scheme.Revoke(s.ctx, stored, resolved)
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)), "nothing was revoked")
}

// preregistered is the Slack fixture pointed at the fake, with the fake's preregistered
// client as the operator's (client_secret_basic), space-separated scopes and no refresh
// policy of its own, so each test sets the one field it is about.
func (s *OAuth2CodeSuite) preregistered(srv *fakeprovider.Server) core.ResolvedManifest {
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	resolved.Client.AuthMethod = ""
	resolved.Scopes = core.ScopePolicy{List: []string{"files:read", "files:write"}}
	resolved.Refresh = core.RefreshPolicy{}
	return resolved
}

// registered is the Linear manifest discovering the fake and registering a public client,
// with its own refresh policy (rotating, 30 minutes of grace).
func (s *OAuth2CodeSuite) registered(srv *fakeprovider.Server) core.ResolvedManifest {
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	return resolved
}

// connected is a scheme on the suite's clock and the stored credentials of one consent
// through it.
func (s *OAuth2CodeSuite) connected(srv *fakeprovider.Server, m core.ResolvedManifest, logger *slog.Logger) (*oauth2code.Scheme, core.StoredCredentials) {
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv), Now: s.clock, Logger: logger})
	stored, _, err := s.connect(srv, scheme, m)
	s.Require().NoError(err)
	return scheme, stored
}

// accessCredentialDue moves the clock past expiry and gets an access credential, which must
// refresh once and succeed.
func (s *OAuth2CodeSuite) accessCredentialDue(scheme *oauth2code.Scheme, stored core.StoredCredentials, m core.ResolvedManifest) core.StoredCredentials {
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, renewed, err := scheme.Retrieve(s.ctx, stored, m, core.RetrieveOptions{})
	s.Require().NoError(err)
	return renewed
}

func (s *OAuth2CodeSuite) clock() time.Time {
	now := s.now
	s.now = s.now.Add(s.step)
	return now
}

func (s *OAuth2CodeSuite) refreshToken(stored core.StoredCredentials) string {
	var payload struct {
		RefreshToken string `json:"refresh_token"`
	}
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	return payload.RefreshToken
}

func (s *OAuth2CodeSuite) outcome(err error) core.Outcome {
	var failed *core.OutcomeError
	s.Require().ErrorAs(err, &failed)
	return failed.Outcome
}

// wrapped is the status of a tool call sent through Wrap with credential.
func (s *OAuth2CodeSuite) wrapped(srv *fakeprovider.Server, scheme *oauth2code.Scheme, credential core.AccessCredential) int {
	client := &http.Client{Transport: scheme.Wrap(srv.Client().Transport, credential)}
	return s.status(client.Do(s.toolCall(srv)))
}

func (s *OAuth2CodeSuite) toolCall(srv *fakeprovider.Server) *http.Request {
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP,
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
	s.Require().NoError(err)
	return request
}

func (s *OAuth2CodeSuite) status(response *http.Response, err error) int {
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// classify runs one answer through a scheme's Classify, as a tool source or AccessCredential hands it over.
func (s *OAuth2CodeSuite) classify(answer func() (*http.Response, []byte, error)) core.Outcome {
	response, body, err := answer()
	return s.scheme(&http.Client{}, oauth2code.Config{Now: s.clock}).Classify(response, body, err)
}

// connectedToken is one consent at the fake straight through its endpoints, as the fake's
// own tests do: the token response.
func (s *OAuth2CodeSuite) connectedToken(srv *fakeprovider.Server) map[string]string {
	resolved := s.preregistered(srv)
	resolved.Scopes.List = []string{"files:read"}
	_, stored := s.connected(srv, resolved, nil)
	return map[string]string{"access_token": s.accessToken(stored), "refresh_token": s.refreshToken(stored)}
}

func (s *OAuth2CodeSuite) refreshAnswer(srv *fakeprovider.Server, refreshToken string) (*http.Response, []byte, error) {
	return s.post(srv, fakeprovider.PathToken, url.Values{"grant_type": {"refresh_token"}, "refresh_token": {refreshToken},
		"client_id": {srv.ClientID}, "client_secret": {srv.ClientSecret}})
}

func (s *OAuth2CodeSuite) post(srv *fakeprovider.Server, path string, form url.Values) (*http.Response, []byte, error) {
	return s.read(srv.Client().PostForm(srv.URL+path, form))
}

func (s *OAuth2CodeSuite) mcpAnswer(srv *fakeprovider.Server, accessToken string) (*http.Response, []byte, error) {
	request := s.toolCall(srv)
	request.Header.Set("Authorization", "Bearer "+accessToken)
	return s.read(srv.Client().Do(request))
}

func (s *OAuth2CodeSuite) read(response *http.Response, err error) (*http.Response, []byte, error) {
	if err != nil {
		return nil, nil, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	return response, body, err
}

// cutOff is an answer whose status and headers arrived and whose body did not, as
// tokenPost hands it on: the response with the read error.
func (s *OAuth2CodeSuite) cutOff(status int, header http.Header) (*http.Response, []byte, error) {
	response, _, _ := s.synthetic(status, header, "")
	return response, nil, io.ErrUnexpectedEOF
}

// bare401 is a 401 with no challenge and no body to a request with header, which net/http
// puts on the response it reads (http.Response.Request).
func (s *OAuth2CodeSuite) bare401(header http.Header) (*http.Response, []byte, error) {
	response, body, err := s.synthetic(http.StatusUnauthorized, nil, "")
	response.Request = httptest.NewRequest(http.MethodPost, "https://provider.example/", nil)
	maps.Copy(response.Request.Header, header)
	return response, body, err
}

// synthetic is an answer no fake personality gives, built as a provider would send it.
func (s *OAuth2CodeSuite) synthetic(status int, header http.Header, body string) (*http.Response, []byte, error) {
	recorder := httptest.NewRecorder()
	for k, v := range header {
		recorder.Header()[k] = v
	}
	recorder.WriteHeader(status)
	_, _ = recorder.WriteString(body)
	return recorder.Result(), []byte(body), nil
}
