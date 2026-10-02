package oauth2code_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

func (s *OAuth2CodeSuite) TestMintHandsOutTheTokenAsItIsUntilItIsInsideTheMargin() {
	srv := fakeprovider.New(s.T())
	scheme, material := s.connected(srv, s.preregistered(srv), nil)

	credential, again, err := scheme.Mint(s.ctx, material, s.preregistered(srv))
	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
	s.Equal(material, again, "nothing was renewed, so the material is the one given")
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestMintRefreshesInsideTheDefaultMinuteBeforeExpiry() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 59*time.Second)
	credential, rotated, err := scheme.Mint(s.ctx, material, profile)
	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
	s.NotEqual(s.accessToken(material), s.accessToken(rotated))
	s.NotEqual(s.refreshToken(material), s.refreshToken(rotated), "the fake rotated, so the new refresh token replaces the old")
	s.Equal(s.now.Add(fakeprovider.AccessTTL).Unix(), credential.ExpiresAt.Unix())
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestOutsideTheDefaultMinuteMintDoesNotRefresh() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 61*time.Second)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Require().NoError(err)
	s.Equal(0, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestTheManifestMarginDecidesWhenARefreshIsDue() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Refresh.Margin = core.Duration(10 * time.Minute)
	scheme, material := s.connected(srv, profile, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - 9*time.Minute)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Require().NoError(err)
	s.Equal(1, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestARefreshSendsNoScopeUnlessThePolicySays() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.mintDue(scheme, material, profile)
	_, sent := srv.RefreshScope()
	s.False(sent)
}

func (s *OAuth2CodeSuite) TestARefreshSendsTheGrantedScopesWhenThePolicySays() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Scopes.SendOnRefresh = true
	scheme, material := s.connected(srv, profile, nil)

	s.mintDue(scheme, material, profile)
	scope, sent := srv.RefreshScope()
	s.True(sent)
	s.Equal("files:read files:write", scope)
}

func (s *OAuth2CodeSuite) TestALostRefreshIsUncertainAndLeavesTheMaterialAlone() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)
	before := bytes.Clone(material.Payload)
	srv.Use(fakeprovider.LostResponse)

	s.now = s.now.Add(fakeprovider.AccessTTL)
	credential, returned, err := scheme.Mint(s.ctx, material, profile)
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(1, srv.Refreshes(), "without a grace window the spent token is never sent again")
	s.Equal(before, []byte(material.Payload), "the caller's material is untouched")
	s.Equal(core.Material{}, returned, "no material to persist")
	s.Equal(core.Credential{}, credential)
}

func (s *OAuth2CodeSuite) TestALostRefreshInsideTheGraceWindowIsRetriedOnceWithTheOldToken() {
	srv := fakeprovider.New(s.T())
	profile := s.registered(srv)
	s.Require().Equal(core.Duration(30*time.Minute), profile.Refresh.Grace, "the Linear manifest's grace")
	scheme, material := s.connected(srv, profile, nil)
	srv.Use(fakeprovider.LostResponseOnce, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	credential, rotated, err := scheme.Mint(s.ctx, material, profile)
	s.Require().NoError(err)
	s.Equal(2, srv.Refreshes(), "the lost one and one retry")
	s.NotEqual(s.refreshToken(material), s.refreshToken(rotated))
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
}

func (s *OAuth2CodeSuite) TestTheGraceRetryIsMadeOnlyOnce() {
	srv := fakeprovider.New(s.T())
	profile := s.registered(srv)
	scheme, material := s.connected(srv, profile, nil)
	srv.Use(fakeprovider.LostResponse, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(2, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestNoGraceRetryOnceTheWindowHasClosed() {
	srv := fakeprovider.New(s.T())
	profile := s.registered(srv)
	scheme, material := s.connected(srv, profile, nil)
	srv.Use(fakeprovider.LostResponseOnce, fakeprovider.RotatingRefreshWithGrace)

	s.now = s.now.Add(24 * time.Hour)
	// Every read of the clock now moves it a whole grace window, so the answer of the first
	// attempt arrives after the window it opened has closed.
	s.step = time.Duration(profile.Refresh.Grace)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Equal(core.OutcomeUncertain, s.outcome(err).Kind)
	s.Equal(1, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestANonRotatingRefreshKeepsTheRefreshToken() {
	srv := fakeprovider.New(s.T(), fakeprovider.NonRotatingRefresh)
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	renewed := s.mintDue(scheme, material, profile)
	s.Equal(s.refreshToken(material), s.refreshToken(renewed))
	s.NotEqual(s.accessToken(material), s.accessToken(renewed))
}

func (s *OAuth2CodeSuite) TestAnExpiredTokenWithNoRefreshTokenIsInvalidGrant() {
	srv := fakeprovider.New(s.T(), fakeprovider.NoRefreshToken)
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.now = s.now.Add(fakeprovider.AccessTTL - time.Second)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Require().NoError(err, "inside the margin and still valid, it is handed out")

	s.now = s.now.Add(time.Second)
	_, _, err = scheme.Mint(s.ctx, material, profile)
	s.Equal(core.OutcomeInvalidGrant, s.outcome(err).Kind)
	s.Equal(0, srv.Refreshes())
}

// TestEachRefusedRefreshIsTheOutcomeItsAnswerMeans is the table of what the token endpoint
// can answer a refresh with and what Mint makes of it. The caller's material never
// changes and no material comes back.
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
		profile := s.preregistered(srv)
		scheme, material := s.connected(srv, profile, nil)
		before := bytes.Clone(material.Payload)
		srv.Use(row.personality)

		s.now = s.now.Add(fakeprovider.AccessTTL)
		_, returned, err := scheme.Mint(s.ctx, material, profile)
		s.Equal(row.want, s.outcome(err), row.personality)
		s.Equal(before, []byte(material.Payload), row.personality)
		s.Equal(core.Material{}, returned, row.personality)
	}
}

func (s *OAuth2CodeSuite) TestARefreshLooksThePreregisteredClientSecretUpAgain() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	secret := srv.ClientSecret
	lookup := func(_ context.Context, ref core.ConnectionRef, _ core.Profile, owner core.ClientOwner) (oauth2code.Client, bool, error) {
		s.Equal(s.ref, ref, "the connection Complete ran for")
		return oauth2code.Client{ID: srv.ClientID, Secret: secret}, owner == core.ClientOperator, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	material, _, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.NotContains(string(material.Payload), srv.ClientSecret, "a preregistered secret is never sealed")

	secret = "rotated-elsewhere"
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err = scheme.Mint(s.ctx, material, profile)
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind, "the fake refuses the new secret with invalid_client")
	var refused *oauth2code.TokenError
	s.Require().ErrorAs(err, &refused)
	s.Equal("invalid_client", refused.Code)
}

func (s *OAuth2CodeSuite) TestARefreshTokenDyingBeforeTheNextRefreshIsWarnedAboutWithoutATokenInTheLog() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Refresh.RefreshTTL = core.Duration(30 * time.Minute)
	var log bytes.Buffer
	scheme, material := s.connected(srv, profile, slog.New(slog.NewJSONHandler(&log, nil)))

	renewed := s.mintDue(scheme, material, profile)
	s.Contains(log.String(), "the refresh token expires before the next refresh")
	s.Contains(log.String(), `"connection":"conn-1"`)
	for _, secret := range []string{s.accessToken(renewed), s.refreshToken(renewed), s.refreshToken(material), srv.ClientSecret} {
		s.NotContains(log.String(), secret)
	}
}

func (s *OAuth2CodeSuite) TestARefreshTokenThatOutlivesTheNextRefreshIsNotWarnedAbout() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Refresh.RefreshTTL = core.Duration(90 * 24 * time.Hour)
	var log bytes.Buffer
	scheme, material := s.connected(srv, profile, slog.New(slog.NewJSONHandler(&log, nil)))

	s.mintDue(scheme, material, profile)
	s.Empty(log.String())
}

func (s *OAuth2CodeSuite) TestARefreshEndpointThatIsNotPublicIsRefusedBeforeTheTokenLeaves() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)
	profile.Endpoints["refresh"] = "https://10.0.0.1/token"

	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err := scheme.Mint(s.ctx, material, profile)
	s.Require().ErrorContains(err, "egress:")
	s.Equal(0, srv.Refreshes())
}

func (s *OAuth2CodeSuite) TestWrapSendsTheAccessTokenAsABearerHeaderOnACopyOfTheRequest() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)
	credential, _, err := scheme.Mint(s.ctx, material, profile)
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
	client := &http.Client{Transport: scheme.Wrap(srv.Client().Transport, core.Credential{})}
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
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeOK}, s.classify(row.answer), row.name)
	}
}

func (s *OAuth2CodeSuite) TestRevokeEndsTheGrantAtTheManifestsEndpoint() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Endpoints["revoke"] = srv.URL + fakeprovider.PathRevoke
	scheme, material := s.connected(srv, profile, nil)

	s.Require().NoError(scheme.Revoke(s.ctx, material, profile))
	s.Equal(1, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusUnauthorized, s.call(srv, s.accessToken(material)), "revoking the refresh token ended the grant")
}

func (s *OAuth2CodeSuite) TestRevokeUsesTheEndpointTheServerAdvertised() {
	srv := fakeprovider.New(s.T())
	profile := s.registered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.Require().NoError(scheme.Revoke(s.ctx, material, profile))
	s.Equal(1, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusUnauthorized, s.call(srv, s.accessToken(material)))
}

func (s *OAuth2CodeSuite) TestRevokeWithNoEndpointSaysSoAndSendsNothing() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	scheme, material := s.connected(srv, profile, nil)

	s.ErrorIs(scheme.Revoke(s.ctx, material, profile), oauth2code.ErrNoRevocationEndpoint)
	s.Equal(0, srv.Hits(fakeprovider.PathRevoke))
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *OAuth2CodeSuite) TestARefusedRevocationIsAnOutcomeError() {
	srv := fakeprovider.New(s.T())
	profile := s.preregistered(srv)
	profile.Endpoints["revoke"] = srv.URL + fakeprovider.PathRevoke
	secret := srv.ClientSecret
	lookup := func(_ context.Context, _ core.ConnectionRef, _ core.Profile, owner core.ClientOwner) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: srv.ClientID, Secret: secret}, owner == core.ClientOperator, nil
	}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: lookup, Now: s.clock})
	material, _, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)

	secret = "rotated-elsewhere"
	err = scheme.Revoke(s.ctx, material, profile)
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)), "nothing was revoked")
}

// preregistered is the Slack fixture pointed at the fake, with the fake's preregistered
// client as the operator's (client_secret_basic), space-separated scopes and no refresh
// policy of its own, so each test sets the one field it is about.
func (s *OAuth2CodeSuite) preregistered(srv *fakeprovider.Server) core.Profile {
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	profile.Capture, profile.Identity = nil, nil
	profile.Client.AuthMethod = ""
	profile.Scopes = core.ScopePolicy{List: []string{"files:read", "files:write"}}
	profile.Refresh = core.RefreshPolicy{}
	return profile
}

// registered is the Linear manifest discovering the fake and registering a public client,
// with its own refresh policy (rotating, 30 minutes of grace).
func (s *OAuth2CodeSuite) registered(srv *fakeprovider.Server) core.Profile {
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	return profile
}

// connected is a scheme on the suite's clock and the material of one consent through it.
func (s *OAuth2CodeSuite) connected(srv *fakeprovider.Server, p core.Profile, logger *slog.Logger) (*oauth2code.Scheme, core.Material) {
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv), Now: s.clock, Logger: logger})
	material, _, err := s.connect(srv, scheme, p)
	s.Require().NoError(err)
	return scheme, material
}

// mintDue moves the clock past expiry and mints, which must refresh once and succeed.
func (s *OAuth2CodeSuite) mintDue(scheme *oauth2code.Scheme, m core.Material, p core.Profile) core.Material {
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, renewed, err := scheme.Mint(s.ctx, m, p)
	s.Require().NoError(err)
	return renewed
}

func (s *OAuth2CodeSuite) clock() time.Time {
	now := s.now
	s.now = s.now.Add(s.step)
	return now
}

func (s *OAuth2CodeSuite) refreshToken(m core.Material) string {
	var payload struct {
		RefreshToken string `json:"refresh_token"`
	}
	s.Require().NoError(json.Unmarshal(m.Payload, &payload))
	return payload.RefreshToken
}

func (s *OAuth2CodeSuite) outcome(err error) core.Outcome {
	var failed *core.OutcomeError
	s.Require().ErrorAs(err, &failed)
	return failed.Outcome
}

// wrapped is the status of a tool call sent through Wrap with credential.
func (s *OAuth2CodeSuite) wrapped(srv *fakeprovider.Server, scheme *oauth2code.Scheme, credential core.Credential) int {
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

// classify runs one answer through a scheme's Classify, as a source or Mint hands it over.
func (s *OAuth2CodeSuite) classify(answer func() (*http.Response, []byte, error)) core.Outcome {
	response, body, err := answer()
	return s.scheme(&http.Client{}, oauth2code.Config{Now: s.clock}).Classify(response, body, err)
}

// connectedToken is one consent at the fake straight through its endpoints, as the fake's
// own tests do: the token response.
func (s *OAuth2CodeSuite) connectedToken(srv *fakeprovider.Server) map[string]string {
	profile := s.preregistered(srv)
	profile.Scopes.List = []string{"files:read"}
	_, material := s.connected(srv, profile, nil)
	return map[string]string{"access_token": s.accessToken(material), "refresh_token": s.refreshToken(material)}
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
