package oauth2cc_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core/contracttest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
)

// TestOAuth2ClientCredentialsSchemeContract runs the scheme contract against the fake
// provider with client credentials on: the fake's preregistered client supplied, its token
// and revoke endpoints in the manifest, and a clock the contract moves with the fake's.
func TestOAuth2ClientCredentialsSchemeContract(t *testing.T) {
	suite.Run(t, &contracttest.SchemeContract{New: func(t *testing.T, _ *slog.Logger) contracttest.Subject {
		srv := fakeprovider.New(t, fakeprovider.ClientCredentials)
		clock := &clock{now: time.Now()}
		scheme, err := oauth2cc.New(oauth2cc.Config{HTTP: srv.Client(), Now: clock.Now, PublicEndpoint: loopbackOrPublic})
		if err != nil {
			t.Fatal(err)
		}
		return contracttest.Subject{
			Scheme:    scheme,
			Manifest:  manifest(srv),
			Supplied:  map[string]string{oauth2cc.SuppliedClientID: srv.ClientID, oauth2cc.SuppliedClientSecret: srv.ClientSecret},
			Transport: srv.Client().Transport,
			Call:      func() *http.Request { return toolCall(srv) },
			Secrets: func(stored core.StoredCredentials) []string {
				var payload struct {
					AccessToken string `json:"access_token"`
				}
				_ = json.Unmarshal(stored.Payload, &payload)
				return []string{payload.AccessToken, srv.ClientSecret}
			},
			Expire: func() {
				clock.Advance(fakeprovider.AccessTTL)
				srv.Advance(fakeprovider.AccessTTL)
			},
			// The first grant is Complete's; a renewal is every one after it.
			Renewals: func() int { return srv.ClientCredentialsGrants() - 1 },
		}
	}})
}

// OAuth2CCSuite covers what is the client credentials scheme's own: when it asks for a new
// token, what it sends, and what each refusal means.
type OAuth2CCSuite struct {
	suite.Suite
	ctx context.Context
	// now is the clock the scheme reads; tests move it.
	now time.Time
}

func TestOAuth2CCSuite(t *testing.T) {
	suite.Run(t, new(OAuth2CCSuite))
}

func (s *OAuth2CCSuite) SetupTest() {
	s.ctx = context.Background()
	s.now = time.Now()
}

func (s *OAuth2CCSuite) TestBeginIsDoneAndSendsNothing() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	out, err := s.scheme(srv.Client()).Begin(s.ctx, core.BeginInput{Manifest: manifest(srv)})
	s.Require().NoError(err)
	s.True(out.Done)
	s.Empty(out.AuthorizeURL)
	s.Zero(srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CCSuite) TestCompleteGetsATokenWithTheSuppliedClientSoAWrongOneFailsAtOnce() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())

	stored := s.complete(srv, scheme, manifest(srv))
	s.Equal(1, srv.ClientCredentialsGrants())
	credential, again, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(stored, again)
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))

	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: map[string]string{
		oauth2cc.SuppliedClientID: srv.ClientID, oauth2cc.SuppliedClientSecret: "not-the-secret",
	}})
	s.Equal(core.OutcomeInvalidGrant, s.outcome(err).Kind, "the fake answers 401 invalid_client")
	s.NotContains(err.Error(), "not-the-secret")
}

func (s *OAuth2CCSuite) TestAServerWithoutTheGrantRefusesItAtComplete() {
	srv := fakeprovider.New(s.T())
	_, _, err := s.scheme(srv.Client()).Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: supplied(srv)})
	s.ErrorContains(err, "unsupported_grant_type")
}

func (s *OAuth2CCSuite) TestATokenIsHandedOutAsStoredUntilItIsInsideTheMargin() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	s.now = s.now.Add(fakeprovider.AccessTTL - 61*time.Second)
	_, again, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(stored, again)
	s.Equal(1, srv.ClientCredentialsGrants(), "outside the one-minute margin nothing is asked for")
}

func (s *OAuth2CCSuite) TestATokenInsideTheMarginIsReplacedWithTheSameClient() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	s.now = s.now.Add(fakeprovider.AccessTTL - 59*time.Second)
	credential, renewed, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.True(s.token(stored) != s.token(renewed), "a new token")
	s.Equal(s.now.Add(fakeprovider.AccessTTL).Unix(), credential.ExpiresAt.Unix())
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))

	again, _, err := scheme.Retrieve(s.ctx, renewed, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants(), "the new token is handed out until it is due in turn")
	s.True(s.token(renewed) == s.secretToken(again))
}

func (s *OAuth2CCSuite) TestATokenPastItsExpiryIsNeverHandedOut() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	s.now = s.now.Add(fakeprovider.AccessTTL + time.Hour)
	srv.Advance(fakeprovider.AccessTTL + time.Hour)
	credential, renewed, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.True(s.token(stored) != s.token(renewed))
	s.True(credential.ExpiresAt.After(s.now))
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential), "the fake refuses the expired one")
}

func (s *OAuth2CCSuite) TestATokenThatExpiresBeforeValidUntilIsReplacedOutsideTheMargin() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))
	before, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)

	s.now = before.ExpiresAt.Add(-10 * time.Minute)
	credential, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{ValidUntil: before.ExpiresAt})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.True(credential.ExpiresAt.After(before.ExpiresAt), "the new token outlives the call")
}

func (s *OAuth2CCSuite) TestATokenTheProviderRefusedIsReplacedWhateverItsExpiry() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))
	_, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(1, srv.ClientCredentialsGrants(), "the token from Complete is still valid")

	_, renewed, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{Refused: true})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.NotEqual(stored, renewed)
}

func (s *OAuth2CCSuite) TestATokenThatOutlivesValidUntilIsNotReplaced() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))
	before, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)

	s.now = before.ExpiresAt.Add(-10 * time.Minute)
	_, again, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{ValidUntil: before.ExpiresAt.Add(-time.Second)})
	s.Require().NoError(err)
	s.Equal(stored, again)
	s.Equal(1, srv.ClientCredentialsGrants())
}

// A token response without expires_in (RFC 6749 section 5.1 only RECOMMENDS it, and
// Salesforce sends none) lives the manifest's refresh.access_ttl, and is replaced inside the
// margin before that ends, though the provider would take it longer.
func (s *OAuth2CCSuite) TestATokenWithoutExpiresInIsReplacedBeforeTheManifestsAccessTTLEnds() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.NoExpiresIn)
	scheme := s.scheme(srv.Client())
	m := manifest(srv)
	m.Refresh.AccessTTL = core.Duration(15 * time.Minute)
	stored := s.complete(srv, scheme, m)
	credential, _, err := scheme.Retrieve(s.ctx, stored, m, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.WithinDuration(s.now.Add(15*time.Minute), credential.ExpiresAt, 0)

	s.now = s.now.Add(15*time.Minute - 61*time.Second)
	_, again, err := scheme.Retrieve(s.ctx, stored, m, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(stored, again, "outside the margin of the access_ttl")

	s.now = s.now.Add(2 * time.Second)
	renewed, next, err := scheme.Retrieve(s.ctx, stored, m, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.True(s.token(stored) != s.token(next))
	s.WithinDuration(s.now.Add(15*time.Minute), renewed.ExpiresAt, 0)
}

// A call that outlives the access_ttl gets a new token for it, even with the margin far off.
func (s *OAuth2CCSuite) TestATokenWithoutExpiresInIsReplacedWhenValidUntilOutlivesTheAccessTTL() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.NoExpiresIn)
	scheme := s.scheme(srv.Client())
	m := manifest(srv)
	m.Refresh.AccessTTL = core.Duration(15 * time.Minute)
	stored := s.complete(srv, scheme, m)

	s.now = s.now.Add(5 * time.Minute)
	credential, _, err := scheme.Retrieve(s.ctx, stored, m, core.RetrieveOptions{ValidUntil: s.now.Add(12 * time.Minute)})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.WithinDuration(s.now.Add(15*time.Minute), credential.ExpiresAt, 0)
}

// With neither expires_in nor refresh.access_ttl a token would be handed out until the
// provider ends it, and then never renewed. Complete refuses it, so nothing is connected.
func (s *OAuth2CCSuite) TestATokenWithNoKnownLifetimeIsRefusedAtComplete() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.NoExpiresIn)
	_, _, err := s.scheme(srv.Client()).Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: supplied(srv)})
	s.ErrorIs(err, oauth2cc.ErrNoLifetime)
	s.ErrorContains(err, "refresh.access_ttl")
}

// Stored credentials without an expiry (none are written now) are due at once.
func (s *OAuth2CCSuite) TestAStoredTokenWithoutAnExpiryIsReplacedAtOnce() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))
	var payload map[string]any
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	delete(payload, "expires_at")
	raw, err := json.Marshal(payload)
	s.Require().NoError(err)
	stored.Payload = raw

	credential, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(2, srv.ClientCredentialsGrants())
	s.False(credential.ExpiresAt.IsZero())
}

// A client credentials request spends nothing (no refresh token, RFC 6749 section 4.4.3),
// so there is nothing for the resolver's checkpoint to guard.
func (s *OAuth2CCSuite) TestReplacingATokenNeverCheckpoints() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	s.now = s.now.Add(fakeprovider.AccessTTL)
	checkpoints := 0
	_, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{Checkpoint: func() error {
		checkpoints++
		return nil
	}})
	s.Require().NoError(err)
	s.Zero(checkpoints)
	s.Equal(2, srv.ClientCredentialsGrants())
}

// RFC 6749 section 2.3.1: client_secret_basic, the method every server must support, is the
// default; the secret travels in the Authorization header, never in the URL or the body.
func (s *OAuth2CCSuite) TestTheSecretGoesInTheAuthorizationHeaderOnly() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	sent := &recorder{next: srv.Client().Transport}
	scheme := s.scheme(&http.Client{Transport: sent})
	s.complete(srv, scheme, manifest(srv))

	requests := sent.all()
	s.Require().Len(requests, 1)
	s.Equal(srv.URL+fakeprovider.PathToken, requests[0].url)
	s.Equal(url.Values{"grant_type": {"client_credentials"}}, requests[0].form, "no secret, no client_id, no scope")
	id, secret, ok := requests[0].basic()
	s.True(ok)
	s.True(id == url.QueryEscape(srv.ClientID) && secret == url.QueryEscape(srv.ClientSecret), "RFC 6749 section 2.3.1 form-encodes both")
}

func (s *OAuth2CCSuite) TestClientSecretPostPutsTheSecretInTheBodyNotTheURL() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	sent := &recorder{next: srv.Client().Transport}
	scheme := s.scheme(&http.Client{Transport: sent})
	m := manifest(srv)
	m.Client.AuthMethod = core.AuthClientSecretPost
	s.complete(srv, scheme, m)

	requests := sent.all()
	s.Require().Len(requests, 1)
	s.NotContains(requests[0].url, srv.ClientSecret)
	s.True(requests[0].form.Get("client_secret") == srv.ClientSecret)
	_, _, ok := requests[0].basic()
	s.False(ok)
}

func (s *OAuth2CCSuite) TestAMethodWithoutASecretIsRefusedAtBegin() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	for _, method := range []core.ClientAuthMethod{core.AuthNone, core.AuthPrivateKeyJWT, core.AuthTLSClientAuth} {
		m := manifest(srv)
		m.Client.AuthMethod = method
		_, err := s.scheme(srv.Client()).Begin(s.ctx, core.BeginInput{Manifest: m})
		s.ErrorContains(err, "is not one this scheme sends", method)
	}
}

func (s *OAuth2CCSuite) TestAManifestWithoutATokenEndpointIsRefusedAtBegin() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	m := manifest(srv)
	delete(m.Endpoints, "token")
	_, err := s.scheme(srv.Client()).Begin(s.ctx, core.BeginInput{Manifest: m})
	s.ErrorContains(err, "has no endpoints.token")
}

// RFC 6749 Appendix A.1, A.2: an id and a secret are VSCHAR, and section 4.4 needs both.
func (s *OAuth2CCSuite) TestAClientRFC6749CannotSendIsRefusedWithoutQuotingIt() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	for _, client := range []map[string]string{
		{oauth2cc.SuppliedClientID: srv.ClientID},
		{oauth2cc.SuppliedClientSecret: srv.ClientSecret},
		{oauth2cc.SuppliedClientID: srv.ClientID, oauth2cc.SuppliedClientSecret: srv.ClientSecret + "\n"},
		{oauth2cc.SuppliedClientID: "idé", oauth2cc.SuppliedClientSecret: srv.ClientSecret},
	} {
		_, _, err := s.scheme(srv.Client()).Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: client})
		s.ErrorContains(err, "oauth2cc: client_")
		s.NotContains(err.Error(), srv.ClientSecret)
	}
	s.Zero(srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CCSuite) TestASuppliedValueItDoesNotTakeIsRefused() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	in := supplied(srv)
	in["api_key"] = srv.ClientSecret
	_, _, err := s.scheme(srv.Client()).Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: in})
	s.ErrorContains(err, `"api_key" is not a value oauth2_client_credentials takes`)
	s.NotContains(err.Error(), srv.ClientSecret)
}

// A secret rotated at the provider makes the stored client useless: only new client
// credentials help, so the resolver moves the connection to needs_reauthorization.
func (s *OAuth2CCSuite) TestARefusedClientIsInvalidGrant() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.withSecret(s.complete(srv, scheme, manifest(srv)), "rotated-at-the-provider")

	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Equal(core.OutcomeInvalidGrant, s.outcome(err).Kind)
	s.NotContains(err.Error(), "rotated-at-the-provider")
}

// The contract reads every log line for secrets on the paths it walks, none of which is a
// refusal; this one is. slog's default is where a stray line would go, written by both
// handlers, since the JSON one marshals a value and the text one formats it.
func (s *OAuth2CCSuite) TestARefusedClientLeavesItsSecretOutOfEveryLogLine() {
	var logs bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewMultiHandler(slog.NewJSONHandler(&logs, nil), slog.NewTextHandler(&logs, nil))))
	defer slog.SetDefault(previous)
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.withSecret(s.complete(srv, scheme, manifest(srv)), "rotated-at-the-provider")

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: map[string]string{
		oauth2cc.SuppliedClientID: srv.ClientID, oauth2cc.SuppliedClientSecret: "not-the-secret",
	}})
	s.Require().Error(err)
	s.now = s.now.Add(fakeprovider.AccessTTL)
	_, _, err = scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Require().Error(err)
	for _, secret := range []string{srv.ClientSecret, "not-the-secret", "rotated-at-the-provider"} {
		s.NotContains(logs.String(), secret)
	}
}

// RFC 6749 section 5.2's codes for a client the server refuses, Salesforce's spelling of
// one, a 401 (which section 5.2 lets a server answer invalid_client with) and invalid_grant:
// the client is the whole grant here, so only new client credentials help.
func (s *OAuth2CCSuite) TestEachAnswerThatRefusesTheClientIsInvalidGrant() {
	for name, answer := range map[string]answer{
		"invalid_client":                 {status: 401, body: `{"error":"invalid_client"}`},
		"unauthorized_client":            {status: 400, body: `{"error":"unauthorized_client"}`},
		"Salesforce's invalid_client_id": {status: 400, body: `{"error":"invalid_client_id","error_description":"client identifier invalid"}`},
		"a bare 401":                     {status: 401},
		"invalid_grant":                  {status: 400, body: `{"error":"invalid_grant"}`},
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeInvalidGrant}, s.refusedWith(answer), name)
	}
}

// A refresh that may have taken effect is Uncertain, since it may have spent a rotating
// refresh token. A client credentials request spends nothing, so the same answers, and any
// other refusal, are Transient: a later request may get a token.
func (s *OAuth2CCSuite) TestEachAnswerThatGaveNoTokenAndSpentNothingIsTransient() {
	for name, answer := range map[string]answer{
		"a 500 (RFC 9110 section 15.6.1)":                      {status: 500, body: `{"error":"server_error"}`},
		"a 502 (RFC 9110 section 15.6.3)":                      {status: 502},
		"a 503 (RFC 9110 section 15.6.4)":                      {status: 503},
		"an error member at 200 (RFC 6749 section 5.2)":        {status: 200, body: `{"error":"invalid_scope"}`},
		"a 200 without an access_token (RFC 6749 section 5.1)": {status: 200, body: `{"token_type":"Bearer"}`},
		"a bare 400":             {status: 400},
		"unsupported_grant_type": {status: 400, body: `{"error":"unsupported_grant_type"}`},
	} {
		s.Equal(core.Outcome{Kind: core.OutcomeTransient}, s.refusedWith(answer), name)
	}
}

// RFC 6585 section 4 for 429, RFC 9110 section 10.2.3 for Retry-After in delay-seconds.
func (s *OAuth2CCSuite) TestA429IsRateLimitedWithItsRetryAfter() {
	s.Equal(core.Outcome{Kind: core.OutcomeRateLimited, RetryAfter: 30 * time.Second},
		s.refusedWith(answer{status: 429, header: http.Header{"Retry-After": {"30"}}}))
}

func (s *OAuth2CCSuite) TestALostAnswerIsTransientSinceARetrySpendsNothing() {
	provider := s.answering(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		conn, _, err := http.NewResponseController(w).Hijack()
		if err == nil {
			_ = conn.Close()
		}
	})
	_, _, err := s.scheme(provider.Client()).Complete(s.ctx, core.CompleteInput{Manifest: at(provider.URL), Supplied: map[string]string{
		oauth2cc.SuppliedClientID: "client", oauth2cc.SuppliedClientSecret: "secret",
	}})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind)
}

func (s *OAuth2CCSuite) TestAFailedReplacementBeforeExpiryHandsOutTheValidToken() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	srv.Use(fakeprovider.ClientCredentials, fakeprovider.Unavailable)
	s.now = s.now.Add(fakeprovider.AccessTTL - 30*time.Second)
	credential, renewed, err := scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind)
	s.Equal(core.StoredCredentials{}, renewed)
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential), "the old token still works")

	s.now = s.now.Add(time.Minute)
	credential, _, err = scheme.Retrieve(s.ctx, stored, manifest(srv), core.RetrieveOptions{})
	s.Error(err)
	s.Empty(credential.Scheme, "past expiry there is no token to hand out")
}

func (s *OAuth2CCSuite) TestATokenEndpointThatIsNotPublicIsRefusedBeforeTheSecretLeaves() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme, err := oauth2cc.New(oauth2cc.Config{HTTP: srv.Client(), Now: s.clock})
	s.Require().NoError(err)
	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Manifest: manifest(srv), Supplied: supplied(srv)})
	s.Equal(core.OutcomeTransient, s.outcome(err).Kind, "nothing was sent, so nothing changed")
	s.Zero(srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CCSuite) TestRevokeWithNoEndpointSaysSoAndSendsNothing() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))
	m := manifest(srv)
	delete(m.Endpoints, "revoke")

	s.ErrorIs(scheme.Revoke(s.ctx, stored, m), oauth2cc.ErrNoRevocationEndpoint)
	s.Zero(srv.Hits(fakeprovider.PathRevoke))
}

func (s *OAuth2CCSuite) TestRevokingAnAccessTokenTheProviderDoesNotRevokeSaysSo() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.AccessTokenNotRevocable)
	scheme := s.scheme(srv.Client())
	stored := s.complete(srv, scheme, manifest(srv))

	s.ErrorIs(scheme.Revoke(s.ctx, stored, manifest(srv)), oauth2cc.ErrTokenTypeNotRevocable)
}

// The built-in Salesforce manifest, every endpoint role it writes pointed at the fake and
// none added: a connection is made from the client alone, and its account id is the
// identity URL the token response carries, as the manifest's capture and identity rules say.
func (s *OAuth2CCSuite) TestTheSalesforceManifestConnectsWithoutABrowser() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientCredentials, fakeprovider.IdentityURL, fakeprovider.NoExpiresIn)
	sent := &recorder{next: srv.Client().Transport}
	scheme := s.scheme(&http.Client{Transport: sent})
	resolved := salesforce(s.T(), nil)
	s.Equal("https://login.salesforce.com/services/oauth2/token", resolved.Endpoints["token"])
	resolved.Endpoints = fake(s.T(), srv, resolved.Endpoints)

	out, err := scheme.Begin(s.ctx, core.BeginInput{Manifest: resolved})
	s.Require().NoError(err)
	s.Require().True(out.Done)
	stored, account, err := scheme.Complete(s.ctx, core.CompleteInput{Manifest: resolved, Supplied: supplied(srv)})
	s.Require().NoError(err)
	s.Equal(srv.IdentityURL, account.AccountID)

	requests := sent.all()
	s.Require().Len(requests, 1, "one token request and no authorize")
	s.Equal(url.Values{"grant_type": {"client_credentials"}, "resource": {srv.URL + fakeprovider.PathMCP}}, requests[0].form)
	credential, _, err := scheme.Retrieve(s.ctx, stored, resolved, core.RetrieveOptions{})
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.wrapped(srv, scheme, credential))
	s.WithinDuration(s.now.Add(15*time.Minute), credential.ExpiresAt, 0, "no expires_in, so the manifest's access_ttl")
}

func (s *OAuth2CCSuite) TestTheSalesforceSandboxGetsItsTokenFromTheSandboxLoginHost() {
	resolved := salesforce(s.T(), map[string]string{"environment": "sandbox"})
	s.Equal("https://test.salesforce.com/services/oauth2/token", resolved.Endpoints["token"])
}

func (s *OAuth2CCSuite) scheme(client *http.Client) *oauth2cc.Scheme {
	scheme, err := oauth2cc.New(oauth2cc.Config{HTTP: client, Now: s.clock, PublicEndpoint: loopbackOrPublic})
	s.Require().NoError(err)
	return scheme
}

func (s *OAuth2CCSuite) clock() time.Time {
	return s.now
}

func (s *OAuth2CCSuite) complete(srv *fakeprovider.Server, scheme *oauth2cc.Scheme, m core.ResolvedManifest) core.StoredCredentials {
	stored, _, err := scheme.Complete(s.ctx, core.CompleteInput{Manifest: m, Supplied: supplied(srv)})
	s.Require().NoError(err)
	return stored
}

// withSecret is stored with another client secret, as after the customer rotated it at the
// provider.
func (s *OAuth2CCSuite) withSecret(stored core.StoredCredentials, secret string) core.StoredCredentials {
	var payload map[string]any
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	payload["client_secret"] = secret
	raw, err := json.Marshal(payload)
	s.Require().NoError(err)
	stored.Payload = raw
	return stored
}

// token is the access token sealed in stored. Compare two with ==, so a failure prints none.
func (s *OAuth2CCSuite) token(stored core.StoredCredentials) string {
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
	return payload.AccessToken
}

func (s *OAuth2CCSuite) secretToken(credential core.AccessCredential) string {
	var secret struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.Unmarshal(credential.Secret(), &secret))
	return secret.AccessToken
}

func (s *OAuth2CCSuite) outcome(err error) core.Outcome {
	var failed *core.OutcomeError
	s.Require().ErrorAs(err, &failed)
	return failed.Outcome
}

// wrapped is the status of a tool call sent through Wrap with credential.
func (s *OAuth2CCSuite) wrapped(srv *fakeprovider.Server, scheme *oauth2cc.Scheme, credential core.AccessCredential) int {
	response, err := (&http.Client{Transport: scheme.Wrap(srv.Client().Transport, credential)}).Do(toolCall(srv))
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

// answer is what a token endpoint answers.
type answer struct {
	status int
	header http.Header
	body   string
}

// refusedWith is the outcome Complete fails with when the token endpoint answers a.
func (s *OAuth2CCSuite) refusedWith(a answer) core.Outcome {
	provider := s.answering(func(w http.ResponseWriter, _ *http.Request) {
		for k, v := range a.header {
			w.Header()[k] = v
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(a.status)
		_, _ = w.Write([]byte(a.body))
	})
	_, _, err := s.scheme(provider.Client()).Complete(s.ctx, core.CompleteInput{Manifest: at(provider.URL), Supplied: map[string]string{
		oauth2cc.SuppliedClientID: "client", oauth2cc.SuppliedClientSecret: "secret",
	}})
	return s.outcome(err)
}

// answering is a TLS token endpoint that answers every request with handler.
func (s *OAuth2CCSuite) answering(handler http.HandlerFunc) *httptest.Server {
	server := httptest.NewTLSServer(handler)
	s.T().Cleanup(server.Close)
	return server
}

// manifest is a connector at the fake: its token and revoke endpoints and its MCP endpoint,
// and no policy of its own, so each test sets the one field it is about.
func manifest(srv *fakeprovider.Server) core.ResolvedManifest {
	return core.ResolvedManifest{
		ConnectorID: "custom_fake",
		Scheme:      oauth2cc.Name,
		Endpoints: map[string]string{
			"token":  srv.URL + fakeprovider.PathToken,
			"revoke": srv.URL + fakeprovider.PathRevoke,
			"mcp":    srv.URL + fakeprovider.PathMCP,
		},
	}
}

// at is a manifest whose token endpoint is base's.
func at(base string) core.ResolvedManifest {
	return core.ResolvedManifest{ConnectorID: "custom_fake", Scheme: oauth2cc.Name, Endpoints: map[string]string{"token": base + "/token"}}
}

func supplied(srv *fakeprovider.Server) map[string]string {
	return map[string]string{oauth2cc.SuppliedClientID: srv.ClientID, oauth2cc.SuppliedClientSecret: srv.ClientSecret}
}

// salesforce is the built-in manifest resolved for this scheme with inputs.
func salesforce(t *testing.T, inputs map[string]string) core.ResolvedManifest {
	raw, err := fs.ReadFile(providers.FS, "salesforce.yaml")
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := core.ParseManifest(raw)
	if err != nil {
		t.Fatal(err)
	}
	resolved, err := parsed.Resolve(oauth2cc.Name, inputs, nil)
	if err != nil {
		t.Fatal(err)
	}
	return resolved
}

// fake points every endpoint role in endpoints at the fake's endpoint for that role, and adds
// none (as providers' consent tests do).
func fake(t *testing.T, srv *fakeprovider.Server, endpoints map[string]string) map[string]string {
	roles := map[string]string{
		"authorize": srv.URL + fakeprovider.PathAuthorize,
		"token":     srv.URL + fakeprovider.PathToken,
		"revoke":    srv.URL + fakeprovider.PathRevoke,
		"mcp":       srv.URL + fakeprovider.PathMCP,
		"resource":  srv.URL + fakeprovider.PathMCP,
	}
	out := map[string]string{}
	for role := range endpoints {
		endpoint, ok := roles[role]
		if !ok {
			t.Fatalf("the fake has no %s endpoint", role)
		}
		out[role] = endpoint
	}
	return out
}

func toolCall(srv *fakeprovider.Server) *http.Request {
	request, _ := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP,
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
	return request
}

// loopbackOrPublic is egress's endpoint check with one hole, a loopback IP literal, where the
// fake listens (as in the oauth2code suite).
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}

// sent is one request the recorder saw: its URL, form body and Authorization header.
type sent struct {
	url           string
	form          url.Values
	authorization string
}

func (r sent) basic() (id, secret string, ok bool) {
	request := http.Request{Header: http.Header{"Authorization": {r.authorization}}}
	return request.BasicAuth()
}

// recorder is the transport under the scheme: it keeps what each request carried and sends
// it on.
type recorder struct {
	next http.RoundTripper
	mu   sync.Mutex
	seen []sent
}

func (r *recorder) RoundTrip(request *http.Request) (*http.Response, error) {
	var body []byte
	if request.Body != nil {
		var err error
		body, err = io.ReadAll(request.Body)
		_ = request.Body.Close()
		if err != nil {
			return nil, err
		}
		request.Body = io.NopCloser(bytes.NewReader(body))
	}
	form, _ := url.ParseQuery(string(body))
	r.mu.Lock()
	r.seen = append(r.seen, sent{url: request.URL.String(), form: form, authorization: request.Header.Get("Authorization")})
	r.mu.Unlock()
	return r.next.RoundTrip(request)
}

func (r *recorder) all() []sent {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]sent(nil), r.seen...)
}

// clock is a time the contract moves, safe to read from the concurrent resolves.
type clock struct {
	mu  sync.Mutex
	now time.Time
}

func (c *clock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *clock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}
