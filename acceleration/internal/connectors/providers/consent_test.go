package providers_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
)

// ConsentSuite runs each built-in manifest's oauth2_code consent, Begin, the browser and
// Complete, against the fake provider. Every endpoint role a manifest writes is pointed at
// the fake and no other is added, so each consent takes the discovery path its manifest
// chose, with its own client policy, scopes and capture rules.
type ConsentSuite struct {
	suite.Suite
	ctx context.Context
	ref core.ConnectionRef
}

func TestConsentSuite(t *testing.T) {
	suite.Run(t, new(ConsentSuite))
}

func (s *ConsentSuite) SetupTest() {
	s.ctx = context.Background()
	s.ref = core.ConnectionRef{CustomerID: "acme", ConnectionID: "conn-1"}
}

func (s *ConsentSuite) TestCalendlyRegistersAPublicClientAtItsPinnedIssuer() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("calendly", nil))
	scheme := s.scheme(srv, nil)

	out := s.begin(scheme, profile)
	s.Equal(0, srv.Hits(fakeprovider.PathProtectedResource), "the issuer is pinned, so the MCP endpoint's metadata is not read")
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"), "the authorize endpoint from the issuer's metadata")
	s.Equal("mcp:scheduling:read mcp:scheduling:write", query.Get("scope"))
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"), "the pinned resource")

	material, captured, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
	s.Empty(captured.AccountID, "no identity")
	s.Empty(captured.Metadata, "owner and organization are optional, and the fake sends neither")
}

func (s *ConsentSuite) TestCalcomDiscoversItsServerFromTheMCPEndpointAndAsksForNoScope() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("calcom", nil))
	scheme := s.scheme(srv, nil)

	out := s.begin(scheme, profile)
	s.Equal(1, srv.Hits(fakeprovider.PathProtectedResource))
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"))
	s.NotContains(query, "scope", "the server names no scopes, so none are asked for")
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"), "the resource from the protected resource metadata")

	material, captured, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
	s.Empty(captured.AccountID)
}

func (s *ConsentSuite) TestGitHubConnectsWithThePreregisteredClientAndRegistersNone() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("github", nil))
	scheme := s.scheme(srv, s.preregistered(srv, core.ClientOperator))

	out := s.begin(scheme, profile)
	s.Equal(0, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"), "the authorize endpoint from the issuer's metadata")
	s.Equal(srv.ClientID, query.Get("client_id"))
	s.Equal("repo read:org read:user user:email read:packages write:packages read:project project gist notifications offline_access", query.Get("scope"))
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"))

	material, captured, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
	s.Equal(profile.Scopes.List, captured.Scopes)
	s.Empty(captured.AccountID)
}

// The fake's preregistered client accepts both secret methods, so only the request on the
// wire shows which one the manifest made the scheme send.
func (s *ConsentSuite) TestGitHubSendsTheClientSecretInTheTokenRequestBody() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("github", nil))
	wire := &tokenWire{next: srv.Client().Transport}
	scheme := s.schemeOver(&http.Client{Transport: wire}, s.preregistered(srv, core.ClientOperator))

	_, _, err := s.complete(srv, scheme, profile, s.begin(scheme, profile))
	s.Require().NoError(err)
	s.Require().Len(wire.sent, 1, "one code exchange")
	s.Empty(wire.sent[0].authorization, "no HTTP Basic credentials")
	s.Equal(srv.ClientID, wire.sent[0].form.Get("client_id"))
	s.Equal(srv.ClientSecret, wire.sent[0].form.Get("client_secret"))
}

func (s *ConsentSuite) TestGitHubTakesACustomersClientBeforeTheOperators() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("github", nil))
	scheme := s.scheme(srv, func(_ context.Context, _ core.ConnectionRef, _ core.Profile, owner core.ClientOwner) (oauth2code.Client, bool, error) {
		if owner == core.ClientCustomer {
			return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, true, nil
		}
		return oauth2code.Client{ID: "operator-client", Secret: "operator-secret"}, true, nil
	})

	out := s.begin(scheme, profile)
	s.Equal(srv.ClientID, s.query(out.AuthorizeURL).Get("client_id"), "the customer's client, though the policy lists the operator first")
	_, _, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
}

// expires_in decides a GitHub token's expiry. GitHub leaves it out only for a token that does
// not expire, so the manifest names no fallback lifetime, which oauth2_code would otherwise
// store on such a token. The fake always sends expires_in, so this is read from the profile.
func (s *ConsentSuite) TestGitHubNamesNoAccessLifetimeForATokenThatDoesNotExpire() {
	refresh := s.profile("github", nil).Refresh
	s.Zero(refresh.AccessTTL)
	s.True(refresh.Rotating)
}

func (s *ConsentSuite) TestGitHubWithoutAPreregisteredClientIsRefusedRatherThanRegistered() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("github", nil))
	scheme := s.scheme(srv, nil)

	_, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: profile, RedirectURI: fakeprovider.RedirectURI})
	s.ErrorIs(err, oauth2code.ErrNoClient)
	s.Equal(0, srv.Hits(fakeprovider.PathRegister), "the fake offers registration; the manifest does not take it")
}

func (s *ConsentSuite) TestGongUsesTheCustomersClientBeforeRegisteringOne() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("gong", nil))
	scheme := s.scheme(srv, s.preregistered(srv, core.ClientCustomer))

	out := s.begin(scheme, profile)
	s.Equal(1, srv.Hits(fakeprovider.PathProtectedResource))
	s.Equal(0, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.Equal(srv.ClientID, query.Get("client_id"))
	s.Equal("mcp:read mcp:write", query.Get("scope"))
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"))

	material, _, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *ConsentSuite) TestGongRegistersAClientWhenTheCustomerHasNone() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("gong", nil))
	scheme := s.scheme(srv, nil)

	out := s.begin(scheme, profile)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	s.NotEqual(srv.ClientID, s.query(out.AuthorizeURL).Get("client_id"))

	material, _, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *ConsentSuite) TestSalesforceProductionIsTheDefaultEnvironment() {
	endpoints := s.profile("salesforce", nil).Endpoints
	s.Equal(map[string]string{
		"authorize": "https://login.salesforce.com/services/oauth2/authorize",
		"token":     "https://login.salesforce.com/services/oauth2/token",
		"revoke":    "https://login.salesforce.com/services/oauth2/revoke",
		"mcp":       "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all",
		"resource":  "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all",
	}, endpoints)
}

func (s *ConsentSuite) TestSalesforceSandboxUsesTheSandboxLoginHostAndMCPPath() {
	endpoints := s.profile("salesforce", map[string]string{"environment": "sandbox"}).Endpoints
	s.Equal(map[string]string{
		"authorize": "https://test.salesforce.com/services/oauth2/authorize",
		"token":     "https://test.salesforce.com/services/oauth2/token",
		"revoke":    "https://test.salesforce.com/services/oauth2/revoke",
		"mcp":       "https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all",
		"resource":  "https://api.salesforce.com/platform/mcp/v1/sandbox/platform/sobject-all",
	}, endpoints)
}

// The fake's token response is RFC 6749's and carries no identity URL, and no personality
// adds one, so this consent ends where the manifest says it must: no account, no connection.
func (s *ConsentSuite) TestSalesforceRefusesATokenResponseWithoutAnIdentityURL() {
	srv := fakeprovider.New(s.T())
	profile := s.atFake(srv, s.profile("salesforce", nil))
	scheme := s.scheme(srv, s.preregistered(srv, core.ClientOperator))

	out := s.begin(scheme, profile)
	s.Equal(0, srv.Hits(fakeprovider.PathProtectedResource), "the endpoints are pinned, so nothing is discovered")
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"))
	s.Equal(srv.ClientID, query.Get("client_id"))
	s.Equal("mcp_api refresh_token", query.Get("scope"))
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"))

	_, _, err := s.complete(srv, scheme, profile, out)
	s.ErrorContains(err, "capture identity_url: token_response has no $.id")
	s.Equal(1, srv.Hits(fakeprovider.PathToken), "the code was exchanged; the response was refused")
}

// profile is the built-in manifest id resolved for oauth2_code with inputs.
func (s *ConsentSuite) profile(id string, inputs map[string]string) core.Profile {
	raw, err := fs.ReadFile(providers.FS, id+".yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	profile, err := manifest.Resolve(oauth2code.Name, inputs, nil)
	s.Require().NoError(err)
	return profile
}

// atFake points every endpoint role the profile has at the fake's endpoint for that role,
// and adds none, so what is pinned and what is discovered stays as the manifest wrote it.
func (s *ConsentSuite) atFake(srv *fakeprovider.Server, p core.Profile) core.Profile {
	fake := map[string]string{
		"issuer":    srv.URL,
		"authorize": srv.URL + fakeprovider.PathAuthorize,
		"token":     srv.URL + fakeprovider.PathToken,
		"revoke":    srv.URL + fakeprovider.PathRevoke,
		"mcp":       srv.URL + fakeprovider.PathMCP,
		"resource":  srv.URL + fakeprovider.PathMCP,
	}
	endpoints := map[string]string{}
	for role := range p.Endpoints {
		endpoint, ok := fake[role]
		s.Require().True(ok, "the fake has no %s endpoint", role)
		endpoints[role] = endpoint
	}
	p.Endpoints = endpoints
	return p
}

// preregistered answers the fake's preregistered client for owner, and no client for any
// other owner.
func (s *ConsentSuite) preregistered(srv *fakeprovider.Server, owner core.ClientOwner) oauth2code.ClientLookup {
	return func(_ context.Context, _ core.ConnectionRef, _ core.Profile, asked core.ClientOwner) (oauth2code.Client, bool, error) {
		if asked != owner {
			return oauth2code.Client{}, false, nil
		}
		return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, true, nil
	}
}

func (s *ConsentSuite) scheme(srv *fakeprovider.Server, clients oauth2code.ClientLookup) *oauth2code.Scheme {
	return s.schemeOver(srv.Client(), clients)
}

// schemeOver is the scheme sending every request through client.
func (s *ConsentSuite) schemeOver(client *http.Client, clients oauth2code.ClientLookup) *oauth2code.Scheme {
	scheme, err := oauth2code.New(oauth2code.Config{HTTP: client, Clients: clients, PublicEndpoint: loopbackOrPublic})
	s.Require().NoError(err)
	return scheme
}

func (s *ConsentSuite) begin(scheme *oauth2code.Scheme, p core.Profile) core.BeginOutput {
	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: p, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	return out
}

// complete plays the browser through the fake's consent and hands the callback to Complete.
func (s *ConsentSuite) complete(srv *fakeprovider.Server, scheme *oauth2code.Scheme, p core.Profile, out core.BeginOutput) (core.Material, core.Captured, error) {
	callback, err := srv.Consent(out.AuthorizeURL)
	s.Require().NoError(err)
	s.Require().Empty(callback.Query().Get("error"), "the fake refused the authorize request")
	return scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: p, State: out.State, Query: callback.Query()})
}

// accessToken reads the access token out of the material, the only way to show it works
// until oauth2_code's Wrap (AI-836) carries it.
func (s *ConsentSuite) accessToken(m core.Material) string {
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.Unmarshal(m.Payload, &payload))
	s.Require().NotEmpty(payload.AccessToken)
	return payload.AccessToken
}

// call is a tools/call to the fake's MCP endpoint with token; its status says whether the
// token is good.
func (s *ConsentSuite) call(srv *fakeprovider.Server, token string) int {
	request, err := http.NewRequest(http.MethodPost, srv.URL+fakeprovider.PathMCP,
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}`))
	s.Require().NoError(err)
	request.Header.Set("Authorization", "Bearer "+token)
	response, err := srv.Client().Do(request)
	s.Require().NoError(err)
	_, _ = io.Copy(io.Discard, response.Body)
	s.Require().NoError(response.Body.Close())
	return response.StatusCode
}

func (s *ConsentSuite) query(raw string) url.Values {
	u, err := url.Parse(raw)
	s.Require().NoError(err)
	return u.Query()
}

// loopbackOrPublic is egress's endpoint check with one hole: a loopback IP literal, where
// the fake listens. Everything else is refused as in the router.
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}

// tokenWire passes every request on to next and keeps what each one to the fake's token
// endpoint carried: its Authorization header and its form body.
type tokenWire struct {
	next http.RoundTripper
	mu   sync.Mutex
	sent []tokenRequest
}

type tokenRequest struct {
	authorization string
	form          url.Values
}

func (w *tokenWire) RoundTrip(r *http.Request) (*http.Response, error) {
	if r.URL.Path != fakeprovider.PathToken || r.Body == nil {
		return w.next.RoundTrip(r)
	}
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	if err := r.Body.Close(); err != nil {
		return nil, err
	}
	form, err := url.ParseQuery(string(raw))
	if err != nil {
		return nil, err
	}
	w.mu.Lock()
	w.sent = append(w.sent, tokenRequest{authorization: r.Header.Get("Authorization"), form: form})
	w.mu.Unlock()
	forwarded := r.Clone(r.Context())
	forwarded.Body = io.NopCloser(bytes.NewReader(raw))
	return w.next.RoundTrip(forwarded)
}
