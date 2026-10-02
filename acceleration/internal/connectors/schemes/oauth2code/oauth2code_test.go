package oauth2code_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
)

// OAuth2CodeSuite drives the scheme against the fake provider, with the built-in manifests
// and core's fixtures where one fits. It is an outside package, so it sees only what the
// router does.
type OAuth2CodeSuite struct {
	suite.Suite
	ctx context.Context
	ref core.ConnectionRef
}

func TestOAuth2CodeSuite(t *testing.T) {
	suite.Run(t, new(OAuth2CodeSuite))
}

func (s *OAuth2CodeSuite) SetupTest() {
	s.ctx = context.Background()
	s.ref = core.ConnectionRef{CustomerID: "acme", ConnectionID: "conn-1"}
}

// TestTheSlackManifestBuildsThePrototypesAuthorizeURL is the T9 acceptance golden test.
// The prototype built Slack's authorize URL in internal/mcp/oauth.go:252-278 on
// codex/connector-support at cf62af0d: response_type, client_id, redirect_uri, state,
// code_challenge, code_challenge_method S256, resource, and the scopes joined with commas
// by the connector.ID == "slack" branch at oauth.go:265-267, all through url.Values.Encode
// onto the catalog's authorization_endpoint. The catalog values are
// internal/mcp/connectors.yaml:9-10,15-44 there, the callback is CallbackPath
// (oauth.go:119) under the PublicURL the prototype's Slack test used
// (oauth_test.go:356). The manifest now says with separator "," what the branch said.
func (s *OAuth2CodeSuite) TestTheSlackManifestBuildsThePrototypesAuthorizeURL() {
	profile := s.profile("../../providers/slack.yaml", nil)
	// Nothing is fetched for a manifest that pins its endpoints, so the client has nowhere
	// to go.
	scheme := s.scheme(&http.Client{}, oauth2code.Config{Clients: func(context.Context, core.ConnectionRef, core.Profile, core.ClientOwner) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: "operator-client", Secret: "operator-secret"}, true, nil
	}})
	out, err := scheme.Begin(s.ctx, core.BeginInput{
		Ref: s.ref, Profile: profile, RedirectURI: "https://router.example/v1/agents/connectors/oauth/callback",
	})
	s.Require().NoError(err)
	s.False(out.Done)

	query := s.query(out.AuthorizeURL)
	prototype := "https://slack.com/oauth/v2_user/authorize?" +
		"client_id=operator-client" +
		"&code_challenge=" + query.Get("code_challenge") +
		"&code_challenge_method=S256" +
		"&redirect_uri=https%3A%2F%2Frouter.example%2Fv1%2Fagents%2Fconnectors%2Foauth%2Fcallback" +
		"&resource=https%3A%2F%2Fmcp.slack.com" +
		"&response_type=code" +
		"&scope=search%3Aread.public%2Csearch%3Aread.private%2Csearch%3Aread.mpim%2Csearch%3Aread.im" +
		"%2Csearch%3Aread.files%2Csearch%3Aread.users%2Cfiles%3Aread%2Cfiles%3Awrite%2Cemoji%3Aread" +
		"%2Cchat%3Awrite%2Cchannels%3Ahistory%2Cgroups%3Ahistory%2Cmpim%3Ahistory%2Cim%3Ahistory" +
		"%2Cchannels%3Awrite%2Cgroups%3Awrite%2Cim%3Awrite%2Cmpim%3Awrite%2Creactions%3Awrite" +
		"%2Ccanvases%3Aread%2Ccanvases%3Awrite%2Cusers%3Aread%2Cusers%3Aread.email%2Cchannels%3Aread" +
		"%2Cgroups%3Aread%2Cmpim%3Aread%2Cim%3Aread%2Clists%3Aread%2Clists%3Awrite" +
		"&state=" + query.Get("state")
	s.Equal(prototype, out.AuthorizeURL)
	s.Len(query.Get("state"), 43, "32 random octets, base64url")
	s.Len(query.Get("code_challenge"), 43, "a SHA-256 digest, base64url")
}

func (s *OAuth2CodeSuite) TestAPreregisteredConfidentialClientConnectsWithPKCE() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	profile.Capture, profile.Identity = nil, nil
	profile.Client.AuthMethod = ""
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	material, captured, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.Equal(oauth2code.Name, material.Scheme)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)), "the token the exchange returned works")
	s.Equal([]string{"channels:history", "chat:write", "users:read"}, captured.Scopes)
}

func (s *OAuth2CodeSuite) TestTheManifestAuthMethodIsHowAPreregisteredClientAuthenticates() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	profile.Capture, profile.Identity = nil, nil
	s.Require().Equal(core.AuthClientSecretPost, profile.Client.AuthMethod)
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	material, _, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *OAuth2CodeSuite) TestTheLinearManifestDiscoversTheServerAndRegistersAPublicClient() {
	srv := fakeprovider.New(s.T())
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: profile, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"), "the discovered authorize endpoint")
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"), "the resource from RFC 9728 metadata")
	s.Equal("read write", query.Get("scope"))

	material, _, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *OAuth2CodeSuite) TestAClientMetadataDocumentIsUsedBeforeDynamicRegistration() {
	srv := fakeprovider.New(s.T(), fakeprovider.ClientMetadataDocuments)
	var clientID string
	document := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(oauth2code.ClientMetadataDocument(clientID, []string{fakeprovider.RedirectURI}))
	}))
	s.T().Cleanup(document.Close)
	clientID = document.URL + "/oauth/client-metadata.json"
	srv.FetchClientMetadataWith(document.Client())

	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	profile.Client.Policy = []core.ClientOwner{core.ClientDCR, core.ClientCIMD}
	scheme := s.scheme(srv.Client(), oauth2code.Config{ClientMetadataURL: clientID})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: profile, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(clientID, s.query(out.AuthorizeURL).Get("client_id"))
	material, _, err := s.complete(srv, scheme, profile, out)
	s.Require().NoError(err)
	s.Equal(0, srv.Hits(fakeprovider.PathRegister), "CIMD first, so nothing was registered")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

func (s *OAuth2CodeSuite) TestWithoutCIMDSupportTheSchemeFallsBackToRegistration() {
	srv := fakeprovider.New(s.T())
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	profile.Client.Policy = []core.ClientOwner{core.ClientCIMD, core.ClientDCR}
	scheme := s.scheme(srv.Client(), oauth2code.Config{ClientMetadataURL: "https://router.example/oauth/client-metadata.json"})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: profile, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	s.NotEqual("https://router.example/oauth/client-metadata.json", s.query(out.AuthorizeURL).Get("client_id"))
}

func (s *OAuth2CodeSuite) TestARegisteredClientSecretBasicClientAuthenticatesWithBasic() {
	s.connectsAsRegistered(core.AuthClientSecretBasic)
}

func (s *OAuth2CodeSuite) TestARegisteredClientSecretPostClientAuthenticatesInTheBody() {
	s.connectsAsRegistered(core.AuthClientSecretPost)
}

func (s *OAuth2CodeSuite) TestADeniedConsentIsAnAccessDeniedError() {
	srv := fakeprovider.New(s.T(), fakeprovider.ConsentDenied)
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := s.connect(srv, scheme, profile)
	var refused *oauth2code.AuthorizationError
	s.Require().ErrorAs(err, &refused)
	s.Equal("access_denied", refused.Code)
	s.Equal(0, srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CodeSuite) TestAReplayedCallbackIsRefusedBeforeItReachesTheProvider() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	profile.Capture, profile.Identity = nil, nil
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, profile)
	first, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.Require().NoError(err)

	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrReplayedState)
	s.Equal(1, srv.Hits(fakeprovider.PathToken), "the replay never reached the token endpoint")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(first)), "so the server did not revoke the grant")
}

func (s *OAuth2CodeSuite) TestACallbackWithAnotherStateIsRefused() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, profile)
	other, _ := s.consent(srv, scheme, profile)

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: other.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrUnknownState)
	callback.Del("state")
	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrUnknownState)
	s.Equal(0, srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CodeSuite) TestAnAttemptOlderThanTenMinutesIsRefused() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	now := time.Now()
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv), Now: func() time.Time { return now }})
	out, callback := s.consent(srv, scheme, profile)

	now = now.Add(10 * time.Minute)
	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrExpiredState)
}

func (s *OAuth2CodeSuite) TestACallbackFromAnotherIssuerIsRefused() {
	srv := fakeprovider.New(s.T(), fakeprovider.ForeignIssuer)
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	_, _, err := s.connect(srv, scheme, profile)
	s.ErrorIs(err, oauth2code.ErrIssuerMismatch)
	s.Equal(0, srv.Hits(fakeprovider.PathToken), "the code never left for the token endpoint")
}

func (s *OAuth2CodeSuite) TestACallbackWithoutIssFromAServerThatSendsItIsRefused() {
	srv := fakeprovider.New(s.T())
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})
	out, callback := s.consent(srv, scheme, profile)
	callback.Del("iss")

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrIssuerMissing)
}

func (s *OAuth2CodeSuite) TestCommaScopesGoOutAndComeBackWithCommas() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, profile)
	s.Equal("channels:history,chat:write,users:read", s.query(out.AuthorizeURL).Get("scope"))
	s.Require().Empty(callback.Get("error"), "the fake refuses a space-separated list with invalid_scope")

	_, captured, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	s.Require().NoError(err)
	s.Equal([]string{"channels:history", "chat:write", "users:read"}, captured.Scopes)
	s.Equal(srv.TeamID, captured.Metadata["team_id"])
	s.Equal(srv.UserID, captured.Metadata["user_id"])
	s.Equal(srv.TeamID+":"+srv.UserID, captured.AccountID)
}

func (s *OAuth2CodeSuite) TestATokenErrorAnsweredWith200IsAnError() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, profile)
	callback.Set("code", "a-code-nobody-issued")

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: profile, State: out.State, Query: callback})
	var refused *oauth2code.TokenError
	s.Require().ErrorAs(err, &refused)
	s.Equal(http.StatusOK, refused.Status)
	s.Equal("invalid_code", refused.Code)
}

func (s *OAuth2CodeSuite) TestARealmIDInTheCallbackIsCapturedAsTheUnverifiedAccount() {
	srv := fakeprovider.New(s.T(), fakeprovider.CallbackRealmID)
	profile := s.static(srv, s.profile("../../core/testdata/manifests/quickbooks.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, captured, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.Equal(srv.RealmID, captured.Metadata["realm_id"])
	s.Equal(srv.RealmID, captured.AccountID)
	s.Equal([]string{"realm_id"}, captured.Unverified)
	s.Equal("8726400", captured.Metadata["refresh_token_expires_in"])
}

func (s *OAuth2CodeSuite) TestASignedCallbackCompletesAndKeepsItsShopUnverified() {
	srv := fakeprovider.New(s.T(), fakeprovider.SignedCallback)
	profile := s.static(srv, s.profile("../../core/testdata/manifests/shopify.yaml", map[string]string{"shop": srv.Shop}))
	profile.Client.Policy = []core.ClientOwner{core.ClientOperator}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, captured, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.Equal(srv.Shop, captured.Metadata["callback_shop"])
	s.Equal([]string{"callback_shop"}, captured.Unverified)
	s.Equal(srv.Shop, captured.AccountID, "the account is the input, not the callback")
}

func (s *OAuth2CodeSuite) TestAPolicyWithNoAvailableClientIsRefused() {
	srv := fakeprovider.New(s.T())
	profile := s.static(srv, s.profile("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	_, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: profile, RedirectURI: fakeprovider.RedirectURI})
	s.ErrorIs(err, oauth2code.ErrNoClient)
}

func (s *OAuth2CodeSuite) TestEnvClientsFindsTheOperatorClientByTheManifestPrefix() {
	profile := s.profile("../../providers/slack.yaml", nil)
	env := map[string]string{"SLACK_MCP_CLIENT_ID": "id-from-env", "SLACK_MCP_CLIENT_SECRET": "secret-from-env"}
	lookup := oauth2code.EnvClients(func(key string) string { return env[key] })

	found, ok, err := lookup(s.ctx, s.ref, profile, core.ClientOperator)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "id-from-env", Secret: "secret-from-env"}, found)
	_, ok, err = lookup(s.ctx, s.ref, profile, core.ClientCustomer)
	s.Require().NoError(err)
	s.False(ok, "the environment holds the operator's client only")
}

func (s *OAuth2CodeSuite) TestTheClientMetadataDocumentNamesItsOwnURLAndNoSecret() {
	document := oauth2code.ClientMetadataDocument("https://router.example/oauth/client-metadata.json", []string{"https://router.example/callback"})
	raw, err := json.Marshal(document)
	s.Require().NoError(err)
	s.JSONEq(`{
		"client_id": "https://router.example/oauth/client-metadata.json",
		"client_name": "Vision Agents",
		"redirect_uris": ["https://router.example/callback"],
		"grant_types": ["authorization_code", "refresh_token"],
		"response_types": ["code"],
		"token_endpoint_auth_method": "none"
	}`, string(raw))
}

func (s *OAuth2CodeSuite) TestAClientMetadataURLThatIsNotACIMDIdentifierIsRefused() {
	for _, bad := range []string{"http://router.example/client", "https://router.example", "https://router.example/", "https://u@router.example/client", "https://router.example/a/../client", "https://router.example/client#x", "https://router.example/client?x=1"} {
		_, err := oauth2code.New(oauth2code.Config{HTTP: &http.Client{}, ClientMetadataURL: bad})
		s.Error(err, bad)
	}
}

// connectsAsRegistered registers a confidential client for method, whose registration the
// fake then holds to that one method, and connects with it.
func (s *OAuth2CodeSuite) connectsAsRegistered(method core.ClientAuthMethod) {
	srv := fakeprovider.New(s.T())
	profile := s.profile("../../providers/linear.yaml", nil)
	profile.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	profile.Client.AuthMethod = method
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	material, _, err := s.connect(srv, scheme, profile)
	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(material)))
}

// profile is a manifest file resolved for oauth2_code.
func (s *OAuth2CodeSuite) profile(path string, inputs map[string]string) core.Profile {
	raw, err := os.ReadFile(path)
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	profile, err := manifest.Resolve(oauth2code.Name, inputs, nil)
	s.Require().NoError(err)
	return profile
}

// static points a profile's pinned endpoints at the fake, as a manifest that names a
// provider's authorize and token URLs does.
func (s *OAuth2CodeSuite) static(srv *fakeprovider.Server, p core.Profile) core.Profile {
	p.Endpoints = map[string]string{"authorize": srv.URL + fakeprovider.PathAuthorize, "token": srv.URL + fakeprovider.PathToken}
	return p
}

// operator answers the fake's preregistered client as the operator's.
func (s *OAuth2CodeSuite) operator(srv *fakeprovider.Server) oauth2code.ClientLookup {
	return func(_ context.Context, _ core.ConnectionRef, _ core.Profile, owner core.ClientOwner) (oauth2code.Client, bool, error) {
		if owner != core.ClientOperator {
			return oauth2code.Client{}, false, nil
		}
		return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, true, nil
	}
}

func (s *OAuth2CodeSuite) scheme(client *http.Client, cfg oauth2code.Config) *oauth2code.Scheme {
	cfg.HTTP = client
	scheme, err := oauth2code.New(cfg)
	s.Require().NoError(err)
	return scheme
}

// consent runs Begin and the fake's browser, and returns the attempt and the callback query.
func (s *OAuth2CodeSuite) consent(srv *fakeprovider.Server, scheme *oauth2code.Scheme, p core.Profile) (core.BeginOutput, url.Values) {
	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Profile: p, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	callback, err := srv.Consent(out.AuthorizeURL)
	s.Require().NoError(err)
	return out, callback.Query()
}

func (s *OAuth2CodeSuite) complete(srv *fakeprovider.Server, scheme *oauth2code.Scheme, p core.Profile, out core.BeginOutput) (core.Material, core.Captured, error) {
	callback, err := srv.Consent(out.AuthorizeURL)
	s.Require().NoError(err)
	return scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: p, State: out.State, Query: callback.Query()})
}

// connect is a whole consent: Begin, the browser, Complete.
func (s *OAuth2CodeSuite) connect(srv *fakeprovider.Server, scheme *oauth2code.Scheme, p core.Profile) (core.Material, core.Captured, error) {
	out, callback := s.consent(srv, scheme, p)
	return scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Profile: p, State: out.State, Query: callback})
}

// accessToken reads the access token out of the material. Part 2's Wrap is what will use
// it; until then this is the only way to show the token works.
func (s *OAuth2CodeSuite) accessToken(m core.Material) string {
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.Unmarshal(m.Payload, &payload))
	s.Require().NotEmpty(payload.AccessToken)
	return payload.AccessToken
}

// call is a tools/call to the fake's MCP endpoint with token; its status says whether the
// token is good.
func (s *OAuth2CodeSuite) call(srv *fakeprovider.Server, token string) int {
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

func (s *OAuth2CodeSuite) query(raw string) url.Values {
	u, err := url.Parse(raw)
	s.Require().NoError(err)
	return u.Query()
}
