package oauth2code_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
)

// OAuth2CodeSuite drives the scheme against the fake provider, with the built-in manifests
// and core's fixtures where one fits. It is an outside package, so it sees only what the
// router does.
type OAuth2CodeSuite struct {
	suite.Suite
	ctx context.Context
	ref core.ConnectionRef
	// now is the clock the access credential tests give the scheme; each read moves it by step.
	now  time.Time
	step time.Duration
}

func TestOAuth2CodeSuite(t *testing.T) {
	suite.Run(t, new(OAuth2CodeSuite))
}

func (s *OAuth2CodeSuite) SetupTest() {
	s.ctx = context.Background()
	s.ref = core.ConnectionRef{CustomerID: "acme", ConnectionID: "conn-1"}
	s.now, s.step = time.Now(), 0
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
	resolved := s.resolve("../../providers/slack.yaml", nil)
	// Nothing is fetched for a manifest that pins its endpoints, so the client has nowhere
	// to go.
	scheme := s.scheme(&http.Client{}, oauth2code.Config{Clients: func(context.Context, core.ConnectionRef, core.ResolvedManifest, core.ClientOwner) (oauth2code.Client, bool, error) {
		return oauth2code.Client{ID: "operator-client", Secret: "operator-secret"}, true, nil
	}})
	out, err := scheme.Begin(s.ctx, core.BeginInput{
		Ref: s.ref, Manifest: resolved, RedirectURI: "https://router.example/v1/agents/connectors/oauth/callback",
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
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	resolved.Client.AuthMethod = ""
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	stored, account, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.Equal(oauth2code.Name, stored.Scheme)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)), "the token the exchange returned works")
	s.Equal([]string{"channels:history", "chat:write", "users:read"}, account.Scopes)
}

func (s *OAuth2CodeSuite) TestTheManifestAuthMethodIsHowAPreregisteredClientAuthenticates() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	s.Require().Equal(core.AuthClientSecretPost, resolved.Client.AuthMethod)
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)))
}

func (s *OAuth2CodeSuite) TestTheLinearManifestDiscoversTheServerAndRegistersAPublicClient() {
	srv := fakeprovider.New(s.T())
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	query := s.query(out.AuthorizeURL)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+fakeprovider.PathAuthorize+"?"), "the discovered authorize endpoint")
	s.Equal(srv.URL+fakeprovider.PathMCP, query.Get("resource"), "the resource from RFC 9728 metadata")
	s.Equal("read write", query.Get("scope"))

	stored, _, err := s.complete(srv, scheme, resolved, out)
	s.Require().NoError(err)
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)))
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

	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	resolved.Client.Policy = []core.ClientOwner{core.ClientDCR, core.ClientCIMD}
	scheme := s.scheme(srv.Client(), oauth2code.Config{ClientMetadataURL: clientID})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(clientID, s.query(out.AuthorizeURL).Get("client_id"))
	stored, _, err := s.complete(srv, scheme, resolved, out)
	s.Require().NoError(err)
	s.Equal(0, srv.Hits(fakeprovider.PathRegister), "CIMD first, so nothing was registered")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)))
}

func (s *OAuth2CodeSuite) TestWithoutCIMDSupportTheSchemeFallsBackToRegistration() {
	srv := fakeprovider.New(s.T())
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	resolved.Client.Policy = []core.ClientOwner{core.ClientCIMD, core.ClientDCR}
	scheme := s.scheme(srv.Client(), oauth2code.Config{ClientMetadataURL: "https://router.example/oauth/client-metadata.json"})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
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
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, _, err := s.connect(srv, scheme, resolved)
	var refused *oauth2code.AuthorizationError
	s.Require().ErrorAs(err, &refused)
	s.Equal("access_denied", refused.Code)
	s.Equal(0, srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CodeSuite) TestAReplayedCallbackIsRefusedBeforeItReachesTheProvider() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	resolved.Capture, resolved.Identity = nil, nil
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, resolved)
	first, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.Require().NoError(err)

	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrReplayedState)
	s.Equal(1, srv.Hits(fakeprovider.PathToken), "the replay never reached the token endpoint")
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(first)), "so the server did not revoke the grant")
}

func (s *OAuth2CodeSuite) TestACallbackWithAnotherStateIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, resolved)
	other, _ := s.consent(srv, scheme, resolved)

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: other.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrUnknownState)
	callback.Del("state")
	_, _, err = scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrUnknownState)
	s.Equal(0, srv.Hits(fakeprovider.PathToken))
}

func (s *OAuth2CodeSuite) TestAnAttemptOlderThanTenMinutesIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	now := time.Now()
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv), Now: func() time.Time { return now }})
	out, callback := s.consent(srv, scheme, resolved)

	now = now.Add(10 * time.Minute)
	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrExpiredState)
}

func (s *OAuth2CodeSuite) TestACallbackFromAnotherIssuerIsRefused() {
	srv := fakeprovider.New(s.T(), fakeprovider.ForeignIssuer)
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	_, _, err := s.connect(srv, scheme, resolved)
	s.ErrorIs(err, oauth2code.ErrIssuerMismatch)
	s.Equal(0, srv.Hits(fakeprovider.PathToken), "the code never left for the token endpoint")
}

func (s *OAuth2CodeSuite) TestACallbackWithoutIssFromAServerThatSendsItIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	scheme := s.scheme(srv.Client(), oauth2code.Config{})
	out, callback := s.consent(srv, scheme, resolved)
	callback.Del("iss")

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.ErrorIs(err, oauth2code.ErrIssuerMissing)
}

func (s *OAuth2CodeSuite) TestCommaScopesGoOutAndComeBackWithCommas() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, resolved)
	s.Equal("channels:history,chat:write,users:read", s.query(out.AuthorizeURL).Get("scope"))
	s.Require().Empty(callback.Get("error"), "the fake refuses a space-separated list with invalid_scope")

	_, account, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	s.Require().NoError(err)
	s.Equal([]string{"channels:history", "chat:write", "users:read"}, account.Scopes)
	s.Equal(srv.TeamID, account.Metadata["team_id"])
	s.Equal(srv.UserID, account.Metadata["user_id"])
	s.Equal(srv.TeamID+":"+srv.UserID, account.AccountID)
}

func (s *OAuth2CodeSuite) TestATokenErrorAnsweredWith200IsAnError() {
	srv := fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})
	out, callback := s.consent(srv, scheme, resolved)
	callback.Set("code", "a-code-nobody-issued")

	_, _, err := scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: resolved, State: out.State, Query: callback})
	var refused *oauth2code.TokenError
	s.Require().ErrorAs(err, &refused)
	s.Equal(http.StatusOK, refused.Status)
	s.Equal("invalid_code", refused.Code)
}

func (s *OAuth2CodeSuite) TestARealmIDInTheCallbackIsCapturedAsTheUnverifiedAccount() {
	srv := fakeprovider.New(s.T(), fakeprovider.CallbackRealmID)
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/quickbooks.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, account, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.Equal(srv.RealmID, account.Metadata["realm_id"])
	s.Equal(srv.RealmID, account.AccountID)
	s.Equal([]string{"realm_id"}, account.Unverified)
	s.Equal("8726400", account.Metadata["refresh_token_expires_in"])
}

func (s *OAuth2CodeSuite) TestASignedCallbackCompletesAndKeepsItsShopUnverified() {
	srv := fakeprovider.New(s.T(), fakeprovider.SignedCallback)
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/shopify.yaml", map[string]string{"shop": srv.Shop}))
	resolved.Client.Policy = []core.ClientOwner{core.ClientOperator}
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: s.operator(srv)})

	_, account, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.Equal(srv.Shop, account.Metadata["callback_shop"])
	s.Equal([]string{"callback_shop"}, account.Unverified)
	s.Equal(srv.Shop, account.AccountID, "the account is the input, not the callback")
}

func (s *OAuth2CodeSuite) TestAPolicyWithNoAvailableClientIsRefused() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../core/testdata/manifests/slack.yaml", nil))
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	_, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
	s.ErrorIs(err, oauth2code.ErrNoClient)
}

func (s *OAuth2CodeSuite) TestEnvClientsFindsTheOperatorClientByTheManifestPrefix() {
	resolved := s.resolve("../../providers/slack.yaml", nil)
	env := map[string]string{"SLACK_MCP_CLIENT_ID": "id-from-env", "SLACK_MCP_CLIENT_SECRET": "secret-from-env"}
	lookup := oauth2code.EnvClients(func(key string) string { return env[key] })

	found, ok, err := lookup(s.ctx, s.ref, resolved, core.ClientOperator)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "id-from-env", Secret: "secret-from-env"}, found)
	_, ok, err = lookup(s.ctx, s.ref, resolved, core.ClientCustomer)
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
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = map[string]string{"mcp": srv.URL + fakeprovider.PathMCP}
	resolved.Client.AuthMethod = method
	scheme := s.scheme(srv.Client(), oauth2code.Config{})

	stored, _, err := s.connect(srv, scheme, resolved)
	s.Require().NoError(err)
	s.Equal(1, srv.Hits(fakeprovider.PathRegister))
	s.Equal(http.StatusOK, s.call(srv, s.accessToken(stored)))
}

func (s *OAuth2CodeSuite) TestADiscoveredEndpointThatIsNotPublicIsRefused() {
	for name, authorize := range map[string]func(base string) string{
		"a private address": func(string) string { return "https://10.0.0.1/authorize" },
		"userinfo": func(base string) string {
			return strings.Replace(base, "https://", "https://user:pass@", 1) + "/authorize"
		},
		"plain http": func(base string) string { return strings.Replace(base, "https://", "http://", 1) + "/authorize" },
		"a fragment": func(base string) string { return base + "/authorize#x" },
	} {
		srv := s.metadataServer(func(base string) map[string]any {
			return map[string]any{"/.well-known/oauth-authorization-server": asMetadata(base, authorize(base))}
		})
		scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: operatorClient})

		_, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"issuer": srv.URL}), RedirectURI: fakeprovider.RedirectURI})
		s.Require().Error(err, name)
		s.Contains(err.Error(), "endpoint", name)
	}
}

func (s *OAuth2CodeSuite) TestADiscoveredAuthorizeEndpointKeepsItsQuery() {
	srv := s.metadataServer(func(base string) map[string]any {
		return map[string]any{"/.well-known/oauth-authorization-server": asMetadata(base, base+"/authorize?tenant=acme")}
	})
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: operatorClient})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"issuer": srv.URL}), RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	query := s.query(out.AuthorizeURL)
	s.Equal("acme", query.Get("tenant"), "RFC 6749 section 3.1: the endpoint's query is retained")
	s.Equal("operator-client", query.Get("client_id"))
}

func (s *OAuth2CodeSuite) TestByDefaultEndpointsAreHeldToTheEgressPolicy() {
	srv := fakeprovider.New(s.T())
	resolved := s.static(srv, s.resolve("../../providers/linear.yaml", nil))
	resolved.Client.Policy = []core.ClientOwner{core.ClientOperator}
	scheme, err := oauth2code.New(oauth2code.Config{HTTP: srv.Client(), Clients: s.operator(srv)})
	s.Require().NoError(err)

	_, err = scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: resolved, RedirectURI: fakeprovider.RedirectURI})
	s.Require().ErrorContains(err, "egress:", "the fake listens on loopback, which egress refuses")
}

func (s *OAuth2CodeSuite) TestAMismatchedPathDocumentFallsBackToTheRootDocument() {
	srv := s.metadataServer(func(base string) map[string]any {
		return map[string]any{
			"/.well-known/oauth-protected-resource/mcp": map[string]any{"resource": "https://elsewhere.example/mcp", "authorization_servers": []string{base}},
			"/.well-known/oauth-protected-resource":     map[string]any{"resource": base, "authorization_servers": []string{base}},
			"/.well-known/oauth-authorization-server":   asMetadata(base, base+"/authorize"),
		}
	})
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: operatorClient})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"mcp": srv.URL + "/mcp"}), RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(srv.URL, s.query(out.AuthorizeURL).Get("resource"), "the root document, after the mismatched one")
}

func (s *OAuth2CodeSuite) TestARefusedRedirectFallsBackToTheRootDocument() {
	// Slack's shape: the path-inserted URL redirects to another host, which egress refuses.
	srv := s.metadataServer(func(base string) map[string]any {
		return map[string]any{
			"/.well-known/oauth-protected-resource/mcp": http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, "https://mcp-9827.example/.well-known/oauth-protected-resource/mcp", http.StatusFound)
			}),
			"/.well-known/oauth-protected-resource":   map[string]any{"resource": base, "authorization_servers": []string{base}},
			"/.well-known/oauth-authorization-server": asMetadata(base, base+"/authorize"),
		}
	})
	client := srv.Client()
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return errors.New("redirect refused") }
	scheme := s.scheme(client, oauth2code.Config{Clients: operatorClient})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"mcp": srv.URL + "/mcp"}), RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.Equal(srv.URL, s.query(out.AuthorizeURL).Get("resource"))
}

func (s *OAuth2CodeSuite) TestAMismatchedIssuerDocumentFallsBackToTheNextURL() {
	srv := s.metadataServer(func(base string) map[string]any {
		issuer := base + "/tenant"
		return map[string]any{
			"/.well-known/oauth-authorization-server/tenant": asMetadata("https://wrong.example/tenant", base+"/wrong/authorize"),
			"/tenant/.well-known/openid-configuration":       asMetadata(issuer, base+"/oidc/authorize"),
		}
	})
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: operatorClient})

	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"issuer": srv.URL + "/tenant"}), RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	s.True(strings.HasPrefix(out.AuthorizeURL, srv.URL+"/oidc/authorize?"), "the third URL's document, after a mismatch and a 404: %s", out.AuthorizeURL)
}

func (s *OAuth2CodeSuite) TestDiscoveryWithNoUsableDocumentNamesEachFailure() {
	srv := s.metadataServer(func(base string) map[string]any {
		return map[string]any{
			"/.well-known/oauth-protected-resource/mcp": map[string]any{"resource": "https://elsewhere.example/mcp", "authorization_servers": []string{base}},
			"/.well-known/oauth-protected-resource":     http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusInternalServerError) }),
		}
	})
	scheme := s.scheme(srv.Client(), oauth2code.Config{Clients: operatorClient})

	_, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: s.discovering(map[string]string{"mcp": srv.URL + "/mcp"}), RedirectURI: fakeprovider.RedirectURI})
	s.Require().Error(err)
	s.Contains(err.Error(), "elsewhere.example")
	s.Contains(err.Error(), "HTTP 500")
}

// resolve is a manifest file resolved for oauth2_code.
func (s *OAuth2CodeSuite) resolve(path string, inputs map[string]string) core.ResolvedManifest {
	raw, err := os.ReadFile(path)
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	resolved, err := manifest.Resolve(oauth2code.Name, inputs, nil)
	s.Require().NoError(err)
	return resolved
}

// static points a resolved manifest's pinned endpoints at the fake, as a manifest that names a
// provider's authorize and token URLs does.
func (s *OAuth2CodeSuite) static(srv *fakeprovider.Server, m core.ResolvedManifest) core.ResolvedManifest {
	m.Endpoints = map[string]string{"authorize": srv.URL + fakeprovider.PathAuthorize, "token": srv.URL + fakeprovider.PathToken}
	return m
}

// operator answers the fake's preregistered client as the operator's.
func (s *OAuth2CodeSuite) operator(srv *fakeprovider.Server) oauth2code.ClientLookup {
	return func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, owner core.ClientOwner) (oauth2code.Client, bool, error) {
		if owner != core.ClientOperator {
			return oauth2code.Client{}, false, nil
		}
		return oauth2code.Client{ID: srv.ClientID, Secret: srv.ClientSecret}, true, nil
	}
}

func (s *OAuth2CodeSuite) scheme(client *http.Client, cfg oauth2code.Config) *oauth2code.Scheme {
	cfg.HTTP = client
	if cfg.PublicEndpoint == nil {
		cfg.PublicEndpoint = loopbackOrPublic
	}
	scheme, err := oauth2code.New(cfg)
	s.Require().NoError(err)
	return scheme
}

// consent runs Begin and the fake's browser, and returns the attempt and the callback query.
func (s *OAuth2CodeSuite) consent(srv *fakeprovider.Server, scheme *oauth2code.Scheme, m core.ResolvedManifest) (core.BeginOutput, url.Values) {
	out, err := scheme.Begin(s.ctx, core.BeginInput{Ref: s.ref, Manifest: m, RedirectURI: fakeprovider.RedirectURI})
	s.Require().NoError(err)
	callback, err := srv.Consent(out.AuthorizeURL)
	s.Require().NoError(err)
	return out, callback.Query()
}

func (s *OAuth2CodeSuite) complete(srv *fakeprovider.Server, scheme *oauth2code.Scheme, m core.ResolvedManifest, out core.BeginOutput) (core.StoredCredentials, core.AccountInfo, error) {
	callback, err := srv.Consent(out.AuthorizeURL)
	s.Require().NoError(err)
	return scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: m, State: out.State, Query: callback.Query()})
}

// connect is a whole consent: Begin, the browser, Complete.
func (s *OAuth2CodeSuite) connect(srv *fakeprovider.Server, scheme *oauth2code.Scheme, m core.ResolvedManifest) (core.StoredCredentials, core.AccountInfo, error) {
	out, callback := s.consent(srv, scheme, m)
	return scheme.Complete(s.ctx, core.CompleteInput{Ref: s.ref, Manifest: m, State: out.State, Query: callback})
}

// accessToken reads the access token out of the stored credentials, so a test can call the fake with
// it directly or tell one issued token from another.
func (s *OAuth2CodeSuite) accessToken(stored core.StoredCredentials) string {
	var payload struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.Unmarshal(stored.Payload, &payload))
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

// loopbackOrPublic is egress's endpoint check with one hole: a loopback IP literal, which
// is where every fake and metadata server in this suite listens. Everything else, private
// addresses and userinfo included, is refused exactly as in the router.
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}

// metadataServer is a TLS server that answers each path with a JSON document, or with the
// handler given for it; anything else is a 404. documents is built after the server
// starts, so a document can name the server's own URL.
func (s *OAuth2CodeSuite) metadataServer(documents func(base string) map[string]any) *httptest.Server {
	var routes map[string]any
	srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch route := routes[r.URL.Path].(type) {
		case nil:
			http.NotFound(w, r)
		case http.HandlerFunc:
			route(w, r)
		default:
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(route)
		}
	}))
	s.T().Cleanup(srv.Close)
	routes = documents(srv.URL)
	return srv
}

// operatorClient answers any lookup for the operator with a fixed public client, for
// resolved manifests whose servers are metadata servers rather than the fake.
func operatorClient(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, owner core.ClientOwner) (oauth2code.Client, bool, error) {
	return oauth2code.Client{ID: "operator-client"}, owner == core.ClientOperator, nil
}

// asMetadata is RFC 8414 metadata for issuer, with its authorize endpoint given.
func asMetadata(issuer, authorize string) map[string]any {
	return map[string]any{
		"issuer":                           issuer,
		"authorization_endpoint":           authorize,
		"token_endpoint":                   issuer + "/token",
		"code_challenge_methods_supported": []string{"S256"},
	}
}

// discovering is the Linear manifest pointed at endpoints, with the operator's client.
func (s *OAuth2CodeSuite) discovering(endpoints map[string]string) core.ResolvedManifest {
	resolved := s.resolve("../../providers/linear.yaml", nil)
	resolved.Endpoints = endpoints
	resolved.Client.Policy = []core.ClientOwner{core.ClientOperator}
	return resolved
}
