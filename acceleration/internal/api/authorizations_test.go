//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/cookiejar"
	"net/netip"
	"net/url"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The router's public URL and dashboard in AuthorizationsSuite. Neither host is dialed: the
// suite's browsers send router.example to the suite's own listener, so the router sees an
// https public URL (and sets Secure cookies) while serving plain http locally.
const (
	consentPublicURL = "https://router.example"
	consentDashboard = "https://dashboard.example/connections"
)

// refusingHook is the before_complete hook a manifest of the suite names to refuse every
// callback.
const refusingHook = "test.refuse_callback"

// AuthorizationsSuite is a connector consent from end to end: the backend starts it, a
// browser opens the launch page, hands off, consents at the fake provider and comes back to
// the callback. One fake provider serves the whole suite, since the scheme the router is
// built with is fixed when the suite starts; each test makes its own app and connection.
type AuthorizationsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
}

func TestAuthorizationsSuite(t *testing.T) {
	runSuite(t, new(AuthorizationsSuite))
}

// SetupSuite builds oauth2_code against the fake, with the fake's preregistered client as
// the operator's, and a hook that refuses every callback.
func (s *AuthorizationsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.CommaScopes)
	s.provider.AllowRedirect(consentPublicURL + ConnectorCallbackPath)
	scheme, err := oauth2code.New(oauth2code.Config{
		HTTP: s.provider.Client(),
		Clients: func(_ context.Context, _ core.ConnectionRef, _ core.ResolvedManifest, source core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
			if source != core.ClientOperator {
				return oauth2code.Client{}, false, nil
			}
			return oauth2code.Client{ID: s.provider.ClientID, Secret: s.provider.ClientSecret}, true, nil
		},
		PublicEndpoint: loopbackOrPublic,
	})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes: map[string]core.Scheme{oauth2code.Name: scheme},
		Hooks: map[string]core.Hook{refusingHook: func(context.Context, *core.HookContext) error {
			return errors.New("the test hook refuses every callback")
		}},
	}
	s.publicURL, s.dashboardURL = consentPublicURL, consentDashboard
	s.RouterSuite.SetupSuite()
}

func (s *AuthorizationsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.provider.Use(fakeprovider.CommaScopes)
}

func (s *AuthorizationsSuite) TestOnlyTheAppsBackendMayStartAConsent() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		id := s.connection("")
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations", nil, nil)
	})
}

func (s *AuthorizationsSuite) TestAliceCannotStartAConsentForBobsConnection() {
	alice, bob := s.data.createUser(), s.data.createUser()
	var bobs Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.actingFor(bob).do(http.MethodPost,
		"/v1/agents/connections", userOwned(s.connector(""), bob), &bobs))

	status, answered := s.serverClient.actingFor(alice).call(http.MethodPost, "/v1/agents/connections/"+bobs.ID+"/authorizations", nil)
	_, missing := s.serverClient.actingFor(alice).call(http.MethodPost, "/v1/agents/connections/"+s.utils.uuid()+"/authorizations", nil)

	s.Equal(http.StatusNotFound, status)
	s.Equal(withoutDuration(missing), withoutDuration(answered), "the same answer as a connection that does not exist")
	s.Equal(http.StatusCreated, s.serverClient.actingFor(bob).do(http.MethodPost,
		"/v1/agents/connections/"+bobs.ID+"/authorizations", nil, nil), "Bob's backend still may")
}

func (s *AuthorizationsSuite) TestAnotherAppCannotStartAConsentForTheAppsConnection() {
	id := s.connection("")

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations", nil, nil)
	})
}

func (s *AuthorizationsSuite) TestAConsentConnectsTheAccountItWasStartedFor() {
	id := s.connection("")
	started := s.start(id)
	s.Equal(AuthorizationKind(store.AttemptConsent), started.Kind)
	s.Equal(consentPublicURL+connectorLaunchPath+started.ID, started.LaunchURL)
	s.WithinDuration(time.Now().Add(10*time.Minute), started.ExpiresAt, time.Minute)

	alice := s.browser()
	finished := alice.finish(s.consent(alice.handOff(started)))

	s.Equal(s.landing(id, consentConnected), finished.Header.Get("Location"))
	connection := s.get(id)
	s.Equal(ConnectionStatus(store.ConnectionConnected), connection.Status)
	s.Equal(2, connection.Revision, "the first credentials")
	s.Equal(s.provider.TeamID, connection.Metadata["team_id"])
	s.Equal(s.provider.UserID, connection.Metadata["user_id"])
	s.NotEmpty(connection.AccountID)
	s.Equal([]string{"channels:history", "chat:write"}, connection.GrantedScopes)
}

func (s *AuthorizationsSuite) TestTheHandoffBindsTheConsentWithAnHttpOnlySecureLaxCookieOnTheCallbackAlone() {
	started := s.start(s.connection(""))

	response := s.browser().post(started.LaunchURL, consentPublicURL, map[string]string{"handoff_token": started.HandoffToken})

	s.Require().Equal(http.StatusOK, response.StatusCode)
	cookies := response.Cookies()
	s.Require().Len(cookies, 1)
	s.Equal(attemptCookiePrefix+started.ID, cookies[0].Name)
	s.Equal(ConnectorCallbackPath, cookies[0].Path)
	s.True(cookies[0].HttpOnly)
	s.True(cookies[0].Secure, "the public URL is https")
	s.Equal(http.SameSiteLaxMode, cookies[0].SameSite)
	s.InDelta(attemptLifetime.Seconds(), float64(cookies[0].MaxAge), 60)
}

func (s *AuthorizationsSuite) TestTheCallbackClearsTheCookie() {
	alice := s.browser()
	callback := s.consent(alice.handOff(s.start(s.connection(""))))

	finished := alice.finish(callback)

	s.Require().Equal(http.StatusFound, finished.StatusCode)
	s.Empty(alice.jar.Cookies(callback), "the browser holds no consent cookie any more")
}

func (s *AuthorizationsSuite) TestADeniedConsentLeavesTheConnectionPending() {
	id := s.connection("")
	alice := s.browser()
	authorize := alice.handOff(s.start(id))
	s.provider.Use(fakeprovider.CommaScopes, fakeprovider.ConsentDenied)
	exchanges := s.provider.Hits(fakeprovider.PathToken)

	finished := alice.finish(s.consent(authorize))

	s.Equal(s.landing(id, consentDenied), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
	s.Equal(1, s.get(id).Revision)
	s.Equal(exchanges, s.provider.Hits(fakeprovider.PathToken), "nothing was exchanged")
}

func (s *AuthorizationsSuite) TestAConsentFinishedInAnotherBrowserIsRefused() {
	id := s.connection("")
	alice, mallory := s.browser(), s.browser()
	callback := s.consent(alice.handOff(s.start(id)))
	exchanges := s.provider.Hits(fakeprovider.PathToken)

	refused := mallory.finish(callback)

	s.Equal(http.StatusForbidden, refused.StatusCode)
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
	s.Equal(exchanges, s.provider.Hits(fakeprovider.PathToken), "the code never left for the provider")
	finished := alice.finish(callback)
	s.Equal(s.landing(id, consentConnected), finished.Header.Get("Location"), "the browser that began it still can")
}

func (s *AuthorizationsSuite) TestAReplayedCallbackIsRefused() {
	id := s.connection("")
	alice := s.browser()
	callback := s.consent(alice.handOff(s.start(id)))
	cookie := alice.jar.Cookies(callback)
	s.Require().Equal(s.landing(id, consentConnected), alice.finish(callback).Header.Get("Location"))
	connected := s.get(id)
	exchanges := s.provider.Hits(fakeprovider.PathToken)
	// The cookie is put back, as a browser that kept a copy of it would send it.
	alice.jar.SetCookies(callback, cookie)

	replayed := alice.finish(callback)

	s.Equal(http.StatusBadRequest, replayed.StatusCode)
	s.Equal(connected, s.get(id), "the grant the first callback stored is untouched")
	s.Equal(exchanges, s.provider.Hits(fakeprovider.PathToken), "the code was not sent again")
}

func (s *AuthorizationsSuite) TestAnExpiredConsentIsRefused() {
	id := s.connection("")
	alice := s.browser()
	started := s.start(id)
	callback := s.consent(alice.handOff(started))
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connector_authorization_attempts SET expires_at = now() - interval '1 second' WHERE id = ?", started.ID)
	s.Require().NoError(err)

	expired := alice.finish(callback)

	s.Equal(http.StatusBadRequest, expired.StatusCode)
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
}

func (s *AuthorizationsSuite) TestACallbackNamingAnotherIssuerIsRefused() {
	id := s.connection("")
	alice := s.browser()
	authorize := alice.handOff(s.start(id))
	s.provider.Use(fakeprovider.CommaScopes, fakeprovider.ForeignIssuer)
	exchanges := s.provider.Hits(fakeprovider.PathToken)

	finished := alice.finish(s.consent(authorize))

	s.Equal(s.landing(id, consentFailed), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
	s.Equal(exchanges, s.provider.Hits(fakeprovider.PathToken), "a mix-up never reaches the token endpoint")
}

func (s *AuthorizationsSuite) TestAReconnectForTheSameAccountReplacesTheGrant() {
	id := s.connection("")
	s.connect(id)

	reconnect := s.start(id)
	alice := s.browser()
	finished := alice.finish(s.consent(alice.handOff(reconnect)))

	s.Equal(AuthorizationKind(store.AttemptReconnect), reconnect.Kind)
	s.Equal(s.landing(id, consentConnected), finished.Header.Get("Location"))
	s.Equal(3, s.get(id).Revision, "new credentials")
}

func (s *AuthorizationsSuite) TestAReconnectForAnotherAccountKeepsTheOldGrantAndSaysSo() {
	id := s.connection("")
	s.connect(id)
	connected := s.get(id)
	s.provider.SwitchAccount()

	alice := s.browser()
	finished := alice.finish(s.consent(alice.handOff(s.start(id))))

	s.Equal(s.landing(id, consentAccountMismatch), finished.Header.Get("Location"))
	kept := s.get(id)
	s.Equal(connected.AccountID, kept.AccountID)
	s.Equal(connected.Metadata, kept.Metadata)
	s.Equal(connected.Revision, kept.Revision, "the credentials were not replaced")
	s.Equal(ConnectionStatus(store.ConnectionConnected), kept.Status)
	stored, err := s.store.ConnectorConnection(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	s.Equal(accountSwitchError, stored.LastError)
}

func (s *AuthorizationsSuite) TestABeforeCompleteHookThatRefusesKeepsTheCodeFromTheProvider() {
	id := s.connection("hooks:\n  before_complete: " + refusingHook + "\n")
	alice := s.browser()
	authorize := alice.handOff(s.start(id))
	exchanges := s.provider.Hits(fakeprovider.PathToken)

	finished := alice.finish(s.consent(authorize))

	s.Equal(s.landing(id, consentFailed), finished.Header.Get("Location"))
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
	s.Equal(exchanges, s.provider.Hits(fakeprovider.PathToken))
}

func (s *AuthorizationsSuite) TestAHandoffFromAnotherOriginIsRefused() {
	started := s.start(s.connection(""))

	response := s.browser().post(started.LaunchURL, "https://elsewhere.example", map[string]string{"handoff_token": started.HandoffToken})

	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Empty(response.Cookies())
}

func (s *AuthorizationsSuite) TestAHandoffWithAnotherConsentsTokenIsRefused() {
	started, other := s.start(s.connection("")), s.start(s.connection(""))

	response := s.browser().post(started.LaunchURL, consentPublicURL, map[string]string{"handoff_token": other.HandoffToken})

	s.Equal(http.StatusForbidden, response.StatusCode)
	s.Empty(response.Cookies())
}

func (s *AuthorizationsSuite) TestTheLaunchPageRunsOnlyItsOwnScriptAndOnlyForTheDashboard() {
	started := s.start(s.connection(""))

	response := s.browser().get(started.LaunchURL)

	s.Require().Equal(http.StatusOK, response.StatusCode)
	page := s.body(response)
	policy := response.Header.Get("Content-Security-Policy")
	nonce := regexp.MustCompile(`'nonce-([A-Za-z0-9_-]+)'`).FindStringSubmatch(policy)
	s.Require().Len(nonce, 2, policy)
	s.Contains(page, `<script nonce="`+nonce[1]+`">`)
	s.Contains(policy, "default-src 'none'")
	s.Contains(policy, "frame-ancestors 'none'")
	s.Contains(page, `const dashboardOrigin = "https://dashboard.example";`)
	s.NotContains(page, started.HandoffToken)
	s.Equal("no-store", response.Header.Get("Cache-Control"))
	s.Equal("DENY", response.Header.Get("X-Frame-Options"))
	s.NotEqual(nonce[1], regexp.MustCompile(`'nonce-([A-Za-z0-9_-]+)'`).
		FindStringSubmatch(s.browser().get(started.LaunchURL).Header.Get("Content-Security-Policy"))[1], "a nonce per response")
}

func (s *AuthorizationsSuite) TestTheClientMetadataDocumentNamesItsOwnURLAndTheCallback() {
	response := s.browser().get(consentPublicURL + ConnectorClientMetadataPath)

	s.Require().Equal(http.StatusOK, response.StatusCode)
	var document oauth2code.ClientMetadata
	s.Require().NoError(json.Unmarshal([]byte(s.body(response)), &document))
	s.Equal(consentPublicURL+ConnectorClientMetadataPath, document.ClientID, "CIMD section 4")
	s.Equal([]string{consentPublicURL + ConnectorCallbackPath}, document.RedirectURIs)
	s.Equal("none", document.TokenEndpointAuthMethod)
}

// connection is a pending app-owned connection to a connector of the test's app at the
// fake provider; extra is more manifest YAML, such as hooks.
func (s *AuthorizationsSuite) connection(extra string) string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(s.connector(extra)), &created))
	return created.ID
}

// connector stores a connector of the test's app whose endpoints are the fake provider's,
// in the shape of core's Slack fixture (testdata/manifests/slack.yaml): comma scopes, and
// the account as team and user from the token response, which CommaScopes answers.
func (s *AuthorizationsSuite) connector(extra string) string {
	id := "custom_fake" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Fake
endpoints:
  authorize: ` + s.provider.URL + fakeprovider.PathAuthorize + `
  token: ` + s.provider.URL + fakeprovider.PathToken + `
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [oauth2_code]
client:
  registration: [operator]
  auth_method: client_secret_post
scopes:
  list: [channels:history, chat:write]
  separator: ","
capture:
  - name: team_id
    from: token_response
    path: $.team.id
  - name: user_id
    from: token_response
    path: $.authed_user.id
identity: [team_id, user_id]
sources:
  - kind: mcp
    endpoint: mcp
` + extra))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// start is the app's backend starting a consent.
func (s *AuthorizationsSuite) start(id string) Authorization {
	var started Authorization
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/authorizations", nil, &started))
	return started
}

// connect is a whole consent that ends connected.
func (s *AuthorizationsSuite) connect(id string) {
	b := s.browser()
	s.Require().Equal(s.landing(id, consentConnected), b.finish(s.consent(b.handOff(s.start(id)))).Header.Get("Location"))
}

// consent is the person at the provider approving: the callback the provider sends the
// browser back to.
func (s *AuthorizationsSuite) consent(authorize string) *url.URL {
	callback, err := s.provider.Consent(authorize)
	s.Require().NoError(err)
	s.Require().True(strings.HasPrefix(callback.String(), consentPublicURL+ConnectorCallbackPath+"?"), callback.String())
	return callback
}

func (s *AuthorizationsSuite) get(id string) Connection {
	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, &connection))
	return connection
}

// landing is where the callback sends the browser for a consent that ended as status.
func (s *AuthorizationsSuite) landing(id, status string) string {
	return consentDashboard + "?" + url.Values{"connection_id": {id}, "status": {status}}.Encode()
}

func (s *AuthorizationsSuite) body(response *http.Response) string {
	defer response.Body.Close()
	read, err := readAll(response)
	s.Require().NoError(err)
	return string(read)
}

// browser is one person's browser: a cookie jar of its own, redirects not followed, and
// router.example reached at the suite's listener.
type browser struct {
	suite  *AuthorizationsSuite
	jar    *cookiejar.Jar
	client *http.Client
}

func (s *AuthorizationsSuite) browser() *browser {
	jar, err := cookiejar.New(nil)
	s.Require().NoError(err)
	listener := s.server.Listener.Addr().String()
	return &browser{suite: s, jar: jar, client: &http.Client{
		Jar: jar,
		Transport: roundTripper(func(r *http.Request) (*http.Response, error) {
			sent := r.Clone(r.Context())
			sent.URL.Scheme, sent.URL.Host, sent.Host = "http", listener, ""
			return http.DefaultTransport.RoundTrip(sent)
		}),
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}}
}

// handOff is the launch page in this browser: it posts the handoff token from the router's
// own origin, as its fetch does, and returns the provider's authorize URL.
func (b *browser) handOff(started Authorization) string {
	response := b.post(started.LaunchURL, consentPublicURL, map[string]string{"handoff_token": started.HandoffToken})
	b.suite.Require().Equal(http.StatusOK, response.StatusCode)
	var answered struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	b.suite.Require().NoError(json.Unmarshal([]byte(b.suite.body(response)), &answered))
	b.suite.Require().True(strings.HasPrefix(answered.AuthorizationURL, b.suite.provider.URL+fakeprovider.PathAuthorize+"?"))
	return answered.AuthorizationURL
}

// finish is the provider's redirect arriving in this browser.
func (b *browser) finish(callback *url.URL) *http.Response {
	response := b.get(callback.String())
	response.Body.Close()
	return response
}

func (b *browser) get(address string) *http.Response {
	response, err := b.client.Get(address)
	b.suite.Require().NoError(err)
	return response
}

func (b *browser) post(address, origin string, body any) *http.Response {
	encoded, err := json.Marshal(body)
	b.suite.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, address, bytes.NewReader(encoded))
	b.suite.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Origin", origin)
	response, err := b.client.Do(request)
	b.suite.Require().NoError(err)
	return response
}

type roundTripper func(*http.Request) (*http.Response, error)

func (f roundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// loopbackOrPublic lets the fake provider's loopback address through and nothing else that
// egress would refuse, as oauth2code's own suite does.
func loopbackOrPublic(ctx context.Context, raw string) error {
	if u, err := url.Parse(raw); err == nil {
		if ip, err := netip.ParseAddr(u.Hostname()); err == nil && ip.IsLoopback() && u.User == nil {
			return nil
		}
	}
	return egress.ValidatePublicHTTPSURL(ctx, raw)
}

// AuthorizationsWithoutAPublicURLSuite is a router that was never told where it is
// reachable, as ROUTER_PUBLIC_URL unset leaves it.
type AuthorizationsWithoutAPublicURLSuite struct {
	RouterSuite
}

func TestAuthorizationsWithoutAPublicURLSuite(t *testing.T) {
	runSuite(t, new(AuthorizationsWithoutAPublicURLSuite))
}

func (s *AuthorizationsWithoutAPublicURLSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: namedScheme(oauth2code.Name)}}
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *AuthorizationsWithoutAPublicURLSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *AuthorizationsWithoutAPublicURLSuite) TestAConsentIsRefusedWhenItStartsNotWhenItComesBack() {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned("linear"), &created))

	status, failure := s.serverClient.failure(http.MethodPost, "/v1/agents/connections/"+created.ID+"/authorizations", nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "ROUTER_PUBLIC_URL")
	var attempts int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_authorization_attempts WHERE connection_id = ?", created.ID).Scan(&attempts))
	s.Zero(attempts, "no attempt waits for a callback that could never arrive")
}

func (s *AuthorizationsWithoutAPublicURLSuite) TestThereIsNoClientMetadataDocument() {
	s.Equal(http.StatusNotFound, s.unauthenticatedClient.do(http.MethodGet, ConnectorClientMetadataPath, nil, nil))
}
