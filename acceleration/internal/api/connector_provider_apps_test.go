//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// providerAppPublicURL is the router's public URL in ConnectorProviderAppsSuite: https, as
// Slack's request and redirect URLs must be. Nothing dials it.
const providerAppPublicURL = "https://router.example"

// ConnectorProviderAppsSuite is the customer's Slack app the router creates, keeps and
// deletes, against one fake Slack for the whole suite, and Stream's own app set for a
// customer by staff. The built-in slack connector (client.registration [managed, operator])
// is seeded; each test has an app of its own. The operator's Slack app is in the router's
// environment, as SLACK_BOT_MCP_*.
type ConnectorProviderAppsSuite struct {
	RouterSuite
	slack *fakeprovider.Server
	// environment is the router's environment: the operator's own Slack app.
	environment map[string]string
	staff       *testClient
}

func TestConnectorProviderAppsSuite(t *testing.T) {
	runSuite(t, new(ConnectorProviderAppsSuite))
}

// SetupSuite builds oauth2_code with the router's own client lookup (ConnectorClients), so a
// consent finds the managed record this suite's PUT writes, and the operator's client in the
// environment otherwise. Its endpoints are slack.com's, which a consent's start never dials,
// so PublicEndpoint lets them through without resolving them.
func (s *ConnectorProviderAppsSuite) SetupSuite() {
	s.slack = fakeprovider.New(s.T())
	s.environment = map[string]string{
		"SLACK_BOT_MCP_APP_ID":         "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))[20:],
		"SLACK_BOT_MCP_CLIENT_ID":      "operator-client-" + s.utils.uuid(),
		"SLACK_BOT_MCP_CLIENT_SECRET":  "operator-secret-" + s.utils.uuid(),
		"SLACK_BOT_MCP_SIGNING_SECRET": "operator-signing-" + s.utils.uuid(),
	}
	getenv := func(name string) string { return s.environment[name] }
	scheme, err := oauth2code.New(oauth2code.Config{
		HTTP: s.slack.Client(),
		Clients: func(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
			return ConnectorClients(s.store, s.sealer, getenv)(ctx, ref, m, registration)
		},
		PublicEndpoint: func(context.Context, string) error { return nil },
	})
	s.Require().NoError(err)
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: scheme}}
	s.slackApps, err = slackapps.New(slackapps.Config{HTTP: s.slack.Client(), BaseURL: s.slack.URL + fakeprovider.PathSlackAPI})
	s.Require().NoError(err)
	s.operatorApps = ConnectorOperatorApps(getenv)
	s.publicURL = providerAppPublicURL
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

func (s *ConnectorProviderAppsSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.slack.Use()
}

func (s *ConnectorProviderAppsSuite) TestOnlyTheAppsBackendMayCreateTheProviderApp() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, providerAppPath("slack_bot"), s.withToken("Acme"), nil)
	})
}

func (s *ConnectorProviderAppsSuite) TestOnlyTheAppsBackendMayDeleteTheProviderApp() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		// Created by the first call, kept by the ones a caller was refused.
		s.Require().Contains([]int{http.StatusCreated, http.StatusOK},
			s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"), s.withToken("Acme"), nil))
		return as.do(http.MethodDelete, providerAppPath("slack_bot"), nil, nil)
	})
}

func (s *ConnectorProviderAppsSuite) TestConnectingSlackCreatesOneAppNamedForTheCustomer() {
	name := "Acme " + s.utils.uuid()[:8]

	created := s.create(name)

	app := s.slackApp(created.ProviderAppID)
	s.Equal(ConnectorProviderApp{
		ConnectorID: "slack_bot", Registration: ConnectorClientRegistrationMethod(core.ClientManaged),
		ProviderAppID: app.AppID, ClientID: app.ClientID, CreatedAt: created.CreatedAt, UpdatedAt: created.UpdatedAt,
	}, created)
	var manifest slackapps.Manifest
	s.Require().NoError(json.Unmarshal(app.Manifest, &manifest))
	s.Equal(name, manifest.DisplayInformation.Name)
	s.True(manifest.Settings.TokenRotationEnabled)
	s.Equal([]string{providerAppPublicURL + ConnectorCallbackPath}, manifest.OAuthConfig.RedirectURLs)
	s.Require().NotNil(manifest.Settings.EventSubscriptions, "the events URL is set once the app id is known")
	s.Equal(providerAppPublicURL+providerAppEventsPath+"slack_bot/"+app.AppID, manifest.Settings.EventSubscriptions.RequestURL)
	s.Equal([]string{"message.channels", "message.im", "tokens_revoked"}, manifest.Settings.EventSubscriptions.BotEvents,
		"slack_bot.yaml's channel.subscriptions")
	s.Equal(1, app.Updates, "created without the URL, then updated with it")

	record, signing, err := ProviderApp(context.Background(), s.store, s.sealer, "slack_bot", app.AppID)
	s.Require().NoError(err)
	s.Equal(s.customerID(), record.CustomerID, "the app's events reach this customer")
	s.Equal(app.SigningSecret, signing)
	client, found, err := s.lookup(core.ClientManaged)
	s.Require().NoError(err)
	s.True(found)
	s.Equal(oauth2code.Client{ID: app.ClientID, Secret: app.ClientSecret}, client)
}

func (s *ConnectorProviderAppsSuite) TestASecondPutReturnsTheSameAppAndCreatesNoOther() {
	created := s.create("Acme")
	before := len(s.slack.SlackApps())

	var again ConnectorProviderApp
	status := s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme renamed"}, &again)

	s.Require().Equal(http.StatusOK, status)
	s.Equal(created.ProviderAppID, again.ProviderAppID)
	s.Equal(before, len(s.slack.SlackApps()), "no second app at Slack")
	app := s.slackApp(created.ProviderAppID)
	s.Equal(2, app.Updates, "the manifest is applied again")
	s.Contains(string(app.Manifest), "Acme renamed")
}

func (s *ConnectorProviderAppsSuite) TestTwoFirstPutsAtOnceCreateOneApp() {
	// Each PUT rotates the token it sent first, slowly, so without the lock both would find
	// no app and create one.
	s.slack.Use(fakeprovider.SlowConfigRotation)
	before := len(s.slack.SlackApps())

	statuses := s.concurrently(s.withToken("Acme"), s.withToken("Acme"))

	s.ElementsMatch([]int{http.StatusCreated, http.StatusOK}, statuses)
	s.Equal(before+1, len(s.slack.SlackApps()), "one app for one customer")
}

func (s *ConnectorProviderAppsSuite) TestAnExpiredConfigTokenIsRotatedOnceUnderTwoConcurrentCalls() {
	s.create("Acme")
	s.expireConfigToken()
	s.slack.Use(fakeprovider.SlowConfigRotation)
	rotations := s.slack.ConfigTokenRotations()

	statuses := s.concurrently(ConnectorProviderAppRequest{Name: "Acme"}, ConnectorProviderAppRequest{Name: "Acme"})

	s.Equal([]int{http.StatusOK, http.StatusOK}, statuses, "the second call used the token the first rotated")
	s.Equal(rotations+1, s.slack.ConfigTokenRotations())
}

func (s *ConnectorProviderAppsSuite) TestAConfigTokenStillGoodIsNotRotated() {
	s.create("Acme")
	rotations := s.slack.ConfigTokenRotations()

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme"}, nil))

	s.Equal(rotations, s.slack.ConfigTokenRotations())
}

func (s *ConnectorProviderAppsSuite) TestTheConfigTokenIsSealedAtRestAndNeverInAnAnswer() {
	given := s.slack.NewConfigToken()

	status, created := s.serverClient.call(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme", ConfigRefreshToken: given})
	_, again := s.serverClient.call(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme"})

	s.Require().Equal(http.StatusCreated, status, string(created))
	for _, body := range [][]byte{created, again} {
		s.NotContains(string(body), given)
		s.NotContains(string(body), "xoxe", "no configuration or refresh token")
		s.NotContains(string(body), "secret")
	}
	var sealed []byte
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT tokens_sealed FROM connector_config_tokens WHERE customer_id = ? AND connector_id = 'slack'", s.customerID()).Scan(&sealed))
	s.False(bytes.Contains(sealed, []byte("xoxe")), "the tokens are sealed")
}

func (s *ConnectorProviderAppsSuite) TestTheConsentUsesTheAppsOwnClient() {
	operators := s.authorizeURL(s.connection())
	s.Equal(s.environment["SLACK_BOT_MCP_CLIENT_ID"], operators.Query().Get("client_id"), "without an app of its own, the operator's")

	created := s.create("Acme")
	own := s.authorizeURL(s.connection())

	s.Equal(created.ClientID, own.Query().Get("client_id"))
	s.Equal("https://slack.com/oauth/v2/authorize", own.Scheme+"://"+own.Host+own.Path)
}

func (s *ConnectorProviderAppsSuite) TestDeletingTheAppDeletesItAtSlackAndRemovesTheRecord() {
	created := s.create("Acme")

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, providerAppPath("slack_bot"), nil, nil))

	s.True(s.slackApp(created.ProviderAppID).Deleted)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
	_, err = s.store.ConnectorConfigToken(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorConfigToken)
	status, failure := s.serverClient.failure(http.MethodDelete, providerAppPath("slack_bot"), nil)
	s.Equal(http.StatusNotFound, status)
	s.Equal(noManagedApp, failure)
}

func (s *ConnectorProviderAppsSuite) TestAnAppAlreadyDeletedInSlackIsRemovedHereToo() {
	created := s.create("Acme")
	s.Require().NoError(s.slackApps.Delete(context.Background(), s.freshToken(), created.ProviderAppID))

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, providerAppPath("slack_bot"), nil, nil))

	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *ConnectorProviderAppsSuite) TestTheSlackToolConnectorListingOnlyTheOperatorRefusesAnAppTheRouterWouldCreate() {
	before := len(s.slack.SlackApps())

	status, failure := s.serverClient.failure(http.MethodPut, providerAppPath("slack"), s.withToken("Acme"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal("slack takes no app the router creates: its client.registration is [operator], which does not list managed", failure)
	s.Equal(before, len(s.slack.SlackApps()), "nothing created at Slack")
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *ConnectorProviderAppsSuite) TestAConnectorWhoseEventsTheAppsOwnRouteCannotVerifyGetsNoApp() {
	id := s.slackConnectorVerifiedWith("operator")
	before := len(s.slack.SlackApps())

	status, failure := s.serverClient.failure(http.MethodPut, providerAppPath(id), s.withToken("Acme"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal(id+" verifies its events with the operator secret, not the app's own (provider_app), so an app the router creates could not deliver them", failure)
	s.Equal(before, len(s.slack.SlackApps()), "nothing created at Slack")
}

func (s *ConnectorProviderAppsSuite) TestTheRequestURLIsTheRouteThatVerifiesTheAppsEvents() {
	created := s.create("Acme")
	app := s.slackApp(created.ProviderAppID)
	var manifest slackapps.Manifest
	s.Require().NoError(json.Unmarshal(app.Manifest, &manifest))
	requestURL, found := strings.CutPrefix(manifest.Settings.EventSubscriptions.RequestURL, providerAppPublicURL)
	s.Require().True(found)
	challenge := "challenge-" + s.utils.uuid()

	status, answer := s.slack.Deliver(s.server.URL+requestURL, app.SigningSecret,
		[]byte(`{"type":"url_verification","challenge":"`+challenge+`"}`), 0)

	s.Equal(http.StatusOK, status, answer)
	s.Equal(challenge, answer, "Slack's handshake on the URL it was given is answered")
	status, _ = s.slack.Deliver(s.server.URL+requestURL, "another-apps-secret",
		[]byte(`{"type":"url_verification","challenge":"x"}`), 0)
	s.Equal(http.StatusUnauthorized, status, "verified with this app's own signing secret")
}

func (s *ConnectorProviderAppsSuite) TestAConnectorThatDoesNotAuthorizeAtSlackGetsNoApp() {
	status, failure := s.serverClient.failure(http.MethodPut, providerAppPath("github"), s.withToken("Acme"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal("github does not authorize at Slack, so the router creates no app for it", failure)
}

func (s *ConnectorProviderAppsSuite) TestTheFirstPutNeedsAConfigToken() {
	status, failure := s.serverClient.failure(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "send config_refresh_token")
}

func (s *ConnectorProviderAppsSuite) TestARefreshTokenSlackRefusesCreatesNothing() {
	before := len(s.slack.SlackApps())

	status, failure := s.serverClient.failure(http.MethodPut, providerAppPath("slack_bot"), ConnectorProviderAppRequest{Name: "Acme", ConfigRefreshToken: "xoxe-not-one-slack-issued"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "Slack refused config_refresh_token")
	s.Equal(before, len(s.slack.SlackApps()))
}

func (s *ConnectorProviderAppsSuite) TestMoreThanTenIPRangesAreRefused() {
	sent := s.withToken("Acme")
	for i := range 11 {
		sent.AllowedIPAddressRanges = append(sent.AllowedIPAddressRanges, fmt.Sprintf("203.0.113.%d", i))
	}

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"), sent, nil))
}

func (s *ConnectorProviderAppsSuite) TestStaffMakeStreamsOwnAppOneCustomersProviderApp() {
	var set ConnectorProviderApp
	s.Require().Equal(http.StatusCreated, s.staff.do(http.MethodPut, operatorAppPath(s.customerID()), nil, &set))
	defer func() {
		s.Equal(http.StatusNoContent, s.staff.do(http.MethodDelete, operatorAppPath(s.customerID()), nil, nil))
	}()

	s.Equal(ConnectorClientRegistrationMethod(core.ClientOperator), set.Registration)
	s.Equal(s.environment["SLACK_BOT_MCP_APP_ID"], set.ProviderAppID)
	record, signing, err := ProviderApp(context.Background(), s.store, s.sealer, "slack_bot", set.ProviderAppID)
	s.Require().NoError(err)
	s.Equal(s.customerID(), record.CustomerID)
	s.Equal(s.environment["SLACK_BOT_MCP_SIGNING_SECRET"], signing)
	client, found, err := s.lookup(core.ClientOperator)
	s.Require().NoError(err)
	s.True(found)
	s.Equal(s.environment["SLACK_BOT_MCP_CLIENT_SECRET"], client.Secret)

	other := s.data.createApp()
	s.Equal(http.StatusConflict, s.staff.do(http.MethodPut, operatorAppPath(other.app.ID), nil, nil), "one customer per app")
}

func (s *ConnectorProviderAppsSuite) TestOnlyStaffMaySetStreamsOwnApp() {
	s.Equal(http.StatusUnauthorized, s.serverClient.do(http.MethodPut, operatorAppPath(s.customerID()), nil, nil))
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// create is the app's backend creating its Slack app with a fresh configuration token.
func (s *ConnectorProviderAppsSuite) create(name string) ConnectorProviderApp {
	var created ConnectorProviderApp
	status, payload := s.serverClient.call(http.MethodPut, providerAppPath("slack_bot"), s.withToken(name))
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &created))
	return created
}

// withToken is a request naming the app name with the refresh token of a configuration token
// the fake Slack's admin just generated.
func (s *ConnectorProviderAppsSuite) withToken(name string) ConnectorProviderAppRequest {
	return ConnectorProviderAppRequest{Name: name, ConfigRefreshToken: s.slack.NewConfigToken()}
}

// freshToken is a configuration token of the test's own, for acting at the fake Slack as
// somebody else in the workspace would.
func (s *ConnectorProviderAppsSuite) freshToken() string {
	rotated, err := s.slackApps.Rotate(context.Background(), s.slack.NewConfigToken())
	s.Require().NoError(err)
	return rotated.Token
}

// slackApp is the fake Slack's app with id.
func (s *ConnectorProviderAppsSuite) slackApp(id string) fakeprovider.SlackApp {
	for _, app := range s.slack.SlackApps() {
		if app.AppID == id {
			return app
		}
	}
	s.FailNow("the fake Slack has no app " + id)
	return fakeprovider.SlackApp{}
}

// expireConfigToken makes the configuration token the router keeps for the test's app one
// that has expired, here and at the fake Slack.
func (s *ConnectorProviderAppsSuite) expireConfigToken() {
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connector_config_tokens SET expires_at = now() - interval '1 minute' WHERE customer_id = ?", s.customerID())
	s.Require().NoError(err)
	s.slack.Advance(fakeprovider.ConfigTokenTTL)
}

// concurrently sends the two PUTs at once, each on a connection of its own, and returns their
// statuses in the order they were given.
func (s *ConnectorProviderAppsSuite) concurrently(first, second ConnectorProviderAppRequest) []int {
	statuses := make([]int, 2)
	errs := make([]error, 2)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i, sent := range []ConnectorProviderAppRequest{first, second} {
		wg.Go(func() {
			<-start
			statuses[i], errs[i] = s.put(sent)
		})
	}
	close(start)
	wg.Wait()
	for _, err := range errs {
		s.Require().NoError(err)
	}
	return statuses
}

// put is the backend's PUT without the suite's assertions, which may not run off the test
// goroutine.
func (s *ConnectorProviderAppsSuite) put(sent ConnectorProviderAppRequest) (int, error) {
	encoded, err := json.Marshal(sent)
	if err != nil {
		return 0, err
	}
	request, err := http.NewRequest(http.MethodPut, s.server.URL+providerAppPath("slack_bot"), bytes.NewReader(encoded))
	if err != nil {
		return 0, err
	}
	request.Header = s.serverClient.header.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := s.server.Client().Do(request)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	_, err = io.Copy(io.Discard, response.Body)
	return response.StatusCode, err
}

// connection is a new app-owned connection of the test's app to the built-in slack.
func (s *ConnectorProviderAppsSuite) connection() string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned("slack_bot"), &created))
	return created.ID
}

// authorizeURL starts a consent for the connection and hands it off as the launch page does,
// and returns the Slack authorize URL the browser would be sent to.
func (s *ConnectorProviderAppsSuite) authorizeURL(connection string) *url.URL {
	var started Authorization
	status, payload := s.serverClient.call(http.MethodPost, "/v1/agents/connections/"+connection+"/authorizations", nil)
	s.Require().Equal(http.StatusCreated, status, string(payload))
	s.Require().NoError(json.Unmarshal(payload, &started))
	encoded, err := json.Marshal(map[string]string{"handoff_token": started.HandoffToken})
	s.Require().NoError(err)
	request, err := http.NewRequest(http.MethodPost, s.server.URL+connectorLaunchPath+started.ID, bytes.NewReader(encoded))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Origin", providerAppPublicURL)
	response, err := s.server.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusOK, response.StatusCode)
	var answered struct {
		AuthorizationURL string `json:"authorization_url"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&answered))
	parsed, err := url.Parse(answered.AuthorizationURL)
	s.Require().NoError(err)
	return parsed
}

// lookup is the client oauth2_code finds for the test's app's slack connection.
func (s *ConnectorProviderAppsSuite) lookup(registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
	return ConnectorClients(s.store, s.sealer, func(string) string { return "" })(context.Background(),
		core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: s.utils.uuid()},
		core.ResolvedManifest{ConnectorID: "slack_bot"}, registration)
}

// slackConnectorVerifiedWith stores a custom bot connector of the test's app that lists
// managed and verifies its events with secret, in the shape of providers/slack_bot.yaml.
func (s *ConnectorProviderAppsSuite) slackConnectorVerifiedWith(secret string) string {
	id := "custom_slack" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Slack, verified with another secret
endpoints:
  authorize: https://slack.com/oauth/v2/authorize
  token: https://slack.com/api/oauth.v2.access
schemes: [oauth2_code]
client:
  registration: [managed, operator]
  env: SLACK_BOT
scopes:
  list: [chat:write]
capture:
  - name: team_id
    from: token_response
    path: $.team.id
identity: [team_id]
channel:
  verifier:
    kind: secret_header
    secret: ` + secret + `
    header: X-Secret
  format: json
  signals:
    - kind: uninstalled
      match:
        $.event.type: app_uninstalled
      identity:
        team_id: $.team_id
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

func providerAppPath(connector string) string {
	return "/v1/agents/connectors/" + connector + "/provider-app"
}

func operatorAppPath(customer string) string {
	return "/v1/ops/customers/" + customer + "/connectors/slack_bot/provider-app"
}
