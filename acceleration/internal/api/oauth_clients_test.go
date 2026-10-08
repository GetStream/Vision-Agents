//go:build integration

package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// OAuthClientsSuite is an app's own OAuth client for a connector: put, replaced, removed, and
// found by the lookup oauth2_code is started with. The built-ins are seeded, so github
// (client.registration [operator, customer]) takes an app's own client and slack ([operator])
// does not. Each test has an app of its own.
type OAuthClientsSuite struct {
	RouterSuite

	// logged is what the router logged, for a test that no secret is in it.
	logged *lockedLog
}

func TestOAuthClientsSuite(t *testing.T) {
	runSuite(t, new(OAuthClientsSuite))
}

func (s *OAuthClientsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: namedScheme(oauth2code.Name)}}
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *OAuthClientsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *OAuthClientsSuite) TestOnlyTheAppsBackendMaySetAnOAuthClient() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPut, oauthClientPath("github"), confidentialClient("secret-"+s.utils.uuid()), nil)
	})
}

func (s *OAuthClientsSuite) TestOnlyTheAppsBackendMayRemoveAnOAuthClient() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		s.put("github", confidentialClient("secret-"+s.utils.uuid()))
		return as.do(http.MethodDelete, oauthClientPath("github"), nil, nil)
	})
}

func (s *OAuthClientsSuite) TestTheFirstPutCreatesTheClientAndTheNextReplacesItsOneRow() {
	first := confidentialClient("first-" + s.utils.uuid())
	second := ConnectorOAuthClientRequest{ClientID: "rotated-id", ClientSecret: "second-" + s.utils.uuid(), AuthMethod: "client_secret_post"}

	var created, replaced ConnectorOAuthClient
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("github"), first, &created))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, oauthClientPath("github"), second, &replaced))

	s.Equal(ConnectorOAuthClient{
		ConnectorID: "github", Registration: ConnectorClientRegistrationMethod(core.ClientCustomer),
		ClientID: "rotated-id", AuthMethod: "client_secret_post",
		CreatedAt: created.CreatedAt, UpdatedAt: replaced.UpdatedAt,
	}, replaced)
	s.True(replaced.UpdatedAt.After(created.UpdatedAt))
	var rows int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_oauth_clients WHERE customer_id = ?", s.customerID()).Scan(&rows))
	s.Equal(1, rows)
}

func (s *OAuthClientsSuite) TestTheSecretIsNeverInAnAnswer() {
	secret := "never-shown-" + s.utils.uuid()

	created, createdBody := s.serverClient.call(http.MethodPut, oauthClientPath("github"), confidentialClient(secret))
	replaced, replacedBody := s.serverClient.call(http.MethodPut, oauthClientPath("github"), confidentialClient(secret))

	s.Equal(http.StatusCreated, created)
	s.Equal(http.StatusOK, replaced)
	for _, body := range []string{string(createdBody), string(replacedBody)} {
		s.NotContains(body, secret)
		s.NotContains(body, "client_secret")
	}
}

// A put without provider_app_id and signing_secret answers with exactly the fields it did
// before them, and stores no app and no signing secret: the control for AI-906.
func (s *OAuthClientsSuite) TestAPutWithoutAProviderAppAnswersAndStoresAsBefore() {
	status, body := s.serverClient.call(http.MethodPut, oauthClientPath("github"), confidentialClient("secret-"+s.utils.uuid()))

	s.Require().Equal(http.StatusCreated, status)
	var fields map[string]any
	s.Require().NoError(json.Unmarshal(body, &fields))
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	// duration is what the server adds to every answered object (server.go, timedResponse).
	s.ElementsMatch([]string{"connector_id", "registration", "client_id", "created_at", "updated_at", "duration"}, keys)
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "github")
	s.Require().NoError(err)
	s.Empty(record.ProviderAppID)
	s.Empty(record.SigningSecretSealed)
	s.Zero(record.SigningKEKVersion)
}

// AI-906's acceptance: a put with both fields is the provider app the events route opens.
func (s *OAuthClientsSuite) TestAPutWithAProviderAppAndItsSigningSecretIsTheAppTheEventsRouteOpens() {
	app, signing := s.slackAppID(), "signing-"+s.utils.uuid()

	var answered ConnectorOAuthClient
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("slack_bot"),
		slackApp(app, signing), &answered))

	s.Equal(app, answered.ProviderAppID)
	record, secret, err := ProviderApp(context.Background(), s.store, s.sealer, "slack_bot", app)
	s.Require().NoError(err)
	s.Equal(signing, secret)
	s.Equal(s.customerID(), record.CustomerID)
	s.Equal(core.ClientCustomer, record.Registration)
	s.NotContains(string(record.SigningSecretSealed), signing, "sealed at rest")
	found, ok, err := ConnectorClients(s.store, s.sealer, func(string) string { return "" })(context.Background(),
		core.ConnectionRef{CustomerID: s.customerID()}, core.ResolvedManifest{ConnectorID: "slack_bot"}, core.ClientCustomer)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "the-apps-client", Secret: "client-secret"}, found,
		"the client secret is sealed apart from the signing secret")
}

func (s *OAuthClientsSuite) TestTheSigningSecretIsNeverInAnAnswerNorInTheLog() {
	app, signing := s.slackAppID(), "never-shown-signing-"+s.utils.uuid()
	sent := slackApp(app, signing)

	created, createdBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), sent)
	replaced, replacedBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), sent)
	refusedPattern := sent
	refusedPattern.SigningSecret = signing + "\n"
	refused, refusedBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), refusedPattern)
	noApp := sent
	noApp.ProviderAppID = ""
	unnamed, unnamedBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), noApp)
	// An unknown field is refused with the whole object as the value it names.
	unknown, unknownBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), map[string]string{
		"client_id": sent.ClientID, "provider_app_id": app, "signing_secret": signing, "unknown": "field",
	})
	s.useApp(s.data.createApp())
	taken, takenBody := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), sent)

	s.Equal(http.StatusCreated, created)
	s.Equal(http.StatusOK, replaced)
	s.Equal(http.StatusBadRequest, refused)
	s.Equal(http.StatusBadRequest, unnamed)
	s.Equal(http.StatusBadRequest, unknown)
	s.Equal(http.StatusConflict, taken)
	for _, body := range []string{string(createdBody), string(replacedBody), string(refusedBody), string(unnamedBody), string(unknownBody), string(takenBody)} {
		s.NotContains(body, signing)
	}
	for _, body := range []string{string(createdBody), string(replacedBody)} {
		s.NotContains(body, "signing_secret")
	}
	s.NotContains(s.logged.String(), signing)
}

// One customer per app: the second customer naming it is a 409, and the app stays the first's.
func (s *OAuthClientsSuite) TestASecondCustomerClaimingTheSameAppIsAConflict() {
	app := s.slackAppID()
	s.put("slack_bot", slackApp(app, "first-signing"))
	first := s.customerID()
	s.useApp(s.data.createApp())

	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("slack_bot"), slackApp(app, "second-signing"))

	s.Equal(http.StatusConflict, status)
	s.Equal(errProviderAppTaken.Message, failure)
	record, secret, err := ProviderApp(context.Background(), s.store, s.sealer, "slack_bot", app)
	s.Require().NoError(err)
	s.Equal(first, record.CustomerID)
	s.Equal("first-signing", secret)
	_, err = s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient, "nothing stored for the second customer")
}

// A put replaces the record whole: put again without the signing secret, the app takes no
// events, as a put without client_secret drops that one.
func (s *OAuthClientsSuite) TestPuttingTheClientAgainWithoutTheSigningSecretRemovesIt() {
	app := s.slackAppID()
	s.put("slack_bot", slackApp(app, "signing-"+s.utils.uuid()))

	var answered ConnectorOAuthClient
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, oauthClientPath("slack_bot"), slackApp(app, ""), &answered))

	s.Equal(app, answered.ProviderAppID)
	_, _, err := ProviderApp(context.Background(), s.store, s.sealer, "slack_bot", app)
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *OAuthClientsSuite) TestASigningSecretWithoutAProviderAppIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("slack_bot"), slackApp("", "signing"))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "signing_secret needs provider_app_id")
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// github has no channel, so no route would ever read a signing secret put for it.
func (s *OAuthClientsSuite) TestASigningSecretForAConnectorThatReadsNoAppsEventsIsRefused() {
	sent := confidentialClient("secret")
	sent.ProviderAppID, sent.SigningSecret = s.slackAppID(), "signing"

	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("github"), sent)

	s.Equal(http.StatusBadRequest, status)
	s.Equal("github does not verify its events with the app's own secret (channel.verifier.secret provider_app), so nothing would read signing_secret: leave it out", failure)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "github")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// AI-863: the customer's Linq account is a provider app with no OAuth client. Its webhook
// subscription's signing secret is put alone, and the events route opens it.
func (s *OAuthClientsSuite) TestALinqAccountIsAProviderAppWithoutAClient() {
	app, signing := "line-"+s.utils.uuid(), "whsec_"+s.utils.uuid()

	var answered ConnectorOAuthClient
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("linq"),
		ConnectorOAuthClientRequest{ProviderAppID: app, SigningSecret: signing}, &answered))

	s.Empty(answered.ClientID)
	s.Equal(app, answered.ProviderAppID)
	record, secret, err := ProviderApp(context.Background(), s.store, s.sealer, "linq", app)
	s.Require().NoError(err)
	s.Equal(signing, secret)
	s.Equal(s.customerID(), record.CustomerID)
}

// oauth2_code reads the client at every consent, so slack_bot's provider app is still a client.
func (s *OAuthClientsSuite) TestAConnectorConsentedThroughOAuth2CodeNeedsAClientID() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("slack_bot"),
		ConnectorOAuthClientRequest{ProviderAppID: s.slackAppID(), SigningSecret: "signing"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "client_id is required")
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// A record without a client is a provider app, which only its signing secret makes one.
func (s *OAuthClientsSuite) TestAPutWithNeitherAClientIDNorASigningSecretIsRefused() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("linq"),
		ConnectorOAuthClientRequest{ProviderAppID: "line-" + s.utils.uuid()})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "client_id is required")
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "linq")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// A client secret or a method belongs to a client, so the provider app alone takes neither.
func (s *OAuthClientsSuite) TestAProviderAppWithoutAClientTakesNoClientSecretOrMethod() {
	for _, sent := range []ConnectorOAuthClientRequest{
		{ProviderAppID: "line-" + s.utils.uuid(), SigningSecret: "signing", ClientSecret: "secret"},
		{ProviderAppID: "line-" + s.utils.uuid(), SigningSecret: "signing", AuthMethod: ConnectorOAuthClientAuthMethod(core.AuthNone)},
	} {
		status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("linq"), sent)

		s.Equal(http.StatusBadRequest, status)
		s.Contains(failure, "client_id is required")
	}
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "linq")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *OAuthClientsSuite) TestAProviderAppThatIsADotSegmentIsRefused() {
	for _, app := range []string{".", ".."} {
		status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("slack_bot"), slackApp(app, "signing"))

		s.Equal(http.StatusBadRequest, status, app)
		s.Contains(failure, "dot segment", app)
	}
}

func (s *OAuthClientsSuite) TestAProviderAppOutsideUnreservedCharactersIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPut, oauthClientPath("slack_bot"), slackApp("A012/ABCD", "signing"))

	s.Equal(http.StatusBadRequest, status)
}

func (s *OAuthClientsSuite) TestAConnectorTakingOnlyTheOperatorsClientRefusesTheAppsOwn() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("slack"), confidentialClient("secret"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal("slack does not take an app's own OAuth client: its client.registration is [operator], which does not list customer", failure)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *OAuthClientsSuite) TestAPublicClientTakesNoSecret() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("github"),
		ConnectorOAuthClientRequest{ClientID: "public", ClientSecret: "a-secret", AuthMethod: "none"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, "auth_method none is a public client")
}

func (s *OAuthClientsSuite) TestAPublicClientIsStoredWithoutASecret() {
	s.Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, oauthClientPath("github"),
		ConnectorOAuthClientRequest{ClientID: "public", AuthMethod: "none"}, nil))

	found, ok, err := s.lookup("github", core.ClientCustomer)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "public", AuthMethod: core.AuthNone}, found)
}

func (s *OAuthClientsSuite) TestASecretMethodNeedsASecret() {
	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath("github"),
		ConnectorOAuthClientRequest{ClientID: "confidential", AuthMethod: "client_secret_basic"})

	s.Equal(http.StatusBadRequest, status)
	s.Equal("auth_method client_secret_basic needs client_secret", failure)
}

func (s *OAuthClientsSuite) TestAMethodNeedingAKeyIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPut, oauthClientPath("github"),
		map[string]string{"client_id": "keyed", "auth_method": "private_key_jwt"})

	s.Equal(http.StatusBadRequest, status)
}

func (s *OAuthClientsSuite) TestAConnectorWhoseMethodNeedsAKeyRefusesAClientThatLeavesTheMethodOut() {
	id := s.customConnector("  registration: [customer]\n  auth_method: private_key_jwt\n  alg: PS256")

	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath(id), confidentialClient("secret"))

	s.Equal(http.StatusBadRequest, status)
	s.Equal("the connector's client.auth_method private_key_jwt needs a key or a certificate this client cannot hold: "+
		"set auth_method to none, client_secret_basic or client_secret_post", failure)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), id)
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

func (s *OAuthClientsSuite) TestAClientIDOutsidePrintableASCIIIsRefused() {
	status, _ := s.serverClient.failure(http.MethodPut, oauthClientPath("github"),
		ConnectorOAuthClientRequest{ClientID: "tab\there", ClientSecret: "secret"})

	s.Equal(http.StatusBadRequest, status)
}

func (s *OAuthClientsSuite) TestAnUnknownConnectorIsNotFound() {
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodPut, oauthClientPath("custom_nothing_"+strings.ReplaceAll(s.utils.uuid(), "-", "")),
		confidentialClient("secret"), nil))
}

func (s *OAuthClientsSuite) TestAnotherAppCannotSetAClientForTheAppsCustomConnector() {
	id := s.customConnector("  registration: [customer]")

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPut, oauthClientPath(id), confidentialClient("secret"), nil)
	})
}

func (s *OAuthClientsSuite) TestAnotherAppCannotRemoveTheAppsClient() {
	s.put("github", confidentialClient("secret"))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodDelete, oauthClientPath("github"), nil, nil)
	})
	_, ok, err := s.lookup("github", core.ClientCustomer)
	s.Require().NoError(err)
	s.True(ok, "still the app's")
}

func (s *OAuthClientsSuite) TestRemovingTheClientForgetsIt() {
	s.put("github", confidentialClient("secret"))

	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, oauthClientPath("github"), nil, nil))

	_, ok, err := s.lookup("github", core.ClientCustomer)
	s.Require().NoError(err)
	s.False(ok)
	status, failure := s.serverClient.failure(http.MethodDelete, oauthClientPath("github"), nil)
	s.Equal(http.StatusNotFound, status)
	s.Equal(noOAuthClient, failure)
}

func (s *OAuthClientsSuite) TestTheOperatorsClientForTheAppCannotBeReplacedThroughTheAPI() {
	_, err := s.store.PutConnectorOAuthClient(context.Background(), &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "github", Registration: core.ClientOperator, ClientID: "operators",
	})
	s.Require().NoError(err)

	s.Equal(http.StatusConflict, s.serverClient.do(http.MethodPut, oauthClientPath("github"), confidentialClient("secret"), nil))
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodDelete, oauthClientPath("github"), nil, nil))
	found, ok, err := s.lookup("github", core.ClientOperator)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal("operators", found.ID)
}

func (s *OAuthClientsSuite) TestTheLookupFindsTheAppsOwnClientWithTheSecretPutLast() {
	s.put("github", ConnectorOAuthClientRequest{ClientID: "the-apps", ClientSecret: "first", AuthMethod: "client_secret_post"})
	s.put("github", ConnectorOAuthClientRequest{ClientID: "the-apps", ClientSecret: "rotated", AuthMethod: "client_secret_post"})

	found, ok, err := s.lookup("github", core.ClientCustomer)

	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "the-apps", Secret: "rotated", AuthMethod: core.AuthClientSecretPost}, found)
}

func (s *OAuthClientsSuite) TestTheAppsOwnClientNeverStandsInForTheOperators() {
	s.put("github", confidentialClient("the-apps-secret"))
	environment := map[string]string{"GITHUB_MCP_CLIENT_ID": "operators", "GITHUB_MCP_CLIENT_SECRET": "operators-secret"}
	lookup := ConnectorClients(s.store, s.sealer, func(name string) string { return environment[name] })

	found, ok, err := lookup(context.Background(), core.ConnectionRef{CustomerID: s.customerID()}, githubManifest(), core.ClientOperator)

	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "operators", Secret: "operators-secret"}, found, "the environment's, read by client.env GITHUB")
	_, ok, err = s.lookup("github", core.ClientOperator)
	s.Require().NoError(err)
	s.False(ok, "no operator client in an empty environment")
}

func (s *OAuthClientsSuite) TestASecretMovedToAnotherConnectorDoesNotOpen() {
	s.put("github", confidentialClient("github-secret"))
	s.put("gong", confidentialClient("gong-secret"))

	s.moveSecret(s.customerID(), "github", s.customerID(), "gong")

	_, _, err := s.lookup("gong", core.ClientCustomer)
	s.ErrorContains(err, "does not open")
}

func (s *OAuthClientsSuite) TestASecretMovedToAnotherAppDoesNotOpen() {
	s.put("github", confidentialClient("first-apps-secret"))
	first := s.customerID()
	s.useApp(s.data.createApp())
	s.put("github", confidentialClient("second-apps-secret"))

	s.moveSecret(first, "github", s.customerID(), "github")

	_, _, err := s.lookup("github", core.ClientCustomer)
	s.ErrorContains(err, "does not open")
}

func (s *OAuthClientsSuite) TestTheLookupFindsTheClientTheRouterCreatedForTheApp() {
	id := s.customConnector("  registration: [managed, customer]")
	s.providerApp(id, core.ClientManaged, "the-routers-secret", "")

	found, ok, err := s.lookup(id, core.ClientManaged)

	s.Require().NoError(err)
	s.True(ok)
	s.Equal(oauth2code.Client{ID: "created-by-the-router", Secret: "the-routers-secret", AuthMethod: core.AuthClientSecretPost}, found)
	_, ok, err = s.lookup(id, core.ClientCustomer)
	s.Require().NoError(err)
	s.False(ok, "the router's app is not the app's own")
}

func (s *OAuthClientsSuite) TestTheClientTheRouterCreatedCannotBeReplacedThroughTheAPI() {
	id := s.customConnector("  registration: [managed, customer]")
	s.providerApp(id, core.ClientManaged, "the-routers-secret", "")

	status, failure := s.serverClient.failure(http.MethodPut, oauthClientPath(id), confidentialClient("secret"))

	s.Equal(http.StatusConflict, status)
	s.Contains(failure, "the router created")
	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodDelete, oauthClientPath(id), nil, nil))
	found, ok, err := s.lookup(id, core.ClientManaged)
	s.Require().NoError(err)
	s.True(ok)
	s.Equal("created-by-the-router", found.ID)
}

func (s *OAuthClientsSuite) TestASigningSecretIsOpenedOnlyForTheProviderAppItSignsFor() {
	first := s.providerApp("github", core.ClientCustomer, "first-client-secret", "first-signing-secret")
	s.useApp(s.data.createApp())
	second := s.providerApp("github", core.ClientCustomer, "second-client-secret", "second-signing-secret")

	record, secret, err := ProviderApp(context.Background(), s.store, s.sealer, "github", first.ProviderAppID)
	s.Require().NoError(err)
	s.Equal(first.CustomerID, record.CustomerID)
	s.Equal("first-signing-secret", secret)

	record, secret, err = ProviderApp(context.Background(), s.store, s.sealer, "github", second.ProviderAppID)
	s.Require().NoError(err)
	s.Equal(second.CustomerID, record.CustomerID)
	s.Equal("second-signing-secret", secret)

	_, _, err = ProviderApp(context.Background(), s.store, s.sealer, "github", "A"+strings.ReplaceAll(s.utils.uuid(), "-", ""))
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
	_, _, err = ProviderApp(context.Background(), s.store, s.sealer, "gong", first.ProviderAppID)
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient, "the app is github's")
}

func (s *OAuthClientsSuite) TestASigningSecretMovedToAnotherProviderAppDoesNotOpen() {
	first := s.providerApp("github", core.ClientCustomer, "first-client-secret", "first-signing-secret")
	s.useApp(s.data.createApp())
	second := s.providerApp("github", core.ClientCustomer, "second-client-secret", "second-signing-secret")

	_, err := s.store.DB().ExecContext(context.Background(), `
UPDATE connector_oauth_clients AS target
SET signing_secret_sealed = source.signing_secret_sealed, signing_kek_version = source.signing_kek_version
FROM connector_oauth_clients AS source
WHERE source.connector_id = 'github' AND source.provider_app_id = ?
  AND target.connector_id = 'github' AND target.provider_app_id = ?`, first.ProviderAppID, second.ProviderAppID)
	s.Require().NoError(err)

	_, _, err = ProviderApp(context.Background(), s.store, s.sealer, "github", second.ProviderAppID)
	s.ErrorContains(err, "does not open")
}

func (s *OAuthClientsSuite) TestAProviderAppWithoutASigningSecretIsNotFound() {
	app := s.providerApp("github", core.ClientCustomer, "client-secret", "")

	_, secret, err := ProviderApp(context.Background(), s.store, s.sealer, "github", app.ProviderAppID)

	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
	s.Empty(secret, "no event verifies under an empty key")
}

func (s *OAuthClientsSuite) TestASigningSecretSealedUnderAnOlderKeyIsSealedAgainOnUse() {
	app := s.providerApp("github", core.ClientCustomer, "client-secret", "signing-secret")
	rotated := s.rotatedKeyring()

	_, secret, err := ProviderApp(context.Background(), s.store, rotated, "github", app.ProviderAppID)

	s.Require().NoError(err)
	s.Equal("signing-secret", secret)
	found, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "github")
	s.Require().NoError(err)
	s.Equal(2, found.SigningKEKVersion)
	_, secret, err = ProviderApp(context.Background(), s.store, s.onlyTheNewKey(), "github", app.ProviderAppID)
	s.Require().NoError(err, "the old key can go")
	s.Equal("signing-secret", secret)
}

func (s *OAuthClientsSuite) TestAClientSecretSealedUnderAnOlderKeyIsSealedAgainOnUse() {
	id := s.customConnector("  registration: [managed]")
	s.providerApp(id, core.ClientManaged, "the-routers-secret", "")
	lookup := func(secrets *auth.Sealer) (oauth2code.Client, bool, error) {
		return ConnectorClients(s.store, secrets, func(string) string { return "" })(context.Background(),
			core.ConnectionRef{CustomerID: s.customerID()}, core.ResolvedManifest{ConnectorID: id}, core.ClientManaged)
	}

	found, ok, err := lookup(s.rotatedKeyring())

	s.Require().NoError(err)
	s.True(ok)
	s.Equal("the-routers-secret", found.Secret)
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), id)
	s.Require().NoError(err)
	s.Equal(2, record.KEKVersion)
	found, _, err = lookup(s.onlyTheNewKey())
	s.Require().NoError(err, "the old key can go")
	s.Equal("the-routers-secret", found.Secret)
}

// rotatedKeyring is the suite's key as version 1 and a new current version 2, as a deployment
// holds them between adding a key and removing the old one.
func (s *OAuthClientsSuite) rotatedKeyring() *auth.Sealer {
	keyring, err := auth.NewSealerWithKeyring(2, map[int]string{1: suiteKEK, 2: suiteKEK + " v2"})
	s.Require().NoError(err)
	return keyring
}

// onlyTheNewKey is the keyring once version 1 is removed.
func (s *OAuthClientsSuite) onlyTheNewKey() *auth.Sealer {
	keyring, err := auth.NewSealerWithKeyring(2, map[int]string{2: suiteKEK + " v2"})
	s.Require().NoError(err)
	return keyring
}

// providerApp stores the test app's record for connector as the router would write one it
// created or was handed (T54): with a provider app id of its own and both secrets sealed, the
// signing secret only when one is given. Written to the store directly, since the API writes
// only a customer record.
func (s *OAuthClientsSuite) providerApp(connector string, registration core.ClientRegistrationMethod, clientSecret, signingSecret string) store.ConnectorOAuthClient {
	record := &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: connector, Registration: registration,
		ClientID: "created-by-the-router", AuthMethod: core.AuthClientSecretPost,
		ProviderAppID: "A" + strings.ReplaceAll(s.utils.uuid(), "-", ""),
	}
	var err error
	record.SecretSealed, err = s.sealer.SealWithAAD(clientSecret, oauthClientAAD(record.CustomerID, connector))
	s.Require().NoError(err)
	record.KEKVersion = s.sealer.CurrentVersion()
	if signingSecret != "" {
		record.SigningSecretSealed, err = s.sealer.SealWithAAD(signingSecret, providerAppAAD(record.CustomerID, connector, record.ProviderAppID))
		s.Require().NoError(err)
		record.SigningKEKVersion = s.sealer.CurrentVersion()
	}
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	return *record
}

// put is the app's backend putting its own client for connector.
func (s *OAuthClientsSuite) put(connector string, client ConnectorOAuthClientRequest) {
	status, payload := s.serverClient.call(http.MethodPut, oauthClientPath(connector), client)
	s.Require().Contains([]int{http.StatusCreated, http.StatusOK}, status, string(payload))
}

// lookup is what oauth2_code finds for the app's connection to connector, with no operator
// variables set.
func (s *OAuthClientsSuite) lookup(connector string, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
	manifest := core.ResolvedManifest{ConnectorID: connector}
	if connector == "github" {
		manifest = githubManifest()
	}
	return ConnectorClients(s.store, s.sealer, func(string) string { return "" })(context.Background(),
		core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: s.utils.uuid()}, manifest, registration)
}

// moveSecret copies one row's sealed secret onto another, as somebody with write access to
// the table could, which the secret's AAD is there to defeat.
func (s *OAuthClientsSuite) moveSecret(fromCustomer, fromConnector, toCustomer, toConnector string) {
	_, err := s.store.DB().ExecContext(context.Background(), `
UPDATE connector_oauth_clients AS target
SET secret_sealed = source.secret_sealed, kek_version = source.kek_version
FROM connector_oauth_clients AS source
WHERE source.customer_id = ? AND source.connector_id = ?
  AND target.customer_id = ? AND target.connector_id = ?`, fromCustomer, fromConnector, toCustomer, toConnector)
	s.Require().NoError(err)
}

// customConnector stores a custom connector of the test's app with client, the lines of its
// client block. Stored directly, since the API resolves a custom connector's endpoint.
func (s *OAuthClientsSuite) customConnector(client string) string {
	id := "custom_byo" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Bring your own
endpoints:
  mcp: https://mcp.example/mcp
schemes: [oauth2_code]
client:
` + client + `
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// githubManifest is the part of the built-in github manifest the lookup reads: its id and the
// prefix of the operator's variables (providers/github.yaml, client.env).
func githubManifest() core.ResolvedManifest {
	return core.ResolvedManifest{ConnectorID: "github", Client: core.ClientPolicy{Env: "GITHUB"}}
}

func oauthClientPath(connector string) string {
	return "/v1/agents/connectors/" + connector + "/oauth-client"
}

// slackAppID is a fresh app id starting with A, as Slack's do (A012ABCD0A0 in
// https://docs.slack.dev/reference/methods/apps.manifest.create), and as long as a UUID so no
// two tests share one.
func (s *OAuthClientsSuite) slackAppID() string {
	return "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))
}

// slackApp is the app's own Slack app for slack_bot: its client, app id and signing secret.
func slackApp(appID, signingSecret string) ConnectorOAuthClientRequest {
	return ConnectorOAuthClientRequest{ClientID: "the-apps-client", ClientSecret: "client-secret",
		ProviderAppID: appID, SigningSecret: signingSecret}
}

func confidentialClient(secret string) ConnectorOAuthClientRequest {
	return ConnectorOAuthClientRequest{ClientID: "the-apps-client", ClientSecret: secret}
}

// OAuthClientsOffSuite is connectors off, the control: the router has no connector keyring, as
// cmd/router gives it none with ROUTER_CONNECTORS_ENABLED unset, so the put answers as on base
// whatever it carries, and the provider app's events route takes nothing.
type OAuthClientsOffSuite struct {
	RouterSuite
}

func TestOAuthClientsOffSuite(t *testing.T) {
	runSuite(t, new(OAuthClientsOffSuite))
}

func (s *OAuthClientsOffSuite) SetupSuite() {
	s.connectorsOff = true
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *OAuthClientsOffSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *OAuthClientsOffSuite) TestThePutAnswersAsBeforeAndStoresNothing() {
	app := "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))
	for _, sent := range []ConnectorOAuthClientRequest{confidentialClient("secret"), slackApp(app, "signing")} {
		status, body := s.serverClient.call(http.MethodPut, oauthClientPath("slack_bot"), sent)

		s.Equal(http.StatusBadRequest, status)
		s.Contains(string(body), `"code":"not_configured"`)
		s.Contains(string(body), "OAuth clients cannot be stored: connectors are not enabled on this deployment")
	}
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
	s.Equal(http.StatusNotFound, s.unauthenticatedClient.do(http.MethodPost, providerAppEventsPath+"slack_bot/"+app,
		map[string]string{"type": "url_verification", "challenge": "c"}, nil))
}

// AI-863: a put without client_id answers as it did when the schema required it, whatever the
// connector and whatever else the body carries.
func (s *OAuthClientsOffSuite) TestAPutWithoutAClientIDIsRefusedAsBeforeProviderApps() {
	for _, connector := range []string{"slack_bot", "linq", "nope"} {
		status, body := s.serverClient.call(http.MethodPut, oauthClientPath(connector),
			ConnectorOAuthClientRequest{ProviderAppID: "line-" + s.utils.uuid(), SigningSecret: "whsec_c2lnbmluZw=="})

		s.Equal(http.StatusBadRequest, status, connector)
		s.Contains(string(body), `"message":"validation failed: expected required property client_id to be present (body)"`, connector)
		s.Contains(string(body), `"type":"invalid_request","code":"validation_failed"`, connector)
	}
}

// AI-881: with connectors off Telnyx's account put alone and a Telnyx-signed event on its
// route answer byte for byte as on base f5462686, captured by a probe of this suite there:
// PUT 400 validation_failed, POST 404 not_found.
func (s *OAuthClientsOffSuite) TestATelnyxAccountAndItsEventsAnswerAsOnBase() {
	app := "profile-" + s.utils.uuid()

	status, body := s.serverClient.call(http.MethodPut, oauthClientPath("telnyx"),
		ConnectorOAuthClientRequest{ProviderAppID: app, SigningSecret: "c2lnbmluZw=="})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"error":{"message":"validation failed: expected required property client_id to be present (body)","type":"invalid_request","code":"validation_failed","doc_url":"https://getstream.io/agents/docs/api/errors/#validation_failed"}}`)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "telnyx")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)

	request, err := http.NewRequest(http.MethodPost, s.server.URL+providerAppEventsPath+"telnyx/"+app,
		strings.NewReader(`{"data":{"event_type":"message.received"}}`))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Telnyx-Signature-Ed25519", "c2ln")
	request.Header.Set("Telnyx-Timestamp", "1700000000")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	answer, err := io.ReadAll(response.Body)
	s.Require().NoError(err)
	s.Equal(http.StatusNotFound, response.StatusCode)
	s.Equal("application/json", response.Header.Get("Content-Type"))
	s.Contains(string(answer), `"error":{"message":"this connector takes no events here","type":"not_found","code":"not_found","doc_url":"https://getstream.io/agents/docs/api/errors/#not_found"}}`)
}

// AI-879: with connectors off a Meta app put alone, a WhatsApp-signed event on its route and
// Meta's handshake there answer byte for byte as on base ad3fffd0, captured by a probe of this
// suite there: PUT 400 validation_failed, POST 404 not_found, GET and HEAD 405
// method_not_allowed with no Allow header.
func (s *OAuthClientsOffSuite) TestAWhatsAppAppItsEventsAndItsHandshakeAnswerAsOnBase() {
	app := "1" + strings.Repeat("0", 14)

	status, body := s.serverClient.call(http.MethodPut, oauthClientPath("whatsapp"),
		ConnectorOAuthClientRequest{ProviderAppID: app, SigningSecret: "app-secret"})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"error":{"message":"validation failed: expected required property client_id to be present (body)","type":"invalid_request","code":"validation_failed","doc_url":"https://getstream.io/agents/docs/api/errors/#validation_failed"}}`)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "whatsapp")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)

	route := s.server.URL + providerAppEventsPath + "whatsapp/" + app
	answer := func(method, target string, payload io.Reader) (*http.Response, string) {
		request, err := http.NewRequest(method, target, payload)
		s.Require().NoError(err)
		request.Header.Set("X-Hub-Signature-256", "sha256=00")
		response, err := http.DefaultClient.Do(request)
		s.Require().NoError(err)
		defer response.Body.Close()
		read, err := io.ReadAll(response.Body)
		s.Require().NoError(err)
		return response, string(read)
	}
	response, read := answer(http.MethodPost, route, strings.NewReader(`{"object":"whatsapp_business_account","entry":[]}`))
	s.Equal(http.StatusNotFound, response.StatusCode)
	s.Contains(read, `"error":{"message":"this connector takes no events here","type":"not_found","code":"not_found","doc_url":"https://getstream.io/agents/docs/api/errors/#not_found"}}`)

	response, read = answer(http.MethodGet, route+"?hub.mode=subscribe&hub.verify_token="+app+"&hub.challenge=987", nil)
	s.Equal(http.StatusMethodNotAllowed, response.StatusCode)
	s.Equal("application/json", response.Header.Get("Content-Type"))
	s.Empty(response.Header.Get("Allow"))
	s.Contains(read, `"error":{"message":"GET is not served on this route","type":"method_not_allowed","code":"method_not_allowed","doc_url":"https://getstream.io/agents/docs/api/errors/#method_not_allowed"}}`)
	s.NotContains(read, "987")

	response, read = answer(http.MethodHead, route, nil)
	s.Equal(http.StatusMethodNotAllowed, response.StatusCode)
	s.Empty(read)
}

// AI-863: Linq's account put alone answers as any put does with connectors off, and its
// events route takes nothing, as slack_bot's.
func (s *OAuthClientsOffSuite) TestALinqAccountIsNotStoredAndItsEventsRouteTakesNothing() {
	app := "line-" + s.utils.uuid()

	status, body := s.serverClient.call(http.MethodPut, oauthClientPath("linq"),
		ConnectorOAuthClientRequest{ClientID: "client", ProviderAppID: app, SigningSecret: "whsec_c2lnbmluZw=="})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"not_configured"`)
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "linq")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
	s.Equal(http.StatusNotFound, s.unauthenticatedClient.do(http.MethodPost, providerAppEventsPath+"linq/"+app,
		map[string]string{"event_type": "message.received"}, nil))
}
