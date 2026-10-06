//go:build integration

package api

import (
	"context"
	"net/http"
	"strings"
	"testing"

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
}

func TestOAuthClientsSuite(t *testing.T) {
	runSuite(t, new(OAuthClientsSuite))
}

func (s *OAuthClientsSuite) SetupSuite() {
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: namedScheme(oauth2code.Name)}}
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

// providerApp stores the test app's record for connector as the router would write one it
// created or was handed (T54): with a provider app id of its own and both secrets sealed, the
// signing secret only when one is given. The API takes neither the app id nor the signing secret.
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

func confidentialClient(secret string) ConnectorOAuthClientRequest {
	return ConnectorOAuthClientRequest{ClientID: "the-apps-client", ClientSecret: secret}
}
