//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/none"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2cc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/egress"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// suiteConnectorTimeout bounds one request a connection's client sends in the router suites:
// the router's connectorHTTPTimeout (cmd/router), 10 s.
const suiteConnectorTimeout = 10 * time.Second

// ConnectionToolsSuite sets a connection's credentials, validates it and reads its tools, end
// to end: the router's resolver and core.Transports over the suite's database, the mcp source,
// and T5's fake provider as the token endpoint and the MCP server. One fake serves the whole
// suite, since the schemes the router is built with are fixed when it starts; each test makes
// its own connector and connection in the suite's app.
type ConnectionToolsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued, which its MCP endpoint takes: the value a
	// bearer connection is given. Synthetic, fresh per suite.
	token string
}

func TestConnectionToolsSuite(t *testing.T) {
	runSuite(t, new(ConnectionToolsSuite))
}

// SetupSuite builds the schemes against the fake: oauth2_code with the operator's client in
// the environment as FAKE_MCP_CLIENT_ID and _SECRET, oauth2_client_credentials, bearer and
// none, and the mcp source. Every connection's client reaches the fake on loopback.
func (s *ConnectionToolsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	environment := map[string]string{"FAKE_MCP_CLIENT_ID": s.provider.ClientID, "FAKE_MCP_CLIENT_SECRET": s.provider.ClientSecret}
	code, err := oauth2code.New(oauth2code.Config{
		HTTP: s.provider.Client(),
		Clients: func(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
			return ConnectorClients(s.store, s.sealer, func(name string) string { return environment[name] })(ctx, ref, m, registration)
		},
		PublicEndpoint: loopbackOrPublic,
	})
	s.Require().NoError(err)
	clientCredentials, err := oauth2cc.New(oauth2cc.Config{HTTP: s.provider.Client(), PublicEndpoint: loopbackOrPublic})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes: map[string]core.Scheme{oauth2code.Name: code, oauth2cc.Name: clientCredentials,
			bearer.Name: bearer.New(), none.Name: none.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.token = s.issue()
	s.RouterSuite.SetupSuite()
}

func (s *ConnectionToolsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ConnectionToolsSuite) TestOnlyTheAppsBackendMaySetCredentials() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		id := s.connection(bearer.Name)
		return as.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil)
	})
}

func (s *ConnectionToolsSuite) TestOnlyTheAppsBackendMayValidateAndReadTools() {
	id := s.connected(bearer.Name)

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, nil)
	})
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, nil)
	})
}

func (s *ConnectionToolsSuite) TestAnotherAppIsToldTheConnectionDoesNotExist() {
	id := s.connected(bearer.Name)

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(2, s.token), nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, nil)
	})
	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, nil)
	})
}

// TestATokenIsStoredAndNeverShownAgain: no answer of any connection operation carries it.
func (s *ConnectionToolsSuite) TestATokenIsStoredAndNeverShownAgain() {
	id := s.connection(bearer.Name)

	status, body := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token))

	s.Require().Equal(http.StatusOK, status, string(body))
	var connection Connection
	s.Require().NoError(json.Unmarshal(body, &connection))
	s.Equal(ConnectionStatus(store.ConnectionConnected), connection.Status)
	s.Equal(2, connection.Revision)
	for _, path := range []string{"", "/tools"} {
		_, read := s.serverClient.call(http.MethodGet, "/v1/agents/connections/"+id+path, nil)
		s.NotContains(string(read), s.token, path)
	}
	_, validated := s.serverClient.call(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil)
	s.NotContains(string(body)+string(validated), s.token)
}

func (s *ConnectionToolsSuite) TestAStaleExpectedRevisionIsAConflict() {
	id := s.connected(bearer.Name)

	status := s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, "another-token"), nil)

	s.Equal(http.StatusConflict, status)
	s.Equal(2, s.get(id).Revision, "the credentials stayed as they were")
}

func (s *ConnectionToolsSuite) TestAValueTheSchemeDoesNotTakeIsRefusedWithoutBeingQuoted() {
	id := s.connection(bearer.Name)

	status, body := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{"password": "do-not-echo-me"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `\"password\" is not a value bearer takes`)
	s.NotContains(string(body), "do-not-echo-me")
	s.Equal(ConnectionStatus(store.ConnectionPending), s.get(id).Status)
}

func (s *ConnectionToolsSuite) TestAConnectorThatNeedsNoCredentialIsActivatedWithNothing() {
	id := s.connection(none.Name)

	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1}, &connection))

	s.Equal(ConnectionStatus(store.ConnectionConnected), connection.Status)
}

// TestAClientCredentialsConnectionIsSetUpWithItsClient: what no endpoint took before this
// one, so an app-owned oauth2_client_credentials connection (a Salesforce integration user)
// could not be set up. The client is tried at the token endpoint at once, and the token it
// gets lists the tools.
func (s *ConnectionToolsSuite) TestAClientCredentialsConnectionIsSetUpWithItsClient() {
	id := s.connection(oauth2cc.Name)

	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{
			oauth2cc.SuppliedClientID: s.provider.ClientID, oauth2cc.SuppliedClientSecret: s.provider.ClientSecret,
		}}, &connection))

	s.Equal(ConnectionStatus(store.ConnectionConnected), connection.Status)
	s.Equal(validationConnected, string(s.validate(id).Status))
}

// TestAnImportedGrantConnectsWithTheConnectorsEndpoints: the grant names no endpoint and no
// client; the scheme takes both from the manifest and the operator's environment.
func (s *ConnectionToolsSuite) TestAnImportedGrantConnectsWithTheConnectorsEndpoints() {
	id := s.connection(oauth2code.Name)

	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		s.importedGrant(s.token, "chat:write"), &connection))

	s.Equal(ConnectionStatus(store.ConnectionConnected), connection.Status)
	s.Equal([]string{"chat:write"}, connection.GrantedScopes)
	s.Equal(validationConnected, string(s.validate(id).Status))
}

func (s *ConnectionToolsSuite) TestValidateListsTheToolsAndToolsReadsThemBack() {
	id := s.connected(bearer.Name)

	validation := s.validate(id)
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, &tools))

	s.Equal(validationConnected, string(validation.Status))
	s.Require().NotNil(validation.CheckedAt)
	s.Equal(validation.ToolsDigest, tools.Digest)
	s.Equal(validation.CheckedAt.UTC(), tools.CheckedAt.UTC())
	var names []string
	for _, tool := range tools.Tools {
		names = append(names, tool.Name)
		s.Len(tool.SchemaDigest, 64, tool.Name)
	}
	s.Equal([]string{"echo", "fail"}, names, "both of the fake's tools/list pages")
}

func (s *ConnectionToolsSuite) TestToolsIsEmptyUntilAValidate() {
	id := s.connected(bearer.Name)

	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, &tools))

	s.Empty(tools.Tools)
	s.Nil(tools.CheckedAt)
}

func (s *ConnectionToolsSuite) TestAPendingConnectionValidatesAsPendingWithoutAskingTheProvider() {
	id := s.connection(bearer.Name)
	before := s.provider.Hits(fakeprovider.PathMCP)

	validation := s.validate(id)

	s.Equal(validationPending, string(validation.Status))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// TestAConnectionThatNeedsAReconnectSaysSoWithoutAskingTheProvider: the provider revoked the
// grant (as a verified tokens_revoked does), and validate opens no MCP session.
func (s *ConnectionToolsSuite) TestAConnectionThatNeedsAReconnectSaysSoWithoutAskingTheProvider() {
	id := s.connected(bearer.Name)
	s.Require().NoError(s.resolver.Revoke(context.Background(), core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: id}, core.SignalRevoked, time.Time{}))
	before := s.provider.Hits(fakeprovider.PathMCP)

	validation := s.validate(id)

	s.Equal(validationNeedsReauthorization, string(validation.Status))
	s.NotEmpty(validation.Error)
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// TestATokenTheProviderRefusesMovesTheConnectionToNeedsReauthorization: the MCP server answers
// 401 invalid_token, nothing renews a static token, so core.Transports invalidates it and the
// validate reports what the connection now needs.
func (s *ConnectionToolsSuite) TestATokenTheProviderRefusesMovesTheConnectionToNeedsReauthorization() {
	id := s.connection(bearer.Name)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, "not-a-token-the-fake-issued"), nil))

	validation := s.validate(id)

	s.Equal(validationNeedsReauthorization, string(validation.Status))
	s.Equal(ConnectionStatus(store.ConnectionNeedsReauthorization), s.get(id).Status)
}

// TestABare401WhoseRefreshIsRefusedValidatesAsNeedsReauthorization: the MCP server refuses an
// oauth2_code token with a bare 401 (resource_metadata, no error), as MCP servers do. The
// transport refreshes, the refresh is refused with invalid_grant, and the validate reports
// what the connection now needs instead of failed.
func (s *ConnectionToolsSuite) TestABare401WhoseRefreshIsRefusedValidatesAsNeedsReauthorization() {
	s.provider.Use(fakeprovider.ClientCredentials, fakeprovider.BareChallenge, fakeprovider.InvalidGrant)
	s.T().Cleanup(func() { s.provider.Use(fakeprovider.ClientCredentials) })
	id := s.connection(oauth2code.Name)
	grant := s.importedGrant("not-a-token-the-fake-issued", "chat:write")
	grant["values"].(map[string]string)[oauth2code.SuppliedRefreshToken] = "nor-a-refresh-token-it-issued"
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", grant, nil))
	refreshes := s.provider.Refreshes()

	validation := s.validate(id)

	s.Equal(validationNeedsReauthorization, string(validation.Status))
	s.Equal(ConnectionStatus(store.ConnectionNeedsReauthorization), s.get(id).Status)
	s.Equal(refreshes+1, s.provider.Refreshes(), "the bare 401 was renewed first")
}

// TestAGrantLackingAScopeAToolNeedsValidatesAsNeedsScopesAndNamesIt is T31's acceptance: a
// Slack-shaped connection granted chat:write only, with a tool that needs channels:read.
func (s *ConnectionToolsSuite) TestAGrantLackingAScopeAToolNeedsValidatesAsNeedsScopesAndNamesIt() {
	id := s.scoped("chat:write")

	validation := s.validate(id)
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id+"/tools", nil, &tools))

	s.Equal(validationNeedsScopes, string(validation.Status))
	s.Equal(codeScopeRequired, validation.Code)
	s.Equal([]string{"channels:read"}, validation.MissingScopes)
	s.Contains(validation.Error, "channels:read")
	s.Require().Len(tools.Tools, 2, "the tools are listed all the same")
	s.Equal([]string{"chat:write"}, tools.Tools[0].NeedsScopes)
	s.Equal([]string{"channels:read"}, tools.Tools[1].NeedsScopes)
}

func (s *ConnectionToolsSuite) TestAGrantWithEveryScopeItsToolsNeedValidatesAsConnected() {
	id := s.scoped("chat:write,channels:read")

	validation := s.validate(id)

	s.Equal(validationConnected, string(validation.Status))
	s.Empty(validation.MissingScopes)
}

func (s *ConnectionToolsSuite) TestOnlyTheToolsTheBodyNamesAreChecked() {
	id := s.scoped("chat:write")

	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate",
		map[string]any{"tools": []string{"echo"}}, &validation))

	s.Equal(validationConnected, string(validation.Status))
}

func (s *ConnectionToolsSuite) TestCheckingAToolTheConnectionDoesNotOfferIsRefused() {
	id := s.scoped("chat:write")

	status, body := s.serverClient.call(http.MethodPost, "/v1/agents/connections/"+id+"/validate",
		map[string]any{"tools": []string{"delete_workspace"}})

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `no tool named \"delete_workspace\"`)
}

// scoped is an oauth2_code connection whose grant was imported with scope, to a connector
// whose echo needs chat:write and fail channels:read.
func (s *ConnectionToolsSuite) scoped(scope string) string {
	connector := s.connector(oauth2code.Name, `    tools:
      - name: echo
        needs_scopes: [chat:write]
      - name: fail
        needs_scopes: [channels:read]
`)
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(connector), &created))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials",
		s.importedGrant(s.token, scope), nil))
	return created.ID
}

// connection is a pending app-owned connection to a new connector of the suite's app at the
// fake, authenticating with scheme.
func (s *ConnectionToolsSuite) connection(scheme string) string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(s.connector(scheme, "")), &created))
	return created.ID
}

// connected is a bearer connection given the fake's token, at revision 2.
func (s *ConnectionToolsSuite) connected(scheme string) string {
	id := s.connection(scheme)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", s.bearerToken(1, s.token), nil))
	return id
}

// connector stores a connector of the suite's app whose endpoints are the fake's, taking
// scheme; tools is more YAML under its mcp source.
func (s *ConnectionToolsSuite) connector(scheme, tools string) string {
	id := "custom_tools" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Fake
endpoints:
  authorize: ` + s.provider.URL + fakeprovider.PathAuthorize + `
  token: ` + s.provider.URL + fakeprovider.PathToken + `
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [` + scheme + `]
client:
  registration: [operator]
  env: FAKE
scopes:
  list: [chat:write, channels:read]
  separator: ","
sources:
  - kind: mcp
    endpoint: mcp
` + tools))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

func (s *ConnectionToolsSuite) bearerToken(revision int, token string) map[string]any {
	return map[string]any{"expected_revision": revision, "values": map[string]string{bearer.SuppliedToken: token}}
}

func (s *ConnectionToolsSuite) importedGrant(token, scope string) map[string]any {
	return map[string]any{"expected_revision": 1, "values": map[string]string{
		oauth2code.SuppliedAccessToken: token,
		oauth2code.SuppliedExpiresAt:   time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
		oauth2code.SuppliedScope:       scope,
	}}
}

func (s *ConnectionToolsSuite) validate(id string) ConnectionValidation {
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+id+"/validate", nil, &validation))
	return validation
}

func (s *ConnectionToolsSuite) get(id string) Connection {
	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+id, nil, &connection))
	return connection
}

// issue is an access token from the fake's client credentials grant.
func (s *ConnectionToolsSuite) issue() string {
	return issuedToken(&s.RouterSuite, s.provider)
}

// issuedToken is an access token from provider's client credentials grant (RFC 6749 section
// 4.4), which its MCP endpoint takes.
func issuedToken(s *RouterSuite, provider *fakeprovider.Server) string {
	form := url.Values{"grant_type": {"client_credentials"}}
	request, err := http.NewRequest(http.MethodPost, provider.URL+fakeprovider.PathToken, strings.NewReader(form.Encode()))
	s.Require().NoError(err)
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	request.SetBasicAuth(provider.ClientID, provider.ClientSecret)
	response, err := provider.Client().Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	var token struct {
		AccessToken string `json:"access_token"`
	}
	s.Require().NoError(json.NewDecoder(response.Body).Decode(&token))
	s.Require().NotEmpty(token.AccessToken)
	return token.AccessToken
}

// loopbackClients is egress.NewClient for a suite whose providers listen on loopback: the same
// redirect policy, over server's transport. Nil server is egress.NewClient itself.
func loopbackClients(server *http.Client) core.NewClientFunc {
	if server == nil {
		return nil
	}
	policy := egress.NewClient(0, nil).CheckRedirect
	return func(timeout time.Duration, wrap func(http.RoundTripper) http.RoundTripper) *http.Client {
		return &http.Client{Transport: wrap(server.Transport), Timeout: timeout, CheckRedirect: policy}
	}
}
