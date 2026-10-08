//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/apikey"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectionTokenSuite exports a connection's access credential (T45, AI-874), end to end: the
// router's resolver over the suite's database, oauth2_code with the operator's client in the
// environment and the app's own client as a record, api_key and bearer.
type ConnectionTokenSuite struct {
	RouterSuite
	provider *fakeprovider.Server
}

func TestConnectionTokenSuite(t *testing.T) {
	runSuite(t, new(ConnectionTokenSuite))
}

func (s *ConnectionTokenSuite) SetupSuite() {
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
	s.connectors = core.Registry{Schemes: map[string]core.Scheme{
		oauth2code.Name: code, apikey.Name: apikey.New(), bearer.Name: bearer.New(),
	}}
	s.connectorHTTP = s.provider.Client()
	s.RouterSuite.SetupSuite()
}

func (s *ConnectionTokenSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectionTokenSuite) TestOnlyTheAppsBackendMayExportAToken() {
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodPost, tokenPath(id), nil, nil)
	})
}

func (s *ConnectionTokenSuite) TestAnotherAppIsToldTheConnectionDoesNotExist() {
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))

	s.assertHiddenFromOtherApps(func(as *testClient) int {
		return as.do(http.MethodPost, tokenPath(id), nil, nil)
	})
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestAGrantIssuedToTheAppsOwnClientExportsItsAccessTokenAndNoRefreshToken() {
	connector := s.connector(oauth2code.Name)
	s.putOwnClient(connector)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "the-refresh-token")

	status, raw := s.serverClient.call(http.MethodPost, tokenPath(id), nil)

	s.Require().Equal(http.StatusOK, status, string(raw))
	var token ConnectionToken
	s.Require().NoError(json.Unmarshal(raw, &token))
	s.Equal(id, token.ConnectionID)
	s.Equal("Authorization", token.Header)
	s.Equal("Bearer the-access-token", token.Value)
	s.Require().NotNil(token.ExpiresAt)
	s.WithinDuration(time.Now().Add(time.Hour), *token.ExpiresAt, time.Minute)
	s.NotContains(string(raw), "the-refresh-token")
}

// TestATokenIsNeverStored: RFC 6749 section 5.1 has a response holding a token say no-store.
func (s *ConnectionTokenSuite) TestATokenIsNeverStored() {
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))

	response := s.serverClient.raw(http.MethodPost, tokenPath(id), nil)

	s.Equal(http.StatusOK, response.StatusCode)
	s.Equal("no-store", response.Header.Get("Cache-Control"))
}

func (s *ConnectionTokenSuite) TestEachExportWritesOneAuditRow() {
	connector := s.connector(oauth2code.Name)
	s.putOwnClient(connector)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "")

	for range 2 {
		s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil))
	}

	exports := s.exports(id)
	s.Require().Len(exports, 2)
	for _, row := range exports {
		s.Equal(connector, row.ConnectorID)
		s.Equal(store.OwnerApp, row.OwnerType)
		s.Equal(2, row.Revision, "the credential revision the import committed")
		s.NotEmpty(row.RequestID)
	}
	s.NotEqual(exports[0].RequestID, exports[1].RequestID)
}

func (s *ConnectionTokenSuite) TestTheExportIsListedInTheConnectorAudit() {
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil))

	var page ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+id, nil, &page))

	actions := make([]ConnectorAuditAction, 0, len(page.Items))
	for _, row := range page.Items {
		actions = append(actions, row.Action)
	}
	s.Equal([]ConnectorAuditAction{store.AuditTokenExport, store.AuditGrantCreated}, actions, "newest first")
}

// TestAGrantIssuedToTheOperatorsClientIsRefused: Stream's app, from the environment, as when
// the app has no client record for the connector.
func (s *ConnectionTokenSuite) TestAGrantIssuedToTheOperatorsClientIsRefused() {
	id := s.connection(appOwned(s.connector(oauth2code.Name)))
	s.importGrant(id, "the-access-token", "the-refresh-token")

	status, body := s.serverClient.call(http.MethodPost, tokenPath(id), nil)

	s.Equal(http.StatusForbidden, status)
	s.Contains(string(body), "not issued to the app's own OAuth client")
	s.NotContains(string(body), "the-access-token")
	s.Empty(s.exports(id))
}

// TestAGrantIssuedToTheOperatorsClientStaysRefusedOnceTheAppPutsItsOwn: the record says whose
// client the connector uses now, not whose client the grant was issued to.
func (s *ConnectionTokenSuite) TestAGrantIssuedToTheOperatorsClientStaysRefusedOnceTheAppPutsItsOwn() {
	connector := s.connector(oauth2code.Name)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "")
	s.putOwnClient(connector)

	status := s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestAGrantIssuedToAClientTheRouterCreatedIsRefused() {
	connector := s.connector(oauth2code.Name)
	s.putManagedClient(connector)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "")

	status := s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestAGrantWhoseClientTheAppRemovedIsRefused() {
	connector := s.connector(oauth2code.Name)
	s.putOwnClient(connector)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "")
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, oauthClientPath(connector), nil, nil))

	status := s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.exports(id))
}

// TestAGrantWhoseClientTheRouterReplacedIsRefused: the app's own client is gone and the
// connector's record is now one the router created.
func (s *ConnectionTokenSuite) TestAGrantWhoseClientTheRouterReplacedIsRefused() {
	connector := s.connector(oauth2code.Name)
	s.putOwnClient(connector)
	id := s.connection(appOwned(connector))
	s.importGrant(id, "the-access-token", "")
	s.Require().NoError(s.store.DeleteConnectorOAuthClient(context.Background(), s.customerID(), connector, core.ClientCustomer))
	s.putManagedClient(connector)

	status := s.serverClient.do(http.MethodPost, tokenPath(id), nil, nil)

	s.Equal(http.StatusForbidden, status)
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestAnAPIKeyConnectionExportsItsKeyInItsHeader() {
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))

	var token ConnectionToken
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, tokenPath(id), nil, &token))

	s.Equal("X-Api-Key", token.Header)
	s.Equal("the-key", token.Value)
	s.Nil(token.ExpiresAt, "a key has no expiry")
	s.Len(s.exports(id), 1)
}

func (s *ConnectionTokenSuite) TestASchemeThatDoesNotExportIsRefused() {
	id := s.connection(appOwned(s.connector(bearer.Name)))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{bearer.SuppliedToken: "the-token"}}, nil))

	status, body := s.serverClient.call(http.MethodPost, tokenPath(id), nil)

	s.Equal(http.StatusForbidden, status)
	s.Contains(string(body), "a bearer connection's credential is not exported")
	s.NotContains(string(body), "the-token")
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestAPendingConnectionHasNothingToExport() {
	id := s.connection(appOwned(s.connector(apikey.Name)))

	status, body := s.serverClient.call(http.MethodPost, tokenPath(id), nil)

	s.Equal(http.StatusConflict, status)
	s.Contains(string(body), "the connection has no credential to export")
	s.Empty(s.exports(id))
}

// TestAUserConnectionExportsOnlyForItsOwnUser: the backend reaches a user's connection only
// while it acts for that user, as for a read.
func (s *ConnectionTokenSuite) TestAUserConnectionExportsOnlyForItsOwnUser() {
	alice := s.data.createUser()
	asAlice := s.serverClient.actingFor(alice)
	var created Connection
	s.Require().Equal(http.StatusCreated, asAlice.do(http.MethodPost, "/v1/agents/connections",
		userOwned(s.connector(apikey.Name), alice), &created))
	s.Require().Equal(http.StatusOK, asAlice.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials", keyValues(), nil))

	s.Equal(http.StatusNotFound, s.serverClient.do(http.MethodPost, tokenPath(created.ID), nil, nil), "acting for no one")
	s.Equal(http.StatusNotFound, s.serverClient.actingFor(s.data.createUser()).do(http.MethodPost, tokenPath(created.ID), nil, nil), "acting for another user")
	s.Equal(http.StatusOK, asAlice.do(http.MethodPost, tokenPath(created.ID), nil, nil))
	s.Len(s.exports(created.ID), 1)
}

func (s *ConnectionTokenSuite) TestAnUnknownConnectionIsNotFound() {
	status, body := s.serverClient.call(http.MethodPost, tokenPath("nope"), nil)

	s.Equal(http.StatusNotFound, status)
	s.Contains(string(body), `"code":"connection_not_found"`)
}

// connector stores a connector of the suite's app taking scheme, whose consent may use the
// app's own client, a client the router created, or the operator's in the environment.
func (s *ConnectionTokenSuite) TestAnExportWhoseAuditRowCannotBeWrittenIsNotHandedOver() {
	ctx := context.Background()
	id := s.keyConnection(appOwned(s.connector(apikey.Name)))
	_, err := s.store.DB().ExecContext(ctx, `CREATE OR REPLACE FUNCTION refuse_token_export() RETURNS trigger AS $$
		BEGIN RAISE EXCEPTION 'refused by the test'; END $$ LANGUAGE plpgsql`)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, `CREATE TRIGGER refuse_token_export BEFORE INSERT ON connector_audit
		FOR EACH ROW WHEN (NEW.connection_id = '`+id+`' AND NEW.action = 'token_export') EXECUTE FUNCTION refuse_token_export()`)
	s.Require().NoError(err)
	defer func() {
		_, err := s.store.DB().ExecContext(ctx, "DROP TRIGGER refuse_token_export ON connector_audit")
		s.Require().NoError(err)
	}()

	status, body := s.serverClient.call(http.MethodPost, tokenPath(id), nil)

	s.Equal(http.StatusInternalServerError, status, string(body))
	s.NotContains(string(body), "the-key")
	s.Empty(s.exports(id))
}

func (s *ConnectionTokenSuite) TestARenewalTheProviderCannotAnswerIsUnavailableAndExportsNothing() {
	id := "custom_export" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Fake
endpoints:
  authorize: ` + s.provider.URL + fakeprovider.PathAuthorize + `
  token: ` + s.provider.URL + fakeprovider.PathToken + `
  refresh: ` + s.provider.URL + `/no-such-path
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [oauth2_code]
client:
  registration: [customer, managed, operator]
  env: FAKE
scopes:
  list: [chat:write]
  separator: ","
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	s.putOwnClient(id)
	conn := s.connection(appOwned(id))
	status, body := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+conn+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{
			oauth2code.SuppliedAccessToken: "the-access-token", oauth2code.SuppliedRefreshToken: "the-refresh-token",
			oauth2code.SuppliedExpiresAt: time.Now().Add(-time.Minute).UTC().Format(time.RFC3339), oauth2code.SuppliedScope: "chat:write"}})
	s.Require().Equal(http.StatusOK, status, string(body))

	status, body = s.serverClient.call(http.MethodPost, tokenPath(conn), nil)

	s.Equal(http.StatusServiceUnavailable, status, string(body))
	s.NotContains(string(body), "the-access-token")
	s.Empty(s.exports(conn))
}

func (s *ConnectionTokenSuite) connector(scheme string) string {
	id := "custom_export" + strings.ReplaceAll(s.utils.uuid(), "-", "")
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
  registration: [customer, managed, operator]
  env: FAKE
scopes:
  list: [chat:write]
  separator: ","
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

func (s *ConnectionTokenSuite) connection(body map[string]any) string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", body, &created))
	return created.ID
}

// keyConnection is an api_key connection given the-key in X-Api-Key.
func (s *ConnectionTokenSuite) keyConnection(body map[string]any) string {
	id := s.connection(body)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+id+"/credentials", keyValues(), nil))
	return id
}

// importGrant gives the connection a grant expiring in an hour, so nothing renews it, with
// refresh when it is not empty. The client is the first one the connector's records and the
// environment have, as an import picks it.
func (s *ConnectionTokenSuite) importGrant(id, access, refresh string) {
	values := map[string]string{
		oauth2code.SuppliedAccessToken: access,
		oauth2code.SuppliedExpiresAt:   time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
		oauth2code.SuppliedScope:       "chat:write",
	}
	if refresh != "" {
		values[oauth2code.SuppliedRefreshToken] = refresh
	}
	status, body := s.serverClient.call(http.MethodPut, "/v1/agents/connections/"+id+"/credentials",
		map[string]any{"expected_revision": 1, "values": values})
	s.Require().Equal(http.StatusOK, status, string(body))
}

func (s *ConnectionTokenSuite) putOwnClient(connector string) {
	status, body := s.serverClient.call(http.MethodPut, oauthClientPath(connector), confidentialClient("the-apps-secret"))
	s.Require().Equal(http.StatusCreated, status, string(body))
}

// putManagedClient stores a client the router created for the app, as T54 will.
func (s *ConnectionTokenSuite) putManagedClient(connector string) {
	_, err := s.store.PutConnectorOAuthClient(context.Background(), &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: connector, Registration: core.ClientManaged,
		ClientID: "the-routers-client", ProviderAppID: "A" + strings.ReplaceAll(s.utils.uuid(), "-", ""),
	})
	s.Require().NoError(err)
}

// exports are the connection's token_export audit rows.
func (s *ConnectionTokenSuite) exports(id string) []store.ConnectorAuditEvent {
	rows, err := s.store.ConnectorAuditEvents(context.Background(), s.customerID(), store.AuditFilter{ConnectionID: id})
	s.Require().NoError(err)
	var exports []store.ConnectorAuditEvent
	for _, row := range rows {
		if row.Action == store.AuditTokenExport {
			exports = append(exports, row)
		}
	}
	return exports
}

func tokenPath(id string) string {
	return "/v1/agents/connections/" + id + "/token"
}

func keyValues() map[string]any {
	return map[string]any{"expected_revision": 1, "values": map[string]string{apikey.SuppliedKey: "the-key", apikey.SuppliedHeader: "X-Api-Key"}}
}

// ConnectionTokenOffSuite is connectors off, the control: cmd/router builds no resolver and no
// scheme with ROUTER_CONNECTORS_ENABLED unset. The export answers as the other connection
// operations do then, and the operations that were there answer as on base ad3fffd0 (probe:
// pr-w3d-t45/author-1.md).
type ConnectionTokenOffSuite struct {
	RouterSuite
}

func TestConnectionTokenOffSuite(t *testing.T) {
	runSuite(t, new(ConnectionTokenOffSuite))
}

func (s *ConnectionTokenOffSuite) SetupSuite() {
	s.connectorsOff = true
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *ConnectionTokenOffSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *ConnectionTokenOffSuite) TestAnUnknownConnectionIsNotFoundAsItsReadIs() {
	read, readBody := s.serverClient.failure(http.MethodGet, "/v1/agents/connections/nope", nil)
	export, exportBody := s.serverClient.failure(http.MethodPost, tokenPath("nope"), nil)

	s.Equal(http.StatusNotFound, read)
	s.Equal(read, export)
	s.Equal(readBody, exportBody)
}

// TestAConnectionMadeBeforeIsNotExportedAndReadsAsOnBase: a row left from when connectors were
// on. Its read answers 200. Its validate answers as cmd/router does with connectors off, where
// there is no resolver and so no transports (main.go): errConnectionToolsOff.
func (s *ConnectionTokenOffSuite) TestAConnectionMadeBeforeIsNotExportedAndReadsAsOnBase() {
	definition, err := s.store.LatestConnectorDefinition(context.Background(), s.customerID(), "linear")
	s.Require().NoError(err)
	connection := store.ConnectorConnection{CustomerID: s.customerID(), ConnectorID: "linear",
		DefinitionRevision: definition.Revision, OwnerType: store.OwnerApp, AuthScheme: oauth2code.Name}
	s.Require().NoError(s.store.CreateConnectorConnection(context.Background(),
		core.Registry{Schemes: map[string]core.Scheme{oauth2code.Name: namedScheme(oauth2code.Name)}}, &connection))

	status, body := s.serverClient.call(http.MethodPost, tokenPath(connection.ID), nil)

	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), `"code":"not_configured"`)
	s.Contains(string(body), "connection tokens cannot be exported: connectors are not enabled on this deployment")
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+connection.ID, nil, nil))
	status, body = s.serverClient.call(http.MethodPost, "/v1/agents/connections/"+connection.ID+"/validate", nil)
	s.Equal(http.StatusBadRequest, status)
	s.Contains(string(body), errConnectionToolsOff.Message)
	rows, err := s.store.ConnectorAuditEvents(context.Background(), s.customerID(), store.AuditFilter{ConnectionID: connection.ID})
	s.Require().NoError(err)
	s.Empty(rows)
}
