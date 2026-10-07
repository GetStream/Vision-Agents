//go:build integration

package api

import (
	"context"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// ConnectionRecordsSuite is what the router keeps about connections' use (T29, T47): the log
// of a connection's tool calls, the audit of its grants, and the user delete that takes a
// user's connections with their log. The fake provider is the MCP server, as in
// SessionConnectorsSuite, so a call in the log is one a session really made.
type ConnectionRecordsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued. Synthetic, fresh per suite.
	token string
}

func TestConnectionRecordsSuite(t *testing.T) {
	runSuite(t, new(ConnectionRecordsSuite))
}

func (s *ConnectionRecordsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
}

func (s *ConnectionRecordsSuite) SetupTest() {
	s.useFixture("standard")
}

func (s *ConnectionRecordsSuite) TestOnlyTheAppsBackendMayReadAConnectionsCalls() {
	id := s.connection(nil, s.connector())

	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id+"/invocations", nil, nil)
	})
}

func (s *ConnectionRecordsSuite) TestOnlyTheAppsBackendMayReadTheAudit() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodGet, "/v1/agents/connector-audit", nil, nil)
	})
}

func (s *ConnectionRecordsSuite) TestOnlyTheAppsBackendMayDeleteAUsersConnections() {
	s.assertPosture(serverOnly, func(as *testClient) int {
		return as.do(http.MethodDelete, "/v1/agents/users/"+s.utils.uuid()+"/connections", nil, nil)
	})
}

// TestAModelsToolCallLeavesOneRowInItsConnectionsLog: the whole path, read back through the
// endpoint by the backend acting for the connection's owner.
func (s *ConnectionRecordsSuite) TestAModelsToolCallLeavesOneRowInItsConnectionsLog() {
	connector := s.connector()
	mine := s.connection(s.client, connector)
	config := s.config(connector, s.client, mine, false)

	opened := s.client.createSession(s.session(config, mine))
	s.toolRanOn(s.serverClient.actingFor(s.client), s.client.opens("/v1/agents/sessions/"+opened.Id+"/events"), opened.Id)

	calls := s.calls(s.serverClient.actingFor(s.client), mine, 1)
	s.Equal(mine, calls[0].ConnectionID)
	s.Equal(config, calls[0].ConfigID)
	s.Equal("crm", calls[0].Binding)
	s.Equal("echo", calls[0].Tool)
	s.Equal(opened.Id, calls[0].SessionID)
	s.Nil(calls[0].ErrorType)
}

func (s *ConnectionRecordsSuite) TestAnIncognitoSessionsCallIsLoggedWithoutTheSession() {
	connector := s.connector()
	mine := s.connection(s.client, connector)
	request := s.session(s.config(connector, s.client, mine, false), mine)
	request.Incognito = pointerTo(true)

	opened := s.client.createSession(request)
	// An incognito conversation is kept nowhere, so it is asked directly, not by command.
	watching := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusAccepted, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: "ask the crm"}, nil))
	s.Require().Equal(connectorEchoText, s.await(watching, "tool_ran")["result"])

	calls := s.calls(s.serverClient.actingFor(s.client), mine, 1)
	s.Empty(calls[0].SessionID)
	s.Equal("echo", calls[0].Tool)
}

func (s *ConnectionRecordsSuite) TestAnotherUsersConnectionsCallsAreNotFound() {
	bobs := s.connection(s.data.createUser(), s.connector())

	s.Equal(http.StatusNotFound, s.serverClient.actingFor(s.client).do(http.MethodGet,
		"/v1/agents/connections/"+bobs+"/invocations", nil, nil))
}

func (s *ConnectionRecordsSuite) TestACursorTheLogDidNotHandOutIsRefused() {
	id := s.connection(nil, s.connector())

	s.Equal(http.StatusBadRequest, s.serverClient.do(http.MethodGet,
		"/v1/agents/connections/"+id+"/invocations?cursor=not-a-cursor", nil, nil))
}

// TestStoringACredentialLeavesOneGrantCreatedRowNamingTheRequest: the credentials write, then
// the delete, each once, and the deleted connection's rows still listed.
func (s *ConnectionRecordsSuite) TestStoringACredentialAndDeletingTheConnectionEachLeaveOneRow() {
	connector := s.connector()
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(connector), bearer.Name), &created))
	request := s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.serverClient.withRequestID(request).do(http.MethodPut,
		"/v1/agents/connections/"+created.ID+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil, nil))

	rows := s.audit(created.ID, 2)
	s.Equal(ConnectorAuditAction(store.AuditGrantRevoked), rows[0].Action)
	s.Equal(store.AuditReasonDeleted, rows[0].Reason)
	s.Equal(ConnectorAuditAction(store.AuditGrantCreated), rows[1].Action)
	s.Equal(store.AuditReasonCredentials, rows[1].Reason)
	s.Equal(2, rows[1].Revision)
	s.Equal(request, rows[1].RequestID)
	s.Equal(ConnectionOwnerType(store.OwnerApp), rows[1].OwnerType)
}

func (s *ConnectionRecordsSuite) TestDeletingAConnectionThatNeverHadAGrantLeavesNoRow() {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections",
		withScheme(appOwned(s.connector()), bearer.Name), &created))

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/connections/"+created.ID, nil, nil))

	s.Empty(s.audit(created.ID, 0))
}

func (s *ConnectionRecordsSuite) TestAnotherAppsAuditIsNotListed() {
	id := s.connection(nil, s.connector())

	var page ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.data.backendOfAnotherApp().do(http.MethodGet,
		"/v1/agents/connector-audit?connection_id="+id, nil, &page))

	s.Empty(page.Items)
	s.Len(s.audit(id, 1), 1)
}

// TestDeletingAUsersConnectionsLeavesTheNextSessionWithNone: the offboarding request. Alice's
// connections go, with their log; Bob's and the app's stay; the audit says which grants
// ended, naming neither user nor account; her next session cannot use what she had.
func (s *ConnectionRecordsSuite) TestDeletingAUsersConnectionsLeavesTheNextSessionWithNone() {
	connector := s.connector()
	alice, bob := s.data.createUser(), s.data.createUser()
	hers := s.connection(alice, connector)
	var pending Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.actingFor(alice).do(http.MethodPost, "/v1/agents/connections",
		withScheme(userOwned(connector, alice), bearer.Name), &pending))
	bobs := s.connection(bob, connector)
	apps := s.connection(nil, connector)
	config := s.config(connector, alice, hers, true)
	opened := alice.createSession(s.session(config, hers))
	s.toolRanOn(s.serverClient.actingFor(alice), alice.opens("/v1/agents/sessions/"+opened.Id+"/events"), opened.Id)
	s.calls(s.serverClient.actingFor(alice), hers, 1)

	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+alice.userID+"/connections", nil, nil))

	for _, id := range []string{hers, pending.ID} {
		s.Equal(http.StatusNotFound, s.serverClient.actingFor(alice).do(http.MethodGet, "/v1/agents/connections/"+id, nil, nil))
	}
	var left int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_connections WHERE customer_id = ? AND owner_id = ?", s.customerID(), alice.userID).Scan(&left))
	s.Zero(left)
	var logged int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM connector_invocations WHERE connection_id = ?", hers).Scan(&logged))
	s.Zero(logged)
	s.Equal(http.StatusOK, s.serverClient.actingFor(bob).do(http.MethodGet, "/v1/agents/connections/"+bobs, nil, nil))
	s.Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+apps, nil, nil))
	revoked := s.audit(hers, 2)
	s.Equal(ConnectorAuditAction(store.AuditGrantRevoked), revoked[0].Action)
	s.Equal(store.AuditReasonUserDeleted, revoked[0].Reason)
	s.Empty(s.audit(pending.ID, 0), "a connection with no grant had none to revoke")
	status, failure := alice.failure(http.MethodPost, "/v1/agents/sessions", s.session(config, hers))
	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `required connector "crm" cannot be used: connection_unavailable`)
}

func (s *ConnectionRecordsSuite) TestDeletingTheConnectionsOfAUserWithNoneIsNotAnError() {
	s.Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/users/"+s.utils.uuid()+"/connections", nil, nil))
}

// TestASessionWithNoConnectorsWritesNoRecord: nothing configured, nothing written, as on base.
func (s *ConnectionRecordsSuite) TestASessionWithNoConnectorsWritesNoRecord() {
	s.useApp(s.data.createApp())
	opened := s.serverClient.createSession(textSession(nil))
	watching := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusAccepted, s.serverClient.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/responses",
		CreateResponseRequest{Text: "What is the capital of France?"}, nil))
	s.await(watching, "responded")
	s.serverClient.stopSession(opened.Id)

	for _, table := range []string{"connector_invocations", "connector_audit"} {
		var rows int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM "+table+" WHERE customer_id = ?", s.customerID()).Scan(&rows))
		s.Zero(rows, table)
	}
}

// calls is a connection's log as as reads it, once it holds count rows.
func (s *ConnectionRecordsSuite) calls(as *testClient, id string, count int) []ConnectionInvocation {
	var page ConnectionInvocationPage
	s.Require().Eventually(func() bool {
		return as.do(http.MethodGet, "/v1/agents/connections/"+id+"/invocations", nil, &page) == http.StatusOK &&
			len(page.Items) >= count
	}, settleFor, 20*time.Millisecond)
	s.Require().Len(page.Items, count)
	return page.Items
}

// audit is a connection's audit rows, newest first, as the app's backend reads them, once
// they hold count.
func (s *ConnectionRecordsSuite) audit(id string, count int) []ConnectorAuditEvent {
	var page ConnectorAuditPage
	s.Require().Eventually(func() bool {
		return s.serverClient.do(http.MethodGet, "/v1/agents/connector-audit?connection_id="+url.QueryEscape(id), nil, &page) == http.StatusOK &&
			len(page.Items) >= count
	}, settleFor, 20*time.Millisecond)
	s.Require().Len(page.Items, count)
	return page.Items
}

// connector stores a connector of the suite's app at the fake, taking a bearer token.
func (s *ConnectionRecordsSuite) connector() string {
	id := "custom_crm" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: CRM
endpoints:
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [bearer]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// connection is a validated connection to connector holding the fake's token, the app's
// when user is nil and otherwise user's, made by the backend acting for them.
func (s *ConnectionRecordsSuite) connection(user *testClient, connector string) string {
	as, body := s.serverClient, appOwned(connector)
	if user != nil {
		as, body = s.serverClient.actingFor(user), userOwned(connector, user)
	}
	var created Connection
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/connections", withScheme(body, bearer.Name), &created))
	s.Require().Equal(http.StatusOK, as.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, as.do(http.MethodPost, "/v1/agents/connections/"+created.ID+"/validate", nil, &validation))
	s.Require().Equal(validationConnected, string(validation.Status), validation.Error)
	return created.ID
}

// config is a new agent config of the suite's app on the connecting model, binding the
// connector's echo under crm to the connection each session picks, at the digest the
// validate of listed found (as the backend acting for owner reads it), required when asked.
func (s *ConnectionRecordsSuite) config(connector string, owner *testClient, listed string, required bool) string {
	as := s.serverClient
	if owner != nil {
		as = s.serverClient.actingFor(owner)
	}
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections/"+listed+"/tools", nil, &tools))
	digest := ""
	for _, tool := range tools.Tools {
		if tool.Name == "echo" {
			digest = tool.SchemaDigest
		}
	}
	s.Require().NotEmpty(digest, "the connection lists no echo")
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "records-" + s.utils.uuid(), "llm": "connecting/connector-model",
		"connectors": []map[string]any{{"name": "crm", "connector_id": connector, "connection": map[string]any{"type": "session"},
			"tools": []map[string]any{{"name": "echo", "schema_digest": digest}}, "required": required}},
	}, &created))
	return created.Id
}

// session is a text session of config choosing connection for crm.
func (s *ConnectionRecordsSuite) session(config, connection string) CreateSessionRequest {
	return CreateSessionRequest{ConfigId: &config, Text: pointerTo(true),
		ConnectorBindings: &[]SessionConnectorBinding{{Name: "crm", ConnectionId: connection}}}
}

// toolRanOn asks the session for a turn, whose model reaches for the connector's echo, and
// waits for the tool to run.
func (s *ConnectionRecordsSuite) toolRanOn(as *testClient, events *websocket.Conn, id string) {
	s.Require().Equal(http.StatusOK, as.do(http.MethodPost, "/v1/agents/sessions/"+id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())}, nil))
	ran := s.await(events, "tool_ran")
	s.Require().Equal(connectorEchoText, ran["result"])
}

// connectorAudit is a connection's audit rows, newest first, as the app's backend reads them.
func (s *RouterSuite) connectorAudit(id string) []ConnectorAuditEvent {
	var page ConnectorAuditPage
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet,
		"/v1/agents/connector-audit?connection_id="+url.QueryEscape(id), nil, &page))
	return page.Items
}

// withRequestID is c naming its requests id, as a caller's proxy may (X-Request-Id).
func (c *testClient) withRequestID(id string) *testClient {
	named := *c
	named.header = c.header.Clone()
	named.header.Set(RequestIDHeader, id)
	return &named
}
