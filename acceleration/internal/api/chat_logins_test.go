//go:build integration

package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// loginCallTool is what the logging-in model calls: crm's echo, through the call_tool a
// binding waiting for a login is offered as.
const loginCallTool = "crm" + mcp.Separator + "call_tool"

// handoffToken is the shape of a handoff token: 32 random octets, base64url without padding
// (handoffBytes, randomToken).
var handoffToken = regexp.MustCompile(`^[A-Za-z0-9_-]{43}$`)

// ChatLoginsSuite is a login in the chat, end to end: a config binds crm as the caller's own
// connection, the session has none that works, and the model reaches for it. The reply asks
// for the login with T17's consent, a browser finishes it at the fake provider, and the
// session carries on by itself through the connection the consent connected.
type ChatLoginsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued, for the twin connection that lists echo's
	// schema digest. Synthetic, fresh per suite.
	token string
}

func TestChatLoginsSuite(t *testing.T) {
	runSuite(t, new(ChatLoginsSuite))
}

// SetupSuite builds oauth2_code against the fake with the operator's client in the
// environment, as AuthorizationsSuite does, and bearer for the twin, with the mcp source.
func (s *ChatLoginsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	s.provider.AllowRedirect(consentPublicURL + ConnectorCallbackPath)
	environment := map[string]string{"FAKE_MCP_CLIENT_ID": s.provider.ClientID, "FAKE_MCP_CLIENT_SECRET": s.provider.ClientSecret}
	code, err := oauth2code.New(oauth2code.Config{
		HTTP: s.provider.Client(),
		Clients: func(ctx context.Context, ref core.ConnectionRef, m core.ResolvedManifest, registration core.ClientRegistrationMethod) (oauth2code.Client, bool, error) {
			return ConnectorClients(s.store, s.sealer, func(name string) string { return environment[name] })(ctx, ref, m, registration)
		},
		PublicEndpoint: loopbackOrPublic,
	})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{oauth2code.Name: code, bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.publicURL, s.dashboardURL = consentPublicURL, consentDashboard
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
}

func (s *ChatLoginsSuite) SetupTest() {
	s.useFixture("standard")
}

// TestASessionBindingWithNoConnectionAsksInTheChatAndCarriesOn: «tell Nash a joke on Slack»
// with nothing connected. The reply carries the login, the person consents, and the agent
// runs the tool with nobody asking again.
func (s *ChatLoginsSuite) TestASessionBindingWithNoConnectionAsksInTheChatAndCarriesOn() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.session(s.config(connector, grant), nil))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	before := s.provider.Hits(fakeprovider.PathMCP)

	question := s.ask(opened.Id)
	asked := s.loginOn(events, "")

	s.Equal("crm", asked["name"])
	s.Equal(connector, asked["connector_id"])
	s.Equal("Connect Fake", asked["title"])
	s.Equal(consentPublicURL+connectorLaunchPath+asked["authorization_id"].(string), asked["launch_url"])
	s.Regexp(handoffToken, asked["handoff_token"])
	connection := s.connectionOf(s.client, asked["connection_id"].(string))
	s.Equal(ConnectionStatus(store.ConnectionPending), connection.Status, "made for the caller, not connected yet")
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "nothing reached the provider before the login")

	alice := newBrowser(&s.RouterSuite, s.provider)
	finished := alice.finish(s.consent(alice.handOff(s.started(asked))))
	s.Equal(s.landing(asked["connection_id"].(string)), finished.Header.Get("Location"))

	after := s.carryOn(events)
	s.Equal(loginCallTool, after.ran["tool"])
	s.Equal(connectorEchoText, after.ran["result"], "the tool ran on the account the login connected")
	s.Empty(after.ran["error"])
	s.Equal([]string{"connected"}, after.statuses, "the button says the login is done")
	s.Empty(slices.DeleteFunc(after.users, func(id string) bool { return id == question.UserMessageId }),
		"nobody is shown having asked again")
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), before)
}

// TestTheAttachmentCarriesNoCredential: the router's launch page, the attempt's id and a
// handoff token the first browser spends. No provider URL, no state, no client secret, no
// code: those reach the browser only from the handoff.
func (s *ChatLoginsSuite) TestTheAttachmentCarriesNoCredential() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.session(s.config(connector, grant), nil))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.ask(opened.Id)
	asked := s.loginOn(events, "")

	keys := make([]string, 0, len(asked))
	for key := range asked {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	s.Equal([]string{"authorization_id", "connection_id", "connector_id", "expires_at", "handoff_token",
		"launch_url", "name", "title", "type"}, keys)
	raw, err := json.Marshal(asked)
	s.Require().NoError(err)
	for _, secret := range []string{s.provider.URL, s.provider.ClientSecret, s.provider.ClientID, "state=", "code="} {
		s.NotContains(string(raw), secret)
	}
}

// TestASecondUserCannotUseTheLaunchURL: the handoff token is spent by the first browser,
// Alice's, and the callback finishes only in the browser holding its cookie. Bob, who got
// the launch URL and the token, can neither hand it off nor finish Alice's consent.
func (s *ChatLoginsSuite) TestASecondUserCannotUseTheLaunchURL() {
	connector, grant := s.connector()
	opened := s.client.createSession(s.session(s.config(connector, grant), nil))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.ask(opened.Id)
	asked := s.loginOn(events, "")
	started := s.started(asked)
	alice, bob := newBrowser(&s.RouterSuite, s.provider), newBrowser(&s.RouterSuite, s.provider)

	callback := s.consent(alice.handOff(started))
	handedOff := bob.post(started.LaunchURL, consentPublicURL, map[string]string{"handoff_token": started.HandoffToken})
	handedOff.Body.Close()
	finishedByBob := bob.finish(callback)

	s.Equal(http.StatusBadRequest, handedOff.StatusCode, "the token is spent")
	s.Equal(http.StatusForbidden, finishedByBob.StatusCode, "Bob's browser holds no cookie for it")
	s.Equal(ConnectionStatus(store.ConnectionPending), s.connectionOf(s.client, asked["connection_id"].(string)).Status)
	s.Equal(s.landing(asked["connection_id"].(string)), alice.finish(callback).Header.Get("Location"), "Alice still finishes it")
	s.Equal(connectorEchoText, s.carryOn(events).ran["result"])
}

// TestAConnectionThatNeedsReauthorizationAsksForAReconnect: the connection the caller chose
// is no longer taken by the provider. The login reconnects that same connection.
func (s *ChatLoginsSuite) TestAConnectionThatNeedsReauthorizationAsksForAReconnect() {
	connector, grant := s.connector()
	mine := s.connected(s.client, connector)
	_, err := s.store.DB().ExecContext(context.Background(),
		"UPDATE connector_connections SET status = ? WHERE id = ?", store.ConnectionNeedsReauthorization, mine)
	s.Require().NoError(err)
	opened := s.client.createSession(s.session(s.config(connector, grant), map[string]string{"crm": mine}))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")

	s.ask(opened.Id)
	asked := s.loginOn(events, "")
	s.Equal(mine, asked["connection_id"])
	started := s.started(asked)
	s.Equal(store.AttemptReconnect, s.attemptKind(started.ID))

	b := newBrowser(&s.RouterSuite, s.provider)
	s.Equal(s.landing(mine), b.finish(s.consent(b.handOff(started))).Header.Get("Location"))

	s.Equal(connectorEchoText, s.carryOn(events).ran["result"])
	s.Equal(ConnectionStatus(store.ConnectionConnected), s.connectionOf(s.client, mine).Status)
}

// TestEachPersonIsAskedForTheirOwnConnection: Bob's session on the same config asks Bob, on a
// connection of Bob's, and Alice's consent carries on in Alice's session only.
func (s *ChatLoginsSuite) TestEachPersonIsAskedForTheirOwnConnection() {
	connector, grant := s.connector()
	config := s.config(connector, grant)
	bob := s.data.createUser()
	alices := s.client.createSession(s.session(config, nil))
	bobs := bob.createSession(s.session(config, nil))
	alicesEvents := s.client.opens("/v1/agents/sessions/" + alices.Id + "/events")
	bobsEvents := bob.opens("/v1/agents/sessions/" + bobs.Id + "/events")

	s.ask(alices.Id)
	askedAlice := s.loginOn(alicesEvents, "")
	s.askAs(bob, bobs.Id)
	askedBob := s.loginOn(bobsEvents, "")

	s.NotEqual(askedAlice["connection_id"], askedBob["connection_id"])
	s.connectionOf(s.client, askedAlice["connection_id"].(string))
	s.connectionOf(bob, askedBob["connection_id"].(string))
	status := s.serverClient.actingFor(bob).do(http.MethodGet, "/v1/agents/connections/"+askedAlice["connection_id"].(string), nil, nil)
	s.Equal(http.StatusNotFound, status, "Alice's connection is not Bob's")

	b := newBrowser(&s.RouterSuite, s.provider)
	b.finish(s.consent(b.handOff(s.started(askedAlice))))
	s.Equal(connectorEchoText, s.carryOn(alicesEvents).ran["result"])
	s.Equal(ConnectionStatus(store.ConnectionPending), s.connectionOf(bob, askedBob["connection_id"].(string)).Status,
		"Alice's consent connected nothing of Bob's")
}

// connector stores a connector of the suite's app at the fake that takes oauth2_code, and
// the grant of its echo at the digest the fake lists, read through a twin connection of the
// app's that takes a bearer token.
func (s *ChatLoginsSuite) connector() (string, map[string]any) {
	id := "custom_crm" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	s.define(id, oauth2code.Name, `
client:
  registration: [operator]
  auth_method: client_secret_post
  env: FAKE
scopes:
  list: [chat:write]
`)
	twin := id + "twin"
	s.define(twin, bearer.Name, "")
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned(twin), &created))
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials",
		map[string]any{"expected_revision": 1, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/connections/"+created.ID+"/validate", nil, &validation))
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodGet, "/v1/agents/connections/"+created.ID+"/tools", nil, &tools))
	for _, tool := range tools.Tools {
		if tool.Name == "echo" {
			return id, map[string]any{"name": tool.Name, "schema_digest": tool.SchemaDigest}
		}
	}
	s.Require().FailNow("the fake lists no echo")
	return "", nil
}

func (s *ChatLoginsSuite) define(id, scheme, extra string) {
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Fake
endpoints:
  authorize: ` + s.provider.URL + fakeprovider.PathAuthorize + `
  token: ` + s.provider.URL + fakeprovider.PathToken + `
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [` + scheme + `]
sources:
  - kind: mcp
    endpoint: mcp
` + extra))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
}

// connected is a connection of user's to connector that a consent their backend began
// connected.
func (s *ChatLoginsSuite) connected(user *testClient, connector string) string {
	as := s.serverClient.actingFor(user)
	var created Connection
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/connections", userOwned(connector, user), &created))
	var started Authorization
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/connections/"+created.ID+"/authorizations", nil, &started))
	b := newBrowser(&s.RouterSuite, s.provider)
	s.Require().Equal(s.landing(created.ID), b.finish(s.consent(b.handOff(started))).Header.Get("Location"))
	return created.ID
}

// config is an agent config on the logging-in model binding connector as crm, the caller's
// own connection, optional, granting echo.
func (s *ChatLoginsSuite) config(connector string, grant map[string]any) string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "logins-" + s.utils.uuid(), "llm": "logging-in/login-model",
		"connectors": []map[string]any{{"name": "crm", "connector_id": connector,
			"connection": map[string]any{"type": "session"}, "tools": []map[string]any{grant}}},
	}, &created))
	return created.Id
}

// session is a text session of config choosing connections by alias.
func (s *ChatLoginsSuite) session(config string, chosen map[string]string) CreateSessionRequest {
	selections := []SessionConnectorBinding{}
	for alias, id := range chosen {
		selections = append(selections, SessionConnectorBinding{Name: alias, ConnectionId: id})
	}
	return CreateSessionRequest{ConfigId: &config, Text: pointerTo(true), ConnectorBindings: &selections}
}

// ask is the caller's backend asking the session something, by command.
func (s *ChatLoginsSuite) ask(id string) CommandReceipt {
	return s.askAs(s.client, id)
}

func (s *ChatLoginsSuite) askAs(user *testClient, id string) CommandReceipt {
	var receipt CommandReceipt
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(user).do(http.MethodPost, "/v1/agents/sessions/"+id+"/respond",
		RespondRequest{Text: "Tell Nash a joke on the crm", CommandId: pointerTo(s.utils.uuid())}, &receipt))
	return receipt
}

// attemptKind is the kind the consent's attempt was stored with.
func (s *ChatLoginsSuite) attemptKind(id string) string {
	var kind string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT kind FROM connector_authorization_attempts WHERE id = ?", id).Scan(&kind))
	return kind
}

// loginOn is the first connector_authorization attachment with status a conversation_updated
// frame on events carries, read through the tool_ran of the call that asked for it. What the
// model read of that call names neither the launch URL nor the handoff token.
func (s *ChatLoginsSuite) loginOn(events *websocket.Conn, status string) map[string]any {
	var found, ran map[string]any
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for found == nil || ran == nil {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		switch frame["type"] {
		case "conversation_updated":
			if login := loginIn(frame, status); login != nil && found == nil {
				found = login
			}
		case "tool_ran":
			if frame["tool"] == loginCallTool {
				ran = frame
			}
		}
	}
	read, _ := ran["result"].(string)
	s.Contains(read, `"status":"authorization_required"`)
	s.NotContains(read, found["launch_url"])
	s.NotContains(read, found["handoff_token"])
	return found
}

// loginIn is the connector_authorization attachment with status in a conversation_updated
// frame, or nil.
func loginIn(frame map[string]any, status string) map[string]any {
	message, _ := frame["message"].(map[string]any)
	attachments, _ := message["attachments"].([]any)
	for _, attachment := range attachments {
		found, _ := attachment.(map[string]any)
		if found["type"] == conversation.ConnectorAuthorizationType && (found["status"] == status || status == "" && found["status"] == nil) {
			return found
		}
	}
	return nil
}

// carriedOn is what events showed between a login finishing and the tool it was asked for
// running: the tool_ran frame, the login statuses the replies showed, and the roles of every
// message written by a person.
type carriedOn struct {
	ran      map[string]any
	statuses []string
	users    []string
}

func (s *ChatLoginsSuite) carryOn(events *websocket.Conn) carriedOn {
	var seen carriedOn
	s.Require().NoError(events.SetReadDeadline(time.Now().Add(settleFor)))
	for {
		var frame map[string]any
		s.Require().NoError(events.ReadJSON(&frame))
		switch frame["type"] {
		case "conversation_updated":
			message, _ := frame["message"].(map[string]any)
			if message["role"] == "user" {
				seen.users = append(seen.users, message["id"].(string))
			}
			if found := loginIn(frame, "connected"); found != nil && !slices.Contains(seen.statuses, "connected") {
				seen.statuses = append(seen.statuses, "connected")
				s.Nil(found["handoff_token"], "a finished login drops its handoff token")
			}
		case "tool_ran":
			if frame["tool"] == loginCallTool {
				seen.ran = frame
				return seen
			}
		}
	}
}

// started is the attachment as the consent createAuthorization would have answered.
func (s *ChatLoginsSuite) started(asked map[string]any) Authorization {
	return Authorization{ID: asked["authorization_id"].(string), LaunchURL: asked["launch_url"].(string),
		HandoffToken: asked["handoff_token"].(string)}
}

// connectionOf is the connection as user's backend reads it, which only the owner may.
func (s *ChatLoginsSuite) connectionOf(user *testClient, id string) Connection {
	var connection Connection
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(user).do(http.MethodGet, "/v1/agents/connections/"+id, nil, &connection))
	return connection
}

func (s *ChatLoginsSuite) consent(authorize string) *url.URL {
	callback, err := s.provider.Consent(authorize)
	s.Require().NoError(err)
	s.Require().True(strings.HasPrefix(callback.String(), consentPublicURL+ConnectorCallbackPath+"?"), callback.String())
	return callback
}

func (s *ChatLoginsSuite) landing(id string) string {
	return consentDashboard + "?" + url.Values{"connection_id": {id}, "status": {consentConnected}}.Encode()
}

// loggingInLLM runs crm's echo through call_tool whenever the last thing it was handed is a
// person's message and it was offered call_tool, and otherwise says it is done. A follow-up
// after a login reaches it as a person's message, so it runs the tool again then.
type loggingInLLM struct {
	mu sync.Mutex
}

func (m *loggingInLLM) Start(context.Context) error { return nil }

func (m *loggingInLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: m.Provider(), Model: m.Model()})
	offered := slices.ContainsFunc(params.Tools, func(tool llm.Tool) bool { return tool.Name == loginCallTool })
	if last := len(params.Input) - 1; offered && last >= 0 && params.Input[last].Role == llm.User {
		script.OutputText("Let me tell Nash.")
		script.ToolCalls(llm.ToolCall{ID: store.NewID(), Name: loginCallTool,
			Arguments: `{"tool":"echo","arguments":{"text":"` + connectorEchoText + `"}}`})
	} else {
		script.OutputText("Done.")
	}
	script.Done()
	return script.Stream(), nil
}

func (m *loggingInLLM) Provider() string               { return "logging-in" }
func (m *loggingInLLM) Model() string                  { return "login-model" }
func (m *loggingInLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (m *loggingInLLM) Close() error                   { return nil }
