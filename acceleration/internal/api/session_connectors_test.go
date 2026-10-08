//go:build integration

package api

import (
	"context"
	"maps"
	"net/http"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/getstream-go/v5"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// connectorEcho is the connector tool the connecting model reaches for on its first turn:
// the fake provider's echo, under the binding called crm. It says back connectorEchoText.
const (
	connectorEcho     = "crm" + mcp.Separator + "echo"
	connectorEchoText = "said through the connector"
)

// SessionConnectorsSuite is connectors in a session, end to end: a connection made, given
// its credential and validated through the API, bound on an agent config through the API,
// chosen when the session is created, and called by the model. The fake provider (T5) is
// the MCP server and issues the bearer token its MCP endpoint takes, so a call that reaches
// it went with the connection's credential.
type SessionConnectorsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued. Synthetic, fresh per suite.
	token string
}

func TestSessionConnectorsSuite(t *testing.T) {
	runSuite(t, new(SessionConnectorsSuite))
}

func (s *SessionConnectorsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
}

func (s *SessionConnectorsSuite) SetupTest() {
	s.useFixture("standard")
}

// TestAModelsToolCallReachesTheProviderWithTheConnectionsCredential: the whole path. The
// end user's own connection, chosen when they open the session, answers the model.
func (s *SessionConnectorsSuite) TestAModelsToolCallReachesTheProviderWithTheConnectionsCredential() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo))
	before := s.provider.Hits(fakeprovider.PathMCP)

	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	// The user watches; their backend, acting for them, asks (respond is server-side only).
	ran := s.toolRanOn(s.serverClient.actingFor(s.client), s.client.opens("/v1/agents/sessions/"+opened.Id+"/events"), opened.Id)

	s.Equal(connectorEcho, ran["tool"])
	s.Equal(connectorEchoText, ran["result"])
	s.Empty(ran["error"])
	s.Greater(s.provider.Hits(fakeprovider.PathMCP), before)
}

// toolStartedOn is the tool_started frame of a session of a config binding crm with policy,
// none when policy is nil, once the model reached for crm__echo.
func (s *SessionConnectorsSuite) toolStartedOn(policy map[string]any) map[string]any {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	binding := s.binding("crm", connector, "session", "", echo)
	if policy != nil {
		binding["policy"] = policy
	}
	opened := s.client.createSession(s.session(s.config(binding), map[string]string{"crm": mine}))
	events := s.client.opens("/v1/agents/sessions/" + opened.Id + "/events")
	s.Require().Equal(http.StatusOK, s.serverClient.actingFor(s.client).do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())}, nil))
	return s.await(events, "tool_started")
}

// TestABindingsPreSpeechIsOnItsToolStarted: the policy written through the config API
// reaches the session's tool_started.
func (s *SessionConnectorsSuite) TestABindingsPreSpeechIsOnItsToolStarted() {
	started := s.toolStartedOn(map[string]any{"pre_speech": "Let me look in the CRM."})

	s.Equal(connectorEcho, started["tool"])
	s.Equal("Let me look in the CRM.", started["pre_speech"])
}

// TestABindingWithoutAPolicyHasTheToolStartedOfBefore: the same keys as base sends, no more.
func (s *SessionConnectorsSuite) TestABindingWithoutAPolicyHasTheToolStartedOfBefore() {
	started := s.toolStartedOn(nil)

	s.ElementsMatch([]string{"type", "tool_call_id", "tool", "turn_id", "started_at"}, slices.Collect(maps.Keys(started)))
}

func (s *SessionConnectorsSuite) TestAFixedAliasCannotBeGivenAConnection() {
	connector := s.connector()
	app, echo := s.connection(nil, connector)
	mine, _ := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "fixed", app, echo))

	status, failure := s.client.failure(http.MethodPost, "/v1/agents/sessions", s.session(config, map[string]string{"crm": mine}))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `connector "crm" has the agent config's fixed connection, which a session cannot replace`)
}

// TestAliceCannotOpenASessionThroughBobsConnection: Bob's connection id is answered as one
// that does not exist, and nothing reaches the provider with his credential.
func (s *SessionConnectorsSuite) TestAliceCannotOpenASessionThroughBobsConnection() {
	connector := s.connector()
	bob := s.data.createUser()
	bobs, echo := s.connection(bob, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo, true))
	before := s.provider.Hits(fakeprovider.PathMCP)

	status, failure := s.client.failure(http.MethodPost, "/v1/agents/sessions", s.session(config, map[string]string{"crm": bobs}))
	_, missing := s.client.failure(http.MethodPost, "/v1/agents/sessions", s.session(config, map[string]string{"crm": "no-such-connection"}))

	s.Equal(http.StatusBadRequest, status)
	s.Contains(failure, `required connector "crm" cannot be used: connection_unavailable`)
	s.Equal(missing, failure)
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// TestAStoredSelectionHoldsNoCredential: the session's row keeps the alias and the
// connection's id, nothing else.
func (s *SessionConnectorsSuite) TestAStoredSelectionHoldsNoCredential() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo))

	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))

	stored := s.storedSelections(opened.Id)
	s.JSONEq(`[{"name": "crm", "connection_id": "`+mine+`"}]`, stored)
	s.NotContains(stored, s.token)
}

// TestAForkOfAnEndedSessionDropsASelectionItsConfigNoLongerDeclares: the fork reads its
// parent's selections from the row and re-resolves them against the config as it is now.
func (s *SessionConnectorsSuite) TestAForkOfAnEndedSessionDropsASelectionItsConfigNoLongerDeclares() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo))
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config,
		map[string]any{"connectors": []map[string]any{s.binding("notes", connector, "session", "", echo)}}, nil))

	var forked Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)}, &forked))
	dropped := s.await(s.client.opens("/v1/agents/sessions/"+forked.Id+"/events"), "connector_unavailable")

	s.Equal("crm", dropped["name"])
	s.Equal("selection_dropped", dropped["reason"])
	s.JSONEq(`[]`, s.storedSelections(forked.Id), "the fork keeps what it re-resolved")
}

// TestAForkForAnotherUserDoesNotUseTheParentsConnection: the backend may fork any session of
// its app, and a fork it asks for in Bob's name re-resolves Alice's selection against Bob,
// whose connection it is not.
func (s *SessionConnectorsSuite) TestAForkForAnotherUserDoesNotUseTheParentsConnection() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo))
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)
	bob := s.serverClient.actingFor(s.data.createUser())

	var forked Session
	s.Require().Equal(http.StatusCreated, bob.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)}, &forked))
	left := s.await(bob.opens("/v1/agents/sessions/"+forked.Id+"/events"), "connector_unavailable")

	s.Equal("crm", left["name"])
	s.Equal("connection_unavailable", left["reason"])
}

// TestAStoppedChatReopensWithTheCallersConnection: a message to a chat whose session ended
// carries it on from its row, with the connection its caller chose. Without it a required
// binding would refuse the message with no_selection.
func (s *SessionConnectorsSuite) TestAStoppedChatReopensWithTheCallersConnection() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo, true))
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)

	status, failure := s.serverClient.actingFor(s.client).failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())})

	s.Equal(http.StatusOK, status, failure)
	s.JSONEq(`[{"name": "crm", "connection_id": "`+mine+`"}]`, s.storedSelections(opened.Id))
}

// TestAStoppedChatWhoseConfigDroppedTheBindingIsStillAnswered: the config no longer declares
// the alias the caller chose, so the reopened chat drops that selection, says so, and answers,
// as it did before the reopen carried selections.
func (s *SessionConnectorsSuite) TestAStoppedChatWhoseConfigDroppedTheBindingIsStillAnswered() {
	connector := s.connector()
	mine, echo := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "session", "", echo))
	opened := s.client.createSession(s.session(config, map[string]string{"crm": mine}))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config,
		map[string]any{"connectors": []map[string]any{}}, nil))

	status, failure := s.serverClient.actingFor(s.client).failure(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())})

	s.Require().Equal(http.StatusOK, status, failure)
	dropped := s.await(s.client.opens("/v1/agents/sessions/"+opened.Id+"/events"), "connector_unavailable")
	s.Equal("crm", dropped["name"])
	s.Equal("selection_dropped", dropped["reason"])
}

// TestAForkWhoseConfigCannotBeReadFails: the fork does not go ahead without the bindings it
// could not read. The config's row is broken so that reading it fails.
func (s *SessionConnectorsSuite) TestAForkWhoseConfigCannotBeReadFails() {
	config := s.config()
	opened := s.client.createSession(s.session(config, nil))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)
	_, err := s.store.DB().ExecContext(context.Background(),
		`UPDATE agent_configs SET connectors = '{"not": "a list"}'::jsonb WHERE id = ?`, config)
	s.Require().NoError(err)

	status := s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork", ForkSessionRequest{Messages: pointerTo(false)}, nil)

	s.Equal(http.StatusInternalServerError, status)
}

// TestAForkOfASessionWhoseConfigWasDeletedGoesAheadAsBefore: a config that is gone binds
// nothing, and its fork is made as one was before connectors existed.
func (s *SessionConnectorsSuite) TestAForkOfASessionWhoseConfigWasDeletedGoesAheadAsBefore() {
	config := s.config()
	opened := s.client.createSession(s.session(config, nil))
	s.storedSelections(opened.Id)
	s.client.stopSession(opened.Id)
	s.Require().Equal(http.StatusNoContent, s.serverClient.do(http.MethodDelete, "/v1/agents/configs/"+config, nil, nil))

	var forked Session
	s.Require().Equal(http.StatusCreated, s.client.do(http.MethodPost, "/v1/agents/sessions/"+opened.Id+"/fork",
		ForkSessionRequest{Messages: pointerTo(false)}, &forked))

	s.Equal(opened.Id, value(forked.ForkedFrom))
}

// TestASlackThreadSessionUsesTheAppsConnectionsOnly: everyone in the Slack thread writes in
// its thread channel, so a session held there uses the app's connection and never one
// person's, even one the request chose. The model's call goes through the app's binding. The
// backend opens it, as a worker does (SlackChannelSuite): a thread channel is no one end
// user's conversation.
func (s *SessionConnectorsSuite) TestASlackThreadSessionUsesTheAppsConnectionsOnly() {
	connector := s.connector()
	app, echo := s.connection(nil, connector)
	mine, _ := s.connection(s.client, connector)
	config := s.config(s.binding("crm", connector, "fixed", app, echo), s.binding("mine", connector, "session", "", echo))
	channel := conversation.ThreadChannelPrefix + s.utils.uuid()
	s.threadChannel(channel, connector, app, config)
	request := s.session(config, map[string]string{"mine": mine})
	request.AgentId = &channel

	opened := s.serverClient.createSession(request)
	events := s.serverClient.opens("/v1/agents/sessions/" + opened.Id + "/events")
	left := s.await(events, "connector_unavailable")
	ran := s.toolRanOn(s.serverClient, events, opened.Id)

	s.Equal("agent:"+channel, value(opened.ConversationId))
	s.Equal("mine", left["name"])
	s.Equal("shared_session", left["reason"])
	s.Equal(connectorEchoText, ran["result"], "the app's binding answers")
}

// threadChannel is a Slack thread's thread channel, linked and created in Stream Chat as the
// channel bridge does on the thread's first message (channelbridge.Bridge.writeInto), with
// two people of the thread writing in it.
func (s *SessionConnectorsSuite) threadChannel(channel, connector, connection, config string) {
	ctx := context.Background()
	_, err := s.store.LinkChannelThread(ctx, &store.ChannelThread{ChannelID: channel, CustomerID: s.customerID(),
		ConnectorID: connector, ProviderUnitID: "T0000TEAM", ThreadKey: s.utils.uuid(), ConnectionID: connection})
	s.Require().NoError(err)
	alice, bob := "slack-U0000ALICE", "slack-U0000BOB"
	_, err = s.chat.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, channel,
		&getstream.GetOrCreateChannelRequest{Data: &getstream.ChannelInput{CreatedByID: &alice, Custom: map[string]any{
			"agent_config_id": config, conversation.CustomerField: s.customerID(), "support_agent_id": channel,
		}}})
	s.Require().NoError(err)
	for _, author := range []string{alice, bob} {
		text := "is the build green?"
		_, err = s.chat.Client.Chat().SendMessage(ctx, chatlog.ChannelType, channel,
			&getstream.SendMessageRequest{Message: getstream.MessageRequest{Text: &text, UserID: &author}})
		s.Require().NoError(err)
	}
}

// connector stores a connector of the suite's app at the fake, taking a bearer token.
func (s *SessionConnectorsSuite) connector() string {
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
// when user is nil and otherwise user's, made by the backend acting for them, and the grant
// of the fake's echo at the digest its validate listed.
func (s *SessionConnectorsSuite) connection(user *testClient, connector string) (string, map[string]any) {
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
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections/"+created.ID+"/tools", nil, &tools))
	for _, tool := range tools.Tools {
		if tool.Name == "echo" {
			return created.ID, map[string]any{"name": tool.Name, "schema_digest": tool.SchemaDigest}
		}
	}
	s.Require().FailNow("the connection lists no echo")
	return "", nil
}

// binding is an agent config's connector binding granting one tool, required when asked.
func (s *SessionConnectorsSuite) binding(alias, connector, selection, connection string, grant map[string]any, required ...bool) map[string]any {
	chosen := map[string]any{"type": selection}
	if connection != "" {
		chosen["connection_id"] = connection
	}
	return map[string]any{"name": alias, "connector_id": connector, "connection": chosen,
		"tools": []map[string]any{grant}, "required": len(required) > 0 && required[0]}
}

// config is a new agent config of the suite's app, on the connecting model, binding
// bindings through the config endpoint (T20).
func (s *SessionConnectorsSuite) config(bindings ...map[string]any) string {
	var created AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "connectors-" + s.utils.uuid(), "llm": "connecting/connector-model", "connectors": bindings,
	}, &created))
	return created.Id
}

// session is a text session of config choosing connections by alias.
func (s *SessionConnectorsSuite) session(config string, chosen map[string]string) CreateSessionRequest {
	selections := []SessionConnectorBinding{}
	for alias, id := range chosen {
		selections = append(selections, SessionConnectorBinding{Name: alias, ConnectionId: id})
	}
	return CreateSessionRequest{ConfigId: &config, Text: pointerTo(true), ConnectorBindings: &selections}
}

// toolRanOn is the tool_ran frame on events of the first turn as asks the session for. A
// persistent conversation is reached only by whose it is (canReadSession), and asked by
// command, as a personal one must be.
func (s *SessionConnectorsSuite) toolRanOn(as *testClient, events *websocket.Conn, id string) map[string]any {
	s.Require().Equal(http.StatusOK, as.do(http.MethodPost, "/v1/agents/sessions/"+id+"/respond",
		RespondRequest{Text: "ask the crm", CommandId: pointerTo(s.utils.uuid())}, nil))
	return s.await(events, "tool_ran")
}

// storedSelections is the session row's connector_selections as stored, once it is written.
func (s *SessionConnectorsSuite) storedSelections(id string) string {
	var stored string
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(context.Background(),
			"SELECT connector_selections::text FROM agent_sessions WHERE id = ?", id).Scan(&stored) == nil
	}, settleFor, 20*time.Millisecond)
	return stored
}
