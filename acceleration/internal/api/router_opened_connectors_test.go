//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/bearer"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/sources/mcp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// RouterOpenedConnectorsSuite pins AI-1000 (Kanat's decision of 2026-10-09): a session the
// router opens itself, for a plugin event (pluginevents.Service.run) or for a message on a
// channel line (channels.Service.answer), gets the implied connection of AI-994 as a session
// an app opens does. A binding of connection type session that names no connection uses the
// caller's only connected connection to its connector (session.impliedSelection), with the
// trust user_plugins already gave these callers: the end user whose login subscribed to the
// event, or the number that wrote (phone:+E164) or the user it was linked to. Two connected
// connections, an app-owned one, or no verified caller imply nothing.
//
// The model is the logging-in one (loggingInLLM): it runs crm's echo when it is offered, so an
// echo result in the session is the implied connection having been used.
type RouterOpenedConnectorsSuite struct {
	RouterSuite
	provider *fakeprovider.Server
	// token is an access token the fake issued, which every bearer connection here holds.
	token string
	// sentry stands in for the plugin server events are subscribed on.
	sentry *eventServer
	// meta stands in for Meta's Cloud API, which answers on a WhatsApp line go to.
	meta *metaAPI
}

func TestRouterOpenedConnectorsSuite(t *testing.T) {
	runSuite(t, new(RouterOpenedConnectorsSuite))
}

func (s *RouterOpenedConnectorsSuite) SetupSuite() {
	s.provider = fakeprovider.New(s.T(), fakeprovider.ClientCredentials)
	s.connectors = core.Registry{
		Schemes:     map[string]core.Scheme{bearer.Name: bearer.New()},
		ToolSources: map[string]core.ToolSource{mcp.Kind: mcp.New()},
	}
	s.connectorHTTP = s.provider.Client()
	s.sentry = &eventServer{subscriptions: map[string]eventSubscription{}}
	s.pluginMCP = httptest.NewTLSServer(http.HandlerFunc(s.sentry.serve))
	s.T().Cleanup(s.pluginMCP.Close)
	s.meta = &metaAPI{}
	s.channelAPI = httptest.NewTLSServer(http.HandlerFunc(s.meta.serve))
	s.T().Cleanup(s.channelAPI.Close)
	s.RouterSuite.SetupSuite()
	s.token = issuedToken(&s.RouterSuite, s.provider)
}

// SetupTest gives each test an app of its own, so the only connections to its connector are
// the ones it makes.
func (s *RouterOpenedConnectorsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *RouterOpenedConnectorsSuite) TestAPluginEventUsesTheSubscribersOneConnectedConnection() {
	user := s.utils.uuid()
	connector := s.connector()
	mine := s.userConnection(connector, user)
	config := s.subscribed(connector, s.grant(mine), &user)

	s.deliverEvent(config)

	s.Eventually(func() bool { return s.echoed(config.Name, user) == 1 }, settleFor, 50*time.Millisecond,
		"the tool ran on the subscriber's one connection, with no connector_bindings")
}

func (s *RouterOpenedConnectorsSuite) TestAPluginEventImpliesNothingWhenTheSubscriberHasTwoConnected() {
	user := s.utils.uuid()
	connector := s.connector()
	first := s.userConnection(connector, user)
	s.userConnection(connector, user)
	config := s.subscribed(connector, s.grant(first), &user)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.deliverEvent(config)

	s.settled(config.Name, user)
	s.Zero(s.echoed(config.Name, user))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "neither connection reached the provider")
}

// An app-owned connection is the app's, not the subscriber's: a session binding never
// implies it.
func (s *RouterOpenedConnectorsSuite) TestAPluginEventImpliesNoAppOwnedConnection() {
	user := s.utils.uuid()
	connector := s.connector()
	config := s.subscribed(connector, s.grant(s.appConnection(connector)), &user)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.deliverEvent(config)

	s.settled(config.Name, user)
	s.Zero(s.echoed(config.Name, user))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// An event subscribed with the app's login opens a conversation as nobody (no Caller), so
// there is no verified person whose connection it could be. The app's own connection is not
// used either.
func (s *RouterOpenedConnectorsSuite) TestAPluginEventTheAppSubscribedImpliesNothing() {
	connector := s.connector()
	config := s.subscribed(connector, s.grant(s.appConnection(connector)), nil)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.deliverEvent(config)

	s.settled(config.Name, "")
	s.Zero(s.echoed(config.Name, ""))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// The number that wrote is the caller, as phone:+E164, and its one connected connection is
// used as the one of a person signed into an app would be.
func (s *RouterOpenedConnectorsSuite) TestAMessageOnALineUsesTheNumbersOneConnectedConnection() {
	writer := s.writer()
	connector := s.connector()
	mine := s.userConnection(connector, "phone:+"+writer)
	line, config := s.lineAgent(connector, s.grant(mine), "")

	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, "tell Nash"))

	s.Eventually(func() bool { return s.echoed(config.Name, "phone:+"+writer) == 1 }, settleFor, 50*time.Millisecond,
		"the tool ran on the number's one connection")
}

// With identity link, the caller is the end user the number was linked to, and that user's
// one connected connection is used once the number is theirs.
func (s *RouterOpenedConnectorsSuite) TestAMessageFromALinkedNumberUsesTheLinkedUsersOneConnectedConnection() {
	owner, writer := s.utils.uuid(), s.writer()
	connector := s.connector()
	mine := s.userConnection(connector, owner)
	line, config := s.lineAgent(connector, s.grant(mine), store.ChannelIdentityLink)
	var minted ChannelLink
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels/links",
		LinkChannelRequest{ConfigId: config.Id, UserId: owner}, &minted))
	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, minted.Code))
	s.Eventually(func() bool { return len(s.meta.to(writer)) == 1 }, settleFor, 50*time.Millisecond, "the claim is answered")

	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, "tell Nash"))

	s.Eventually(func() bool { return s.echoed(config.Name, owner) == 1 }, settleFor, 50*time.Millisecond,
		"the tool ran on the linked user's one connection")
}

// A number not linked yet is nobody's: it is asked for a code, no conversation opens, and the
// connection of the user it may later be linked to is not used.
func (s *RouterOpenedConnectorsSuite) TestAMessageFromANumberNotLinkedYetImpliesNothing() {
	owner, writer := s.utils.uuid(), s.writer()
	connector := s.connector()
	mine := s.userConnection(connector, owner)
	line, config := s.lineAgent(connector, s.grant(mine), store.ChannelIdentityLink)
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, "tell Nash"))

	s.Eventually(func() bool { return len(s.meta.to(writer)) == 1 }, settleFor, 50*time.Millisecond)
	s.Contains(s.meta.to(writer)[0], "code")
	s.Empty(s.sessionsOf(config.Name, ""), "no conversation opened")
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

func (s *RouterOpenedConnectorsSuite) TestAMessageOnALineImpliesNothingWhenTheNumberHasTwoConnected() {
	writer := s.writer()
	connector := s.connector()
	first := s.userConnection(connector, "phone:+"+writer)
	s.userConnection(connector, "phone:+"+writer)
	line, config := s.lineAgent(connector, s.grant(first), "")
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, "tell Nash"))

	s.settled(config.Name, "phone:+"+writer)
	s.Zero(s.echoed(config.Name, "phone:+"+writer))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP), "neither connection reached the provider")
}

func (s *RouterOpenedConnectorsSuite) TestAMessageOnALineImpliesNoAppOwnedConnection() {
	writer := s.writer()
	connector := s.connector()
	line, config := s.lineAgent(connector, s.grant(s.appConnection(connector)), "")
	before := s.provider.Hits(fakeprovider.PathMCP)

	s.Require().Equal(http.StatusAccepted, s.deliverMessage(line, writer, "tell Nash"))

	s.settled(config.Name, "phone:+"+writer)
	s.Zero(s.echoed(config.Name, "phone:+"+writer))
	s.Equal(before, s.provider.Hits(fakeprovider.PathMCP))
}

// connector stores a connector of the app's at the fake that takes a bearer token.
func (s *RouterOpenedConnectorsSuite) connector() string {
	id := "custom_crm" + strings.ReplaceAll(s.utils.uuid(), "-", "")
	manifest, err := core.ParseManifest([]byte(`
id: ` + id + `
revision: 1
name: Fake
endpoints:
  mcp: ` + s.provider.URL + fakeprovider.PathMCP + `
schemes: [` + bearer.Name + `]
sources:
  - kind: mcp
    endpoint: mcp
`))
	s.Require().NoError(err)
	_, err = s.store.CreateConnectorDefinition(context.Background(), s.customerID(), manifest)
	s.Require().NoError(err)
	return id
}

// userConnection is a connected connection of userID's to connector holding the fake's token.
func (s *RouterOpenedConnectorsSuite) userConnection(connector, userID string) string {
	user := &testClient{userID: userID}
	return s.withToken(s.serverClient.actingFor(user), userOwned(connector, user))
}

// appConnection is a connected connection of the app's to connector holding the fake's token.
func (s *RouterOpenedConnectorsSuite) appConnection(connector string) string {
	return s.withToken(s.serverClient, appOwned(connector))
}

func (s *RouterOpenedConnectorsSuite) withToken(as *testClient, sent map[string]any) string {
	sent["auth_scheme"] = bearer.Name
	var created Connection
	s.Require().Equal(http.StatusCreated, as.do(http.MethodPost, "/v1/agents/connections", sent, &created))
	s.Require().Equal(http.StatusOK, as.do(http.MethodPut, "/v1/agents/connections/"+created.ID+"/credentials",
		map[string]any{"expected_revision": created.Revision, "values": map[string]string{bearer.SuppliedToken: s.token}}, nil))
	stored, err := s.store.ConnectorConnection(context.Background(), s.customerID(), created.ID)
	s.Require().NoError(err)
	s.Require().Equal(store.ConnectionConnected, stored.Status)
	return created.ID
}

// grant is echo at the schema digest the fake lists, read through connection once a validate
// listed its tools.
func (s *RouterOpenedConnectorsSuite) grant(connection string) map[string]any {
	connected, err := s.store.ConnectorConnection(context.Background(), s.customerID(), connection)
	s.Require().NoError(err)
	as := s.serverClient
	if connected.OwnerType == store.OwnerUser {
		as = as.actingFor(&testClient{userID: connected.OwnerID})
	}
	var validation ConnectionValidation
	s.Require().Equal(http.StatusOK, as.do(http.MethodPost, "/v1/agents/connections/"+connection+"/validate", nil, &validation))
	var tools ConnectionTools
	s.Require().Equal(http.StatusOK, as.do(http.MethodGet, "/v1/agents/connections/"+connection+"/tools", nil, &tools))
	for _, tool := range tools.Tools {
		if tool.Name == "echo" {
			return map[string]any{"name": tool.Name, "schema_digest": tool.SchemaDigest}
		}
	}
	s.Require().FailNow("the fake lists no echo")
	return nil
}

// binding is crm, the caller's own connection to connector (connection type session),
// optional, granting echo.
func binding(connector string, grant map[string]any) []map[string]any {
	return []map[string]any{{"name": "crm", "connector_id": connector,
		"connection": map[string]any{"type": "session"}, "tools": []map[string]any{grant}}}
}

// subscribed is a text agent on the logging-in model binding crm, declaring sentry's
// issue.created, logged into Sentry by the app or by the end user named, and subscribed.
func (s *RouterOpenedConnectorsSuite) subscribed(connector string, grant map[string]any, userID *string) AgentConfig {
	request := map[string]any{
		"name": "watcher-" + s.utils.uuid(), "mode": "text", "llm": "logging-in/login-model",
		"plugin_events": []map[string]any{{"plugin": "sentry", "event": "issue.created",
			"arguments": map[string]any{"project": "web"}}},
		"connectors": binding(connector, grant),
	}
	if userID == nil {
		request["plugins"] = []map[string]any{{"name": "sentry", "user": false}}
	} else {
		request["plugins"] = []map[string]any{{"name": "sentry", "user": true}}
	}
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &config))

	login := store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: config.Id, PluginID: "sentry",
		Status: store.PluginConnected, AccessToken: "token-" + s.utils.uuid(),
	}
	if userID != nil {
		login.UserID = *userID
	}
	s.sentry.name(login.AccessToken, config.Name)
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &login))
	stored, err := s.store.AgentConfig(context.Background(), s.customerID(), config.Id)
	s.Require().NoError(err)
	s.events.Reconcile(context.Background(), stored)
	return config
}

// deliverEvent posts one issue.created to config's subscription, signed as Sentry signs it.
func (s *RouterOpenedConnectorsSuite) deliverEvent(config AgentConfig) {
	id := "evt-" + s.utils.uuid()
	body, err := json.Marshal(map[string]any{
		"eventId": id, "name": "issue.created", "timestamp": time.Now().UTC().Format(time.RFC3339),
		"data": map[string]any{"project": "web", "title": "TypeError in checkout"}, "cursor": nil,
	})
	s.Require().NoError(err)
	sub := s.sentry.subscription(s.T(), config.Name)
	response, err := signedPost(sub.url, sub.secret, id, body)
	s.Require().NoError(err)
	defer response.Body.Close()
	s.Require().Equal(http.StatusAccepted, response.StatusCode)
}

// lineAgent is a WhatsApp line connected for a number nobody else in the run uses, and a text
// agent on the logging-in model answering on it, binding crm, identifying a writer the way
// named.
func (s *RouterOpenedConnectorsSuite) lineAgent(connector string, grant map[string]any, identity string) (ChannelAccount, AgentConfig) {
	number := "+1555" + digits(s.utils.uuid())
	var line ChannelAccount
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost, "/v1/agents/channels",
		ConnectChannelRequest{
			Kind:      string(channels.WhatsApp),
			Number:    number,
			AccountId: pointerTo(strings.TrimPrefix(number, "+")),
			Token:     pointerTo("meta-token"),
			Signing:   pointerTo(appSecret),
			Challenge: pointerTo("verify-" + s.app.key),
		}, &line))
	answering := map[string]any{"whatsapp": map[string]any{"number": number}}
	if identity != "" {
		answering["identity"] = identity
	}
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", map[string]any{
		"name": "channelled-" + s.utils.uuid(), "mode": "text", "llm": "logging-in/login-model",
		"channels": answering, "connectors": binding(connector, grant),
	}, &config))
	return line, config
}

// deliverMessage posts one text from a number to a line's webhook, signed as Meta signs it.
func (s *RouterOpenedConnectorsSuite) deliverMessage(line ChannelAccount, from, text string) int {
	body, err := json.Marshal(map[string]any{"entry": []any{map[string]any{
		"changes": []any{map[string]any{"value": map[string]any{
			"metadata": map[string]any{"display_phone_number": "15556325550"},
			"contacts": []any{map[string]any{"wa_id": from, "profile": map[string]any{"name": "Thierry"}}},
			"messages": []any{map[string]any{
				"id": "msg-" + s.utils.uuid(), "from": from, "type": "text", "text": map[string]any{"body": text},
			}},
		}}},
	}}})
	s.Require().NoError(err)
	mac := hmac.New(sha256.New, []byte(appSecret))
	mac.Write(body)
	request, err := http.NewRequest(http.MethodPost, s.server.URL+line.WebhookUrl, bytes.NewReader(body))
	s.Require().NoError(err)
	request.Header.Set("X-Hub-Signature-256", "sha256="+hex.EncodeToString(mac.Sum(nil)))
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

// writer is somebody's phone, as WhatsApp sends it: digits, no plus.
func (s *RouterOpenedConnectorsSuite) writer() string { return "1347" + digits(s.utils.uuid()) }

// sessionsOf are the conversations of agent held as userID, empty for one held as nobody.
func (s *RouterOpenedConnectorsSuite) sessionsOf(agent, userID string) []store.AgentSession {
	found, err := s.store.QuerySessions(context.Background(), s.customerID(),
		store.SessionFilter{AgentName: agent, UserID: userID})
	s.Require().NoError(err)
	return found
}

// settled waits for agent's one conversation as userID to have closed, which both callers do
// once the turn settled.
func (s *RouterOpenedConnectorsSuite) settled(agent, userID string) {
	s.Require().Eventually(func() bool {
		found := s.sessionsOf(agent, userID)
		return len(found) == 1 && found[0].ClosedAt != nil
	}, settleFor, 50*time.Millisecond, "the conversation %s opened as %q did not close", agent, userID)
}

// echoed is how many times crm's echo ran in agent's conversations as userID.
func (s *RouterOpenedConnectorsSuite) echoed(agent, userID string) int {
	ran := 0
	for _, session := range s.sessionsOf(agent, userID) {
		var count int
		s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
			"SELECT count(*) FROM agent_response_items WHERE session_id = ? AND kind = ? AND tool_name = ? AND text = ?",
			session.ID, store.ItemToolResult, connectorEcho, connectorEchoText).Scan(&count))
		ran += count
	}
	return ran
}
