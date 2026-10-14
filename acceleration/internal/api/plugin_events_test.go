//go:build integration

package api

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// PluginEventsSuite covers MCP events end to end: the router subscribes on a plugin's
// server, the server checks the callback, and an event it delivers opens a conversation.
// Sentry's server is stood in for by eventServer, which does what the MCP Events draft
// asks of one.
type PluginEventsSuite struct {
	RouterSuite
	mcp *eventServer
	// logged is what the router logged, for the tests of the deprecation it warns of.
	logged *lockedLog
}

func TestPluginEventsSuite(t *testing.T) {
	runSuite(t, new(PluginEventsSuite))
}

func (s *PluginEventsSuite) SetupSuite() {
	s.mcp = &eventServer{subscriptions: map[string]eventSubscription{}}
	s.pluginMCP = httptest.NewTLSServer(http.HandlerFunc(s.mcp.serve))
	s.T().Cleanup(s.pluginMCP.Close)
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
}

func (s *PluginEventsSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

func (s *PluginEventsSuite) TestAnEventTheAgentSubscribedToOpensAConversation() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)

	s.Equal(http.StatusAccepted, s.deliver(sub, "evt-"+s.utils.uuid()))

	s.Eventually(func() bool { return len(s.conversations(config.Name, "")) == 1 }, settleFor, 50*time.Millisecond)
}

func (s *PluginEventsSuite) TestTheServerCheckedTheCallbackBeforeSubscribing() {
	config := s.subscribedAgent(nil)

	sub := s.mcp.subscription(s.T(), config.Name)
	s.True(sub.verified, "the callback echoed the challenge")
	held, err := s.store.PluginEventSubscriptions(context.Background(), s.customerID(), config.Id)
	s.Require().NoError(err)
	s.Require().Len(held, 1)
	s.Equal(store.PluginEventActive, held[0].Status)
}

func (s *PluginEventsSuite) TestADeliveryRetriedOpensNoSecondConversation() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)
	id := "evt-" + s.utils.uuid()

	s.Equal(http.StatusAccepted, s.deliver(sub, id))
	s.Equal(http.StatusOK, s.deliver(sub, id))
}

func (s *PluginEventsSuite) TestADeliveryNotSignedWithTheSecretIsRefused() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)
	forged, err := plugins.NewWebhookSecret()
	s.Require().NoError(err)
	sub.secret = forged

	s.Equal(http.StatusUnauthorized, s.deliver(sub, "evt-"+s.utils.uuid()))
}

func (s *PluginEventsSuite) TestADeliveryWarnsOfTheDeprecationAndStillOpensAConversation() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)

	s.Equal(http.StatusAccepted, s.deliver(sub, "evt-"+s.utils.uuid()))

	lines := deprecations(s.logged, plugins.PathEventDelivery, config.Id)
	s.Require().Len(lines, 1)
	s.Contains(lines[0], "customer="+s.customerID()+" config="+config.Id+" plugin=sentry event=issue.created")
	s.Eventually(func() bool { return len(s.conversations(config.Name, "")) == 1 }, settleFor, 50*time.Millisecond)
}

func (s *PluginEventsSuite) TestADeliveryNotSignedWithTheSecretWarnsOfNothing() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)
	forged, err := plugins.NewWebhookSecret()
	s.Require().NoError(err)
	sub.secret = forged

	s.Equal(http.StatusUnauthorized, s.deliver(sub, "evt-"+s.utils.uuid()))
	s.Empty(deprecations(s.logged, plugins.PathEventDelivery, config.Id))
}

func (s *PluginEventsSuite) TestAnEventNoLongerDeclaredIsUnsubscribedAndGone() {
	config := s.subscribedAgent(nil)
	sub := s.mcp.subscription(s.T(), config.Name)

	var patched AgentConfig
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPatch, "/v1/agents/configs/"+config.Id,
		AgentConfigPatch{PluginEvents: &[]PluginEvent{}}, &patched))
	s.reconcile(config.Id)

	s.False(s.mcp.subscribed(sub.url), "the server was told to stop")
	s.Equal(http.StatusGone, s.deliver(sub, "evt-"+s.utils.uuid()))
}

func (s *PluginEventsSuite) TestAnEndUsersLoginSubscribesAndTheConversationIsTheirs() {
	userID := s.utils.uuid()
	config := s.subscribedAgent(&userID)
	sub := s.mcp.subscription(s.T(), config.Name)

	s.Equal(http.StatusAccepted, s.deliver(sub, "evt-"+s.utils.uuid()))

	s.Eventually(func() bool { return len(s.conversations(config.Name, userID)) == 1 }, settleFor, 50*time.Millisecond)
}

func (s *PluginEventsSuite) TestAnEventOnAPluginTheConfigDoesNotNameIsRefused() {
	status := s.serverClient.do(http.MethodPost, "/v1/agents/configs", AgentConfigRequest{
		Name:         "watcher-" + s.utils.uuid(),
		PluginEvents: &[]PluginEvent{{Plugin: "sentry", Event: "issue.created"}},
	}, nil)

	s.Equal(http.StatusBadRequest, status)
}

// subscribedAgent is a text agent declaring sentry's issue.created, logged into Sentry by the
// app, or by the end user named, and subscribed.
func (s *PluginEventsSuite) subscribedAgent(userID *string) AgentConfig {
	request := AgentConfigRequest{
		Name: "watcher-" + s.utils.uuid(),
		Mode: pointerTo(AgentModeText),
		Llm:  pointerTo("llm-flow"),
		PluginEvents: &[]PluginEvent{{
			Plugin: "sentry", Event: "issue.created",
			Arguments:    &map[string]any{"project": "web"},
			Instructions: pointerTo("Say what broke."),
		}},
	}
	request.Plugins = &[]PluginEntry{{Name: "sentry", User: pointerTo(userID != nil)}}
	var config AgentConfig
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/configs", request, &config))

	login := store.PluginConnection{
		CustomerID: s.customerID(), ConfigID: config.Id, PluginID: "sentry",
		Status: store.PluginConnected, AccessToken: "token-" + s.utils.uuid(),
	}
	if userID != nil {
		login.UserID = *userID
	}
	s.mcp.name(login.AccessToken, config.Name)
	s.Require().NoError(s.store.UpsertPluginConnection(context.Background(), &login))
	s.reconcile(config.Id)
	return config
}

func (s *PluginEventsSuite) reconcile(configID string) {
	config, err := s.store.AgentConfig(context.Background(), s.customerID(), configID)
	s.Require().NoError(err)
	s.events.Reconcile(context.Background(), config)
}

// deliver posts one issue.created to a subscription's callback, signed as the server signs it.
func (s *PluginEventsSuite) deliver(sub eventSubscription, id string) int {
	body, err := json.Marshal(map[string]any{
		"eventId": id, "name": "issue.created", "timestamp": time.Now().UTC().Format(time.RFC3339),
		"data": map[string]any{"project": "web", "title": "TypeError in checkout"}, "cursor": nil,
	})
	s.Require().NoError(err)
	response, err := signedPost(sub.url, sub.secret, id, body)
	s.Require().NoError(err)
	defer response.Body.Close()
	return response.StatusCode
}

func (s *PluginEventsSuite) conversations(agent, userID string) []store.AgentSession {
	found, err := s.store.QuerySessions(context.Background(), s.customerID(),
		store.SessionFilter{AgentName: agent, UserID: userID})
	s.Require().NoError(err)
	return found
}

// eventServer is an MCP server offering events, as the draft describes one: it checks a
// callback with a signed challenge before it accepts a subscription to it.
type eventServer struct {
	mu sync.Mutex
	// agents says which agent each login's token belongs to, so a test finds its own.
	agents        map[string]string
	subscriptions map[string]eventSubscription
}

type eventSubscription struct {
	agent, url, secret string
	verified           bool
}

func (e *eventServer) name(token, agent string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.agents == nil {
		e.agents = map[string]string{}
	}
	e.agents[token] = agent
}

func (e *eventServer) subscription(t *testing.T, agent string) eventSubscription {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, sub := range e.subscriptions {
		if sub.agent == agent {
			return sub
		}
	}
	t.Fatalf("nobody subscribed for %s", agent)
	return eventSubscription{}
}

func (e *eventServer) subscribed(url string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	_, ok := e.subscriptions[url]
	return ok
}

func (e *eventServer) serve(w http.ResponseWriter, r *http.Request) {
	var request struct {
		ID     int    `json:"id"`
		Method string `json:"method"`
		Params struct {
			Name      string            `json:"name"`
			Arguments map[string]any    `json:"arguments"`
			Delivery  map[string]string `json:"delivery"`
		} `json:"params"`
	}
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	token := r.Header.Get("Authorization")[len("Bearer "):]
	switch request.Method {
	case "server/discover":
		answer(w, request.ID, map[string]any{
			"resultType":        "complete",
			"supportedVersions": []string{plugins.EventsVersion},
			"capabilities":      map[string]any{"tools": map[string]any{}, "events": map[string]any{}},
		})
	case "events/subscribe":
		url, secret := request.Params.Delivery["url"], request.Params.Delivery["secret"]
		verified, err := challenge(url, secret)
		if err != nil {
			_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": request.ID,
				"error": map[string]any{"code": -32015, "message": "CallbackEndpointError", "data": map[string]string{"reason": err.Error()}}})
			return
		}
		e.mu.Lock()
		e.subscriptions[url] = eventSubscription{agent: e.agents[token], url: url, secret: secret, verified: verified}
		e.mu.Unlock()
		answer(w, request.ID, map[string]any{
			"id": "sub_" + strconv.Itoa(len(url)), "refreshBefore": time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
			"cursor": nil, "truncated": false,
		})
	case "events/unsubscribe":
		e.mu.Lock()
		delete(e.subscriptions, request.Params.Delivery["url"])
		e.mu.Unlock()
		answer(w, request.ID, map[string]any{})
	default:
		http.Error(w, "unexpected "+request.Method, http.StatusBadRequest)
	}
}

// challenge checks a callback the way the draft asks: a signed, single-use challenge it
// has to echo.
func challenge(url, secret string) (bool, error) {
	raw := make([]byte, 16)
	if _, err := rand.Read(raw); err != nil {
		return false, err
	}
	sent := hex.EncodeToString(raw)
	body, _ := json.Marshal(map[string]string{"type": "verification", "challenge": sent})
	response, err := signedPost(url, secret, "msg_verification_"+sent, body)
	if err != nil {
		return false, err
	}
	defer response.Body.Close()
	var echoed struct {
		Challenge string `json:"challenge"`
	}
	if err := json.NewDecoder(response.Body).Decode(&echoed); err != nil || response.StatusCode/100 != 2 {
		return false, fmt.Errorf("challenge_failed: %d", response.StatusCode)
	}
	return echoed.Challenge == sent, nil
}

func signedPost(url, secret, id string, body []byte) (*http.Response, error) {
	now := time.Now()
	signature, err := plugins.SignWebhook(secret, id, now, body)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("webhook-id", id)
	request.Header.Set("webhook-timestamp", strconv.FormatInt(now.Unix(), 10))
	request.Header.Set("webhook-signature", signature)
	return http.DefaultClient.Do(request)
}

func answer(w http.ResponseWriter, id int, result any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"jsonrpc": "2.0", "id": id, "result": result})
}
