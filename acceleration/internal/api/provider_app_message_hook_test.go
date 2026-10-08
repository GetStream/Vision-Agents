//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/slackapps"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// messageHookHarness is a router in app mode whose customer registered a Stream app of its
// own, with a key the suite's Stream answers as that app: the app a provider app is pinned to
// (T48, AI-887). Each suite on it starts the router with connectors on or off, and with or
// without ROUTER_PUBLIC_URL. The operator's Slack app is in the router's environment, as
// SLACK_BOT_MCP_*.
type messageHookHarness struct {
	RouterSuite
	// slack is the suite's Slack, for the managed app and the bot's events. Nil with
	// connectors off.
	slack       *fakeprovider.Server
	environment map[string]string
	staff       *testClient
	logged      *lockedLog
	// apiKey and secret are the key of the test customer's own Stream app.
	apiKey, secret string
}

// start builds the router: connectors on gives it Slack's app API and the operator's app, as
// cmd/router does only with connectors enabled; off leaves both out, as cmd/router does.
func (s *messageHookHarness) start(connectorsOn bool, publicURL string) {
	s.appMode = true
	s.environment = map[string]string{
		"SLACK_BOT_MCP_APP_ID":         "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))[20:],
		"SLACK_BOT_MCP_CLIENT_ID":      "operator-client-" + s.utils.uuid(),
		"SLACK_BOT_MCP_CLIENT_SECRET":  "operator-secret-" + s.utils.uuid(),
		"SLACK_BOT_MCP_SIGNING_SECRET": "operator-signing-" + s.utils.uuid(),
	}
	verifier := hmacheader.New()
	// Retrieve never reaches a token endpoint here: the stored bot token has no expiry and no
	// refresh token, so it is handed out as stored.
	code, err := oauth2code.New(oauth2code.Config{HTTP: http.DefaultClient})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{oauth2code.Name: code},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	if connectorsOn {
		s.slack = fakeprovider.New(s.T(), fakeprovider.SlackChannel)
		s.slackApps, err = slackapps.New(slackapps.Config{HTTP: s.slack.Client(), BaseURL: s.slack.URL + fakeprovider.PathSlackAPI})
		s.Require().NoError(err)
		s.operatorApps = ConnectorOperatorApps(func(name string) string { return s.environment[name] })
		s.channelProvider = func() string { return strings.TrimPrefix(s.slack.URL, "https://") }
	}
	s.publicURL = publicURL
	s.logged = &lockedLog{}
	s.logs = s.logged
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
	s.staff = &testClient{suite: &s.RouterSuite, header: http.Header{opsKeyHeader: {suiteOpsKey}}, kind: noCredential}
}

// SetupTest makes a customer whose id is its Stream app's, and registers that app's key.
func (s *messageHookHarness) SetupTest() {
	s.useApp(s.numberedApp(streamAppID()))
	s.apiKey, s.secret = "own-key-"+s.utils.uuid(), "own-secret-"+s.utils.uuid()
	s.answering(false)
	s.registered(0, key(s.apiKey, s.secret))
	if s.slack != nil {
		s.slack.Use(fakeprovider.SlackChannel)
	}
}

// answering makes the suite's Stream answer the customer's key as the customer's app, set up
// as the router needs, or refuse it.
func (s *messageHookHarness) answering(refuses bool) {
	s.chat.SetAppFor(s.apiKey, chattest.App{
		ID:           s.appID(),
		ChannelTypes: map[string]map[string][]string{"agent": {"channel_member": {"read-channel", "create-message"}}},
		CallTypes:    []string{"agent"},
		Refuses:      refuses,
	})
}

// hookURL is where the customer's app delivers its messages once the router pointed it.
func (s *messageHookHarness) hookURL(public string) string {
	return public + chat.MessageHookPath + "/" + strconv.FormatInt(s.appID(), 10)
}

// pointed is the one message hook the router points: webhook, message.new only, on.
func pointed(url string) getstream.EventHook {
	return getstream.EventHook{
		HookType: pointerTo("webhook"), WebhookUrl: pointerTo(url), Enabled: pointerTo(true), EventTypes: []string{"message.new"},
	}
}

// operatorApp makes Stream's own Slack app the test customer's provider app, and takes it
// back when the test ends, since an app serves one customer.
func (s *messageHookHarness) operatorApp() int {
	status := s.staff.do(http.MethodPut, operatorAppPath(s.customerID()), nil, nil)
	s.T().Cleanup(func() { s.staff.do(http.MethodDelete, operatorAppPath(s.customerID()), nil, nil) })
	return status
}

// asked are the requests made of Stream with a key whose path ends in suffix.
func (s *messageHookHarness) asked(apiKey, method, suffix string) int {
	count := 0
	for _, request := range s.chat.Requests(apiKey) {
		if request.Method == method && strings.HasSuffix(request.Path, suffix) {
			count++
		}
	}
	return count
}

// ProviderAppMessageHookSuite is connectors on, with ROUTER_PUBLIC_URL set: a provider app
// PUT points the message hook of the Stream app the provider app is pinned to.
type ProviderAppMessageHookSuite struct {
	messageHookHarness
}

func TestProviderAppMessageHookSuite(t *testing.T) {
	runSuite(t, new(ProviderAppMessageHookSuite))
}

func (s *ProviderAppMessageHookSuite) SetupSuite() {
	s.start(true, providerAppPublicURL)
}

func (s *ProviderAppMessageHookSuite) TestCreatingTheAppPointsThePinnedAppsMessageHookAtTheRouter() {
	status, payload := s.serverClient.call(http.MethodPut, providerAppPath("slack_bot"),
		ConnectorProviderAppRequest{Name: "Acme", ConfigRefreshToken: s.slack.NewConfigToken()})

	s.Require().Equal(http.StatusCreated, status, string(payload))
	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Equal(s.appID(), record.StreamAppPK, "pinned to the customer's own app")
	s.Equal([]getstream.EventHook{pointed(s.hookURL(providerAppPublicURL))}, s.chat.EventHooks(s.apiKey))
	s.Zero(s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app"), "the deployment's own app is left alone")
}

// A PUT again points the hook again: it updates the one it pointed rather than adding a
// second, puts it back when it was removed, and leaves a hook of the customer's own.
func (s *ProviderAppMessageHookSuite) TestAPutAgainPointsTheHookAgainAndLeavesTheAppsOtherHooks() {
	client, err := getstream.NewClient(s.apiKey, s.secret, getstream.WithBaseUrl(s.chat.URL))
	s.Require().NoError(err)
	customers := chat.StreamOf(client)
	_, err = customers.PointMessageHook(context.Background(), "https://customer.example/hooks")
	s.Require().NoError(err)
	both := []getstream.EventHook{pointed("https://customer.example/hooks"), pointed(s.hookURL(providerAppPublicURL))}
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"),
		ConnectorProviderAppRequest{Name: "Acme", ConfigRefreshToken: s.slack.NewConfigToken()}, nil))

	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"),
		ConnectorProviderAppRequest{Name: "Acme"}, nil))
	s.Equal(both, s.chat.EventHooks(s.apiKey), "updated, not added again")

	_, err = customers.RemoveMessageHook(context.Background(), s.hookURL(providerAppPublicURL))
	s.Require().NoError(err)
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, providerAppPath("slack_bot"),
		ConnectorProviderAppRequest{Name: "Acme"}, nil))
	s.Equal(both, s.chat.EventHooks(s.apiKey), "put back")
}

func (s *ProviderAppMessageHookSuite) TestStaffSettingStreamsOwnAppPointsThePinnedAppsMessageHook() {
	s.Require().Equal(http.StatusCreated, s.operatorApp())

	s.Equal([]getstream.EventHook{pointed(s.hookURL(providerAppPublicURL))}, s.chat.EventHooks(s.apiKey))
}

// Stream refusing fails the PUT, as it fails `router phone hooks`, and keeps the provider
// app: a PUT again points the hook.
func (s *ProviderAppMessageHookSuite) TestAHookStreamRefusesFailsThePutAndAPutAgainPointsIt() {
	s.answering(true)

	s.Equal(http.StatusServiceUnavailable, s.operatorApp())

	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err, "the provider app is kept")
	s.Empty(s.chat.EventHooks(s.apiKey))
	s.answering(false)
	s.Equal(http.StatusOK, s.operatorApp())
	s.Equal([]getstream.EventHook{pointed(s.hookURL(providerAppPublicURL))}, s.chat.EventHooks(s.apiKey))
}

// A customer with no app of its own is pinned to the deployment's, whose hooks are the
// operator's: nothing is asked of Stream.
func (s *ProviderAppMessageHookSuite) TestAProviderAppPinnedToTheDeploymentsAppPointsNoHook() {
	s.useApp(s.data.createApp())
	before := s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app")

	s.Require().Equal(http.StatusCreated, s.operatorApp())

	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Equal(int64(suiteStreamApp), record.StreamAppPK)
	s.Equal(before, s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app"))
	s.Empty(s.chat.EventHooks(suiteStreamKey))
}

// The acceptance of T48 with T55's closer: a Slack thread's card is written into the app the
// provider app was pinned to at its first message, and its summary after the idle period
// lands there too, never in the deployment's app.
func (s *ProviderAppMessageHookSuite) TestACardWrittenAfterTheIdlePeriodLandsInThePinnedApp() {
	ctx := context.Background()
	s.Require().Equal(http.StatusCreated, s.operatorApp())
	workspace := s.slack.TeamID
	bot := s.connectedBot(workspace, s.slack.InstallBot())
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "slack-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "recites/recites-model",
		Connectors: []store.ConnectorBinding{{
			Name: "slack", ConnectorID: "slack_bot",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: bot.ConnectionID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(ctx, &config))

	status, _ := s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/"+s.environment["SLACK_BOT_MCP_APP_ID"],
		s.environment["SLACK_BOT_MCP_SIGNING_SECRET"], s.slackMessage(workspace, "U0000ALICE", "Can you check the build?"), 0)
	s.Require().Equal(http.StatusOK, status)
	var episode string
	var pin int64
	s.Require().Eventually(func() bool {
		return s.store.DB().QueryRowContext(ctx, "SELECT id, stream_app_pk FROM episodes WHERE customer_id = ?", s.customerID()).Scan(&episode, &pin) == nil
	}, settleFor, 10*time.Millisecond)
	s.Equal(s.appID(), pin, "the episode is pinned to the provider app's app")
	var cid string
	s.Require().NoError(s.store.DB().QueryRowContext(ctx,
		"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'slack' AND address = ?",
		s.customerID(), config.ID, workspace+":U0000ALICE").Scan(&cid))
	omni := strings.TrimPrefix(cid, "agent:")
	s.Require().Eventually(func() bool { return len(s.chat.Stored(omni)) == 1 }, settleFor, 10*time.Millisecond)
	card, _ := s.chat.Stored(omni)[0]["id"].(string)
	// Idle for two hours, past the closer's hour. Episodes left in progress from earlier runs
	// are older still and would fill the sweep's batch, so they are put out of its way; the
	// other suites' episodes are newer than an hour and stay.
	_, err := s.store.DB().ExecContext(ctx, "UPDATE episodes SET started_at = now() - interval '2 hours' WHERE id = ?", episode)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx, "UPDATE episode_activity SET last_message_at = now() - interval '2 hours' WHERE episode_id = ?", episode)
	s.Require().NoError(err)
	_, err = s.store.DB().ExecContext(ctx,
		"UPDATE episodes SET status = 'summary_failed' WHERE status IN ('in_progress', 'ended') AND started_at < now() - interval '1 hour' AND id <> ?", episode)
	s.Require().NoError(err)

	s.Require().NoError(s.episodes.Sweep(ctx))

	stored := s.chat.Stored(omni)
	s.Require().Len(stored, 1, "the card is updated in place")
	s.Contains(stored[0]["text"], "Can you check the build?")
	custom, _ := stored[0]["custom"].(map[string]any)
	s.Equal(store.EpisodeSummarized, custom["status"])
	s.Equal(1, s.asked(s.apiKey, http.MethodPost, "/agent/"+omni+"/message"), "the card is written in the pinned app")
	s.Equal(2, s.asked(s.apiKey, http.MethodPut, "/chat/messages/"+card), "ended, then summarized, in the pinned app")
	s.Zero(s.asked(suiteStreamKey, http.MethodPost, "/agent/"+omni+"/message"))
	s.Zero(s.asked(suiteStreamKey, http.MethodPut, "/chat/messages/"+card))
}

// connectedBot is the app's slack_bot connection of a workspace, made through the API and then
// given the bot token as an install leaves it, as SlackChannelSuite makes it.
func (s *ProviderAppMessageHookSuite) connectedBot(team, token string) core.ConnectionRef {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned("slack_bot"), &created))
	ref := core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: created.ID}
	payload, err := json.Marshal(map[string]string{"access_token": token, "token_type": "bot"})
	s.Require().NoError(err)
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(context.Background(), ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = core.StoredCredentials{Scheme: oauth2code.Name, Version: 1, Payload: payload}
		state.Status = store.ConnectionConnected
		state.AccountID = team
		state.Metadata = map[string]string{"team_id": team}
		state.ConnectedAt = time.Now().UTC()
		return true, nil
	}))
	return ref
}

// slackMessage is a message.channels event that starts a thread in C0000CHAN
// (https://docs.slack.dev/reference/events/message.channels), in an event_callback
// (https://docs.slack.dev/apis/events-api/) to the operator's app.
func (s *ProviderAppMessageHookSuite) slackMessage(team, user, text string) []byte {
	inner, err := json.Marshal(map[string]string{
		"type": "message", "channel": "C0000CHAN", "user": user, "text": text, "ts": "1759740000.000100", "channel_type": "channel",
	})
	s.Require().NoError(err)
	body, err := json.Marshal(map[string]any{
		"token": "synthetic", "team_id": team, "api_app_id": s.environment["SLACK_BOT_MCP_APP_ID"],
		"event": json.RawMessage(inner), "type": "event_callback",
		"event_id": "Ev" + strings.ReplaceAll(s.utils.uuid(), "-", ""), "event_time": time.Now().Unix(),
	})
	s.Require().NoError(err)
	return body
}

// ProviderAppMessageHookNoAppSuite is connectors on in app mode with no fallback to the
// deployment's app: a customer with no app of its own is pinned to zero while the deployment's
// own app id is known, as in a deployment-mode router in production.
type ProviderAppMessageHookNoAppSuite struct {
	messageHookHarness
}

func TestProviderAppMessageHookNoAppSuite(t *testing.T) {
	runSuite(t, new(ProviderAppMessageHookNoAppSuite))
}

func (s *ProviderAppMessageHookNoAppSuite) SetupSuite() {
	s.appRefuses = true
	s.start(true, providerAppPublicURL)
}

func (s *ProviderAppMessageHookNoAppSuite) SetupTest() {
	s.useApp(s.data.createApp())
}

// A pin of zero names no app, even when the deployment's own app is not zero: nothing is
// asked of Stream, and the deployment's hooks are left alone.
func (s *ProviderAppMessageHookNoAppSuite) TestAProviderAppPinnedToZeroPointsNoHookOfTheDeploymentsApp() {
	before := s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app")

	s.Require().Equal(http.StatusCreated, s.operatorApp())

	record, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.Require().NoError(err)
	s.Zero(record.StreamAppPK)
	s.NotZero(s.stream.DeploymentApp())
	s.Equal(before, s.asked(suiteStreamKey, http.MethodPatch, "/api/v2/app"))
	s.Empty(s.chat.EventHooks(suiteStreamKey))
}

// ProviderAppMessageHookWithoutPublicURLSuite is connectors on without ROUTER_PUBLIC_URL:
// Stream's own app is set all the same, with no hook and one warning. The managed app needs
// the URL for Slack and refuses as before.
type ProviderAppMessageHookWithoutPublicURLSuite struct {
	messageHookHarness
}

func TestProviderAppMessageHookWithoutPublicURLSuite(t *testing.T) {
	runSuite(t, new(ProviderAppMessageHookWithoutPublicURLSuite))
}

func (s *ProviderAppMessageHookWithoutPublicURLSuite) SetupSuite() {
	s.start(true, "")
}

func (s *ProviderAppMessageHookWithoutPublicURLSuite) TestStaffSettingStreamsOwnAppPointsNoHookAndWarnsOnce() {
	before := strings.Count(s.logged.String(), "ROUTER_PUBLIC_URL is not set")
	asked := len(s.chat.Requests(s.apiKey))

	s.Require().Equal(http.StatusCreated, s.operatorApp())

	s.Empty(s.chat.EventHooks(s.apiKey))
	s.Len(s.chat.Requests(s.apiKey), asked, "nothing is asked of the customer's Stream app")
	s.Equal(before+1, strings.Count(s.logged.String(), "ROUTER_PUBLIC_URL is not set"))
	s.Contains(s.logged.String(), "stream_app="+strconv.FormatInt(s.appID(), 10))
}

// ProviderAppMessageHookOffSuite is connectors off, the control: the provider app PUTs answer
// as on base, and nothing is asked of the customer's Stream app.
type ProviderAppMessageHookOffSuite struct {
	messageHookHarness
}

func TestProviderAppMessageHookOffSuite(t *testing.T) {
	runSuite(t, new(ProviderAppMessageHookOffSuite))
}

func (s *ProviderAppMessageHookOffSuite) SetupSuite() {
	s.start(false, providerAppPublicURL)
}

func (s *ProviderAppMessageHookOffSuite) TestTheProviderAppPutsAnswerAsBeforeAndAskNothingOfStream() {
	asked := len(s.chat.Requests(s.apiKey))

	managed, managedBody := s.serverClient.call(http.MethodPut, providerAppPath("slack_bot"),
		ConnectorProviderAppRequest{Name: "Acme", ConfigRefreshToken: "xoxe-synthetic"})
	operator, operatorBody := s.staff.call(http.MethodPut, operatorAppPath(s.customerID()), nil)

	s.Equal(http.StatusBadRequest, managed)
	s.Contains(string(managedBody), `"code":"not_configured"`)
	s.Contains(string(managedBody), "provider apps cannot be created: connectors are not enabled on this deployment")
	s.Equal(http.StatusBadRequest, operator)
	s.Contains(string(operatorBody), `"code":"not_configured"`)
	s.Contains(string(operatorBody), "operator apps cannot be set: connectors are not enabled on this deployment")
	s.Len(s.chat.Requests(s.apiKey), asked, "nothing is asked of the customer's Stream app")
	s.Empty(s.chat.EventHooks(s.apiKey))
	_, err := s.store.ConnectorOAuthClient(context.Background(), s.customerID(), "slack_bot")
	s.ErrorIs(err, store.ErrNoConnectorOAuthClient)
}

// lockedLog is what the router logs, written from any goroutine and read by the test.
type lockedLog struct {
	mu      sync.Mutex
	written bytes.Buffer
}

func (l *lockedLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.written.Write(p)
}

func (l *lockedLog) String() string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.written.String()
}
