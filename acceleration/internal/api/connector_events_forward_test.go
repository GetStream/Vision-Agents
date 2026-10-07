//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// EventForwardingSuite is a customer's Slack app's events going on to the customer's event
// destinations (T46): the fake provider plays Slack, delivering to
// POST /v1/connectors/events/slack_bot/{app id} signed with the app's signing secret; the
// router acts on the event as it does without destinations, acks, and the suite's forwarder
// sends the raw event to local TLS servers that stand for the customer's URLs.
type EventForwardingSuite struct {
	RouterSuite

	slack     *fakeprovider.Server
	app       store.ConnectorOAuthClient
	secret    string
	workspace string
}

func TestEventForwardingSuite(t *testing.T) {
	runSuite(t, new(EventForwardingSuite))
}

// SetupSuite registers hmac_header, oauth2_code and the real channel bridge, as
// SlackChannelSuite does, and an event forwarder.
func (s *EventForwardingSuite) SetupSuite() {
	verifier := hmacheader.New()
	code, err := oauth2code.New(oauth2code.Config{HTTP: http.DefaultClient})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{oauth2code.Name: code},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return strings.TrimPrefix(s.slack.URL, "https://") }
	s.forwardHTTP = destinationClient()
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *EventForwardingSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.slack = fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	s.workspace = s.slack.TeamID
	s.secret = "synthetic-signing-" + s.utils.uuid()
	s.app = s.providerApp(s.secret)
}

// A Block Kit button click is an interaction the router does not act on. It reaches the
// customer as Slack sent it, so Slack Bolt verifies it with the app's own signing secret.
func (s *EventForwardingSuite) TestABlockActionsReachesTheDestinationWithSlacksBodyAndHeadersAsTheyCame() {
	target := newDestination(s.T())
	created := s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.interaction(`{"type":"block_actions","team":{"id":"` + s.workspace + `"},"user":{"id":"U0000ALICE"},` +
		`"trigger_id":"synthetic.trigger","actions":[{"action_id":"approve","block_id":"b1","type":"button","value":"yes"}]}`)

	status, _, sent := s.slack.DeliverInteraction(s.eventsURL(), s.secret, body)

	s.Require().Equal(http.StatusOK, status)
	got := s.forwardedTo(target, 1)[0]
	s.Equal(body, got.body, "the raw body, byte for byte")
	for _, name := range []string{"Content-Type", "X-Slack-Signature", "X-Slack-Request-Timestamp"} {
		s.Equal(sent.Get(name), got.header.Get(name), name)
	}
	s.NoError(plugins.VerifyWebhook(created.Secret, got.header, got.body, time.Now()), "signed with the destination's secret")
}

func (s *EventForwardingSuite) TestAForwardSignedForOneCustomerDoesNotVerifyWithAnothersKey() {
	mine := newDestination(s.T())
	created := s.createDestination("slack_bot", mine.URL, store.ForwardUnhandled)
	s.useApp(s.data.createApp())
	theirs := s.createDestination("slack_bot", newDestination(s.T()).URL, store.ForwardUnhandled)

	s.deliver(s.reaction(), 0)

	got := s.forwardedTo(mine, 1)[0]
	s.NoError(plugins.VerifyWebhook(created.Secret, got.header, got.body, time.Now()))
	s.ErrorIs(plugins.VerifyWebhook(theirs.Secret, got.header, got.body, time.Now()), plugins.ErrBadSignature)
}

// A person's message an agent answers is the router's: a destination of unhandled events is
// not sent it, and one of every event is.
func (s *EventForwardingSuite) TestAMessageAnAgentAnswersGoesOnlyToADestinationOfEveryEvent() {
	s.answering(s.connectedBot())
	unhandled := newDestination(s.T())
	s.createDestination("slack_bot", unhandled.URL, store.ForwardUnhandled)
	every := newDestination(s.T())
	s.createDestination("slack_bot", every.URL, store.ForwardAll)
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	status, _ := s.deliver(body, 0)

	s.Equal(http.StatusOK, status)
	s.Equal(body, s.forwardedTo(every, 1)[0].body)
	s.Never(func() bool { return len(unhandled.requests()) > 0 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return len(s.chat.Stored(s.threadChannel())) == 1 }, settleFor, 10*time.Millisecond,
		"the agent's thread channel still gets the message")
}

// An answer is counted before the bridge writes the message into its thread channel, after
// the ack. When that write fails no agent got the message, so a destination of unhandled
// events gets it then (AI-924), not neither of them.
func (s *EventForwardingSuite) TestAMessageWhoseThreadChannelWriteFailsGoesToADestinationOfUnhandledEvents() {
	s.answering(s.connectedBot())
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	// Stream refuses every write of the customer's: its app is read only, as app mode leaves
	// the deployment's app once the fallback is off.
	s.setApps(s.customerID(), func(apps *suiteApps) { apps.readOnly[s.customerID()] = true })
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	status, _ := s.deliver(body, 0)

	s.Equal(http.StatusOK, status)
	s.Equal(body, s.forwardedTo(target, 1)[0].body)
	s.Empty(s.chat.Stored(s.threadChannel()), "the thread channel did not get it")
}

// Slack's event_id is «A unique identifier for this specific event»
// (https://docs.slack.dev/apis/events-api/), and slack_bot.yaml keys a forward by it: two
// deliveries of one event are one forward even when their bodies differ.
func (s *EventForwardingSuite) TestTwoDeliveriesOfOneSlackEventAreOneForward() {
	target := newDestination(s.T())
	release := target.holding()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	first := s.reaction()
	again := []byte(strings.Replace(string(first), `"type":"event_callback"`, `"type":"event_callback","is_ext_shared_channel":false`, 1))
	s.Require().NotEqual(first, again)

	s.deliver(first, 0)
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond)
	s.deliver(again, 1)
	release()

	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
	s.Len(target.requests(), 1)
}

// Mode C: the customer runs its own agent, so no agent config binds the bot connection, and a
// message is one the router answers in no way.
func (s *EventForwardingSuite) TestAMessageNoAgentAnswersIsUnhandled() {
	s.connectedBot()
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100")

	s.deliver(body, 0)

	s.Equal(body, s.forwardedTo(target, 1)[0].body)
}

// https://docs.slack.dev/reference/events/url_verification: the handshake proves the router's
// URL to Slack; the customer's URL is not Slack's to call.
func (s *EventForwardingSuite) TestAURLVerificationIsNotForwarded() {
	target := newDestination(s.T())
	s.createDestination("slack_bot", target.URL, store.ForwardAll)

	status, answer := s.deliver([]byte(`{"token":"synthetic","challenge":"synthetic-challenge-value","type":"url_verification"}`), 0)

	s.Equal(http.StatusOK, status)
	s.Equal("synthetic-challenge-value", answer)
	s.Never(func() bool { return len(target.requests()) > 0 }, dropped, 20*time.Millisecond)
}

// Slack waits three seconds for its answer (https://docs.slack.dev/apis/events-api/); a
// customer's URL may take fifteen.
func (s *EventForwardingSuite) TestTheAckToSlackDoesNotWaitForTheDestination() {
	target := newDestination(s.T())
	release := target.holding()
	defer release()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)

	started := time.Now()
	status, _ := s.deliver(s.reaction(), 0)

	s.Equal(http.StatusOK, status)
	s.Less(time.Since(started), time.Second, "answered while the destination still holds the forward")
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond,
		"the forward is on its way")
}

func (s *EventForwardingSuite) TestAForwardTheDestinationAnswers503IsSentAgainAndTakenOnce() {
	target := newDestination(s.T())
	target.answer(http.StatusServiceUnavailable)
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.reaction()

	s.deliver(body, 0)

	sent := s.forwardedTo(target, 2)
	s.Equal(body, sent[1].body)
	s.Equal(sent[0].header.Get("webhook-id"), sent[1].header.Get("webhook-id"), "one webhook-id on every attempt")
	s.Never(func() bool { return len(target.requests()) > 2 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
}

func (s *EventForwardingSuite) TestAForwardTheDestinationAnswers400IsNotSentAgain() {
	target := newDestination(s.T())
	target.answer(http.StatusBadRequest)
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)

	s.deliver(s.reaction(), 0)

	s.forwardedTo(target, 1)
	s.Never(func() bool { return len(target.requests()) > 1 }, dropped, 20*time.Millisecond)
	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
}

// Slack retries a delivery it got no answer to in time, with X-Slack-Retry-Num
// (https://docs.slack.dev/apis/events-api/, «Retries»); the forward of the first one still
// pending is not queued twice.
func (s *EventForwardingSuite) TestADeliverySlackRetriesWhileItsForwardIsPendingIsQueuedOnce() {
	target := newDestination(s.T())
	release := target.holding()
	s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	body := s.reaction()
	s.deliver(body, 0)
	s.Require().Eventually(func() bool { return len(target.requests()) == 1 }, settleFor, 10*time.Millisecond)

	s.deliver(body, 1)
	release()

	s.Require().Eventually(func() bool { return s.pendingForwards() == 0 }, settleFor, 10*time.Millisecond)
	s.Len(target.requests(), 1)
}

// The Standard Webhooks rotation: «the webhook is signed both using the current key, and using
// an old key (for a set period of time)» (https://www.standardwebhooks.com/).
func (s *EventForwardingSuite) TestAfterARotationAForwardVerifiesWithTheOldSecretAndTheNew() {
	target := newDestination(s.T())
	created := s.createDestination("slack_bot", target.URL, store.ForwardUnhandled)
	var rotated ConnectorEventDestinationSecret
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPost,
		destinationPath("slack_bot", created.Destination.ID)+"/rotate-secret", nil, &rotated))

	s.deliver(s.reaction(), 0)

	got := s.forwardedTo(target, 1)[0]
	s.NoError(plugins.VerifyWebhook(created.Secret, got.header, got.body, time.Now()), "the old secret")
	s.NoError(plugins.VerifyWebhook(rotated.Secret, got.header, got.body, time.Now()), "the new secret")
}

// forwardedTo waits until a destination took count requests, and returns them.
func (s *EventForwardingSuite) forwardedTo(target *destination, count int) []forwarded {
	s.Require().Eventually(func() bool { return len(target.requests()) >= count }, settleFor, 10*time.Millisecond,
		"the destination took %d forwards, not %d", len(target.requests()), count)
	return target.requests()
}

// pendingForwards is how many forwards to the test app's destinations are not done with.
func (s *EventForwardingSuite) pendingForwards() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		`SELECT count(*) FROM connector_event_deliveries d JOIN connector_event_destinations t ON t.id = d.destination_id
		 WHERE t.customer_id = ?`, s.customerID()).Scan(&count))
	return count
}

func (s *EventForwardingSuite) eventsURL() string {
	return s.server.URL + providerAppEventsPath + "slack_bot/" + s.app.ProviderAppID
}

// deliver posts a Slack event to the test app's events URL, signed with its secret.
func (s *EventForwardingSuite) deliver(body []byte, retry int) (int, string) {
	return s.slack.Deliver(s.eventsURL(), s.secret, body, retry)
}

// event is a Slack event_callback in the test's workspace (https://docs.slack.dev/apis/events-api/).
func (s *EventForwardingSuite) event(inner string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + s.workspace + `","api_app_id":"` + s.app.ProviderAppID +
		`","event":` + inner + `,"type":"event_callback","event_id":"Ev` + strings.ReplaceAll(s.utils.uuid(), "-", "") +
		`","event_time":` + fmt.Sprint(time.Now().Unix()) + `}`)
}

// message is a message.channels event by user in C0000CHAN
// (https://docs.slack.dev/reference/events/message.channels).
func (s *EventForwardingSuite) message(user, text, ts string) []byte {
	raw, err := json.Marshal(map[string]string{"type": "message", "channel": "C0000CHAN", "user": user, "text": text, "ts": ts, "channel_type": "channel"})
	s.Require().NoError(err)
	return s.event(string(raw))
}

// reaction is a reaction_added event (https://docs.slack.dev/reference/events/reaction_added),
// which the router reads nothing from.
func (s *EventForwardingSuite) reaction() []byte {
	return s.event(`{"type":"reaction_added","user":"U0000ALICE","reaction":"thumbsup","item_user":"U0000BOT",` +
		`"item":{"type":"message","channel":"C0000CHAN","ts":"1759740000.000100"},"event_ts":"1759740001.000200"}`)
}

// interaction is the form body Slack posts an interaction payload in
// (https://docs.slack.dev/interactivity/handling-user-interaction).
func (s *EventForwardingSuite) interaction(payload string) []byte {
	return []byte(url.Values{"payload": {payload}}.Encode())
}

// providerApp is a Slack app of the test's customer with signingSecret sealed for it, as
// SlackChannelSuite.providerApp writes one.
func (s *EventForwardingSuite) providerApp(signingSecret string) store.ConnectorOAuthClient {
	record := &store.ConnectorOAuthClient{
		CustomerID: s.customerID(), ConnectorID: "slack_bot", Registration: core.ClientCustomer,
		ClientID: "synthetic-client", AuthMethod: core.AuthClientSecretPost,
		ProviderAppID: "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", "")),
	}
	var err error
	record.SecretSealed, err = s.sealer.SealWithAAD("synthetic-client-secret", oauthClientAAD(record.CustomerID, "slack_bot"))
	s.Require().NoError(err)
	record.KEKVersion = s.sealer.CurrentVersion()
	record.SigningSecretSealed, err = s.sealer.SealWithAAD(signingSecret, providerAppAAD(record.CustomerID, "slack_bot", record.ProviderAppID))
	s.Require().NoError(err)
	record.SigningKEKVersion = s.sealer.CurrentVersion()
	_, err = s.store.PutConnectorOAuthClient(context.Background(), record)
	s.Require().NoError(err)
	return *record
}

// connectedBot is the app's slack_bot connection of the test's workspace, connected with a bot
// token, as SlackChannelSuite.connectedBot makes one.
func (s *EventForwardingSuite) connectedBot() string {
	var created Connection
	s.Require().Equal(http.StatusCreated, s.serverClient.do(http.MethodPost, "/v1/agents/connections", appOwned("slack_bot"), &created))
	ref := core.ConnectionRef{CustomerID: s.customerID(), ConnectionID: created.ID}
	payload, err := json.Marshal(map[string]string{"access_token": s.slack.InstallBot(), "token_type": "bot"})
	s.Require().NoError(err)
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(context.Background(), ref, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.Credentials = core.StoredCredentials{Scheme: oauth2code.Name, Version: 1, Payload: payload}
		state.Status = store.ConnectionConnected
		state.AccountID = s.workspace
		state.Metadata = map[string]string{"team_id": s.workspace}
		state.ConnectedAt = time.Now().UTC()
		return true, nil
	}))
	return created.ID
}

// answering is an agent config of the test's app that binds the bot connection as fixed.
func (s *EventForwardingSuite) answering(connectionID string) {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "slack-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
		Connectors: []store.ConnectorBinding{{
			Name: "slack", ConnectorID: "slack_bot",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: connectionID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
}

// threadChannel is the test customer's one thread channel, once the bridge linked it.
func (s *EventForwardingSuite) threadChannel() string {
	var channel string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT channel_id FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&channel))
	return channel
}
