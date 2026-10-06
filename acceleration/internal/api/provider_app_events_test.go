//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// SlackChannelSuite is Slack as an inbound channel end to end, on the built-in slack_bot
// manifest: the fake provider plays Slack, delivering Events API events to
// POST /v1/connectors/events/slack_bot/{app id}, signed with the customer's app's own signing
// secret, and taking replies at chat.postMessage. The channel bridge is the router's own,
// writing into the suite's Stream Chat; the suite delivers the message.new Stream Chat would
// send for what was written there, signed as Stream signs it.
type SlackChannelSuite struct {
	RouterSuite

	// slack is the test's Slack, one per test, which the bridge's replies dial.
	slack *fakeprovider.Server
	// app is the test app's Slack app, with its signing secret; bot is its workspace's bot
	// connection, botToken that connection's token, config the agent that answers there.
	app       store.ConnectorOAuthClient
	secret    string
	bot       core.ConnectionRef
	botToken  string
	config    store.AgentConfig
	workspace string
}

func TestSlackChannelSuite(t *testing.T) {
	runSuite(t, new(SlackChannelSuite))
}

// SetupSuite registers hmac_header and the real oauth2_code, whose Wrap puts the bot token on
// a reply, and the real channel bridge, with sessions that write their transcripts.
func (s *SlackChannelSuite) SetupSuite() {
	verifier := hmacheader.New()
	// Retrieve never reaches a token endpoint here: the stored bot token has no expiry and no
	// refresh token, so it is handed out as stored.
	code, err := oauth2code.New(oauth2code.Config{HTTP: http.DefaultClient})
	s.Require().NoError(err)
	s.connectors = core.Registry{
		Schemes:   map[string]core.Scheme{oauth2code.Name: code},
		Verifiers: map[string]core.Verifier{verifier.Name(): verifier},
	}
	s.channelProvider = func() string { return strings.TrimPrefix(s.slack.URL, "https://") }
	s.transcripts = true
	s.RouterSuite.SetupSuite()
	s.Require().NoError(s.store.SeedConnectorDefinitions(context.Background(), providers.FS))
}

func (s *SlackChannelSuite) SetupTest() {
	s.useApp(s.data.createApp())
	s.slack = fakeprovider.New(s.T(), fakeprovider.SlackChannel)
	s.workspace = s.slack.TeamID
	s.secret = "synthetic-signing-" + s.utils.uuid()
	s.app = s.providerApp(s.secret)
	s.botToken = s.slack.InstallBot()
	s.bot = s.connectedBot(s.workspace, s.botToken)
	s.config = s.answering(s.bot.ConnectionID)
}

func (s *SlackChannelSuite) TestAMessageIsWrittenIntoANewThreadChannelAsThePersonWithoutASource() {
	status, _ := s.deliver(s.message("U0000ALICE", "Can you check the build?", "1759740000.000100", ""), 0)

	s.Equal(http.StatusOK, status)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	stored := s.written(channel, 1)
	s.Equal("Can you check the build?", stored[0]["text"])
	s.Empty(stored[0]["custom"], "without source, so the message hook takes it as written to the agent")
	author, _ := stored[0]["user_id"].(string)
	user, found := s.chat.User(author)
	s.Require().True(found, "the Slack author is a Stream Chat user")
	s.Equal("U0000ALICE", user["name"])
	data, found := s.chat.Channel(channel)
	s.Require().True(found)
	custom, _ := data["custom"].(map[string]any)
	s.Equal(s.config.ID, custom[ConfigField], "the channel names the agent that answers in it")
}

// One Slack thread is one thread channel however many people write in it: a mention starts
// it, and two others reply in it.
func (s *SlackChannelSuite) TestThreePeopleInOneThreadAreOneThreadChannel() {
	s.deliver(s.message("U0000ALICE", "<@U0000BOT> is the build green?", "1759740000.000100", ""), 0)
	s.deliver(s.message("U0000BOB", "it was an hour ago", "1759740000.000200", "1759740000.000100"), 0)
	s.deliver(s.message("U0000CAROL", "mine failed", "1759740000.000300", "1759740000.000100"), 0)

	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	stored := s.written(channel, 3)
	authors := map[any]bool{}
	for _, message := range stored {
		authors[message["user_id"]] = true
	}
	s.Len(authors, 3, "each person writes as a user of their own")
	s.Equal(1, s.threadChannels(), "one thread channel for the thread")
}

func (s *SlackChannelSuite) TestAnotherThreadIsAnotherThreadChannel() {
	s.deliver(s.message("U0000ALICE", "first", "1759740000.000100", ""), 0)
	s.deliver(s.message("U0000ALICE", "second", "1759740000.000500", ""), 0)

	s.NotEqual(s.threadChannel("C0000CHAN:1759740000.000100"), s.threadChannel("C0000CHAN:1759740000.000500"))
}

// Slack retries an event it got no 2xx for within three seconds, with X-Slack-Retry-Num
// (https://docs.slack.dev/apis/events-api/, «Retries»).
func (s *SlackChannelSuite) TestARetriedDeliveryIsWrittenOnce() {
	event := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100", "")
	s.deliver(event, 0)

	status, _ := s.deliver(event, 1)

	s.Equal(http.StatusOK, status, "a retry is taken, so Slack stops retrying")
	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	s.written(channel, 1)
	s.Never(func() bool { return len(s.chat.Messages(channel)) > 1 }, dropped, 20*time.Millisecond)
}

// chat.postMessage answers the bot's reply with its bot_id
// (https://docs.slack.dev/reference/methods/chat.postMessage), and the event Slack sends for it
// carries the same.
func (s *SlackChannelSuite) TestTheBotsOwnMessageIsNotWritten() {
	status, _ := s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000BOT","bot_id":"B0000BOT",`+
		`"text":"Here is the build status.","ts":"1759740000.000400","thread_ts":"1759740000.000100","channel_type":"channel"}`), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
}

// https://docs.slack.dev/reference/events/message: «message events can have a subtype».
func (s *SlackChannelSuite) TestAMessageWithASubtypeIsIgnored() {
	for _, event := range []string{
		`{"type":"message","subtype":"channel_join","channel":"C0000CHAN","user":"U0000ALICE","text":"<@U0000ALICE> has joined the channel","ts":"1759740000.000600"}`,
		`{"type":"message","subtype":"message_changed","channel":"C0000CHAN","user":"U0000ALICE","text":"edited","ts":"1759740000.000700"}`,
	} {
		status, _ := s.deliver(s.event(event), 0)
		s.Equal(http.StatusOK, status)
	}

	s.nothingLinked()
}

func (s *SlackChannelSuite) TestAnEventSignedWithAnotherAppsSecretIsRefusedOnThisAppsURL() {
	other := s.providerApp("synthetic-signing-" + s.utils.uuid())
	body := s.message("U0000ALICE", "Can you check the build?", "1759740000.000100", "")

	status, answer := s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/"+other.ProviderAppID, s.secret, body, 0)

	s.Equal(http.StatusUnauthorized, status)
	s.Contains(answer, `"code":"unauthenticated"`)
	s.nothingLinked()
}

func (s *SlackChannelSuite) TestAProviderAppNobodyHasTakesNoEvents() {
	status, answer := s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/A"+s.utils.uuid(), s.secret,
		s.message("U0000ALICE", "hello", "1759740000.000100", ""), 0)

	s.Equal(http.StatusNotFound, status)
	s.Contains(answer, "this connector takes no events here")
}

// https://docs.slack.dev/reference/events/url_verification
func (s *SlackChannelSuite) TestAURLVerificationIsAnsweredWithItsChallenge() {
	status, answer := s.deliver([]byte(`{"token":"synthetic","challenge":"synthetic-challenge-value","type":"url_verification"}`), 0)

	s.Equal(http.StatusOK, status)
	s.Equal("synthetic-challenge-value", answer)
}

// Another customer's app installed in the same workspace holds a bot connection of the same
// team; the event of this customer's app moves only this customer's.
func (s *SlackChannelSuite) TestATokensRevokedOnTheAppsURLRevokesOnlyItsCustomersConnection() {
	mine := s.bot
	s.useApp(s.data.createApp())
	theirs := s.connectedBot(s.workspace, s.slack.InstallBot())

	status, _ := s.deliver(s.revokedBot(time.Now()), 0)

	s.Equal(http.StatusOK, status)
	s.Equal(store.ConnectionNeedsReauthorization, s.status(mine))
	s.Equal(store.ConnectionConnected, s.status(theirs))
}

// Slack retries after 1 and 5 minutes (https://docs.slack.dev/apis/events-api/, «Retries»).
// The admin reconnected the app between the first delivery and this retry, so the event is
// about the grant that ended, not the one the connection holds now.
func (s *SlackChannelSuite) TestAStaleRetriedTokensRevokedAfterAReconnectDoesNotRevoke() {
	status, _ := s.deliver(s.revokedBot(time.Now().Add(-5*time.Minute)), 3)

	s.Equal(http.StatusOK, status)
	s.Equal(store.ConnectionConnected, s.status(s.bot))
}

// The whole loop: Slack's message lands in the thread channel; the message hook wakes the
// text session running on it; the session's reply, written back into the thread channel,
// leaves through the manifest's chat.postMessage template, into the same Slack thread, with
// the bot connection's token.
func (s *SlackChannelSuite) TestTheSessionsReplyLeavesIntoTheSameSlackThreadWithTheBotToken() {
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()
	s.deliver(s.message("U0000ALICE", "<@U0000BOT> is the build green?", "1759740000.000100", ""), 0)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	s.written(channel, 1)

	// Nothing runs on the thread channel yet, so the hook hands the message to a worker of the
	// customer's, with the agent the channel names, as for any agent channel.
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	select {
	case handed := <-worker.Messages():
		s.Equal(channel, handed.ChannelID)
		s.Equal(s.config.ID, handed.ConfigID)
		s.Equal("<@U0000BOT> is the build green?", handed.Text)
	case <-time.After(settleFor):
		s.Fail("the message never reached the worker")
	}

	// The worker's session on the thread channel. Opened through the manager, because one
	// opened through POST /v1/agents/sessions keeps its conversation in a channel of its own.
	opened, err := s.manager.Create(context.Background(), session.Spec{
		CustomerID: s.customerID(), ConfigID: s.config.ID, Text: true, AgentID: channel,
		UserID: "agent-" + s.utils.uuid(), LLMTarget: "en-low-latency",
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _, _ = s.manager.Close(opened.ID(), session.OwnerOf(opened.Spec())) })

	// The next message in the thread wakes the session, which answers into the thread
	// channel with source agent; Stream Chat delivers that too.
	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.written(channel, 2)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 1))
	replies := s.written(channel, 3)
	s.Equal("Hello.", replies[2]["text"])
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 2))

	s.Require().Eventually(func() bool { return len(s.slack.Posts()) == 1 }, settleFor, 10*time.Millisecond)
	post := s.slack.Posts()[0]
	s.Equal("C0000CHAN", post.Channel)
	s.Equal("1759740000.000100", post.ThreadTS, "the reply goes into the thread the message came from")
	s.Equal("Hello.", post.Text)
	s.True(post.Token == s.botToken, "sent with the bot connection's token")
}

func (s *SlackChannelSuite) TestAnAgentsReplyInAChannelLinkedToNoThreadLeavesNothing() {
	channel := "chat-" + s.utils.uuid()

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.agentReply(channel, "Hello.", false)))

	s.Never(func() bool { return len(s.slack.Posts()) > 0 }, dropped, 20*time.Millisecond)
}

func (s *SlackChannelSuite) TestAReplyStillBeingWrittenIsNotSent() {
	s.deliver(s.message("U0000ALICE", "hello", "1759740000.000100", ""), 0)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.agentReply(channel, "Hel", true)))

	s.Never(func() bool { return len(s.slack.Posts()) > 0 }, dropped, 20*time.Millisecond)
}

func (s *SlackChannelSuite) TestAFinishedReplyInAThreadChannelIsSent() {
	s.deliver(s.message("U0000ALICE", "hello", "1759740000.000100", ""), 0)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")

	s.Require().Equal(http.StatusOK, s.signedly("/v1/chat/hooks/stream", s.agentReply(channel, "Hi there.", false)))

	s.Require().Eventually(func() bool { return len(s.slack.Posts()) == 1 }, settleFor, 10*time.Millisecond)
	s.Equal("Hi there.", s.slack.Posts()[0].Text)
}

// providerApp is a Slack app of the test's customer, its record written as T40's store
// writes it, with signingSecret sealed for it.
func (s *SlackChannelSuite) providerApp(signingSecret string) store.ConnectorOAuthClient {
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

// connectedBot is the app's slack_bot connection of a workspace, made through the API and then
// given the bot token as an install leaves it: the workspace as the account, connected now.
func (s *SlackChannelSuite) connectedBot(team, token string) core.ConnectionRef {
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

// answering is an agent config of the test's app that binds the bot connection as fixed: the
// agent that answers in the workspace.
func (s *SlackChannelSuite) answering(connectionID string) store.AgentConfig {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "slack-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "en-low-latency",
		Connectors: []store.ConnectorBinding{{
			Name: "slack", ConnectorID: "slack_bot",
			Connection: store.ConnectionBinding{Type: "fixed", ConnectionID: connectionID},
		}},
	}
	s.Require().NoError(s.store.CreateAgentConfig(context.Background(), &config))
	return config
}

// deliver posts a Slack event to the test app's events URL, signed with its secret.
func (s *SlackChannelSuite) deliver(body []byte, retry int) (int, string) {
	return s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/"+s.app.ProviderAppID, s.secret, body, retry)
}

// event is a Slack event_callback in the test's workspace (https://docs.slack.dev/apis/events-api/).
func (s *SlackChannelSuite) event(inner string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + s.workspace + `","api_app_id":"` + s.app.ProviderAppID +
		`","event":` + inner + `,"type":"event_callback","event_id":"Ev` + strings.ReplaceAll(s.utils.uuid(), "-", "") +
		`","event_time":` + fmt.Sprint(time.Now().Unix()) + `}`)
}

// message is a message.channels event by user in C0000CHAN
// (https://docs.slack.dev/reference/events/message.channels), a reply in thread when
// threadTS is set.
func (s *SlackChannelSuite) message(user, text, ts, threadTS string) []byte {
	inner := map[string]string{"type": "message", "channel": "C0000CHAN", "user": user, "text": text, "ts": ts, "channel_type": "channel"}
	if threadTS != "" {
		inner["thread_ts"] = threadTS
	}
	raw, err := json.Marshal(inner)
	s.Require().NoError(err)
	return s.event(string(raw))
}

// revokedBot is tokens_revoked for the workspace's bot user, dispatched at at
// (https://docs.slack.dev/reference/events/tokens_revoked).
func (s *SlackChannelSuite) revokedBot(at time.Time) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + s.workspace + `","api_app_id":"` + s.app.ProviderAppID +
		`","event":{"type":"tokens_revoked","tokens":{"bot":["U0000BOT"]}},"type":"event_callback","event_id":"Ev0000REVOKED","event_time":` +
		fmt.Sprint(at.Unix()) + `}`)
}

// threadChannel is the thread channel the test's customer has for a thread key, once the
// bridge linked it.
func (s *SlackChannelSuite) threadChannel(threadKey string) string {
	var channel string
	err := s.store.DB().QueryRowContext(context.Background(),
		"SELECT channel_id FROM channel_threads WHERE customer_id = ? AND thread_key = ?", s.customerID(), threadKey).Scan(&channel)
	s.Require().NoError(err, "no thread channel for %s", threadKey)
	return channel
}

// threadChannels is how many thread channels the test's customer has.
func (s *SlackChannelSuite) threadChannels() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_threads WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}

// nothingLinked fails when the test's customer has a thread channel by the time a drop shows.
func (s *SlackChannelSuite) nothingLinked() {
	s.Never(func() bool { return s.threadChannels() > 0 }, dropped, 20*time.Millisecond)
}

// written waits until a thread channel holds count messages, written off the request, and
// returns them.
func (s *SlackChannelSuite) written(channel string, count int) []map[string]any {
	s.Require().Eventually(func() bool { return len(s.chat.Stored(channel)) >= count }, settleFor, 10*time.Millisecond,
		"%s holds %d messages, not %d", channel, len(s.chat.Stored(channel)), count)
	return s.chat.Stored(channel)
}

// streamDelivers delivers the message.new Stream Chat sends for the index-th message of a
// thread channel, as Stream holds it, signed with the app's secret.
func (s *SlackChannelSuite) streamDelivers(channel string, index int) int {
	stored := s.chat.Stored(channel)[index]
	data, _ := s.chat.Channel(channel)
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"channel_custom": data["custom"],
		"message": map[string]any{
			"id": stored["id"], "text": stored["text"], "user": stored["user"], "custom": stored["custom"],
		},
	})
	s.Require().NoError(err)
	return s.signedly("/v1/chat/hooks/stream", string(payload))
}

// agentReply is the message.new for an agent's written reply in a channel, as chatlog writes
// it: source agent, and generating while it is still being written.
func (s *SlackChannelSuite) agentReply(channel, text string, generating bool) string {
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"message": map[string]any{
			"id": s.utils.uuid(), "text": text, "user": map[string]string{"id": "agent", "name": "Agent"},
			"custom": map[string]any{"source": "agent", "generating": generating, "interrupted": false},
		},
	})
	s.Require().NoError(err)
	return string(payload)
}

// status is a connection's status as stored.
func (s *SlackChannelSuite) status(ref core.ConnectionRef) string {
	connection, err := s.store.ConnectorConnection(context.Background(), ref.CustomerID, ref.ConnectionID)
	s.Require().NoError(err)
	return connection.Status
}
