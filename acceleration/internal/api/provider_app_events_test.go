//go:build integration

package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/credentialstores/pgsealed"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/fakeprovider"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/schemes/oauth2code"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/verifiers/hmacheader"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
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
	// transcribed is the channel each voice session's transcript was opened for.
	transcribed *openedTranscripts
	// logged is what the router logged, at debug and up.
	logged *lockedLog
}

// botMention is how a message names the test workspace's bot, U0000BOT (fakeprovider's
// bot_user_id): https://docs.slack.dev/messaging/formatting-message-text.
const botMention = "<@U0000BOT> "

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
	s.transcribed = &openedTranscripts{}
	s.transcripts = s.transcribed.open
	s.logged = &lockedLog{}
	s.logs = s.logged
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
	s.Equal("Can you check the build?", stored[0]["text"], "without the bot's mention (AI-990 F29)")
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

// AI-906: the app's backend puts its own Slack app through the oauth-client PUT, and Slack's
// events for that app, signed with the secret put, reach the bridge. The app the put replaced
// takes none.
func (s *SlackChannelSuite) TestAnAppPutThroughTheAPIHasItsSignedEventsWrittenIntoAThreadChannel() {
	replaced, replacedSecret := s.app.ProviderAppID, s.secret
	s.app.ProviderAppID = "A" + strings.ToUpper(strings.ReplaceAll(s.utils.uuid(), "-", ""))
	s.secret = "synthetic-put-signing-" + s.utils.uuid()
	s.Require().Equal(http.StatusOK, s.serverClient.do(http.MethodPut, oauthClientPath("slack_bot"), ConnectorOAuthClientRequest{
		ClientID: "synthetic-client", ClientSecret: "synthetic-client-secret",
		ProviderAppID: s.app.ProviderAppID, SigningSecret: s.secret,
	}, nil))

	status, _ := s.deliver(s.message("U0000ALICE", "Can you check the build?", "1759740000.000100", ""), 0)

	s.Equal(http.StatusOK, status)
	stored := s.written(s.threadChannel("C0000CHAN:1759740000.000100"), 1)
	s.Equal("Can you check the build?", stored[0]["text"])
	stale, _ := s.slack.Deliver(s.server.URL+providerAppEventsPath+"slack_bot/"+replaced, replacedSecret,
		s.message("U0000ALICE", "still there?", "1759740000.000200", ""), 0)
	s.Equal(http.StatusNotFound, stale, "the app the put replaced takes no events")
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

// A thread is one episode: its first message opens one card in the omni-channel of whoever
// started it, and the messages after it, the starter's or anybody's, open none (T41).
func (s *SlackChannelSuite) TestAThreadIsOneEpisodeCardInTheOmniChannelOfWhoeverStartedIt() {
	s.deliver(s.message("U0000ALICE", "<@U0000BOT> is the build green?", "1759740000.000100", ""), 0)
	s.deliver(s.message("U0000ALICE", "the release one", "1759740000.000200", "1759740000.000100"), 0)
	s.deliver(s.message("U0000BOB", "it was an hour ago", "1759740000.000300", "1759740000.000100"), 0)

	thread := s.threadChannel("C0000CHAN:1759740000.000100")
	s.written(thread, 3)
	omni := s.omniChannel("U0000ALICE")
	card := s.written(omni, 1)[0]
	s.Never(func() bool { return len(s.chat.Stored(omni)) > 1 }, dropped, 20*time.Millisecond)
	custom, _ := card["custom"].(map[string]any)
	s.Equal("slack", custom["source"])
	s.Equal("in_progress", custom["status"])
	s.Equal("agent:"+thread, custom["thread_channel"])
	s.NotEmpty(custom["started_at"])
	s.Equal(1, s.threadChannels(), "one thread channel for the thread")
}

// One person's threads are episodes of one omni-channel: Monday's thread is a card beside
// Tuesday's, which is how the agent on Tuesday learns of Monday (T56).
func (s *SlackChannelSuite) TestTheSamePersonsNextThreadIsAnotherCardInTheSameOmniChannel() {
	s.deliver(s.message("U0000ALICE", "first", "1759740000.000100", ""), 0)
	s.deliver(s.message("U0000ALICE", "second", "1759740000.000500", ""), 0)

	omni := s.omniChannel("U0000ALICE")
	cards := s.written(omni, 2)
	threads := map[any]bool{}
	for _, card := range cards {
		custom, _ := card["custom"].(map[string]any)
		threads[custom["thread_channel"]] = true
	}
	s.Len(threads, 2)
}

// A card has a source, so the message.new Stream Chat sends for it reaches nobody: the
// omni-channel names its agent config, and a message there without a source would be handed
// to a worker.
func (s *SlackChannelSuite) TestAnEpisodeCardStartsNoSession() {
	s.deliver(s.message("U0000ALICE", "Can you check the build?", "1759740000.000100", ""), 0)
	omni := s.omniChannel("U0000ALICE")
	s.written(omni, 1)
	worker, release := s.dispatch.Register(s.customerID(), dispatch.Registration{Capacity: 1})
	defer release()

	s.Equal(http.StatusOK, s.streamDelivers(omni, 0))

	select {
	case message := <-worker.Messages():
		s.Failf("an episode card started a session", "it reached a worker in %s", message.ChannelID)
	case <-time.After(dropped):
	}
}

// omniChannel is the id of the omni-channel the contact map gives a Slack user of the test's
// workspace, for the test's agent.
func (s *SlackChannelSuite) omniChannel(user string) string {
	var cid string
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT conversation_id FROM contact_map WHERE customer_id = ? AND agent_config_id = ? AND kind = 'slack' AND address = ?",
		s.customerID(), s.config.ID, s.workspace+":"+user).Scan(&cid), "no contact for %s", user)
	return strings.TrimPrefix(cid, "agent:")
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

// AI-990 F21, F30, AI-1053 F67: a message the manifest skips is answered 200 as before, and
// logged at info with the provider app and the rule that skipped it, without its text.
func (s *SlackChannelSuite) TestASkippedMessageIsLoggedWithTheRuleThatSkippedIt() {
	for rule, body := range map[string][]byte{
		"skip_if_present $.event.subtype": s.event(`{"type":"message","subtype":"channel_join","channel":"C0000CHAN","user":"U0000ALICE",` +
			`"text":"synthetic join text F21","ts":"1759740000.002100"}`),
		"match $.authorizations[0].is_bot": s.eventFor(`{"type":"message","channel":"D0000PEOPLE","user":"U0000ALICE",`+
			`"text":"synthetic direct text F30","ts":"1759740000.002200","channel_type":"im"}`,
			`{"team_id":"`+s.workspace+`","user_id":"U0000KANAT","is_bot":false}`),
	} {
		status, _ := s.deliver(body, 0)
		s.Equal(http.StatusOK, status)
		var line string
		for _, l := range strings.Split(s.logged.String(), "\n") {
			if strings.Contains(l, "skipped a connector event's message") && strings.Contains(l, `rule="`+rule+`"`) {
				line = l
			}
		}
		s.Contains(line, "level=INFO")
		s.Contains(line, "connector=slack_bot")
		s.Contains(line, "provider_app="+s.app.ProviderAppID)
	}
	s.nothingLinked()
	s.NotContains(s.logged.String(), "synthetic join text F21")
	s.NotContains(s.logged.String(), "synthetic direct text F30")
}

// AI-989: an agent's slack_send_message through a person's user token posts as that person,
// with no bot_id and no subtype; the event names the posting app in app_id. The shape is a
// live event of 2026-10-09 (slack_bot.yaml, revision 4), with synthetic ids. It is skipped
// even when it mentions the bot.
func (s *SlackChannelSuite) TestAnAppsPostThroughAPersonsUserTokenIsNotWritten() {
	status, _ := s.deliver(s.event(`{"type":"message","user":"U0000JUSTIN","ts":"1759740000.000800","app_id":"A0000OTHERAPP",`+
		`"text":"<@U0000BOT> hello from another agent *Sent using* Another app","team":"`+s.workspace+`",`+
		`"blocks":[{"type":"context","block_id":"ctx","elements":[{"type":"mrkdwn","text":"*Sent using* Another app","verbatim":false}]}],`+
		`"channel":"C0000CHAN","event_ts":"1759740000.000800","channel_type":"channel"}`), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
}

// A message a person types in a Slack client carries client_msg_id and no app_id (a live
// event of 2026-10-09, synthetic ids), and is written when it mentions the bot.
func (s *SlackChannelSuite) TestAMessageAPersonTypesInSlackIsWritten() {
	status, _ := s.deliver(s.event(`{"type":"message","user":"U0000ALICE","client_msg_id":"00000000-0000-4000-8000-000000000001",`+
		`"ts":"1759740000.000900","text":"<@U0000BOT> Hello","team":"`+s.workspace+`",`+
		`"blocks":[{"type":"rich_text","block_id":"rt","elements":[{"type":"rich_text_section","elements":[{"type":"text","text":"Hello"}]}]}],`+
		`"channel":"C0000CHAN","event_ts":"1759740000.000900","channel_type":"channel"}`), 0)

	s.Equal(http.StatusOK, status)
	stored := s.written(s.threadChannel("C0000CHAN:1759740000.000900"), 1)
	s.Equal("Hello", stored[0]["text"])
}

// AI-989: an app a person also installed with user scopes gets that person's direct messages
// with other people under the person's install (authorizations is_bot false, seen live on
// 2026-10-09). None is a message to the bot.
func (s *SlackChannelSuite) TestAnEventOnlyAPersonsInstallSeesIsNotWritten() {
	status, _ := s.deliver(s.eventFor(`{"type":"message","channel":"D0000PEOPLE","user":"U0000ALICE","text":"see you at the sync",`+
		`"ts":"1759740000.001000","channel_type":"im"}`, `{"team_id":"`+s.workspace+`","user_id":"U0000KANAT","is_bot":false}`), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
}

// AI-989: a thread the bot was mentioned in is linked, and the bot answers there without a
// mention. A top-level message that mentions nobody, in the same channel, is another thread.
func (s *SlackChannelSuite) TestAnotherTopLevelMessageInAChannelWithALinkedThreadIsNotWritten() {
	s.messaged("U0000ALICE", "check the build", "1759740000.003100", "")

	s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000BOB","text":"lunch?",`+
		`"ts":"1759740000.003200","channel_type":"channel"}`), 0)

	s.Never(func() bool { return s.threadChannels() > 1 }, dropped, 20*time.Millisecond)
}

// AI-989: connections pinned to revision 3 of slack_bot keep working: its rule answers a
// channel message without a mention, and does not skip an app's post or an install of a person.
// Revision 3 is the manifest as shipped before revision 4 (testdata/slack_bot_rev3.yaml).
func (s *SlackChannelSuite) TestAConnectionPinnedToRevisionThreeStillAnswersAChannelMessageWithoutAMention() {
	raw, err := os.ReadFile("testdata/slack_bot_rev3.yaml")
	s.Require().NoError(err)
	manifest, err := core.ParseManifest(raw)
	s.Require().NoError(err)
	s.Require().Equal(3, manifest.Revision)
	_, err = s.store.DB().NewInsert().Model(&store.ConnectorDefinition{
		CustomerID: store.BuiltinCustomer, ID: manifest.ID, Revision: manifest.Revision, Name: manifest.Name,
		Category: manifest.Category, Description: manifest.Description, Manifest: manifest, CreatedAt: time.Now().UTC(),
	}).On("CONFLICT DO NOTHING").Exec(context.Background())
	s.Require().NoError(err)
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	s.Require().NoError(credentials.Update(context.Background(), s.bot, func(state *core.CredentialState, _ func() error) (bool, error) {
		state.DefinitionRevision = 3
		return true, nil
	}))

	s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000ALICE","text":"lunch?",`+
		`"ts":"1759740000.002100","channel_type":"channel"}`), 0)
	s.written(s.threadChannel("C0000CHAN:1759740000.002100"), 1)

	s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000JUSTIN","app_id":"A0000OTHERAPP","text":"hi",`+
		`"ts":"1759740000.002200","channel_type":"channel"}`), 0)
	s.deliver(s.eventFor(`{"type":"message","channel":"D0000PEOPLE","user":"U0000ALICE","text":"x",`+
		`"ts":"1759740000.002300","channel_type":"im"}`, `{"team_id":"`+s.workspace+`","user_id":"U0000KANAT","is_bot":false}`), 0)
	s.Never(func() bool { return s.threadChannels() > 1 }, dropped, 20*time.Millisecond)
}

// AI-989: in a channel the bot answers only a message that mentions it.
func (s *SlackChannelSuite) TestAChannelMessageThatDoesNotMentionTheBotIsNotWritten() {
	status, _ := s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000ALICE","text":"lunch?",`+
		`"ts":"1759740000.001100","channel_type":"channel"}`), 0)

	s.Equal(http.StatusOK, status)
	s.nothingLinked()
}

// AI-989: once mentioned in a thread, the bot answers every reply there without a mention.
func (s *SlackChannelSuite) TestAReplyInAThreadTheBotWasMentionedInIsWrittenWithoutAMention() {
	channel := s.messaged("U0000ALICE", "can you check the build?", "1759740000.001200", "")

	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.001300", "1759740000.001200"), 0)

	stored := s.written(channel, 2)
	s.Equal("and the deploy?", stored[1]["text"])
}

// AI-989: a mention in a thread of people starts the bot there; the replies before it are
// not written, the ones after it are.
func (s *SlackChannelSuite) TestAMentionInAThreadOfPeopleStartsTheBotThere() {
	s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.001500", "1759740000.001400"), 0)
	s.nothingLinked()

	s.deliver(s.message("U0000BOB", botMention+"do you know?", "1759740000.001600", "1759740000.001400"), 0)
	s.deliver(s.message("U0000ALICE", "thanks", "1759740000.001700", "1759740000.001400"), 0)

	stored := s.written(s.threadChannel("C0000CHAN:1759740000.001400"), 2)
	s.Equal([]any{"do you know?", "thanks"}, []any{stored[0]["text"], stored[1]["text"]})
}

// AI-989: a direct message to the bot is to it without a mention.
func (s *SlackChannelSuite) TestADirectMessageIsWrittenWithoutAMention() {
	status, _ := s.deliver(s.event(`{"type":"message","channel":"D0000BOT","user":"U0000ALICE","text":"Blah",`+
		`"ts":"1759740000.001800","channel_type":"im"}`), 0)

	s.Equal(http.StatusOK, status)
	s.Equal("Blah", s.written(s.threadChannel("D0000BOT:1759740000.001800"), 1)[0]["text"])
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

// The production path end to end: Slack's message lands in the thread channel; Stream Chat's
// message.new reaches the message hook; the Router opens a persistent text session on the
// thread channel itself; the reply's final text is stored in the thread channel; the
// conversation hands it to the bridge, which posts it into the same Slack thread with the
// bot connection's token.
func (s *SlackChannelSuite) TestTheRoutersSessionAnswersInTheThreadChannelAndTheReplyLeavesIntoTheSameSlackThread() {
	channel := s.messaged("U0000ALICE", "<@U0000BOT> is the build green?", "1759740000.000100", "")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	post := s.posted(1)[0]
	s.Equal("C0000CHAN", post.Channel)
	s.Equal("1759740000.000100", post.ThreadTS, "the reply goes into the thread the message came from")
	s.Equal("Noted.", post.Text)
	s.True(post.Token == s.botToken, "sent with the bot connection's token")
	reply := s.agentsReply(channel)
	s.Equal("Noted.", reply["text"], "the reply is kept in the thread channel, not in a support channel of its own")
	s.True(s.told("is the build green?"), "the model is told the person's message")
}

// The session stays on the thread channel for DetachedGrace, so the second message is the
// same conversation, and the model reads the first one in it.
func (s *SlackChannelSuite) TestASecondMessageInTheThreadIsAnsweredInTheSameConversation() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.posted(1)

	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	stored := s.written(channel, 3)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, len(stored)-1))

	posts := s.posted(2)
	s.Equal("1759740000.000100", posts[1].ThreadTS)
	s.True(s.told("is the build green?", "and the deploy?"), "the second turn reads the first message as history")
}

// Stream may deliver one message.new twice; the session answers it once.
func (s *SlackChannelSuite) TestAMessageStreamDeliversTwiceIsAnsweredOnce() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.posted(1)
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// A finished reply is handed over each time it is written; a message a login later marks is
// written again (conversation.Service.OnFinishedReply). It leaves once.
func (s *SlackChannelSuite) TestAFinishedReplyHandedOverTwiceIsSentOnce() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.posted(1)
	reply := s.agentsReply(channel)

	s.bridge.(*channelbridge.Bridge).Reply(conversation.FinishedReply{
		Customer: s.customerID(), CID: "agent:" + channel, MessageID: reply["id"].(string), Text: "Noted.",
	})

	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// Slack answering 503 twice is a reply sent the third time, once.
func (s *SlackChannelSuite) TestAReplyTheProviderFailsToTakeIsSentAgainOnce() {
	s.slack.FailPosts(2)
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Equal("Noted.", s.posted(1)[0].Text)
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// A reply every send of which failed is unclaimed again, so the next hand-off of it sends it,
// and only once however often it is handed over after that.
func (s *SlackChannelSuite) TestAReplyThatNeverGotThroughIsSentByTheNextHandOff() {
	s.slack.FailPosts(3)
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Require().Eventually(func() bool { return s.slack.Hits(fakeprovider.PathChatPostMessage) == 3 }, settleFor, 10*time.Millisecond,
		"the first send and the two the backoff allows")
	s.Require().Eventually(func() bool { return s.claimed(channel, "reply") == 0 }, settleFor, 10*time.Millisecond,
		"after its last failed send the reply is unclaimed")
	reply := s.agentsReply(channel)
	s.Empty(s.slack.Posts())
	finished := conversation.FinishedReply{Customer: s.customerID(), CID: "agent:" + channel, MessageID: reply["id"].(string), Text: "Noted."}

	s.bridge.(*channelbridge.Bridge).Reply(finished)
	s.posted(1)
	s.bridge.(*channelbridge.Bridge).Reply(finished)

	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// AI-990 F28: Slack refusing a reply for a reason other than its token, HTTP 200 with
// «"ok": false» and an error name (https://docs.slack.dev/reference/methods/chat.postMessage),
// leaves no reply row behind, so the next hand-off sends it, once. The log names Slack's error.
func (s *SlackChannelSuite) TestAReplySlackRefusesIsUnclaimedAndItsErrorIsLogged() {
	s.slack.RefusePosts(1, "not_in_channel")
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Require().Eventually(func() bool { return s.slack.Hits(fakeprovider.PathChatPostMessage) == 1 }, settleFor, 10*time.Millisecond)

	s.Require().Eventually(func() bool { return s.claimed(channel, "reply") == 0 }, settleFor, 10*time.Millisecond,
		"a refused reply is not kept as sent")
	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), "not_in_channel") }, settleFor, 10*time.Millisecond,
		"the log names the error Slack refused the reply with")
	s.Empty(s.slack.Posts())
	finished := conversation.FinishedReply{Customer: s.customerID(), CID: "agent:" + channel, MessageID: s.agentsReply(channel)["id"].(string), Text: "Noted."}

	s.bridge.(*channelbridge.Bridge).Reply(finished)
	s.posted(1)
	s.Require().Eventually(func() bool { return s.claimed(channel, "reply") == 1 }, settleFor, 10*time.Millisecond,
		"a reply Slack took is kept as sent")
	s.bridge.(*channelbridge.Bridge).Reply(finished)

	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// AI-990 F28: a 2xx answer the bridge cannot read may be of a reply Slack posted, so the reply
// keeps its claim and a later hand-off does not post it again. The log says why.
func (s *SlackChannelSuite) TestAReplyAnsweredWithABodyTheBridgeCannotReadKeepsItsClaim() {
	s.slack.GarblePosts(1)
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.posted(1)
	s.Require().Eventually(func() bool { return strings.Contains(s.logged.String(), "a body it does not read") }, settleFor, 10*time.Millisecond,
		"the log says the answer was not read")

	s.Equal(1, s.claimed(channel, "reply"), "the reply Slack may have posted is kept as sent")
	s.bridge.(*channelbridge.Bridge).Reply(conversation.FinishedReply{
		Customer: s.customerID(), CID: "agent:" + channel, MessageID: s.agentsReply(channel)["id"].(string), Text: "Noted.",
	})
	s.Never(func() bool { return len(s.slack.Posts()) > 1 }, dropped, 20*time.Millisecond)
}

// AI-990 F31a: a mention's first delivery linked its thread and failed before it took the
// replies that waited, so Slack retries it. The retry takes them, though the link is not new.
func (s *SlackChannelSuite) TestAMentionRetriedAfterItLinkedItsThreadTakesTheRepliesThatWaited() {
	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.Require().Equal(1, s.waiting())
	channel := s.linkedAsAFailedDeliveryLeftIt("C0000CHAN:1759740000.000100")

	status, _ := s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100", ""), 1)

	s.Require().Equal(http.StatusOK, status)
	stored := s.written(channel, 2)
	s.Equal([]any{"is the build green?", "and the deploy?"}, []any{stored[0]["text"], stored[1]["text"]})
	s.Zero(s.waiting(), "the reply no longer waits")
}

// AI-990 F31a: a retried mention that was taken before takes the replies that waited for it
// all the same, and each of them is handed out once when Slack delivers the mention twice at once.
func (s *SlackChannelSuite) TestAMentionDeliveredTwiceWritesEachReplyThatWaitedOnce() {
	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.Require().Equal(1, s.waiting())
	mention := s.message("U0000ALICE", "is the build green?", "1759740000.000100", "")
	channel := s.linkedAsAFailedDeliveryLeftIt("C0000CHAN:1759740000.000100")
	fresh, err := s.store.ClaimChannelThreadMessage(context.Background(), channel, store.ClaimInbound, "1759740000.000100")
	s.Require().NoError(err)
	s.Require().True(fresh, "the failed delivery had claimed the mention too")

	var delivered sync.WaitGroup
	for retry := 1; retry <= 2; retry++ {
		delivered.Add(1)
		go func() {
			defer delivered.Done()
			s.deliver(mention, retry)
		}()
	}
	delivered.Wait()

	stored := s.written(channel, 1)
	s.Equal("and the deploy?", stored[0]["text"], "the mention was claimed, so only the reply is written")
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 1 }, dropped, 20*time.Millisecond)
	s.Zero(s.waiting())
}

// AI-990 F31a: Slack retries a mention whose first delivery failed
// (https://docs.slack.dev/apis/events-api/, «Retries»), so a reply in its thread can arrive
// before the mention links the thread. The reply waits for the link and is written after it.
func (s *SlackChannelSuite) TestAReplyThatArrivesBeforeTheRetriedMentionIsWrittenAfterIt() {
	status, _ := s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.Require().Equal(http.StatusOK, status)
	s.nothingLinked()

	status, _ = s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100", ""), 1)

	s.Require().Equal(http.StatusOK, status)
	stored := s.written(s.threadChannel("C0000CHAN:1759740000.000100"), 2)
	s.Equal([]any{"is the build green?", "and the deploy?"}, []any{stored[0]["text"], stored[1]["text"]},
		"the mention first, then the reply that waited for it")
	s.Zero(s.waiting(), "the reply no longer waits")
}

// AI-990 F31a: mentions and replies in their threads that arrive together are all written,
// whichever of each pair the router reads first.
func (s *SlackChannelSuite) TestRepliesArrivingWithTheirMentionsAreAllWritten() {
	const threads = 50
	var delivered sync.WaitGroup
	for i := range threads {
		ts := fmt.Sprintf("1759740000.%06d", 1000+10*i)
		for _, body := range [][]byte{
			s.message("U0000ALICE", "is the build green?", ts, ""),
			s.message("U0000BOB", "and the deploy?", fmt.Sprintf("1759740000.%06d", 1000+10*i+1), ts),
		} {
			delivered.Add(1)
			go func() {
				defer delivered.Done()
				s.deliver(body, 0)
			}()
		}
	}
	delivered.Wait()

	for i := range threads {
		s.written(s.threadChannel(fmt.Sprintf("C0000CHAN:1759740000.%06d", 1000+10*i)), 2)
	}
}

// AI-990 F29, F31a: a reply that waited comes back through take when the mention links its
// thread, so the bot's mention is left out of it too. It waits with the mention only while the
// connection has no bot_user_id to read it by, so the test takes that away and gives it back.
func (s *SlackChannelSuite) TestAReplyThatWaitedIsWrittenWithoutTheBotsMention() {
	credentials, err := pgsealed.New(s.store, s.sealer)
	s.Require().NoError(err)
	metadata := func(values map[string]string) {
		s.Require().NoError(credentials.Update(context.Background(), s.bot, func(state *core.CredentialState, _ func() error) (bool, error) {
			state.Metadata = values
			return true, nil
		}))
	}
	metadata(map[string]string{"team_id": s.workspace})
	s.deliver(s.message("U0000BOB", botMention+"and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	s.Require().Equal(1, s.waiting(), "the reply waits")
	metadata(map[string]string{"team_id": s.workspace, "bot_user_id": "U0000BOT"})

	s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100", ""), 1)

	stored := s.written(s.threadChannel("C0000CHAN:1759740000.000100"), 2)
	s.Equal([]any{"is the build green?", "and the deploy?"}, []any{stored[0]["text"], stored[1]["text"]})
}

// AI-990 F31a: a message that starts its thread and is not to the bot is not kept: no later
// message takes it, since the one that links its thread is a reply after it (AI-989).
func (s *SlackChannelSuite) TestAChannelMessageNotToTheBotDoesNotWait() {
	s.deliver(s.event(`{"type":"message","channel":"C0000CHAN","user":"U0000ALICE","text":"lunch?",`+
		`"ts":"1759740000.001100","channel_type":"channel"}`), 0)
	s.deliver(s.message("U0000BOB", "sure", "1759740000.001200", "1759740000.001100"), 0)

	s.Require().Eventually(func() bool { return s.waiting() == 1 }, settleFor, 10*time.Millisecond, "the reply waits")
	s.Never(func() bool { return s.waiting() > 1 }, dropped, 20*time.Millisecond, "the message that started the thread does not")
}

// AI-990 F31a: a reply that waited for a mention is written once, though Slack delivers it
// again once the thread is linked.
func (s *SlackChannelSuite) TestAReplyThatWaitedAndIsDeliveredAgainIsWrittenOnce() {
	reply := s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100")
	s.deliver(reply, 0)
	s.deliver(s.message("U0000ALICE", "is the build green?", "1759740000.000100", ""), 1)
	channel := s.threadChannel("C0000CHAN:1759740000.000100")
	s.written(channel, 2)

	s.deliver(reply, 1)

	s.Never(func() bool { return len(s.chat.Stored(channel)) > 2 }, dropped, 20*time.Millisecond)
}

// Another router holds the thread's turn: this one waits, and answers once it is let go.
func (s *SlackChannelSuite) TestAThreadAnotherRouterIsAnsweringWaitsForItsTurn() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	taken, err := s.store.TakeChannelThreadTurn(context.Background(), channel, "another-router", time.Now().Add(time.Minute))
	s.Require().NoError(err)
	s.Require().True(taken)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Never(func() bool { return len(s.slack.Posts()) > 0 }, 500*time.Millisecond, 20*time.Millisecond,
		"two routers never answer one thread at once")
	s.Require().NoError(s.store.ReleaseChannelThreadTurn(context.Background(), channel, "another-router"))
	s.posted(1)
}

// A worker that opens a session for a thread channel through POST /v1/agents/sessions names
// it by agent_id, as the Go SDK's Dispatch.Conversation does; the session keeps its
// conversation in the thread channel, where the hook's turns and the bridge find it.
func (s *SlackChannelSuite) TestASessionOpenedThroughTheAPIForAThreadChannelHoldsItsConversationThere() {
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	opened := s.serverClient.createSession(CreateSessionRequest{Agent: &s.config.Name, AgentId: &channel})

	s.Equal("agent:"+channel, value(opened.ConversationId))
	s.Equal(channel, opened.AgentId, "the conversation names its agent, the channel's")
}

// Slack refuses a revoked bot token with HTTP 200 and «"ok": false, "error": "invalid_auth"»
// (https://docs.slack.dev/reference/methods/chat.postMessage), which the transport does not
// read; the bridge has the scheme classify it and the resolver end the grant.
func (s *SlackChannelSuite) TestAReplySlackRefusesForItsTokenMovesTheConnection() {
	s.slack.RevokeBot(s.botToken)
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))

	s.Require().Eventually(func() bool {
		return s.status(s.bot) == store.ConnectionNeedsReauthorization
	}, settleFor, 10*time.Millisecond)
	s.Empty(s.slack.Posts())
}

// messaged is the thread channel a Slack message lands in, once the bridge wrote it there.
func (s *SlackChannelSuite) messaged(user, text, ts, threadTS string) string {
	s.deliver(s.message(user, text, ts, threadTS), 0)
	channel := s.threadChannel("C0000CHAN:" + firstNonEmpty(threadTS, ts))
	s.written(channel, 1)
	return channel
}

// posted waits until Slack took count replies, and returns them.
func (s *SlackChannelSuite) posted(count int) []fakeprovider.Post {
	s.Require().Eventually(func() bool { return len(s.slack.Posts()) >= count }, settleFor, 10*time.Millisecond,
		"Slack took %d replies, not %d", len(s.slack.Posts()), count)
	return s.slack.Posts()
}

// agentsReply is the agent's reply in a thread channel, written as the channel's own agent.
func (s *SlackChannelSuite) agentsReply(channel string) map[string]any {
	for _, stored := range s.chat.Stored(channel) {
		if stored["user_id"] == channel {
			return stored
		}
	}
	s.FailNow("the thread channel holds no reply of its agent")
	return nil
}

// linkedAsAFailedDeliveryLeftIt links the Slack thread key to a thread channel of the test's
// bot, as a mention's delivery does before it fails, and returns the channel.
func (s *SlackChannelSuite) linkedAsAFailedDeliveryLeftIt(key string) string {
	channel, ts, _ := strings.Cut(key, ":")
	thread := store.ChannelThread{
		ChannelID: conversation.ThreadChannelPrefix + s.utils.uuid(), CustomerID: s.customerID(),
		ConnectorID: "slack_bot", ProviderUnitID: s.workspace, ThreadKey: key,
		ConnectionID: s.bot.ConnectionID, ThreadParts: map[string]string{"channel": channel, "thread_ts": ts},
		StreamAppPK: s.app.StreamAppPK,
	}
	created, err := s.store.LinkChannelThread(context.Background(), &thread)
	s.Require().NoError(err)
	s.Require().True(created)
	return thread.ChannelID
}

// waiting is how many replies of the test's customer wait for their thread's link.
func (s *SlackChannelSuite) waiting() int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_waiting WHERE customer_id = ?", s.customerID()).Scan(&count))
	return count
}

// claimed is how many of a thread channel's messages one step holds claimed.
func (s *SlackChannelSuite) claimed(channel, kind string) int {
	var count int
	s.Require().NoError(s.store.DB().QueryRowContext(context.Background(),
		"SELECT count(*) FROM channel_thread_messages WHERE channel_id = ? AND kind = ?", channel, kind).Scan(&count))
	return count
}

// told is whether one request to a model of the suite's held every text in its input.
func (s *SlackChannelSuite) told(texts ...string) bool {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	for _, model := range s.noted {
		for _, request := range model.requests() {
			input := fmt.Sprint(request.Input)
			held := true
			for _, text := range texts {
				held = held && strings.Contains(input, text)
			}
			if held {
				return true
			}
		}
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
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
		state.Metadata = map[string]string{"team_id": team, "bot_user_id": "U0000BOT"}
		state.ConnectedAt = time.Now().UTC()
		return true, nil
	}))
	return ref
}

// answering is an agent config of the test's app that binds the bot connection as fixed: the
// agent that answers in the workspace.
func (s *SlackChannelSuite) answering(connectionID string) store.AgentConfig {
	config := store.AgentConfig{
		CustomerID: s.customerID(), Name: "slack-" + s.utils.uuid(), Mode: store.AgentModeText, LLM: "noted/noted-model",
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

// event is a Slack event_callback in the test's workspace (https://docs.slack.dev/apis/events-api/),
// visible to the app's bot install.
func (s *SlackChannelSuite) event(inner string) []byte {
	return s.eventFor(inner, `{"team_id":"`+s.workspace+`","user_id":"U0000BOT","is_bot":true}`)
}

// eventFor is a Slack event_callback whose authorizations names the one install authorization.
func (s *SlackChannelSuite) eventFor(inner, authorization string) []byte {
	return []byte(`{"token":"synthetic","team_id":"` + s.workspace + `","api_app_id":"` + s.app.ProviderAppID +
		`","event":` + inner + `,"type":"event_callback","event_id":"Ev` + strings.ReplaceAll(s.utils.uuid(), "-", "") +
		`","event_time":` + fmt.Sprint(time.Now().Unix()) + `,"authorizations":[` + authorization + `]}`)
}

// message is a message.channels event by user in C0000CHAN
// (https://docs.slack.dev/reference/events/message.channels), a reply in thread when
// threadTS is set. A message that starts a thread mentions the bot, which it must to be
// answered (AI-989), so its text is the mention and text.
func (s *SlackChannelSuite) message(user, text, ts, threadTS string) []byte {
	if threadTS == "" {
		text = botMention + text
	}
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
// thread channel, shaped as Stream sends it, signed with the app's secret.
func (s *SlackChannelSuite) streamDelivers(channel string, index int) int {
	data, _ := s.chat.Channel(channel)
	payload, err := json.Marshal(map[string]any{
		"type": "message.new", "cid": "agent:" + channel, "channel_id": channel, "channel_type": "agent",
		"channel_custom": data["custom"],
		"message":        s.chat.Delivered(channel)[index],
	})
	s.Require().NoError(err)
	return s.signedly("/v1/chat/hooks/stream", string(payload))
}

// status is a connection's status as stored.
func (s *SlackChannelSuite) status(ref core.ConnectionRef) string {
	connection, err := s.store.ConnectorConnection(context.Background(), ref.CustomerID, ref.ConnectionID)
	s.Require().NoError(err)
	return connection.Status
}
