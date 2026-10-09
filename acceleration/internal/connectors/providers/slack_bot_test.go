package providers_test

import (
	"encoding/json"
	"io/fs"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/providers"
)

// SlackBotSuite reads the built-in slack_bot manifest against events shaped as live Events
// API deliveries of 2026-10-09 were, with synthetic ids (AI-989).
type SlackBotSuite struct {
	suite.Suite
	manifest core.Manifest
	// installed is what the bot install captured: the connection's metadata.
	installed map[string]string
}

func TestSlackBotSuite(t *testing.T) {
	suite.Run(t, new(SlackBotSuite))
}

func (s *SlackBotSuite) SetupSuite() {
	raw, err := fs.ReadFile(providers.FS, "slack_bot.yaml")
	s.Require().NoError(err)
	s.manifest, err = core.ParseManifest(raw)
	s.Require().NoError(err)
	s.Require().NotNil(s.manifest.Channel)
	s.installed = map[string]string{"team_id": "T0000TEAM", "bot_user_id": "U0000BOT"}
}

// A person types in a channel the bot is in: the bot install sees it.
func (s *SlackBotSuite) TestAPersonsChannelMessageIsRead() {
	s.Len(s.read(s.event(person("hello"), bot)), 1)
}

// An app posting with a person's user token, such as an agent's slack_send_message, posts
// as that person with no bot_id and no subtype; the event names the app in app_id.
func (s *SlackBotSuite) TestAnAppsPostThroughAPersonsUserTokenIsNotRead() {
	post := person("hello from another agent *Sent using* Another app")
	delete(post, "client_msg_id")
	post["app_id"] = "A0000OTHER"
	s.Empty(s.read(s.event(post, bot)))
}

// The bot's own reply carries bot_id and app_id.
func (s *SlackBotSuite) TestTheBotsOwnReplyIsNotRead() {
	reply := person("Hello! How can I help you today?")
	delete(reply, "client_msg_id")
	reply["user"], reply["bot_id"], reply["app_id"], reply["thread_ts"] = "U0000BOT", "B0000BOT", "A0000APP", "1759740000.000100"
	s.Empty(s.read(s.event(reply, bot)))
}

// A person who also installed the app with user scopes has their direct messages with other
// people delivered under their own install; none is a message to the bot.
func (s *SlackBotSuite) TestAnEventOnlyAPersonsInstallSeesIsNotRead() {
	dm := person("a direct message between two people")
	dm["channel"], dm["channel_type"] = "D0000PEOPLE", "im"
	s.Empty(s.read(s.event(dm, map[string]any{"team_id": "T0000TEAM", "user_id": "U0000KANAT", "is_bot": false})))
}

// In a channel, a message that mentions the bot addresses it; one that does not, does not.
func (s *SlackBotSuite) TestAChannelMessageAddressesTheBotOnlyWhenItMentionsIt() {
	mentioned := s.read(s.event(person("<@U0000BOT> can you check the build?"), bot))
	other := s.read(s.event(person("<@U0000ALICE> can you check the build?"), bot))

	s.Require().Len(mentioned, 1)
	s.Require().Len(other, 1)
	s.True(s.manifest.Channel.Messages.Addresses(mentioned[0], s.installed))
	s.False(s.manifest.Channel.Messages.Addresses(other[0], s.installed))
}

// A direct message to the bot addresses it without a mention.
func (s *SlackBotSuite) TestADirectMessageAddressesTheBot() {
	dm := person("Blah")
	dm["channel"], dm["channel_type"] = "D0000BOT", "im"
	read := s.read(s.event(dm, bot))

	s.Require().Len(read, 1)
	s.True(s.manifest.Channel.Messages.Addresses(read[0], s.installed))
}

// A connection whose install captured no bot_user_id is never mentioned: a mention of the
// empty user id is not one of the bot.
func (s *SlackBotSuite) TestAConnectionWithoutABotUserIsNeverMentioned() {
	read := s.read(s.event(person("<@> hello"), bot))

	s.Require().Len(read, 1)
	s.False(s.manifest.Channel.Messages.Addresses(read[0], map[string]string{"team_id": "T0000TEAM"}))
}

// AI-990 F29: the agent gets what the person said to the bot, without the bot's own mention.
// A mention of anybody else stays, and a message that is only the mention keeps it.
func (s *SlackBotSuite) TestTheAgentGetsTheTextWithoutTheBotsMention() {
	rule := s.manifest.Channel.Messages
	for text, want := range map[string]string{
		"<@U0000BOT> is the build green?":          "is the build green?",
		"is the build green, <@U0000BOT>?":         "is the build green, ?",
		"ask <@U0000BOT> about <@U0000ALICE>'s PR": "ask about <@U0000ALICE>'s PR",
		"<@U0000BOT> <@U0000BOT> hello":            "hello",
		"<@U0000BOT>\nfirst line\nsecond line":     "first line\nsecond line",
		"<@U0000ALICE> can you check the build?":   "<@U0000ALICE> can you check the build?",
		"<@U0000BOT>":                              "<@U0000BOT>",
		"Blah":                                     "Blah",
	} {
		s.Equal(want, rule.WithoutMention(text, s.installed), text)
	}
	s.Equal("<@U0000BOT> hi", rule.WithoutMention("<@U0000BOT> hi", map[string]string{"team_id": "T0000TEAM"}),
		"a connection without a bot_user_id has no mention to take out")
}

// AI-990 F21, F30: every event slack_bot leaves out names the rule that did, for the events
// route to log at debug.
func (s *SlackBotSuite) TestEachSkippedEventNamesItsRule() {
	reply := person("Here is the build status.")
	reply["bot_id"], reply["app_id"] = "B0000BOT", "A0000APP"
	join := person("<@U0000ALICE> has joined the channel")
	join["subtype"] = "channel_join"
	post := person("hello from another agent")
	post["app_id"] = "A0000OTHER"
	for rule, body := range map[string][]byte{
		"skip_if_present $.event.bot_id":   s.event(reply, bot),
		"skip_if_present $.event.subtype":  s.event(join, bot),
		"skip_if_present $.event.app_id":   s.event(post, bot),
		"match $.authorizations[0].is_bot": s.event(person("hi"), map[string]any{"team_id": "T0000TEAM", "user_id": "U0000KANAT", "is_bot": false}),
		"match $.event.type":               s.event(map[string]any{"type": "reaction_added", "user": "U0000ALICE"}, bot),
	} {
		read, err := s.manifest.Channel.Read("slack_bot", body)
		s.Require().NoError(err)
		s.Empty(read.Messages, rule)
		s.Equal([]string{rule}, read.Skipped)
	}
}

// bot is the authorizations entry of the app's bot install.
var bot = map[string]any{"team_id": "T0000TEAM", "user_id": "U0000BOT", "is_bot": true}

// person is a message a person typed in a Slack client, in a channel.
func person(text string) map[string]any {
	return map[string]any{
		"type": "message", "user": "U0000ALICE", "client_msg_id": "00000000-0000-4000-8000-000000000001",
		"ts": "1759740000.000100", "text": text, "team": "T0000TEAM", "channel": "C0000CHAN",
		"event_ts": "1759740000.000100", "channel_type": "channel",
	}
}

// event is the event_callback envelope around event, with one authorizations entry.
func (s *SlackBotSuite) event(event, authorization map[string]any) []byte {
	raw, err := json.Marshal(map[string]any{
		"team_id": "T0000TEAM", "api_app_id": "A0000APP", "event": event, "type": "event_callback",
		"event_id": "Ev0000", "event_time": 1759740000, "authorizations": []any{authorization},
	})
	s.Require().NoError(err)
	return raw
}

func (s *SlackBotSuite) read(body []byte) []core.ChannelMessage {
	read, err := s.manifest.Channel.Read("slack_bot", body)
	s.Require().NoError(err)
	return read.Messages
}
