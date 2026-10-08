package core

import (
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// ChannelSuite runs the manifest channel block against the four channel fixtures in
// testdata/manifests (slack_bot, linq, telegram, whatsapp) and their synthetic events in
// testdata/recorded. Every id, number and text there is made up.
type ChannelSuite struct {
	suite.Suite
}

func TestChannelSuite(t *testing.T) {
	suite.Run(t, new(ChannelSuite))
}

// channelFixtures are the fixtures that carry a channel block.
var channelFixtures = []string{"slack_bot", "linq", "telegram", "whatsapp"}

// baseChannel is a valid channel block that a test changes one part of.
const baseChannel = `channel:
  verifier:
    kind: secret_header
    secret: provider_app
    header: X-Secret
  format: json
  messages:
    thread_key:
      - name: chat
        path: $.chat
    author_id: $.from
    provider_message_id: $.id
    text: $.text
  reply:
    url: https://api.example.com/chats/{thread.chat}/messages
    body:
      text: "{text}"
`

func (s *ChannelSuite) load(id string) Manifest {
	raw, err := os.ReadFile(filepath.Join("testdata", "manifests", id+".yaml"))
	s.Require().NoError(err)
	m, err := ParseManifest(raw)
	s.Require().NoError(err, id)
	return m
}

func (s *ChannelSuite) recorded(name string) []byte {
	raw, err := os.ReadFile(filepath.Join("testdata", "recorded", name))
	s.Require().NoError(err)
	return raw
}

// read reads one recorded event with a fixture's channel block.
func (s *ChannelSuite) read(id, event string) ChannelEvent {
	m := s.load(id)
	s.Require().NotNil(m.Channel)
	read, err := m.Channel.Read(m.ID, s.recorded(event))
	s.Require().NoError(err)
	return read
}

// variant is baseChannel with old replaced by new, parsed.
func (s *ChannelSuite) variant(old, new string) error {
	s.Require().Contains(baseChannel, old)
	_, err := ParseManifest(minimal(strings.Replace(baseChannel, old, new, 1)))
	return err
}

func (s *ChannelSuite) TestAManifestWithOnlyAChannelBlockLoads() {
	for _, id := range []string{"telegram", "linq"} {
		m := s.load(id)
		s.Empty(m.Sources, id)
		s.NotNil(m.Channel, id)
	}
	_, err := ParseManifest(minimal(baseChannel))
	s.Require().NoError(err)
}

func (s *ChannelSuite) TestAManifestWithSourcesAndAChannelLoads() {
	for _, id := range []string{"slack_bot", "whatsapp"} {
		m := s.load(id)
		s.NotEmpty(m.Sources, id)
		s.NotNil(m.Channel, id)
	}
}

// The seeded built-ins have no channel block, and the store compares them as JSON to decide
// on a new revision, so a manifest without one must marshal as it did before the block.
func (s *ChannelSuite) TestAManifestWithoutAChannelMarshalsWithoutTheKey() {
	raw, err := json.Marshal(s.load("custom_crm"))
	s.Require().NoError(err)
	s.NotContains(string(raw), `"channel"`)
}

func (s *ChannelSuite) TestAChannelManifestStoredAsJSONReadsBackTheSame() {
	for _, id := range channelFixtures {
		m := s.load(id)
		raw, err := json.Marshal(m)
		s.Require().NoError(err)
		back, err := ParseManifest(raw)
		s.Require().NoError(err, id)
		s.Equal(m, back, id)
	}
}

func (s *ChannelSuite) TestSubscriptionsAreKeptInTheirOrder() {
	m, err := ParseManifest(minimal(baseChannel + "  subscriptions: [message.channels, message.im, tokens_revoked]\n"))

	s.Require().NoError(err)
	s.Equal([]string{"message.channels", "message.im", "tokens_revoked"}, m.Channel.Subscriptions)
}

func (s *ChannelSuite) TestASubscriptionThatIsNotAnEventTypeIsRefusedWithItsField() {
	_, err := ParseManifest(minimal(baseChannel + "  subscriptions: [message.im, \"Message IM\"]\n"))

	s.ErrorContains(err, `channel.subscriptions[1]: "Message IM" is not a lowercase event type such as message.im`)
}

func (s *ChannelSuite) TestASubscriptionListedTwiceIsRefused() {
	_, err := ParseManifest(minimal(baseChannel + "  subscriptions: [message.im, message.im]\n"))

	s.ErrorContains(err, `channel.subscriptions[1]: "message.im" is listed twice`)
}

func (s *ChannelSuite) TestAnUnknownVerifierKindIsRefusedWithItsField() {
	err := s.variant("kind: secret_header", "kind: jwt_set")
	s.ErrorContains(err, `channel.verifier.kind: "jwt_set" is not one of [hmac_header secret_header standard_webhooks ed25519]`)
}

func (s *ChannelSuite) TestAReplyBodyNamingAnUndeclaredInputIsRefused() {
	err := s.variant(`text: "{text}"`, `text: "{text} {signature}"`)
	s.ErrorContains(err, "channel.reply.body.text: {signature} is not text, a declared thread key part or provider_unit_id, an input, a vars entry or a captured name")
}

func (s *ChannelSuite) TestAReplyURLNamingAnUndeclaredInputIsRefused() {
	err := s.variant("/messages", "/{account}/messages")
	s.ErrorContains(err, "channel.reply.url: {account} is not a declared input, a vars entry or a captured name")
}

func (s *ChannelSuite) TestAReplyNamingAnUndeclaredThreadKeyPartIsRefused() {
	err := s.variant("{thread.chat}", "{thread.room}")
	s.ErrorContains(err, "channel.reply.url: {thread.room} is not a declared thread key part or provider_unit_id")

	err = s.variant(`text: "{text}"`, `text: "{thread.room}"`)
	s.ErrorContains(err, "channel.reply.body.text: {thread.room} is not text")
}

// provider_unit_id is a reply value only when the block reads one.
func (s *ChannelSuite) TestAReplyCannotNameAProviderUnitTheBlockDoesNotRead() {
	err := s.variant("/messages", "/{provider_unit_id}/messages")
	s.ErrorContains(err, "channel.reply.url: {provider_unit_id} is not a declared thread key part or provider_unit_id")
}

func (s *ChannelSuite) TestAReplyMayNameAnInput() {
	_, err := ParseManifest(minimal("inputs:\n  - name: workspace\n    pattern: \"[a-z]+\"\n" +
		strings.Replace(baseChannel, "/messages", "/{workspace}/messages", 1)))
	s.Require().NoError(err)
}

func (s *ChannelSuite) TestTheReplyTextIsNotInTheURL() {
	err := s.variant("/messages", "/{text}")
	s.ErrorContains(err, "channel.reply.url: {text} goes in the body, not the URL")
}

func (s *ChannelSuite) TestAMessageValueCannotPickTheReplyHost() {
	for _, url := range []string{"https://{thread.chat}.example.com/messages", `"{thread.chat}/messages"`} {
		err := s.variant("https://api.example.com/chats/{thread.chat}/messages", url)
		s.ErrorContains(err, "channel.reply.url: {thread.chat} is in the host", url)
	}
}

func (s *ChannelSuite) TestAnInputCannotShareAReplyValuesName() {
	_, err := ParseManifest(minimal("inputs:\n  - name: text\n    pattern: \"[a-z]+\"\n" + baseChannel))
	s.ErrorContains(err, `inputs: "text" is a name a reply template already uses`)
}

func (s *ChannelSuite) TestAReplyBodyHoldsOnlyStringsObjectsAndLists() {
	err := s.variant(`text: "{text}"`, "text: \"{text}\"\n      silent: true")
	s.ErrorContains(err, "channel.reply.body.silent: is bool; a body holds strings, objects and lists")
}

func (s *ChannelSuite) TestAVerifierParameterItsKindDoesNotReadIsRefused() {
	err := s.variant("header: X-Secret", "header: X-Secret\n    algorithm: sha256")
	s.ErrorContains(err, "channel.verifier.algorithm: is not read by secret_header")
}

func (s *ChannelSuite) TestAnHMACMustSignTheBody() {
	hmac := "kind: hmac_header\n    secret: provider_app\n    header: X-Signature\n    algorithm: sha256\n    encoding: hex\n"
	err := s.variant("kind: secret_header\n    secret: provider_app\n    header: X-Secret\n", hmac+"    signed: \"{timestamp}\"\n")
	s.ErrorContains(err, "channel.verifier.signed: \"{timestamp}\" must name {body} once")

	err = s.variant("kind: secret_header\n    secret: provider_app\n    header: X-Secret\n", hmac+"    signed: \"{nonce}.{body}\"\n")
	s.ErrorContains(err, "channel.verifier.signed: \"{nonce}.{body}\" names a value other than [body timestamp]")
}

func (s *ChannelSuite) TestASignedTimestampNeedsItsHeaderAndAnAge() {
	hmac := "kind: hmac_header\n    secret: provider_app\n    header: X-Signature\n    algorithm: sha256\n    encoding: hex\n" +
		"    signed: \"{timestamp}.{body}\"\n"
	err := s.variant("kind: secret_header\n    secret: provider_app\n    header: X-Secret\n", hmac)
	s.ErrorContains(err, "channel.verifier.signed: {timestamp} and timestamp_header go together")

	err = s.variant("kind: secret_header\n    secret: provider_app\n    header: X-Secret\n", hmac+"    timestamp_header: X-Timestamp\n")
	s.ErrorContains(err, "channel.verifier.max_age: is set exactly when timestamp_header is")
}

// AI-881: ed25519 reads hmac_header's header, signed template and timestamp, and fixes the
// rest: Telnyx's signature is «Base64-encoded Ed25519»
// (https://developers.telnyx.com/docs/messaging/messages/receiving-webhooks).
func (s *ChannelSuite) TestAnEd25519VerifierTakesAHeaderASignedTemplateAndATimestamp() {
	secretHeader := "kind: secret_header\n    secret: provider_app\n    header: X-Secret\n"
	ed := "kind: ed25519\n    secret: provider_app\n    header: Telnyx-Signature-Ed25519\n"
	_, err := ParseManifest(minimal(strings.Replace(baseChannel, secretHeader,
		ed+"    signed: \"{timestamp}|{body}\"\n    timestamp_header: Telnyx-Timestamp\n    max_age: 5m\n", 1)))
	s.Require().NoError(err)

	err = s.variant(secretHeader, "kind: ed25519\n    secret: provider_app\n    signed: \"{body}\"\n")
	s.ErrorContains(err, "channel.verifier.header: is empty")

	err = s.variant(secretHeader, ed+"    signed: \"{timestamp}\"\n")
	s.ErrorContains(err, "channel.verifier.signed: \"{timestamp}\" must name {body} once")

	err = s.variant(secretHeader, ed+"    signed: \"{timestamp}|{body}\"\n")
	s.ErrorContains(err, "channel.verifier.signed: {timestamp} and timestamp_header go together")

	for _, parameter := range []string{"algorithm: sha256", "encoding: hex", "prefix: v1="} {
		err = s.variant(secretHeader, ed+"    signed: \"{body}\"\n    "+parameter+"\n")
		s.ErrorContains(err, "channel.verifier."+strings.Split(parameter, ":")[0]+": is not read by ed25519")
	}
}

func (s *ChannelSuite) TestAnOperatorSecretNeedsClientEnv() {
	err := s.variant("secret: provider_app", "secret: operator")
	s.ErrorContains(err, "channel.verifier.secret: operator needs client.env")
}

// AI-879: a handshake is one of the closed list, named with its field.
func (s *ChannelSuite) TestAnUnknownHandshakeIsRefusedWithItsField() {
	err := s.variant("  format: json", "  handshake: crc_token\n  format: json")
	s.ErrorContains(err, `channel.handshake: "crc_token" is not one of [hub_challenge]`)
}

// AI-879: the verify token is the provider app's id, so an operator's secret has no handshake.
func (s *ChannelSuite) TestAHandshakeNeedsAProviderAppSecret() {
	s.Require().NoError(s.variant("  format: json", "  handshake: hub_challenge\n  format: json"))

	err := s.variant("    secret: provider_app\n    header: X-Secret\n",
		"    secret: operator\n    header: X-Secret\n  handshake: hub_challenge\n")

	s.ErrorContains(err, "channel.handshake: needs a provider_app secret")
}

// hubQuery is a hub_challenge handshake's query (PubSubHubbub 0.3, 6.2.1; Meta's webhooks).
func hubQuery(mode, token, challenge string) url.Values {
	return url.Values{"hub.mode": {mode}, "hub.verify_token": {token}, "hub.challenge": {challenge}}
}

// AI-879: the challenge is echoed only for subscribe, the URL's own verify token, and digits.
func (s *ChannelSuite) TestAHubChallengeIsEchoedOnlyWhenItsQueryAgrees() {
	block := ChannelRule{Handshake: HandshakeHubChallenge}

	challenge, ok := block.AnswerHandshake(hubQuery("subscribe", "1234", "1158201444"), "1234")
	s.True(ok)
	s.Equal("1158201444", challenge)

	for name, query := range map[string]url.Values{
		"unsubscribe":      hubQuery("unsubscribe", "1234", "1158201444"),
		"another token":    hubQuery("subscribe", "12345", "1158201444"),
		"no token":         {"hub.mode": {"subscribe"}, "hub.challenge": {"1158201444"}},
		"no challenge":     hubQuery("subscribe", "1234", ""),
		"markup":           hubQuery("subscribe", "1234", "<script>1</script>"),
		"a signed integer": hubQuery("subscribe", "1234", "-1"),
	} {
		_, ok := block.AnswerHandshake(query, "1234")
		s.False(ok, name)
	}
	_, ok = block.AnswerHandshake(hubQuery("subscribe", "", "1"), "")
	s.False(ok, "an empty verify token agrees with nothing")
}

// AI-879: a block that declares no handshake echoes nothing, whatever the query says.
func (s *ChannelSuite) TestABlockWithoutAHandshakeEchoesNothing() {
	_, ok := ChannelRule{}.AnswerHandshake(hubQuery("subscribe", "1234", "1"), "1234")
	s.False(ok)
}

func (s *ChannelSuite) TestAPathWhoseWildcardIsNotEachsIsRefused() {
	err := s.variant("author_id: $.from", "author_id: $.items[*].from")
	s.ErrorContains(err, `channel.messages.author_id: "$.items[*].from": has [*] but messages.each is empty`)

	_, err = ParseManifest(minimal(strings.NewReplacer(
		"thread_key:", "each: $.entry[*].messages[*]\n    thread_key:",
		"author_id: $.from", "author_id: $.other[*].messages[*].from",
	).Replace(baseChannel)))
	s.ErrorContains(err, `channel.messages.author_id: "$.other[*].messages[*].from": its [*] is not where messages.each has one`)
}

func (s *ChannelSuite) TestAPathWithAFilterIsRefused() {
	err := s.variant("text: $.text", "text: $.parts[?@.type=='text'].value")
	s.ErrorContains(err, "channel.messages.text: \"$.parts[?@.type=='text'].value\" is not a path of member names")
}

func (s *ChannelSuite) TestAWindowNeedsTheBodySentAfterIt() {
	err := s.variant(`text: "{text}"`, "text: \"{text}\"\n    window: 24h")
	s.ErrorContains(err, "channel.reply.after_window: is set exactly when window is")
}

func (s *ChannelSuite) TestSlackReadsAThreadReplyByChannelAndParent() {
	body := s.recorded("slack_bot.message.json")
	read := s.read("slack_bot", "slack_bot.message.json")
	s.Empty(read.Challenge)
	s.Require().Len(read.Messages, 1)
	s.Equal(ChannelMessage{
		InboundMessage: InboundMessage{
			ConnectorID:       "slack_bot",
			ProviderUnitID:    "T0000TEAM",
			ThreadKey:         "C0000CHAN:1759740000.000100",
			AuthorID:          "U0000USER",
			Text:              "Can you \"check\" the build?\nThanks",
			ProviderMessageID: "1759740000.000200",
			Raw:               body,
		},
		ThreadParts: map[string]string{"channel": "C0000CHAN", "thread_ts": "1759740000.000100"},
	}, read.Messages[0])
}

func (s *ChannelSuite) TestASlackMessageThatStartsAThreadIsItsOwnThread() {
	read := s.read("slack_bot", "slack_bot.thread_start.json")
	s.Require().Len(read.Messages, 1)
	s.Equal("D0000IM:1759740000.000300", read.Messages[0].ThreadKey)
	s.Equal("1759740000.000300", read.Messages[0].ThreadParts["thread_ts"])
}

func (s *ChannelSuite) TestTheSlackBotsOwnMessageIsNotRead() {
	s.Empty(s.read("slack_bot", "slack_bot.own.json").Messages)
}

// The customer's app may subscribe to other events on the same Request URL. channel_created's
// event.channel is an object, so reading it as a message would fail the whole body.
func (s *ChannelSuite) TestASlackEventThatIsNotAMessageIsNoMessageAndNoError() {
	s.Equal(ChannelEvent{}, s.read("slack_bot", "slack_bot.channel_created.json"))
}

func (s *ChannelSuite) TestALinqReadReceiptIsNoMessage() {
	s.Equal(ChannelEvent{}, s.read("linq", "linq.read.json"))
}

// An Update holds at most one of its optional fields, so an edit has no message.
func (s *ChannelSuite) TestATelegramUpdateWithoutAMessageIsNoMessage() {
	s.Equal(ChannelEvent{}, s.read("telegram", "telegram.edited.json"))
}

// A WhatsApp Business Account webhook also posts other fields, such as
// message_template_status_update, whose value has no messages.
func (s *ChannelSuite) TestAWhatsAppTemplateStatusIsNoMessage() {
	s.Equal(ChannelEvent{}, s.read("whatsapp", "whatsapp.template_status.json"))
}

// An app subscribed to app_mention and message.channels can get one post as both events;
// only the message is read.
func (s *ChannelSuite) TestASlackPostThatMentionsTheAppIsOneMessage() {
	messages := append(s.read("slack_bot", "slack_bot.app_mention.json").Messages,
		s.read("slack_bot", "slack_bot.message.json").Messages...)
	s.Require().Len(messages, 1)
	s.Equal("1759740000.000200", messages[0].ProviderMessageID)
}

func (s *ChannelSuite) TestASlackHandshakeIsOnlyAChallenge() {
	s.Equal(ChannelEvent{Challenge: "synthetic-challenge-value"}, s.read("slack_bot", "slack_bot.challenge.json"))
}

func (s *ChannelSuite) TestASlackReplyGoesToTheSameThread() {
	resolved, err := s.load("slack_bot").Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)
	read := s.read("slack_bot", "slack_bot.message.json")

	url, body, err := resolved.Reply(ReplyValues{Text: `Done: "green"`, ThreadParts: read.Messages[0].ThreadParts})
	s.Require().NoError(err)
	s.Equal("https://slack.com/api/chat.postMessage", url)
	s.JSONEq(`{"channel":"C0000CHAN","thread_ts":"1759740000.000100","text":"Done: \"green\""}`, string(body))
}

func (s *ChannelSuite) TestLinqReadsAReceivedMessageOnTheCustomersLine() {
	read := s.read("linq", "linq.received.json")
	s.Require().Len(read.Messages, 1)
	message := read.Messages[0]
	s.Equal("+12025550100", message.ProviderUnitID)
	s.Equal("00000000-0000-4000-8000-0000000000c1", message.ThreadKey)
	s.Equal("+12025550199", message.AuthorID)
	s.Equal("00000000-0000-4000-8000-0000000000e1", message.ProviderMessageID)
	s.Equal("Hi, is my order ready?", message.Text)
}

func (s *ChannelSuite) TestAMessageTheLinqLineSentIsNotRead() {
	s.Empty(s.read("linq", "linq.sent.json").Messages)
}

func (s *ChannelSuite) TestALinqReplyGoesToTheChat() {
	resolved, err := s.load("linq").Resolve("bearer", nil, nil)
	s.Require().NoError(err)
	url, body, err := resolved.Reply(ReplyValues{
		Text: "Yes", ProviderUnitID: "+12025550100",
		ThreadParts: map[string]string{"chat_id": "00000000-0000-4000-8000-0000000000c1"},
	})
	s.Require().NoError(err)
	s.Equal("https://api.linqapp.com/api/partner/v3/chats/00000000-0000-4000-8000-0000000000c1/messages", url)
	s.JSONEq(`{"message":{"parts":[{"type":"text","value":"Yes"}]}}`, string(body))
}

// Telegram ids can pass 32 bits, so they are read as written, never through a float.
func (s *ChannelSuite) TestTelegramReadsAnUpdateWithItsIDsAsWritten() {
	read := s.read("telegram", "telegram.update.json")
	s.Require().Len(read.Messages, 1)
	message := read.Messages[0]
	s.Empty(message.ProviderUnitID)
	s.Equal("-1004503599627370497", message.ThreadKey)
	s.Equal("4503599627370497", message.AuthorID)
	s.Equal("42", message.ProviderMessageID)
	s.Equal("What time do you open?", message.Text)
}

func (s *ChannelSuite) TestAWhatsAppBatchIsOneMessageEachWithItsOwnNumber() {
	read := s.read("whatsapp", "whatsapp.batch.json")
	s.Require().Len(read.Messages, 3)
	var got [][]string
	for _, message := range read.Messages {
		got = append(got, []string{message.ProviderUnitID, message.ThreadKey, message.ProviderMessageID, message.Text})
	}
	s.Equal([][]string{
		{"200000000000001", "15550001111", "wamid.synthetic-1", "First"},
		// An image has no text.body, and is still a message.
		{"200000000000001", "15550001111", "wamid.synthetic-2", ""},
		// The statuses change before it has no messages, so it adds none.
		{"200000000000002", "15550002222", "wamid.synthetic-3", "Second number"},
	}, got)
	s.Equal(s.recorded("whatsapp.batch.json"), read.Messages[2].Raw)
}

func (s *ChannelSuite) TestAWhatsAppReplyInsideTheWindowIsText() {
	resolved, err := s.load("whatsapp").Resolve("oauth2_code",
		map[string]string{"reopen_template": "follow_up", "template_language": "en_US"}, nil)
	s.Require().NoError(err)
	url, body, err := resolved.Reply(ReplyValues{
		Text: "Blue and red", ProviderUnitID: "200000000000001",
		ThreadParts:  map[string]string{"from": "15550001111"},
		SinceInbound: 23 * time.Hour,
	})
	s.Require().NoError(err)
	s.Equal("https://graph.facebook.com/v25.0/200000000000001/messages", url)
	s.JSONEq(`{"messaging_product":"whatsapp","recipient_type":"individual","to":"+15550001111","type":"text","text":{"body":"Blue and red"}}`, string(body))
}

func (s *ChannelSuite) TestAWhatsAppReplyAfterTheWindowIsTheTemplate() {
	resolved, err := s.load("whatsapp").Resolve("oauth2_code",
		map[string]string{"reopen_template": "follow_up", "template_language": "en_US"}, nil)
	s.Require().NoError(err)
	_, body, err := resolved.Reply(ReplyValues{
		Text: "Blue and red", ProviderUnitID: "200000000000001",
		ThreadParts:  map[string]string{"from": "15550001111"},
		SinceInbound: 24 * time.Hour,
	})
	s.Require().NoError(err)
	s.JSONEq(`{"messaging_product":"whatsapp","recipient_type":"individual","to":"+15550001111","type":"template","template":{"name":"follow_up","language":{"code":"en_US"}}}`, string(body))
}

// A thread key part goes into the reply URL as an input does, so a value from a message
// cannot add a path segment or remove one.
func (s *ChannelSuite) TestAMessageValueCannotChangeTheReplyPath() {
	resolved, err := s.load("linq").Resolve("bearer", nil, nil)
	s.Require().NoError(err)
	for _, chat := range []string{"../../admin", "..", "a/b", "a?b"} {
		_, _, err := resolved.Reply(ReplyValues{Text: "x", ThreadParts: map[string]string{"chat_id": chat}})
		s.ErrorContains(err, "channel.reply.url: {thread.chat_id}", chat)
	}
}

// Parts are joined with ":", so a part that holds one is encoded: otherwise a:b + c and
// a + b:c would be one thread.
func (s *ChannelSuite) TestAThreadKeyPartWithAColonCannotMergeTwoThreads() {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel,
		"      - name: chat\n        path: $.chat\n",
		"      - name: chat\n        path: $.chat\n      - name: topic\n        path: $.topic\n", 1)))
	s.Require().NoError(err)
	first, err := m.Channel.Read(m.ID, []byte(`{"chat":"a:b","topic":"c","from":"u","id":"1"}`))
	s.Require().NoError(err)
	second, err := m.Channel.Read(m.ID, []byte(`{"chat":"a","topic":"b:c","from":"u","id":"2"}`))
	s.Require().NoError(err)
	s.Equal("a%3Ab:c", first.Messages[0].ThreadKey)
	s.Equal("a:b%3Ac", second.Messages[0].ThreadKey)
}

// A form body, as Twilio posts one, is read with the same paths: each field is a member.
func (s *ChannelSuite) TestAFormBodyIsReadByFieldName() {
	m, err := ParseManifest(minimal(strings.NewReplacer(
		"format: json", "format: form",
		"$.chat", "$.From",
		"$.from", "$.From",
		"$.id", "$.MessageSid",
		"$.text", "$.Body",
	).Replace(baseChannel)))
	s.Require().NoError(err)
	read, err := m.Channel.Read(m.ID, []byte("MessageSid=SM00000000000000000000000000000001&From=%2B15550001111&To=%2B15550000001&Body=Hi+there"))
	s.Require().NoError(err)
	s.Require().Len(read.Messages, 1)
	s.Equal("+15550001111", read.Messages[0].AuthorID)
	s.Equal("SM00000000000000000000000000000001", read.Messages[0].ProviderMessageID)
	s.Equal("Hi there", read.Messages[0].Text)
}

func (s *ChannelSuite) TestAFormFieldGivenTwiceIsRefused() {
	m, err := ParseManifest(minimal(strings.NewReplacer("format: json", "format: form").Replace(baseChannel)))
	s.Require().NoError(err)
	_, err = m.Channel.Read(m.ID, []byte("chat=a&chat=b&from=u&id=1"))
	s.ErrorContains(err, "inbound form has 2 values for chat")
}

func (s *ChannelSuite) TestAFormPathIsOneField() {
	err := s.variant("format: json", "format: form")
	s.Require().NoError(err)
	_, err = ParseManifest(minimal(strings.NewReplacer("format: json", "format: form", "$.text", "$.body.text").Replace(baseChannel)))
	s.ErrorContains(err, `channel.messages.text: "$.body.text": a form is flat`)
}

func (s *ChannelSuite) TestAMessageWithoutAnAuthorIsNotRead() {
	m, err := ParseManifest(minimal(baseChannel))
	s.Require().NoError(err)
	read, err := m.Channel.Read(m.ID, []byte(`{"chat":"a","id":"1","text":"hi"}`))
	s.Require().NoError(err)
	s.Empty(read.Messages)
}

func (s *ChannelSuite) TestABodyThatIsNotJSONIsAnError() {
	m, err := ParseManifest(minimal(baseChannel))
	s.Require().NoError(err)
	_, err = m.Channel.Read(m.ID, []byte("not json"))
	s.ErrorContains(err, "inbound body is not a JSON object")
}

// signalsOnly is a channel block with signals and no messages, in the shape of the built-in
// user-token Slack connector: the account is the team and the user.
const signalsOnly = `capture:
  - name: team_id
    from: token_response
    path: $.team.id
  - name: user_id
    from: token_response
    path: $.authed_user.id
identity: [team_id, user_id]
channel:
  verifier:
    kind: secret_header
    secret: provider_app
    header: X-Secret
  format: json
  challenge: $.challenge
  signals:
    - kind: revoked
      match:
        $.event.type: tokens_revoked
      each: $.event.tokens.oauth[*]
      identity:
        team_id: $.team_id
        user_id: $.event.tokens.oauth[*]
    - kind: uninstalled
      match:
        $.event.type: app_uninstalled
      identity:
        team_id: $.team_id
`

// signalsVariant is signalsOnly with old replaced by new, parsed.
func (s *ChannelSuite) signalsVariant(old, new string) error {
	s.Require().Contains(signalsOnly, old)
	_, err := ParseManifest(minimal(strings.Replace(signalsOnly, old, new, 1)))
	return err
}

// readSignalsOnly reads one recorded event with signalsOnly.
func (s *ChannelSuite) readSignalsOnly(event string) ChannelEvent {
	m, err := ParseManifest(minimal(signalsOnly))
	s.Require().NoError(err)
	read, err := m.Channel.Read(m.ID, s.recorded(event))
	s.Require().NoError(err)
	return read
}

func (s *ChannelSuite) TestARevokedBotTokenEndsTheWorkspacesBotGrant() {
	read := s.read("slack_bot", "slack_bot.tokens_revoked.json")

	s.Equal([]Signal{{ConnectorID: "slack_bot", Identity: map[string]string{"team_id": "T0000TEAM"}, Kind: SignalRevoked}}, read.Signals)
	s.Empty(read.Messages)
}

func (s *ChannelSuite) TestARevokedUserTokenEndsNoBotGrant() {
	read := s.read("slack_bot", "slack_bot.user_tokens_revoked.json")

	s.Empty(read.Signals)
	s.Empty(read.Messages)
}

func (s *ChannelSuite) TestASlackUninstallNamesTheWorkspace() {
	read := s.read("slack_bot", "slack_bot.app_uninstalled.json")

	s.Equal([]Signal{{ConnectorID: "slack_bot", Identity: map[string]string{"team_id": "T0000TEAM"}, Kind: SignalUninstalled}}, read.Signals)
}

func (s *ChannelSuite) TestAMessageIsNoSignal() {
	read := s.read("slack_bot", "slack_bot.message.json")

	s.Empty(read.Signals)
	s.Len(read.Messages, 1)
}

func (s *ChannelSuite) TestARevocationOfUserTokensIsOneSignalForEachUser() {
	read := s.readSignalsOnly("slack_bot.tokens_revoked.json")

	s.Equal([]Signal{
		{ConnectorID: "example", Identity: map[string]string{"team_id": "T0000TEAM", "user_id": "U0000USER"}, Kind: SignalRevoked},
		{ConnectorID: "example", Identity: map[string]string{"team_id": "T0000TEAM", "user_id": "U0000OTHER"}, Kind: SignalRevoked},
	}, read.Signals)
}

// An uninstall names the team and no user, which is every account of the workspace.
func (s *ChannelSuite) TestAnUninstallMayNameFewerPartsThanTheIdentity() {
	read := s.readSignalsOnly("slack_bot.app_uninstalled.json")

	s.Equal([]Signal{{ConnectorID: "example", Identity: map[string]string{"team_id": "T0000TEAM"}, Kind: SignalUninstalled}}, read.Signals)
}

func (s *ChannelSuite) TestABlockWithOnlySignalsReadsNoMessagesAndStillAnswersAHandshake() {
	s.Empty(s.readSignalsOnly("slack_bot.message.json").Messages)
	s.Equal(ChannelEvent{Challenge: "synthetic-challenge-value"}, s.readSignalsOnly("slack_bot.challenge.json"))
}

func (s *ChannelSuite) TestABlockWithOnlySignalsStoredAsJSONReadsBackTheSameWithoutMessages() {
	m, err := ParseManifest(minimal(signalsOnly))
	s.Require().NoError(err)
	raw, err := json.Marshal(m)
	s.Require().NoError(err)
	s.NotContains(string(raw), `"messages"`)
	s.NotContains(string(raw), `"reply"`)
	back, err := ParseManifest(raw)
	s.Require().NoError(err)
	s.Equal(m, back)
}

func (s *ChannelSuite) TestABlockWithOnlySignalsHasNoReply() {
	m, err := ParseManifest(minimal(signalsOnly))
	s.Require().NoError(err)
	resolved, err := m.Resolve("oauth2_code", nil, nil)
	s.Require().NoError(err)

	_, _, err = resolved.Reply(ReplyValues{Text: "hi"})
	s.ErrorContains(err, "has no channel reply")
}

func (s *ChannelSuite) TestABlockWithNeitherMessagesNorSignalsIsRefused() {
	_, err := ParseManifest(minimal(signalsOnly[:strings.Index(signalsOnly, "  signals:")]))
	s.ErrorContains(err, "channel: reads neither messages nor signals")
}

func (s *ChannelSuite) TestAReplyWithoutMessagesIsRefused() {
	err := s.signalsVariant("  signals:", "  reply:\n    url: https://api.example.com/messages\n    body:\n      text: \"{text}\"\n  signals:")
	s.ErrorContains(err, "channel.messages: is empty: a reply goes back to the thread a message came from")
}

// signalsWithAt is signalsOnly whose uninstall says when it happened, as Slack's envelope
// does: «event_time: The epoch timestamp in seconds indicating when this event was
// dispatched» (https://docs.slack.dev/apis/events-api/).
func (s *ChannelSuite) signalsWithAt() Manifest {
	m, err := ParseManifest(minimal(strings.Replace(signalsOnly,
		"        $.event.type: app_uninstalled\n", "        $.event.type: app_uninstalled\n      at: $.event_time\n", 1)))
	s.Require().NoError(err)
	return m
}

func (s *ChannelSuite) TestASignalSaysWhenTheEventHappened() {
	m := s.signalsWithAt()

	read, err := m.Channel.Read(m.ID, s.recorded("slack_bot.app_uninstalled.json"))

	s.Require().NoError(err)
	s.Require().Len(read.Signals, 1)
	s.Equal(time.Unix(1759740000, 0).UTC(), read.Signals[0].At)
}

func (s *ChannelSuite) TestASignalRuleWithoutATimeSaysNothingOfWhen() {
	read := s.readSignalsOnly("slack_bot.app_uninstalled.json")

	s.Require().Len(read.Signals, 1)
	s.True(read.Signals[0].At.IsZero())
}

func (s *ChannelSuite) TestASignalTimeThatIsNotWholeSecondsIsNotTheBlocksShape() {
	m := s.signalsWithAt()

	_, err := m.Channel.Read(m.ID, []byte(`{"team_id":"T0000TEAM","event":{"type":"app_uninstalled"},"event_time":"yesterday"}`))

	s.ErrorContains(err, `$.event_time: "yesterday" is not whole Unix seconds`)
}

func (s *ChannelSuite) TestASignalTimeWithAWildcardIsRefused() {
	err := s.signalsVariant("      each: $.event.tokens.oauth[*]\n", "      each: $.event.tokens.oauth[*]\n      at: $.event.tokens.oauth[*]\n")
	s.ErrorContains(err, "channel.signals[0].at: an event happened once, so it has no [*]")
}

// Slack answers a refused chat.postMessage with HTTP 200 and «"ok": false»
// (https://docs.slack.dev/reference/methods/chat.postMessage).
func (s *ChannelSuite) TestAReplyIsSentOnlyWhenTheAnswerHasTheAcceptedValues() {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "      text: \"{text}\"\n", "      text: \"{text}\"\n    accepted:\n      $.ok: \"true\"\n", 1)))
	s.Require().NoError(err)

	sent, err := m.Channel.Reply.Accepts([]byte(`{"ok":true,"ts":"1759740000.000500"}`))
	s.Require().NoError(err)
	s.True(sent)
	sent, err = m.Channel.Reply.Accepts([]byte(`{"ok":false,"error":"invalid_auth"}`))
	s.Require().NoError(err)
	s.False(sent)
	sent, err = m.Channel.Reply.Accepts([]byte(`{"error":"invalid_auth"}`))
	s.Require().NoError(err)
	s.False(sent, "a value that is absent is not the accepted one")
}

func (s *ChannelSuite) TestAReplyWithoutAcceptedValuesIsSentOnAny2xx() {
	m, err := ParseManifest(minimal(baseChannel))
	s.Require().NoError(err)

	sent, err := m.Channel.Reply.Accepts([]byte(`not json`))

	s.Require().NoError(err)
	s.True(sent)
}

func (s *ChannelSuite) TestAnEmptyAcceptedValueIsRefused() {
	err := s.variant("      text: \"{text}\"\n", "      text: \"{text}\"\n    accepted:\n      $.ok: \"\"\n")
	s.ErrorContains(err, "channel.reply.accepted.$.ok: is empty")
}

func (s *ChannelSuite) TestASignalWithoutAMatchIsRefused() {
	err := s.signalsVariant("      match:\n        $.event.type: app_uninstalled\n", "")
	s.ErrorContains(err, "channel.signals[1].match: is empty: without one every delivery would be this signal")
}

func (s *ChannelSuite) TestASignalOfAnUnknownKindIsRefused() {
	err := s.signalsVariant("kind: uninstalled", "kind: paused")
	s.ErrorContains(err, `channel.signals[1].kind: "paused" is not one of [revoked uninstalled rotated]`)
}

func (s *ChannelSuite) TestASignalNamingAPartOutsideTheIdentityIsRefused() {
	err := s.signalsVariant("      identity:\n        team_id: $.team_id\n", "      identity:\n        enterprise_id: $.enterprise_id\n")
	s.ErrorContains(err, `channel.signals[0].identity.enterprise_id: "enterprise_id" is not one of identity [team_id user_id]`)
}

func (s *ChannelSuite) TestASignalPathWhoseWildcardIsNotEachsIsRefused() {
	err := s.signalsVariant("      each: $.event.tokens.oauth[*]\n", "")
	s.ErrorContains(err, `channel.signals[0].identity.user_id: "$.event.tokens.oauth[*]": has [*] but channel.signals[0].each is empty`)
}

func (s *ChannelSuite) TestASignalMatchIsAboutTheWholeEvent() {
	err := s.signalsVariant("        $.event.type: tokens_revoked", "        $.event.tokens.oauth[*]: U0000USER")
	s.ErrorContains(err, "a match is about the whole event, not one of its accounts")
}

// eventIDs is baseChannel reading Slack's two ids: event_id on an event, and trigger_id in an
// interaction's payload form field.
func (s *ChannelSuite) eventIDs() *ChannelRule {
	m, err := ParseManifest(minimal(strings.Replace(baseChannel, "  format: json\n",
		"  format: json\n  event_id: [$.event_id, $.payload.trigger_id]\n", 1)))
	s.Require().NoError(err)
	return m.Channel
}

func (s *ChannelSuite) TestADeliveryEventIDIsTheProvidersIdOfAnEvent() {
	s.Equal("Ev0000ONE", s.eventIDs().DeliveryEventID([]byte(`{"type":"event_callback","event_id":"Ev0000ONE"}`)))
}

// https://docs.slack.dev/interactivity/handling-user-interaction: an interaction is a form
// whose payload field is JSON, posted to the same URL as the JSON events.
func (s *ChannelSuite) TestADeliveryEventIDIsReadInsideAFormFieldsJSON() {
	body := url.Values{"payload": {`{"type":"block_actions","trigger_id":"1.2.abc"}`}}.Encode()

	s.Equal("1.2.abc", s.eventIDs().DeliveryEventID([]byte(body)))
}

func (s *ChannelSuite) TestADeliveryWithoutAnEventIDHasNone() {
	s.Empty(s.eventIDs().DeliveryEventID([]byte(`{"type":"app_rate_limited"}`)))
	s.Empty(s.load("linq").Channel.DeliveryEventID(s.recorded("linq.received.json")), "a block that names no event_id")
}

func (s *ChannelSuite) TestAnEventIDIsOneValueNotOnePerMessage() {
	err := s.variant("  format: json\n", "  format: json\n  event_id:\n    - $.events[*].id\n")
	s.ErrorContains(err, "a delivery has one id, not one per message")
}
