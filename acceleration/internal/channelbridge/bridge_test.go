package channelbridge

import (
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// BridgeSuite is what the bridge decides before it reaches a store or Stream Chat. Its whole
// flow, from a Slack event to a reply, runs in internal/api's SlackChannelSuite against
// Postgres, the suite's Stream Chat and the fake provider.
type BridgeSuite struct {
	suite.Suite
}

func TestBridgeSuite(t *testing.T) {
	suite.Run(t, new(BridgeSuite))
}

func (s *BridgeSuite) TestAnAuthorIsTheSameUserOnEveryMessage() {
	s.Equal(AuthorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), AuthorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

// One Slack workspace can install two customers' apps; its person is a user of each, never
// one user both customers write as.
func (s *BridgeSuite) TestTheSamePersonForAnotherCustomerIsAnotherUser() {
	s.NotEqual(AuthorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), AuthorUserID("globex", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

func (s *BridgeSuite) TestPartsThatJoinAlikeAreStillTwoUsers() {
	s.NotEqual(AuthorUserID("acme", "slack_bot", "T0", "0U"), AuthorUserID("acme", "slack_bot", "T00", "U"))
}

// A phone number's "+" or any provider's id stays out of the user id.
func (s *BridgeSuite) TestAnAuthorUserIDKeepsToLettersDigitsUnderscoreAndHyphen() {
	s.Regexp(regexp.MustCompile(`^[A-Za-z0-9_-]+$`), AuthorUserID("acme", "linq", "+12025550100", "+12025550199"))
}

// A Linq chat is one person's when its event says is_group false, as the 2026-02-03 payload
// writes it.
func (s *BridgeSuite) TestALinqChatIsOnePersonsWhenItsEventSaysItIsNoGroup() {
	s.True(linqOneToOne([]byte(`{"event_type":"message.received","data":{"chat":{"id":"c1","is_group":false,"owner_handle":{"handle":"+12025550100"}}}}`)))
}

// Anything but an explicit false is a group's chat, whose tasks would reach everyone in it.
func (s *BridgeSuite) TestALinqChatIsAGroupsUnlessItsEventSaysOtherwise() {
	for name, raw := range map[string]string{
		"a group":                 `{"data":{"chat":{"id":"c1","is_group":true}}}`,
		"no is_group":             `{"data":{"chat":{"id":"c1"}}}`,
		"is_group null":           `{"data":{"chat":{"id":"c1","is_group":null}}}`,
		"is_group a string":       `{"data":{"chat":{"id":"c1","is_group":"false"}}}`,
		"is_group a number":       `{"data":{"chat":{"id":"c1","is_group":0}}}`,
		"a chat that is a string": `{"data":{"chat":"c1"}}`,
		"no chat":                 `{"data":{}}`,
		"data null":               `{"data":null}`,
		"an empty object":         `{}`,
		"an array":                `[{"data":{"chat":{"is_group":false}}}]`,
		"null":                    `null`,
		"empty":                   ``,
		"not JSON":                `is_group=false`,
		"is_group one level up":   `{"data":{"is_group":false,"chat":{"id":"c1"}}}`,
		// A decode that fails after it read is_group false says two things at once.
		"a second chat after it": `{"data":{"chat":{"id":"c1","is_group":false},"chat":"c2"}}`,
	} {
		s.False(linqOneToOne([]byte(raw)), name)
	}
}

func (s *BridgeSuite) TestABridgeNeedsAStoreStreamTransportsAndAResolver() {
	_, err := New(Options{})
	s.ErrorContains(err, "a store, Stream clients, transports and a resolver are required")
}

// AI-881: a keyword is read without its case or punctuation (CTIA 5.1.3), and only when it is
// the whole message.
func (s *BridgeSuite) TestAKeywordIsReadWithoutItsCaseOrPunctuation() {
	for text, word := range map[string]string{
		"Stop.":           "STOP",
		"  stop  ":        "STOP",
		"opt-out":         "OPT OUT",
		"Stop all!":       "STOP ALL",
		"help?":           "HELP",
		"stop texting me": "STOP TEXTING ME",
	} {
		s.Equal(word, keywordOf(text), text)
	}
	s.Contains(stopWords, keywordOf("Opt out"))
	s.NotContains(stopWords, keywordOf("stop texting me"), "a sentence is the agent's to read")
}

// Only a connector whose episode source names an opt-out channel has keywords and passes the
// sandbox gate: SMS, WhatsApp (AI-879) and, since T62a (AI-921), iMessage, so a STOP in Slack
// reaches the agent as before.
func (s *BridgeSuite) TestOnlySMSWhatsAppAndIMessageHaveKeywords() {
	for connector, source := range episodeSources {
		switch connector {
		case "telnyx":
			s.Equal("sms", source.optOuts)
		case "whatsapp":
			s.Equal("whatsapp", source.optOuts)
		case "linq":
			s.Equal("imessage", source.optOuts)
		default:
			s.Empty(source.optOuts, connector)
		}
	}
}

// T62a (AI-921): a reply's files reach the external thread as links after its text, one a
// line, on every provider.
func (s *BridgeSuite) TestAReplysFilesAreLinksAfterItsText() {
	graph := sandbox.Attachment{Name: "graph.png", MIME: "image/png", URL: "https://cdn.example/graph.png"}
	report := sandbox.Attachment{Name: "report.pdf", MIME: "application/pdf", URL: "https://cdn.example/report.pdf"}

	s.Equal("Here they are.\n\nhttps://cdn.example/graph.png\nhttps://cdn.example/report.pdf",
		withFiles("Here they are.", []sandbox.Attachment{graph, report}))
	s.Equal("https://cdn.example/graph.png", withFiles("", []sandbox.Attachment{graph}), "a reply that is only a file")
	s.Equal("Noted.", withFiles("Noted.", nil), "a reply without files is its text, as before")
	s.Equal("Noted.", withFiles("Noted.", []sandbox.Attachment{{Name: "lost.png"}}), "a file with no link adds nothing")
}

// AI-879: Meta writes a WhatsApp author as digits with no +, and the contact map and the
// opt-outs key the person by the number in E.164; an SMS number is E.164 already.
func (s *BridgeSuite) TestAWhatsAppAuthorIsTheirNumberInE164() {
	whatsapp, telnyx := episodeSources["whatsapp"], episodeSources["telnyx"]

	person, err := whatsapp.person("106540352242922", "16505551234")

	s.Require().NoError(err)
	s.Equal("+16505551234", person.Address)
	s.Equal("+16505551234", whatsapp.recipientOf("16505551234"))
	s.Equal("+13125550001", telnyx.recipientOf("+13125550001"))
}

// WhatsApp takes free text for 24 hours after the person's last message; past that, a reply
// reaches them only when the manifest has a template to send instead.
func (s *BridgeSuite) TestAWhatsAppReplyPastTheDayIsPastTheWindowUnlessTheManifestHasATemplate() {
	free := core.ReplyRule{}
	templated := core.ReplyRule{Window: core.Duration(24 * time.Hour), AfterWindow: map[string]any{"type": "template"}}

	s.False(pastWindow("whatsapp", free, 23*time.Hour), "within the day")
	s.True(pastWindow("whatsapp", free, 24*time.Hour), "at the day's end")
	s.False(pastWindow("whatsapp", templated, 25*time.Hour), "the template is sent instead")
}

// SMS and Slack take free text whenever the person last wrote.
func (s *BridgeSuite) TestAnSMSOrSlackReplyIsNeverPastAWindow() {
	s.False(pastWindow("telnyx", core.ReplyRule{}, 30*24*time.Hour))
	s.False(pastWindow("slack_bot", core.ReplyRule{}, 30*24*time.Hour))
}

// A bridge that holds nothing but replies: a held command's reply never reaches the line,
// so none of its other parts are needed.
func (s *BridgeSuite) TestAReleasedCommandIsNoLongerHeld() {
	b := &Bridge{held: map[string]chan conversation.FinishedReply{}}
	_, release := b.Hold("cmd-1")
	s.Len(b.held, 1)
	release()
	s.Empty(b.held)
}

func (s *BridgeSuite) TestAHeldCommandsReplyGoesToItsHolderAndClearsTheEntry() {
	b := &Bridge{held: map[string]chan conversation.FinishedReply{}}
	replies, release := b.Hold("cmd-1")
	defer release()
	b.Reply(conversation.FinishedReply{CommandID: "cmd-1", Text: "done"})
	select {
	case got := <-replies:
		s.Equal("done", got.Text)
	default:
		s.Fail("the holder did not receive the reply")
	}
	s.Empty(b.held)
}

// threadChannel is a thread channel id as take makes one: 43 characters.
const threadChannel = threadChannelPrefix + "0b7c8f4e-2a51-4c1e-9d3f-6a2b8e1c7d90"

// A provider id of the characters Stream Chat took is in the message id as it is, after the
// thread channel: a Slack ts, a Telnyx or Linq UUID, a WhatsApp wamid without padding.
func (s *BridgeSuite) TestAProviderIDIsReadableInItsMessageID() {
	for _, id := range []string{"1793125611.816139", "40385f64-5717-4562-b3fc-2c963f66afa6", "wamid.HBgLMTY1MDUxNzY1NDMVAgARGBI5QTNDQTVCM0Q0Q0Q2RTY3RTcA"} {
		s.Equal(threadChannel+"-"+id, inboundMessageID(threadChannel, id))
	}
}

// The same provider id in two thread channels is two messages: Stream Chat keeps a message id
// unique in the whole app.
func (s *BridgeSuite) TestTheSameProviderIDInTwoThreadsIsTwoMessageIDs() {
	other := threadChannelPrefix + "1c8d9f5a-3b62-4d2f-8e4a-7b3c9f2d8ea1"
	s.NotEqual(inboundMessageID(threadChannel, "1793125611.816139"), inboundMessageID(other, "1793125611.816139"))
}

// A wamid with base64's padding, or + or /, which Stream Chat was not tried with, and an id
// too long for 255 characters are hashed: within the limit, and still one id each.
func (s *BridgeSuite) TestAProviderIDStreamChatMightRefuseIsHashed() {
	ids := []string{
		"wamid.HBgLMTY1MDM4NzQzOTkVAgASGBQzQTRBNjU5OUFFRTAzODEwMTQ0RgA=",
		"wamid.HBgL+TY1/DM4",
		"",
		strings.Repeat("a", maxMessageID-len(threadChannel)),
		strings.Repeat("a", 1000),
		threadChannel + "=x",
	}
	seen := map[string]bool{}
	for _, id := range ids {
		message := inboundMessageID(threadChannel, id)
		s.Equal(threadChannel+"=", message[:len(threadChannel)+1], id)
		s.Len(message, len(threadChannel)+1+64, id)
		s.False(seen[message], id)
		seen[message] = true
	}
}

// The longest readable provider id fits the 255 characters exactly.
func (s *BridgeSuite) TestTheLongestReadableIDFillsTheLimit() {
	id := strings.Repeat("a", maxMessageID-len(threadChannel)-1)

	s.Equal(threadChannel+"-"+id, inboundMessageID(threadChannel, id))
	s.Len(inboundMessageID(threadChannel, id), maxMessageID)
}

// Stream Chat's answer to a second SendMessage with the same id, however it is wrapped.
func (s *BridgeSuite) TestADuplicateMessageIDIsAlreadyWritten() {
	duplicate := &getstream.StreamError{StatusCode: http.StatusBadRequest, Code: 4,
		Message: `SendMessage failed with error: "a message with ID thread-1-m already exists"`}

	s.True(alreadyWritten(duplicate))
	s.True(alreadyWritten(stack.Wrap(fmt.Errorf("omnichannel: write the card: %w", duplicate))))
}

// Every other refusal, and the same words in an error Stream Chat did not send, is a failure.
func (s *BridgeSuite) TestAnyOtherRefusalIsNotAlreadyWritten() {
	for name, err := range map[string]error{
		"no error":         nil,
		"another code":     &getstream.StreamError{StatusCode: http.StatusBadRequest, Code: 16, Message: "a message with ID m already exists"},
		"another status":   &getstream.StreamError{StatusCode: http.StatusTooManyRequests, Code: 4, Message: "a message with ID m already exists"},
		"another input":    &getstream.StreamError{StatusCode: http.StatusBadRequest, Code: 4, Message: "message.id must be a maximum of 255 characters in length"},
		"a channel exists": &getstream.StreamError{StatusCode: http.StatusBadRequest, Code: 4, Message: "a channel with ID c already exists"},
		"no message id":    &getstream.StreamError{StatusCode: http.StatusBadRequest, Code: 4, Message: "a message with ID m"},
		"not from Stream":  errors.New("a message with ID m already exists"),
	} {
		s.False(alreadyWritten(err), name)
	}
}

func (s *BridgeSuite) TestTheInboundRetriesAreTwoTenThirtySecondsTwoAndTenMinutes() {
	s.Equal([]time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second, 2 * time.Minute, 10 * time.Minute}, defaultInboundRetries)
}
