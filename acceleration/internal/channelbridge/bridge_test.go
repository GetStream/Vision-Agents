package channelbridge

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
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
	s.Equal(authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

// One Slack workspace can install two customers' apps; its person is a user of each, never
// one user both customers write as.
func (s *BridgeSuite) TestTheSamePersonForAnotherCustomerIsAnotherUser() {
	s.NotEqual(authorUserID("acme", "slack_bot", "T0000TEAM", "U0000ALICE"), authorUserID("globex", "slack_bot", "T0000TEAM", "U0000ALICE"))
}

func (s *BridgeSuite) TestPartsThatJoinAlikeAreStillTwoUsers() {
	s.NotEqual(authorUserID("acme", "slack_bot", "T0", "0U"), authorUserID("acme", "slack_bot", "T00", "U"))
}

// A phone number's "+" or any provider's id stays out of the user id.
func (s *BridgeSuite) TestAnAuthorUserIDKeepsToLettersDigitsUnderscoreAndHyphen() {
	s.Regexp(regexp.MustCompile(`^[A-Za-z0-9_-]+$`), authorUserID("acme", "linq", "+12025550100", "+12025550199"))
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
