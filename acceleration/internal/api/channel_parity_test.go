//go:build integration

package api

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// T62a (AI-921): what internal/channels does on its lines that the channel bridge now does
// for a bridge connection too: the number's use-case texts for a keyword and a reply's files
// as links (here, on TelnyxChannelSuite's number, whose customer the sandbox does not hold),
// and the sandbox gate (TelnyxSandboxSuite).

// A keyword is answered with the texts of the use case the number is assigned to, before the
// customer's default, as internal/channels/keywords.go answers it.
func (s *TelnyxChannelSuite) TestKeywordsAreAnsweredWithTheNumbersUseCaseTexts() {
	s.useCase("Default", true)
	assigned := s.useCase("Line", false)
	ctx := context.Background()
	s.Require().NoError(s.store.RecordNumber(ctx, &store.PhoneNumber{E164: s.line, Vendor: "telnyx", Country: "US", CustomerID: s.customerID()}))
	// One customer at a time holds a number (phone_numbers_held_idx), and every test's line is
	// the same one.
	s.T().Cleanup(func() {
		_, err := s.store.DB().ExecContext(ctx, "DELETE FROM phone_numbers WHERE customer_id = ?", s.customerID())
		s.NoError(err)
	})
	s.Require().NoError(s.store.AssignNumbers(ctx, s.customerID(), assigned.ID, []string{s.line}))
	person := "+13125550001"

	for i, word := range []string{"STOP", "START", "HELP"} {
		s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, person, word), time.Now()))
		s.took(i + 1)
	}

	s.Equal([]string{"Line: stopped.", "Line: started again.", "Line: help."}, s.texts())
}

// A number assigned to no use case answers with the customer's default one's texts.
func (s *TelnyxChannelSuite) TestAKeywordIsAnsweredWithTheDefaultUseCasesTexts() {
	s.useCase("Default", true)

	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, "+13125550001", "HELP"), time.Now()))

	s.Equal("Default: help.", s.took(1)[0].text)
}

// A finished reply's files reach the person as links after its text: the conversation hands
// them over (conversation.FinishedReply.Files) and the bridge sends them in the text.
func (s *TelnyxChannelSuite) TestAReplysFilesReachThePersonAsLinks() {
	person := "+13125550001"
	s.deliver(s.received(s.line, person, "Send me the invoice"), time.Now())
	channel := s.threadChannel(person)
	s.written(channel, 1)
	bridge, ok := s.bridge.(*channelbridge.Bridge)
	s.Require().True(ok)

	bridge.Reply(conversation.FinishedReply{
		Customer: s.customerID(), CID: "agent:" + channel, MessageID: s.utils.uuid(), Text: "Here it is.",
		Files: []sandbox.Attachment{{Name: "invoice.pdf", MIME: "application/pdf", URL: "https://cdn.example/invoice.pdf"}},
	})

	s.Equal(telnyxMessage{from: s.line, to: person, text: "Here it is.\n\nhttps://cdn.example/invoice.pdf"},
		s.took(1)[0].withoutAuthorization())
}

// useCase records a use case of the test's customer whose texts start with name.
func (s *TelnyxChannelSuite) useCase(name string, isDefault bool) store.UseCase {
	useCase := store.UseCase{
		CustomerID: s.customerID(), Name: name, IsDefault: isDefault, Status: dlc.Draft,
		OptOutMessage: name + ": stopped.", OptInMessage: name + ": started again.", HelpMessage: name + ": help.",
	}
	s.Require().NoError(s.store.CreateUseCase(context.Background(), &useCase))
	return useCase
}

// texts are the texts the test's Telnyx took, in order.
func (s *TelnyxChannelSuite) texts() []string {
	var texts []string
	for _, sent := range s.telnyx.sent() {
		texts = append(texts, sent.text)
	}
	return texts
}

// TelnyxSandboxSuite is the Telnyx number of a customer the sandbox holds: no use case
// approved, so it may text only its sandbox recipients, one message a day here. The bridge
// asks dlc.Gate before it hands a message to the agent and before each reply, and counts each
// reply sent, as internal/channels does on its lines.
type TelnyxSandboxSuite struct {
	telnyxLine
}

func TestTelnyxSandboxSuite(t *testing.T) {
	runSuite(t, new(TelnyxSandboxSuite))
}

// recipient is the one number the test's customer may reach.
const sandboxRecipient = "+13125550001"

func (s *TelnyxSandboxSuite) SetupSuite() {
	s.sandbox = dlc.Sandbox{Enabled: true, Recipients: 2, MessagesPerDay: 1}
	s.telnyxLine.SetupSuite()
}

func (s *TelnyxSandboxSuite) SetupTest() {
	s.telnyxLine.SetupTest()
	s.Require().NoError(s.store.SetSandboxRecipients(context.Background(), s.customerID(), []string{sandboxRecipient}))
}

// A person the sandbox does not list reaches no agent and is sent nothing.
func (s *TelnyxSandboxSuite) TestAPersonOutsideTheSandboxReachesNoAgent() {
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, "+13125550099", "Hi"), time.Now()))

	channel := s.threadChannel("+13125550099")
	s.Never(func() bool { return len(s.chat.Stored(channel)) > 0 || len(s.telnyx.sent()) > 0 }, dropped, 20*time.Millisecond)
}

// A sandbox recipient is answered until the day's one message is sent; a reply after it is
// held back, and so is the next message.
func (s *TelnyxSandboxSuite) TestARecipientIsAnsweredUntilTheDailyLimit() {
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, sandboxRecipient, "first"), time.Now()))
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, sandboxRecipient, "second"), time.Now()))
	channel := s.threadChannel(sandboxRecipient)
	s.written(channel, 2)

	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.Equal("Noted.", s.took(1)[0].text)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 1))
	// The agent did answer the second message, in the thread channel; the gate keeps that
	// answer from the person.
	s.Require().Eventually(func() bool { return s.answers(channel) == 2 }, settleFor, 10*time.Millisecond)
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, sandboxRecipient, "third"), time.Now()))

	s.Never(func() bool { return len(s.telnyx.sent()) > 1 }, dropped, 20*time.Millisecond)
	for _, message := range s.chat.Stored(channel) {
		s.NotEqual("third", message["text"], "the message past the limit reaches no agent")
	}
}

// A keyword is answered whatever the sandbox says: the confirmation is the one message a
// person is still owed (internal/channels/keywords.go).
func (s *TelnyxSandboxSuite) TestAKeywordIsAnsweredPastTheSandbox() {
	s.Require().Equal(http.StatusOK, s.deliver(s.received(s.line, "+13125550099", "HELP"), time.Now()))

	s.Equal(telnyxHelp, s.took(1)[0].text)
}

// answers counts the agent's replies in a thread channel.
func (s *TelnyxSandboxSuite) answers(channel string) int {
	count := 0
	for _, message := range s.chat.Stored(channel) {
		if message["text"] == "Noted." {
			count++
		}
	}
	return count
}
