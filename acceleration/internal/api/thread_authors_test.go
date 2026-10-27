//go:build integration

package api

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// AI-2014: in a thread channel several people write in, the model is told who wrote each
// message. Each person goes under "Speaker N": the bridge names an author's Stream user after
// the provider's id for them, a phone number on Linq, which the model is never told.

// severalWriters is the note a model answering several people in writing is given (the
// agent's own constant, which this package cannot read).
const severalWriters = "Several people are writing in this conversation. Each of their messages starts with " +
	"the name of who wrote it, then a colon. Never start your reply with such a label."

// forgetNoted forgets the noted models earlier tests of the suite opened.
func (s *RouterSuite) forgetNoted() {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	s.noted = nil
}

// askedWith is the request a noted model was sent for the turn text was written in, and every
// request any noted model was sent, as one string.
func (s *RouterSuite) askedWith(text string) (llm.ResponseParams, string) {
	s.notedMu.Lock()
	defer s.notedMu.Unlock()
	var found []llm.ResponseParams
	var everything strings.Builder
	for _, model := range s.noted {
		for _, request := range model.requests() {
			fmt.Fprint(&everything, request.Instructions, request.Input)
			if last := request.Input[len(request.Input)-1]; last.Role == llm.User && strings.HasSuffix(last.Content, text) {
				found = append(found, request)
			}
		}
	}
	s.Require().Len(found, 1, "one turn was written as %q", text)
	return found[0], everything.String()
}

func contentsOf(input []llm.Message) []string {
	contents := make([]string, 0, len(input))
	for _, message := range input {
		contents = append(contents, message.Content)
	}
	return contents
}

// The repro of AI-2014 (staging, 2026-10-27): Kanat and Ana in one iMessage group chat.
func (s *LinqChannelSuite) TestAGroupChatTellsTheModelWhoWroteEachMessage() {
	s.forgetNoted()
	chat := s.utils.uuid()
	kanat, ana := "+13125550001", "+13125550002"

	s.linqTurn(chat, kanat, true, "Hi all, I'm Kanat")
	s.linqTurn(chat, ana, true, "I'm Ana. What is your return policy?")
	s.linqTurn(chat, kanat, true, "What are our names?")
	s.linqTurn(chat, kanat, true, "hi")

	names, _ := s.askedWith("What are our names?")
	s.Equal(severalWriters, names.Instructions)
	s.Equal([]string{
		"Speaker 1: Hi all, I'm Kanat", "Noted.",
		"Speaker 2: I'm Ana. What is your return policy?", "Noted.",
		"Speaker 1: What are our names?",
	}, contentsOf(names.Input))
	hi, everything := s.askedWith("hi")
	s.Equal("Speaker 1: hi", hi.Input[len(hi.Input)-1].Content, "Kanat's hi is Kanat's")
	s.NotContains(everything, "+1312555000", "no model is told a phone number")
	s.NotContains(everything, "5550001", "nor its digits")
}

// The control: a 1:1 chat sends the model what base af17ef057 sends. The expected request is
// the one a probe of these two turns printed on af17ef057.
func (s *LinqChannelSuite) TestAOneToOneChatReachesTheModelAsAtBase() {
	s.forgetNoted()
	chat := s.utils.uuid()

	s.linqTurn(chat, "+13125550001", false, "Hi all, I'm Kanat")
	s.linqTurn(chat, "+13125550001", false, "What is my name?")

	asked, _ := s.askedWith("What is my name?")
	s.Empty(asked.Instructions)
	s.Equal([]string{"Hi all, I'm Kanat", "Noted.", "What is my name?"}, contentsOf(asked.Input))
	for _, message := range asked.Input {
		s.Empty(message.Speaker)
	}
}

// A Slack thread goes the same way (api.answerThread): Alice asks, Bob replies in the thread,
// Alice again.
func (s *SlackChannelSuite) TestAThreadOfTwoPeopleTellsTheModelWhoWroteEachMessage() {
	s.forgetNoted()
	channel := s.messaged("U0000ALICE", "is the build green?", "1759740000.000100", "")
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, 0))
	s.posted(1)
	s.threadTurnIsFree(channel)
	s.deliver(s.message("U0000BOB", "and the deploy?", "1759740000.000200", "1759740000.000100"), 0)
	stored := s.written(channel, 3)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, len(stored)-1))
	s.posted(2)
	s.threadTurnIsFree(channel)
	s.deliver(s.message("U0000ALICE", "thanks", "1759740000.000300", "1759740000.000100"), 0)
	stored = s.written(channel, 5)
	s.Require().Equal(http.StatusOK, s.streamDelivers(channel, len(stored)-1))
	s.posted(3)

	bob, _ := s.askedWith("and the deploy?")
	s.Equal(severalWriters, bob.Instructions, "Bob is the second person")
	s.Equal([]string{"is the build green?", "Noted.", "Speaker 2: and the deploy?"}, contentsOf(bob.Input),
		"Alice's turn was read back before anybody else wrote")
	alice, everything := s.askedWith("thanks")
	s.Equal([]string{
		"Speaker 1: is the build green?", "Noted.",
		"Speaker 2: and the deploy?", "Noted.",
		"Speaker 1: thanks",
	}, contentsOf(alice.Input))
	s.NotContains(everything, "U0000", "nor a Slack user id")
}
