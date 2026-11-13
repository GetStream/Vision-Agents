//go:build integration

package session

import (
	"net/http"
	"time"

	"github.com/google/uuid"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
)

// A branch brought back is sent again in the order of the tree, though a question above it was
// sent again by an earlier activation and so has a newer copy than the branch's own messages.
func (s *BranchesSuite) TestABranchBroughtBackUnderAnAncestorSentAgainKeepsTheTreesOrder() {
	session := s.opens()
	first := s.asks(session, "Q1")
	second := s.asks(session, "Q2")
	_, err := session.Edit(s.ctx, uuid.NewString(), second, "Q2 edited", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	editedFirst, err := session.Edit(s.ctx, uuid.NewString(), first, "Q1 edited", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	_, err = session.Activate(s.ctx, first)
	s.Require().NoError(err)
	s.written()
	_, err = session.Activate(s.ctx, editedFirst.ResponseID)
	s.Require().NoError(err)
	s.written()

	_, shown, err := BranchShown(s.ctx, s.store, s.customer, session.ID(), second)
	s.Require().NoError(err)
	var read []string
	for _, m := range shown {
		text, _ := m.Message["text"].(string)
		read = append(read, text)
	}
	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.written()

	want := []string{"Q1", "Answer 1 to Q1", "Q2", "Answer 2 to Q2"}
	s.Equal(want, read, "the branch reads in the tree's order")
	s.Equal(want, s.texts(session), "Chat shows the branch in the tree's order")
	s.Equal(want, history(session), "the model holds it in the same order")
	reopened := s.reopens(session)
	s.Equal(want, s.texts(reopened))
	s.Equal(want, history(reopened), "a reopen reads it back in the same order")
}

// written waits until the conversation's writes reached Chat.
func (s *BranchesSuite) written() {
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
}

// A regenerate whose rewind Chat did not answer is kept while that rewind can still land. When
// it lands after all, the next question finds it and answers from the conversation Chat shows,
// rather than after the reply Chat hid.
func (s *BranchesSuite) TestARewindThatLandsLateIsFoundByTheNextQuestion() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	onScreen := s.visible(session)
	question, reply := onScreen[2].id, onScreen[3].id
	s.chat.FailWith(http.MethodPost, "/rewind", http.StatusGatewayTimeout)

	_, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})

	s.Require().Error(err)
	s.False(settledSwitch(session), "the switch may still land, so the next turn looks again")
	s.chat.Heal(http.MethodPost, "/rewind")
	// The rewind the router sent lands after all.
	_, err = persistent.NewRewinder(s.chat.Client).Rewind(s.ctx, "agent", session.ID(), question, false, reply, "agent")
	s.Require().NoError(err)

	third := s.asks(session, "And Sunday?")

	s.Equal(first, s.response(session, third).ParentResponseID, "the hidden reply is off the conversation")
	s.NotNil(s.response(session, second).RewoundAt)
	s.Equal(s.texts(session), history(session), "the model holds what Chat shows")
	s.True(settledSwitch(session))
}

// An edit that cannot be asked, empty or under an id that is not one, is refused before Chat
// hides anything: the conversation keeps the turn it would have replaced.
func (s *BranchesSuite) TestAnEditThatCannotBeAskedIsRefusedBeforeAnythingIsHidden() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	before := s.texts(session)

	_, empty := session.Edit(s.ctx, uuid.NewString(), second, "", "", options.LLM{})
	_, badID := session.Edit(s.ctx, "not an id!", second, "And on Sunday?", "", options.LLM{})

	s.Require().Error(empty)
	s.Require().Error(badID)
	s.Equal(before, s.texts(session), "Chat still shows the turn")
	s.Equal(before, history(session))
	s.Nil(s.response(session, second).RewoundAt)
	for _, request := range s.chat.Requests(s.chat.Key) {
		s.NotContains(request.Path, "/rewind", "Chat was never asked to rewind")
	}
}
