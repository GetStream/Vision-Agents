//go:build integration

package session

import (
	"time"

	"github.com/google/uuid"

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
