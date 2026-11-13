//go:build integration

package session

import (
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// rewound reports whether Chat was asked to rewind the session's channel.
func (s *BranchesSuite) rewound() bool {
	for _, request := range s.chat.Requests(s.chat.Key) {
		if request.Method == http.MethodPost && strings.HasSuffix(request.Path, "/rewind") {
			return true
		}
	}
	return false
}

// AI-3209: a conversation kept in Chat, every non-incognito session, was refused a rewind.
func (s *BranchesSuite) TestRewindingAConversationKeptInChatHidesTheLaterTurnsAndCarriesOnFromTheKeptOne() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.asks(session, "And Sunday?")

	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?"}, s.texts(session), "Chat hides every later turn")
	s.Equal(s.texts(session), history(session), "the model holds the conversation Chat shows")
	s.NotNil(s.response(session, second).RewoundAt)
	s.Nil(s.response(session, first).RewoundAt, "the response rewound to is kept")

	next := s.asks(session, "And next week?")
	s.Equal(first, s.response(session, next).ParentResponseID, "the next turn follows the kept response")
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And next week?"}, s.texts(session)[:3])

	reopened := s.reopens(session)
	s.Equal(s.texts(reopened), history(reopened), "a reopen does not bring the rewound turns back")
	s.NotContains(s.texts(reopened), "And tomorrow?")
}

// The rewound turns are kept as a branch: the turn asked after the rewind is an alternative to
// them, which can be read and brought back rather than answering 410.
func (s *BranchesSuite) TestTheTurnsARewindHidAreABranchThatCanBeBroughtBack() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))
	next := s.asks(session, "And next week?")

	siblings, err := s.store.Siblings(s.ctx, s.customer, session.ID(), next, store.SiblingsOfQuestion)
	s.Require().NoError(err)
	s.Len(siblings, 2, "the question asked after the rewind stands beside the rewound one")
	active, shown, err := BranchShown(s.ctx, s.store, s.customer, session.ID(), second)
	s.Require().NoError(err)
	s.False(active)
	s.NotEmpty(shown, "the rewound turn's messages were copied before Chat hid them")

	activated, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
	s.Equal(second, activated.LeafResponseID)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
}

func (s *BranchesSuite) TestRewindingAKeptConversationToItsNewestResponseHidesNothing() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")

	s.Require().NoError(session.Rewind(s.ctx, s.store, second))

	s.False(s.rewound(), "there is nothing after the newest response to hide")
	s.Len(s.texts(session), 4)
}

func (s *BranchesSuite) TestARewindToAResponseOffTheConversationIsRefused() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	err := session.Rewind(s.ctx, s.store, second)

	s.ErrorIs(err, persistent.ErrBranchConflict, "a rewound response is brought back with activate, not rewound to")
}

func (s *BranchesSuite) TestARewindToAResponseTheSessionDoesNotHaveIsNotFound() {
	session := s.opens()
	s.asks(session, "Is it raining?")

	err := session.Rewind(s.ctx, s.store, uuid.NewString())

	s.ErrorIs(err, store.ErrUnknownResponse)
}

func (s *BranchesSuite) TestARewindChatHasTurnedOffIsRefusedAndTheConversationIsLeftAsItWas() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	s.asks(session, "And tomorrow?")
	s.chat.DisableRewind()

	err := session.Rewind(s.ctx, s.store, first)

	s.Require().Error(err)
	s.True(persistent.Refused(err), "Chat refused: nothing changed, nothing is left to settle")
	s.Len(s.texts(session), 4)
	s.Equal(s.texts(session), history(session))
}
