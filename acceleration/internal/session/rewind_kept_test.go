//go:build integration

package session

import (
	"errors"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"

	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
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

	onScreen := s.visible(session)
	s.Require().Len(onScreen, 4)
	s.Equal(map[string]any{"response_id": next, "parent_response_id": first, "question_response_id": next,
		"of": "question", "index": float64(2), "count": float64(2)}, mark(onScreen[2]),
		"the question asked after the rewind is shown as 2 of 2, beside the rewound one")

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

// A rewind right after a turn, whose response the writer has not recorded yet, still hides
// that turn: the turns are read only once everything said is recorded.
func (s *BranchesSuite) TestARewindRightAfterATurnNotYetRecordedHidesIt() {
	gate := &gatedRecords{sessionRecords: s.manager.records}
	s.manager.records = gate
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	_, second, err := session.RespondCommand(s.ctx, uuid.NewString(), "And tomorrow?", "", nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().Eventually(func() bool {
		last := s.visible(session)
		return len(last) == 4 && last[3].state == "completed" && !s.service.Writing()
	}, 5*time.Second, 5*time.Millisecond, "the reply was never written")

	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?"}, s.texts(session))
	s.NotNil(s.response(session, second).RewoundAt)
}

// A rewind whose answer Chat lost, asked again, finishes the switch it began from the message
// it rewound to, which Chat may hide by now, rather than refusing it as unmarked.
func (s *BranchesSuite) TestARewindAskedAgainAfterItsAnswerWasLostFinishesIt() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.chat.LoseRewindAnswer()
	waiting, release := s.chat.Hold("/rewind")
	failed := make(chan error, 1)
	go func() { failed <- session.Rewind(s.ctx, s.store, first) }()
	<-waiting
	// Chat cannot be read once it rewound, so whether it did is not known.
	s.chat.Fail(http.MethodPost, "/query")
	release()
	s.Require().Error(<-failed)
	s.chat.Heal(http.MethodPost, "/query")

	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	s.NotNil(s.response(session, second).RewoundAt)
	next := s.asks(session, "And next week?")
	s.Equal(first, s.response(session, next).ParentResponseID)
}

// Rewinds and activations around a regenerate bring each branch back in the tree's order, in
// Chat, in the model's history and after a reopen.
func (s *BranchesSuite) TestRewindsAndActivationsAroundARegenerateKeepTheTreesOrder() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))
	_, err = session.Activate(s.ctx, branched.ResponseID)
	s.Require().NoError(err)
	s.written()
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.written()

	want := []string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?"}
	s.Equal(want, s.texts(session))
	s.Equal(want, history(session))
	reopened := s.reopens(session)
	s.Equal(want, s.texts(reopened))
	s.Equal(want, history(reopened))
}

// The question asked after a rewind is shown beside the rewound one though the session was
// reopened in between: its place is read from the tree, not held by the session that rewound.
func (s *BranchesSuite) TestTheQuestionAskedAfterARewindAndAReopenIsShownBesideTheRewoundOne() {
	session := s.opens()
	first := s.asks(session, "Q1")
	s.asks(session, "Q2")
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))
	reopened := s.reopens(session)

	next := s.asks(reopened, "Q3")

	onScreen := s.visible(reopened)
	s.Require().Len(onScreen, 4)
	s.Equal(map[string]any{"response_id": next, "parent_response_id": first, "question_response_id": next,
		"of": "question", "index": float64(2), "count": float64(2)}, mark(onScreen[2]))
}

// The same rewind made again after its turn was brought back, with Chat's answer lost, is a
// switch of its own: the next question finds it and is answered from what Chat shows.
func (s *BranchesSuite) TestARewindMadeAgainAfterItsTurnWasBroughtBackIsSettledWhenItsAnswerIsLost() {
	session := s.opens()
	first := s.asks(session, "Q1")
	second := s.asks(session, "Q2")
	s.Require().NoError(session.Rewind(s.ctx, s.store, first))
	_, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.written()
	s.chat.LoseRewindAnswer()
	waiting, release := s.chat.Hold("/rewind")
	failed := make(chan error, 1)
	go func() { failed <- session.Rewind(s.ctx, s.store, first) }()
	<-waiting
	// Chat cannot be read once it rewound, so whether it did is not known.
	s.chat.Fail(http.MethodPost, "/query")
	release()
	s.Require().Error(<-failed)
	s.chat.Heal(http.MethodPost, "/query")

	next := s.asks(session, "Q3")

	s.Equal(first, s.response(session, next).ParentResponseID, "Chat hid Q2, so Q3 follows Q1")
	s.NotNil(s.response(session, second).RewoundAt)
	s.Equal(s.texts(session), history(session), "the model holds what Chat shows")
	reopened := s.reopens(session)
	s.Equal(s.texts(reopened), history(reopened))
}

// A rewind asked while the turn after the kept response is still on its way to Chat waits for
// it, rather than reading a channel without it and refusing the turn as unmarked.
func (s *BranchesSuite) TestARewindWaitsForTheNextTurnsMessagesToReachChat() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	waiting, release := s.chat.Hold("/message")
	request := uuid.NewString()
	_, second, err := session.RespondCommand(s.ctx, request, "And tomorrow?", "", nil, options.LLM{})
	s.Require().NoError(err)
	<-waiting
	s.Require().Eventually(func() bool {
		done, err := session.Command(request)
		return err == nil && done.State == "completed"
	}, 5*time.Second, 5*time.Millisecond, "the reply is finished, though not in Chat yet")
	go func() {
		time.Sleep(100 * time.Millisecond)
		release()
	}()

	s.Require().NoError(session.Rewind(s.ctx, s.store, first))

	s.written()
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?"}, s.texts(session))
	s.NotNil(s.response(session, second).RewoundAt)
}

// A rewind that would hide a response whose messages carry no branch mark, as a turn spoken on
// a call writes them, is refused: nothing finds its messages in Chat to hide them by.
func (s *BranchesSuite) TestARewindThatWouldHideAnUnmarkedResponseIsRefused() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	spoken := store.AgentResponse{ID: uuid.NewString(), SessionID: session.ID(), CustomerID: s.customer,
		ParentResponseID: first, Said: "Said on a call", CreatedAt: time.Now().UTC()}
	s.Require().NoError(s.store.StartResponse(s.ctx, &spoken))
	before := s.texts(session)

	err := session.Rewind(s.ctx, s.store, first)

	unsupported, ok := errors.AsType[*persistent.BranchesUnsupportedError](err)
	s.Require().True(ok, "%v", err)
	s.Equal(persistent.UnsupportedUnmarked, unsupported.Reason)
	s.Equal(before, s.texts(session))
	s.False(s.rewound())
}
